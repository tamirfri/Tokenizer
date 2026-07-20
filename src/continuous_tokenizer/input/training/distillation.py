from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from continuous_tokenizer.backbone.assets import ModelAssets, load_frozen_causal_lm
from continuous_tokenizer.codec.batches import (
    build_span_batch,
    byte_reconstruction_loss,
)
from continuous_tokenizer.codec.checkpoints import save_checkpoint
from continuous_tokenizer.data.corpus import joined_prefix, sample_token_windows
from continuous_tokenizer.input.adapter import (
    InputEmbeddingAdapter,
    SegmentationAlignment,
)
from continuous_tokenizer.input.alignment import embedding_alignment_loss
from continuous_tokenizer.input.segmentation import EncodedSpan
from continuous_tokenizer.runtime.device import module_device, module_dtype, resolve_model_device
from continuous_tokenizer.runtime.environment import runtime_environment
from continuous_tokenizer.runtime.progress import ProgressTracker, log_event
from continuous_tokenizer.runtime.resume import capture_torch_rng_state, restore_torch_rng_state
from continuous_tokenizer.runtime.tensors import module_state_snapshot, parameter_fingerprint
from continuous_tokenizer.training.optimizers import (
    TokenizerOptimizers,
    build_tokenizer_optimizers,
    optimizer_metadata,
)

if TYPE_CHECKING:
    from continuous_tokenizer.runtime.resume import ResumeManager


@final
@dataclass(frozen=True, slots=True)
class DistillationOptions:
    epochs: int = 1
    windows: int = 512
    prompt_tokens: int = 64
    continuation_tokens: int = 16
    vocabulary_replay: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    seed: int = 17
    alignment: SegmentationAlignment = "aligned"

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.windows,
            self.prompt_tokens,
            self.continuation_tokens,
            self.vocabulary_replay,
            self.learning_rate,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("distillation counts and learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("distillation weight decay must be non-negative")
        if self.alignment not in {"aligned", "arbitrary"}:
            raise ValueError("distillation alignment must be aligned or arbitrary")


@final
@dataclass(frozen=True, slots=True)
class DistillationResult:
    steps: int
    mean_loss: float
    mean_kl: float
    mean_embedding_loss: float
    mean_reconstruction_loss: float
    trainable_parameters: tuple[str, ...]
    model_fingerprint: str
    options: dict[str, Any]
    epochs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _DistillationState:
    totals: dict[str, float]
    pending_totals: Tensor | None = None
    steps: int = 0
    start_epoch: int = 0
    resume_order: list[int] | None = None
    resume_window: int = 0
    epoch_metrics: list[dict[str, Any]] = field(default_factory=list)
    resume_epoch_totals: dict[str, float] | None = None
    resume_epoch_steps: int = 0


@final
@dataclass(frozen=True, slots=True)
class _DistillationEpoch:
    index: int
    number: int
    order: list[int]
    window_start: int
    start_totals: dict[str, float]
    start_steps: int
    started: float


@final
@dataclass(frozen=True, slots=True)
class _DistillationRuntime:
    optimizers: TokenizerOptimizers
    randomizer: random.Random
    resume_manager: ResumeManager | None
    phase: str
    epoch_boundary: Callable[[str, int], None] | None


@final
@dataclass(frozen=True, slots=True)
class _SnapshotPosition:
    next_window: int
    completed: bool


@final
@dataclass(frozen=True, slots=True)
class DistillationRequest:
    assets: ModelAssets
    checkpoint: Path
    output: Path
    documents: Sequence[bytes]
    options: DistillationOptions
    device: torch.device | None = None
    frozen_model: nn.Module | None = None
    resume_manager: ResumeManager | None = None
    epoch_boundary: Callable[[str, int], None] | None = None


@final
class FrozenBackboneDistiller:
    def __init__(
        self,
        model: nn.Module,
        adapter: InputEmbeddingAdapter,
        assets: ModelAssets,
        options: DistillationOptions,
    ) -> None:
        self.model = model.eval()
        self.adapter = adapter
        self.assets = assets
        self.options = options
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.adapter.codec.set_trainable_components(encoder=True, decoder=False)

    @property
    def device(self) -> torch.device:
        return module_device(self.model)

    @property
    def model_dtype(self) -> torch.dtype:
        return module_dtype(self.model)

    def _native_logits(self, prompt: tuple[int, ...], continuation: tuple[int, ...]) -> Tensor:
        full = prompt + continuation
        input_ids = torch.tensor([full], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, use_cache=False).logits[0]
        start = len(prompt) - 1
        return logits[start : start + len(continuation)].float()

    def _segmented_logits(
        self,
        prompt: tuple[int, ...],
        continuation: tuple[int, ...],
    ) -> tuple[Tensor, tuple[bytes, ...]]:
        segmented = self.adapter.encode_token_ids(
            prompt,
            mode="segmented",
            alignment=self.options.alignment,
        )
        continuation_encoding = self.adapter.encode_compatibility(
            continuation,
            position_offset=len(prompt),
        )
        embeddings = torch.cat((segmented.embeddings, continuation_encoding.embeddings)).to(
            device=self.device,
            dtype=self.model_dtype,
        )
        position_ids = torch.cat((segmented.position_ids, continuation_encoding.position_ids)).to(self.device)
        logits = self.model(
            inputs_embeds=embeddings.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            use_cache=False,
        ).logits[0]
        start = len(segmented.positions) - 1
        dynamic = tuple(position.data for position in segmented.positions if isinstance(position, EncodedSpan) and len(position.data) > 1)
        return logits[start : start + len(continuation)].float(), dynamic

    def _replay_losses(
        self,
        dynamic_spans: tuple[bytes, ...],
        randomizer: random.Random,
    ) -> tuple[Tensor, Tensor]:
        compatibility = self.assets.vocabulary.compatibility_ids
        replay_ids = randomizer.choices(compatibility, k=self.options.vocabulary_replay)
        vocabulary_spans = [self.assets.vocabulary.bytes_for(token_id) for token_id in replay_ids]
        spans = vocabulary_spans + list(dynamic_spans)
        static_rows = self.options.vocabulary_replay + self.options.prompt_tokens
        if len(spans) > static_rows:
            raise RuntimeError("distillation replay spans exceed the declared static row count")
        logical_rows = len(spans)
        spans.extend([spans[-1]] * (static_rows - logical_rows))
        batch = build_span_batch(
            spans,
            max_span=self.adapter.codec.max_span,
            device=self.adapter.device,
        )
        latent, logits = self.adapter.codec.reconstruction_logits(
            batch.byte_values,
            batch.valid_mask,
        )
        targets = self.assets.input_embeddings[replay_ids].to(
            device=self.adapter.device,
            dtype=latent.dtype,
        )
        embedding_loss = embedding_alignment_loss(latent[: len(replay_ids)], targets)
        reconstruction_loss = byte_reconstruction_loss(
            logits[:logical_rows],
            batch.framed_targets[:logical_rows],
            batch.target_mask[:logical_rows],
        )
        return embedding_loss, reconstruction_loss

    def _training_state(
        self,
        phase: str,
        optimizers: TokenizerOptimizers,
        randomizer: random.Random,
        resume_manager: ResumeManager | None,
    ) -> _DistillationState:
        state = _DistillationState(
            totals={
                "loss": 0.0,
                "kl": 0.0,
                "embedding": 0.0,
                "reconstruction": 0.0,
            }
        )
        snapshot = None if resume_manager is None else resume_manager.latest(phase)
        if snapshot is None:
            return state
        self.adapter.codec.load_state_dict(snapshot["codec"])
        optimizers.load_resume_state(snapshot["optimizers"])
        randomizer.setstate(snapshot["randomizer"])
        restore_torch_rng_state(self.device, snapshot["torch_rng"])
        return _DistillationState(
            totals={name: float(value) for name, value in snapshot["totals"].items()},
            pending_totals=(None if snapshot.get("pending_totals") is None else snapshot["pending_totals"].to(self.device)),
            steps=int(snapshot["steps"]),
            start_epoch=int(snapshot["epoch_index"]),
            resume_order=[int(value) for value in snapshot["order"]],
            resume_window=int(snapshot["next_window"]),
            epoch_metrics=list(snapshot.get("epoch_metrics", [])),
            resume_epoch_totals={
                name: float(value)
                for name, value in snapshot.get(
                    "epoch_start_totals",
                    snapshot["totals"],
                ).items()
            },
            resume_epoch_steps=int(snapshot.get("epoch_start_steps", snapshot["steps"])),
        )

    def _epoch(
        self,
        index: int,
        window_count: int,
        state: _DistillationState,
        randomizer: random.Random,
    ) -> _DistillationEpoch:
        resuming = index == state.start_epoch and state.resume_order is not None
        order = state.resume_order if resuming else list(range(window_count))
        if not resuming:
            randomizer.shuffle(order)
        resume_totals = state.resume_epoch_totals
        return _DistillationEpoch(
            index=index,
            number=index + 1,
            order=order,
            window_start=state.resume_window if resuming else 0,
            start_totals=resume_totals if index == state.start_epoch and resume_totals is not None else state.totals.copy(),
            start_steps=state.resume_epoch_steps if index == state.start_epoch and resume_totals is not None else state.steps,
            started=perf_counter(),
        )

    def _snapshot(
        self,
        epoch: _DistillationEpoch,
        state: _DistillationState,
        runtime: _DistillationRuntime,
        position: _SnapshotPosition,
    ) -> dict[str, Any]:
        return {
            "completed": position.completed,
            "codec": module_state_snapshot(self.adapter.codec),
            "optimizers": runtime.optimizers.resume_state(),
            "randomizer": runtime.randomizer.getstate(),
            "torch_rng": capture_torch_rng_state(self.device),
            "totals": state.totals,
            "pending_totals": (None if state.pending_totals is None else state.pending_totals.detach().cpu()),
            "steps": state.steps,
            "epoch_index": epoch.number if position.completed else epoch.index,
            "order": epoch.order,
            "next_window": position.next_window,
            "epoch_metrics": state.epoch_metrics,
            "epoch_start_totals": epoch.start_totals,
            "epoch_start_steps": epoch.start_steps,
        }

    def _train_window(
        self,
        window: tuple[tuple[int, ...], tuple[int, ...]],
        state: _DistillationState,
        optimizers: TokenizerOptimizers,
        randomizer: random.Random,
    ) -> None:
        prompt, continuation = window
        teacher = self._native_logits(prompt, continuation)
        student, dynamic_spans = self._segmented_logits(prompt, continuation)
        kl = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="batchmean",
        )
        embedding_loss, reconstruction_loss = self._replay_losses(
            dynamic_spans,
            randomizer,
        )
        loss = kl + embedding_loss + reconstruction_loss
        optimizers.optimize(loss)
        values = torch.stack(
            (
                loss.detach(),
                kl.detach(),
                embedding_loss.detach(),
                reconstruction_loss.detach(),
            )
        )
        state.pending_totals = values if state.pending_totals is None else state.pending_totals + values
        state.steps += 1

    def _record_epoch(
        self,
        epoch: _DistillationEpoch,
        state: _DistillationState,
        optimizers: TokenizerOptimizers,
    ) -> None:
        if state.pending_totals is not None:
            values = state.pending_totals.detach().cpu().tolist()
            for name, value in zip(
                ("loss", "kl", "embedding", "reconstruction"),
                values,
                strict=True,
            ):
                state.totals[name] += float(value)
            state.pending_totals = None
        optimizer_telemetry = optimizers.epoch_telemetry(reset=True)
        environment = runtime_environment(self.device)
        epoch_steps = state.steps - epoch.start_steps
        state.epoch_metrics.append(
            {
                "epoch": epoch.number,
                "epoch_wall_seconds": perf_counter() - epoch.started,
                "component_losses": {
                    "total": (state.totals["loss"] - epoch.start_totals["loss"]) / epoch_steps,
                    "kl": (state.totals["kl"] - epoch.start_totals["kl"]) / epoch_steps,
                    "embedding_alignment": (state.totals["embedding"] - epoch.start_totals["embedding"]) / epoch_steps,
                    "byte_reconstruction": (state.totals["reconstruction"] - epoch.start_totals["reconstruction"]) / epoch_steps,
                },
                "gradient_norms": {name: value for name, value in optimizer_telemetry.items() if "gradient_norm" in name},
                "optimizer_steps": optimizer_telemetry["optimizer_steps"],
                "peak_memory": {
                    "cpu_rss_bytes": optimizer_telemetry["peak_cpu_rss_bytes"],
                    "mps_allocated_bytes": optimizer_telemetry["peak_mps_allocated_bytes"],
                    "mps_driver_allocated_bytes": optimizer_telemetry["peak_mps_driver_allocated_bytes"],
                    "process_peak_rss_bytes": environment["peak_rss_bytes"],
                },
                "optimization_dtype": str(self.adapter.codec.dtype),
                "selection_dtype": str(self.assets.input_embeddings.dtype),
            }
        )

    def _run_epoch(
        self,
        windows: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
        epoch: _DistillationEpoch,
        state: _DistillationState,
        runtime: _DistillationRuntime,
    ) -> None:
        log_event(
            "epoch_started",
            phase=f"input_distillation_{self.options.alignment}",
            epoch=epoch.number,
            total_epochs=self.options.epochs,
        )
        progress = ProgressTracker(
            f"input_distillation_{self.options.alignment}_windows",
            len(epoch.order),
            {"epoch": epoch.number},
        )
        for window_index in range(epoch.window_start, len(epoch.order)):
            window_number = window_index + 1
            self._train_window(
                windows[epoch.order[window_index]],
                state,
                runtime.optimizers,
                runtime.randomizer,
            )
            progress.update(window_number)
            global_boundary = epoch.index * len(epoch.order) + window_number
            if runtime.resume_manager is not None and runtime.resume_manager.should_snapshot(global_boundary):
                runtime.resume_manager.save(
                    runtime.phase,
                    epoch.number,
                    self._snapshot(
                        epoch,
                        state,
                        runtime,
                        _SnapshotPosition(window_number, completed=False),
                    ),
                )
        self._record_epoch(epoch, state, runtime.optimizers)
        if runtime.resume_manager is not None and epoch.number == self.options.epochs:
            runtime.resume_manager.save(
                runtime.phase,
                epoch.number,
                self._snapshot(
                    epoch,
                    state,
                    runtime,
                    _SnapshotPosition(len(epoch.order), completed=True),
                ),
            )
        log_event(
            "epoch_completed",
            phase=f"input_distillation_{self.options.alignment}",
            epoch=epoch.number,
            cumulative_mean_loss=state.totals["loss"] / state.steps,
        )
        if runtime.epoch_boundary is not None:
            runtime.epoch_boundary(
                f"input_distillation_{self.options.alignment}",
                epoch.number,
            )

    def run(
        self,
        windows: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
        *,
        resume_manager: ResumeManager | None = None,
        epoch_boundary: Callable[[str, int], None] | None = None,
    ) -> DistillationResult:
        if not windows:
            raise ValueError("distillation windows must not be empty")
        if self.adapter.codec.dtype != torch.float32:
            raise ValueError("distillation requires FP32 tokenizer parameters")
        before = parameter_fingerprint(self.model)
        trainable = tuple(name for name, parameter in self.adapter.codec.named_parameters() if parameter.requires_grad)
        parameters = tuple(parameter for parameter in self.adapter.codec.parameters() if parameter.requires_grad)
        muon_parameters, adamw_parameters = self.adapter.codec.optimizer_parameter_groups(parameters)
        optimizers = build_tokenizer_optimizers(
            muon_parameters,
            adamw_parameters,
            learning_rate=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )
        randomizer = random.Random(self.options.seed)
        phase = f"input-distillation-{self.options.alignment}"
        state = self._training_state(phase, optimizers, randomizer, resume_manager)
        runtime = _DistillationRuntime(
            optimizers=optimizers,
            randomizer=randomizer,
            resume_manager=resume_manager,
            phase=phase,
            epoch_boundary=epoch_boundary,
        )
        self.adapter.codec.train()
        for index in range(state.start_epoch, self.options.epochs):
            self._run_epoch(
                windows,
                self._epoch(index, len(windows), state, randomizer),
                state,
                runtime,
            )
        self.adapter.codec.eval()
        after = parameter_fingerprint(self.model)
        if before != after:
            raise RuntimeError("frozen backbone parameters changed during distillation")
        return DistillationResult(
            steps=state.steps,
            mean_loss=state.totals["loss"] / state.steps,
            mean_kl=state.totals["kl"] / state.steps,
            mean_embedding_loss=state.totals["embedding"] / state.steps,
            mean_reconstruction_loss=state.totals["reconstruction"] / state.steps,
            trainable_parameters=trainable,
            model_fingerprint=before,
            options={
                **asdict(self.options),
                "trainable_component": "encoder",
                "frozen_component": "decoder",
                "optimizer": optimizer_metadata(),
                "optimization_dtype": str(self.adapter.codec.dtype),
            },
            epochs=tuple(state.epoch_metrics),
        )


def distill_checkpoint(request: DistillationRequest) -> DistillationResult:
    selected_device = resolve_model_device(request.device, request.frozen_model)
    loaded = InputEmbeddingAdapter.from_checkpoint(
        request.assets,
        request.checkpoint,
        device=selected_device,
    )
    adapter = loaded.adapter
    metadata = loaded.metadata
    codec = adapter.codec
    deployment_dtype = codec.dtype
    codec.to(dtype=torch.float32)
    codec.byte_embeddings = codec.byte_embeddings.to(dtype=deployment_dtype)
    if selected_device.type == "mps":
        codec.compile_neural_paths(static_rows=(request.options.vocabulary_replay + request.options.prompt_tokens,))
    model = load_frozen_causal_lm(request.assets, selected_device) if request.frozen_model is None else request.frozen_model
    data = joined_prefix(request.documents, max_bytes=8 * 1024 * 1024)
    token_ids = request.assets.tokenizer.encode(data.decode("utf-8"), add_special_tokens=False)
    windows = sample_token_windows(
        token_ids,
        count=request.options.windows,
        prompt_tokens=request.options.prompt_tokens,
        continuation_tokens=request.options.continuation_tokens,
    )
    result = FrozenBackboneDistiller(
        model,
        adapter,
        request.assets,
        request.options,
    ).run(
        windows,
        resume_manager=request.resume_manager,
        epoch_boundary=request.epoch_boundary,
    )
    save_checkpoint(
        request.output,
        codec,
        {
            **metadata,
            "distillation": {
                **result.to_dict(),
                "deployment_dtype": str(deployment_dtype),
            },
        },
        control_ids=adapter.control_ids,
        control_embeddings=adapter.control_embeddings,
    )
    return result
