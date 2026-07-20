from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import torch
from torch import Tensor
from torch.nn import functional as F

from continuous_tokenizer.artifacts.store import write_json_atomic
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.batches import byte_reconstruction_loss
from continuous_tokenizer.codec.output import OutputByteCodec, decode_output_batch
from continuous_tokenizer.output.evaluation import exact_output_predictions
from continuous_tokenizer.output.trajectory_cache import PreparedOutputCorpus
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
class OutputTrainingOptions:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size) < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch size, and learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight decay must be non-negative")


@final
@dataclass(frozen=True, slots=True)
class OutputTrainerContext:
    backbone: FrozenBackbone
    vocabulary: ByteVocabulary
    deployment_dtype: torch.dtype
    progress_directory: Path | None = None
    resume_manager: ResumeManager | None = None
    epoch_boundary: Callable[[str, int], None] | None = None


@dataclass(slots=True)
class _TrainingProgress:
    best_score: float = -1.0
    best_state: dict[str, Tensor] | None = None
    total_loss: float = 0.0
    total_steps: int = 0
    completed_epochs: int = 0
    start_epoch: int = 0


@final
@dataclass(frozen=True, slots=True)
class OutputTrainingResult:
    epochs: int
    steps: int
    mean_loss: float
    exact_event_agreement: float
    trainable_parameters: tuple[str, ...]
    backbone_fingerprint: str
    options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@final
class OutputCodecTrainer:
    def __init__(
        self,
        codec: OutputByteCodec,
        options: OutputTrainingOptions,
        context: OutputTrainerContext,
    ) -> None:
        if codec.config.control_count != len(context.vocabulary.control_ids):
            raise ValueError("output codec controls do not match the native vocabulary")
        self.backbone = context.backbone
        self.codec = codec
        self.options = options
        self.deployment_dtype = context.deployment_dtype
        self.progress_directory = context.progress_directory
        self.resume_manager = context.resume_manager
        self.epoch_boundary = context.epoch_boundary

    def _batch_loss(
        self,
        corpus: PreparedOutputCorpus,
        start: int,
        stop: int,
    ) -> Tensor:
        local_hidden, event_targets, byte_targets, byte_mask = corpus.batch(
            start,
            stop,
            device=self.codec.device,
            dtype=self.codec.dtype,
            narrow_targets=True,
        )
        byte_logits, control_logits = decode_output_batch(
            self.codec,
            local_hidden,
            maximum_rows=self.options.batch_size,
        )
        event_loss = F.cross_entropy(control_logits, event_targets)
        row_ids = (event_targets == 0).nonzero(as_tuple=False).flatten()
        byte_loss = (
            byte_reconstruction_loss(
                byte_logits[:, : byte_targets.shape[1]].index_select(0, row_ids),
                byte_targets.index_select(0, row_ids),
                byte_mask.index_select(0, row_ids),
            )
            if row_ids.numel()
            else event_loss.new_zeros(())
        )
        return event_loss + byte_loss

    @torch.no_grad()
    def _evaluate(
        self,
        corpus: PreparedOutputCorpus,
        codec: OutputByteCodec,
        *,
        epoch: int | None = None,
    ) -> float:
        exact = 0
        examples = 0
        progress = ProgressTracker(
            "output_selection_sequences",
            corpus.sequences,
            {} if epoch is None else {"epoch": epoch},
        )
        for sequence in range(corpus.sequences):
            sequence_start, sequence_stop = corpus.bounds(sequence)
            for start in range(sequence_start, sequence_stop, self.options.batch_size):
                stop = min(start + self.options.batch_size, sequence_stop)
                local_hidden, event_targets, byte_targets, byte_mask = corpus.batch(
                    start,
                    stop,
                    device=codec.device,
                    dtype=codec.dtype,
                    narrow_targets=True,
                )
                byte_logits, control_logits = decode_output_batch(
                    codec,
                    local_hidden,
                    maximum_rows=self.options.batch_size,
                )
                exact += int(
                    exact_output_predictions(
                        byte_logits[:, : byte_targets.shape[1]],
                        control_logits,
                        event_targets,
                        byte_targets,
                        byte_mask,
                    )
                    .sum()
                    .item()
                )
                examples += stop - start
            progress.update(sequence + 1)
        return exact / max(examples, 1)

    def _restore_progress(
        self,
        optimizers: TokenizerOptimizers,
        randomizer: random.Random,
    ) -> _TrainingProgress:
        progress = _TrainingProgress()
        state = None if self.resume_manager is None else self.resume_manager.latest("output-codec")
        if state is None:
            return progress
        self.codec.load_state_dict(state["codec"])
        optimizers.load_resume_state(state["optimizers"])
        randomizer.setstate(state["randomizer"])
        restore_torch_rng_state(self.codec.device, state["torch_rng"])
        progress.best_score = float(state["best_score"])
        progress.best_state = state["best_codec"]
        progress.total_loss = float(state["total_loss"])
        progress.total_steps = int(state["total_steps"])
        progress.completed_epochs = int(state["epoch"])
        progress.start_epoch = self.options.epochs if state["completed"] else progress.completed_epochs
        return progress

    def _deployment_evaluator(self) -> OutputByteCodec:
        deployed = copy.deepcopy(self.codec).to(dtype=self.deployment_dtype).requires_grad_(False).eval()
        if deployed.device.type == "mps":
            log_event(
                "compilation_started",
                phase="output_codec_deployment_evaluator",
            )
            deployed.compile_neural_paths()
            log_event(
                "compilation_completed",
                phase="output_codec_deployment_evaluator",
            )
        return deployed

    def _train_epoch(
        self,
        corpus: PreparedOutputCorpus,
        optimizers: TokenizerOptimizers,
        randomizer: random.Random,
        epoch: int,
    ) -> tuple[float, int]:
        order = list(range(corpus.sequences))
        randomizer.shuffle(order)
        progress = ProgressTracker(
            "output_training_sequences",
            len(order),
            {"epoch": epoch},
        )
        epoch_loss: Tensor | None = None
        epoch_steps = 0
        for sequence_number, sequence in enumerate(order, start=1):
            sequence_start, sequence_stop = corpus.bounds(sequence)
            for start in range(
                sequence_start,
                sequence_stop,
                self.options.batch_size,
            ):
                stop = min(start + self.options.batch_size, sequence_stop)
                loss = self._batch_loss(corpus, start, stop)
                optimizers.optimize(loss)
                epoch_loss = loss.detach() if epoch_loss is None else epoch_loss + loss.detach()
                epoch_steps += 1
            progress.update(sequence_number)
        return (
            0.0 if epoch_loss is None else float(epoch_loss.item()),
            epoch_steps,
        )

    def _record_epoch(
        self,
        *,
        epoch: int,
        epoch_loss: float,
        epoch_steps: int,
        score: float,
        selected: bool,
    ) -> None:
        mean_loss = epoch_loss / max(epoch_steps, 1)
        if self.progress_directory is not None:
            write_json_atomic(
                self.progress_directory / f"output-{epoch:04d}.json",
                {
                    "phase": "output_codec",
                    "epoch": epoch,
                    "training_loss": mean_loss,
                    "exact_event_agreement": score,
                    "selected": selected,
                },
            )
        log_event(
            "evaluation_completed",
            phase="output_codec",
            epoch=epoch,
            training_loss=mean_loss,
            exact_event_agreement=score,
            selected=selected,
        )

    def _save_progress(
        self,
        progress: _TrainingProgress,
        optimizers: TokenizerOptimizers,
        randomizer: random.Random,
        *,
        completed: bool,
    ) -> None:
        if self.resume_manager is None:
            return
        self.resume_manager.save(
            "output-codec",
            progress.completed_epochs,
            {
                "completed": completed,
                "codec": module_state_snapshot(self.codec),
                "optimizers": optimizers.resume_state(),
                "randomizer": randomizer.getstate(),
                "torch_rng": capture_torch_rng_state(self.codec.device),
                "best_codec": progress.best_state,
                "best_score": progress.best_score,
                "total_loss": progress.total_loss,
                "total_steps": progress.total_steps,
            },
        )

    def run(  # noqa: C901 - Epoch optimization and selection stay atomic.
        self,
        training_corpus: PreparedOutputCorpus,
        selection_corpus: PreparedOutputCorpus,
    ) -> OutputTrainingResult:
        if training_corpus.sequences < 1 or selection_corpus.sequences < 1:
            raise ValueError("output training and selection sequences must not be empty")
        if self.codec.dtype != torch.float32:
            raise ValueError("output codec optimization requires FP32 parameters")
        before = parameter_fingerprint(self.backbone.source_model)
        if any(parameter.requires_grad for parameter in self.backbone.source_model.parameters()):
            raise RuntimeError("frozen backbone exposes trainable parameters")
        trainable_names = tuple(name for name, parameter in self.codec.named_parameters() if parameter.requires_grad)
        muon, adamw = self.codec.optimizer_parameter_groups()
        optimizers = build_tokenizer_optimizers(
            muon,
            adamw,
            learning_rate=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )
        randomizer = random.Random(self.options.seed)
        progress = self._restore_progress(optimizers, randomizer)
        self.codec.train()
        deployed = self._deployment_evaluator()
        for epoch in range(progress.start_epoch, self.options.epochs):
            progress.completed_epochs = epoch + 1
            log_event(
                "epoch_started",
                phase="output_codec",
                epoch=progress.completed_epochs,
                total_epochs=self.options.epochs,
            )
            epoch_loss, epoch_steps = self._train_epoch(
                training_corpus,
                optimizers,
                randomizer,
                progress.completed_epochs,
            )
            progress.total_loss += epoch_loss
            progress.total_steps += epoch_steps
            deployed.load_state_dict(self.codec.state_dict())
            log_event(
                "evaluation_started",
                phase="output_codec",
                epoch=progress.completed_epochs,
            )
            score = self._evaluate(
                selection_corpus,
                deployed,
                epoch=progress.completed_epochs,
            )
            selected = score > progress.best_score
            self._record_epoch(
                epoch=progress.completed_epochs,
                epoch_loss=epoch_loss,
                epoch_steps=epoch_steps,
                score=score,
                selected=selected,
            )
            if selected:
                progress.best_score = score
                progress.best_state = module_state_snapshot(self.codec)
            completed = score == 1.0 or progress.completed_epochs == self.options.epochs
            if completed or (self.resume_manager is not None and self.resume_manager.should_snapshot(progress.completed_epochs)):
                self._save_progress(
                    progress,
                    optimizers,
                    randomizer,
                    completed=completed,
                )
            if self.epoch_boundary is not None:
                self.epoch_boundary(
                    "output_codec",
                    progress.completed_epochs,
                )
            if completed:
                break

        if progress.best_state is None:
            raise RuntimeError("output codec training did not select a checkpoint")
        self.codec.load_state_dict(progress.best_state)
        self.codec.eval()
        after = parameter_fingerprint(self.backbone.source_model)
        if before != after:
            raise RuntimeError("frozen backbone parameters changed during output codec training")
        return OutputTrainingResult(
            epochs=progress.completed_epochs,
            steps=progress.total_steps,
            mean_loss=progress.total_loss / max(progress.total_steps, 1),
            exact_event_agreement=progress.best_score,
            trainable_parameters=trainable_names,
            backbone_fingerprint=before,
            options={
                **asdict(self.options),
                "optimizer": optimizer_metadata(),
                "optimization_dtype": str(torch.float32),
                "deployment_dtype": str(self.deployment_dtype),
                "telemetry": {name: value for name, value in optimizers.epoch_telemetry().items() if not name.startswith("peak_")},
            },
        )
