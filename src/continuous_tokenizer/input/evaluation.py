from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.artifacts.source import find_project_root, source_state
from continuous_tokenizer.artifacts.store import write_json_atomic, write_text_atomic
from continuous_tokenizer.backbone.assets import (
    ModelAssets,
    load_frozen_causal_lm,
)
from continuous_tokenizer.backbone.config import tie_word_embeddings
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import InputEvaluationSpec
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.data.corpus import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    TokenWindowSampling,
    load_corpus_documents,
    sample_document_token_windows,
)
from continuous_tokenizer.input.adapter import (
    InputEmbeddingAdapter,
    InputEncoding,
    InputMode,
    LoadedInputAdapter,
    SegmentationAlignment,
)
from continuous_tokenizer.input.alignment import tensor_quantile
from continuous_tokenizer.input.benchmark.prefill import (
    PrefillBenchmarkOptions,
    benchmark_model_prefill,
)
from continuous_tokenizer.input.evaluation_calibration import (
    CALIBRATION_ROWS,
    PRODUCTION_BATCH_SIZE,
    UNIQUE_SAMPLE_COUNT,
    CalibrationIdentity,
    CalibrationTolerance,
    load_or_build_calibration,
)
from continuous_tokenizer.input.generation import (
    InputOnlyCausalLM,
    native_greedy_generate,
)
from continuous_tokenizer.runtime.device import module_device, module_dtype, resolve_model_device
from continuous_tokenizer.runtime.environment import runtime_environment
from continuous_tokenizer.runtime.resume import capture_torch_rng_state, restore_torch_rng_state
from continuous_tokenizer.runtime.tensors import parameter_fingerprint
from continuous_tokenizer.runtime.timing import timed_call

if TYPE_CHECKING:
    from continuous_tokenizer.runtime.resume import ResumeManager


_STUDENT_MODES: tuple[InputMode, ...] = ("compatibility", "segmented")
_ACCUMULATOR_MODES = (*_STUDENT_MODES, "segmented_native_continuation")


def _tensor_digest(tensors: tuple[Tensor, ...]) -> str:
    digest = hashlib.sha256()
    statistics: list[Tensor] = []
    for tensor in tensors:
        value = tensor.detach()
        descriptor = f"{value.dtype}:{tuple(value.shape)}".encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        measured = value.float()
        statistics.append(
            torch.stack(
                (
                    measured.sum(),
                    measured.abs().sum(),
                    measured.square().sum(),
                    measured.amin(),
                    measured.amax(),
                )
            )
        )
    if statistics:
        raw = torch.stack(statistics).cpu().contiguous().numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@final
@dataclass(frozen=True, slots=True)
class TeacherForcedBatchPolicy:
    batch_size: int = 8
    maximum_calibration_kl: float = 1e-4
    maximum_calibration_nll_delta: float = 1e-3
    minimum_calibration_top1_agreement: float = 1.0
    maximum_calibration_logit_error: float = 1e-2

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise ValueError("teacher-forced batch size must be at least two")
        self.calibration_tolerance()

    def calibration_tolerance(self) -> CalibrationTolerance:
        return CalibrationTolerance(
            maximum_kl=self.maximum_calibration_kl,
            maximum_nll_delta=self.maximum_calibration_nll_delta,
            minimum_top1_agreement=self.minimum_calibration_top1_agreement,
            maximum_logit_error=self.maximum_calibration_logit_error,
        )


@final
@dataclass(frozen=True, slots=True)
class NativeBaselineIdentity:
    model_id: str
    model_revision: str
    model_fingerprint: str
    prompt_window_sha256: str
    sample_order_sha256: str
    seed: int
    dtype: str
    device: str
    generation_samples: int
    max_new_tokens: int
    eos_token_ids: tuple[int, ...]
    teacher_forced_batch_size: int

    @property
    def key(self) -> str:
        return mapping_fingerprint(asdict(self))


@final
@dataclass(frozen=True, slots=True)
class NativeGeneration:
    token_ids: tuple[int, ...]
    seconds: float
    model_forwards: int


@final
@dataclass(frozen=True, slots=True)
class NativeBaselineBundle:
    identity: NativeBaselineIdentity
    teacher_logits: tuple[Tensor, ...]
    generations: tuple[NativeGeneration, ...]
    content_sha256: str

    @classmethod
    def create(
        cls,
        identity: NativeBaselineIdentity,
        teacher_logits: tuple[Tensor, ...],
        generations: tuple[NativeGeneration, ...],
    ) -> NativeBaselineBundle:
        owned = tuple(value.detach().clone() for value in teacher_logits)
        content_sha256 = mapping_fingerprint(
            {
                "identity": identity.key,
                "teacher_logits": _tensor_digest(owned),
                "generations": mapping_fingerprint([asdict(value) for value in generations]),
            }
        )
        return cls(identity, owned, generations, content_sha256)

    def verify(self, identity: NativeBaselineIdentity) -> None:
        if self.identity != identity:
            raise ValueError("native baseline identity does not match the requested evaluation")
        expected = NativeBaselineBundle.create(
            self.identity,
            self.teacher_logits,
            self.generations,
        ).content_sha256
        if expected != self.content_sha256:
            raise ValueError("native baseline bundle content was modified")


@final
class EvaluationSession:
    def __init__(self) -> None:
        self._model_fingerprint: str | None = None
        self._model_identity: int | None = None
        self._baselines: dict[str, NativeBaselineBundle] = {}
        self._adapters: dict[tuple[str, str], LoadedInputAdapter] = {}
        self._sample_sets: dict[str, tuple[PromptSample, ...]] = {}
        self._telemetry = {
            "checkpoint_loads": 0,
            "checkpoint_hashes": 0,
            "checkpoint_compiles": 0,
            "native_baseline_builds": 0,
            "native_baseline_reuses": 0,
            "native_model_forwards_avoided": 0,
            "native_baseline_integrity_transfers": 0,
            "corpus_preparation_builds": 0,
            "corpus_preparation_reuses": 0,
        }

    def bind_model(self, model: nn.Module) -> str:
        identity = id(model)
        if self._model_identity is None:
            self._model_identity = identity
            self._model_fingerprint = parameter_fingerprint(model)
        elif self._model_identity != identity:
            raise ValueError("evaluation session cannot be shared across frozen model instances")
        if self._model_fingerprint is None:
            raise RuntimeError("evaluation session did not capture a model fingerprint")
        return self._model_fingerprint

    def verify_model(self, model: nn.Module) -> str:
        before = self.bind_model(model)
        after = parameter_fingerprint(model)
        if before != after:
            raise RuntimeError("frozen model parameters changed during evaluation")
        return before

    def register_adapter(
        self,
        checkpoint: Path,
        device: torch.device,
        loaded: LoadedInputAdapter,
    ) -> None:
        key = (str(checkpoint.resolve()), str(device))
        existing = self._adapters.get(key)
        if existing is not None and existing.fingerprint != loaded.fingerprint:
            raise ValueError("checkpoint path changed during the bounded evaluation")
        if existing is None:
            self._telemetry["checkpoint_loads"] += 1
            self._telemetry["checkpoint_hashes"] += 1
            if device.type == "mps" and not loaded.adapter.codec.neural_paths_compiled:
                loaded.adapter.codec.compile_neural_paths()
                self._telemetry["checkpoint_compiles"] += 1
        self._adapters[key] = loaded

    def load_adapter(
        self,
        assets: ModelAssets,
        checkpoint: Path,
        device: torch.device,
    ) -> LoadedInputAdapter:
        key = (str(checkpoint.resolve()), str(device))
        loaded = self._adapters.get(key)
        if loaded is not None:
            return loaded
        loaded = InputEmbeddingAdapter.from_checkpoint(assets, checkpoint, device=device)
        self.register_adapter(checkpoint, device, loaded)
        return loaded

    def baseline(self, identity: NativeBaselineIdentity) -> NativeBaselineBundle | None:
        baseline = self._baselines.get(identity.key)
        if baseline is None:
            return None
        baseline.verify(identity)
        self._telemetry["native_baseline_reuses"] += 1
        self._telemetry["native_baseline_integrity_transfers"] += any(tensor.device.type != "cpu" for tensor in baseline.teacher_logits)
        forwards = math.ceil(len(baseline.teacher_logits) / baseline.identity.teacher_forced_batch_size)
        forwards += sum(item.model_forwards for item in baseline.generations)
        self._telemetry["native_model_forwards_avoided"] += forwards
        return baseline

    def prepared_samples(
        self,
        assets: ModelAssets,
        options: EvaluationOptions,
    ) -> list[PromptSample]:
        descriptor = {
            "model_id": assets.model_id,
            "model_revision": assets.revision,
            "dataset_id": options.dataset_id,
            "dataset_config": options.dataset_config,
            "dataset_revision": options.dataset_revision,
            "dataset_split": options.dataset_split,
            "corpus_max_rows": options.corpus_max_rows,
            "samples": options.samples,
            "prompt_tokens": options.prompt_tokens,
            "continuation_tokens": options.continuation_tokens,
            "seed": options.seed,
        }
        key = mapping_fingerprint(descriptor)
        prepared = self._sample_sets.get(key)
        if prepared is None:
            prepared = tuple(_samples(assets, options))
            self._sample_sets[key] = prepared
            self._telemetry["corpus_preparation_builds"] += 1
        else:
            self._telemetry["corpus_preparation_reuses"] += 1
        return list(prepared)

    def store_baseline(self, baseline: NativeBaselineBundle) -> None:
        existing = self._baselines.get(baseline.identity.key)
        if existing is not None:
            existing.verify(baseline.identity)
            if existing.content_sha256 != baseline.content_sha256:
                raise ValueError("native baseline identity resolved to different content")
            return
        baseline.verify(baseline.identity)
        self._baselines[baseline.identity.key] = baseline
        self._telemetry["native_baseline_builds"] += 1

    def telemetry(self) -> dict[str, int]:
        return dict(self._telemetry)


@final
@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    output_dir: Path
    samples: int = 128
    prompt_tokens: int = 256
    continuation_tokens: int = 64
    generation_samples: int = 32
    max_new_tokens: int = 64
    warmups: int = 5
    repetitions: int = 20
    performance_prompts: int = 4
    seed: int = 17
    segmentation_alignment: SegmentationAlignment = "arbitrary"
    dataset_id: str = DATASET_ID
    dataset_config: str = DATASET_CONFIG
    dataset_revision: str = DATASET_REVISION
    corpus_max_rows: int = 4096
    dataset_split: str = "test"

    def __post_init__(self) -> None:
        positive = (
            self.samples,
            self.prompt_tokens,
            self.continuation_tokens,
            self.max_new_tokens,
            self.repetitions,
            self.performance_prompts,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("evaluation counts and token lengths must be positive")
        if self.generation_samples < 0 or self.warmups < 0:
            raise ValueError("generation samples and warmups must be non-negative")
        if self.corpus_max_rows < 1:
            raise ValueError("evaluation corpus row bound must be positive")
        if self.segmentation_alignment not in {"aligned", "arbitrary"}:
            raise ValueError("segmented alignment must be aligned or arbitrary")
        if self.dataset_split not in {"validation", "test"}:
            raise ValueError("evaluation dataset split must be validation or test")


def evaluation_options_from_spec(
    experiment: ExperimentSpec,
    output_dir: Path,
    *,
    segmentation_alignment: SegmentationAlignment,
    seed: int | None = None,
    dataset_split: Literal["validation", "test"] = "test",
) -> EvaluationOptions:
    evaluation = experiment.evaluation
    if not isinstance(evaluation, InputEvaluationSpec):
        raise ValueError("input evaluation requires input-only settings")
    return EvaluationOptions(
        output_dir=output_dir,
        samples=evaluation.samples,
        prompt_tokens=evaluation.prompt_tokens,
        continuation_tokens=evaluation.continuation_tokens,
        generation_samples=evaluation.generation_samples,
        max_new_tokens=evaluation.max_new_tokens,
        warmups=evaluation.warmups,
        repetitions=evaluation.repetitions,
        performance_prompts=evaluation.performance_prompts,
        seed=experiment.seed if seed is None else seed,
        segmentation_alignment=segmentation_alignment,
        dataset_id=experiment.dataset.dataset_id,
        dataset_config=experiment.dataset.config,
        dataset_revision=experiment.dataset.revision,
        corpus_max_rows=experiment.runtime.corpus_max_rows,
        dataset_split=dataset_split,
    )


def teacher_forced_policy_from_spec(
    experiment: ExperimentSpec,
) -> TeacherForcedBatchPolicy:
    evaluation = experiment.evaluation
    if not isinstance(evaluation, InputEvaluationSpec):
        raise ValueError("input evaluation requires input-only settings")
    return TeacherForcedBatchPolicy(
        batch_size=evaluation.batch_size,
        maximum_calibration_kl=evaluation.calibration_maximum_kl,
        maximum_calibration_nll_delta=evaluation.calibration_maximum_nll_delta,
        minimum_calibration_top1_agreement=evaluation.calibration_minimum_top1_agreement,
        maximum_calibration_logit_error=evaluation.calibration_maximum_logit_error,
    )


@final
@dataclass(frozen=True, slots=True)
class EvaluationRuntime:
    device: torch.device | None = None
    frozen_model: nn.Module | None = None
    resume_manager: ResumeManager | None = None
    resume_phase: str = "input-evaluation"
    session: EvaluationSession | None = None
    teacher_forced_policy: TeacherForcedBatchPolicy = TeacherForcedBatchPolicy()
    calibration_cache_directory: Path | None = None
    dependency_lock_sha256: str | None = None


@final
@dataclass(frozen=True, slots=True)
class _EvaluationResume:
    manager: ResumeManager | None
    phase: str


@final
@dataclass(frozen=True, slots=True)
class _TeacherForcedRuntime:
    resume: _EvaluationResume
    native_embeddings: Tensor
    baseline: NativeBaselineBundle | None = None
    policy: TeacherForcedBatchPolicy = TeacherForcedBatchPolicy()


@final
@dataclass(frozen=True, slots=True)
class PromptSample:
    prompt: tuple[int, ...]
    continuation: tuple[int, ...]


@final
@dataclass(frozen=True, slots=True)
class GenerationSample:
    sample: int
    native_ids: tuple[int, ...]
    compatibility_ids: tuple[int, ...]
    segmented_ids: tuple[int, ...]
    compatibility_prefix: int
    segmented_prefix: int
    compatibility_token_similarity: float
    segmented_token_similarity: float
    compatibility_byte_similarity: float
    segmented_byte_similarity: float
    native_seconds: float
    compatibility_seconds: float
    segmented_seconds: float


@final
class LogitAccumulator:
    def __init__(self, measurement_dtype: torch.dtype) -> None:
        self.measurement_dtype = measurement_dtype
        self.tokens = 0
        self.teacher_nll = 0.0
        self.student_nll = 0.0
        self.kl = 0.0
        self.js = 0.0
        self.top1 = 0
        self.top5_overlap = 0.0
        self.row_mean_errors: list[Tensor] = []
        self.row_max_errors: list[Tensor] = []
        self.compact_transfers = 0

    def sufficient_statistics(
        self,
        teacher: Tensor,
        student: Tensor,
        targets: Tensor,
    ) -> Tensor:
        if teacher.dtype != self.measurement_dtype or student.dtype != self.measurement_dtype:
            raise ValueError("behavioral logits must be measured in the source embedding dtype")
        teacher_float = teacher.float()
        student_float = student.float()
        teacher_log = F.log_softmax(teacher_float, dim=-1)
        student_log = F.log_softmax(student_float, dim=-1)
        teacher_probabilities = teacher_log.exp()
        student_probabilities = student_log.exp()
        mixture = (teacher_probabilities + student_probabilities) / 2
        mixture_log = mixture.clamp_min(1e-30).log()
        teacher_top5 = teacher_float.topk(5, dim=-1).indices
        student_top5 = student_float.topk(5, dim=-1).indices
        overlap = (teacher_top5[:, :, None] == student_top5[:, None, :]).any(dim=2).sum(dim=1)
        difference = (teacher_float - student_float).abs()
        row_means = difference.mean(dim=1)
        row_maxima = difference.max(dim=1).values
        scalars = torch.stack(
            (
                -teacher_log.gather(1, targets[:, None]).sum(),
                -student_log.gather(1, targets[:, None]).sum(),
                (teacher_probabilities * (teacher_log - student_log)).sum(),
                (0.5 * teacher_probabilities * (teacher_log - mixture_log) + 0.5 * student_probabilities * (student_log - mixture_log)).sum(),
                (teacher_top5[:, 0] == student_top5[:, 0]).sum(),
                (overlap.float() / 5).sum(),
            )
        )
        return torch.cat((scalars, row_means, row_maxima)).detach()

    def consume(self, compact: Tensor, rows: int) -> None:
        self.compact_transfers += 1
        self.tokens += rows
        self.teacher_nll += float(compact[0])
        self.student_nll += float(compact[1])
        self.kl += float(compact[2])
        self.js += float(compact[3])
        self.top1 += int(compact[4])
        self.top5_overlap += float(compact[5])
        self.row_mean_errors.append(compact[6 : 6 + rows])
        self.row_max_errors.append(compact[6 + rows : 6 + (2 * rows)])

    def result(self) -> dict[str, float | int | str]:
        count = max(self.tokens, 1)
        means = torch.cat(self.row_mean_errors) if self.row_mean_errors else torch.empty(0)
        maxima = torch.cat(self.row_max_errors) if self.row_max_errors else torch.empty(0)
        student_nll = self.student_nll / count
        return {
            "measurement_dtype": str(self.measurement_dtype),
            "tokens": self.tokens,
            "teacher_nll": self.teacher_nll / count,
            "student_nll": student_nll,
            "student_perplexity": math.exp(min(student_nll, 80)),
            "mean_kl": self.kl / count,
            "mean_js": self.js / count,
            "top1_agreement": self.top1 / count,
            "top5_overlap": self.top5_overlap / count,
            "mean_absolute_logit_error": float(means.mean().item()) if means.numel() else 0.0,
            "maximum_logit_error_p50": tensor_quantile(maxima, 0.50),
            "maximum_logit_error_p95": tensor_quantile(maxima, 0.95),
            "maximum_logit_error": float(maxima.max().item()) if maxima.numel() else 0.0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "teacher_nll": self.teacher_nll,
            "student_nll": self.student_nll,
            "kl": self.kl,
            "js": self.js,
            "top1": self.top1,
            "top5_overlap": self.top5_overlap,
            "row_mean_errors": self.row_mean_errors,
            "row_max_errors": self.row_max_errors,
            "compact_transfers": self.compact_transfers,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.tokens = int(state["tokens"])
        self.teacher_nll = float(state["teacher_nll"])
        self.student_nll = float(state["student_nll"])
        self.kl = float(state["kl"])
        self.js = float(state["js"])
        self.top1 = int(state["top1"])
        self.top5_overlap = float(state["top5_overlap"])
        self.row_mean_errors = list(state["row_mean_errors"])
        self.row_max_errors = list(state["row_max_errors"])
        self.compact_transfers = int(state.get("compact_transfers", 0))


def _new_accumulators(dtype: torch.dtype) -> dict[str, LogitAccumulator]:
    return {mode: LogitAccumulator(dtype) for mode in _ACCUMULATOR_MODES}


def _new_prompt_positions() -> dict[str, int]:
    return dict.fromkeys(("native", *_STUDENT_MODES), 0)


def _teacher_forced_result(
    accumulators: dict[str, LogitAccumulator],
    prompt_positions: dict[str, int],
    sample_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {mode: accumulators[mode].result() for mode in _STUDENT_MODES}
    diagnostic = {
        "label": "mechanism_only_native_continuation",
        "purpose": "isolate_compressed_prompt_effects",
        "acceptance_scope": "excluded",
        "claims_scope": "excluded",
        "performance_scope": "excluded",
        "teacher_forced": accumulators["segmented_native_continuation"].result(),
    }
    positions = {name: value / max(sample_count, 1) for name, value in prompt_positions.items()}
    positions["native_positions_per_compatibility_position"] = positions["native"] / max(
        positions["compatibility"],
        1,
    )
    positions["native_positions_per_segmented_position"] = positions["native"] / max(
        positions["segmented"],
        1,
    )
    return metrics, positions, diagnostic


def _samples(assets: ModelAssets, options: EvaluationOptions) -> list[PromptSample]:
    documents = load_corpus_documents(
        options.dataset_split,
        dataset_id=options.dataset_id,
        config=options.dataset_config,
        revision=options.dataset_revision,
        max_rows=options.corpus_max_rows,
    )

    def encode(text: str) -> list[int]:
        return assets.tokenizer.encode(text, add_special_tokens=False)

    windows = sample_document_token_windows(
        documents,
        encode,
        TokenWindowSampling(
            count=options.samples,
            prompt_tokens=options.prompt_tokens,
            continuation_tokens=options.continuation_tokens,
            seed=options.seed,
        ),
    )
    return list(starmap(PromptSample, windows))


def _eos_ids(assets: ModelAssets) -> tuple[int, ...]:
    eos = assets.tokenizer.eos_token_id
    if eos is None:
        return ()
    if isinstance(eos, int):
        return (eos,)
    return tuple(eos)


def _baseline_identity(  # noqa: PLR0913 - Identity fields are intentionally explicit.
    assets: ModelAssets,
    model: nn.Module,
    model_fingerprint: str,
    samples: list[PromptSample],
    options: EvaluationOptions,
    policy: TeacherForcedBatchPolicy,
) -> NativeBaselineIdentity:
    content_digests = [mapping_fingerprint(asdict(sample)) for sample in samples]
    return NativeBaselineIdentity(
        model_id=assets.model_id,
        model_revision=assets.revision,
        model_fingerprint=model_fingerprint,
        prompt_window_sha256=mapping_fingerprint(sorted(content_digests)),
        sample_order_sha256=mapping_fingerprint(content_digests),
        seed=options.seed,
        dtype=str(module_dtype(model)),
        device=str(module_device(model)),
        generation_samples=min(options.generation_samples, len(samples)),
        max_new_tokens=options.max_new_tokens,
        eos_token_ids=_eos_ids(assets),
        teacher_forced_batch_size=policy.batch_size,
    )


def _require_source_dtype(
    assets: ModelAssets,
    adapter: InputEmbeddingAdapter,
    model: nn.Module,
) -> None:
    source_dtype = assets.input_embeddings.dtype
    if adapter.codec.dtype != source_dtype or module_dtype(model) != source_dtype:
        raise ValueError("input behavior evaluation requires source-dtype codec and model states")


def _selected_logits(
    model: nn.Module,
    *,
    indices: Tensor,
    input_ids: Tensor | None = None,
    encoding: InputEncoding | None = None,
) -> Tensor:
    device = module_device(model)
    kwargs: dict[str, Any] = {"use_cache": False, "logits_to_keep": indices.to(device)}
    if input_ids is not None:
        if encoding is not None:
            raise ValueError("provide exactly one of input IDs or an input encoding")
        kwargs["input_ids"] = input_ids.to(device)
    else:
        if encoding is None:
            raise ValueError("provide exactly one of input IDs or an input encoding")
        kwargs["inputs_embeds"] = encoding.embeddings.to(device=device, dtype=module_dtype(model)).unsqueeze(0)
        kwargs["position_ids"] = encoding.position_ids.to(device).unsqueeze(0)
    return model(**kwargs).logits[0]


def _native_batch_logits(
    model: nn.Module,
    samples: list[tuple[int, PromptSample]],
) -> dict[int, Tensor]:
    if not samples:
        return {}
    device = module_device(model)
    lengths = [len(sample.prompt) + len(sample.continuation) for _, sample in samples]
    maximum = max(lengths)
    input_ids = torch.zeros(
        (len(samples), maximum),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    position_ids = torch.zeros_like(input_ids)
    for row, ((_, sample), length) in enumerate(zip(samples, lengths, strict=True)):
        input_ids[row, :length] = torch.tensor(
            sample.prompt + sample.continuation,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :length] = 1
        position_ids[row, :length] = torch.arange(length, device=device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits
    selected: dict[int, Tensor] = {}
    for row, (sample_index, sample) in enumerate(samples):
        indices = torch.arange(
            len(sample.prompt) - 1,
            len(sample.prompt) + len(sample.continuation) - 1,
            device=device,
        )
        selected[sample_index] = outputs[row].index_select(0, indices).detach()
    return selected


def _build_native_baseline(
    model: nn.Module,
    samples: list[PromptSample],
    identity: NativeBaselineIdentity,
) -> NativeBaselineBundle:
    device = module_device(model)
    teacher_logits: list[Tensor] = []
    generations: list[NativeGeneration] = []
    with torch.inference_mode():
        for start in range(0, len(samples), identity.teacher_forced_batch_size):
            batch = samples[start : start + identity.teacher_forced_batch_size]
            logits = _native_batch_logits(
                model,
                list(enumerate(batch, start=start)),
            )
            teacher_logits.extend(logits[index] for index in range(start, start + len(batch)))
        for sample in samples[: identity.generation_samples]:
            generated, seconds = timed_call(
                lambda prompt=sample.prompt: native_greedy_generate(
                    model,
                    prompt,
                    eos_token_ids=identity.eos_token_ids,
                    max_new_tokens=identity.max_new_tokens,
                ),
                device,
            )
            generations.append(
                NativeGeneration(
                    generated.token_ids,
                    seconds,
                    max(len(generated.token_ids), 1),
                )
            )
    return NativeBaselineBundle.create(
        identity,
        tuple(teacher_logits),
        tuple(generations),
    )


def _native_baseline(  # noqa: PLR0913 - Baseline construction binds all source inputs.
    session: EvaluationSession,
    assets: ModelAssets,
    model: nn.Module,
    samples: list[PromptSample],
    options: EvaluationOptions,
    policy: TeacherForcedBatchPolicy,
) -> NativeBaselineBundle:
    identity = _baseline_identity(
        assets,
        model,
        session.bind_model(model),
        samples,
        options,
        policy,
    )
    baseline = session.baseline(identity)
    if baseline is None:
        baseline = _build_native_baseline(model, samples, identity)
        session.store_baseline(baseline)
    return baseline


def _baseline_measurement(baseline: NativeBaselineBundle) -> dict[str, Any]:
    return {
        "identity": asdict(baseline.identity),
        "key": baseline.identity.key,
        "content_sha256": baseline.content_sha256,
        "model_call_order": "native_baseline_then_candidate_compatibility_then_candidate_segmented",
    }


def _evaluation_telemetry(
    session: EvaluationSession,
    sample_count: int,
    policy: TeacherForcedBatchPolicy,
    device: torch.device,
) -> dict[str, int | float]:
    batches = math.ceil(sample_count / policy.batch_size)
    scalar_forwards = 4 * sample_count
    batched_forwards = 4 * batches
    return session.telemetry() | {
        "teacher_forced_batch_fill_ratio": sample_count / max(batches * policy.batch_size, 1),
        "teacher_forced_model_forwards": batched_forwards,
        "scalar_teacher_forced_model_forwards": scalar_forwards,
        "teacher_forced_model_forwards_avoided": scalar_forwards - batched_forwards,
        "compact_metric_transfers": batches,
        "synchronization_count": batches if device.type in {"cuda", "mps"} else 0,
    }


def _student_sequences(
    adapter: InputEmbeddingAdapter,
    sample: PromptSample,
    mode: InputMode,
    alignment: SegmentationAlignment,
    native_embeddings: Tensor,
) -> tuple[InputEncoding, Tensor, InputEncoding | None]:
    prompt = adapter.encode_token_ids(
        sample.prompt,
        mode=mode,
        cache=adapter.codec.encoding_cache,
        alignment=alignment,
    )
    continuation = adapter.encode_compatibility(
        sample.continuation,
        cache=adapter.codec.encoding_cache,
        position_offset=len(sample.prompt),
    )
    embeddings = torch.cat((prompt.embeddings, continuation.embeddings), dim=0)
    positions = prompt.positions + continuation.positions
    position_ids = torch.cat((prompt.position_ids, continuation.position_ids))
    indices = torch.arange(
        len(prompt.positions) - 1,
        len(prompt.positions) + len(sample.continuation) - 1,
        device=adapter.device,
    )
    canonical = InputEncoding(embeddings, positions, position_ids)
    diagnostic = None
    if mode == "segmented":
        continuation_ids = torch.tensor(
            sample.continuation,
            dtype=torch.long,
            device=native_embeddings.device,
        )
        exact_continuation = native_embeddings.index_select(
            0,
            continuation_ids,
        ).to(device=adapter.device, dtype=adapter.codec.dtype)
        diagnostic = InputEncoding(
            torch.cat((prompt.embeddings, exact_continuation), dim=0),
            positions,
            position_ids,
        )
    return canonical, indices, diagnostic


type _EncodingEntry = tuple[int, InputEncoding, Tensor]


def _student_sequence_entries(
    adapter: InputEmbeddingAdapter,
    samples: list[PromptSample],
    alignment: SegmentationAlignment,
    native_embeddings: Tensor,
    *,
    start: int = 0,
) -> dict[str, list[_EncodingEntry]]:
    entries: dict[str, list[_EncodingEntry]] = {mode: [] for mode in _ACCUMULATOR_MODES}
    for index, sample in enumerate(samples[start:], start=start):
        for mode in _STUDENT_MODES:
            encoding, indices, diagnostic = _student_sequences(
                adapter,
                sample,
                mode,
                alignment,
                native_embeddings,
            )
            entries[mode].append((index, encoding, indices))
            if diagnostic is not None:
                entries["segmented_native_continuation"].append(
                    (index, diagnostic, indices),
                )
    return entries


def _batched_encoding_logits(
    model: nn.Module,
    entries: list[_EncodingEntry],
) -> dict[int, Tensor]:
    if not entries:
        return {}
    device = module_device(model)
    dtype = module_dtype(model)
    lengths = [entry.embeddings.shape[0] for _, entry, _ in entries]
    maximum = max(lengths)
    width = entries[0][1].embeddings.shape[1]
    embeddings = torch.zeros(
        (len(entries), maximum, width),
        device=device,
        dtype=dtype,
    )
    position_ids = torch.zeros(
        (len(entries), maximum),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(position_ids)
    for row, ((_, encoding, _), length) in enumerate(
        zip(entries, lengths, strict=True),
    ):
        embeddings[row, :length] = encoding.embeddings.to(
            device=device,
            dtype=dtype,
        )
        position_ids[row, :length] = encoding.position_ids.to(device)
        attention_mask[row, :length] = 1
    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits
    return {sample_index: outputs[row].index_select(0, indices.to(device)) for row, (sample_index, _, indices) in enumerate(entries)}


def _bucketed_encoding_logits(
    model: nn.Module,
    entries: list[_EncodingEntry],
    batch_size: int,
) -> dict[int, Tensor]:
    ordered = sorted(
        entries,
        key=lambda entry: (entry[1].embeddings.shape[0], entry[0]),
    )
    logits: dict[int, Tensor] = {}
    for start in range(0, len(ordered), batch_size):
        logits.update(
            _batched_encoding_logits(
                model,
                ordered[start : start + batch_size],
            )
        )
    return logits


def _teacher_forced_metrics(  # noqa: C901 - One bounded batching orchestration.
    model: nn.Module,
    adapter: InputEmbeddingAdapter,
    samples: list[PromptSample],
    alignment: SegmentationAlignment,
    runtime: _TeacherForcedRuntime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = runtime.policy
    if runtime.baseline is None:
        raise ValueError("batched teacher forcing requires a native baseline bundle")
    if len(runtime.baseline.teacher_logits) != len(samples):
        raise ValueError("native baseline teacher logits do not match evaluation samples")
    accumulators = _new_accumulators(adapter.codec.dtype)
    prompt_positions = _new_prompt_positions()
    device = module_device(model)
    resume = runtime.resume
    manager = resume.manager
    start_sample = 0
    state = None if manager is None else manager.latest(resume.phase)
    if state is not None:
        for mode, accumulator in accumulators.items():
            accumulator.load_state_dict(state["accumulators"][mode])
        prompt_positions = {name: int(value) for name, value in state["prompt_positions"].items()}
        start_sample = int(state["next_sample"])
        restore_torch_rng_state(device, state["torch_rng"])
    targets = [torch.tensor(sample.continuation, dtype=torch.long, device=device) for sample in samples]
    with torch.inference_mode():
        prepared = _student_sequence_entries(
            adapter,
            samples,
            alignment,
            runtime.native_embeddings,
            start=start_sample,
        )
        logits = {
            mode: _bucketed_encoding_logits(
                model,
                entries,
                policy.batch_size,
            )
            for mode, entries in prepared.items()
        }
        student_prompt_positions = {mode: {index: int(indices[0]) + 1 for index, _, indices in prepared[mode]} for mode in _STUDENT_MODES}
        for start in range(start_sample, len(samples), policy.batch_size):
            stop = min(start + policy.batch_size, len(samples))
            teacher = torch.cat([runtime.baseline.teacher_logits[index].to(device) for index in range(start, stop)])
            target = torch.cat(targets[start:stop])
            compact_by_mode: list[Tensor] = []
            compact_sizes: list[int] = []
            for mode, accumulator in accumulators.items():
                student = torch.cat([logits[mode][index] for index in range(start, stop)])
                compact = accumulator.sufficient_statistics(
                    teacher,
                    student,
                    target,
                )
                compact_by_mode.append(compact)
                compact_sizes.append(compact.numel())
            transferred = torch.cat(compact_by_mode).cpu()
            offset = 0
            for mode, size in zip(accumulators, compact_sizes, strict=True):
                accumulators[mode].consume(
                    transferred[offset : offset + size],
                    target.numel(),
                )
                offset += size
            for sample_index in range(start, stop):
                sample = samples[sample_index]
                prompt_positions["native"] += len(sample.prompt)
                for mode in _STUDENT_MODES:
                    prompt_positions[mode] += student_prompt_positions[mode][sample_index]
            if manager is not None and (manager.should_snapshot(stop) or stop == len(samples)):
                manager.save(
                    resume.phase,
                    stop,
                    {
                        "completed": stop == len(samples),
                        "accumulators": {mode: accumulator.state_dict() for mode, accumulator in accumulators.items()},
                        "prompt_positions": prompt_positions,
                        "next_sample": stop,
                        "torch_rng": capture_torch_rng_state(device),
                    },
                )

    return _teacher_forced_result(accumulators, prompt_positions, len(samples))


def _calibration_logits(  # noqa: PLR0913 - Calibration operands stay explicit.
    model: nn.Module,
    adapter: InputEmbeddingAdapter,
    samples: list[PromptSample],
    alignment: SegmentationAlignment,
    native_embeddings: Tensor,
    *,
    batched: bool,
) -> dict[str, list[Tensor]]:
    paths = {mode: [] for mode in ("native", *_ACCUMULATOR_MODES)}
    prepared = _student_sequence_entries(
        adapter,
        samples,
        alignment,
        native_embeddings,
    )
    if batched:
        native = _native_batch_logits(model, list(enumerate(samples)))
        encoded = {mode: _batched_encoding_logits(model, entries) for mode, entries in prepared.items()}
        for index in range(len(samples)):
            paths["native"].append(native[index])
            for mode in _ACCUMULATOR_MODES:
                paths[mode].append(encoded[mode][index])
        return paths

    device = module_device(model)
    for sample in samples:
        full_ids = torch.tensor(
            [sample.prompt + sample.continuation],
            dtype=torch.long,
            device=device,
        )
        native_indices = torch.arange(
            len(sample.prompt) - 1,
            len(sample.prompt) + len(sample.continuation) - 1,
            device=device,
        )
        paths["native"].append(
            _selected_logits(
                model,
                input_ids=full_ids,
                indices=native_indices,
            )
        )
    for mode, entries in prepared.items():
        paths[mode] = [
            _selected_logits(
                model,
                encoding=encoding,
                indices=indices,
            )
            for _, encoding, indices in entries
        ]
    return paths


def _registered_calibration_samples(
    samples: list[PromptSample],
) -> tuple[list[PromptSample], list[PromptSample]]:
    unique = samples[:UNIQUE_SAMPLE_COUNT]
    if len(unique) != UNIQUE_SAMPLE_COUNT:
        raise ValueError("input evaluation calibration requires two samples")
    return unique, unique * (CALIBRATION_ROWS // UNIQUE_SAMPLE_COUNT)


def _calibration_measurements(
    model: nn.Module,
    adapter: InputEmbeddingAdapter,
    samples: list[PromptSample],
    alignment: SegmentationAlignment,
    native_embeddings: Tensor,
) -> dict[str, float | int]:
    unique_samples, calibration_samples = _registered_calibration_samples(samples)
    with torch.inference_mode():
        scalar = _calibration_logits(
            model,
            adapter,
            unique_samples,
            alignment,
            native_embeddings,
            batched=False,
        )
        batched = _calibration_logits(
            model,
            adapter,
            calibration_samples,
            alignment,
            native_embeddings,
            batched=True,
        )
    maximum_kl = maximum_nll_delta = maximum_logit_error = 0.0
    top1_matches = rows = tokens = 0
    device = module_device(model)
    targets = [torch.tensor(sample.continuation, dtype=torch.long, device=device) for sample in calibration_samples]
    for mode in scalar:
        for row, (actual, target) in enumerate(
            zip(batched[mode], targets, strict=True),
        ):
            expected = scalar[mode][row % UNIQUE_SAMPLE_COUNT]
            expected_float = expected.float()
            actual_float = actual.float()
            expected_log = F.log_softmax(expected_float, dim=-1)
            actual_log = F.log_softmax(actual_float, dim=-1)
            row_kl = (expected_log.exp() * (expected_log - actual_log)).sum(dim=-1)
            expected_nll = -expected_log.gather(1, target[:, None]).squeeze(1)
            actual_nll = -actual_log.gather(1, target[:, None]).squeeze(1)
            maximum_kl = max(maximum_kl, float(row_kl.max()))
            maximum_nll_delta = max(
                maximum_nll_delta,
                float((expected_nll - actual_nll).abs().max()),
            )
            maximum_logit_error = max(
                maximum_logit_error,
                float((expected_float - actual_float).abs().max()),
            )
            top1_matches += int((expected_float.argmax(dim=-1) == actual_float.argmax(dim=-1)).sum())
            rows += expected.shape[0]
            tokens += target.numel()
    return {
        "unique_sample_count": UNIQUE_SAMPLE_COUNT,
        "calibration_rows": CALIBRATION_ROWS,
        "production_batch_size": PRODUCTION_BATCH_SIZE,
        "paths": len(scalar),
        "tokens": tokens,
        "scalar_model_forwards": UNIQUE_SAMPLE_COUNT * len(scalar),
        "batched_model_forwards": len(batched),
        "maximum_kl": maximum_kl,
        "maximum_nll_delta": maximum_nll_delta,
        "top1_agreement": top1_matches / max(rows, 1),
        "maximum_logit_error": maximum_logit_error,
    }


def _calibration_root(
    options: EvaluationOptions,
) -> Path:
    try:
        return find_project_root(options.output_dir)
    except FileNotFoundError:
        return find_project_root(Path.cwd())


def _ensure_calibration(  # noqa: PLR0913 - Calibration identity inputs stay explicit.
    assets: ModelAssets,
    model: nn.Module,
    adapter: InputEmbeddingAdapter,
    samples: list[PromptSample],
    options: EvaluationOptions,
    runtime: EvaluationRuntime,
    model_fingerprint: str,
    checkpoint_fingerprint: str,
) -> dict[str, Any]:
    project_root = _calibration_root(options)
    dependency_lock_sha256 = sha256_file(project_root / "uv.lock") if runtime.dependency_lock_sha256 is None else runtime.dependency_lock_sha256
    cache_directory = (
        project_root / ".cache" / "input-evaluation-calibration" if runtime.calibration_cache_directory is None else runtime.calibration_cache_directory
    )
    selected, calibration_rows = _registered_calibration_samples(samples)
    unique_samples_sha256 = mapping_fingerprint([asdict(sample) for sample in selected])
    calibration_rows_sha256 = mapping_fingerprint(
        [asdict(sample) for sample in calibration_rows],
    )
    source_commit, _, source_state_sha256 = source_state(project_root)
    identity = CalibrationIdentity(
        model_id=assets.model_id,
        model_revision=assets.revision,
        tokenizer_revision=assets.revision,
        model_fingerprint=model_fingerprint,
        codec_checkpoint_fingerprint=checkpoint_fingerprint,
        segmentation_alignment=options.segmentation_alignment,
        source_commit=source_commit,
        source_state_sha256=source_state_sha256,
        dtype=str(module_dtype(model)),
        device=str(module_device(model)),
        production_batch_size=runtime.teacher_forced_policy.batch_size,
        unique_sample_count=UNIQUE_SAMPLE_COUNT,
        calibration_rows=CALIBRATION_ROWS,
        implementation_sha256=mapping_fingerprint(
            {
                "evaluation": sha256_file(Path(__file__)),
                "calibration": sha256_file(Path(__file__).with_name("evaluation_calibration.py")),
            }
        ),
        unique_samples_sha256=unique_samples_sha256,
        calibration_rows_sha256=calibration_rows_sha256,
        tolerance=runtime.teacher_forced_policy.calibration_tolerance(),
        dependency_lock_sha256=dependency_lock_sha256,
    )
    adapter.codec.clear_runtime_caches()
    record = load_or_build_calibration(
        cache_directory,
        identity,
        lambda: _calibration_measurements(
            model,
            adapter,
            selected,
            options.segmentation_alignment,
            assets.input_embeddings,
        ),
    )
    adapter.codec.clear_runtime_caches()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    return record.materialize(
        options.output_dir / "evaluation-calibration.json",
    )


def _prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    count = 0
    for first, second in zip(left, right, strict=False):
        if first != second:
            break
        count += 1
    return count


def _ordinary_bytes(assets: ModelAssets, token_ids: tuple[int, ...]) -> bytes:
    return b"".join(value for token_id in token_ids if (value := assets.vocabulary.payload_for(token_id)) is not None)


def _generation_metrics(  # noqa: PLR0913 - Existing evaluation inputs plus immutable baseline.
    wrapper: InputOnlyCausalLM,
    assets: ModelAssets,
    samples: list[PromptSample],
    options: EvaluationOptions,
    resume: _EvaluationResume,
    baseline: NativeBaselineBundle | None = None,
) -> tuple[list[GenerationSample], dict[str, Any]]:
    eos_ids = _eos_ids(assets)
    model = wrapper.model
    device = module_device(model)
    results: list[GenerationSample] = []
    selected = samples[: min(options.generation_samples, len(samples))]
    if baseline is not None and len(baseline.generations) != len(selected):
        raise ValueError("native baseline generations do not match evaluation samples")
    start = 0
    manager = resume.manager
    state = None if manager is None else manager.latest(resume.phase)
    if state is not None:
        results = [GenerationSample(**item) for item in state["results"]]
        start = int(state["next_sample"])
        restore_torch_rng_state(device, state["torch_rng"])

    for index in range(start, len(selected)):
        sample = selected[index]
        prompt = sample.prompt
        if baseline is None:
            generated, native_seconds = timed_call(
                lambda prompt=prompt: native_greedy_generate(
                    model,
                    prompt,
                    eos_token_ids=eos_ids,
                    max_new_tokens=options.max_new_tokens,
                ),
                device,
            )
            native = NativeGeneration(
                generated.token_ids,
                native_seconds,
                max(len(generated.token_ids), 1),
            )
        else:
            native = baseline.generations[index]
        compatibility, compatibility_seconds = timed_call(
            lambda prompt=prompt: wrapper.generate(
                prompt,
                mode="compatibility",
                eos_token_ids=eos_ids,
                max_new_tokens=options.max_new_tokens,
            ),
            device,
        )
        segmented, segmented_seconds = timed_call(
            lambda prompt=prompt: wrapper.generate(
                prompt,
                mode="segmented",
                eos_token_ids=eos_ids,
                max_new_tokens=options.max_new_tokens,
            ),
            device,
        )
        native_bytes = _ordinary_bytes(assets, native.token_ids)
        compatibility_bytes = _ordinary_bytes(assets, compatibility.token_ids)
        segmented_bytes = _ordinary_bytes(assets, segmented.token_ids)
        results.append(
            GenerationSample(
                sample=index,
                native_ids=native.token_ids,
                compatibility_ids=compatibility.token_ids,
                segmented_ids=segmented.token_ids,
                compatibility_prefix=_prefix_length(native.token_ids, compatibility.token_ids),
                segmented_prefix=_prefix_length(native.token_ids, segmented.token_ids),
                compatibility_token_similarity=SequenceMatcher(None, native.token_ids, compatibility.token_ids).ratio(),
                segmented_token_similarity=SequenceMatcher(None, native.token_ids, segmented.token_ids).ratio(),
                compatibility_byte_similarity=SequenceMatcher(None, native_bytes, compatibility_bytes).ratio(),
                segmented_byte_similarity=SequenceMatcher(None, native_bytes, segmented_bytes).ratio(),
                native_seconds=native.seconds,
                compatibility_seconds=compatibility_seconds,
                segmented_seconds=segmented_seconds,
            )
        )
        boundary = index + 1
        if manager is not None and (manager.should_snapshot(boundary) or boundary == len(selected)):
            manager.save(
                resume.phase,
                boundary,
                {
                    "completed": boundary == len(selected),
                    "results": [asdict(item) for item in results],
                    "next_sample": boundary,
                    "torch_rng": capture_torch_rng_state(device),
                },
            )

    count = max(len(results), 1)
    summary = {
        "samples": len(results),
        "compatibility_exact_fraction": sum(item.native_ids == item.compatibility_ids for item in results) / count,
        "segmented_exact_fraction": sum(item.native_ids == item.segmented_ids for item in results) / count,
        "compatibility_mean_prefix": sum(item.compatibility_prefix for item in results) / count,
        "segmented_mean_prefix": sum(item.segmented_prefix for item in results) / count,
        "compatibility_mean_byte_similarity": sum(item.compatibility_byte_similarity for item in results) / count,
        "segmented_mean_byte_similarity": sum(item.segmented_byte_similarity for item in results) / count,
        "native_mean_seconds": sum(item.native_seconds for item in results) / count,
        "compatibility_mean_seconds": sum(item.compatibility_seconds for item in results) / count,
        "segmented_mean_seconds": sum(item.segmented_seconds for item in results) / count,
    }
    return results, summary


def _markdown(metrics: dict[str, Any]) -> str:
    positions = metrics["positions"]
    segmented_ratio = positions["native_positions_per_segmented_position"]
    lines = [
        "# Input-Only Continuous Tokenizer Evaluation",
        "",
        f"- Model: `{metrics['model']['id']}`",
        f"- Revision: `{metrics['model']['revision']}`",
        f"- Native prompt positions: `{positions['native']:.2f}`",
        f"- Compatibility positions: `{positions['compatibility']:.2f}`",
        f"- Segmented positions: `{positions['segmented']:.2f}`",
        f"- Native positions/segmented position: `{segmented_ratio:.4f}`",
        "",
        "## Behavioral comparison",
        "",
        "| Mode | KL | JS | Top-1 | NLL | Perplexity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in _STUDENT_MODES:
        item = metrics["teacher_forced"][mode]
        lines.append(
            f"| {mode} | {item['mean_kl']:.6f} | {item['mean_js']:.6f} | "
            f"{item['top1_agreement']:.4f} | {item['student_nll']:.6f} | "
            f"{item['student_perplexity']:.4f} |"
        )
    native_continuation = metrics["diagnostics"]["native_continuation"]
    lines += [
        "",
        "## Mechanism-only diagnostic",
        "",
        (f"- Native-continuation compressed-prompt KL: `{native_continuation['teacher_forced']['mean_kl']:.6f}`"),
        "- This diagnostic is excluded from acceptance, claims, and performance evidence.",
        "",
        "The backbone, LM head, output tokenizer, and model control embeddings were frozen.",
        "Segmented generation is exploratory; this report makes no output-density claim.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_input_replacement(
    assets: ModelAssets,
    checkpoint: Path,
    options: EvaluationOptions,
    runtime: EvaluationRuntime | None = None,
) -> dict[str, Any]:
    runtime = EvaluationRuntime() if runtime is None else runtime
    selected_device = resolve_model_device(runtime.device, runtime.frozen_model)
    torch.manual_seed(options.seed)
    own_session = runtime.session is None
    session = EvaluationSession() if runtime.session is None else runtime.session
    loaded = session.load_adapter(assets, checkpoint, selected_device)
    adapter = loaded.adapter
    fingerprint = loaded.fingerprint
    model = load_frozen_causal_lm(assets, selected_device) if runtime.frozen_model is None else runtime.frozen_model
    _require_source_dtype(assets, adapter, model)
    wrapper = InputOnlyCausalLM(model, adapter, segmentation_alignment=options.segmentation_alignment)
    before = session.bind_model(model)
    samples = session.prepared_samples(assets, options)
    calibration = _ensure_calibration(
        assets,
        model,
        adapter,
        samples,
        options,
        runtime,
        before,
        fingerprint,
    )
    baseline = _native_baseline(
        session,
        assets,
        model,
        samples,
        options,
        runtime.teacher_forced_policy,
    )
    if options.performance_prompts > len(samples):
        raise ValueError(
            "performance prompts must not exceed sampled evaluation prompts",
        )
    adapter.codec.clear_runtime_caches()
    teacher_runtime = _TeacherForcedRuntime(
        _EvaluationResume(
            runtime.resume_manager,
            f"{runtime.resume_phase}-teacher-forced",
        ),
        assets.input_embeddings,
        baseline,
        runtime.teacher_forced_policy,
    )
    teacher_forced, positions, native_continuation_diagnostic = _teacher_forced_metrics(
        model,
        adapter,
        samples,
        options.segmentation_alignment,
        teacher_runtime,
    )
    adapter.codec.clear_runtime_caches()
    performance = benchmark_model_prefill(
        model,
        adapter,
        tuple(sample.prompt for sample in samples[: options.performance_prompts]),
        PrefillBenchmarkOptions(
            warmups=options.warmups,
            repetitions=options.repetitions,
            segmentation_alignment=options.segmentation_alignment,
        ),
    )
    adapter.codec.clear_runtime_caches()
    generations, generation_summary = _generation_metrics(
        wrapper,
        assets,
        samples,
        options,
        _EvaluationResume(runtime.resume_manager, f"{runtime.resume_phase}-generation"),
        baseline,
    )
    if own_session:
        session.verify_model(model)

    metrics: dict[str, Any] = {
        "kind": "llm_metrics",
        "model": {
            "id": assets.model_id,
            "revision": assets.revision,
            "dtype": str(module_dtype(model)),
            "source_dtype": str(assets.input_embeddings.dtype),
            "parameter_fingerprint": before,
            "tie_word_embeddings": tie_word_embeddings(assets.config),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": fingerprint,
        },
        "options": {**asdict(options), "output_dir": str(options.output_dir)},
        "teacher_forced": teacher_forced,
        "diagnostics": {
            "native_continuation": native_continuation_diagnostic,
        },
        "positions": positions,
        "segmentation_alignment": options.segmentation_alignment,
        "generation": generation_summary,
        "performance": performance,
        "environment": runtime_environment(selected_device),
        "measurement": {
            "process_isolated": False,
            "output_density_in_scope": False,
            "model_timing_input_cache": "warm",
            "performance": performance["measurement"],
            "native_baseline": _baseline_measurement(baseline),
            "teacher_forced_policy": asdict(runtime.teacher_forced_policy),
            "calibration": calibration,
            "evaluation_telemetry": _evaluation_telemetry(
                session,
                len(samples),
                runtime.teacher_forced_policy,
                selected_device,
            ),
        },
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(options.output_dir / "llm-metrics.json", metrics)
    performance_artifact = {
        "kind": "performance_metrics",
        "model": metrics["model"],
        "checkpoint": metrics["checkpoint"],
        "performance": performance,
        "measurement": performance["measurement"],
        "prompt_set_sha256": performance["measurement"]["prompt_set_sha256"],
        "analytical_flops_scope": "frozen_text_backbone_full_and_linear_attention_estimate",
        "prompt_cache_measurement": "materialized_cache_tensor_storage",
        "generation_latency": generation_summary,
        "environment": metrics["environment"],
    }
    write_json_atomic(options.output_dir / "performance-metrics.json", performance_artifact)
    sample_rows = "".join(json.dumps(asdict(sample), sort_keys=True) + "\n" for sample in generations)
    write_text_atomic(options.output_dir / "samples.jsonl", sample_rows)
    write_text_atomic(options.output_dir / "llm-report.md", _markdown(metrics))
    return metrics


def evaluate_input_selection(
    assets: ModelAssets,
    checkpoint: Path,
    options: EvaluationOptions,
    runtime: EvaluationRuntime | None = None,
) -> dict[str, Any]:
    if options.dataset_split != "validation":
        raise ValueError("input candidate selection requires the validation split")
    runtime = EvaluationRuntime(resume_phase="input-selection-evaluation") if runtime is None else runtime
    selected_device = resolve_model_device(runtime.device, runtime.frozen_model)
    torch.manual_seed(options.seed)
    own_session = runtime.session is None
    session = EvaluationSession() if runtime.session is None else runtime.session
    loaded = session.load_adapter(assets, checkpoint, selected_device)
    model = load_frozen_causal_lm(assets, selected_device) if runtime.frozen_model is None else runtime.frozen_model
    _require_source_dtype(assets, loaded.adapter, model)
    before = session.bind_model(model)
    samples = session.prepared_samples(assets, options)
    calibration = _ensure_calibration(
        assets,
        model,
        loaded.adapter,
        samples,
        options,
        runtime,
        before,
        loaded.fingerprint,
    )
    baseline = _native_baseline(
        session,
        assets,
        model,
        samples,
        options,
        runtime.teacher_forced_policy,
    )
    loaded.adapter.codec.clear_runtime_caches()
    teacher_forced, positions, native_continuation_diagnostic = _teacher_forced_metrics(
        model,
        loaded.adapter,
        samples,
        options.segmentation_alignment,
        _TeacherForcedRuntime(
            _EvaluationResume(
                runtime.resume_manager,
                f"{runtime.resume_phase}-teacher-forced",
            ),
            assets.input_embeddings,
            baseline,
            runtime.teacher_forced_policy,
        ),
    )
    wrapper = InputOnlyCausalLM(
        model,
        loaded.adapter,
        segmentation_alignment=options.segmentation_alignment,
    )
    loaded.adapter.codec.clear_runtime_caches()
    _, generation_summary = _generation_metrics(
        wrapper,
        assets,
        samples,
        options,
        _EvaluationResume(
            runtime.resume_manager,
            f"{runtime.resume_phase}-generation",
        ),
        baseline,
    )
    if own_session:
        session.verify_model(model)
    return {
        "kind": "input_selection_metrics",
        "model": {
            "id": assets.model_id,
            "revision": assets.revision,
            "dtype": str(module_dtype(model)),
            "source_dtype": str(assets.input_embeddings.dtype),
            "parameter_fingerprint": before,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": loaded.fingerprint,
        },
        "selection_split": options.dataset_split,
        "untouched_by_training": True,
        "segmentation_alignment": options.segmentation_alignment,
        "teacher_forced": teacher_forced,
        "diagnostics": {
            "native_continuation": native_continuation_diagnostic,
        },
        "positions": positions,
        "generation": generation_summary,
        "samples": len(samples),
        "options": {**asdict(options), "output_dir": str(options.output_dir)},
        "environment": runtime_environment(selected_device),
        "measurement": {
            "native_baseline": _baseline_measurement(baseline),
            "teacher_forced_policy": asdict(runtime.teacher_forced_policy),
            "calibration": calibration,
            "evaluation_telemetry": _evaluation_telemetry(
                session,
                len(samples),
                runtime.teacher_forced_policy,
                selected_device,
            ),
        },
    }
