from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Self, final

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.artifacts.store import load_json_object, write_json_atomic
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.constants import CODEC_EOS
from continuous_tokenizer.contracts.output import OutputGateSpec
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.output.events import ControlEvent
from continuous_tokenizer.output.feedback import NativeByteSegmenter
from continuous_tokenizer.output.targets import (
    NativeTrajectoryOptions,
    OutputPackingInfeasibleError,
    native_head_trajectory,
    pack_native_tokens,
    pack_native_trajectory,
)
from continuous_tokenizer.runtime.tensors import tensor_bytes
from continuous_tokenizer.runtime.timing import timed_call

OUTPUT_FEEDBACK_POLICY: Final = "exact_selected_native_token_feedback"
OUTPUT_TRAJECTORY_CACHE_ENV: Final = "CONTINUOUS_TOKENIZER_TRAJECTORY_CACHE_DIR"
TRAJECTORY_TERMINATION_CODES: Final = {
    "max_native_tokens": 0,
    "max_bytes": 1,
    "stop_control": 2,
}


@final
@dataclass(frozen=True, slots=True)
class OutputTrajectoryOptions:
    max_span: int
    stop_control_ids: frozenset[int] = frozenset()
    max_native_tokens: int = 16
    max_bytes: int = 4096

    def __post_init__(self) -> None:
        if min(self.max_span, self.max_native_tokens, self.max_bytes) < 1:
            raise ValueError("output trajectory limits must be positive")

    @property
    def native(self) -> NativeTrajectoryOptions:
        return NativeTrajectoryOptions(
            stop_control_ids=self.stop_control_ids,
            max_native_tokens=self.max_native_tokens,
            max_bytes=self.max_bytes,
        )


def _strict_offsets(
    offsets: tuple[int, ...],
    *,
    expected_parts: int,
    total: int,
) -> bool:
    return (
        bool(offsets)
        and len(offsets) == expected_parts + 1
        and offsets[0] == 0
        and offsets[-1] == total
        and all(left < right for left, right in pairwise(offsets))
    )


@final
@dataclass(frozen=True, slots=True)
class PreparedOutputCorpus:
    hidden: Tensor
    event_targets: Tensor
    byte_targets: Tensor
    byte_mask: Tensor
    target_native_token_ids: Tensor
    target_native_token_offsets: tuple[int, ...]
    stop_targets: Tensor
    payload_bytes: Tensor
    sequence_offsets: tuple[int, ...]
    native_token_ids: Tensor
    native_sequence_offsets: tuple[int, ...]
    termination_reasons: Tensor

    def __post_init__(self) -> None:
        self._validate_tensor_shapes()
        self._validate_offsets()

    def _validate_tensor_shapes(self) -> None:
        examples = self.hidden.shape[0]
        if self.hidden.ndim != 2:
            raise ValueError("prepared output hidden states must be a matrix")
        if self.event_targets.shape != (examples,):
            raise ValueError("prepared output event targets have the wrong shape")
        if self.byte_targets.shape != self.byte_mask.shape:
            raise ValueError("prepared output byte targets and mask must have identical shapes")
        if self.byte_targets.ndim != 2 or self.byte_targets.shape[0] != examples:
            raise ValueError("prepared output byte targets have the wrong shape")
        for name, tensor in (
            ("stop targets", self.stop_targets),
            ("payload bytes", self.payload_bytes),
        ):
            if tensor.shape != (examples,):
                raise ValueError(f"prepared output {name} have the wrong shape")

    def _validate_offsets(self) -> None:
        examples = self.hidden.shape[0]
        if not _strict_offsets(
            self.target_native_token_offsets,
            expected_parts=examples,
            total=self.target_native_token_ids.shape[0],
        ):
            raise ValueError("prepared target native-token offsets must partition every event")
        if not _strict_offsets(
            self.sequence_offsets,
            expected_parts=len(self.sequence_offsets) - 1,
            total=examples,
        ):
            raise ValueError("prepared output sequence offsets must partition every example")
        if not _strict_offsets(
            self.native_sequence_offsets,
            expected_parts=self.sequences,
            total=self.native_token_ids.shape[0],
        ):
            raise ValueError("prepared native-token offsets must partition every trajectory")
        if self.termination_reasons.shape != (self.sequences,):
            raise ValueError("prepared termination reasons must align with trajectories")

    @property
    def sequences(self) -> int:
        return len(self.sequence_offsets) - 1

    @property
    def examples(self) -> int:
        return self.hidden.shape[0]

    @property
    def tensor_bytes(self) -> int:
        tensors = (
            self.hidden,
            self.event_targets,
            self.byte_targets,
            self.byte_mask,
            self.target_native_token_ids,
            self.stop_targets,
            self.payload_bytes,
            self.native_token_ids,
            self.termination_reasons,
        )
        offsets = len(self.target_native_token_offsets) + len(self.sequence_offsets) + len(self.native_sequence_offsets)
        return sum(tensor_bytes(tensor) for tensor in tensors) + (offsets * self.event_targets.element_size())

    def bounds(self, sequence: int) -> tuple[int, int]:
        if not 0 <= sequence < self.sequences:
            raise IndexError("prepared output sequence index is out of range")
        return self.sequence_offsets[sequence], self.sequence_offsets[sequence + 1]

    def batch(
        self,
        start: int,
        stop: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        narrow_targets: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        target_width = self.byte_targets.shape[1]
        if narrow_targets:
            payload_lengths = self.payload_bytes[start:stop]
            target_width = int(payload_lengths.max().item()) + 1
        return (
            self.hidden[start:stop].to(device=device, dtype=dtype),
            self.event_targets[start:stop].to(device),
            self.byte_targets[start:stop, :target_width].to(device),
            self.byte_mask[start:stop, :target_width].to(device),
        )

    def target_native_tokens(self, event: int) -> tuple[int, ...]:
        if not 0 <= event < self.examples:
            raise IndexError("prepared output event index is out of range")
        start = self.target_native_token_offsets[event]
        stop = self.target_native_token_offsets[event + 1]
        return tuple(int(value) for value in self.target_native_token_ids[start:stop].tolist())

    def staged(self, *, device: torch.device, dtype: torch.dtype) -> PreparedOutputCorpus:
        return PreparedOutputCorpus(
            hidden=self.hidden.to(device=device, dtype=dtype),
            event_targets=self.event_targets.to(device),
            byte_targets=self.byte_targets.to(device),
            byte_mask=self.byte_mask.to(device),
            target_native_token_ids=self.target_native_token_ids,
            target_native_token_offsets=self.target_native_token_offsets,
            stop_targets=self.stop_targets.to(device),
            payload_bytes=self.payload_bytes,
            sequence_offsets=self.sequence_offsets,
            native_token_ids=self.native_token_ids,
            native_sequence_offsets=self.native_sequence_offsets,
            termination_reasons=self.termination_reasons,
        )

    def tensors(self) -> dict[str, Tensor]:
        return {
            "hidden": self.hidden.detach().cpu().contiguous(),
            "event_targets": self.event_targets.detach().cpu().contiguous(),
            "byte_targets": self.byte_targets.detach().cpu().contiguous(),
            "byte_mask": self.byte_mask.detach().cpu().contiguous(),
            "target_native_token_ids": self.target_native_token_ids.detach().cpu().contiguous(),
            "target_native_token_offsets": torch.tensor(
                self.target_native_token_offsets,
                dtype=torch.long,
            ),
            "stop_targets": self.stop_targets.detach().cpu().contiguous(),
            "payload_bytes": self.payload_bytes.detach().cpu().contiguous(),
            "sequence_offsets": torch.tensor(self.sequence_offsets, dtype=torch.long),
            "native_token_ids": self.native_token_ids.detach().cpu().contiguous(),
            "native_sequence_offsets": torch.tensor(self.native_sequence_offsets, dtype=torch.long),
            "termination_reasons": self.termination_reasons.detach().cpu().contiguous(),
        }


@final
@dataclass(frozen=True, slots=True)
class OutputCacheIdentity:
    source_commit: str
    source_dirty: bool
    source_state_sha256: str
    dependency_lock_sha256: str
    model_revision: str
    model_config_sha256: str
    frozen_backbone_fingerprint: str
    tokenizer_revision: str


@final
@dataclass(frozen=True, slots=True)
class OutputCorpusPreparation:
    identity: OutputCacheIdentity
    split: str
    trajectory: OutputTrajectoryOptions
    cache_directory: Path | None = None


@final
@dataclass(frozen=True, slots=True)
class _OutputCacheInfoContext:
    key: str
    split: str
    hit: bool
    directory: Path
    corpus_sha256: str
    build_seconds: float = 0.0
    load_seconds: float = 0.0
    write_seconds: float = 0.0


@final
@dataclass(frozen=True, slots=True)
class OutputCorpusCacheInfo:
    key: str
    split: str
    hit: bool
    path: str
    corpus_sha256: str
    trajectory_sha256: str
    feedback_policy: str
    build_seconds: float
    load_seconds: float
    write_seconds: float
    tensor_bytes: int
    sequences: int
    examples: int

    @classmethod
    def from_corpus(
        cls,
        corpus: PreparedOutputCorpus,
        context: _OutputCacheInfoContext,
        *,
        trajectory_sha256: str,
    ) -> Self:
        return cls(
            key=context.key,
            split=context.split,
            hit=context.hit,
            path=str(context.directory),
            corpus_sha256=context.corpus_sha256,
            trajectory_sha256=trajectory_sha256,
            feedback_policy=OUTPUT_FEEDBACK_POLICY,
            build_seconds=context.build_seconds,
            load_seconds=context.load_seconds,
            write_seconds=context.write_seconds,
            tensor_bytes=corpus.tensor_bytes,
            sequences=corpus.sequences,
            examples=corpus.examples,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prepared_output_tensors_digest(tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode())
        dtype = str(value.dtype).encode()
        digest.update(len(dtype).to_bytes(8, "big"))
        digest.update(dtype)
        digest.update(len(value.shape).to_bytes(8, "big"))
        for dimension in value.shape:
            digest.update(dimension.to_bytes(8, "big"))
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def prepared_output_corpus_digest(corpus: PreparedOutputCorpus) -> str:
    return _prepared_output_tensors_digest(corpus.tensors())


def output_trajectory_cache_directory() -> Path:
    configured = os.environ.get(OUTPUT_TRAJECTORY_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path(sys.prefix) / ".cache" / "continuous-tokenizer" / "output-trajectories"


def oracle_ceiling_passes_gates(
    ceiling: Mapping[str, float | int | bool | None],
    gates: OutputGateSpec,
) -> bool:
    exact_rate = ceiling["exact_native_sequence_rate_ceiling"]
    token_density = ceiling["native_tokens_per_attempted_macro_step_ceiling"]
    return (
        ceiling["feasible"] is True
        and exact_rate is not None
        and exact_rate >= gates.minimum_direct_feedback_equality
        and token_density is not None
        and token_density >= gates.minimum_native_tokens_per_attempted_macro_step
    )


def native_head_oracle_ceilings(
    corpus: PreparedOutputCorpus,
    vocabulary: ByteVocabulary,
    *,
    span_limits: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, dict[str, float | int | bool | None]]:
    segmenter = NativeByteSegmenter(vocabulary)
    native_sequences = tuple(
        tuple(int(value) for value in corpus.native_token_ids[corpus.native_sequence_offsets[index] : corpus.native_sequence_offsets[index + 1]].tolist())
        for index in range(corpus.sequences)
    )
    if not native_sequences:
        raise ValueError("native-head oracle ceilings require generated native tokens")
    native_tokens = sum(len(sequence) for sequence in native_sequences)
    ceilings: dict[str, dict[str, float | int | bool | None]] = {}
    for max_span in span_limits:
        if max_span < 1:
            raise ValueError("oracle ceiling span limits must be positive")
        exact_events = 0
        events = 0
        payload_bytes = 0
        try:
            for sequence in native_sequences:
                sequence_events, targets = pack_native_tokens(
                    sequence,
                    vocabulary,
                    max_span=max_span,
                )
                for event, target in zip(sequence_events, targets, strict=True):
                    exact_events += segmenter.feedback(event).token_ids == target
                    payload_bytes += 0 if isinstance(event, ControlEvent) else len(event.data)
                events += len(sequence_events)
        except OutputPackingInfeasibleError:
            ceilings[str(max_span)] = {
                "feasible": False,
                "native_tokens": native_tokens,
                "events": None,
                "exact_native_sequence_rate_ceiling": None,
                "bytes_per_event_ceiling": None,
                "native_tokens_per_attempted_macro_step_ceiling": None,
            }
            continue
        ceilings[str(max_span)] = {
            "feasible": True,
            "native_tokens": native_tokens,
            "events": events,
            "exact_native_sequence_rate_ceiling": exact_events / max(events, 1),
            "bytes_per_event_ceiling": payload_bytes / max(events, 1),
            "native_tokens_per_attempted_macro_step_ceiling": native_tokens / max(events, 1),
        }
    return ceilings


def _sequence_digest(sequences: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(len(sequence).to_bytes(8, "big"))
        for token_id in sequence:
            digest.update(token_id.to_bytes(8, "big"))
    return digest.hexdigest()


@final
@dataclass(frozen=True, slots=True)
class _OutputCacheEnvironment:
    transformers_version: str
    model_implementation: str
    dtype: torch.dtype


def _cache_descriptor(
    preparation: OutputCorpusPreparation,
    environment: _OutputCacheEnvironment,
    corpus_sha256: str,
) -> dict[str, Any]:
    trajectory = preparation.trajectory
    return {
        **asdict(preparation.identity),
        "transformers_version": environment.transformers_version,
        "model_implementation": environment.model_implementation,
        "corpus_sha256": corpus_sha256,
        "split": preparation.split,
        "max_span": trajectory.max_span,
        "stop_control_ids": sorted(trajectory.stop_control_ids),
        "max_native_tokens": trajectory.max_native_tokens,
        "max_bytes": trajectory.max_bytes,
        "dtype": str(environment.dtype),
        "feedback_policy": OUTPUT_FEEDBACK_POLICY,
    }


def _cache_key(descriptor: dict[str, Any]) -> str:
    return mapping_fingerprint(descriptor)


@torch.no_grad()
def build_prepared_output_corpus(
    backbone: FrozenBackbone,
    vocabulary: ByteVocabulary,
    sequences: tuple[tuple[int, ...], ...],
    options: OutputTrajectoryOptions,
) -> PreparedOutputCorpus:
    if not sequences:
        raise ValueError("output corpus must contain at least one sequence")
    max_span = options.max_span
    control_rows = {token_id: row + 1 for row, token_id in enumerate(vocabulary.control_ids)}
    hidden_parts = []
    event_targets: list[int] = []
    byte_targets: list[list[int]] = []
    byte_masks: list[list[bool]] = []
    target_native_token_ids: list[int] = []
    target_native_token_offsets = [0]
    stop_targets: list[bool] = []
    payload_bytes: list[int] = []
    native_token_ids: list[int] = []
    termination_reasons: list[int] = []
    offsets = [0]
    native_offsets = [0]
    native_options = options.native
    for prompt_token_ids in sequences:
        trajectory = native_head_trajectory(
            backbone,
            vocabulary,
            prompt_token_ids,
            native_options,
        )
        packed = pack_native_trajectory(
            trajectory,
            vocabulary,
            max_span=max_span,
        )
        hidden_parts.append(packed.hidden)
        native_token_ids.extend(trajectory.native_token_ids)
        termination_reasons.append(TRAJECTORY_TERMINATION_CODES[trajectory.termination_reason])
        for event, native_targets in zip(
            packed.events,
            packed.target_native_token_ids,
            strict=True,
        ):
            target = [0] * (max_span + 1)
            mask = [False] * (max_span + 1)
            if isinstance(event, ControlEvent):
                event_targets.append(control_rows[event.token_id])
                stop_target = event.token_id in native_options.stop_control_ids
                event_payload_bytes = 0
            else:
                event_targets.append(0)
                values = [*event.data, CODEC_EOS]
                target[: len(values)] = values
                mask[: len(values)] = [True] * len(values)
                stop_target = False
                event_payload_bytes = len(event.data)
            byte_targets.append(target)
            byte_masks.append(mask)
            target_native_token_ids.extend(native_targets)
            target_native_token_offsets.append(len(target_native_token_ids))
            stop_targets.append(stop_target)
            payload_bytes.append(event_payload_bytes)
        offsets.append(len(event_targets))
        native_offsets.append(len(native_token_ids))
    return PreparedOutputCorpus(
        hidden=torch.cat(hidden_parts),
        event_targets=torch.tensor(event_targets, dtype=torch.long),
        byte_targets=torch.tensor(byte_targets, dtype=torch.long),
        byte_mask=torch.tensor(byte_masks, dtype=torch.bool),
        target_native_token_ids=torch.tensor(target_native_token_ids, dtype=torch.long),
        target_native_token_offsets=tuple(target_native_token_offsets),
        stop_targets=torch.tensor(stop_targets, dtype=torch.bool),
        payload_bytes=torch.tensor(payload_bytes, dtype=torch.long),
        sequence_offsets=tuple(offsets),
        native_token_ids=torch.tensor(native_token_ids, dtype=torch.long),
        native_sequence_offsets=tuple(native_offsets),
        termination_reasons=torch.tensor(termination_reasons, dtype=torch.uint8),
    )


def _load_prepared_output_corpus(
    directory: Path,
    descriptor: dict[str, Any],
) -> tuple[PreparedOutputCorpus, str] | None:
    metadata_path = directory / "metadata.json"
    tensors_path = directory / "tensors.safetensors"
    if not metadata_path.is_file() or not tensors_path.is_file():
        return None
    try:
        metadata = load_json_object(metadata_path)
        expected_metadata = {
            "descriptor",
            "tensors_sha256",
            "trajectory_sha256",
            "tensor_bytes",
            "sequences",
            "examples",
        }
        valid_metadata = (
            set(metadata) == expected_metadata and metadata.get("descriptor") == descriptor and metadata.get("tensors_sha256") == sha256_file(tensors_path)
        )
        if not valid_metadata:
            return None
        tensors = load_file(tensors_path, device="cpu")
        corpus = PreparedOutputCorpus(
            hidden=tensors["hidden"],
            event_targets=tensors["event_targets"],
            byte_targets=tensors["byte_targets"],
            byte_mask=tensors["byte_mask"],
            target_native_token_ids=tensors["target_native_token_ids"],
            target_native_token_offsets=_integer_tuple(tensors["target_native_token_offsets"]),
            stop_targets=tensors["stop_targets"],
            payload_bytes=tensors["payload_bytes"],
            sequence_offsets=_integer_tuple(tensors["sequence_offsets"]),
            native_token_ids=tensors["native_token_ids"],
            native_sequence_offsets=_integer_tuple(tensors["native_sequence_offsets"]),
            termination_reasons=tensors["termination_reasons"],
        )
    except Exception:  # noqa: BLE001 - Any corrupt cache artifact should trigger a rebuild.
        return None
    trajectory_sha256 = _prepared_output_tensors_digest(tensors)
    if metadata["trajectory_sha256"] != trajectory_sha256:
        return None
    return corpus, trajectory_sha256


def _integer_tuple(tensor: Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.tolist())


def _store_prepared_output_corpus(
    directory: Path,
    descriptor: dict[str, Any],
    corpus: PreparedOutputCorpus,
    tensors: dict[str, Tensor],
    *,
    trajectory_sha256: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensors_path = directory / "tensors.safetensors"
    temporary = tensors_path.with_name(f".{tensors_path.name}.{os.getpid()}.tmp")
    save_file(tensors, temporary)
    Path(temporary).replace(tensors_path)
    write_json_atomic(
        directory / "metadata.json",
        {
            "descriptor": descriptor,
            "tensors_sha256": sha256_file(tensors_path),
            "trajectory_sha256": trajectory_sha256,
            "tensor_bytes": corpus.tensor_bytes,
            "sequences": corpus.sequences,
            "examples": corpus.examples,
        },
    )


def prepare_output_corpus(
    backbone: FrozenBackbone,
    vocabulary: ByteVocabulary,
    sequences: tuple[tuple[int, ...], ...],
    preparation: OutputCorpusPreparation,
) -> tuple[PreparedOutputCorpus, OutputCorpusCacheInfo]:
    corpus_sha256 = _sequence_digest(sequences)
    descriptor = _cache_descriptor(
        preparation,
        _OutputCacheEnvironment(
            transformers_version=version("transformers"),
            model_implementation=(f"{type(backbone.source_model).__module__}.{type(backbone.source_model).__qualname__}"),
            dtype=backbone.dtype,
        ),
        corpus_sha256=corpus_sha256,
    )
    key = _cache_key(descriptor)
    root = output_trajectory_cache_directory() if preparation.cache_directory is None else preparation.cache_directory
    directory = root / key
    loaded_result, load_seconds = timed_call(
        lambda: _load_prepared_output_corpus(directory, descriptor),
        backbone.device,
    )
    if loaded_result is not None:
        loaded, trajectory_sha256 = loaded_result
        return loaded, OutputCorpusCacheInfo.from_corpus(
            loaded,
            _OutputCacheInfoContext(
                key=key,
                split=preparation.split,
                hit=True,
                directory=directory,
                corpus_sha256=corpus_sha256,
                load_seconds=load_seconds,
            ),
            trajectory_sha256=trajectory_sha256,
        )
    prepared, build_seconds = timed_call(
        lambda: build_prepared_output_corpus(
            backbone,
            vocabulary,
            sequences,
            preparation.trajectory,
        ),
        backbone.device,
    )
    started = perf_counter()
    prepared_tensors = prepared.tensors()
    prepared_sha256 = _prepared_output_tensors_digest(prepared_tensors)
    _store_prepared_output_corpus(
        directory,
        descriptor,
        prepared,
        prepared_tensors,
        trajectory_sha256=prepared_sha256,
    )
    write_seconds = perf_counter() - started
    stored_result = _load_prepared_output_corpus(directory, descriptor)
    if stored_result is None:
        raise RuntimeError("stored output trajectory differs from the prepared corpus")
    stored, trajectory_sha256 = stored_result
    return stored, OutputCorpusCacheInfo.from_corpus(
        stored,
        _OutputCacheInfoContext(
            key=key,
            split=preparation.split,
            hit=False,
            directory=directory,
            corpus_sha256=corpus_sha256,
            build_seconds=build_seconds,
            load_seconds=load_seconds,
            write_seconds=write_seconds,
        ),
        trajectory_sha256=trajectory_sha256,
    )
