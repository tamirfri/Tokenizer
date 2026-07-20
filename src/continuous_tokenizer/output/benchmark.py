from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from functools import partial
from statistics import median
from typing import Any, Literal, final

import torch

from continuous_tokenizer.codec.compute import backbone_forward_flops
from continuous_tokenizer.contracts.claim_derivation import (
    MINIMUM_LATENCY_REPETITIONS,
    MINIMUM_LATENCY_WARMUPS,
)
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.output.events import ControlEvent, OutputEvent
from continuous_tokenizer.output.generation import OutputGenerationResult, OutputOnlyGenerator
from continuous_tokenizer.output.targets import (
    NativeHeadGeneration,
    NativeTrajectoryOptions,
    native_head_generation,
    native_output_head,
)
from continuous_tokenizer.runtime.progress import log_event
from continuous_tokenizer.runtime.tensors import module_bytes
from continuous_tokenizer.runtime.timing import (
    TIMING_OBSERVATION_SCHEMA_VERSION,
    TimingObservation,
    timed_observation,
    timing_summary,
)

OUTPUT_BENCHMARK_SCHEMA_VERSION = 1
type _SnapshotCache = dict[tuple[str, object], dict[str, Any]]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _event_output(
    events: tuple[OutputEvent, ...],
) -> tuple[bytes, tuple[int, ...], tuple[tuple[str, int], ...]]:
    data = bytearray()
    controls: list[int] = []
    semantics: list[tuple[str, int]] = []
    for event in events:
        if isinstance(event, ControlEvent):
            controls.append(event.token_id)
            semantics.append(("control", event.token_id))
        else:
            data.extend(event.data)
            semantics.extend(("byte", value) for value in event.data)
    return bytes(data), tuple(controls), tuple(semantics)


def _semantic_snapshot(
    data: bytes,
    controls: tuple[int, ...],
    semantics: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    return {
        "bytes": len(data),
        "bytes_sha256": _sha256_bytes(data),
        "control_ids": controls,
        "semantic_trajectory": semantics,
        "semantic_sha256": mapping_fingerprint(semantics),
    }


@final
@dataclass(frozen=True, slots=True)
class OutputBenchmark:
    schema_version: int
    measurement: dict[str, Any]
    raw_repetitions: tuple[dict[str, Any], ...]
    native: dict[str, float | int]
    candidate: dict[str, float | int]
    exact_trajectory_equivalence: bool
    speedup_claimable: bool
    latency_speedup: float | None
    candidate_state_bytes: int
    reference_state_bytes: int
    candidate_reference_state_ratio: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@final
@dataclass(frozen=True, slots=True)
class OutputBenchmarkOptions:
    warmups: int
    repetitions: int
    stop_control_ids: frozenset[int]
    max_macro_steps: int
    max_bytes: int

    def __post_init__(self) -> None:
        if self.repetitions < 1 or self.warmups < 0:
            raise ValueError("benchmark repetitions must be positive and warmups non-negative")


def _native_output(
    trajectory: NativeHeadGeneration,
    generator: OutputOnlyGenerator,
) -> tuple[bytes, tuple[int, ...], tuple[tuple[str, int], ...]]:
    data = bytearray()
    controls: list[int] = []
    semantics: list[tuple[str, int]] = []
    vocabulary = generator.segmenter.vocabulary
    for token_id in trajectory.native_token_ids:
        payload = vocabulary.payload_for(token_id)
        if payload is None:
            controls.append(token_id)
            semantics.append(("control", token_id))
        else:
            data.extend(payload)
            semantics.extend(("byte", value) for value in payload)
    return bytes(data), tuple(controls), tuple(semantics)


def _native_compute(
    trajectory: NativeHeadGeneration,
    generator: OutputOnlyGenerator,
    prompt_tokens: int,
) -> tuple[int, int]:
    config = getattr(generator.backbone.source_model, "config", None)
    if config is None:
        return 0, 0
    backbone = backbone_forward_flops(config, prompt_tokens)
    context = prompt_tokens
    feedback_tokens = trajectory.model_calls - 1
    for _ in range(feedback_tokens):
        context += 1
        backbone += backbone_forward_flops(config, 1, context)
    hidden = int(config.hidden_size)
    vocabulary = int(config.vocab_size)
    native_head = trajectory.attempted_native_tokens * 2 * hidden * vocabulary
    return backbone, native_head


def _native_observation(
    trajectory: NativeHeadGeneration,
    timing: TimingObservation,
    generator: OutputOnlyGenerator,
    prompt_tokens: int,
    snapshots: _SnapshotCache,
) -> dict[str, Any]:
    snapshot_started = time.perf_counter()
    key = (
        "native",
        (
            trajectory.native_token_ids,
            trajectory.attempted_native_tokens,
            trajectory.model_calls,
            prompt_tokens,
        ),
    )
    snapshot = snapshots.get(key)
    if snapshot is None:
        data, controls, semantics = _native_output(trajectory, generator)
        backbone_flops, head_flops = _native_compute(
            trajectory,
            generator,
            prompt_tokens,
        )
        snapshot = {
            **_semantic_snapshot(data, controls, semantics),
            "analytical_backbone_flops": backbone_flops,
            "analytical_local_flops": head_flops,
            "total_analytical_flops": backbone_flops + head_flops,
        }
        snapshots[key] = snapshot
    snapshot_hash_seconds = time.perf_counter() - snapshot_started
    telemetry = dict(trajectory.runtime_telemetry)
    telemetry["synchronization_count"] = int(telemetry["synchronization_count"]) + timing.synchronization_count
    return {
        "schema_version": OUTPUT_BENCHMARK_SCHEMA_VERSION,
        "latency_seconds": timing.wall_seconds,
        "timing": timing.to_dict(),
        "subphases": telemetry,
        "snapshot_hash_seconds": snapshot_hash_seconds,
        **snapshot,
        "stop_control": trajectory.termination_reason == "stop_control",
        "native_tokens": len(trajectory.native_token_ids),
        "attempted_macro_steps": trajectory.attempted_native_tokens,
        "model_calls": trajectory.model_calls,
        "final_cache_bytes": trajectory.final_cache_bytes,
        "peak_cache_bytes": trajectory.peak_cache_bytes,
        "native_head_invocations": trajectory.attempted_native_tokens,
    }


def _candidate_observation(
    result: OutputGenerationResult,
    timing: TimingObservation,
    snapshots: _SnapshotCache,
) -> dict[str, Any]:
    snapshot_started = time.perf_counter()
    key = ("candidate", result.events)
    snapshot = snapshots.get(key)
    if snapshot is None:
        snapshot = _semantic_snapshot(*_event_output(result.events))
        snapshots[key] = snapshot
    snapshot_hash_seconds = time.perf_counter() - snapshot_started
    telemetry = dict(result.runtime_telemetry)
    telemetry["synchronization_count"] = int(telemetry["synchronization_count"]) + timing.synchronization_count
    return {
        "schema_version": OUTPUT_BENCHMARK_SCHEMA_VERSION,
        "latency_seconds": timing.wall_seconds,
        "timing": timing.to_dict(),
        "subphases": telemetry,
        "snapshot_hash_seconds": snapshot_hash_seconds,
        **snapshot,
        "stop_control": result.termination_reason == "stop_control",
        "native_tokens": result.native_tokens_represented,
        "attempted_macro_steps": result.macro_steps,
        "model_calls": result.model_calls,
        "final_cache_bytes": result.final_cache_bytes,
        "peak_cache_bytes": result.peak_cache_bytes,
        "analytical_backbone_flops": result.analytical_backbone_flops,
        "analytical_local_flops": result.analytical_codec_decode_flops,
        "total_analytical_flops": (result.analytical_backbone_flops + result.analytical_codec_decode_flops),
        "native_head_invocations": result.native_head_invocations,
    }


def _summary(
    observations: list[dict[str, Any]],
) -> dict[str, float | int]:
    latency = timing_summary(
        [float(observation["latency_seconds"]) for observation in observations],
    )
    numeric = (
        "bytes",
        "native_tokens",
        "attempted_macro_steps",
        "model_calls",
        "final_cache_bytes",
        "peak_cache_bytes",
        "analytical_backbone_flops",
        "analytical_local_flops",
        "total_analytical_flops",
        "native_head_invocations",
    )
    result: dict[str, float | int] = {
        "latency_median_seconds": latency["median"],
        "latency_p95_seconds": latency["p95"],
        **{name: int(median(int(observation[name]) for observation in observations)) for name in numeric},
    }
    for phase in (
        "preparation",
        "backbone",
        "output_decode",
        "feedback",
        "cache_accounting",
    ):
        summary = timing_summary([float(observation["subphases"][f"{phase}_seconds"]) for observation in observations])
        result[f"{phase}_median_seconds"] = summary["median"]
        result[f"{phase}_p95_seconds"] = summary["p95"]
    for name in (
        "host_to_device_bytes",
        "device_to_host_bytes",
        "synchronization_count",
        "backbone_calls",
        "output_decode_calls",
        "feedback_calls",
        "cache_snapshots",
    ):
        result[name] = int(median(int(observation["subphases"][name]) for observation in observations))
    for name in (
        "peak_rss_bytes",
        "peak_mps_allocated_bytes",
        "peak_mps_driver_bytes",
    ):
        result[name] = max(int(observation["timing"][name]) for observation in observations)
    return result


def _equivalent(
    native: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    return all(
        native[name] == candidate[name]
        for name in (
            "semantic_trajectory",
            "semantic_sha256",
            "bytes_sha256",
            "control_ids",
            "stop_control",
        )
    )


type _BenchmarkVariant = Literal["native", "candidate"]


def _generate_candidate(
    generator: OutputOnlyGenerator,
    prompt: tuple[int, ...],
    options: OutputBenchmarkOptions,
) -> OutputGenerationResult:
    return generator.generate(
        prompt,
        stop_control_ids=options.stop_control_ids,
        max_macro_steps=options.max_macro_steps,
        max_bytes=options.max_bytes,
        collect_runtime_telemetry=True,
    )


def _generate_native(
    generator: OutputOnlyGenerator,
    prompt: tuple[int, ...],
    options: OutputBenchmarkOptions,
    native_token_horizon: int,
) -> NativeHeadGeneration:
    return native_head_generation(
        generator.backbone,
        generator.segmenter.vocabulary,
        prompt,
        NativeTrajectoryOptions(
            stop_control_ids=options.stop_control_ids,
            max_native_tokens=native_token_horizon,
            max_bytes=options.max_bytes,
            collect_runtime_telemetry=True,
        ),
    )


def _warm_benchmark(
    generator: OutputOnlyGenerator,
    prompts: tuple[tuple[int, ...], ...],
    options: OutputBenchmarkOptions,
    horizons: tuple[int, ...],
) -> None:
    for prompt, horizon in zip(prompts, horizons, strict=True):
        for _ in range(options.warmups):
            _generate_native(generator, prompt, options, horizon)
            _generate_candidate(generator, prompt, options)


def _setup_benchmark(
    generator: OutputOnlyGenerator,
    prompts: tuple[tuple[int, ...], ...],
    prompt_hashes: tuple[str, ...],
    options: OutputBenchmarkOptions,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    horizons: list[int] = []
    setup_runs: list[dict[str, Any]] = []
    for prompt_index, (prompt, prompt_sha256) in enumerate(
        zip(prompts, prompt_hashes, strict=True),
    ):
        candidate = _generate_candidate(generator, prompt, options)
        horizon = max(1, candidate.native_tokens_represented)
        _generate_native(generator, prompt, options, horizon)
        horizons.append(horizon)
        setup_runs.append(
            {
                "prompt_index": prompt_index,
                "prompt_sha256": prompt_sha256,
                "order": [
                    "candidate_horizon_discovery",
                    "native_matched_horizon",
                ],
                "candidate_runs": 1,
                "native_runs": 1,
                "native_token_horizon": horizon,
                "timed": False,
            },
        )
    return tuple(horizons), setup_runs


def _measure_variant(  # noqa: PLR0913 - Horizon and timing inputs stay explicit.
    generator: OutputOnlyGenerator,
    prompt: tuple[int, ...],
    options: OutputBenchmarkOptions,
    variant: _BenchmarkVariant,
    snapshots: _SnapshotCache,
    native_token_horizon: int,
) -> dict[str, Any]:
    if variant == "native":
        trajectory, timing = timed_observation(
            partial(
                _generate_native,
                generator,
                prompt,
                options,
                native_token_horizon,
            ),
            generator.backbone.device,
        )
        return _native_observation(
            trajectory,
            timing,
            generator,
            len(prompt),
            snapshots,
        )
    result, timing = timed_observation(
        partial(_generate_candidate, generator, prompt, options),
        generator.backbone.device,
    )
    return _candidate_observation(result, timing, snapshots)


def _paired_observation(  # noqa: PLR0913 - Pair identity remains explicit.
    generator: OutputOnlyGenerator,
    prompt: tuple[int, ...],
    options: OutputBenchmarkOptions,
    repetition: int,
    prompt_offset: int,
    snapshots: _SnapshotCache,
    native_token_horizon: int,
) -> dict[str, Any]:
    order: tuple[_BenchmarkVariant, ...] = ("native", "candidate")
    if (repetition + prompt_offset) % 2:
        order = order[::-1]
    measured = {
        variant: _measure_variant(
            generator,
            prompt,
            options,
            variant,
            snapshots,
            native_token_horizon,
        )
        for variant in order
    }
    return {
        "schema_version": OUTPUT_BENCHMARK_SCHEMA_VERSION,
        "repetition": repetition,
        "order": order,
        "native": measured["native"],
        "candidate": measured["candidate"],
        "exact_byte_control_stop_equivalence": _equivalent(
            measured["native"],
            measured["candidate"],
        ),
    }


@torch.inference_mode()
def benchmark_output_generation(
    generator: OutputOnlyGenerator,
    prompts: tuple[tuple[int, ...], ...],
    options: OutputBenchmarkOptions,
) -> OutputBenchmark:
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("output benchmark requires registered non-empty prompts")

    hash_started = time.perf_counter()
    prompt_hashes = tuple(mapping_fingerprint(prompt) for prompt in prompts)
    prompt_set_sha256 = mapping_fingerprint(prompts)
    prompt_hash_seconds = time.perf_counter() - hash_started
    horizons, setup_runs = _setup_benchmark(
        generator,
        prompts,
        prompt_hashes,
        options,
    )
    _warm_benchmark(generator, prompts, options, horizons)
    raw: list[dict[str, Any]] = []
    snapshots: _SnapshotCache = {}
    prompt_indices = tuple(range(len(prompts)))
    for repetition in range(options.repetitions):
        rotation = repetition % len(prompts)
        ordered_indices = prompt_indices[rotation:] + prompt_indices[:rotation]
        for prompt_offset, prompt_index in enumerate(ordered_indices):
            prompt = prompts[prompt_index]
            pair = _paired_observation(
                generator,
                prompt,
                options,
                repetition,
                prompt_offset,
                snapshots,
                horizons[prompt_index],
            )
            raw.append(
                {
                    **pair,
                    "prompt_index": prompt_index,
                    "prompt_sha256": prompt_hashes[prompt_index],
                },
            )

    exact_equivalence = all(bool(row["exact_byte_control_stop_equivalence"]) for row in raw)
    native_summary = _summary([pair["native"] for pair in raw])
    candidate_summary = _summary([pair["candidate"] for pair in raw])
    speedup = float(native_summary["latency_median_seconds"]) / float(candidate_summary["latency_median_seconds"]) if exact_equivalence else None
    candidate_state_bytes = module_bytes(generator.codec)
    reference_state_bytes = module_bytes(
        native_output_head(generator.backbone),
    )
    log_event(
        "benchmark_work_avoided",
        work="output_serialization_and_hash",
        avoided_observations=len(raw) * 2 - len(snapshots),
        distinct_semantics=len(snapshots),
    )
    return OutputBenchmark(
        schema_version=OUTPUT_BENCHMARK_SCHEMA_VERSION,
        measurement={
            "schema_version": OUTPUT_BENCHMARK_SCHEMA_VERSION,
            "timing_observation_schema_version": TIMING_OBSERVATION_SCHEMA_VERSION,
            "prompt_count": len(prompts),
            "prompt_sha256": list(prompt_hashes),
            "prompt_set_sha256": prompt_set_sha256,
            "prompt_hash_seconds": prompt_hash_seconds,
            "setup_runs": setup_runs,
            "warmups": options.warmups,
            "repetitions": options.repetitions,
            "expected_raw_repetitions": len(prompts) * options.repetitions,
            "recorded_raw_repetitions": len(raw),
            "prompt_order": "cyclic_rotation_by_repetition",
            "variant_order": "alternating_by_repetition_and_prompt_offset",
            "pairing": "same_prompt_and_repetition",
            "latency_timing": "direct_end_to_end_generation_call",
            "semantic_horizon": "candidate_native_tokens_represented",
        },
        raw_repetitions=tuple(raw),
        native=native_summary,
        candidate=candidate_summary,
        exact_trajectory_equivalence=exact_equivalence,
        speedup_claimable=(exact_equivalence and options.warmups >= MINIMUM_LATENCY_WARMUPS and options.repetitions >= MINIMUM_LATENCY_REPETITIONS),
        latency_speedup=speedup,
        candidate_state_bytes=candidate_state_bytes,
        reference_state_bytes=reference_state_bytes,
        candidate_reference_state_ratio=(candidate_state_bytes / reference_state_bytes if reference_state_bytes else None),
    )
