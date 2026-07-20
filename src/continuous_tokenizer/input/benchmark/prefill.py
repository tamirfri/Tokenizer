from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Literal

import torch
from torch import Tensor, nn

from continuous_tokenizer.codec.batches import span_bucket_width
from continuous_tokenizer.codec.compute import (
    backbone_forward_flops,
    input_encode_batch_flops,
    input_validation_flops,
)
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.input.adapter import (
    ByteRun,
    InputEmbeddingAdapter,
    SegmentationAlignment,
)
from continuous_tokenizer.input.segmentation import (
    DYNAMIC_SEGMENTATION_MAX_BYTES,
    SEGMENTATION_FRONTIERS,
    candidate_group_rows,
    segment_bytes,
)
from continuous_tokenizer.runtime.tensors import cache_tensor_bytes
from continuous_tokenizer.runtime.timing import (
    TIMING_OBSERVATION_SCHEMA_VERSION,
    timed_call,
    timed_observation,
    timing_summary,
)

PrefillMode = Literal["native", "compatibility", "segmented"]
PREFILL_BENCHMARK_SCHEMA_VERSION = 1
_PREFILL_MODES: tuple[PrefillMode, ...] = (
    "native",
    "compatibility",
    "segmented",
)


@dataclass(frozen=True, slots=True)
class PrefillBenchmarkOptions:
    warmups: int
    repetitions: int
    segmentation_alignment: SegmentationAlignment = "arbitrary"


@dataclass(frozen=True, slots=True)
class _PreparedInputContext:
    native_input_ids: Tensor
    device: torch.device
    dtype: torch.dtype
    segmentation_alignment: SegmentationAlignment


@dataclass(frozen=True, slots=True)
class _BenchmarkContext:
    model: nn.Module
    adapter: InputEmbeddingAdapter
    prompts: tuple[tuple[int, ...], ...]
    prompt_hashes: tuple[str, ...]
    prompt_hash_seconds: float
    inputs: tuple[_PreparedInputContext, ...]
    compute: dict[PrefillMode, tuple[_CodecCompute, ...]]
    device: torch.device


@dataclass(slots=True)
class _BenchmarkMeasurements:
    prompt_metrics: dict[PrefillMode, dict[int, _PromptMetrics]]
    preparation: dict[PrefillMode, list[float]]
    prefill: dict[PrefillMode, list[float]]
    cache_accounting: dict[PrefillMode, list[float]]
    end_to_end: dict[PrefillMode, list[float]]
    raw_pairs: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _PairExecution:
    prompt_index: int
    repetition: int
    prompt_execution_order: int
    pair_execution_order: int
    record: bool


@dataclass(frozen=True, slots=True)
class _PromptMetrics:
    prompt_index: int
    prompt_sha256: str
    positions: int
    materialized_cache_bytes: int
    candidate_lengths: dict[int, int]
    logical_candidates: int
    neural_candidate_rows: int
    speculative_discarded_rows: int
    padded_neural_rows: int
    neural_invocations: int
    graph_signature_counts: dict[str, int]
    analytical_backbone_flops: int
    analytical_codec_encode_flops: int
    analytical_codec_validation_decode_flops: int


def _estimated_prefill_flops(config: Any, positions: int) -> int:
    return backbone_forward_flops(config, positions)


@dataclass(frozen=True, slots=True)
class _CodecCompute:
    encode_flops: int
    validation_flops: int
    candidate_lengths: dict[int, int]
    logical_candidates: int
    neural_candidate_rows: int
    speculative_discarded_rows: int
    padded_neural_rows: int
    neural_invocations: int
    graph_signature_counts: dict[str, int]


def _segmentation_neural_work(
    data: bytes,
    span_lengths: tuple[int, ...],
    *,
    candidate_limit: int,
) -> tuple[dict[int, int], int, int, int, dict[str, int]]:
    neural_lengths: Counter[int] = Counter()
    signatures: Counter[str] = Counter()
    evaluated_rows = 0
    logical_rows = 0
    padded_rows = 0
    position = 0
    span_index = 0
    while position < len(data):
        window_start = position
        frontier = SEGMENTATION_FRONTIERS[-1]
        window_end = min(position + frontier, len(data))
        widths: Counter[int] = Counter()
        for offset in range(window_start, window_end):
            maximum_length = min(candidate_limit, len(data) - offset)
            for length in range(2, maximum_length + 1):
                widths[span_bucket_width(length, max_span=candidate_limit)] += 1
        evaluated_rows += sum(widths.values())
        for width, rows in widths.items():
            target_rows = candidate_group_rows(width, frontier)
            if rows > target_rows:
                raise RuntimeError("prefill candidate group exceeds bounded rows")
            neural_lengths[width] += target_rows
            padded_rows += target_rows - rows
            signatures[f"encode_validate:{target_rows}x{width}"] += 1
        while position < window_end:
            maximum_length = min(candidate_limit, len(data) - position)
            logical_rows += max(maximum_length - 1, 0)
            position += span_lengths[span_index]
            span_index += 1
    if span_index != len(span_lengths):
        raise RuntimeError("prefill span lengths do not cover the byte run")
    return (
        dict(sorted(neural_lengths.items())),
        logical_rows,
        evaluated_rows - logical_rows,
        padded_rows,
        dict(sorted(signatures.items())),
    )


def _codec_compute(
    adapter: InputEmbeddingAdapter,
    prompt: Sequence[int],
    mode: PrefillMode,
) -> _CodecCompute:
    if mode == "native":
        return _CodecCompute(0, 0, {}, 0, 0, 0, 0, 0, {})
    if mode == "compatibility":
        histogram = dict(
            sorted(
                Counter(len(payload) for token_id in prompt if (payload := adapter.vocabulary.payload_for(token_id)) is not None and len(payload) > 1).items()
            )
        )
        invocations = int(bool(histogram))
        encode = input_encode_batch_flops(
            adapter.codec.config,
            histogram,
            neural_invocations=invocations,
        )
        rows = sum(histogram.values())
        width = max(histogram, default=0)
        signatures = {f"encode:{rows}x{width}": 1} if rows else {}
        return _CodecCompute(
            encode,
            0,
            histogram,
            rows,
            rows,
            0,
            0,
            invocations,
            signatures,
        )

    counts: Counter[int] = Counter()
    neural_lengths: Counter[int] = Counter()
    logical_candidates = 0
    speculative_rows = 0
    padded_rows = 0
    signatures: Counter[str] = Counter()
    for piece in adapter.pieces_from_token_ids(prompt):
        if not isinstance(piece, ByteRun):
            continue
        result = segment_bytes(
            adapter.codec,
            piece.data,
            cache=None,
            namespace=adapter.namespace,
        )
        counts.update(result.stats.candidate_lengths)
        work = _segmentation_neural_work(
            piece.data,
            tuple(len(span.data) for span in result.spans),
            candidate_limit=min(DYNAMIC_SEGMENTATION_MAX_BYTES, adapter.codec.max_span),
        )
        neural_lengths.update(work[0])
        logical_candidates += work[1]
        speculative_rows += work[2]
        padded_rows += work[3]
        signatures.update(work[4])
    histogram = dict(sorted(counts.items()))
    invocations = sum(signatures.values())
    encode = input_encode_batch_flops(
        adapter.codec.config,
        neural_lengths,
        neural_invocations=invocations,
    )
    validation = sum(count * input_validation_flops(adapter.codec.config, length) for length, count in neural_lengths.items())
    return _CodecCompute(
        encode,
        validation,
        histogram,
        logical_candidates,
        sum(neural_lengths.values()),
        speculative_rows,
        padded_rows,
        invocations,
        dict(sorted(signatures.items())),
    )


def _prepare_model_input(
    adapter: InputEmbeddingAdapter,
    prompt: Sequence[int],
    mode: PrefillMode,
    context: _PreparedInputContext,
) -> tuple[Tensor, Tensor | None, int]:
    if mode == "native":
        return context.native_input_ids, None, len(prompt)
    encoding = adapter.encode_token_ids(
        prompt,
        mode=mode,
        cache=adapter.codec.encoding_cache,
        alignment=context.segmentation_alignment,
    )
    return (
        encoding.embeddings.to(
            device=context.device,
            dtype=context.dtype,
        ).unsqueeze(0),
        encoding.position_ids.to(context.device).unsqueeze(0),
        len(encoding.positions),
    )


def _prefill_model(
    model: nn.Module,
    mode: PrefillMode,
    model_input: Tensor,
    position_ids: Tensor | None,
) -> Any:
    if mode == "native":
        return model(input_ids=model_input, use_cache=True, logits_to_keep=1)
    return model(
        inputs_embeds=model_input,
        position_ids=position_ids,
        use_cache=True,
        logits_to_keep=1,
    )


def _prompt_order(prompt_count: int, repetition: int) -> tuple[int, ...]:
    offset = repetition % prompt_count
    indices = tuple(range(prompt_count))
    return indices[offset:] + indices[:offset]


def _path_order(pair_execution_order: int) -> tuple[PrefillMode, ...]:
    offset = pair_execution_order % len(_PREFILL_MODES)
    return _PREFILL_MODES[offset:] + _PREFILL_MODES[:offset]


def _run_pair(
    context: _BenchmarkContext,
    measurements: _BenchmarkMeasurements,
    execution: _PairExecution,
) -> None:
    prompt_index = execution.prompt_index
    prompt = context.prompts[prompt_index]
    paths = _path_order(execution.pair_execution_order)
    raw_paths: dict[str, object] = {}
    for execution_order, mode in enumerate(paths):

        def preparation_and_prefill(
            path_mode: PrefillMode = mode,
        ) -> tuple[Any, int, float, float]:
            prepared, preparation_seconds = timed_call(
                lambda: _prepare_model_input(
                    context.adapter,
                    prompt,
                    path_mode,
                    context.inputs[prompt_index],
                ),
                context.device,
            )
            model_input, model_position_ids, positions = prepared
            output, prefill_seconds = timed_call(
                partial(
                    _prefill_model,
                    context.model,
                    path_mode,
                    model_input,
                    model_position_ids,
                ),
                context.device,
            )
            return output, positions, preparation_seconds, prefill_seconds

        (
            (
                output,
                positions,
                preparation_seconds,
                prefill_seconds,
            ),
            end_to_end,
        ) = timed_observation(
            preparation_and_prefill,
            context.device,
        )
        cache_started = time.perf_counter()
        cache_bytes = cache_tensor_bytes(output.past_key_values)
        cache_accounting_seconds = time.perf_counter() - cache_started
        if not execution.record:
            continue
        measurements.preparation[mode].append(preparation_seconds)
        measurements.prefill[mode].append(prefill_seconds)
        measurements.cache_accounting[mode].append(
            cache_accounting_seconds,
        )
        measurements.end_to_end[mode].append(end_to_end.wall_seconds)
        compute = context.compute[mode][prompt_index]
        static = _PromptMetrics(
            prompt_index=prompt_index,
            prompt_sha256=context.prompt_hashes[prompt_index],
            positions=positions,
            materialized_cache_bytes=cache_bytes,
            candidate_lengths=compute.candidate_lengths,
            logical_candidates=compute.logical_candidates,
            neural_candidate_rows=compute.neural_candidate_rows,
            speculative_discarded_rows=compute.speculative_discarded_rows,
            padded_neural_rows=compute.padded_neural_rows,
            neural_invocations=compute.neural_invocations,
            graph_signature_counts=compute.graph_signature_counts,
            analytical_backbone_flops=_estimated_prefill_flops(
                context.model.config,
                positions,
            ),
            analytical_codec_encode_flops=compute.encode_flops,
            analytical_codec_validation_decode_flops=compute.validation_flops,
        )
        previous = measurements.prompt_metrics[mode].setdefault(
            prompt_index,
            static,
        )
        if previous != static:
            raise RuntimeError(
                "prefill prompt metrics changed across paired repetitions",
            )
        raw_paths[mode] = {
            "schema_version": PREFILL_BENCHMARK_SCHEMA_VERSION,
            "execution_order": execution_order,
            "input_preparation_seconds": preparation_seconds,
            "model_prefill_seconds": prefill_seconds,
            "cache_accounting_seconds": cache_accounting_seconds,
            "time_to_first_logit_seconds": end_to_end.wall_seconds,
            "subphase_sum_seconds": preparation_seconds + prefill_seconds,
            "timing": end_to_end.to_dict(),
        }
    if execution.record:
        measurements.raw_pairs.append(
            {
                "schema_version": PREFILL_BENCHMARK_SCHEMA_VERSION,
                "prompt_index": prompt_index,
                "prompt_sha256": context.prompt_hashes[prompt_index],
                "repetition": execution.repetition,
                "prompt_execution_order": execution.prompt_execution_order,
                "pair_execution_order": execution.pair_execution_order,
                "path_order": list(paths),
                "paths": raw_paths,
            }
        )


def _run_schedule(
    context: _BenchmarkContext,
    measurements: _BenchmarkMeasurements,
    options: PrefillBenchmarkOptions,
) -> None:
    prompt_count = len(context.prompts)
    with torch.inference_mode():
        context.adapter.codec.encoding_cache.clear()
        for prompt in context.prompts:
            for mode in ("compatibility", "segmented"):
                context.adapter.encode_token_ids(
                    prompt,
                    mode=mode,
                    cache=context.adapter.codec.encoding_cache,
                    alignment=options.segmentation_alignment,
                )
        for warmup in range(options.warmups):
            for execution_order, prompt_index in enumerate(
                _prompt_order(prompt_count, warmup),
            ):
                _run_pair(
                    context,
                    measurements,
                    _PairExecution(
                        prompt_index,
                        warmup,
                        execution_order,
                        warmup * prompt_count + execution_order,
                        False,
                    ),
                )
        for repetition in range(options.repetitions):
            for execution_order, prompt_index in enumerate(
                _prompt_order(prompt_count, repetition),
            ):
                _run_pair(
                    context,
                    measurements,
                    _PairExecution(
                        prompt_index,
                        repetition,
                        execution_order,
                        repetition * prompt_count + execution_order,
                        True,
                    ),
                )


def _measurement_metadata(
    context: _BenchmarkContext,
    measurements: _BenchmarkMeasurements,
    options: PrefillBenchmarkOptions,
) -> dict[str, object]:
    prompt_count = len(context.prompts)
    return {
        "schema_version": PREFILL_BENCHMARK_SCHEMA_VERSION,
        "timing_observation_schema_version": TIMING_OBSERVATION_SCHEMA_VERSION,
        "prompt_count": prompt_count,
        "prompt_set_sha256": mapping_fingerprint(context.prompts),
        "prompt_sha256": list(context.prompt_hashes),
        "prompt_content": "native_token_id_sequence",
        "hash_algorithm": "sha256",
        "prompt_hash_serialization": "compact_json_integer_array",
        "prompt_set_hash_serialization": ("compact_json_array_of_integer_arrays"),
        "prompt_hash_seconds": context.prompt_hash_seconds,
        "warmups": options.warmups,
        "repetitions": options.repetitions,
        "expected_raw_pairs": prompt_count * options.repetitions,
        "recorded_raw_pairs": len(measurements.raw_pairs),
        "warmup_scope": "each_registered_prompt_and_path",
        "warmup_executions": (prompt_count * len(_PREFILL_MODES) * options.warmups),
        "prompt_order": "cyclic_rotation_by_repetition",
        "path_order": "cyclic_rotation_by_pair_execution_order",
        "pairing": "same_prompt_and_repetition",
        "end_to_end_timing": "direct_preparation_plus_prefill_call",
        "timing_denominator": ("all registered prompts times repetitions per mode"),
        "aggregate_denominator": "registered_unique_prompt_set",
        "aggregate_semantics": {
            "positions": "sum_over_registered_prompts",
            "materialized_cache_bytes": "sum_over_registered_prompts",
            "analytical_flops": "sum_over_registered_prompts",
            "candidate_lengths": "sum_over_registered_prompts",
            "logical_candidates": "sum_over_registered_prompts",
            "neural_candidate_rows": "sum_over_registered_prompts",
            "graph_signature_counts": "sum_over_registered_prompts",
        },
    }


def _mode_result(
    context: _BenchmarkContext,
    measurements: _BenchmarkMeasurements,
    mode: PrefillMode,
) -> dict[str, object]:
    prompt_count = len(context.prompts)
    per_prompt = tuple(measurements.prompt_metrics[mode][prompt_index] for prompt_index in range(prompt_count))
    candidate_lengths = dict(
        sorted(
            sum(
                (Counter(item.candidate_lengths) for item in per_prompt),
                Counter(),
            ).items()
        )
    )
    positions = sum(item.positions for item in per_prompt)
    cache_bytes = sum(item.materialized_cache_bytes for item in per_prompt)
    backbone_flops = sum(item.analytical_backbone_flops for item in per_prompt)
    encode_flops = sum(item.analytical_codec_encode_flops for item in per_prompt)
    validation_flops = sum(item.analytical_codec_validation_decode_flops for item in per_prompt)
    total_flops = backbone_flops + encode_flops + validation_flops
    preparation_summary = timing_summary(measurements.preparation[mode])
    prefill_summary = timing_summary(measurements.prefill[mode])
    cache_accounting_summary = timing_summary(
        measurements.cache_accounting[mode],
    )
    end_to_end_summary = timing_summary(measurements.end_to_end[mode])
    graph_signature_counts = dict(
        sorted(
            sum(
                (Counter(item.graph_signature_counts) for item in per_prompt),
                Counter(),
            ).items()
        )
    )
    return {
        "positions": positions,
        "mean_positions_per_prompt": positions / prompt_count,
        "input_cache": "not_applicable" if mode == "native" else "warm",
        "input_preparation_median_seconds": preparation_summary["median"],
        "input_preparation_p95_seconds": preparation_summary["p95"],
        "model_prefill_median_seconds": prefill_summary["median"],
        "model_prefill_p95_seconds": prefill_summary["p95"],
        "cache_accounting_median_seconds": cache_accounting_summary["median"],
        "cache_accounting_p95_seconds": cache_accounting_summary["p95"],
        "time_to_first_logit_median_seconds": end_to_end_summary["median"],
        "time_to_first_logit_p95_seconds": end_to_end_summary["p95"],
        "timing_observations": len(measurements.end_to_end[mode]),
        "candidate_lengths": candidate_lengths,
        "logical_candidates": sum(item.logical_candidates for item in per_prompt),
        "neural_candidate_rows": sum(item.neural_candidate_rows for item in per_prompt),
        "speculative_discarded_rows": sum(item.speculative_discarded_rows for item in per_prompt),
        "padded_neural_rows": sum(item.padded_neural_rows for item in per_prompt),
        "neural_invocations": sum(item.neural_invocations for item in per_prompt),
        "graph_signature_counts": graph_signature_counts,
        "materialized_cache_bytes": cache_bytes,
        "mean_materialized_cache_bytes_per_prompt": (cache_bytes / prompt_count),
        "analytical_backbone_flops": backbone_flops,
        "analytical_codec_encode_flops": encode_flops,
        "analytical_codec_validation_decode_flops": validation_flops,
        "total_analytical_flops": total_flops,
        "mean_total_analytical_flops_per_prompt": (total_flops / prompt_count),
        "per_prompt": [asdict(item) for item in per_prompt],
    }


def benchmark_model_prefill(
    model: nn.Module,
    adapter: InputEmbeddingAdapter,
    prompts: tuple[tuple[int, ...], ...],
    options: PrefillBenchmarkOptions,
) -> dict[str, Any]:
    if options.warmups < 0 or options.repetitions < 1:
        raise ValueError("prefill warmups must be non-negative and repetitions positive")
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("prefill prompts must be a non-empty tuple of non-empty prompts")
    parameter = next(model.parameters())
    device = parameter.device
    model_dtype = parameter.dtype
    hash_started = time.perf_counter()
    prompt_hashes = tuple(mapping_fingerprint(prompt) for prompt in prompts)
    prompt_hash_seconds = time.perf_counter() - hash_started
    inputs = tuple(
        _PreparedInputContext(
            torch.tensor([prompt], dtype=torch.long, device=device),
            device,
            model_dtype,
            options.segmentation_alignment,
        )
        for prompt in prompts
    )
    compute = {mode: tuple(_codec_compute(adapter, prompt, mode) for prompt in prompts) for mode in _PREFILL_MODES}
    context = _BenchmarkContext(
        model,
        adapter,
        prompts,
        prompt_hashes,
        prompt_hash_seconds,
        inputs,
        compute,
        device,
    )
    measurements = _BenchmarkMeasurements(
        {mode: {} for mode in _PREFILL_MODES},
        {mode: [] for mode in _PREFILL_MODES},
        {mode: [] for mode in _PREFILL_MODES},
        {mode: [] for mode in _PREFILL_MODES},
        {mode: [] for mode in _PREFILL_MODES},
        [],
    )
    _run_schedule(context, measurements, options)
    result: dict[str, Any] = {
        "schema_version": PREFILL_BENCHMARK_SCHEMA_VERSION,
        "measurement": _measurement_metadata(
            context,
            measurements,
            options,
        ),
        "raw_pairs": measurements.raw_pairs,
    }
    for mode in _PREFILL_MODES:
        result[mode] = _mode_result(context, measurements, mode)
    return result
