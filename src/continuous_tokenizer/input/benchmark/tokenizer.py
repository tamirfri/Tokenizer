from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, final

import psutil
import torch

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.backbone.config import input_table_is_removable, tie_word_embeddings
from continuous_tokenizer.codec.batches import build_byte_batch, span_bucket_width
from continuous_tokenizer.codec.checkpoints import checkpoint_tensor_inventory
from continuous_tokenizer.codec.compilation import TOKENIZER_RECOMPILE_LIMIT
from continuous_tokenizer.codec.encoding_cache import EncodingCache
from continuous_tokenizer.codec.input import InputByteCodec
from continuous_tokenizer.codec.layers import gqa_metadata
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.data.corpus import ContentWindow
from continuous_tokenizer.input.adapter import LoadedInputAdapter
from continuous_tokenizer.input.alignment import (
    EmbeddingEvaluationRequest,
    EmbeddingFitTargets,
    evaluate_embeddings,
)
from continuous_tokenizer.input.evidence import (
    SegmentationEvidence,
    alias_group_evidence,
    segmentation_evidence,
)
from continuous_tokenizer.input.segmentation import (
    DYNAMIC_SEGMENTATION_MAX_BYTES,
    SEGMENTATION_FRONTIERS,
    EncodedSpan,
    SegmentationResult,
    candidate_group_rows,
    reconstruct,
    segment_bytes,
)
from continuous_tokenizer.runtime.compiler import compiler_cache_directory
from continuous_tokenizer.runtime.environment import runtime_environment
from continuous_tokenizer.runtime.progress import log_event
from continuous_tokenizer.runtime.tensors import module_bytes, tensor_bytes
from continuous_tokenizer.runtime.timing import (
    TIMING_OBSERVATION_SCHEMA_VERSION,
    timed_observation,
    timing_summary,
)

TOKENIZER_BENCHMARK_SCHEMA_VERSION: Final = 1
_BOOL_BYTES: Final = torch.empty((), dtype=torch.bool).element_size()
_LONG_BYTES: Final = torch.empty((), dtype=torch.long).element_size()


@final
@dataclass(frozen=True, slots=True)
class SegmentationRun:
    schema_version: int
    mode: str
    execution_mode: str
    spans: int
    seconds: float
    p95_seconds: float
    repetitions: int
    bytes_per_span: float
    round_trip: bool
    source_bookkeeping_round_trip: bool
    semantic_sha256: str
    span_evidence: tuple[dict[str, Any], ...]
    cache_entries: int
    cache_tensor_bytes: int
    cache_capacity_bytes: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    cache_stores: int
    cache_evictions: int
    cache_coalesced: int
    process_rss_bytes: int
    atomic_spans: int
    candidates: int
    candidate_lengths: dict[int, int]
    valid_candidates: int
    invalid_by_length: dict[int, int]
    span_lengths: dict[int, int]
    content_windows: int
    workload_sha256: str
    logical_candidates: int
    neural_candidate_rows: int
    speculative_discarded_rows: int
    padded_neural_rows: int
    neural_invocations: int
    graph_signature_counts: dict[str, int]
    host_to_device_bytes: int
    device_to_host_bytes: int
    synchronization_count: int
    peak_mps_allocated_bytes: int
    peak_mps_driver_bytes: int
    cache_accounting_median_seconds: float
    snapshot_hash_median_seconds: float
    raw_observations: tuple[dict[str, Any], ...]


@final
@dataclass(frozen=True, slots=True)
class TokenizerMetricRequest:
    test_windows: tuple[ContentWindow, ...]
    batch_size: int
    retrieval_rows: int | None
    repetitions: int
    dataset_id: str
    dataset_config: str
    dataset_revision: str
    dataset_split: str = "test"

    @property
    def test_data(self) -> bytes:
        return b"".join(window.payload for window in self.test_windows)


@dataclass(frozen=True, slots=True)
class _SegmentationExecution:
    mode: str
    cache: EncodingCache | None
    namespace: str
    repetition: int
    execution_order: int


@dataclass(frozen=True, slots=True)
class _SemanticSnapshot:
    results: tuple[SegmentationResult, ...]
    evidence: tuple[SegmentationEvidence, ...]
    semantic_sha256: str
    span_evidence: tuple[dict[str, Any], ...]


def _same_segmentation(
    left: tuple[SegmentationResult, ...],
    right: tuple[SegmentationResult, ...],
) -> bool:
    if len(left) != len(right):
        return False
    for left_result, right_result in zip(left, right, strict=True):
        if left_result.stats != right_result.stats or len(left_result.spans) != len(
            right_result.spans,
        ):
            return False
        for left_span, right_span in zip(
            left_result.spans,
            right_result.spans,
            strict=True,
        ):
            if (
                left_span.data != right_span.data
                or left_span.atomic != right_span.atomic
                or left_span.latent.dtype != right_span.latent.dtype
                or left_span.latent.shape != right_span.latent.shape
                or not torch.equal(left_span.latent, right_span.latent)
            ):
                return False
    return True


def _semantic_snapshot(
    codec: InputByteCodec,
    windows: tuple[ContentWindow, ...],
    results: tuple[SegmentationResult, ...],
    snapshots: list[_SemanticSnapshot],
) -> _SemanticSnapshot:
    for snapshot in snapshots:
        if _same_segmentation(results, snapshot.results):
            return snapshot
    evidence = tuple(
        segmentation_evidence(
            codec,
            result.spans,
            window.payload,
            source_dtype=codec.byte_embeddings.dtype,
        )
        for window, result in zip(windows, results, strict=True)
    )
    semantic_sha256 = mapping_fingerprint(
        [
            {
                "content_sha256": window.sha256,
                "semantic_sha256": window_evidence.semantic_sha256,
            }
            for window, window_evidence in zip(windows, evidence, strict=True)
        ],
    )
    snapshot = _SemanticSnapshot(
        results=results,
        evidence=evidence,
        semantic_sha256=semantic_sha256,
        span_evidence=tuple(
            {
                "window_index": window_index,
                **asdict(row),
            }
            for window_index, window_evidence in enumerate(evidence)
            for row in window_evidence.rows
        ),
    )
    snapshots.append(snapshot)
    return snapshot


def _counter_sum(values: list[dict[int, int]]) -> dict[int, int]:
    return dict(sorted(sum((Counter(value) for value in values), Counter()).items()))


def _candidate_groups_by_width(
    data: bytes,
    position: int,
    window_end: int,
    *,
    candidate_limit: int,
) -> dict[int, list[bytes]]:
    groups: dict[int, list[bytes]] = {}
    for offset in range(position, window_end):
        maximum_length = min(candidate_limit, len(data) - offset)
        for length in range(2, maximum_length + 1):
            width = span_bucket_width(length, max_span=candidate_limit)
            groups.setdefault(width, []).append(data[offset : offset + length])
    return groups


def _window_neural_work(  # noqa: PLR0913
    data: bytes,
    span_lengths: tuple[int, ...],
    *,
    candidate_limit: int,
    mode: str,
    warm_cache: bool,
    cached_spans: set[tuple[bytes, int, int]] | None = None,
) -> dict[str, Any]:
    cached = set() if cached_spans is None else cached_spans
    signatures: Counter[str] = Counter()
    neural_rows = 0
    padded_rows = 0
    speculative_rows = 0
    neural_invocations = 0
    host_to_device_bytes = len(data) * _LONG_BYTES
    device_to_host_bytes = 0
    position = 0
    span_index = 0
    while position < len(data):
        window_start = position
        frontier = SEGMENTATION_FRONTIERS[-1]
        window_end = min(position + frontier, len(data))
        groups = _candidate_groups_by_width(
            data,
            window_start,
            window_end,
            candidate_limit=candidate_limit,
        )
        evaluated_rows = sum(map(len, groups.values()))
        logical_rows = 0
        while position < window_end:
            maximum_length = min(candidate_limit, len(data) - position)
            logical_rows += max(maximum_length - 1, 0)
            position += span_lengths[span_index]
            span_index += 1
        speculative_rows += evaluated_rows - logical_rows
        for width, spans in groups.items():
            target_rows = candidate_group_rows(width, frontier)
            padding = target_rows - len(spans)
            if padding < 0:
                raise RuntimeError("benchmark candidate group exceeds bounded rows")
            padded_rows += padding
            host_to_device_bytes += target_rows * 2 * _LONG_BYTES
            device_to_host_bytes += len(spans) * _BOOL_BYTES
            if mode == "disabled":
                signatures[f"encode_validate:{target_rows}x{width}"] += 1
                neural_rows += target_rows
                neural_invocations += 1
                continue
            cache_keys = {(span, target_rows, width) for span in spans}
            missing = set() if warm_cache else cache_keys - cached
            if missing:
                signatures[f"encode:{target_rows}x{width}"] += 1
                neural_rows += target_rows
                neural_invocations += 1
                host_to_device_bytes += target_rows * width * (_LONG_BYTES + _BOOL_BYTES)
                cached.update(missing)
            signatures[f"validate:{target_rows}x{width}"] += 1
            neural_rows += target_rows
            neural_invocations += 1
    if span_index != len(span_lengths):
        raise RuntimeError("benchmark span lengths do not cover the content window")
    return {
        "neural_candidate_rows": neural_rows,
        "speculative_discarded_rows": speculative_rows,
        "padded_neural_rows": padded_rows,
        "neural_invocations": neural_invocations,
        "graph_signature_counts": dict(sorted(signatures.items())),
        "host_to_device_bytes": host_to_device_bytes,
        "device_to_host_bytes": device_to_host_bytes,
    }


@torch.inference_mode()
def _run_segmentation(
    codec: InputByteCodec,
    windows: tuple[ContentWindow, ...],
    execution: _SegmentationExecution,
    snapshots: list[_SemanticSnapshot],
    workload_sha256: str,
) -> SegmentationRun:
    mode = execution.mode
    cache = execution.cache
    namespace = execution.namespace
    before = cache.info() if cache is not None else None
    results, timing = timed_observation(
        lambda: tuple(
            segment_bytes(
                codec,
                window.payload,
                cache=cache,
                namespace=namespace,
            )
            for window in windows
        ),
        codec.device,
    )
    cache_started = time.perf_counter()
    if cache is None:
        info = None
    else:
        if before is None:
            raise RuntimeError("cache statistics were not captured before segmentation")
        info = cache.info().since(before)
    cache_accounting_seconds = time.perf_counter() - cache_started
    snapshot_started = time.perf_counter()
    snapshot = _semantic_snapshot(
        codec,
        windows,
        results,
        snapshots,
    )
    evidence = snapshot.evidence
    semantic_sha256 = snapshot.semantic_sha256
    span_evidence = snapshot.span_evidence
    snapshot_hash_seconds = time.perf_counter() - snapshot_started
    data_bytes = sum(len(window.payload) for window in windows)
    stats = tuple(result.stats for result in results)
    shared_cache: set[tuple[bytes, int, int]] = set()
    work_rows = []
    for window, result in zip(windows, results, strict=True):
        work_rows.append(
            _window_neural_work(
                window.payload,
                tuple(len(span.data) for span in result.spans),
                candidate_limit=min(DYNAMIC_SEGMENTATION_MAX_BYTES, codec.max_span),
                mode=mode,
                warm_cache=mode == "warm",
                cached_spans=shared_cache,
            )
        )
    graph_signature_counts = dict(
        sorted(
            sum(
                (Counter(row["graph_signature_counts"]) for row in work_rows),
                Counter(),
            ).items()
        )
    )
    logical_candidates = sum(row.candidates for row in stats)
    neural_candidate_rows = sum(int(row["neural_candidate_rows"]) for row in work_rows)
    speculative_discarded_rows = sum(int(row["speculative_discarded_rows"]) for row in work_rows)
    padded_neural_rows = sum(int(row["padded_neural_rows"]) for row in work_rows)
    neural_invocations = sum(int(row["neural_invocations"]) for row in work_rows)
    host_to_device_bytes = sum(int(row["host_to_device_bytes"]) for row in work_rows)
    device_to_host_bytes = sum(int(row["device_to_host_bytes"]) for row in work_rows)
    observation = {
        "schema_version": TOKENIZER_BENCHMARK_SCHEMA_VERSION,
        "repetition": execution.repetition,
        "execution_order": execution.execution_order,
        "mode": mode,
        "subphases": {
            "segmentation_encoding_seconds": timing.wall_seconds,
            "cache_accounting_seconds": cache_accounting_seconds,
            "snapshot_hash_seconds": snapshot_hash_seconds,
        },
        "timing": timing.to_dict(),
        "semantic_sha256": semantic_sha256,
        "logical_candidates": logical_candidates,
        "neural_candidate_rows": neural_candidate_rows,
        "speculative_discarded_rows": speculative_discarded_rows,
        "padded_neural_rows": padded_neural_rows,
        "neural_invocations": neural_invocations,
        "graph_signature_counts": graph_signature_counts,
        "host_to_device_bytes": host_to_device_bytes,
        "device_to_host_bytes": device_to_host_bytes,
    }
    return SegmentationRun(
        schema_version=TOKENIZER_BENCHMARK_SCHEMA_VERSION,
        mode=mode,
        execution_mode=("warm_compile" if codec.neural_paths_compiled else "eager"),
        spans=sum(len(result.spans) for result in results),
        seconds=timing.wall_seconds,
        p95_seconds=timing.wall_seconds,
        repetitions=1,
        bytes_per_span=data_bytes / max(sum(len(result.spans) for result in results), 1),
        round_trip=all(row.empirical_round_trip for row in evidence),
        source_bookkeeping_round_trip=all(row.source_bookkeeping_round_trip for row in evidence),
        semantic_sha256=semantic_sha256,
        span_evidence=span_evidence,
        cache_entries=0 if info is None else info.entries,
        cache_tensor_bytes=0 if info is None else info.tensor_bytes,
        cache_capacity_bytes=0 if info is None else info.capacity_bytes,
        cache_hits=0 if info is None else info.hits,
        cache_misses=0 if info is None else info.misses,
        cache_hit_rate=0.0 if info is None else info.hit_rate,
        cache_stores=0 if info is None else info.stores,
        cache_evictions=0 if info is None else info.evictions,
        cache_coalesced=0 if info is None else info.coalesced,
        process_rss_bytes=psutil.Process().memory_info().rss,
        atomic_spans=sum(row.atomic_spans for row in stats),
        candidates=sum(row.candidates for row in stats),
        candidate_lengths=_counter_sum([row.candidate_lengths for row in stats]),
        valid_candidates=sum(row.valid_candidates for row in stats),
        invalid_by_length=_counter_sum([row.invalid_by_length for row in stats]),
        span_lengths=_counter_sum([row.span_lengths for row in stats]),
        content_windows=len(windows),
        workload_sha256=workload_sha256,
        logical_candidates=logical_candidates,
        neural_candidate_rows=neural_candidate_rows,
        speculative_discarded_rows=speculative_discarded_rows,
        padded_neural_rows=padded_neural_rows,
        neural_invocations=neural_invocations,
        graph_signature_counts=graph_signature_counts,
        host_to_device_bytes=host_to_device_bytes,
        device_to_host_bytes=device_to_host_bytes,
        synchronization_count=timing.synchronization_count,
        peak_mps_allocated_bytes=timing.peak_mps_allocated_bytes,
        peak_mps_driver_bytes=timing.peak_mps_driver_bytes,
        cache_accounting_median_seconds=cache_accounting_seconds,
        snapshot_hash_median_seconds=snapshot_hash_seconds,
        raw_observations=(observation,),
    )


def _aggregate_segmentation_runs(runs: list[SegmentationRun]) -> SegmentationRun:
    if not runs:
        raise ValueError("segmentation benchmark requires at least one repetition")
    first = runs[0]
    comparable = replace(
        first,
        seconds=0.0,
        p95_seconds=0.0,
        repetitions=0,
        process_rss_bytes=0,
        synchronization_count=0,
        peak_mps_allocated_bytes=0,
        peak_mps_driver_bytes=0,
        cache_accounting_median_seconds=0.0,
        snapshot_hash_median_seconds=0.0,
        raw_observations=(),
    )
    for run in runs[1:]:
        candidate = replace(
            run,
            seconds=0.0,
            p95_seconds=0.0,
            repetitions=0,
            process_rss_bytes=0,
            synchronization_count=0,
            peak_mps_allocated_bytes=0,
            peak_mps_driver_bytes=0,
            cache_accounting_median_seconds=0.0,
            snapshot_hash_median_seconds=0.0,
            raw_observations=(),
        )
        if candidate != comparable:
            raise RuntimeError(f"{first.mode} segmentation changed between repetitions")
    timing = timing_summary([run.seconds for run in runs])
    cache_accounting = timing_summary(
        [run.cache_accounting_median_seconds for run in runs],
    )
    snapshot_hash = timing_summary(
        [run.snapshot_hash_median_seconds for run in runs],
    )
    return replace(
        first,
        seconds=timing["median"],
        p95_seconds=timing["p95"],
        repetitions=len(runs),
        process_rss_bytes=max(run.process_rss_bytes for run in runs),
        synchronization_count=sum(run.synchronization_count for run in runs),
        peak_mps_allocated_bytes=max(run.peak_mps_allocated_bytes for run in runs),
        peak_mps_driver_bytes=max(run.peak_mps_driver_bytes for run in runs),
        cache_accounting_median_seconds=cache_accounting["median"],
        snapshot_hash_median_seconds=snapshot_hash["median"],
        raw_observations=tuple(observation for run in runs for observation in run.raw_observations),
    )


def _benchmark_segmentation_cache(
    codec: InputByteCodec,
    windows: tuple[ContentWindow, ...],
    *,
    namespace: str,
    repetitions: int,
) -> tuple[SegmentationRun, SegmentationRun, SegmentationRun]:
    if repetitions < 1:
        raise ValueError("segmentation benchmark repetitions must be positive")
    modes = ("disabled", "cold", "warm")
    observations: dict[str, list[SegmentationRun]] = {mode: [] for mode in modes}
    cache = codec.encoding_cache
    snapshots: list[_SemanticSnapshot] = []
    workload_sha256 = mapping_fingerprint(
        [window.to_dict() for window in windows],
    )
    for repetition in range(repetitions):
        rotation = repetition % len(modes)
        order = modes[rotation:] + modes[:rotation]
        for execution_order, mode in enumerate(order):
            if mode == "disabled":
                run_cache = None
            else:
                cache.clear()
                if mode == "warm":
                    with torch.inference_mode():
                        for window in windows:
                            segment_bytes(
                                codec,
                                window.payload,
                                cache=cache,
                                namespace=namespace,
                            )
                run_cache = cache
            run = _run_segmentation(
                codec,
                windows,
                _SegmentationExecution(
                    mode,
                    run_cache,
                    namespace,
                    repetition,
                    execution_order,
                ),
                snapshots,
                workload_sha256,
            )
            observations[mode].append(run)
    log_event(
        "benchmark_work_avoided",
        work="input_semantic_evidence_and_hashes",
        avoided_runs=repetitions * len(modes) - len(snapshots),
        distinct_semantics=len(snapshots),
    )
    return (
        _aggregate_segmentation_runs(observations["disabled"]),
        _aggregate_segmentation_runs(observations["cold"]),
        _aggregate_segmentation_runs(observations["warm"]),
    )


def _window_metrics(
    codec: InputByteCodec,
    window: ContentWindow,
    *,
    native_tokens: int | None,
    namespace: str,
    source_dtype: torch.dtype,
) -> dict[str, Any]:
    result = segment_bytes(codec, window.payload, namespace=namespace)
    evidence = segmentation_evidence(
        codec,
        result.spans,
        window.payload,
        source_dtype=source_dtype,
    )
    return {
        **window.to_dict(),
        "bytes": len(window.payload),
        "native_tokens": native_tokens,
        "continuous_tokens": len(result.spans),
        "native_tokens_per_continuous_token": (None if native_tokens is None else native_tokens / max(len(result.spans), 1)),
        "atomic_spans": result.stats.atomic_spans,
        "span_lengths": result.stats.span_lengths,
        "semantic_sha256": evidence.semantic_sha256,
        "source_bookkeeping_round_trip": evidence.source_bookkeeping_round_trip,
        "empirical_round_trip": evidence.empirical_round_trip,
        "span_evidence": [asdict(row) for row in evidence.rows],
    }


def _vocabulary_density_rows(
    assets: ModelAssets,
    codec: InputByteCodec,
    *,
    count: int,
    namespace: str,
) -> list[dict[str, Any]]:
    ranked = []
    for token_id in assets.vocabulary.compatibility_ids:
        payload = assets.vocabulary.bytes_for(token_id)
        rank = hashlib.sha256(token_id.to_bytes(8, "big") + payload).digest()
        ranked.append((rank, token_id, payload))
    selected = sorted(ranked)[:count]
    rows = []
    for _, token_id, payload in selected:
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        rows.append(
            {
                "token_id": token_id,
                **_window_metrics(
                    codec,
                    ContentWindow(payload_sha256, 0, payload, payload_sha256),
                    native_tokens=1,
                    namespace=namespace,
                    source_dtype=assets.input_embeddings.dtype,
                ),
            }
        )
    return rows


@torch.inference_mode()
def _warm_compiled_tokenizer(codec: InputByteCodec, data: bytes) -> dict[str, Any]:
    if not codec.neural_paths_compiled:
        return {
            "schema_version": TOKENIZER_BENCHMARK_SCHEMA_VERSION,
            "enabled": False,
            "backend": None,
            "fullgraph": None,
            "dynamic": None,
            "shape_policy": None,
            "recompile_limit": None,
            "cache_directory": None,
            "warmup_bytes": 0,
            "warmup_seconds": 0.0,
            "graph_signatures": [],
            "warmup_coverage": "not_applicable",
            "eager": {
                "status": "complete",
                "reason": "codec uses eager tensor paths",
            },
            "cold_compile": {
                "status": "not_applicable",
                "process_isolated": False,
                "reason": "compilation is disabled",
            },
            "warm_compile": {
                "status": "not_applicable",
                "reason": "compilation is disabled",
            },
        }
    warmup_data = data[: min(len(data), 512)]
    if not warmup_data:
        raise ValueError("compiler warm-up requires non-empty data")
    candidate_limit = min(DYNAMIC_SEGMENTATION_MAX_BYTES, codec.max_span)
    widths = sorted({span_bucket_width(length, max_span=candidate_limit) for length in range(2, candidate_limit + 1)})
    signatures = [
        {
            "rows": candidate_group_rows(width, SEGMENTATION_FRONTIERS[-1]),
            "width": width,
        }
        for width in widths
    ]

    def warm_compiled_paths() -> None:
        for signature in signatures:
            rows = int(signature["rows"])
            width = int(signature["width"])
            spans = [bytes((row + column) % 256 for column in range(width)) for row in range(rows)]
            batch = build_byte_batch(
                spans,
                max_span=codec.max_span,
                device=codec.device,
            )
            codec.encode(batch.byte_values, batch.valid_mask)
            codec.reconstruction_logits(batch.byte_values, batch.valid_mask)
            codec.encode_and_reconstruction_matches(
                batch.byte_values,
                batch.valid_mask,
            )

    _, observation = timed_observation(warm_compiled_paths, codec.device)
    return {
        "schema_version": TOKENIZER_BENCHMARK_SCHEMA_VERSION,
        "enabled": True,
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "shape_policy": "bounded_static_specialization",
        "recompile_limit": TOKENIZER_RECOMPILE_LIMIT,
        "cache_directory": str(compiler_cache_directory()),
        "warmup_bytes": len(warmup_data),
        "warmup_seconds": observation.wall_seconds,
        "graph_signatures": signatures,
        "warmup_coverage": "all_bounded_encode_reconstruction_validation_signatures",
        "eager": {
            "status": "unavailable",
            "reason": "loaded codec exposes compiled tensor paths",
        },
        "cold_compile": {
            "status": "unavailable",
            "process_isolated": False,
            "reason": "true cold compilation requires a fresh isolated process",
        },
        "warm_compile": {
            "status": "complete",
            "reason": "every bounded tensor signature was executed before timing",
            "timing": observation.to_dict(),
        },
    }


def _raw_byte_fixtures() -> dict[str, bytes]:
    text = "Tokenizer: שלום, 世界, Δx"
    return {
        "all_bytes": bytes(range(256)),
        "binary": bytes((index * 73 + 19) % 256 for index in range(1024)),
        "invalid_utf8": b"\x80\xff\xc0\xaf\x00\xfe",
        "code": b"def encode(data: bytes) -> bytes:\n    return data.hex().encode()\n",
        "utf8": text.encode("utf-8"),
        "utf16_le": text.encode("utf-16-le"),
        "utf16_be": text.encode("utf-16-be"),
        "utf32_le": text.encode("utf-32-le"),
        "utf32_be": text.encode("utf-32-be"),
    }


def tokenizer_metrics(
    assets: ModelAssets,
    checkpoint: Path,
    loaded: LoadedInputAdapter,
    request: TokenizerMetricRequest,
) -> dict[str, Any]:
    adapter = loaded.adapter
    codec = adapter.codec
    test_data = request.test_data
    native_ids_by_window = tuple(
        tuple(
            assets.tokenizer.encode(
                window.payload.decode("utf-8", errors="strict"),
                add_special_tokens=False,
            )
        )
        for window in request.test_windows
    )
    native_ids = tuple(token_id for window_ids in native_ids_by_window for token_id in window_ids)
    compilation = _warm_compiled_tokenizer(codec, test_data)
    embedding_metrics = evaluate_embeddings(
        codec,
        assets.vocabulary,
        assets.input_embeddings,
        EmbeddingEvaluationRequest(
            batch_size=request.batch_size,
            device=codec.device,
            retrieval=True,
            retrieval_rows=request.retrieval_rows,
        ),
    )
    uncached, cold, warm = _benchmark_segmentation_cache(
        codec,
        request.test_windows,
        namespace=adapter.namespace,
        repetitions=request.repetitions,
    )
    if len({run.semantic_sha256 for run in (uncached, cold, warm)}) != 1:
        raise RuntimeError("encoding cache changed the canonical segmentation evidence digest")

    source_bytes = tensor_bytes(assets.input_embeddings)
    adapter.eval()
    with torch.inference_mode():
        native_aligned = tuple(
            adapter.encode_token_ids(
                window_ids,
                mode="segmented",
                alignment="aligned",
            )
            for window_ids in native_ids_by_window
        )
    native_aligned_spans = tuple(tuple(position for position in encoding.positions if isinstance(position, EncodedSpan)) for encoding in native_aligned)
    native_aligned_positions = sum(len(encoding.positions) for encoding in native_aligned)
    codec_bytes = module_bytes(codec)
    control_bytes = tensor_bytes(adapter.control_ids) + tensor_bytes(adapter.control_embeddings)
    candidate_state_bytes = codec_bytes + control_bytes
    native_tokens_per_continuous_token = len(native_ids) / max(uncached.spans, 1)
    binary_results: dict[str, dict[str, Any]] = {}
    for name, fixture in _raw_byte_fixtures().items():
        digest = hashlib.sha256(fixture).hexdigest()
        row = _window_metrics(
            codec,
            ContentWindow(digest, 0, fixture, digest),
            native_tokens=None,
            namespace=adapter.namespace,
            source_dtype=assets.input_embeddings.dtype,
        )
        binary_results[name] = {
            **row,
            "spans": row["continuous_tokens"],
            "bytes_per_span": len(fixture) / max(int(row["continuous_tokens"]), 1),
            "round_trip": row["empirical_round_trip"],
        }
    wikitext_rows = [
        _window_metrics(
            codec,
            window,
            native_tokens=len(window_ids),
            namespace=adapter.namespace,
            source_dtype=assets.input_embeddings.dtype,
        )
        for window, window_ids in zip(
            request.test_windows,
            native_ids_by_window,
            strict=True,
        )
    ]
    vocabulary_rows = _vocabulary_density_rows(
        assets,
        codec,
        count=min(256, len(assets.vocabulary.compatibility_ids)),
        namespace=adapter.namespace,
    )
    environment = runtime_environment(codec.device)
    checkpoint_bytes = checkpoint.stat().st_size
    metrics: dict[str, Any] = {
        "kind": "tokenizer_metrics",
        "benchmark_contract": {
            "schema_version": TOKENIZER_BENCHMARK_SCHEMA_VERSION,
            "timing_observation_schema_version": TIMING_OBSERVATION_SCHEMA_VERSION,
            "content_window_boundaries_preserved": True,
            "cache_modes": ["disabled", "cold", "warm"],
            "cache_mode_order": "cyclic_rotation_by_repetition",
            "raw_observations": request.repetitions * 3,
        },
        "model": {
            "id": assets.model_id,
            "revision": assets.revision,
            "embedding_tensor": assets.embedding_tensor_name,
            "source_dtype": str(assets.input_embeddings.dtype),
            "tie_word_embeddings": tie_word_embeddings(assets.config),
            "separate_input_table": not tie_word_embeddings(assets.config),
        },
        "dataset": {
            "id": request.dataset_id,
            "config": request.dataset_config,
            "revision": request.dataset_revision,
            "split": request.dataset_split,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": loaded.fingerprint,
            "serialized_bytes": checkpoint_bytes,
        },
        "codec": gqa_metadata(codec.config.query_heads),
        "compilation": compilation,
        "embedding_fit": embedding_metrics.to_dict(),
        "compatibility_scope": {
            "definition": "reachable canonical ordinary-token payload rows",
            "canonical_rows": len(assets.vocabulary.compatibility_ids),
            "ordinary_rows": len(assets.vocabulary.ordinary_ids),
            "noncanonical_alias_rows": len(assets.vocabulary.alias_ids),
        },
        "alias_groups": alias_group_evidence(assets),
        "compactness": {
            "reference_state_bytes": source_bytes,
            "candidate_codec_state_bytes": codec_bytes,
            "serialized_checkpoint_bytes": checkpoint_bytes,
            "candidate_control_state_bytes": control_bytes,
            "candidate_state_bytes": candidate_state_bytes,
            "candidate_reference_state_ratio": (candidate_state_bytes / source_bytes),
            "warm_cache_tensor_bytes": warm.cache_tensor_bytes,
            "candidate_state_and_warm_cache_bytes": (candidate_state_bytes + warm.cache_tensor_bytes),
            "candidate_reference_state_ratio_with_warm_cache": (candidate_state_bytes + warm.cache_tensor_bytes) / source_bytes,
        },
        "runtime_memory": {
            "rss_bytes": environment["rss_bytes"],
            "peak_rss_bytes": environment["peak_rss_bytes"],
            "mps_allocated_bytes": environment["mps_allocated_bytes"],
            "mps_driver_allocated_bytes": environment["mps_driver_allocated_bytes"],
        },
        "density": {
            "test_bytes": len(test_data),
            "native_tokens": len(native_ids),
            "continuous_tokens": uncached.spans,
            "bytes_per_native_token": len(test_data) / max(len(native_ids), 1),
            "bytes_per_continuous_token": uncached.bytes_per_span,
            "native_tokens_per_continuous_token": native_tokens_per_continuous_token,
            "round_trip": uncached.round_trip,
            "alignment": "arbitrary",
        },
        "native_aligned_segmentation": {
            "test_bytes": len(test_data),
            "native_tokens": len(native_ids),
            "continuous_tokens": native_aligned_positions,
            "bytes_per_continuous_token": len(test_data)
            / max(
                native_aligned_positions,
                1,
            ),
            "native_tokens_per_continuous_token": len(native_ids)
            / max(
                native_aligned_positions,
                1,
            ),
            "round_trip": all(
                reconstruct(spans) == window.payload
                for spans, window in zip(
                    native_aligned_spans,
                    request.test_windows,
                    strict=True,
                )
            ),
            "alignment": "native_token",
            "content_window_boundaries_preserved": True,
        },
        "segmentation_runs": [asdict(run) for run in (uncached, cold, warm)],
        "raw_byte_fixtures": binary_results,
        "density_strata": {
            "vocabulary": {
                "sampling": "lowest_sha256(token_id || payload), capped at 256 canonical rows",
                "windows": vocabulary_rows,
            },
            "wikitext": {
                "sampling": "fixed content-hashed document windows",
                "windows": wikitext_rows,
            },
            "deterministic_binary": {
                "sampling": "registered deterministic fixtures",
                "windows": [{"name": name, **result} for name, result in binary_results.items()],
            },
        },
        "deployment_tensors": checkpoint_tensor_inventory(checkpoint),
    }
    embedding_targets = EmbeddingFitTargets.from_mapping(loaded.metadata.get("embedding_targets"))
    compactness_target = float(
        loaded.metadata.get("candidate_reference_state_target", 0.5),
    )
    minimum_native_tokens_per_continuous_token = float(loaded.metadata["minimum_native_tokens_per_continuous_token"])
    compactness_passed = candidate_state_bytes / source_bytes <= compactness_target
    round_trip_passed = uncached.round_trip
    raw_fixtures_passed = all(result["round_trip"] for result in binary_results.values())
    wikitext_windows_passed = all(result["empirical_round_trip"] for result in wikitext_rows)
    vocabulary_windows_passed = all(result["empirical_round_trip"] for result in vocabulary_rows)
    acceptance = {
        "embedding_fit": embedding_targets.accepts(embedding_metrics),
        "compactness": compactness_passed,
        "density": (
            round_trip_passed
            and native_tokens_per_continuous_token >= minimum_native_tokens_per_continuous_token
            and raw_fixtures_passed
            and wikitext_windows_passed
            and vocabulary_windows_passed
        ),
    }
    acceptance["overall"] = bool(
        acceptance["embedding_fit"] and acceptance["density"] and (acceptance["compactness"] or not input_table_is_removable(assets.config))
    )
    metrics["acceptance"] = acceptance
    metrics["gates"] = {
        "maximum_normalized_rmse": {
            "measured": embedding_metrics.normalized_rmse,
            "operator": "<=",
            "threshold": embedding_targets.maximum_normalized_rmse,
            "passed": embedding_metrics.normalized_rmse <= embedding_targets.maximum_normalized_rmse,
        },
        "minimum_cosine_p01": {
            "measured": embedding_metrics.cosine_similarity_p01,
            "operator": ">=",
            "threshold": embedding_targets.minimum_cosine_p01,
            "passed": embedding_metrics.cosine_similarity_p01 >= embedding_targets.minimum_cosine_p01,
        },
        "minimum_cosine_p50": {
            "measured": embedding_metrics.cosine_similarity_p50,
            "operator": ">=",
            "threshold": embedding_targets.minimum_cosine_p50,
            "passed": embedding_metrics.cosine_similarity_p50 >= embedding_targets.minimum_cosine_p50,
        },
        "vocabulary_reconstruction": {
            "measured": embedding_metrics.reconstruction_fraction,
            "operator": "==",
            "threshold": 1.0,
            "passed": embedding_metrics.reconstruction_fraction == 1.0,
        },
        "exact_byte_round_trip": {
            "measured": round_trip_passed,
            "operator": "==",
            "threshold": True,
            "passed": round_trip_passed,
        },
        "raw_binary_fixtures": {
            "measured": sum(result["round_trip"] for result in binary_results.values()),
            "operator": "==",
            "threshold": len(binary_results),
            "passed": raw_fixtures_passed,
        },
        "independent_wikitext_reconstruction": {
            "measured": sum(result["empirical_round_trip"] for result in wikitext_rows),
            "operator": "==",
            "threshold": len(wikitext_rows),
            "passed": wikitext_windows_passed,
        },
        "independent_vocabulary_reconstruction": {
            "measured": sum(result["empirical_round_trip"] for result in vocabulary_rows),
            "operator": "==",
            "threshold": len(vocabulary_rows),
            "passed": vocabulary_windows_passed,
        },
        "minimum_native_tokens_per_continuous_token": {
            "measured": native_tokens_per_continuous_token,
            "operator": ">=",
            "threshold": minimum_native_tokens_per_continuous_token,
            "passed": native_tokens_per_continuous_token >= minimum_native_tokens_per_continuous_token,
        },
        "maximum_candidate_reference_state_ratio": {
            "measured": candidate_state_bytes / source_bytes,
            "operator": "<=",
            "threshold": compactness_target,
            "passed": compactness_passed,
        },
    }
    return metrics


def tokenizer_report(metrics: dict[str, Any]) -> str:
    fit = metrics["embedding_fit"]
    compactness = metrics["compactness"]
    density = metrics["density"]
    native_aligned_density = metrics["native_aligned_segmentation"]
    compactness_result = "PASS" if metrics["acceptance"]["compactness"] else "FAIL"
    if not metrics["model"]["separate_input_table"]:
        compactness_result = "NOT APPLICABLE"
    lines = [
        "# Continuous Byte Tokenizer Benchmark",
        "",
        f"- Model: `{metrics['model']['id']}`",
        f"- Revision: `{metrics['model']['revision']}`",
        f"- Tokenizer encoder attention: `{metrics['codec']['query_heads']}Q/{metrics['codec']['key_value_heads']}KV GQA`",
        f"- Exact embedding rows: `{fit['exact_rows']}/{fit['rows']}`",
        f"- Exact vocabulary reconstruction: `{fit['reconstruction_fraction']:.2%}`",
        f"- Reference input-table state: `{compactness['reference_state_bytes']:,}` bytes",
        f"- Candidate codec state: `{compactness['candidate_codec_state_bytes']:,}` bytes",
        f"- Candidate control state: `{compactness['candidate_control_state_bytes']:,}` bytes",
        f"- Total candidate state: `{compactness['candidate_state_bytes']:,}` bytes",
        f"- Warm encoding-cache tensor payload: `{compactness['warm_cache_tensor_bytes']:,}` bytes",
        f"- Candidate/reference state ratio: `{compactness['candidate_reference_state_ratio']:.4f}`",
        "- This ratio measures state compactness, not runtime allocation.",
        f"- Bytes/native token: `{density['bytes_per_native_token']:.4f}`",
        f"- Bytes/continuous token: `{density['bytes_per_continuous_token']:.4f}`",
        f"- Native tokens/continuous token: `{density['native_tokens_per_continuous_token']:.4f}`",
        f"- Native tokens/continuous token (native-token-aligned): `{native_aligned_density['native_tokens_per_continuous_token']:.4f}`",
        "- Compiler warm-up excluded from cache timings: "
        + (f"`{metrics['compilation']['warmup_seconds']:.6f}` seconds" if metrics["compilation"]["enabled"] else "`not applicable`"),
        "",
        "## Acceptance",
        "",
        f"- Embedding fit: **{'PASS' if metrics['acceptance']['embedding_fit'] else 'FAIL'}**",
        f"- Candidate-state compactness: **{compactness_result}**",
        f"- Input density: **{'PASS' if metrics['acceptance']['density'] else 'FAIL'}**",
        f"- Overall: **{'PASS' if metrics['acceptance']['overall'] else 'FAIL'}**",
        "",
        "## Cache comparison",
        "",
        (
            "| Mode | Median seconds | P95 seconds | Runs | Spans | Bytes/span | Hits | Misses | "
            "Hit rate | Logical candidates | Neural rows | Speculative rows | Padded rows | "
            "Synchronizations | H2D bytes | D2H bytes | Cache tensor bytes | Process RSS |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {run['mode']} | {run['seconds']:.6f} | {run['p95_seconds']:.6f} | "
        f"{run['repetitions']} | {run['spans']} | "
        f"{run['bytes_per_span']:.4f} | {run['cache_hits']} | {run['cache_misses']} | "
        f"{run['cache_hit_rate']:.2%} | {run['logical_candidates']} | "
        f"{run['neural_candidate_rows']} | {run['speculative_discarded_rows']} | "
        f"{run['padded_neural_rows']} | {run['synchronization_count']} | "
        f"{run['host_to_device_bytes']} | {run['device_to_host_bytes']} | "
        f"{run['cache_tensor_bytes']} | "
        f"{run['process_rss_bytes']} |"
        for run in metrics["segmentation_runs"]
    )
    lines += [
        "",
        (
            "Cache tensor bytes count stored tensor payload only. Process RSS is the measured "
            "whole-process resident set and includes allocator and runtime overhead."
        ),
        "",
        "The benchmark is tokenizer-only and makes no language-model quality claim.",
    ]
    return "\n".join(lines) + "\n"
