from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any, cast

from continuous_tokenizer.contracts.parsing import (
    is_lowercase_sha256,
    mapping_fingerprint,
)

TIMING_OBSERVATION_FIELDS = {
    "schema_version",
    "wall_seconds",
    "synchronization_count",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "rss_before_bytes",
    "rss_after_bytes",
    "peak_rss_bytes",
    "mps_allocated_before_bytes",
    "mps_allocated_after_bytes",
    "peak_mps_allocated_bytes",
    "mps_driver_before_bytes",
    "mps_driver_after_bytes",
    "peak_mps_driver_bytes",
    "mps_peak_method",
}
OUTPUT_SUBPHASE_FIELDS = {
    "schema_version",
    "preparation_seconds",
    "backbone_seconds",
    "output_decode_seconds",
    "feedback_seconds",
    "cache_accounting_seconds",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "synchronization_count",
    "backbone_calls",
    "output_decode_calls",
    "feedback_calls",
    "cache_snapshots",
    "graph_signature_counts",
}
OUTPUT_OBSERVATION_FIELDS = {
    "schema_version",
    "latency_seconds",
    "timing",
    "subphases",
    "snapshot_hash_seconds",
    "bytes",
    "bytes_sha256",
    "control_ids",
    "semantic_trajectory",
    "semantic_sha256",
    "stop_control",
    "native_tokens",
    "attempted_macro_steps",
    "model_calls",
    "final_cache_bytes",
    "peak_cache_bytes",
    "analytical_backbone_flops",
    "analytical_local_flops",
    "total_analytical_flops",
    "native_head_invocations",
}
OUTPUT_SETUP_FIELDS = {
    "prompt_index",
    "prompt_sha256",
    "order",
    "candidate_runs",
    "native_runs",
    "native_token_horizon",
    "timed",
}
OUTPUT_EQUIVALENCE_FIELDS = (
    "semantic_trajectory",
    "semantic_sha256",
    "bytes_sha256",
    "control_ids",
    "stop_control",
)
OUTPUT_PAIR_FIELDS = {
    "schema_version",
    "repetition",
    "order",
    "native",
    "candidate",
    "exact_byte_control_stop_equivalence",
    "prompt_index",
    "prompt_sha256",
}
OUTPUT_SUMMARY_PHASES = (
    "preparation",
    "backbone",
    "output_decode",
    "feedback",
    "cache_accounting",
)
SEGMENTATION_RUN_FIELDS = {
    "schema_version",
    "mode",
    "execution_mode",
    "spans",
    "seconds",
    "p95_seconds",
    "repetitions",
    "bytes_per_span",
    "round_trip",
    "source_bookkeeping_round_trip",
    "semantic_sha256",
    "span_evidence",
    "cache_entries",
    "cache_tensor_bytes",
    "cache_capacity_bytes",
    "cache_hits",
    "cache_misses",
    "cache_hit_rate",
    "cache_stores",
    "cache_evictions",
    "cache_coalesced",
    "process_rss_bytes",
    "atomic_spans",
    "candidates",
    "candidate_lengths",
    "valid_candidates",
    "invalid_by_length",
    "span_lengths",
    "content_windows",
    "workload_sha256",
    "logical_candidates",
    "neural_candidate_rows",
    "speculative_discarded_rows",
    "padded_neural_rows",
    "neural_invocations",
    "graph_signature_counts",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "synchronization_count",
    "peak_mps_allocated_bytes",
    "peak_mps_driver_bytes",
    "cache_accounting_median_seconds",
    "snapshot_hash_median_seconds",
    "raw_observations",
}
PERFORMANCE_ABLATION_CONDITIONS = frozenset(
    {
        "cold_compile",
        "warm_compile",
        "cache_disabled",
        "cache_cold",
        "cache_warm",
    }
)
_PERFORMANCE_ABLATION_IDENTITY_FIELDS = {
    "source_state_sha256",
    "dependency_lock_sha256",
    "model_id",
    "model_revision",
    "data_sha256",
    "checkpoint_sha256",
    "workload_sha256",
}
_PERFORMANCE_ABLATION_SUMMARY_FIELDS = {
    "baseline_median_seconds",
    "optimized_median_seconds",
    "median_ratio",
    "geometric_mean_ratio",
    "confidence_method",
    "confidence_level",
    "confidence_95_low",
    "confidence_95_high",
}
PERFORMANCE_CLAIM_CONTEXT_FIELDS = {
    "complete_condition_matrix",
    "semantic_equivalent",
    "semantic_sha256",
    "no_concurrent_accelerator_work",
    "no_concurrent_processes",
    "denominators_registered",
    "all_registered_gates_passed",
    "tokenizer_warmups",
    "tokenizer_repetitions",
}


def _table(value: object, name: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return None
    return cast(Mapping[str, Any], value)


def _strict_fields(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
    errors: list[str],
) -> None:
    if set(value) != expected:
        errors.append(f"{name} fields are not canonical")


def _timing_errors(value: object, name: str) -> list[str]:
    errors: list[str] = []
    timing = _table(value, name, errors)
    if timing is None:
        return errors
    _strict_fields(timing, TIMING_OBSERVATION_FIELDS, name, errors)
    if timing.get("schema_version") != 1:
        errors.append(f"{name} has an unsupported schema version")
    for field in TIMING_OBSERVATION_FIELDS - {"mps_peak_method"}:
        raw = timing.get(field)
        if field == "schema_version":
            continue
        if not isinstance(raw, int | float) or isinstance(raw, bool) or raw < 0:
            errors.append(f"{name}.{field} must be non-negative")
    return errors


def _summary(values: Sequence[float]) -> tuple[float, float]:
    median = statistics.median(values)
    p95 = values[0] if len(values) == 1 else statistics.quantiles(values, n=100, method="inclusive")[94]
    return median, p95


def tokenizer_performance_errors(  # noqa: C901, PLR0912, PLR0915
    tokenizer: object,
) -> list[str]:
    if not isinstance(tokenizer, Mapping) or not isinstance(
        tokenizer.get("benchmark_contract"),
        Mapping,
    ):
        return []
    errors: list[str] = []
    root = _table(tokenizer, "tokenizer metrics", errors)
    if root is None:
        return errors
    contract = root.get("benchmark_contract")
    if not isinstance(contract, Mapping) or contract.get("schema_version") != 1:
        return errors
    runs = root.get("segmentation_runs")
    if not isinstance(runs, Sequence) or isinstance(runs, str | bytes):
        return ["tokenizer segmentation runs are missing"]
    by_mode: dict[str, Mapping[str, Any]] = {}
    for run_index, raw_run in enumerate(runs):
        run = _table(raw_run, f"segmentation run {run_index}", errors)
        if run is None:
            continue
        _strict_fields(run, SEGMENTATION_RUN_FIELDS, f"segmentation run {run_index}", errors)
        mode = str(run.get("mode"))
        by_mode[mode] = run
        raw = run.get("raw_observations")
        if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
            errors.append(f"{mode} segmentation raw observations are missing")
            continue
        if run.get("repetitions") != len(raw):
            errors.append(f"{mode} segmentation repetition count differs from raw observations")
        seconds: list[float] = []
        cache_seconds: list[float] = []
        snapshot_seconds: list[float] = []
        synchronization_count = 0
        for observation_index, raw_observation in enumerate(raw):
            observation = _table(
                raw_observation,
                f"{mode} segmentation observation {observation_index}",
                errors,
            )
            if observation is None:
                continue
            _strict_fields(
                observation,
                {
                    "schema_version",
                    "repetition",
                    "execution_order",
                    "mode",
                    "subphases",
                    "timing",
                    "semantic_sha256",
                    "logical_candidates",
                    "neural_candidate_rows",
                    "speculative_discarded_rows",
                    "padded_neural_rows",
                    "neural_invocations",
                    "graph_signature_counts",
                    "host_to_device_bytes",
                    "device_to_host_bytes",
                },
                f"{mode} segmentation observation {observation_index}",
                errors,
            )
            errors.extend(
                _timing_errors(
                    observation.get("timing"),
                    f"{mode} segmentation timing",
                )
            )
            subphases = _table(
                observation.get("subphases"),
                f"{mode} segmentation subphases",
                errors,
            )
            timing = observation.get("timing")
            if subphases is None or not isinstance(timing, Mapping):
                continue
            _strict_fields(
                subphases,
                {
                    "segmentation_encoding_seconds",
                    "cache_accounting_seconds",
                    "snapshot_hash_seconds",
                },
                f"{mode} segmentation subphases",
                errors,
            )
            if subphases.get("segmentation_encoding_seconds") != timing.get("wall_seconds"):
                errors.append(f"{mode} segmentation phase differs from raw timing")
            seconds.append(float(timing["wall_seconds"]))
            cache_seconds.append(float(subphases["cache_accounting_seconds"]))
            snapshot_seconds.append(float(subphases["snapshot_hash_seconds"]))
            synchronization_count += int(timing["synchronization_count"])
            fields = (
                "semantic_sha256",
                "logical_candidates",
                "neural_candidate_rows",
                "speculative_discarded_rows",
                "padded_neural_rows",
                "neural_invocations",
                "graph_signature_counts",
                "host_to_device_bytes",
                "device_to_host_bytes",
            )
            errors.extend(f"{mode} segmentation {field} differs from raw observation" for field in fields if observation.get(field) != run.get(field))
        if seconds:
            median, p95 = _summary(seconds)
            if run.get("seconds") != median or run.get("p95_seconds") != p95:
                errors.append(f"{mode} segmentation timing summary differs from raw observations")
            if run.get("cache_accounting_median_seconds") != _summary(cache_seconds)[0]:
                errors.append(f"{mode} cache timing differs from raw observations")
            if run.get("snapshot_hash_median_seconds") != _summary(snapshot_seconds)[0]:
                errors.append(f"{mode} snapshot timing differs from raw observations")
            if run.get("synchronization_count") != synchronization_count:
                errors.append(f"{mode} synchronization count differs from raw observations")
    if set(by_mode) != {"disabled", "cold", "warm"}:
        errors.append("tokenizer cache timing matrix is incomplete")
    elif len({run.get("semantic_sha256") for run in by_mode.values()}) != 1:
        errors.append("tokenizer cache modes have different semantic digests")
    return errors


def prefill_performance_errors(  # noqa: C901, PLR0912
    performance: object,
) -> list[str]:
    if not isinstance(performance, Mapping) or performance.get("schema_version") != 1:
        return []
    errors: list[str] = []
    root = _table(performance, "prefill performance", errors)
    if root is None or root.get("schema_version") != 1:
        return errors
    _strict_fields(
        root,
        {
            "schema_version",
            "measurement",
            "raw_pairs",
            "native",
            "compatibility",
            "segmented",
        },
        "prefill performance",
        errors,
    )
    measurement = _table(root.get("measurement"), "prefill measurement", errors)
    raw = root.get("raw_pairs")
    if measurement is None or not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        errors.append("prefill raw pairs are missing")
        return errors
    expected = measurement.get("expected_raw_pairs")
    if expected != len(raw) or measurement.get("recorded_raw_pairs") != len(raw):
        errors.append("prefill raw pair count differs from measurement metadata")
    observations: dict[str, dict[str, list[float]]] = {
        mode: {
            "input_preparation": [],
            "model_prefill": [],
            "cache_accounting": [],
            "time_to_first_logit": [],
        }
        for mode in ("native", "compatibility", "segmented")
    }
    for pair_index, raw_pair in enumerate(raw):
        pair = _table(raw_pair, f"prefill raw pair {pair_index}", errors)
        if pair is None:
            continue
        _strict_fields(
            pair,
            {
                "schema_version",
                "prompt_index",
                "prompt_sha256",
                "repetition",
                "prompt_execution_order",
                "pair_execution_order",
                "path_order",
                "paths",
            },
            f"prefill raw pair {pair_index}",
            errors,
        )
        paths = _table(pair.get("paths"), f"prefill raw pair {pair_index}.paths", errors)
        if paths is None or set(paths) != set(observations):
            errors.append(f"prefill raw pair {pair_index} has incomplete paths")
            continue
        for mode, raw_path in paths.items():
            path = _table(raw_path, f"prefill {mode} path", errors)
            if path is None:
                continue
            _strict_fields(
                path,
                {
                    "schema_version",
                    "execution_order",
                    "input_preparation_seconds",
                    "model_prefill_seconds",
                    "cache_accounting_seconds",
                    "time_to_first_logit_seconds",
                    "subphase_sum_seconds",
                    "timing",
                },
                f"prefill {mode} path",
                errors,
            )
            errors.extend(_timing_errors(path.get("timing"), f"prefill {mode} timing"))
            timing = path.get("timing")
            if isinstance(timing, Mapping) and path.get("time_to_first_logit_seconds") != timing.get("wall_seconds"):
                errors.append(f"prefill {mode} direct timing differs from its raw observation")
            if path.get("subphase_sum_seconds") != path.get("input_preparation_seconds", 0) + path.get("model_prefill_seconds", 0):
                errors.append(f"prefill {mode} subphase arithmetic is invalid")
            for phase in observations[str(mode)]:
                observations[str(mode)][phase].append(float(path[f"{phase}_seconds"]))
    for mode, phases in observations.items():
        summary = _table(root.get(mode), f"prefill {mode} summary", errors)
        if summary is None:
            continue
        for phase, values in phases.items():
            if not values:
                continue
            median, p95 = _summary(values)
            if summary.get(f"{phase}_median_seconds") != median or summary.get(f"{phase}_p95_seconds") != p95:
                errors.append(f"prefill {mode} {phase} summary differs from raw observations")
    return errors


def performance_claim_context_errors(value: object) -> list[str]:
    if value is None:
        return []
    errors: list[str] = []
    context = _table(value, "performance claim context", errors)
    if context is None:
        return errors
    _strict_fields(
        context,
        PERFORMANCE_CLAIM_CONTEXT_FIELDS,
        "performance claim context",
        errors,
    )
    errors.extend(
        f"performance claim context {name} must be boolean"
        for name in (
            "complete_condition_matrix",
            "semantic_equivalent",
            "no_concurrent_accelerator_work",
            "no_concurrent_processes",
            "denominators_registered",
            "all_registered_gates_passed",
        )
        if not isinstance(context.get(name), bool)
    )
    if not is_lowercase_sha256(context.get("semantic_sha256")):
        errors.append("performance claim context semantic digest is invalid")
    for name in ("tokenizer_warmups", "tokenizer_repetitions"):
        raw = context.get(name)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            errors.append(f"performance claim context {name} must be non-negative")
    return errors


def _output_setup_horizons(
    measurement: Mapping[str, Any],
    prompt_count: object,
    prompt_hashes: Sequence[object],
    errors: list[str],
) -> tuple[dict[int, int], bool]:
    raw_runs = measurement.get("setup_runs")
    balanced = True
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, str | bytes) or not isinstance(prompt_count, int) or len(raw_runs) != prompt_count:
        errors.append("output setup run inventory is invalid")
        raw_runs = ()
        balanced = False
    horizons: dict[int, int] = {}
    for index, raw_run in enumerate(raw_runs):
        run = _table(raw_run, f"output setup run {index}", errors)
        if run is None:
            continue
        _strict_fields(
            run,
            OUTPUT_SETUP_FIELDS,
            f"output setup run {index}",
            errors,
        )
        horizon = run.get("native_token_horizon")
        valid = (
            run.get("prompt_index") == index
            and index < len(prompt_hashes)
            and run.get("prompt_sha256") == prompt_hashes[index]
            and run.get("order")
            == [
                "candidate_horizon_discovery",
                "native_matched_horizon",
            ]
            and run.get("candidate_runs") == 1
            and run.get("native_runs") == 1
            and run.get("timed") is False
            and isinstance(horizon, int)
            and not isinstance(horizon, bool)
            and horizon > 0
        )
        if not valid:
            errors.append(f"output setup run {index} is invalid")
            balanced = False
            continue
        horizons[index] = horizon
    return horizons, balanced


def _validate_output_semantic_snapshot(
    observation: Mapping[str, Any],
    variant: str,
    errors: list[str],
) -> None:
    semantic = observation.get("semantic_trajectory")
    if (
        not isinstance(semantic, Sequence)
        or isinstance(semantic, str | bytes)
        or not all(
            isinstance(unit, Sequence)
            and not isinstance(unit, str | bytes)
            and len(unit) == 2
            and unit[0] in {"byte", "control"}
            and isinstance(unit[1], int)
            and not isinstance(unit[1], bool)
            and (0 <= unit[1] <= 255 if unit[0] == "byte" else unit[1] >= 0)
            for unit in semantic
        )
    ):
        errors.append(f"output {variant} semantic trajectory is invalid")
        return
    canonical = [(str(unit[0]), int(unit[1])) for unit in cast(Sequence[Sequence[Any]], semantic)]
    data = bytes(value for kind, value in canonical if kind == "byte")
    controls = [value for kind, value in canonical if kind == "control"]
    observed_controls = observation.get("control_ids")
    if (
        observation.get("semantic_sha256") != mapping_fingerprint(canonical)
        or observation.get("bytes") != len(data)
        or observation.get("bytes_sha256") != hashlib.sha256(data).hexdigest()
        or not isinstance(observed_controls, Sequence)
        or isinstance(observed_controls, str | bytes)
        or list(observed_controls) != controls
    ):
        errors.append(f"output {variant} semantic snapshot is inconsistent")


def _output_observations_equivalent(
    observations: Mapping[str, Mapping[str, Any]],
) -> bool:
    native = observations["native"]
    candidate = observations["candidate"]
    return all(native.get(name) == candidate.get(name) for name in OUTPUT_EQUIVALENCE_FIELDS)


def _output_latency_speedup(
    root: Mapping[str, Any],
    *,
    exact_equivalence: bool,
) -> float | None:
    native = root.get("native")
    candidate = root.get("candidate")
    if (
        not exact_equivalence
        or not isinstance(native, Mapping)
        or not isinstance(candidate, Mapping)
        or not isinstance(native.get("latency_median_seconds"), int | float)
        or not isinstance(candidate.get("latency_median_seconds"), int | float)
        or float(candidate["latency_median_seconds"]) <= 0
    ):
        return None
    return float(native["latency_median_seconds"]) / float(
        candidate["latency_median_seconds"],
    )


def _validate_output_pairs(  # noqa: C901, PLR0912, PLR0913 - Artifact dimensions stay explicit.
    raw_pairs: Sequence[object],
    *,
    prompt_count: object,
    repetitions: object,
    prompt_hashes: Sequence[object],
    setup_horizons: Mapping[int, int],
    complete: bool,
    balanced: bool,
    errors: list[str],
) -> tuple[
    dict[str, list[Mapping[str, Any]]],
    tuple[bool, ...],
    bool,
    bool,
]:
    observations: dict[str, list[Mapping[str, Any]]] = {
        "native": [],
        "candidate": [],
    }
    observed_pairs: set[tuple[int, int]] = set()
    equivalences: list[bool] = []
    for pair_index, raw_pair in enumerate(raw_pairs):
        pair = _table(raw_pair, f"output raw pair {pair_index}", errors)
        if pair is None:
            continue
        _strict_fields(
            pair,
            OUTPUT_PAIR_FIELDS,
            f"output raw pair {pair_index}",
            errors,
        )
        repetition = pair.get("repetition")
        prompt_index = pair.get("prompt_index")
        if (
            not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or not isinstance(prompt_index, int)
            or isinstance(prompt_index, bool)
            or not isinstance(prompt_count, int)
            or not isinstance(repetitions, int)
            or repetition not in range(repetitions)
            or prompt_index not in range(prompt_count)
            or (repetition, prompt_index) in observed_pairs
        ):
            complete = False
            errors.append(
                f"output raw pair {pair_index} has an invalid pairing identity",
            )
        else:
            observed_pairs.add((repetition, prompt_index))
            if prompt_index >= len(prompt_hashes) or pair.get("prompt_sha256") != prompt_hashes[prompt_index]:
                complete = False
                errors.append(
                    f"output raw pair {pair_index} prompt hash differs from measurement",
                )
        pair_observations: dict[str, Mapping[str, Any]] = {}
        for variant, variant_observations in observations.items():
            observation = _table(
                pair.get(variant),
                f"output {variant} observation",
                errors,
            )
            if observation is None:
                continue
            _strict_fields(
                observation,
                OUTPUT_OBSERVATION_FIELDS,
                f"output {variant} observation",
                errors,
            )
            errors.extend(
                _timing_errors(
                    observation.get("timing"),
                    f"output {variant} timing",
                ),
            )
            subphases = _table(
                observation.get("subphases"),
                f"output {variant} subphases",
                errors,
            )
            if subphases is not None:
                _strict_fields(
                    subphases,
                    OUTPUT_SUBPHASE_FIELDS,
                    f"output {variant} subphases",
                    errors,
                )
            timing = observation.get("timing")
            if isinstance(timing, Mapping) and observation.get("latency_seconds") != timing.get("wall_seconds"):
                errors.append(
                    f"output {variant} latency differs from its raw observation",
                )
            if (
                variant == "candidate"
                and isinstance(prompt_index, int)
                and not isinstance(prompt_index, bool)
                and prompt_index in setup_horizons
                and observation.get("native_tokens") != setup_horizons[prompt_index]
            ):
                balanced = False
                errors.append(
                    f"output candidate horizon differs from setup run {prompt_index}",
                )
            _validate_output_semantic_snapshot(observation, variant, errors)
            variant_observations.append(observation)
            pair_observations[variant] = observation
        if set(pair_observations) == {"native", "candidate"}:
            equivalent = _output_observations_equivalent(pair_observations)
            equivalences.append(equivalent)
            if pair.get("exact_byte_control_stop_equivalence") is not equivalent:
                errors.append(
                    f"output raw pair {pair_index} equivalence verdict is inconsistent",
                )
    if (
        isinstance(prompt_count, int)
        and isinstance(repetitions, int)
        and observed_pairs != {(repetition, prompt_index) for repetition in range(repetitions) for prompt_index in range(prompt_count)}
    ):
        complete = False
        errors.append("output raw paired observation matrix is incomplete")
    return observations, tuple(equivalences), complete, balanced


def output_performance_errors(  # noqa: C901, PLR0912
    output: object,
) -> list[str]:
    if not isinstance(output, Mapping) or output.get("schema_version") != 1:
        return []
    errors: list[str] = []
    root = _table(output, "output performance", errors)
    if root is None or root.get("schema_version") != 1:
        return errors
    measurement = _table(root.get("measurement"), "output measurement", errors)
    raw = root.get("raw_repetitions")
    if measurement is None or not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        errors.append("output raw repetitions are missing")
        return errors
    prompt_count = measurement.get("prompt_count")
    repetitions = measurement.get("repetitions")
    warmups = measurement.get("warmups")
    complete_pairs = (
        isinstance(prompt_count, int)
        and not isinstance(prompt_count, bool)
        and prompt_count > 0
        and isinstance(repetitions, int)
        and not isinstance(repetitions, bool)
        and repetitions > 0
        and len(raw) == prompt_count * repetitions
        and measurement.get("expected_raw_repetitions") == len(raw)
        and measurement.get("recorded_raw_repetitions") == len(raw)
    )
    if not complete_pairs:
        errors.append("output raw repetition count differs from measurement metadata")
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        errors.append("output warmup count is invalid")
    prompt_hashes = measurement.get("prompt_sha256")
    if (
        not isinstance(prompt_hashes, Sequence)
        or isinstance(prompt_hashes, str | bytes)
        or not isinstance(prompt_count, int)
        or len(prompt_hashes) != prompt_count
    ):
        errors.append("output prompt hash inventory is invalid")
        prompt_hashes = ()
    setup_horizons, balanced_setup = _output_setup_horizons(
        measurement,
        prompt_count,
        prompt_hashes,
        errors,
    )
    observations, pair_equivalence, complete_pairs, balanced_setup = _validate_output_pairs(
        raw,
        prompt_count=prompt_count,
        repetitions=repetitions,
        prompt_hashes=prompt_hashes,
        setup_horizons=setup_horizons,
        complete=complete_pairs,
        balanced=balanced_setup,
        errors=errors,
    )
    for variant, rows in observations.items():
        summary = _table(root.get(variant), f"output {variant} summary", errors)
        if summary is None or not rows:
            continue
        median, p95 = _summary([float(row["latency_seconds"]) for row in rows])
        if summary.get("latency_median_seconds") != median or summary.get("latency_p95_seconds") != p95:
            errors.append(f"output {variant} latency summary differs from raw observations")
        for phase in OUTPUT_SUMMARY_PHASES:
            values = [float(row["subphases"][f"{phase}_seconds"]) for row in rows]
            phase_median, phase_p95 = _summary(values)
            if summary.get(f"{phase}_median_seconds") != phase_median or summary.get(f"{phase}_p95_seconds") != phase_p95:
                errors.append(f"output {variant} {phase} summary differs from raw observations")
    exact_equivalence = bool(pair_equivalence) and all(pair_equivalence)
    if root.get("exact_trajectory_equivalence") is not exact_equivalence:
        errors.append("output exact trajectory equivalence verdict is inconsistent")
    claimable = (
        exact_equivalence
        and complete_pairs
        and balanced_setup
        and isinstance(warmups, int)
        and warmups >= 5
        and isinstance(repetitions, int)
        and repetitions >= 20
    )
    if root.get("speedup_claimable") is not claimable:
        errors.append("output speedup claimability verdict is inconsistent")
    expected_speedup = _output_latency_speedup(
        root,
        exact_equivalence=exact_equivalence,
    )
    if root.get("latency_speedup") != expected_speedup:
        errors.append("output latency speedup is inconsistent")
    return errors


def _ablation_interval(
    baseline: Sequence[float],
    optimized: Sequence[float],
) -> tuple[float, float, float]:
    log_ratios = [
        math.log(optimized_value / baseline_value)
        for baseline_value, optimized_value in zip(
            baseline,
            optimized,
            strict=True,
        )
    ]
    mean = statistics.fmean(log_ratios)
    if len(log_ratios) == 1:
        return math.exp(mean), math.exp(mean), math.exp(mean)
    standard_error = statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
    half_width = NormalDist().inv_cdf(0.975) * standard_error
    return math.exp(mean), math.exp(mean - half_width), math.exp(mean + half_width)


def performance_ablation_errors(  # noqa: C901, PLR0912, PLR0915
    artifact: object,
) -> list[str]:
    errors: list[str] = []
    root = _table(artifact, "performance ablation", errors)
    if root is None:
        return errors
    _strict_fields(
        root,
        {
            "schema_version",
            "artifact_kind",
            "mode",
            "evidence_scope",
            "operational_status",
            "final_evidence",
            "evidence_role",
            "baseline",
            "optimized",
            "optimization_ids",
            "semantic_sha256",
            "conditions",
        },
        "performance ablation",
        errors,
    )
    if (
        root.get("schema_version") != 1
        or root.get("artifact_kind") != "performance_ablation"
        or root.get("mode") not in {"input_only", "output_only"}
        or root.get("evidence_scope") != "operational_secondary"
        or root.get("operational_status") != "completed"
        or root.get("final_evidence") is not False
        or root.get("evidence_role") != "operational_and_secondary_only"
    ):
        errors.append("performance ablation classification is invalid")
    semantic_sha256 = root.get("semantic_sha256")
    if not is_lowercase_sha256(semantic_sha256):
        errors.append("performance ablation semantic digest is invalid")
    identities = []
    for name in ("baseline", "optimized"):
        identity = _table(root.get(name), f"performance ablation {name}", errors)
        if identity is None:
            continue
        _strict_fields(
            identity,
            _PERFORMANCE_ABLATION_IDENTITY_FIELDS,
            f"performance ablation {name}",
            errors,
        )
        errors.extend(
            f"performance ablation {name}.{field} is invalid"
            for field in (
                "source_state_sha256",
                "dependency_lock_sha256",
                "data_sha256",
                "checkpoint_sha256",
                "workload_sha256",
            )
            if not is_lowercase_sha256(identity.get(field))
        )
        errors.extend(
            f"performance ablation {name}.{field} is invalid"
            for field in ("model_id", "model_revision")
            if not isinstance(identity.get(field), str) or not identity[field]
        )
        identities.append(identity)
    if len(identities) == 2:
        baseline, optimized = identities
        shared = _PERFORMANCE_ABLATION_IDENTITY_FIELDS - {
            "source_state_sha256",
        }
        if any(baseline.get(field) != optimized.get(field) for field in shared):
            errors.append("performance ablation baseline and optimized identities are not paired")
    optimization_ids = root.get("optimization_ids")
    if (
        not isinstance(optimization_ids, Sequence)
        or isinstance(optimization_ids, str | bytes)
        or not optimization_ids
        or any(not isinstance(value, str) or not value for value in optimization_ids)
        or len(set(optimization_ids)) != len(optimization_ids)
    ):
        errors.append("performance ablation optimization IDs are not canonical")
    conditions = _table(root.get("conditions"), "performance ablation conditions", errors)
    if conditions is None:
        return errors
    if set(conditions) != PERFORMANCE_ABLATION_CONDITIONS:
        errors.append("performance ablation condition matrix is incomplete")
    for name, raw_condition in conditions.items():
        condition = _table(
            raw_condition,
            f"performance ablation condition {name}",
            errors,
        )
        if condition is None:
            continue
        _strict_fields(
            condition,
            {
                "warmups",
                "repetitions",
                "no_concurrent_accelerator_work",
                "no_concurrent_processes",
                "semantic_sha256",
                "raw_pairs",
                "summary",
            },
            f"performance ablation condition {name}",
            errors,
        )
        repetitions = condition.get("repetitions")
        raw_pairs = condition.get("raw_pairs")
        if (
            not isinstance(condition.get("warmups"), int)
            or isinstance(condition.get("warmups"), bool)
            or condition["warmups"] < 5
            or not isinstance(repetitions, int)
            or isinstance(repetitions, bool)
            or repetitions < 20
            or condition.get("no_concurrent_accelerator_work") is not True
            or condition.get("no_concurrent_processes") is not True
            or condition.get("semantic_sha256") != semantic_sha256
            or not isinstance(raw_pairs, Sequence)
            or isinstance(raw_pairs, str | bytes)
            or len(raw_pairs) != repetitions
        ):
            errors.append(f"performance ablation condition {name} is incomplete")
            continue
        baseline_values: list[float] = []
        optimized_values: list[float] = []
        for index, raw_pair in enumerate(raw_pairs):
            pair = _table(
                raw_pair,
                f"performance ablation condition {name} pair {index}",
                errors,
            )
            if pair is None:
                continue
            _strict_fields(
                pair,
                {
                    "repetition",
                    "baseline_seconds",
                    "optimized_seconds",
                    "baseline_semantic_sha256",
                    "optimized_semantic_sha256",
                },
                f"performance ablation condition {name} pair {index}",
                errors,
            )
            baseline_seconds = pair.get("baseline_seconds")
            optimized_seconds = pair.get("optimized_seconds")
            if (
                pair.get("repetition") != index
                or pair.get("baseline_semantic_sha256") != semantic_sha256
                or pair.get("optimized_semantic_sha256") != semantic_sha256
                or isinstance(baseline_seconds, bool)
                or not isinstance(baseline_seconds, int | float)
                or baseline_seconds <= 0
                or isinstance(optimized_seconds, bool)
                or not isinstance(optimized_seconds, int | float)
                or optimized_seconds <= 0
            ):
                errors.append(f"performance ablation condition {name} pair {index} is invalid")
                continue
            baseline_values.append(float(baseline_seconds))
            optimized_values.append(float(optimized_seconds))
        if len(baseline_values) != repetitions:
            continue
        summary = _table(
            condition.get("summary"),
            f"performance ablation condition {name} summary",
            errors,
        )
        if summary is None:
            continue
        _strict_fields(
            summary,
            _PERFORMANCE_ABLATION_SUMMARY_FIELDS,
            f"performance ablation condition {name} summary",
            errors,
        )
        baseline_median = statistics.median(baseline_values)
        optimized_median = statistics.median(optimized_values)
        geometric_ratio, low, high = _ablation_interval(
            baseline_values,
            optimized_values,
        )
        expected = {
            "baseline_median_seconds": baseline_median,
            "optimized_median_seconds": optimized_median,
            "median_ratio": optimized_median / baseline_median,
            "geometric_mean_ratio": geometric_ratio,
            "confidence_method": "paired_log_ratio_normal",
            "confidence_level": 0.95,
            "confidence_95_low": low,
            "confidence_95_high": high,
        }
        for field, value in expected.items():
            actual = summary.get(field)
            equal = (
                math.isclose(float(actual), value)
                if isinstance(value, float) and isinstance(actual, int | float) and not isinstance(actual, bool)
                else actual == value
            )
            if not equal:
                errors.append(
                    f"performance ablation condition {name} summary differs from raw pairs",
                )
                break
    return errors
