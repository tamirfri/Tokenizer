from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from operator import eq, ge, le, lt
from statistics import median
from typing import Any, Final, Literal, cast

from continuous_tokenizer.contracts.claims import (
    CLAIM_VERDICTS,
    ClaimVerdict,
    combine_claim_verdicts,
    directional_claims,
)
from continuous_tokenizer.contracts.input_study import (
    alignment_feasibility_verdict,
)
from continuous_tokenizer.contracts.parsing import is_lowercase_sha256
from continuous_tokenizer.contracts.performance import (
    PERFORMANCE_CLAIM_CONTEXT_FIELDS,
)

QWEN_MODEL: Final = (
    "Qwen/Qwen3.5-0.8B",
    "2fc06364715b967f1860aea9cf38778875588b17",
)
GEMMA_MODEL: Final = (
    "google/gemma-3-270m-it",
    "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
)
WIKITEXT_DATASET: Final = (
    "Salesforce/wikitext",
    "b08601e04326c79dfdd32d625aee71d232d685c3",
)
FINAL_VERIFICATION_CHECKS: Final = frozenset(
    {
        "ruff_format",
        "ruff_lint",
        "types",
        "fast_tests",
        "slow_synthetic",
        "streamlit",
        "model_tokenizers",
        "model_access",
        "representative_mps",
        "hardware_model_preflight",
    }
)
MINIMUM_LATENCY_WARMUPS: Final = 5
MINIMUM_LATENCY_REPETITIONS: Final = 20
INPUT_HEADLINE_CLAIMS: Final = (
    "input.held_out_position_compression",
    "input.registered_behavioral_similarity_tolerances",
)
_NUMERIC_COMPARISONS: Final[Mapping[str, Callable[[int | float, int | float], bool]]] = {
    "<": lt,
    "<=": le,
    ">=": ge,
    "==": eq,
}
type NumericGate = tuple[str, str, Literal["<", "<=", ">=", "=="]]
_INPUT_EMBEDDING_GATES: Final[tuple[NumericGate, ...]] = (
    ("normalized_rmse", "maximum_normalized_rmse", "<="),
    ("cosine_similarity_p01", "minimum_cosine_p01", ">="),
    ("cosine_similarity_p50", "minimum_cosine_p50", ">="),
)
_INPUT_BEHAVIOR_GATES: Final[tuple[NumericGate, ...]] = (
    ("segmented_mean_kl", "maximum_segmented_mean_kl", "<="),
    ("segmented_nll_delta", "maximum_segmented_nll_delta", "<="),
    (
        "segmented_top1_agreement",
        "minimum_segmented_top1_agreement",
        ">=",
    ),
    (
        "segmented_generation_byte_similarity",
        "minimum_segmented_generation_byte_similarity",
        ">=",
    ),
)


def _verdict(
    values: Sequence[object],
    *,
    complete: bool,
    supported: bool,
) -> ClaimVerdict:
    if not complete or not values or any(value is None for value in values):
        return "incomplete"
    return "supported" if supported else "unsupported"


def _combine_gate_verdicts(
    verdicts: Sequence[ClaimVerdict],
    *,
    complete: bool,
) -> ClaimVerdict:
    if not complete or not verdicts or "incomplete" in verdicts:
        return "incomplete"
    return "supported" if all(verdict == "supported" for verdict in verdicts) else "unsupported"


def derive_input_headline_verdict(
    claim_verdicts: Mapping[str, object],
) -> ClaimVerdict:
    verdicts = tuple(claim_verdicts.get(claim_id) for claim_id in INPUT_HEADLINE_CLAIMS)
    if any(verdict not in CLAIM_VERDICTS for verdict in verdicts):
        raise ValueError("input headline requires canonical compression and behavioral-similarity claims")
    return combine_claim_verdicts(cast(ClaimVerdict, verdict) for verdict in verdicts)


def _boolean_true_gate(
    rows: Sequence[Mapping[str, object]],
    name: str,
    *,
    complete: bool,
) -> ClaimVerdict:
    values = [row.get(name) for row in rows]
    if not complete or not values or any(not isinstance(value, bool) for value in values):
        return "incomplete"
    return "supported" if all(values) else "unsupported"


def _all_numeric(
    rows: Sequence[Mapping[str, object]],
    name: str,
) -> list[int | float] | None:
    values = [row.get(name) for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        return None
    return [value for value in values if isinstance(value, int | float)]


def _numeric_gate(
    metrics: Sequence[Mapping[str, object]],
    thresholds: Sequence[Mapping[str, object]],
    gate: NumericGate,
    *,
    complete: bool,
) -> ClaimVerdict:
    metric_name, threshold_name, operator = gate
    values = _all_numeric(metrics, metric_name)
    limits = _all_numeric(thresholds, threshold_name)
    if values is None or limits is None or len(values) != len(metrics) or len(limits) != len(metrics):
        return "incomplete"
    return _verdict(
        values,
        complete=complete,
        supported=all(_NUMERIC_COMPARISONS[operator](value, limit) for value, limit in zip(values, limits, strict=True)),
    )


def _numeric_gates(
    metrics: Sequence[Mapping[str, object]],
    thresholds: Sequence[Mapping[str, object]],
    gates: Sequence[NumericGate],
    *,
    complete: bool,
) -> tuple[ClaimVerdict, ...]:
    return tuple(
        _numeric_gate(
            metrics,
            thresholds,
            gate,
            complete=complete,
        )
        for gate in gates
    )


def _classification_evidence(
    values: Sequence[object],
    thresholds: Sequence[Mapping[str, object]],
    gates: tuple[tuple[str, str], ...],
) -> tuple[list[object], bool]:
    measured: list[object] = []
    supported = len(values) == len(thresholds)
    for value, threshold in zip(values, thresholds, strict=False):
        if not isinstance(value, Mapping):
            return [], False
        row = {name: value.get(name) for name, _ in gates}
        limits = {name: threshold.get(limit) for name, limit in gates}
        measured.extend(row.values())
        supported = supported and all(_meets_minimum(row[name], limits[name]) for name, _ in gates)
    return measured, supported


def _meets_minimum(value: object, limit: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and isinstance(limit, int | float) and not isinstance(limit, bool) and value >= limit


def _performance_eligible(
    row: Mapping[str, object],
    *,
    tokenizer: bool = False,
) -> bool:
    context = row.get("performance_context")
    if (
        row.get("evidence_scope") != "final"
        or not isinstance(context, Mapping)
        or set(context) != PERFORMANCE_CLAIM_CONTEXT_FIELDS
        or context.get("complete_condition_matrix") is not True
        or context.get("semantic_equivalent") is not True
        or not is_lowercase_sha256(context.get("semantic_sha256"))
        or context.get("no_concurrent_accelerator_work") is not True
        or context.get("no_concurrent_processes") is not True
        or context.get("denominators_registered") is not True
        or context.get("all_registered_gates_passed") is not True
    ):
        return False
    if not tokenizer:
        return True
    warmups = context.get("tokenizer_warmups")
    repetitions = context.get("tokenizer_repetitions")
    return (
        isinstance(warmups, int)
        and not isinstance(warmups, bool)
        and warmups >= MINIMUM_LATENCY_WARMUPS
        and isinstance(repetitions, int)
        and not isinstance(repetitions, bool)
        and repetitions >= MINIMUM_LATENCY_REPETITIONS
    )


def _performance_ratio_verdict(
    rows: Sequence[Mapping[str, object]],
    ratio_name: str,
    numerator_name: str,
    denominator_name: str,
    *,
    complete: bool,
) -> ClaimVerdict:
    ratios: list[float] = []
    for row in rows:
        if not _performance_eligible(row) or _input_latency_verdict((row,), complete=True) == "incomplete":
            return "incomplete"
        numerator = row.get(numerator_name)
        denominator = row.get(denominator_name)
        ratio = row.get(ratio_name)
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int | float)
            or numerator < 0
            or isinstance(denominator, bool)
            or not isinstance(denominator, int | float)
            or denominator <= 0
            or isinstance(ratio, bool)
            or not isinstance(ratio, int | float)
            or not math.isclose(float(ratio), float(numerator) / float(denominator))
        ):
            return "incomplete"
        ratios.append(float(ratio))
    return _verdict(
        ratios,
        complete=complete,
        supported=all(ratio < 1 for ratio in ratios),
    )


def _input_latency_verdict(  # noqa: PLR0911
    rows: Sequence[Mapping[str, object]],
    *,
    complete: bool,
) -> ClaimVerdict:
    modes = ("native", "compatibility", "segmented")
    measured: list[object] = []
    supported = True
    for row in rows:
        if not _performance_eligible(row):
            return "incomplete"
        timing = row.get("latency")
        if not isinstance(timing, Mapping):
            return "incomplete"
        registered_prompt_count = timing.get("registered_prompt_count")
        registered_warmups = timing.get("registered_warmups")
        registered_repetitions = timing.get("registered_repetitions")
        prompt_count = timing.get("prompt_count")
        warmups = timing.get("warmups")
        repetitions = timing.get("repetitions")
        expected_raw_pairs = timing.get("expected_raw_pairs")
        recorded_raw_pairs = timing.get("recorded_raw_pairs")
        prompt_set_sha256 = timing.get("prompt_set_sha256")
        prompt_sha256 = timing.get("prompt_sha256")
        pairs = timing.get("raw_pairs")
        if (
            not isinstance(registered_prompt_count, int)
            or isinstance(registered_prompt_count, bool)
            or registered_prompt_count < 1
            or not isinstance(registered_warmups, int)
            or isinstance(registered_warmups, bool)
            or registered_warmups < MINIMUM_LATENCY_WARMUPS
            or not isinstance(registered_repetitions, int)
            or isinstance(registered_repetitions, bool)
            or registered_repetitions < MINIMUM_LATENCY_REPETITIONS
            or not isinstance(prompt_count, int)
            or isinstance(prompt_count, bool)
            or prompt_count < 1
            or prompt_count != registered_prompt_count
            or warmups != registered_warmups
            or repetitions != registered_repetitions
            or not isinstance(warmups, int)
            or isinstance(warmups, bool)
            or warmups < MINIMUM_LATENCY_WARMUPS
            or not isinstance(repetitions, int)
            or isinstance(repetitions, bool)
            or repetitions < MINIMUM_LATENCY_REPETITIONS
            or expected_raw_pairs != prompt_count * repetitions
            or recorded_raw_pairs != expected_raw_pairs
            or not is_lowercase_sha256(prompt_set_sha256)
            or not isinstance(prompt_sha256, Sequence)
            or isinstance(prompt_sha256, str | bytes)
            or len(prompt_sha256) != prompt_count
            or any(not is_lowercase_sha256(digest) for digest in prompt_sha256)
            or timing.get("prompt_content") != "native_token_id_sequence"
            or timing.get("hash_algorithm") != "sha256"
            or timing.get("prompt_hash_serialization") != "compact_json_integer_array"
            or timing.get("prompt_set_hash_serialization") != "compact_json_array_of_integer_arrays"
            or timing.get("warmup_scope") != "each_registered_prompt_and_path"
            or timing.get("prompt_order") != "cyclic_rotation_by_repetition"
            or timing.get("path_order") != "cyclic_rotation_by_pair_execution_order"
            or timing.get("pairing") != "same_prompt_and_repetition"
            or not isinstance(pairs, Sequence)
            or isinstance(pairs, str | bytes)
            or len(pairs) != expected_raw_pairs
        ):
            return "incomplete"
        native: list[float] = []
        segmented: list[float] = []
        for pair_execution_order, pair in enumerate(pairs):
            current_pair = pair if isinstance(pair, Mapping) else {}
            repetition, prompt_execution_order = divmod(
                pair_execution_order,
                prompt_count,
            )
            prompt_index = (repetition + prompt_execution_order) % prompt_count
            path_offset = pair_execution_order % len(modes)
            expected_path_order = modes[path_offset:] + modes[:path_offset]
            stored_path_order = current_pair.get("path_order")
            paths = current_pair.get("paths")
            if (
                current_pair.get("repetition") != repetition
                or current_pair.get("prompt_execution_order") != prompt_execution_order
                or current_pair.get("pair_execution_order") != pair_execution_order
                or current_pair.get("prompt_index") != prompt_index
                or current_pair.get("prompt_sha256") != prompt_sha256[prompt_index]
                or not isinstance(stored_path_order, Sequence)
                or isinstance(stored_path_order, str | bytes)
                or tuple(stored_path_order) != expected_path_order
                or not isinstance(paths, Mapping)
                or set(paths) != set(modes)
            ):
                return "incomplete"
            path_times: dict[str, float] = {}
            for execution_order, mode in enumerate(expected_path_order):
                path = paths.get(mode)
                if not isinstance(path, Mapping) or path.get("execution_order") != execution_order:
                    return "incomplete"
                preparation_value = path.get("input_preparation_seconds")
                prefill_value = path.get("model_prefill_seconds")
                end_to_end_value = path.get("time_to_first_logit_seconds")
                if any(
                    isinstance(value, bool) or not isinstance(value, int | float) or value <= 0
                    for value in (
                        preparation_value,
                        prefill_value,
                        end_to_end_value,
                    )
                ):
                    return "incomplete"
                path_times[mode] = float(
                    cast(int | float, end_to_end_value),
                )
            native.append(path_times["native"])
            segmented.append(path_times["segmented"])
        measured.extend((native, segmented))
        supported = supported and median(segmented) < median(native)
    return _verdict(measured, complete=complete, supported=supported)


def _tokenizer_latency_verdict(  # noqa: C901, PLR0911
    rows: Sequence[Mapping[str, object]],
    *,
    complete: bool,
) -> ClaimVerdict:
    measured: list[float] = []
    supported = True
    expected_modes = {"disabled", "cold", "warm"}
    for row in rows:
        if not _performance_eligible(row, tokenizer=True):
            return "incomplete"
        performance = row.get("tokenizer_performance")
        if not isinstance(performance, Mapping):
            return "incomplete"
        contract = performance.get("benchmark_contract")
        runs = performance.get("segmentation_runs")
        if (
            not isinstance(contract, Mapping)
            or contract.get("schema_version") != 1
            or contract.get("content_window_boundaries_preserved") is not True
            or contract.get("cache_modes") != ["disabled", "cold", "warm"]
            or contract.get("cache_mode_order") != "cyclic_rotation_by_repetition"
            or not isinstance(runs, Sequence)
            or isinstance(runs, str | bytes)
        ):
            return "incomplete"
        by_mode = {str(run.get("mode")): run for run in runs if isinstance(run, Mapping)}
        context = cast(Mapping[str, object], row["performance_context"])
        repetitions = context["tokenizer_repetitions"]
        if set(by_mode) != expected_modes:
            return "incomplete"
        digests = {run.get("semantic_sha256") for run in by_mode.values()}
        if digests != {context["semantic_sha256"]}:
            return "incomplete"
        raw_values: dict[str, list[float]] = {}
        for mode, run in by_mode.items():
            raw = run.get("raw_observations")
            if run.get("repetitions") != repetitions or not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or len(raw) != repetitions:
                return "incomplete"
            seconds = []
            for observation in raw:
                timing = observation.get("timing") if isinstance(observation, Mapping) else None
                value = timing.get("wall_seconds") if isinstance(timing, Mapping) else None
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or value <= 0
                    or not isinstance(observation, Mapping)
                    or observation.get("semantic_sha256") != context["semantic_sha256"]
                ):
                    return "incomplete"
                seconds.append(float(value))
            summary_seconds = run.get("seconds")
            if isinstance(summary_seconds, bool) or not isinstance(summary_seconds, int | float) or not math.isclose(float(summary_seconds), median(seconds)):
                return "incomplete"
            raw_values[mode] = seconds
        measured.extend((*raw_values["disabled"], *raw_values["warm"]))
        supported = supported and median(raw_values["warm"]) < median(
            raw_values["disabled"],
        )
    return _verdict(measured, complete=complete, supported=supported)


def derive_input_claim_verdicts(
    metrics: Sequence[Mapping[str, object]],
    thresholds: Sequence[Mapping[str, object]],
    *,
    complete: bool,
) -> dict[str, ClaimVerdict]:
    unit_thresholds = tuple({"one": 1.0} for _ in metrics)
    embedding = (
        *_numeric_gates(
            metrics,
            thresholds,
            _INPUT_EMBEDDING_GATES,
            complete=complete,
        ),
        _numeric_gate(
            metrics,
            unit_thresholds,
            ("reconstruction_fraction", "one", "=="),
            complete=complete,
        ),
    )
    position_compression = _combine_gate_verdicts(
        (
            _boolean_true_gate(
                metrics,
                "density_exact",
                complete=complete,
            ),
            _numeric_gate(
                metrics,
                thresholds,
                (
                    "native_tokens_per_continuous_token",
                    "minimum_native_tokens_per_continuous_token",
                    ">=",
                ),
                complete=complete,
            ),
        ),
        complete=complete,
    )
    behavior_gates = _numeric_gates(
        metrics,
        thresholds,
        _INPUT_BEHAVIOR_GATES,
        complete=complete,
    )
    return {
        "input.fixed_subset_alignment_feasibility": "incomplete",
        "input.full_vocabulary_embedding_compatibility": _combine_gate_verdicts(
            embedding,
            complete=complete,
        ),
        "input.held_out_position_compression": position_compression,
        "input.registered_behavioral_similarity_tolerances": _combine_gate_verdicts(
            behavior_gates,
            complete=complete,
        ),
        "input.tokenizer_latency_improvement": _tokenizer_latency_verdict(
            metrics,
            complete=complete,
        ),
        "input.prompt_cache_reduction": _performance_ratio_verdict(
            metrics,
            "prompt_cache_ratio",
            "segmented_prompt_cache_bytes",
            "native_prompt_cache_bytes",
            complete=complete,
        ),
        "input.end_to_end_latency_improvement": _input_latency_verdict(
            metrics,
            complete=complete,
        ),
        "input.prefill_compute_reduction": _performance_ratio_verdict(
            metrics,
            "prefill_flops_ratio",
            "segmented_prefill_flops",
            "native_prefill_flops",
            complete=complete,
        ),
        "input.codec_reference_compactness": _numeric_gate(
            metrics,
            thresholds,
            (
                "candidate_reference_state_ratio",
                "maximum_candidate_reference_state_ratio",
                "<=",
            ),
            complete=complete,
        ),
        "input.physical_input_table_omission": "incomplete",
        "input.input_table_removability": "incomplete",
        "input.cross_model_confirmation": "incomplete",
    }


def derive_output_claim_verdicts(
    metrics: Sequence[Mapping[str, object]],
    thresholds: Sequence[Mapping[str, object]],
    *,
    complete: bool,
    structurally_unrepresentable: bool = False,
) -> dict[str, ClaimVerdict]:
    if structurally_unrepresentable:
        verdicts: dict[str, ClaimVerdict] = {claim.claim_id: "incomplete" for claim in directional_claims("output_only")}
        verdicts["output.semi_autoregressive_density"] = "unsupported"
        return verdicts

    direct_values = [
        value
        for row in metrics
        for value in (
            row.get("direct_feedback_byte_equality"),
            row.get("direct_feedback_token_equality"),
        )
    ]
    controls = [row.get("control_evidence") for row in metrics]
    stops = [row.get("stop_control") for row in metrics]
    control_measured, control_supported = _classification_evidence(
        controls,
        thresholds,
        (
            ("coverage", "minimum_control_prompt_coverage"),
            ("precision", "minimum_control_precision"),
            ("recall", "minimum_control_recall"),
        ),
    )
    stop_measured, stop_supported = _classification_evidence(
        stops,
        thresholds,
        (
            ("precision", "minimum_stop_precision"),
            ("recall", "minimum_stop_recall"),
        ),
    )
    native_head_counts = _all_numeric(metrics, "native_head_invocations")
    direct_feedback = _verdict(
        direct_values,
        complete=complete,
        supported=all(value == 1.0 for value in direct_values),
    )
    valid_termination = _numeric_gate(
        metrics,
        thresholds,
        (
            "valid_non_empty_termination",
            "minimum_valid_non_empty_termination",
            ">=",
        ),
        complete=complete,
    )
    no_invalid_events = _numeric_gate(
        metrics,
        thresholds,
        ("invalid_events", "maximum_invalid_events", "<="),
        complete=complete,
    )
    rollout_fidelity = _numeric_gate(
        metrics,
        thresholds,
        (
            "rollout_event_agreement",
            "minimum_rollout_event_agreement",
            ">=",
        ),
        complete=complete,
    )
    density = _numeric_gate(
        metrics,
        thresholds,
        (
            "native_tokens_per_attempted_macro_step",
            "minimum_native_tokens_per_attempted_macro_step",
            ">=",
        ),
        complete=complete,
    )
    density_prerequisites = (
        direct_feedback,
        no_invalid_events,
        valid_termination,
        rollout_fidelity,
    )
    if "incomplete" in density_prerequisites:
        conditional_density: ClaimVerdict = "incomplete"
    elif any(verdict != "supported" for verdict in density_prerequisites):
        conditional_density = "unsupported"
    else:
        conditional_density = density
    return {
        "output.direct_feedback_exactness": direct_feedback,
        "output.valid_non_empty_termination": valid_termination,
        "output.no_invalid_events": no_invalid_events,
        "output.control_exactness": _verdict(
            control_measured,
            complete=complete and len(control_measured) == 3 * len(metrics),
            supported=control_supported,
        ),
        "output.stop_exactness": _verdict(
            stop_measured,
            complete=complete and len(stop_measured) == 2 * len(metrics),
            supported=stop_supported,
        ),
        "output.rollout_fidelity": rollout_fidelity,
        "output.semi_autoregressive_density": conditional_density,
        "output.codec_reference_compactness": _numeric_gate(
            metrics,
            thresholds,
            (
                "candidate_reference_state_ratio",
                "maximum_candidate_reference_state_ratio",
                "<=",
            ),
            complete=complete,
        ),
        "output.native_head_bypass": _verdict(
            [] if native_head_counts is None else native_head_counts,
            complete=complete,
            supported=native_head_counts is not None and all(value == 0 for value in native_head_counts),
        ),
        "output.physical_output_head_omission": "incomplete",
        "output.output_head_removability": "incomplete",
        "output.cross_model_confirmation": "incomplete",
    }


def derive_project_alignment_verdict(evidence: object) -> ClaimVerdict:
    rows = _primary_model_rows(evidence)
    if rows is None:
        return "incomplete"
    verdicts: list[ClaimVerdict] = []
    for row in rows:
        result = row.get("result")
        if not isinstance(result, Mapping):
            verdicts.append("incomplete")
            continue
        study = result.get("study")
        if (
            not isinstance(study, Mapping)
            or study.get("prospective") is not True
            or result.get("prospective") is not True
            or result.get("final_evidence") is not False
        ):
            verdicts.append("incomplete")
            continue
        try:
            verdicts.append(alignment_feasibility_verdict(result))
        except TypeError, ValueError:
            verdicts.append("incomplete")
    return combine_claim_verdicts(verdicts)


def derive_project_deployment_verdicts(
    mode: Literal["input_only", "output_only"],
    evidence: object,
) -> dict[str, ClaimVerdict]:
    claim_ids = (
        ("input.physical_input_table_omission", "input.input_table_removability")
        if mode == "input_only"
        else (
            "output.physical_output_head_omission",
            "output.output_head_removability",
        )
    )
    incomplete: dict[str, ClaimVerdict] = dict.fromkeys(
        claim_ids,
        "incomplete",
    )
    rows = _primary_model_rows(evidence)
    if rows is None:
        return incomplete
    verdicts = [derive_deployment_claim_verdicts(row.get("result")) for row in rows]
    return {
        claim_ids[0]: combine_claim_verdicts(verdict[0] for verdict in verdicts),
        claim_ids[1]: combine_claim_verdicts(verdict[1] for verdict in verdicts),
    }


def derive_deployment_claim_verdicts(
    result: object,
) -> tuple[ClaimVerdict, ClaimVerdict]:
    applicability = result.get("applicability") if isinstance(result, Mapping) else None
    applicable = applicability.get("applicable") if isinstance(applicability, Mapping) else None
    if applicable is False:
        return "inapplicable", "inapplicable"
    if applicable is not True or not isinstance(result, Mapping):
        return "incomplete", "incomplete"
    omission: ClaimVerdict = (
        "supported"
        if result.get("physical_reference_tensor_absent") is True and result.get("output_equivalent") is True and result.get("hidden_equivalent") is True
        else "unsupported"
    )
    return omission, "supported"


def _primary_model_rows(
    evidence: object,
) -> list[Mapping[str, Any]] | None:
    if not isinstance(evidence, Sequence) or isinstance(evidence, str | bytes):
        return None
    rows = [cast(Mapping[str, Any], row) for row in evidence if isinstance(row, Mapping)]
    models = {
        (
            cast(Mapping[str, Any], row.get("model", {})).get("id"),
            cast(Mapping[str, Any], row.get("model", {})).get("revision"),
        )
        for row in rows
    }
    return rows if len(rows) == 2 and models == {QWEN_MODEL, GEMMA_MODEL} else None
