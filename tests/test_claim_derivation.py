from __future__ import annotations

import unittest
from typing import cast

from continuous_tokenizer.contracts.claim_derivation import (
    derive_input_claim_verdicts,
    derive_input_headline_verdict,
    derive_output_claim_verdicts,
)
from tests.test_replication import performance_measurement


def test_output_control_claim_uses_coverage_precision_and_recall() -> None:
    metrics = (
        {
            "control_evidence": {
                "coverage": 1.0,
                "precision": 1.0,
                "recall": 1.0,
            },
        },
    )
    thresholds = (
        {
            "minimum_control_prompt_coverage": 1.0,
            "minimum_control_precision": 1.0,
            "minimum_control_recall": 1.0,
        },
    )

    claims = derive_output_claim_verdicts(metrics, thresholds, complete=True)

    assert claims["output.control_exactness"] == "supported"


def test_output_control_claim_rejects_legacy_correctness_shape() -> None:
    metrics = (
        {
            "control_evidence": {
                "correctness": 1.0,
            },
        },
    )
    thresholds = (
        {
            "minimum_control_prompt_coverage": 1.0,
            "minimum_control_precision": 1.0,
            "minimum_control_recall": 1.0,
        },
    )

    claims = derive_output_claim_verdicts(metrics, thresholds, complete=True)

    assert claims["output.control_exactness"] == "incomplete"


def test_output_native_head_bypass_is_derived_from_invocations() -> None:
    thresholds = ({},)

    bypassed = derive_output_claim_verdicts(
        ({"native_head_invocations": 0},),
        thresholds,
        complete=True,
    )
    invoked = derive_output_claim_verdicts(
        ({"native_head_invocations": 1},),
        thresholds,
        complete=True,
    )

    assert bypassed["output.native_head_bypass"] == "supported"
    assert invoked["output.native_head_bypass"] == "unsupported"


def test_input_latency_requires_registered_raw_timing() -> None:
    insufficient = (
        {
            "latency": {
                "warmups": 4,
                "repetitions": 1,
                "raw_pairs": [
                    {
                        "native_seconds": 2.0,
                        "segmented_seconds": 1.0,
                    },
                ],
            },
        },
    )

    claims = derive_input_claim_verdicts(insufficient, ({},), complete=True)

    assert claims["input.end_to_end_latency_improvement"] == "incomplete"


def test_input_compression_requires_exact_density() -> None:
    claims = derive_input_claim_verdicts(
        (
            {
                "density_exact": False,
                "native_tokens_per_continuous_token": 2.0,
            },
        ),
        ({"minimum_native_tokens_per_continuous_token": 1.1},),
        complete=True,
    )

    assert claims["input.held_out_position_compression"] == "unsupported"


def test_input_compression_is_independent_from_alignment() -> None:
    claims = derive_input_claim_verdicts(
        (
            {
                "normalized_rmse": 0.5,
                "cosine_similarity_p01": 0.5,
                "cosine_similarity_p50": 0.5,
                "reconstruction_fraction": 1.0,
                "density_exact": True,
                "native_tokens_per_continuous_token": 1.2,
            },
        ),
        (
            {
                "maximum_normalized_rmse": 0.01,
                "minimum_cosine_p01": 0.999,
                "minimum_cosine_p50": 0.9999,
                "minimum_native_tokens_per_continuous_token": 1.1,
            },
        ),
        complete=True,
    )

    assert claims["input.full_vocabulary_embedding_compatibility"] == "unsupported"
    assert claims["input.held_out_position_compression"] == "supported"


def test_input_behavior_blocks_headline_without_rewriting_compression() -> None:
    claims = derive_input_claim_verdicts(
        (
            {
                "density_exact": True,
                "native_tokens_per_continuous_token": 1.2,
                "segmented_mean_kl": 0.2,
                "segmented_nll_delta": 0.01,
                "segmented_top1_agreement": 1.0,
                "segmented_generation_byte_similarity": 1.0,
            },
        ),
        (
            {
                "minimum_native_tokens_per_continuous_token": 1.1,
                "maximum_segmented_mean_kl": 0.1,
                "maximum_segmented_nll_delta": 0.1,
                "minimum_segmented_top1_agreement": 0.9,
                "minimum_segmented_generation_byte_similarity": 0.5,
            },
        ),
        complete=True,
    )

    assert claims["input.held_out_position_compression"] == "supported"
    assert claims["input.registered_behavioral_similarity_tolerances"] == ("unsupported")
    assert derive_input_headline_verdict(claims) == "unsupported"


def _registered_latency() -> dict[str, object]:
    performance = performance_measurement(prompt_count=2)
    measurement = cast(dict[str, object], performance["measurement"])
    return {
        "registered_prompt_count": measurement["prompt_count"],
        "registered_warmups": measurement["warmups"],
        "registered_repetitions": measurement["repetitions"],
        **measurement,
        "raw_pairs": performance["raw_pairs"],
    }


def _performance_context(**changes: object) -> dict[str, object]:
    context: dict[str, object] = {
        "complete_condition_matrix": True,
        "semantic_equivalent": True,
        "semantic_sha256": "a" * 64,
        "no_concurrent_accelerator_work": True,
        "no_concurrent_processes": True,
        "denominators_registered": True,
        "all_registered_gates_passed": True,
        "tokenizer_warmups": 5,
        "tokenizer_repetitions": 20,
    }
    context.update(changes)
    return context


def _performance_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "evidence_scope": "final",
        "performance_context": _performance_context(),
        "latency": _registered_latency(),
        "prompt_cache_ratio": 0.75,
        "native_prompt_cache_bytes": 100,
        "segmented_prompt_cache_bytes": 75,
        "prefill_flops_ratio": 0.75,
        "native_prefill_flops": 100,
        "segmented_prefill_flops": 75,
    }
    row.update(changes)
    return row


def test_input_latency_requires_complete_rotated_pairs() -> None:
    latency = _registered_latency()

    claims = derive_input_claim_verdicts(
        (_performance_row(latency=latency),),
        ({},),
        complete=True,
    )
    assert claims["input.end_to_end_latency_improvement"] == "supported"

    pairs = latency["raw_pairs"]
    assert isinstance(pairs, list)
    first = cast(dict[str, object], pairs[0])
    first["path_order"] = ["segmented", "compatibility", "native"]
    claims = derive_input_claim_verdicts(
        (_performance_row(latency=latency),),
        ({},),
        complete=True,
    )
    assert claims["input.end_to_end_latency_improvement"] == "incomplete"


def test_input_performance_is_final_only_and_rejects_contamination() -> None:
    prospective = derive_input_claim_verdicts(
        (_performance_row(evidence_scope="selection"),),
        ({},),
        complete=True,
    )
    contaminated = derive_input_claim_verdicts(
        (
            _performance_row(
                performance_context=_performance_context(
                    no_concurrent_processes=False,
                ),
            ),
        ),
        ({},),
        complete=True,
    )

    for claims in (prospective, contaminated):
        assert claims["input.prompt_cache_reduction"] == "incomplete"
        assert claims["input.end_to_end_latency_improvement"] == "incomplete"
        assert claims["input.prefill_compute_reduction"] == "incomplete"


def test_input_performance_requires_raw_matrix_and_denominators() -> None:
    missing_raw = _performance_row(latency={})
    zero_denominator = _performance_row(native_prompt_cache_bytes=0)

    missing = derive_input_claim_verdicts((missing_raw,), ({},), complete=True)
    denominator = derive_input_claim_verdicts(
        (zero_denominator,),
        ({},),
        complete=True,
    )

    assert missing["input.end_to_end_latency_improvement"] == "incomplete"
    assert denominator["input.prompt_cache_reduction"] == "incomplete"


def test_input_performance_regression_is_unsupported() -> None:
    row = _performance_row(
        prompt_cache_ratio=1.1,
        segmented_prompt_cache_bytes=110,
        prefill_flops_ratio=1.1,
        segmented_prefill_flops=110,
    )
    latency = cast(dict[str, object], row["latency"])
    pairs = cast(list[dict[str, object]], latency["raw_pairs"])
    for pair in pairs:
        paths = cast(dict[str, dict[str, object]], pair["paths"])
        paths["segmented"]["time_to_first_logit_seconds"] = 1.1

    claims = derive_input_claim_verdicts((row,), ({},), complete=True)

    assert claims["input.prompt_cache_reduction"] == "unsupported"
    assert claims["input.end_to_end_latency_improvement"] == "unsupported"
    assert claims["input.prefill_compute_reduction"] == "unsupported"


def test_tokenizer_latency_requires_complete_exact_cache_matrix() -> None:
    digest = "a" * 64

    def run(mode: str, seconds: float) -> dict[str, object]:
        return {
            "mode": mode,
            "semantic_sha256": digest,
            "repetitions": 20,
            "seconds": seconds,
            "raw_observations": [
                {
                    "semantic_sha256": digest,
                    "timing": {"wall_seconds": seconds},
                }
                for _ in range(20)
            ],
        }

    tokenizer_performance = {
        "benchmark_contract": {
            "schema_version": 1,
            "content_window_boundaries_preserved": True,
            "cache_modes": ["disabled", "cold", "warm"],
            "cache_mode_order": "cyclic_rotation_by_repetition",
        },
        "segmentation_runs": [
            run("disabled", 2.0),
            run("cold", 1.5),
            run("warm", 1.0),
        ],
    }
    row = _performance_row(tokenizer_performance=tokenizer_performance)

    supported = derive_input_claim_verdicts((row,), ({},), complete=True)
    tokenizer_performance["segmentation_runs"].pop()
    incomplete = derive_input_claim_verdicts((row,), ({},), complete=True)

    assert supported["input.tokenizer_latency_improvement"] == "supported"
    assert incomplete["input.tokenizer_latency_improvement"] == "incomplete"


def test_structural_output_infeasibility_remains_unsupported() -> None:
    claims = derive_output_claim_verdicts(
        (),
        (),
        complete=True,
        structurally_unrepresentable=True,
    )

    assert claims["output.semi_autoregressive_density"] == "unsupported"


def test_output_density_requires_all_fidelity_prerequisites() -> None:
    thresholds = (
        {
            "maximum_invalid_events": 0,
            "minimum_valid_non_empty_termination": 1.0,
            "minimum_rollout_event_agreement": 0.5,
            "minimum_native_tokens_per_attempted_macro_step": 1.1,
        },
    )
    supported = {
        "direct_feedback_byte_equality": 1.0,
        "direct_feedback_token_equality": 1.0,
        "invalid_events": 0,
        "valid_non_empty_termination": 1.0,
        "rollout_event_agreement": 1.0,
        "native_tokens_per_attempted_macro_step": 2.0,
    }

    claims = derive_output_claim_verdicts((supported,), thresholds, complete=True)
    invalid = derive_output_claim_verdicts(
        ({**supported, "invalid_events": 1},),
        thresholds,
        complete=True,
    )
    incomplete = derive_output_claim_verdicts(
        ({key: value for key, value in supported.items() if key != "rollout_event_agreement"},),
        thresholds,
        complete=True,
    )

    assert claims["output.semi_autoregressive_density"] == "supported"
    assert invalid["output.semi_autoregressive_density"] == "unsupported"
    assert incomplete["output.semi_autoregressive_density"] == "incomplete"


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_output_control_claim_uses_coverage_precision_and_recall,
            test_output_control_claim_rejects_legacy_correctness_shape,
            test_output_native_head_bypass_is_derived_from_invocations,
            test_input_latency_requires_registered_raw_timing,
            test_input_compression_requires_exact_density,
            test_input_compression_is_independent_from_alignment,
            test_input_behavior_blocks_headline_without_rewriting_compression,
            test_input_latency_requires_complete_rotated_pairs,
            test_input_performance_is_final_only_and_rejects_contamination,
            test_input_performance_requires_raw_matrix_and_denominators,
            test_input_performance_regression_is_unsupported,
            test_tokenizer_latency_requires_complete_exact_cache_matrix,
            test_structural_output_infeasibility_remains_unsupported,
            test_output_density_requires_all_fidelity_prerequisites,
        )
    )
