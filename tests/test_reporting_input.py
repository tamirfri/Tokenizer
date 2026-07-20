from __future__ import annotations

import unittest

from continuous_tokenizer.reporting.artifact_markdown import artifact_report
from continuous_tokenizer.reporting.discovery import (
    acceptance_rows,
)


def test_artifact_report_marks_inapplicable_memory() -> None:
    result = {
        "mode": "input_only",
        "evidence_scope": "final",
        "operational_status": "completed",
        "scientific_verdict": "supported",
        "vocabulary": {
            "vocabulary_size": 270,
            "ordinary_tokens": 264,
            "compatibility_tokens": 263,
            "duplicate_aliases": 1,
            "ambiguous_byte_sequences": 1,
            "control_tokens": 2,
            "unavailable_rows": 3,
            "out_of_table_controls": 1,
            "atomic_bytes": 256,
        },
        "verification": {
            "provided": True,
            "all_passed": True,
            "source_commit": "commit",
            "source_state_sha256": "source",
            "dependency_lock_sha256": "lock",
            "checks": {
                "ruff_lint": {
                    "passed": True,
                    "return_code": 0,
                    "seconds": 0.25,
                    "log_sha256": "log",
                }
            },
        },
        "experiment": {
            "name": "synthetic",
            "device": "cpu",
            "gates": {
                "maximum_normalized_rmse": 0.01,
                "minimum_cosine_p01": 0.999,
                "minimum_cosine_p50": 0.9999,
                "minimum_native_tokens_per_continuous_token": 1.1,
                "maximum_candidate_reference_state_ratio": 0.5,
            },
        },
        "training": {
            "optimizer": {
                "hidden_matrix_parameters": "Muon",
                "output_and_non_matrix_parameters": "AdamW",
                "muon_adjust_lr_fn": "match_rms_adamw",
            }
        },
        "training_progress": [
            {
                "phase": "alignment",
                "epoch": 5,
                "training_loss": 0.25,
                "embedding_metrics": {
                    "normalized_rmse": 0.02,
                    "cosine_similarity_p01": 0.98,
                    "cosine_similarity_p50": 0.99,
                    "exact_fraction": 0.25,
                    "reconstruction_fraction": 0.5,
                },
            }
        ],
        "tokenizer": {
            "model": {"id": "synthetic/model", "revision": "revision"},
            "checkpoint": {"sha256": "checkpoint"},
            "codec": {"query_heads": 4, "key_value_heads": 2, "enable_gqa": True},
            "acceptance": {
                "embedding_fit": True,
                "density": True,
                "compactness": False,
                "overall": True,
            },
            "gates": {
                "maximum_normalized_rmse": {
                    "measured": 0.0,
                    "operator": "<=",
                    "threshold": 0.01,
                    "passed": True,
                },
                "maximum_candidate_reference_state_ratio": {
                    "measured": 12.0,
                    "operator": "<=",
                    "threshold": 0.5,
                    "passed": None,
                },
            },
            "embedding_fit": {
                "normalized_rmse": 0.0,
                "cosine_similarity_p01": 1.0,
                "cosine_similarity_p50": 1.0,
                "reconstruction_fraction": 1.0,
                "exact_fraction": 1.0,
                "retrieval_queries": 16,
                "retrieval_candidates": 263,
                "retrieval_top1_fraction": 0.75,
                "retrieval_top5_fraction": 0.9,
            },
            "density": {
                "native_tokens_per_continuous_token": 1.5,
                "round_trip": True,
                "alignment": "arbitrary",
                "native_tokens": 12,
                "continuous_tokens": 8,
                "bytes_per_continuous_token": 2.0,
            },
            "native_aligned_segmentation": {
                "native_tokens_per_continuous_token": 1.2,
                "round_trip": True,
                "alignment": "native_token",
                "native_tokens": 12,
                "continuous_tokens": 10,
                "bytes_per_continuous_token": 1.6,
            },
            "raw_byte_fixtures": {
                "binary": {
                    "bytes": 4,
                    "spans": 2,
                    "bytes_per_span": 2.0,
                    "round_trip": True,
                    "atomic_spans": 0,
                    "span_lengths": {"2": 2},
                },
                "invalid_utf8": {
                    "bytes": 2,
                    "spans": 2,
                    "bytes_per_span": 1.0,
                    "round_trip": True,
                    "atomic_spans": 2,
                    "span_lengths": {"1": 2},
                },
            },
            "compactness": {
                "candidate_reference_state_ratio": 12.0,
                "reference_state_bytes": 100,
                "candidate_state_bytes": 1200,
                "warm_cache_tensor_bytes": 64,
            },
            "compilation": {"enabled": True, "warmup_seconds": 0.5},
            "segmentation_runs": [
                {
                    "mode": "disabled",
                    "seconds": 0.3,
                    "p95_seconds": 0.33,
                    "repetitions": 5,
                    "cache_hit_rate": 0.0,
                    "cache_tensor_bytes": 0,
                },
                {
                    "mode": "cold",
                    "seconds": 0.2,
                    "p95_seconds": 0.22,
                    "repetitions": 5,
                    "cache_hit_rate": 0.0,
                    "cache_tensor_bytes": 64,
                },
                {
                    "mode": "warm",
                    "seconds": 0.1,
                    "p95_seconds": 0.11,
                    "repetitions": 5,
                    "cache_hit_rate": 1.0,
                    "cache_tensor_bytes": 64,
                },
            ],
        },
        "distillation": {"aligned": {"mean_kl": 0.02, "steps": 4}},
        "ablations": {
            "reconstruction_only": {
                "tokenizer": {"density": {"native_tokens_per_continuous_token": 1.25}},
                "llm": {"teacher_forced": {"segmented": {"mean_kl": 0.03}}},
            }
        },
        "llm": {
            "options": {
                "warmups": 5,
                "repetitions": 20,
                "performance_prompts": 4,
            },
            "teacher_forced": {
                "compatibility": {"mean_kl": 0.001},
                "segmented": {"mean_kl": 0.01},
            },
            "positions": {"native": 100.0, "segmented": 50.0, "native_positions_per_segmented_position": 2.0},
            "performance": {
                "measurement": {
                    "prompt_count": 4,
                    "prompt_set_sha256": "a" * 64,
                    "warmups": 5,
                    "repetitions": 20,
                    "expected_raw_pairs": 80,
                    "recorded_raw_pairs": 80,
                    "prompt_order": "cyclic_rotation_by_repetition",
                    "path_order": "cyclic_rotation_by_pair_execution_order",
                },
                "raw_pairs": [{} for _ in range(80)],
                "native": {
                    "materialized_cache_bytes": 100,
                    "total_analytical_flops": 1000,
                    "time_to_first_logit_median_seconds": 2.0,
                },
                "segmented": {
                    "materialized_cache_bytes": 50,
                    "total_analytical_flops": 400,
                    "time_to_first_logit_median_seconds": 1.0,
                },
            },
            "generation": {"segmented_exact_fraction": 0.75},
        },
    }

    report = artifact_report(result)

    assert "Native positions/segmented position: `2.0000x`" in report

    assert "Stored tokenizer component-gate bundle: **PASSED**" in report
    assert "- Mode: `input_only`" in report
    assert "- Evidence scope: `final`" in report
    assert "- Operational status: `completed`" in report
    assert "- Scientific verdict: `supported`" in report
    assert "Tokenizer encoder attention: `4Q/2KV GQA`" in report
    assert ("Tokenizer optimizer: `hidden matrices=Muon; output and non-matrices=AdamW; Muon LR adjustment=match_rms_adamw`") in report
    assert "`input.codec_reference_compactness`" in report
    assert "Verdict: **INCOMPLETE**" in report
    assert "Denominator / sample context" in report
    assert "## Registered Gates" in report
    assert all(row["passed"] is not False for row in acceptance_rows(result))
    assert "Ambiguous byte payloads: `1`" in report
    assert "Unavailable rows: `3`" in report
    assert "## Lifecycle Verification" in report
    assert "| ruff_lint | True | 0 | 0.250 | `log` |" in report
    assert "Retrieval top-1 fraction: `0.750000`" in report
    assert "| binary | 4 | 2 | 2.000000 | 0 | True |" in report
    assert "| invalid_utf8 | 2 | 2 | 1.000000 | 2 | True |" in report
    assert "| arbitrary | 12 | 8 | 2.000000 | 1.500000 | True |" in report
    assert "| native_token | 12 | 10 | 1.600000 | 1.200000 | True |" in report
    assert "`aligned.mean_kl` | `0.02`" in report
    assert "`reconstruction_only.llm.teacher_forced.segmented.mean_kl` | `0.03`" in report
    assert "Registered-prompt-set model-cache reduction: `50.00%`" in report
    assert "Registered-prompt-set analytical prefill FLOP reduction: `60.00%`" in report
    assert "Registered performance prompt set: `4` prompts" in report
    assert "time-to-first-logit speedup: `2.0000x`" in report
    assert "Registered-prompt-set model-cache reduction: **OBSERVED**" in report
    assert "Warm encoding-cache median speedup: `3.0000x`" in report
    assert "| warm | 0.100000 | 0.110000 | 5 | 100.00% | 64 |" in report
    assert "## Training Convergence" in report
    assert "Convergence rows, checkpoint selection" in report
    assert ("| alignment | 5 | 0.250000 | 0.020000 | 0.980000 | 0.990000 | 25.00% | 50.00% | n/a | n/a |") in report
    assert "Registered-prompt-set analytical FLOP reduction: **OBSERVED**" in report
    assert "End-to-end time-to-first-logit improvement: **OBSERVED**" in report
    assert ("Frozen-model behavioral similarity: **REGISTERED GATES REPORTED IN THE HEADLINE**") in report

    result["llm"]["performance"]["segmented"].update(
        materialized_cache_bytes=120,
        total_analytical_flops=1200,
        time_to_first_logit_median_seconds=3.0,
    )
    failed = artifact_report(result)
    assert "Registered-prompt-set model-cache reduction: **NOT OBSERVED**" in failed
    assert "Registered-prompt-set analytical FLOP reduction: **NOT OBSERVED**" in failed
    assert "End-to-end time-to-first-logit improvement: **NOT OBSERVED**" in failed

    result["llm"]["options"] = {
        "warmups": 0,
        "repetitions": 1,
        "performance_prompts": 4,
    }
    pilot = artifact_report(result)
    assert "End-to-end time-to-first-logit improvement: **INSUFFICIENT MEASUREMENT**" in pilot


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (test_artifact_report_marks_inapplicable_memory,))
