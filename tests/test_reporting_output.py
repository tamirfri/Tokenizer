from __future__ import annotations

import unittest

from continuous_tokenizer.reporting.artifact_markdown import artifact_report


def test_output_report_publishes_policy_thresholds_and_applicability() -> None:
    result = {
        "mode": "output_only",
        "evidence_scope": "final",
        "operational_status": "completed",
        "scientific_verdict": "unsupported",
        "vocabulary": {
            "vocabulary_size": 260,
            "ordinary_tokens": 256,
            "compatibility_tokens": 255,
            "duplicate_aliases": 1,
            "ambiguous_byte_sequences": 1,
            "control_tokens": 1,
            "unavailable_rows": 3,
            "out_of_table_controls": 1,
            "atomic_bytes": 256,
        },
        "verification": {"provided": False},
        "experiment": {
            "name": "output",
            "training": {"profile": "large"},
            "gates": {
                "minimum_direct_feedback_equality": 0.99,
                "minimum_native_tokens_per_attempted_macro_step": 1.1,
                "minimum_rollout_event_agreement": 0.5,
                "maximum_candidate_reference_state_ratio": 0.5,
            },
        },
        "output": {
            "direct_feedback_equality": 0.95,
            "native_tokens_per_attempted_macro_step": 1.2,
            "rollout_event_agreement": 0.75,
            "rollout_byte_agreement": 0.99,
            "bytes_per_macro_step": 1.5,
            "candidate_reference_state_ratio": 0.25,
            "stop_control": {"policy": "structural_eos_only", "token_ids": [259]},
            "deployment": None,
        },
        "gates": {
            "direct_feedback": False,
            "native_tokens_per_attempted_macro_step": True,
            "rollout_event_agreement": True,
            "candidate_reference_state_ratio": True,
        },
    }

    report = artifact_report(result)

    assert "Stop-control policy: `structural_eos_only`" in report
    assert "| Direct Feedback | `0.95` | `>= 0.99` | FAIL |" in report
    assert "measured by independent deployment evidence" in report
    assert "Unsupported: Direct Feedback" in report
    assert "Ambiguous byte payloads: `1`" in report
    assert "Preflight verification: `not provided`" in report


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (test_output_report_publishes_policy_thresholds_and_applicability,))
