from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from continuous_tokenizer.campaigns.lifecycle import ExperimentLifecycle
from continuous_tokenizer.commands.verify import verify
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.profiles import profile_named
from continuous_tokenizer.diagnostics.preflight import PreflightCheck, run_preflight
from continuous_tokenizer.diagnostics.verification import CheckResult, run_verification


def test_verification_command_fails_when_any_check_fails() -> None:
    arguments = argparse.Namespace(
        output_dir=Path("verification"),
        slow=False,
        streamlit=False,
        model_tokenizers=False,
        complete=False,
    )
    with (
        patch("continuous_tokenizer.commands.verify.find_project_root", return_value=Path()),
        patch(
            "continuous_tokenizer.commands.verify.run_verification",
            return_value={"all_passed": False},
        ),
        unittest.TestCase().assertRaisesRegex(RuntimeError, "preflight verification failed"),
    ):
        verify(arguments)


def test_complete_verification_targets_reorganized_gated_tests() -> None:
    result = CheckResult((), True, 0, 0.0, "log", "hash")
    with (
        tempfile.TemporaryDirectory() as directory,
        patch(
            "continuous_tokenizer.diagnostics.verification._run_check",
            return_value=result,
        ) as run_check,
        patch(
            "continuous_tokenizer.diagnostics.verification.source_state",
            return_value=("commit", True, "source"),
        ),
        patch(
            "continuous_tokenizer.diagnostics.verification.sha256_file",
            return_value="lock",
        ),
    ):
        root = Path(directory)
        artifact = run_verification(
            root,
            root / "verification",
            slow=True,
            streamlit=True,
            model_tokenizers=True,
            complete=True,
        )

    assert artifact["kind"] == "complete_verification"
    assert artifact["source_commit"] == "commit"
    assert artifact["source_dirty"] is True
    assert artifact["source_state_sha256"] == "source"
    assert artifact["dependency_lock_sha256"] == "lock"
    commands = {item.args[2]: item.args[3] for item in run_check.call_args_list}
    assert commands["streamlit"][-2:] == ("test_dashboard_*.py", "-v")
    assert commands["model_tokenizers"][-2:] == ("test_input_vocabulary.py", "-v")
    assert commands["model_access"][-2:] == ("test_real_model_diagnostics.py", "-v")
    assert commands["representative_mps"][-2:] == (
        "tests.test_codec_compilation.test_mps_gqa_forward_and_backward_compile_as_full_graphs",
        "tests.test_codec_compilation.test_mps_qwen_sized_vocabulary_evaluation_uses_static_rows",
    )
    assert commands["hardware_model_preflight"][-2:] == (
        "test_hardware_preflight.py",
        "-v",
    )
    assert {
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
    }.issubset(commands)
    assert not set(commands["slow_synthetic"][4:]) & set(
        commands["representative_mps"][4:],
    )


def test_non_complete_verification_runs_only_selected_inventory() -> None:
    result = CheckResult((), True, 0, 0.0, "log", "hash")
    cases = (
        ("default", False, False, False, ()),
        (
            "diagnostic",
            True,
            True,
            True,
            ("slow_synthetic", "streamlit", "model_tokenizers"),
        ),
    )
    with (
        tempfile.TemporaryDirectory() as directory,
        patch(
            "continuous_tokenizer.diagnostics.verification._run_check",
            return_value=result,
        ) as run_check,
        patch(
            "continuous_tokenizer.diagnostics.verification.source_state",
            return_value=("commit", True, "source"),
        ),
        patch(
            "continuous_tokenizer.diagnostics.verification.sha256_file",
            return_value="lock",
        ),
    ):
        root = Path(directory)
        for name, slow, streamlit, model_tokenizers, selected in cases:
            with unittest.TestCase().subTest(name=name):
                run_check.reset_mock()
                artifact = run_verification(
                    root,
                    root / name,
                    slow=slow,
                    streamlit=streamlit,
                    model_tokenizers=model_tokenizers,
                )
                assert artifact["kind"] == "verification"
                assert [call.args[2] for call in run_check.call_args_list] == [
                    "ruff_format",
                    "ruff_lint",
                    "types",
                    "fast_tests",
                    *selected,
                ]


def test_preflight_reuses_representative_mps_verification() -> None:
    repository = Path(__file__).parents[1]
    spec = ExperimentSpec.load(repository / "experiments/synthetic/input-smoke.toml")
    passed = PreflightCheck(True, {})
    with (
        tempfile.TemporaryDirectory() as directory,
        patch(
            "continuous_tokenizer.diagnostics.preflight.storage_check",
            return_value=passed,
        ),
        patch(
            "continuous_tokenizer.diagnostics.preflight._mps_check",
            return_value=passed,
        ),
        patch(
            "continuous_tokenizer.diagnostics.preflight._cache_check",
            return_value=passed,
        ),
        patch(
            "continuous_tokenizer.diagnostics.preflight._dataset_check",
            return_value=passed,
        ),
        patch(
            "continuous_tokenizer.diagnostics.preflight._cold_compile_check",
            side_effect=AssertionError("cold compile must be reused"),
        ),
    ):
        artifact, model = run_preflight(
            spec,
            Path(directory) / "preflight.json",
            device=torch.device("mps"),
            load_full_model=None,
            identity={},
            representative_mps_verified=True,
        )

    assert model is None
    assert artifact["checks"]["cold_inductor_compilation"]["work_avoided"] == "duplicate_cold_inductor_probe"


def test_final_large_runs_require_source_bound_complete_verification() -> None:
    repository = Path(__file__).parents[1]
    candidate = ExperimentSpec.load(repository / "experiments/campaigns/input/qwen35-0.8b/seed-17.toml")
    lifecycle = object.__new__(ExperimentLifecycle)
    lifecycle.spec = replace(candidate, evidence_scope="final")
    lifecycle.profile = profile_named(candidate.training.profile)
    lifecycle.source_state = ("commit", True, "source")
    lifecycle.dependency_lock_sha256 = "lock"

    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "require a verification artifact",
    ):
        lifecycle._load_verification(None)

    checks = {
        name: {"passed": True}
        for name in (
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
        )
    }
    artifact = {
        "kind": "complete_verification",
        "source_commit": "wrong",
        "source_state_sha256": "source",
        "dependency_lock_sha256": "lock",
        "all_passed": True,
        "checks": checks,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "verification.json"
        path.write_text(json.dumps(artifact))
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "source commit",
        ):
            lifecycle._load_verification(path)
        artifact["source_commit"] = "commit"
        path.write_text(json.dumps(artifact))
        assert lifecycle._load_verification(path)["provided"]
        artifact["kind"] = "verification"
        path.write_text(json.dumps(artifact))
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "canonical complete verification",
        ):
            lifecycle._load_verification(path)

    diagnostic = ExperimentSpec.load(repository / "experiments/diagnostics/qwen35-0.8b-input.toml")
    lifecycle.spec = diagnostic
    lifecycle.profile = profile_named(diagnostic.training.profile)
    assert lifecycle._load_verification(None) == {"provided": False}


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_verification_command_fails_when_any_check_fails,
            test_complete_verification_targets_reorganized_gated_tests,
            test_non_complete_verification_runs_only_selected_inventory,
            test_preflight_reuses_representative_mps_verification,
            test_final_large_runs_require_source_bound_complete_verification,
        )
    )
