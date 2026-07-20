from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import unittest
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Final, cast, final

from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.artifacts.source import source_state
from continuous_tokenizer.artifacts.store import RunDirectory
from continuous_tokenizer.runtime.progress import log_event

BASE_CHECKS: Final = {
    "ruff_format": ("ruff", "format", "--check", "."),
    "ruff_lint": ("ruff", "check", "."),
    "types": ("ty", "check", "src", "tests"),
    "fast_tests": (
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ),
}

SLOW_CHECK: Final = (
    sys.executable,
    "-m",
    "continuous_tokenizer.diagnostics.verification",
    "tests.test_codec_compilation.test_neural_paths_are_full_graph_compilable",
    "tests.test_input_campaign.test_synthetic_spec_runs_complete_offline_artifact",
    "tests.test_output_campaign.OutputModeTests.test_synthetic_output_campaign_proves_end_to_end_path",
)

COMPLETE_CHECKS: Final = {
    "slow_synthetic": SLOW_CHECK,
    "streamlit": (
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_dashboard_*.py",
        "-v",
    ),
    "model_tokenizers": (
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_input_vocabulary.py",
        "-v",
    ),
    "model_access": (
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_real_model_diagnostics.py",
        "-v",
    ),
    "representative_mps": (
        sys.executable,
        "-m",
        "continuous_tokenizer.diagnostics.verification",
        "tests.test_codec_compilation.test_mps_gqa_forward_and_backward_compile_as_full_graphs",
        "tests.test_codec_compilation.test_mps_qwen_sized_vocabulary_evaluation_uses_static_rows",
    ),
    "hardware_model_preflight": (
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_hardware_preflight.py",
        "-v",
    ),
}


@final
@dataclass(frozen=True, slots=True)
class CheckResult:
    command: tuple[str, ...]
    passed: bool
    return_code: int
    seconds: float
    log: str
    log_sha256: str


def run_verification(  # noqa: PLR0913 - Selection flags mirror the CLI contract.
    project_root: Path,
    output_dir: Path,
    *,
    slow: bool,
    streamlit: bool,
    model_tokenizers: bool,
    complete: bool = False,
) -> dict[str, object]:
    output = RunDirectory(output_dir)
    checks = dict(BASE_CHECKS)
    if complete:
        checks.update(COMPLETE_CHECKS)
    else:
        for enabled, name in (
            (slow, "slow_synthetic"),
            (streamlit, "streamlit"),
            (model_tokenizers, "model_tokenizers"),
        ):
            if enabled:
                checks[name] = COMPLETE_CHECKS[name]

    results = {name: _run_check(project_root, output, name, command) for name, command in checks.items()}
    commit, dirty, state_sha256 = source_state(project_root)
    artifact = {
        "kind": "complete_verification" if complete else "verification",
        "source_commit": commit,
        "source_dirty": dirty,
        "source_state_sha256": state_sha256,
        "dependency_lock_sha256": sha256_file(project_root / "uv.lock"),
        "all_passed": all(result.passed for result in results.values()),
        "checks": {name: asdict(result) for name, result in results.items()},
    }
    output.write_json("verification.json", artifact)
    return artifact


def _run_check(
    project_root: Path,
    output: RunDirectory,
    name: str,
    command: tuple[str, ...],
) -> CheckResult:
    environment = os.environ.copy()
    if name in {"slow_synthetic", "representative_mps"}:
        environment["RUN_SLOW_TESTS"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(project_root / "tests"), environment.get("PYTHONPATH")),
            ),
        )
    elif name == "streamlit":
        environment["RUN_STREAMLIT_TESTS"] = "1"
    elif name == "model_tokenizers":
        environment["RUN_MODEL_TESTS"] = "1"
    elif name == "model_access":
        environment["RUN_REAL_MODEL_TESTS"] = "1"
    elif name == "hardware_model_preflight":
        environment["RUN_HARDWARE_TESTS"] = "1"
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    log_event("verification_check_started", check=name, command=list(command))
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = perf_counter() - started
    log = completed.stdout + completed.stderr
    log_path = output.write_text(f"logs/{name}.log", log)
    result = CheckResult(
        command=command,
        passed=completed.returncode == 0,
        return_code=completed.returncode,
        seconds=seconds,
        log=str(log_path.relative_to(output.root)),
        log_sha256=hashlib.sha256(log.encode()).hexdigest(),
    )
    log_event(
        "verification_check_completed",
        check=name,
        passed=result.passed,
        return_code=result.return_code,
        elapsed_seconds=round(result.seconds, 1),
        log=result.log,
    )
    return result


def _selected_test(specification: str) -> unittest.TestSuite:
    parts = specification.split(".")
    for index in range(len(parts), 0, -1):
        try:
            value: object = importlib.import_module(".".join(parts[:index]))
        except ModuleNotFoundError:
            continue
        parent = value
        for part in parts[index:]:
            parent = value
            value = getattr(value, part)
        if isinstance(parent, type) and issubclass(parent, unittest.TestCase):
            return unittest.defaultTestLoader.loadTestsFromName(specification)
        if callable(value):
            test = cast(Callable[[], object], value)
            return unittest.TestSuite((unittest.FunctionTestCase(test),))
        break
    raise ValueError(f"invalid selected test: {specification}")


def _run_selected_tests(specifications: list[str]) -> int:
    suite = unittest.TestSuite(_selected_test(specification) for specification in specifications)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_run_selected_tests(sys.argv[1:]))
