from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
from huggingface_hub import HfApi

from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.diagnostics.preflight import (
    _cache_check,
    _cold_compile_check,
    _dataset_check,
    _mps_check,
    storage_check,
)


def _campaign_specs() -> tuple[ExperimentSpec, ...]:
    repository = Path(__file__).parents[1]
    paths = sorted((repository / "experiments/campaigns").glob("**/*.toml"))
    return tuple(ExperimentSpec.load(path) for path in paths)


def _registered_check_failures(
    repository: Path,
    device: torch.device,
    specs: tuple[ExperimentSpec, ...],
) -> list[str]:
    failures = []
    for spec in specs:
        for name, check in (
            ("storage", storage_check(repository, spec)),
            ("mps", _mps_check(device, spec)),
            ("inductor_cache", _cache_check(spec)),
        ):
            if not check.passed:
                failures.append(f"{spec.name}:{name}:{check.details}")
    return failures


def _model_access_failures(specs: tuple[ExperimentSpec, ...]) -> list[str]:
    failures = []
    for model_id, revision in sorted({(spec.model.model_id, spec.model.revision) for spec in specs}):
        try:
            resolved = HfApi().model_info(model_id, revision=revision).sha
        except Exception as error:  # noqa: BLE001 - The diagnostic must preserve access failures.
            failures.append(f"{model_id}:model_access:{type(error).__name__}: {error}")
        else:
            if resolved != revision:
                failures.append(f"{model_id}:resolved {resolved}, expected {revision}")
    return failures


@unittest.skipUnless(os.environ.get("RUN_HARDWARE_TESTS") == "1", "set RUN_HARDWARE_TESTS=1")
def test_registered_models_and_hardware_pass_bounded_preflight() -> None:
    repository = Path(__file__).parents[1]
    device = torch.device("mps")
    specs = _campaign_specs()
    if not specs:
        raise AssertionError("no registered campaign specifications were found")
    failures = _registered_check_failures(repository, device, specs)
    dataset = _dataset_check(specs[0])
    if not dataset.passed:
        failures.append(f"dataset_access:{dataset.details}")
    compilation = _cold_compile_check(device)
    if not compilation.passed:
        failures.append(f"cold_inductor_compilation:{compilation.details}")
    failures.extend(_model_access_failures(specs))
    if failures:
        raise AssertionError("\n".join(failures))


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite((unittest.FunctionTestCase(test_registered_models_and_hardware_pass_bounded_preflight),))
