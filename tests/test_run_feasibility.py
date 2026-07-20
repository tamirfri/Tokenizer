from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.data.corpus import stream_corpus_documents
from continuous_tokenizer.diagnostics.preflight import storage_check
from continuous_tokenizer.runtime.resume import ResumeManager


def test_operational_snapshot_is_replaced_and_finalized_immutably() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "run"
        manager = ResumeManager(
            root,
            "experiment",
            "commit",
            "source",
            "lock",
            True,
        )
        manager.save("phase", 1, {"completed": False, "value": torch.tensor(1)})
        first = manager.recovery_root / "phase.pt"
        manager.save("phase", 2, {"completed": False, "value": torch.tensor(2)})

        assert tuple(manager.recovery_root.iterdir()) == (first,)
        operational = manager.latest("phase")
        assert operational is not None
        assert operational["value"].item() == 2

        manager.save("phase", 3, {"completed": True, "value": torch.tensor(3)})
        assert not first.exists()
        assert (root / "phase-final/phase.pt").is_file()
        assert (root / "phase-final/phase.json").is_file()
        final = manager.latest("phase")
        assert final is not None
        assert final["value"].item() == 3
        with unittest.TestCase().assertRaisesRegex(
            FileExistsError,
            "phase-final evidence",
        ):
            manager.save(
                "phase",
                4,
                {"completed": True, "value": torch.tensor(4)},
            )


def test_resume_snapshot_invalidates_for_every_bound_identity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "run"
        manager = ResumeManager(
            root,
            "experiment",
            "commit",
            "source",
            "lock",
            True,
        )
        manager.save("phase", 1, {"completed": False, "value": torch.tensor(1)})
        identities = (
            ("other-experiment", "commit", "source", "lock"),
            ("experiment", "other-commit", "source", "lock"),
            ("experiment", "commit", "other-source", "lock"),
            ("experiment", "commit", "source", "other-lock"),
        )
        for identity in identities:
            with (
                unittest.TestCase().subTest(identity=identity),
                unittest.TestCase().assertRaisesRegex(ValueError, "source contract"),
            ):
                ResumeManager(root, *identity, True).latest("phase")


def test_wikitext_streaming_stops_at_declared_row_bound() -> None:
    rows = [{"text": f"row-{index}"} for index in range(10)]
    with mock.patch(
        "continuous_tokenizer.data.corpus.load_dataset",
        return_value=iter(rows),
    ) as load_dataset:
        documents = list(
            stream_corpus_documents(
                "train",
                dataset_id="dataset",
                config="config",
                revision="revision",
                max_rows=3,
            )
        )

    assert documents == [b"row-0", b"row-1", b"row-2"]
    assert load_dataset.call_args.kwargs["streaming"] is True
    assert load_dataset.call_args.kwargs["revision"] == "revision"


def test_storage_projection_uses_next_run_and_reserve_only() -> None:
    repository = Path(__file__).parents[1]
    spec = ExperimentSpec.load(repository / "experiments/synthetic/input-smoke.toml")
    policy = replace(
        spec.runtime,
        projected_run_bytes=60,
        inductor_cache_estimate_bytes=20,
        storage_reserve_bytes=30,
    )
    with mock.patch(
        "continuous_tokenizer.diagnostics.preflight.shutil.disk_usage",
        return_value=SimpleNamespace(total=200, used=100, free=100),
    ):
        check = storage_check(repository, replace(spec, runtime=policy))

    assert check.passed
    assert check.details["required_free_bytes"] == 90
    assert check.details["projected_free_after_run_and_cache_bytes"] == 40
    assert check.details["existing_artifact_bytes"] > 0
    assert check.details["inductor_cache_bytes"] >= 0
    with mock.patch(
        "continuous_tokenizer.diagnostics.preflight.shutil.disk_usage",
        return_value=SimpleNamespace(total=200, used=111, free=89),
    ):
        insufficient = storage_check(repository, replace(spec, runtime=policy))
    assert not insufficient.passed


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_operational_snapshot_is_replaced_and_finalized_immutably,
            test_resume_snapshot_invalidates_for_every_bound_identity,
            test_wikitext_streaming_stops_at_declared_row_bound,
            test_storage_projection_uses_next_run_and_reserve_only,
        )
    )
