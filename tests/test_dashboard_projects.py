from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from reporting_fixtures import refresh_manifest, write_manifest

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    write_evidence_manifest,
)
from continuous_tokenizer.contracts.claim_derivation import GEMMA_MODEL, QWEN_MODEL
from continuous_tokenizer.contracts.claims import (
    CLAIM_VOCABULARY_SHA256,
)
from continuous_tokenizer.reporting.project import assemble_project_artifact
from continuous_tokenizer.reporting.state_budget_markdown import (
    state_budget_report,
)
from tests.test_artifacts import _performance_ablation
from tests.test_reporting_projects import _replication
from tests.test_state_budget import _result


def seal(
    directory: Path,
    kind: Literal[
        "performance_ablation",
        "project",
        "replication",
        "search",
        "state_budget",
    ],
    mode: str,
    status: str,
    artifact: Path,
) -> None:
    artifacts = {kind: artifact}
    reports = tuple(directory.glob("*-report.md"))
    if reports:
        artifacts["report"] = reports[0]
    write_evidence_manifest(
        directory,
        EvidenceManifest(
            artifact_kind=kind,
            mode=mode,
            status=status,
            identity=EvidenceIdentity(
                source_commit="commit",
                source_dirty=False,
                source_state_sha256="a" * 64,
                dependency_lock_sha256="b" * 64,
                installed_package={
                    "name": "continuous-byte-tokenizer",
                    "version": "0.1.0",
                    "content_sha256": "c" * 64,
                },
                claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
                source_assets={},
                verification={"provided": False},
                model_id="Qwen/Qwen3.5-0.8B",
                model_revision="revision",
            ),
            parents={},
            inputs={},
            artifacts=artifacts,
        ),
    )


@unittest.skipUnless(
    os.environ.get("RUN_STREAMLIT_TESTS") == "1",
    "set RUN_STREAMLIT_TESTS=1 and install the ui dependency group",
)
def test_streamlit_dashboard_renders_project_artifact() -> None:
    AppTest = import_module("streamlit.testing.v1").AppTest

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        project = root / "project"
        assemble_project_artifact(
            (
                _replication(root, model=QWEN_MODEL),
                _replication(root, model=GEMMA_MODEL),
            ),
            project,
        )

        repository = Path(__file__).parents[1]
        with patch.dict(
            os.environ,
            {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)},
        ):
            app = AppTest.from_file(repository / "src/continuous_tokenizer/app.py").run(timeout=30)

        assert not app.exception
        assert [tab.label for tab in app.tabs] == ["Project hypotheses", "Report"]
        metrics = {(metric.label, metric.value) for metric in app.metric}
        assert ("Operational status", "COMPLETED") in metrics
        assert ("Scientific verdict", "SUPPORTED") in metrics
        assert any("Project Evidence" in markdown.value for markdown in app.markdown)
        subheaders = [item.value for item in app.subheader]
        assert "Protocol proofs" in subheaders
        assert "Software validation" in subheaders
        assert "Per-model empirical support" in subheaders
        assert "Cross-model confirmation" in subheaders
        assert "Project claims" in subheaders
        assert "Primary and independent metrics with 95% intervals" in subheaders
        assert "Secondary final-only performance" in subheaders
        assert subheaders.index("Headline verdict") < subheaders.index(
            "Secondary final-only performance",
        )


@unittest.skipUnless(
    os.environ.get("RUN_STREAMLIT_TESTS") == "1",
    "set RUN_STREAMLIT_TESTS=1 and install the ui dependency group",
)
def test_streamlit_dashboard_renders_replication_rows_and_intervals() -> None:
    AppTest = import_module("streamlit.testing.v1").AppTest
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _replication(root, model=QWEN_MODEL, mode="output_only")

        repository = Path(__file__).parents[1]
        with patch.dict(
            os.environ,
            {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)},
        ):
            app = AppTest.from_file(repository / "src/continuous_tokenizer/app.py").run(timeout=30)

        assert not app.exception
        tab_labels = [tab.label for tab in app.tabs]
        assert tab_labels == [
            "Overview",
            "Runs and 95% CIs",
            "Report",
        ], tab_labels
        assert ("Operational status", "COMPLETED") in [(metric.label, metric.value) for metric in app.metric]
        subheaders = [item.value for item in app.subheader]
        assert "Per-seed runs" in subheaders
        assert "Primary and independent metrics with 95% intervals" in subheaders


@unittest.skipUnless(
    os.environ.get("RUN_STREAMLIT_TESTS") == "1",
    "set RUN_STREAMLIT_TESTS=1 and install the ui dependency group",
)
def test_streamlit_dashboard_renders_search_and_run_failures() -> None:
    AppTest = import_module("streamlit.testing.v1").AppTest
    repository = Path(__file__).parents[1]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        search = root / "search"
        search.mkdir()
        (search / "search.json").write_text(
            json.dumps(
                {
                    "mode": "output_only",
                    "evidence_scope": "search",
                    "operational_status": "failed",
                    "scientific_verdict": "not_applicable_search",
                    "name": "output-search",
                    "model_id": "Qwen/Qwen3.5-0.8B",
                    "requested_trials": 2,
                    "finished_trials": 1,
                    "completed_trials": 0,
                    "failed_trials": 1,
                    "selected_trial": None,
                    "trials": [
                        {
                            "number": 0,
                            "state": "FAIL",
                            "parameters": {"learning_rate": 0.001},
                            "metrics": None,
                            "gates": None,
                            "failure": {"type": "RuntimeError", "message": "trial failed"},
                        }
                    ],
                    "failure": {"type": "RuntimeError", "message": "search failed"},
                }
            )
        )
        seal(
            search,
            "search",
            "output_only",
            "failed",
            search / "search.json",
        )
        with patch.dict(
            os.environ,
            {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)},
        ):
            search_app = AppTest.from_file(repository / "src/continuous_tokenizer/app.py").run(timeout=30)

        assert not search_app.exception
        assert [tab.label for tab in search_app.tabs] == [
            "Overview",
            "Trials and failures",
            "Report",
        ]
        assert any("search failed" in item.value for item in search_app.error)
        assert "All search attempts" in [item.value for item in search_app.subheader]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        failed = root / "failed"
        write_manifest(failed, experiment="failed-run", status="failed")
        (failed / "failure.json").write_text(
            json.dumps(
                {
                    "type": "RuntimeError",
                    "message": "run failed",
                }
            )
        )
        refresh_manifest(failed)
        with patch.dict(
            os.environ,
            {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)},
        ):
            run_app = AppTest.from_file(repository / "src/continuous_tokenizer/app.py").run(timeout=30)

        assert not run_app.exception
        assert [tab.label for tab in run_app.tabs] == ["Overview", "Evidence", "Report"]
        assert any("run failed" in item.value for item in run_app.error)
        assert ("Operational status", "FAILED") in [(metric.label, metric.value) for metric in run_app.metric]


@unittest.skipUnless(
    os.environ.get("RUN_STREAMLIT_TESTS") == "1",
    "set RUN_STREAMLIT_TESTS=1 and install the ui dependency group",
)
def test_streamlit_dashboard_renders_state_budget_as_future_prerequisite() -> None:
    AppTest = import_module("streamlit.testing.v1").AppTest
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        budget_dir = root / "joint-budget"
        budget_dir.mkdir()
        result = _result()
        artifact = budget_dir / "joint-state-budget.json"
        artifact.write_text(
            json.dumps(result.to_dict()),
            encoding="utf-8",
        )
        (budget_dir / "joint-state-budget-report.md").write_text(
            state_budget_report(result),
            encoding="utf-8",
        )
        seal(
            budget_dir,
            "state_budget",
            "cross_directional",
            "completed",
            artifact,
        )
        repository = Path(__file__).parents[1]
        with (
            patch.dict(
                os.environ,
                {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)},
            ),
            patch(
                "continuous_tokenizer.reporting.discovery.verify_artifact",
                return_value={"valid": True, "errors": []},
            ),
        ):
            app = AppTest.from_file(
                repository / "src/continuous_tokenizer/app.py",
            ).run(timeout=30)

        assert not app.exception
        assert [tab.label for tab in app.tabs] == [
            "Future prerequisite",
            "Report",
        ]
        metrics = {(metric.label, metric.value) for metric in app.metric}
        assert ("Stored verdict", "SUPPORTED") in metrics
        assert ("Worst-case ratio", "0.550000") in metrics
        assert ("Registered maximum", "1.000000") in metrics
        subheaders = [item.value for item in app.subheader]
        assert "Mandatory non-claims" in subheaders
        assert "Per-model, per-seed tensor arithmetic" in subheaders
        assert any("not memory evidence" in item.value for item in app.warning)
        assert any("Future prerequisite verdict" in item.value for item in app.markdown)


@unittest.skipUnless(
    os.environ.get("RUN_STREAMLIT_TESTS") == "1",
    "set RUN_STREAMLIT_TESTS=1 and install the ui dependency group",
)
def test_streamlit_dashboard_classifies_performance_ablation() -> None:
    AppTest = import_module("streamlit.testing.v1").AppTest
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ablation_dir = root / "ablation"
        ablation_dir.mkdir()
        artifact = ablation_dir / "performance-ablation.json"
        ablation = _performance_ablation()
        ablation["optimized"]["source_state_sha256"] = "a" * 64
        artifact.write_text(
            json.dumps(ablation),
            encoding="utf-8",
        )
        seal(
            ablation_dir,
            "performance_ablation",
            "input_only",
            "completed",
            artifact,
        )
        repository = Path(__file__).parents[1]
        with patch.dict(
            os.environ,
            {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)},
        ):
            app = AppTest.from_file(
                repository / "src/continuous_tokenizer/app.py",
            ).run(timeout=30)

        assert not app.exception
        assert [tab.label for tab in app.tabs] == [
            "Operational ablation",
            "Report",
        ]
        assert "Operational performance ablation" in [item.value for item in app.subheader]
        assert any("operational and secondary evidence only" in item.value for item in app.warning)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_streamlit_dashboard_renders_project_artifact,
            test_streamlit_dashboard_renders_replication_rows_and_intervals,
            test_streamlit_dashboard_renders_search_and_run_failures,
            test_streamlit_dashboard_renders_state_budget_as_future_prerequisite,
            test_streamlit_dashboard_classifies_performance_ablation,
        )
    )
