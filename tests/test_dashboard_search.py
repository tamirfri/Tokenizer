from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    write_evidence_manifest,
)
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256


@unittest.skipUnless(
    os.environ.get("RUN_STREAMLIT_TESTS") == "1",
    "set RUN_STREAMLIT_TESTS=1 and install the ui and search dependency groups",
)
def test_streamlit_dashboard_renders_search_artifact() -> None:
    AppTest = import_module("streamlit.testing.v1").AppTest
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = root / "search"
        result.mkdir()
        search = {
            "mode": "input_only",
            "evidence_scope": "search",
            "operational_status": "completed",
            "scientific_verdict": "not_applicable_search",
            "name": "alignment-search",
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "revision",
            "profile": "large",
            "requested_trials": 1,
            "finished_trials": 1,
            "completed_trials": 1,
            "selected_trial": 0,
            "selected_alignment_passed": True,
            "selected_compactness_passed": True,
            "trials": [
                {
                    "number": 0,
                    "state": "complete",
                    "parameters": {
                        "learning_rate": 0.0003,
                        "weight_decay": 0.0,
                        "batch_size": 32,
                    },
                    "metrics": {
                        "normalized_rmse": 0.01,
                        "cosine_p01": 0.999,
                        "cosine_p50": 0.9999,
                    },
                }
            ],
        }
        (result / "search.json").write_text(json.dumps(search), encoding="utf-8")
        (result / "search-report.md").write_text("# Search pilot\n", encoding="utf-8")
        write_evidence_manifest(
            result,
            EvidenceManifest(
                artifact_kind="search",
                mode="input_only",
                status="completed",
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
                artifacts={
                    "search": result / "search.json",
                    "report": result / "search-report.md",
                },
            ),
        )
        repository = Path(__file__).parents[1]
        with patch.dict(os.environ, {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)}):
            app = AppTest.from_file(repository / "src/continuous_tokenizer/app.py").run(timeout=30)

        assert not app.exception
        assert [(metric.label, metric.value) for metric in app.metric[-4:]] == [
            ("Finished trials", "1 / 1"),
            ("Selected trial", "0"),
            ("Selected NRMSE", "0.010000"),
            ("Alignment gate", "PASS"),
        ]
        assert any("Search pilot" in markdown.value for markdown in app.markdown)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (test_streamlit_dashboard_renders_search_artifact,))
