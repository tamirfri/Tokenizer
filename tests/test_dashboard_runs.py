from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from reporting_fixtures import refresh_manifest, write_manifest

from continuous_tokenizer.reporting.discovery import (
    artifact_index,
    discover_artifact_runs,
)


def mark_output_manifest(directory: Path, *, model: str, revision: str) -> None:
    path = directory / "manifest-final.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        mode="output_only",
        codec_direction="output_only",
        model_id=model,
        model_revision=revision,
        native_head_used=False,
        feedback_policy="longest_native_byte_match",
        codec_attention={"type": "none"},
    )
    path.write_text(json.dumps(manifest))


def test_dashboard_artifact_query_selects_directory_name_or_path() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_manifest(root / "first", experiment="first", status="passed")
        write_manifest(root / "second", experiment="second", status="passed")
        artifacts = discover_artifact_runs(root)
        second = next(item for item in artifacts if item.directory.name == "second")

        assert artifact_index(artifacts, "second") == artifacts.index(second)
        assert artifact_index(artifacts, str(second.directory)) == artifacts.index(second)
        assert artifact_index(artifacts, "missing") == 0


@unittest.skipUnless(
    os.environ.get("RUN_STREAMLIT_TESTS") == "1",
    "set RUN_STREAMLIT_TESTS=1 and install the ui dependency group",
)
def test_streamlit_dashboard_renders_completed_artifact() -> None:
    AppTest = import_module("streamlit.testing.v1").AppTest

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "completed"
        write_manifest(run, experiment="completed-run", status="passed")
        (run / "artifact-report.md").write_text("# Verified artifact\n", encoding="utf-8")
        (run / "attention").mkdir()
        (run / "attention/report.md").write_text(
            "# BertViz Attention Diagnostic\n",
            encoding="utf-8",
        )
        tokenizer = {
            "model": {"id": "continuous-tokenizer/synthetic-model"},
            "acceptance": {"overall": True, "compactness": True, "density": True},
            "embedding_fit": {
                "retrieval_queries": 4,
                "retrieval_candidates": 8,
                "retrieval_top1_fraction": 0.75,
                "retrieval_top5_fraction": 1.0,
            },
            "density": {"round_trip": True, "native_tokens_per_continuous_token": 1.25, "alignment": "arbitrary"},
            "native_aligned_segmentation": {"round_trip": True, "native_tokens_per_continuous_token": 1.1, "alignment": "native_token"},
            "raw_byte_fixtures": {"binary": {"bytes": 4, "spans": 2, "round_trip": True}},
            "compactness": {"candidate_reference_state_ratio": 0.25},
            "gates": {
                "density": {
                    "measured": 1.25,
                    "operator": ">=",
                    "threshold": 1.1,
                    "passed": True,
                }
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
        }
        (run / "tokenizer-metrics.json").write_text(
            json.dumps(tokenizer),
            encoding="utf-8",
        )
        llm = {
            "positions": {"native": 100.0, "segmented": 50.0, "native_positions_per_segmented_position": 2.0},
            "performance": {
                "measurement": {
                    "prompt_count": 4,
                    "repetitions": 20,
                    "prompt_set_sha256": "a" * 64,
                },
                "native": {
                    "time_to_first_logit_median_seconds": 2.0,
                    "materialized_cache_bytes": 100,
                    "total_analytical_flops": 1_000,
                    "positions": 1_024,
                    "timing_observations": 80,
                },
                "compatibility": {
                    "time_to_first_logit_median_seconds": 2.1,
                    "materialized_cache_bytes": 100,
                    "total_analytical_flops": 1_200,
                    "positions": 1_024,
                    "timing_observations": 80,
                },
                "segmented": {
                    "time_to_first_logit_median_seconds": 1.0,
                    "materialized_cache_bytes": 60,
                    "total_analytical_flops": 600,
                    "positions": 512,
                    "timing_observations": 80,
                },
            },
            "teacher_forced": {"compatibility": {}, "segmented": {}},
        }
        (run / "llm-metrics.json").write_text(json.dumps(llm), encoding="utf-8")
        (run / "result.json").write_text(
            json.dumps(
                {
                    "mode": "input_only",
                    "evidence_scope": "synthetic",
                    "operational_status": "completed",
                    "scientific_verdict": "supported",
                    "experiment": {"name": "completed-run"},
                    "vocabulary": {
                        "vocabulary_size": 270,
                        "compatibility_tokens": 263,
                        "duplicate_aliases": 1,
                        "ambiguous_byte_sequences": 1,
                        "unavailable_rows": 3,
                        "control_tokens": 2,
                    },
                    "tokenizer": tokenizer,
                    "llm": llm,
                    "verification": {
                        "provided": True,
                        "all_passed": True,
                        "source_commit": "commit",
                        "source_state_sha256": "source",
                        "dependency_lock_sha256": "lock",
                        "checks": {
                            "offline": {
                                "passed": True,
                                "return_code": 0,
                                "seconds": 1.0,
                                "log_sha256": "log",
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "checkpoints").mkdir()
        (run / "checkpoints" / "input.pt").write_bytes(b"checkpoint placeholder")
        progress = run / "checkpoints" / "progress"
        progress.mkdir(parents=True)
        (progress / "small-vocabulary-alignment-005.json").write_text(
            json.dumps(
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
            ),
            encoding="utf-8",
        )
        refresh_manifest(run)

        repository = Path(__file__).parents[1]
        with patch.dict(
            os.environ,
            {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)},
        ):
            app = AppTest.from_file(repository / "src/continuous_tokenizer/app.py").run(timeout=30)

        assert not app.exception
        assert [title.value for title in app.title] == ["Continuous Byte Tokenizer"]
        assert [tab.label for tab in app.tabs] == [
            "Overview",
            "Evidence",
            "Report",
            "Explore",
            "Attention",
        ]
        metrics = {(metric.label, metric.value) for metric in app.metric}
        assert ("Operational status", "COMPLETED") in metrics
        assert ("Scientific verdict", "SUPPORTED") in metrics
        assert ("Evidence scope", "SYNTHETIC") in metrics
        assert ("Mode", "INPUT ONLY") in metrics
        assert ("Verification checks", "PASSED") in metrics
        assert any("Verified artifact" in markdown.value for markdown in app.markdown)
        subheaders = {item.value for item in app.subheader}
        assert "Vocabulary coverage" in subheaders
        assert "Retrieval findings" in subheaders
        assert "Input density findings" in subheaders
        assert "Raw byte fixtures" in subheaders
        assert "Registered input performance measurement" in subheaders
        assert "Verification and preflight" in subheaders


@unittest.skipUnless(
    os.environ.get("RUN_STREAMLIT_TESTS") == "1",
    "set RUN_STREAMLIT_TESTS=1 and install the ui dependency group",
)
def test_streamlit_dashboard_renders_output_artifact() -> None:
    AppTest = import_module("streamlit.testing.v1").AppTest
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "output"
        write_manifest(run, experiment="output-run", status="passed")
        mark_output_manifest(
            run,
            model="continuous-tokenizer/synthetic-model",
            revision="revision",
        )
        result = {
            "mode": "output_only",
            "evidence_scope": "synthetic",
            "operational_status": "completed",
            "experiment": {
                "name": "output-run",
                "mode": "output_only",
                "training": {"profile": "large"},
            },
            "output": {
                "direct_feedback_equality": 1.0,
                "byte_accuracy": 1.0,
                "valid_non_empty_termination": 1.0,
                "rollout_byte_agreement": 1.0,
                "bytes_per_macro_step": 2.0,
                "native_tokens_represented": 4,
                "stop_control": {"policy": "structural_eos_only", "token_ids": [259]},
                "deployment": {
                    "native_head_used": False,
                    "physical_output_head_omission": "not_applicable_tied_input_feedback",
                },
            },
            "gates": {"exact": True},
            "scientific_verdict": "supported",
        }
        (run / "result.json").write_text(json.dumps(result))
        (run / "artifact-report.md").write_text("# Output artifact\n")
        refresh_manifest(run)
        repository = Path(__file__).parents[1]
        with patch.dict(
            os.environ,
            {"CONTINUOUS_TOKENIZER_ARTIFACT_ROOT": str(root)},
        ):
            app = AppTest.from_file(repository / "src/continuous_tokenizer/app.py").run(timeout=30)

        assert not app.exception
        assert [tab.label for tab in app.tabs] == ["Overview", "Evidence", "Report"]
        assert ("Direct feedback", "100.00%") in [(metric.label, metric.value) for metric in app.metric]
        assert ("Output verdict", "SUPPORTED") in [(metric.label, metric.value) for metric in app.metric]
        assert "Stop control" in [item.value for item in app.subheader]
        assert "Deployment" in [item.value for item in app.subheader]
        assert any("Output artifact" in markdown.value for markdown in app.markdown)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_dashboard_artifact_query_selects_directory_name_or_path,
            test_streamlit_dashboard_renders_completed_artifact,
            test_streamlit_dashboard_renders_output_artifact,
        )
    )
