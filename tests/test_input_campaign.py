from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch
from input_training_fixtures import (
    TEST_PROFILE,
    limited_torch_threads,
    synthetic_assets,
)

import continuous_tokenizer.campaigns.input as runner_module
import continuous_tokenizer.cli as cli_module
import continuous_tokenizer.input.benchmark.run as benchmark_module
import continuous_tokenizer.input.benchmark.tokenizer as tokenizer_benchmark_module
import continuous_tokenizer.input.training.reconstruction as reconstruction_module
import continuous_tokenizer.input.training.run as training_module
import continuous_tokenizer.input.training.runtime as training_runtime_module
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.campaigns.dispatch import create_experiment_runner
from continuous_tokenizer.codec.checkpoints import load_checkpoint
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.input.training.run import TrainingOptions, train_experiment


def test_training_pipeline_writes_reloadable_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            limited_torch_threads(),
            patch.object(training_module, "load_corpus_documents", return_value=[b"abc"]),
            patch.object(
                reconstruction_module.ReconstructionFitter,
                "fit",
                side_effect=AssertionError("zero reconstruction epochs must skip the stage"),
            ),
        ):
            options = TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=256,
                vocabulary_epochs=1,
                reconstruction_epochs=0,
                reconstruction_samples=0,
                validation_bytes=3,
                patience=1,
            )

            result = train_experiment(
                synthetic_assets(root),
                options,
                device=torch.device("cpu"),
            )
        checkpoint = Path(result.checkpoint)
        loaded = load_checkpoint(checkpoint)
        codec = loaded.codec
        metadata = loaded.metadata

        assert checkpoint.is_file()
        assert (options.output_dir / "run-manifest.json").is_file()
        assert metadata["model_id"] == "synthetic/model"
        assert metadata["checkpoint_stage"] == "vocabulary"
        assert metadata["codec_attention"] == {
            "query_heads": 4,
            "key_value_heads": 2,
            "enable_gqa": True,
        }
        assert metadata["optimizer"] == {
            "hidden_matrix_parameters": "Muon",
            "muon_adjust_lr_fn": "match_rms_adamw",
            "muon_ns_steps": 5,
            "output_and_non_matrix_parameters": "AdamW",
        }
        assert codec.config.embedding_dim == 8
        assert result.round_trip


def test_failed_embedding_profile_still_measures_density() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = synthetic_assets(root)
        assets = replace(
            assets,
            input_embeddings=torch.cat((assets.input_embeddings, torch.full((1, 8), 100.0))),
            vocabulary=ByteVocabulary(
                token_bytes=(*assets.vocabulary.token_bytes, b"ab"),
                ordinary_ids=tuple(range(257)),
                control_ids=(),
                byte_token_ids=assets.vocabulary.byte_token_ids,
                max_token_bytes=2,
            ),
        )
        options = TrainingOptions(
            output_dir=root / "checkpoints",
            profile=TEST_PROFILE,
            vocabulary_epochs=0,
            reconstruction_epochs=0,
            reconstruction_samples=0,
            validation_bytes=3,
        )

        with (
            limited_torch_threads(),
            patch.object(training_module, "load_corpus_documents", return_value=[b"abc"]),
        ):
            result = train_experiment(assets, options, device=torch.device("cpu"))

        progress = json.loads((options.output_dir / "progress/small-density.json").read_text(encoding="utf-8"))
        assert not result.compatibility_passed
        assert result.native_tokens_per_continuous_token > 0.0
        assert result.round_trip
        assert "skipped" not in progress
        assert Path(result.checkpoint).is_file()


def test_reconstruction_carries_selected_density_without_final_rescan() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        options = TrainingOptions(
            output_dir=root / "checkpoints",
            profile=TEST_PROFILE,
            batch_size=256,
            vocabulary_epochs=1,
            reconstruction_epochs=1,
            reconstruction_samples=1,
            validation_bytes=3,
            patience=10,
            evaluation_interval=1,
        )
        original_density = training_runtime_module.TrainingRuntime.density_metrics
        with (
            limited_torch_threads(),
            patch.object(training_module, "load_corpus_documents", return_value=[b"abc"]),
            patch.object(
                training_runtime_module.TrainingRuntime,
                "density_metrics",
                autospec=True,
                side_effect=original_density,
            ) as density,
        ):
            train_experiment(
                synthetic_assets(root),
                options,
                device=torch.device("cpu"),
            )

        progress = json.loads((options.output_dir / "progress/small-density.json").read_text(encoding="utf-8"))
        assert density.call_count == 2
        assert progress["reused_selected_reconstruction_measurement"] is True
        assert progress["seconds"] == 0.0
        assert progress["identity"]["implementation"] == "exhaustive-greedy-density-v1"


def test_runner_measures_failed_tokenizer_hypotheses() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repository = Path(__file__).parents[1]
        spec = ExperimentSpec.load(repository / "experiments/synthetic/input-smoke.toml")
        spec = replace(
            spec,
            stages=("vocabulary",),
            training=replace(
                spec.training,
                vocabulary_epochs=0,
                reconstruction_epochs=0,
                reconstruction_samples=0,
            ),
            gates=replace(spec.gates, minimum_cosine_p01=2.0),
        )
        tokenizer_metrics: dict[str, object] = {
            "acceptance": {
                "overall": False,
                "density": False,
                "embedding_fit": False,
                "compactness": True,
            },
        }

        def measure_tokenizer(
            assets: object,
            model: object,
            request: object,
        ) -> tuple[dict[str, object], None]:
            del assets, model
            assert isinstance(request, runner_module._MeasurementRequest)
            output_dir = request.output_dir
            (output_dir / "tokenizer-metrics.json").write_text(
                json.dumps(tokenizer_metrics),
                encoding="utf-8",
            )
            (output_dir / "tokenizer-report.md").write_text(
                "# tokenizer\n",
                encoding="utf-8",
            )
            return tokenizer_metrics, None

        with (
            limited_torch_threads(),
            patch.object(
                runner_module.InputExperimentRunner,
                "_measure",
                side_effect=measure_tokenizer,
            ) as measure,
            patch(
                "continuous_tokenizer.campaigns.input.artifact_report",
                return_value="# report\n",
            ),
        ):
            result = create_experiment_runner(spec, root / "run", repository).run()

        measure.assert_called_once()
        assert not result["training"]["passed"]
        assert result["tokenizer"] == tokenizer_metrics
        assert result["vocabulary"]["atomic_bytes"] == 256
        assert result["verification"] == {"provided": False}
        assert (root / "run/artifact-report.md").is_file()


@unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS") == "1", "set RUN_SLOW_TESTS=1")
def test_synthetic_spec_runs_complete_offline_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository = Path(__file__).parents[1]
        output = Path(directory) / "run"
        stdout = StringIO()

        with (
            patch.object(
                tokenizer_benchmark_module,
                "_raw_byte_fixtures",
                return_value={"binary": b"abc"},
            ),
            patch.object(training_module, "default_device", return_value=torch.device("cpu")),
            patch.object(benchmark_module, "default_device", return_value=torch.device("cpu")),
            redirect_stdout(stdout),
        ):
            cli_module.main(
                [
                    "run",
                    str(repository / "experiments/synthetic/input-smoke.toml"),
                    "--output-dir",
                    str(output),
                ]
            )

        assert stdout.getvalue()
        assert (output / "manifest-final.json").is_file()
        assert (output / "artifact-report.md").is_file()
        assert (output / "tokenizer-metrics.json").is_file()
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert result["tokenizer"]["acceptance"]["overall"]
        assert result["mode"] == "input_only"
        assert result["evidence_scope"] == "synthetic"
        assert result["operational_status"] == "completed"
        assert result["scientific_verdict"] == "supported"
        start = json.loads((output / "manifest-start.json").read_text(encoding="utf-8"))
        final = json.loads((output / "manifest-final.json").read_text(encoding="utf-8"))
        assert final["status"] == "passed"
        assert start["artifact_hashes"] == {}
        assert final["codec_attention"]["enable_gqa"] is True
        assert set(final["artifact_hashes"]) == {name for name, relative in final["artifacts"].items() if relative is not None and (output / relative).exists()}
        assert start["source_state_sha256"] == final["source_state_sha256"]


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_training_pipeline_writes_reloadable_checkpoint,
            test_failed_embedding_profile_still_measures_density,
            test_reconstruction_carries_selected_density_without_final_rescan,
            test_runner_measures_failed_tokenizer_hypotheses,
            test_synthetic_spec_runs_complete_offline_artifact,
        )
    )
