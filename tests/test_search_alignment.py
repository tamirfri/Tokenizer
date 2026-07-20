from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from search_fixtures import _optional_search_module

from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.input.training.vocabulary import AlignmentResult


def _search_module() -> ModuleType:
    return _optional_search_module("continuous_tokenizer.search.alignment")


def _experiment() -> str:
    return """
name = "synthetic-search-base"
mode = "input_only"
evidence_scope = "search"
device = "cpu"
seed = 17
stages = ["vocabulary"]

[model]
id = "synthetic/model"
revision = "revision"
evaluation = "full"

[dataset]
id = "synthetic/data"
config = "default"
revision = "data-revision"

[runtime]
corpus_max_rows = 16
cache_chunk_rows = 8
snapshot_interval = 1
projected_run_bytes = 1024
storage_reserve_bytes = 1024
inductor_cache_estimate_bytes = 1024
minimum_mps_memory_bytes = 1024

[training]
profile = "small"
batch_size = 32
learning_rate = 0.0003
weight_decay = 0.0
vocabulary_epochs = 2
reconstruction_epochs = 0
reconstruction_samples = 0
reconstruction_vocabulary_fraction = 0.75
validation_bytes = 16
patience = 1
evaluation_interval = 1
distillation_epochs = 0
distillation_windows = 0
distillation_prompt_tokens = 2
distillation_continuation_tokens = 1

[evaluation]
batch_size = 8
samples = 2
prompt_tokens = 2
continuation_tokens = 1
generation_samples = 0
max_new_tokens = 1
warmups = 0
repetitions = 1
performance_prompts = 1
tokenizer_repetitions = 1
retrieval_queries = 16
max_test_bytes = 16

[gates]
maximum_normalized_rmse = 0.01
minimum_cosine_p01 = 0.999
minimum_cosine_p50 = 0.9999
minimum_native_tokens_per_continuous_token = 1.1
maximum_candidate_reference_state_ratio = 0.5
"""


def _search() -> str:
    return """
name = "synthetic-search"
experiment = "experiment.toml"
final_experiment = "experiment.toml"
trials = 2
sampler_seed = 7
vocabulary_rows = 2
vocabulary_epochs = 2
patience = 1
evaluation_interval = 1

[space]
learning_rate_min = 0.0001
learning_rate_max = 0.001
weight_decays = [0.0, 0.001]
batch_sizes = [16, 32]
"""


def _alignment(options) -> AlignmentResult:
    distance = abs(options.learning_rate - 0.0003)
    score = 0.001 + distance
    return AlignmentResult(
        profile="small",
        optimizer={"hidden_matrix_parameters": "Muon"},
        embedding_metrics={
            "normalized_rmse": score,
            "cosine_similarity_p01": 1.0 - score,
            "cosine_similarity_p50": 1.0 - score / 2,
        },
        candidate_state_bytes=40,
        reference_state_bytes=100,
        candidate_reference_state_ratio=0.4,
    )


def _assert_completed_search(values) -> None:
    output_dir = values["output_dir"]
    prepared = values["prepared"]
    prepared_report = values["prepared_report"]
    result = values["result"]
    resumed = values["resumed"]
    fit_alignment = values["fit_alignment"]
    assert prepared["status"] == "running"
    assert prepared["mode"] == "input_only"
    assert prepared["evidence_scope"] == "search"
    assert prepared["operational_status"] == "running"
    assert prepared["scientific_verdict"] == "not_applicable_search"
    assert prepared["finished_trials"] == 0
    assert "Selected alignment gate: `pending`" in prepared_report
    assert "| 0 | waiting | n/a | n/a | n/a | n/a | n/a | n/a |" in prepared_report
    assert result["completed_trials"] == 2
    assert result["finished_trials"] == 2
    assert result["failed_trials"] == 0
    assert result["status"] == "complete"
    assert result["operational_status"] == "completed"
    assert not result["selected_alignment_passed"]
    assert result["selected_compactness_passed"]
    assert result["vocabulary_rows"] == 2
    assert len(json.loads((output_dir / "vocabulary-sample.json").read_text())["token_ids"]) == 2
    assert fit_alignment.call_count == 2
    assert all(len(call.kwargs["token_ids"]) == 2 for call in fit_alignment.call_args_list)
    assert resumed == result
    assert result["selected_trial"] == 0
    assert result["source_commit"] == "commit"
    assert result["source_dirty"] is True
    assert result["source_state_sha256"] == "a" * 64
    assert len(result["dependency_lock_sha256"]) == 64
    assert result["selection_feasible"] is False
    assert result["selected_parameters"] == {
        "batch_size": 32,
        "learning_rate": 0.0003,
        "weight_decay": 0.0,
    }
    assert (output_dir / "optuna-journal.log").is_file()
    assert "Vocabulary Alignment Search" in (output_dir / "search-report.md").read_text()
    with (output_dir / "selected-experiment.toml").open("rb") as handle:
        selected = tomllib.load(handle)
    assert selected["training"]["profile"] == "small"
    assert selected["training"]["learning_rate"] == 0.0003
    selected_spec = ExperimentSpec.load(output_dir / "selected-experiment.toml")
    assert selected_spec.training.batch_size == 32
    assert selected_spec.evidence_scope == "final"
    assert selected_spec.search_selections[0].search_kind == "alignment"
    assert selected_spec.search_selections[0].artifact_sha256
    assert not selected_spec.search_selections[0].feasible


def test_search_is_strict_and_writes_a_frozen_selected_experiment() -> None:
    search = _search_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        experiment_path = root / "experiment.toml"
        search_path = root / "search.toml"
        output_dir = root / "result"
        experiment_path.write_text(_experiment(), encoding="utf-8")
        search_path.write_text(_search(), encoding="utf-8")

        with (
            patch.object(search, "find_project_root", return_value=Path(__file__).parents[1]),
            patch.object(
                search,
                "source_state",
                return_value=("commit", True, "a" * 64),
            ),
            patch.object(
                search,
                "load_model_assets",
                return_value=SimpleNamespace(
                    vocabulary=SimpleNamespace(
                        ordinary_ids=(1, 2, 3),
                        compatibility_ids=(1, 2, 3),
                        bytes_for=lambda token_id: bytes((token_id, token_id)),
                    )
                ),
            ),
            patch.object(
                search,
                "fit_vocabulary_alignment",
                side_effect=lambda _assets, options, device, token_ids: _alignment(options),  # noqa: ARG005 - Matches keyword call protocol.
            ) as fit_alignment,
        ):
            prepared = search.run_vocabulary_search(search_path, output_dir, prepare_only=True)
            prepared_report = (output_dir / "search-report.md").read_text()
            assert fit_alignment.call_count == 0
            result = search.run_vocabulary_search(search_path, output_dir, resume=True)
            resumed = search.run_vocabulary_search(search_path, output_dir, resume=True)
            with (
                patch.object(
                    search,
                    "source_state",
                    return_value=("commit", True, "b" * 64),
                ),
                unittest.TestCase().assertRaisesRegex(ValueError, "different specification"),
            ):
                search.run_vocabulary_search(search_path, output_dir, resume=True)

        _assert_completed_search(
            {
                "output_dir": output_dir,
                "prepared": prepared,
                "prepared_report": prepared_report,
                "result": result,
                "resumed": resumed,
                "fit_alignment": fit_alignment,
            }
        )

        assertions = unittest.TestCase()
        with assertions.assertRaisesRegex(FileExistsError, "already exists"):
            search.run_vocabulary_search(search_path, output_dir)


def test_search_rejects_unknown_fields() -> None:
    search = _search_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        experiment_path = root / "experiment.toml"
        search_path = root / "search.toml"
        experiment_path.write_text(_experiment())
        search_path.write_text(_search() + "\nunknown = true\n", encoding="utf-8")
        assertions = unittest.TestCase()
        with assertions.assertRaisesRegex(ValueError, "unknown .* fields"):
            search.SearchSpec.load(search_path)


def test_alignment_budget_and_resume_reconcile_terminal_states() -> None:
    search = _search_module()
    trials = [
        SimpleNamespace(state=search.TrialState.COMPLETE),
        SimpleNamespace(state=search.TrialState.PRUNED),
        SimpleNamespace(state=search.TrialState.FAIL),
        SimpleNamespace(state=search.TrialState.WAITING),
    ]
    assert search._remaining_trials(trials, requested=5) == 2

    running = SimpleNamespace(number=7, state=search.TrialState.RUNNING)
    study = SimpleNamespace(
        get_trials=lambda **_kwargs: [running],
        tell=Mock(),
    )
    search._reconcile_running_trials(study)
    study.tell.assert_called_once_with(7, state=search.TrialState.FAIL)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_search_is_strict_and_writes_a_frozen_selected_experiment,
            test_search_rejects_unknown_fields,
            test_alignment_budget_and_resume_reconcile_terminal_states,
        )
    )
