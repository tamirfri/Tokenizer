from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from search_fixtures import _optional_search_module

from continuous_tokenizer.backbone.synthetic import synthetic_model_assets
from continuous_tokenizer.campaigns.output import (
    OutputPilotCorpus,
    OutputRunnerOptions,
    _output_sequence_corpora,
)
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.output import OutputTrainingSpec

REPOSITORY = Path(__file__).parents[1]
OUTPUT_SEARCH_SPEC = REPOSITORY / "experiments/searches/qwen35-0.8b-output.toml"
PILOT_DOCUMENTS = tuple(f"pilot document {index}".encode() for index in range(256))


def _output_search_module() -> ModuleType:
    return _optional_search_module("continuous_tokenizer.search.output")


def test_output_search_prepare_contract_resumes_without_training() -> None:
    module = _output_search_module()
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "search"

        with (
            patch.object(module, "load_corpus_documents", return_value=PILOT_DOCUMENTS),
            patch.object(module, "source_state", return_value=("commit", True, "a" * 64)),
            patch.object(module, "sha256_file", return_value="b" * 64),
        ):
            prepared = module.run_output_search(
                OUTPUT_SEARCH_SPEC,
                output,
                resume=False,
                prepare_only=True,
            )
            with (
                patch.object(module, "source_state", return_value=("commit", True, "c" * 64)),
                unittest.TestCase().assertRaisesRegex(ValueError, "different specification"),
            ):
                module.run_output_search(
                    OUTPUT_SEARCH_SPEC,
                    output,
                    resume=True,
                    prepare_only=True,
                )
            resumed = module.run_output_search(
                OUTPUT_SEARCH_SPEC,
                output,
                resume=True,
                prepare_only=True,
            )

        assert prepared == resumed
        assert prepared["mode"] == "output_only"
        assert prepared["evidence_scope"] == "search"
        assert prepared["operational_status"] == "running"
        assert prepared["scientific_verdict"] == "not_applicable_search"
        assert prepared["profile"] == "large"
        assert prepared["source_commit"] == "commit"
        assert prepared["source_dirty"] is True
        assert prepared["source_state_sha256"] == "a" * 64
        assert prepared["dependency_lock_sha256"] == "b" * 64
        assert prepared["pilot_corpus"]["training_documents"] == 4
        assert prepared["pilot_corpus"]["checkpoint_selection_documents"] == 4
        assert prepared["pilot_corpus"]["oracle_validation_documents"] == 4
        assert (
            len(
                {
                    prepared["pilot_corpus"]["training_sha256"],
                    prepared["pilot_corpus"]["checkpoint_selection_sha256"],
                    prepared["pilot_corpus"]["oracle_validation_sha256"],
                }
            )
            == 3
        )
        assert len(prepared["pilot_corpus"]["sha256"]) == 64
        assert (output / "search-spec.json").is_file()
        prepared_report = (output / "search-report.md").read_text()
        assert "Output Tokenizer Search" in prepared_report
        assert "Evidence scope: `search`" in prepared_report
        assert "NO TRIAL OR SELECTION IS FINAL MODEL EVIDENCE" in prepared_report


def test_infeasible_oracle_materializes_unsupported_search_without_training() -> None:
    module = _output_search_module()
    runner = Mock()
    oracle = {
        "artifact": "/sealed/oracle/result.json",
        "artifact_sha256": "a" * 64,
        "study_fingerprint": "b" * 64,
        "max_span": 2,
        "selection_feasible": False,
    }
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.object(module, "load_corpus_documents", return_value=PILOT_DOCUMENTS),
        patch.object(module, "_oracle_selection", return_value=oracle),
        patch.object(module, "OutputExperimentRunner", runner),
    ):
        output = Path(directory) / "search"
        result = module.run_output_search(
            OUTPUT_SEARCH_SPEC,
            output,
            resume=False,
            prepare_only=False,
            oracle_study_artifact=Path("oracle"),
        )
        selected = ExperimentSpec.load(
            output / "selected-experiment.toml",
        )

    runner.assert_not_called()
    assert result["operational_status"] == "completed"
    assert result["status"] == "completed_unsupported"
    assert result["selection_feasible"] is False
    assert result["selected_metrics"]["training_performed"] is False
    assert isinstance(selected.training, OutputTrainingSpec)
    assert selected.training.max_span == 2
    assert selected.search_selections[0].feasible is False


def test_output_search_prefers_all_gate_feasible_candidate() -> None:
    module = _output_search_module()

    class FakeRunner:
        def __init__(
            self,
            experiment,
            _output_dir,
            _project_root,
            options: OutputRunnerOptions,
        ) -> None:
            self.experiment = experiment
            assert experiment.seed == 17
            assert options.pilot_corpus is not None
            assert options.pilot_corpus.training_documents

        def run(self) -> dict:
            trial = int(self.experiment.name.rsplit("-", 1)[1])
            feasible = trial == 1
            return {
                "evidence_scope": "search",
                "output": {
                    "exact_full_sequence_rate": 0.99 - trial / 100,
                    "oracle_feasible": True,
                },
                "gates": {
                    "direct_feedback": feasible,
                    "candidate_reference_state_ratio": feasible,
                },
            }

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "search"
        with (
            patch.object(module, "load_corpus_documents", return_value=PILOT_DOCUMENTS),
            patch.object(module, "OutputExperimentRunner", FakeRunner),
        ):
            result = module.run_output_search(
                OUTPUT_SEARCH_SPEC,
                output,
                resume=False,
                prepare_only=False,
            )

        assert result["trials"][0]["state"] == "COMPLETE"
        assert result["selected_trial"] == 1
        assert result["selection_feasible"]
        assert result["selection_policy"] == "all_gates_feasible"
        assert all(not trial["final_evidence"] for trial in result["trials"])
        assert (output / "selected-experiment.toml").is_file()
        selected = ExperimentSpec.load(output / "selected-experiment.toml")
        assert selected.evidence_scope == "final"
        assert selected.search_selections[0].search_kind == "output"
        assert selected.search_selections[0].artifact_sha256
        assert selected.search_selections[0].feasible
        report = (output / "search-report.md").read_text()
        assert "Status: `completed`" in report
        assert "| 0 | COMPLETE |" in report
        assert "## Selected Candidate" in report
        assert "Gates: `" in report


def test_output_search_reports_no_candidate_and_failure_states() -> None:
    module = _output_search_module()

    class PrunedRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> dict:
            return {
                "output": {
                    "exact_full_sequence_rate": 0.5,
                    "candidate_reference_state_ratio": 0.75,
                    "oracle_feasible": True,
                },
                "gates": {"candidate_reference_state_ratio": False},
            }

    class FailedRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> dict:
            raise RuntimeError("candidate training failed")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        no_candidate_output = root / "no-candidate"
        with (
            patch.object(module, "load_corpus_documents", return_value=PILOT_DOCUMENTS),
            patch.object(module, "OutputExperimentRunner", PrunedRunner),
        ):
            no_candidate = module.run_output_search(
                OUTPUT_SEARCH_SPEC,
                no_candidate_output,
                resume=False,
                prepare_only=False,
            )
        assert no_candidate["status"] == "completed"
        assert no_candidate["finished_trials"] == no_candidate["requested_trials"]
        assert no_candidate["selection_feasible"] is False
        assert no_candidate["selection_policy"] == "best_exact_full_sequence_rate"
        assert "Status: `completed`" in (no_candidate_output / "search-report.md").read_text()

        failed_output = root / "failed"
        with (
            patch.object(module, "load_corpus_documents", return_value=PILOT_DOCUMENTS),
            patch.object(module, "OutputExperimentRunner", FailedRunner),
        ):
            failed = module.run_output_search(
                OUTPUT_SEARCH_SPEC,
                failed_output,
                resume=False,
                prepare_only=False,
            )
        assert failed == json.loads((failed_output / "search.json").read_text())
        failed_report = (failed_output / "search-report.md").read_text()
        assert failed["status"] == "completed_no_candidate"
        assert failed["operational_status"] == "completed"
        assert failed["finished_trials"] == failed["requested_trials"]
        assert failed["failed_trials"] == failed["requested_trials"]
        assert failed["trials"][0]["state"] == "FAIL"
        assert "Status: `completed_no_candidate`" in failed_report
        assert "candidate training failed" in failed_report


def test_output_search_prunes_infeasible_native_head_oracle_before_training() -> None:
    module = _output_search_module()
    ceilings = {
        str(limit): {
            "native_tokens": 4,
            "events": 4,
            "feasible": True,
            "exact_native_sequence_rate_ceiling": 1.0,
            "bytes_per_event_ceiling": 2.0,
            "native_tokens_per_attempted_macro_step_ceiling": 1.0,
        }
        for limit in (1, 2, 4, 8)
    }

    class OracleInfeasibleRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> dict:
            raise module.OutputOracleCeilingError(ceilings, 8)

    with (
        tempfile.TemporaryDirectory() as directory,
        patch.object(module, "load_corpus_documents", return_value=PILOT_DOCUMENTS),
        patch.object(module, "OutputExperimentRunner", OracleInfeasibleRunner),
    ):
        result = module.run_output_search(
            OUTPUT_SEARCH_SPEC,
            Path(directory) / "search",
            resume=False,
            prepare_only=False,
        )

    assert result["status"] == "completed_no_candidate"
    assert all(trial["state"] == "PRUNED" for trial in result["trials"])
    assert result["trials"][0]["metrics"]["native_head_oracle_ceilings"] == ceilings


def test_output_resume_reconciles_running_without_spending_extra_budget() -> None:
    module = _output_search_module()
    running = SimpleNamespace(
        number=7,
        state=module.optuna.trial.TrialState.RUNNING,
    )
    study = SimpleNamespace(
        get_trials=lambda **_kwargs: [running],
        tell=Mock(),
    )

    module._reconcile_running_trials(study)

    study.tell.assert_called_once_with(
        7,
        state=module.optuna.trial.TrialState.FAIL,
    )
    trials = [
        SimpleNamespace(state=module.optuna.trial.TrialState.COMPLETE),
        SimpleNamespace(state=module.optuna.trial.TrialState.PRUNED),
        SimpleNamespace(state=module.optuna.trial.TrialState.FAIL),
    ]
    assert module._remaining_trials(trials, requested=4) == 1


def test_output_search_sequence_corpora_never_load_final_prompts() -> None:
    spec = ExperimentSpec.load(
        REPOSITORY / "experiments/searches/qwen35-0.8b-output-base.toml",
    )
    pilot = OutputPilotCorpus(
        (b"training",),
        (b"checkpoint",),
        (b"oracle",),
    )

    with patch(
        "continuous_tokenizer.campaigns.output._registered_prompt_sequences",
        side_effect=AssertionError("search loaded final prompts"),
    ):
        corpora = _output_sequence_corpora(
            synthetic_model_assets(),
            spec,
            pilot,
            limit=1,
        )

    assert corpora.final_test == corpora.oracle_validation
    assert corpora.final_test_metadata == {"scope": "not_executed_in_search"}


def test_prospective_output_sequence_corpora_use_validation_without_final_prompts() -> None:
    spec = ExperimentSpec.load(
        REPOSITORY / "experiments/campaigns/output/qwen35-0.8b/seed-17.toml",
    )
    selected = (((1,),), {"scope": "validation"}, frozenset())
    with (
        patch(
            "continuous_tokenizer.campaigns.output._corpus_token_sequences",
            return_value=selected,
        ),
        patch(
            "continuous_tokenizer.campaigns.output._registered_prompt_sequences",
            side_effect=AssertionError("prospective run loaded final prompts"),
        ),
    ):
        corpora = _output_sequence_corpora(
            synthetic_model_assets(),
            spec,
            None,
            limit=1,
            validation_only=True,
        )

    assert corpora.final_test == corpora.oracle_validation
    assert corpora.final_test_metadata["final_test_loaded"] is False


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_output_search_prepare_contract_resumes_without_training,
            test_infeasible_oracle_materializes_unsupported_search_without_training,
            test_output_search_prefers_all_gate_feasible_candidate,
            test_output_search_reports_no_candidate_and_failure_states,
            test_output_search_prunes_infeasible_native_head_oracle_before_training,
            test_output_resume_reconciles_running_without_spending_extra_budget,
            test_output_search_sequence_corpora_never_load_final_prompts,
            test_prospective_output_sequence_corpora_use_validation_without_final_prompts,
        )
    )
