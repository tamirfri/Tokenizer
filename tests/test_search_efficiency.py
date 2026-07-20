from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from search_fixtures import _optional_search_module

from continuous_tokenizer.backbone.synthetic import synthetic_model_assets
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import InputGateSpec, InputTrainingSpec
from continuous_tokenizer.contracts.search import EfficiencyPilotSpec
from continuous_tokenizer.input.training.vocabulary import AlignmentResult


def _efficiency_search_module() -> ModuleType:
    return _optional_search_module("continuous_tokenizer.search.efficiency")


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


def test_efficiency_search_pre_registers_fixed_factors_without_training() -> None:
    module = _efficiency_search_module()
    repository = Path(__file__).parents[1]
    path = repository / "experiments/searches/qwen35-0.8b-efficiency.toml"
    spec = EfficiencyPilotSpec.load(path)
    assert spec.space.batch_sizes == (256, 512, 1024)
    assert spec.space.projection_multipliers == (2, 4, 8)
    assert spec.space.muon_ns_steps == (3, 5)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "efficiency"
        with (
            patch.object(
                module,
                "load_model_assets",
                return_value=synthetic_model_assets(),
            ),
            patch.object(
                module,
                "_vocabulary_sample",
                return_value=((256,), "0" * 64),
            ),
            patch.object(module, "source_state", return_value=("commit", True, "a" * 64)),
            patch.object(module, "sha256_file", return_value="b" * 64),
        ):
            prepared = module.run_efficiency_search(
                path,
                output,
                resume=False,
                prepare_only=True,
            )
            with (
                patch.object(module, "source_state", return_value=("commit", True, "c" * 64)),
                unittest.TestCase().assertRaisesRegex(ValueError, "different contract"),
            ):
                module.run_efficiency_search(
                    path,
                    output,
                    resume=True,
                    prepare_only=True,
                )

        assert prepared["status"] == "prepared"
        assert prepared["evidence_scope"] == "search"
        assert prepared["scientific_verdict"] == "not_applicable_search"
        assert prepared["source_commit"] == "commit"
        assert prepared["source_dirty"] is True
        assert prepared["source_state_sha256"] == "a" * 64
        assert prepared["dependency_lock_sha256"] == "b" * 64
        assert prepared["selected_trial"] is None
        assert (output / "search-spec.json").is_file()
        assert (output / "vocabulary-sample.json").is_file()


def test_efficiency_trials_retain_compiler_state_between_models() -> None:
    module = _efficiency_search_module()
    options = SimpleNamespace(
        batch_size=32,
        vocabulary_epochs=2,
        learning_rate=0.0003,
        muon_ns_steps=5,
        profile=SimpleNamespace(projection_multiplier=8),
    )
    with (
        patch.object(module.torch.compiler, "reset") as reset,
        patch.object(
            module,
            "fit_vocabulary_alignment",
            side_effect=lambda *_args, **_kwargs: _alignment(options),
        ),
    ):
        for _ in range(2):
            module._run_alignment(
                synthetic_model_assets(),
                options,
                (256, 257),
                module.torch.device("cpu"),
            )

    reset.assert_not_called()


def test_efficiency_trial_budget_counts_every_terminal_state() -> None:
    module = _efficiency_search_module()
    trials = [
        SimpleNamespace(state=module.TrialState.COMPLETE),
        SimpleNamespace(state=module.TrialState.PRUNED),
        SimpleNamespace(state=module.TrialState.FAIL),
        SimpleNamespace(state=module.TrialState.WAITING),
    ]

    assert module._remaining_trials(trials, requested=5) == 2


def test_efficiency_resume_reconciles_persisted_running_trials() -> None:
    module = _efficiency_search_module()
    running = SimpleNamespace(number=7, state=module.TrialState.RUNNING)
    study = SimpleNamespace(
        get_trials=lambda **_kwargs: [running],
        tell=Mock(),
    )

    module._reconcile_running_trials(study)

    study.tell.assert_called_once_with(7, state=module.TrialState.FAIL)


def test_efficiency_selection_prefers_feasible_trial_before_runtime() -> None:
    module = _efficiency_search_module()
    gates = InputGateSpec()

    def trial(number: int, duration: float, normalized_rmse: float) -> SimpleNamespace:
        return SimpleNamespace(
            number=number,
            state=module.TrialState.COMPLETE,
            user_attrs={
                "result": {
                    "duration_seconds": duration,
                    "candidate_reference_state_ratio": 0.4,
                    "embedding_metrics": {
                        "normalized_rmse": normalized_rmse,
                        "cosine_similarity_p01": 1.0,
                        "cosine_similarity_p50": 1.0,
                    },
                }
            },
        )

    selected = module._selected_trial(
        [trial(0, 1.0, 1.0), trial(1, 8.0, 0.001)],
        baseline_seconds=10.0,
        minimum_improvement=0.1,
        gates=gates,
    )

    assert selected is not None
    assert selected.number == 1


def test_efficiency_selected_experiment_is_complete_and_reloadable() -> None:
    module = _efficiency_search_module()
    repository = Path(__file__).parents[1]
    search = EfficiencyPilotSpec.load(repository / "experiments/searches/qwen35-0.8b-efficiency.toml")
    final = ExperimentSpec.load(repository / "experiments/campaigns/input/qwen35-0.8b/seed-17.toml")
    assert isinstance(final.training, InputTrainingSpec)
    assert isinstance(final.gates, InputGateSpec)
    selected_result = {
        "duration_seconds": 8.0,
        "candidate_reference_state_ratio": 0.4,
        "embedding_metrics": {
            "normalized_rmse": 0.001,
            "cosine_similarity_p01": 0.9999,
            "cosine_similarity_p50": 0.99999,
        },
    }
    trial = SimpleNamespace(
        number=3,
        state=module.TrialState.COMPLETE,
        params={
            "learning_rate": 0.0002,
            "batch_size": 256,
            "projection_multiplier": 8,
            "muon_ns_steps": 5,
        },
        user_attrs={"result": selected_result},
    )
    study = SimpleNamespace(get_trials=lambda **_kwargs: [trial])
    baseline = {**selected_result, "duration_seconds": 10.0}

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        summary = module._write_summary(
            module._SummaryRequest(
                output_dir=output,
                contract={"artifacts": {}},
                study=study,
                baseline=baseline,
                gates=final.gates,
                spec=search,
                final_experiment=final,
            )
        )
        selected = ExperimentSpec.load(output / "selected-experiment.toml")

    assert summary["selected_parameters"]["weight_decay"] == final.training.weight_decay
    assert selected.training.weight_decay == final.training.weight_decay
    assert selected.efficiency_pilot_sha256
    assert selected.efficiency_pilot is not None
    assert selected.search_selections[0].artifact
    assert selected.search_selections[0].selected_parameters["weight_decay"] == final.training.weight_decay


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_efficiency_search_pre_registers_fixed_factors_without_training,
            test_efficiency_trials_retain_compiler_state_between_models,
            test_efficiency_trial_budget_counts_every_terminal_state,
            test_efficiency_resume_reconciles_persisted_running_trials,
            test_efficiency_selection_prefers_feasible_trial_before_runtime,
            test_efficiency_selected_experiment_is_complete_and_reloadable,
        )
    )
