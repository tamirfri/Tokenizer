from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from input_training_fixtures import TEST_PROFILE, pair_assets

import continuous_tokenizer.input.study as study_module
from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.artifacts.store import RunDirectory
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.cli import build_parser
from continuous_tokenizer.contracts.input import InputTrainingSpec
from continuous_tokenizer.contracts.input_study import (
    INPUT_ALIGNMENT_CONTINUATION_RULE,
    INPUT_ALIGNMENT_FEASIBILITY_STAGES,
    INPUT_ALIGNMENT_SUBSET_SEED,
    INPUT_ALIGNMENT_TRAINING_SEEDS,
    INPUT_COMPRESSION_CONTINUATION_RULE,
    INPUT_COMPRESSION_FINAL_ACTION,
    INPUT_COMPRESSION_TRAINING_SEEDS,
    INPUT_COMPRESSION_VOCABULARY_ROWS,
    INPUT_SELECTION_CANDIDATES,
    INPUT_SELECTION_RULE,
    InputAlignmentFeasibilityStudySpec,
    InputCompressionFeasibilityStudySpec,
    InputSelectionStudySpec,
)
from continuous_tokenizer.input.studies import (
    CandidateLengthRequest,
    VocabularySubset,
    candidate_length_report,
    deterministic_binary_spans,
    registered_vocabulary_subset,
    select_input_candidate,
)
from continuous_tokenizer.input.training.run import (
    TrainingOptions,
    TrainingResult,
)
from continuous_tokenizer.input.training.runtime import TrainingRuntime
from continuous_tokenizer.input.training.vocabulary import AlignmentResult


def _candidate(
    name: str,
    *,
    mean_kl: float,
    passed: bool = True,
) -> dict:
    return {
        "name": name,
        "tokenizer": {
            "acceptance": {
                "overall": passed,
                "density": passed,
                "embedding_fit": False,
                "compactness": True,
            },
            "density": {
                "round_trip": passed,
                "native_tokens_per_continuous_token": 1.25,
            },
        },
        "validation": {
            "teacher_forced": {
                "segmented": {
                    "mean_kl": mean_kl,
                    "mean_js": mean_kl / 2,
                    "teacher_nll": 1.0,
                    "student_nll": 1 + mean_kl,
                    "top1_agreement": 1 - mean_kl,
                },
            },
            "generation": {
                "segmented_mean_byte_similarity": 1 - mean_kl,
            },
            "positions": {"native_positions_per_segmented_position": 1.25},
        },
    }


def test_checked_in_input_studies_are_strict_large_seed_17_contracts() -> None:
    root = Path(__file__).parents[1]
    paths = sorted(
        path
        for path in (root / "experiments/studies/input").glob("*.toml")
        if "alignment-feasibility" not in path.name and "compression-feasibility" not in path.name
    )

    specs = [InputSelectionStudySpec.load(path) for path in paths]

    assert len(specs) == 2
    assert {spec.kind for spec in specs} == {"scaling"}
    assert all(spec.selection_rule == INPUT_SELECTION_RULE for spec in specs)
    assert all(not spec.run_selection for spec in specs)
    assert all(spec.load_experiment(path).training.profile == "large" for spec, path in zip(specs, paths, strict=True))


def test_input_study_contract_rejects_unregistered_candidate_lengths() -> None:
    root = Path(__file__).parents[1]
    source = (root / "experiments/studies/input/qwen35-0.8b-scaling.toml").read_text()
    source = source.replace(
        "candidate_lengths = [2, 8, 32]",
        "candidate_lengths = [2, 8]",
    )
    with tempfile.TemporaryDirectory(dir=root / "experiments/studies/input") as directory:
        path = Path(directory) / "invalid.toml"
        path.write_text(source)
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "exact candidate lengths",
        ):
            InputSelectionStudySpec.load(path)


def test_vocabulary_and_binary_study_inputs_are_content_hashed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assets = pair_assets(Path(directory))
        first = registered_vocabulary_subset(assets, 1, 17)
        second = registered_vocabulary_subset(assets, 1, 17)
        binary_first = deterministic_binary_spans((2, 4), 3, 17)
        binary_second = deterministic_binary_spans((2, 4), 3, 17)

    assert first == second
    assert first.token_ids == (256,)
    assert len(first.sha256) == 64
    assert binary_first == binary_second
    assert binary_first[2] != deterministic_binary_spans((2,), 3, 23)[2]


def test_vocabulary_subset_excludes_aliases_and_stratifies_span_lengths() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assets = pair_assets(Path(directory))
        vocabulary = assets.vocabulary
        alias_id = len(vocabulary.token_bytes) + 2
        assets = replace(
            assets,
            vocabulary=ByteVocabulary(
                token_bytes=(
                    *vocabulary.token_bytes,
                    b"abc",
                    b"abcd",
                    b"ab",
                ),
                ordinary_ids=(*vocabulary.ordinary_ids, 257, 258, alias_id),
                control_ids=vocabulary.control_ids,
                byte_token_ids=vocabulary.byte_token_ids,
                max_token_bytes=4,
                compatibility_ids=(*vocabulary.compatibility_ids, 257, 258),
            ),
            input_embeddings=torch.cat(
                (assets.input_embeddings, torch.randn(3, 8)),
            ),
        )

        subset = registered_vocabulary_subset(assets, 3, 17)

    assert alias_id not in subset.token_ids
    assert {len(assets.vocabulary.bytes_for(token_id)) for token_id in subset.token_ids} == {2, 3, 4}
    assert subset.algorithm == "content_hashed_length_stratified_non_atomic_compatibility_rows"


def test_every_input_study_manifest_binds_source_and_dependency_provenance() -> None:
    repository = Path(__file__).parents[1]
    paths = sorted(
        path
        for path in (repository / "experiments/studies/input").glob("*.toml")
        if "alignment-feasibility" not in path.name and "compression-feasibility" not in path.name
    )
    with (
        tempfile.TemporaryDirectory() as directory,
        mock.patch.object(
            study_module,
            "source_state",
            return_value=("commit", True, "a" * 64),
        ),
        mock.patch.object(study_module, "sha256_file", return_value="b" * 64),
    ):
        root = Path(directory)
        assets = pair_assets(root)
        for index, path in enumerate(paths):
            study = InputSelectionStudySpec.load(path)
            experiment = study.load_experiment(path)
            run = RunDirectory(root / f"study-{index}")
            run.write_json("result.json", {"study": study.name})
            registered = study_module._registered_study(
                path,
                study,
                experiment,
            )
            manifest = study_module._study_manifest(
                study_module._StudyContext(
                    run,
                    assets,
                    experiment,
                    b"",
                    study,
                ),
                {"result": "result.json"},
                registered,
            )

            assert manifest["source_commit"] == "commit"
            assert manifest["source_dirty"] is True
            assert manifest["source_state_sha256"] == "a" * 64
            assert manifest["dependency_lock_sha256"] == "b" * 64
            assert manifest["study_fingerprint"] == study.fingerprint()
            assert manifest["experiment_fingerprint"] == experiment.fingerprint()
            assert manifest["artifact_hashes"]["result"]


def test_candidate_length_report_covers_all_registered_sources_and_lengths() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        runtime = TrainingRuntime(
            assets,
            TrainingOptions(output_dir=root / "checkpoints", profile=TEST_PROFILE),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(TEST_PROFILE).eval()

        report = candidate_length_report(
            codec,
            CandidateLengthRequest(
                assets=assets,
                validation_data=bytes(range(128)),
                candidate_lengths=(2, 8, 32),
                binary_samples_per_length=2,
                seed=17,
                batch_size=2,
            ),
        )

    assert set(report) >= {
        "vocabulary",
        "wikitext_validation",
        "arbitrary_binary",
    }
    for source in ("vocabulary", "wikitext_validation", "arbitrary_binary"):
        assert set(report[source]["metrics"]) == {"2", "8", "32"}
    assert report["vocabulary"]["metrics"]["2"]["candidates"] == 1
    assert report["arbitrary_binary"]["metrics"]["32"]["source_bytes"] == 64


def test_selection_uses_untouched_metrics_instead_of_aligned_construction() -> None:
    candidates = [
        _candidate("reconstruction_only", mean_kl=0.03),
        _candidate("token_aligned_distillation", mean_kl=0.02),
        _candidate("arbitrary_boundary_distillation", mean_kl=0.01),
    ]

    selection = select_input_candidate(candidates)

    assert selection["selected_candidate"] == "arbitrary_boundary_distillation"
    assert selection["selection_metrics_split"] == "validation"
    assert selection["untouched_by_training"] is True
    assert selection["selection_rule"] == INPUT_SELECTION_RULE

    candidates[2]["tokenizer"]["acceptance"]["density"] = False
    candidates[0]["tokenizer"]["acceptance"]["density"] = False
    selection = select_input_candidate(candidates)
    assert selection["selected_candidate"] == "token_aligned_distillation"
    assert selection["selection_feasible"] is True
    assert tuple(candidate["name"] for candidate in candidates) == INPUT_SELECTION_CANDIDATES


def test_cli_registers_study_orchestration() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        ["study", "study.toml", "--output-dir", "result"],
    )

    assert parsed.command == "study"
    assert parsed.spec == Path("study.toml")
    assert parsed.output_dir == Path("result")
    assert parsed.resume is False
    assert (
        build_parser()
        .parse_args(
            ["study", "study.toml", "--output-dir", "result", "--resume"],
        )
        .resume
    )


def test_input_study_resume_retains_hash_verified_completed_trial() -> None:
    with tempfile.TemporaryDirectory() as directory:
        trial_dir = Path(directory) / "trial"
        trial_dir.mkdir()
        checkpoint = trial_dir / "checkpoint.pt"
        lengths = trial_dir / "candidate-lengths.json"
        checkpoint.write_bytes(b"checkpoint")
        lengths.write_text("{}\n", encoding="utf-8")
        subset = {"token_ids": [1], "sha256": "a" * 64}
        trial = {
            "vocabulary_subset": subset,
            "training": {"checkpoint": str(checkpoint)},
            "candidate_lengths": {},
            "artifact_hashes": {
                "checkpoint": sha256_file(checkpoint),
                "candidate_lengths": sha256_file(lengths),
            },
        }
        trial_path = trial_dir / "trial.json"
        RunDirectory(Path(directory), resume=True).write_json(
            "trial/trial.json",
            trial,
        )

        resumed = study_module._completed_trial(trial_path, subset)
        assert resumed == trial

        checkpoint.write_bytes(b"changed")
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "artifact hash mismatch",
        ):
            study_module._completed_trial(trial_path, subset)


def _alignment_metrics(*, passed: bool) -> dict[str, float]:
    return {
        "normalized_rmse": 0.005 if passed else 0.02,
        "cosine_similarity_p01": 0.9995,
        "cosine_similarity_p50": 0.99995,
        "reconstruction_fraction": 0.0,
    }


def _run_alignment_feasibility(
    root: Path,
    outcomes: list[bool],
    *,
    resume: bool = False,
) -> tuple[dict, mock.Mock]:
    repository = Path(__file__).parents[1]
    study_path = repository / "experiments/studies/input/qwen35-0.8b-alignment-feasibility.toml"
    experiment = InputAlignmentFeasibilityStudySpec.load(study_path).load_experiment(
        study_path,
    )
    assets = replace(
        pair_assets(root),
        model_id=experiment.model.model_id,
        revision=experiment.model.revision,
    )
    outcome_iterator = iter(outcomes)

    def fit(_assets, options, *, device, token_ids):
        del _assets, device, token_ids
        result = AlignmentResult(
            profile="large",
            optimizer={},
            embedding_metrics=_alignment_metrics(passed=next(outcome_iterator)),
            candidate_state_bytes=1,
            reference_state_bytes=2,
            candidate_reference_state_ratio=0.5,
        )
        RunDirectory(options.output_dir).write_json(
            "alignment-result.json",
            asdict(result),
        )
        return result

    fit_mock = mock.Mock(side_effect=fit)
    with (
        mock.patch.object(study_module, "load_model_assets", return_value=assets),
        mock.patch.object(
            study_module,
            "declared_device",
            return_value=torch.device("cpu"),
        ),
        mock.patch.object(
            study_module,
            "dependency_environment",
            return_value={"device": "cpu"},
        ),
        mock.patch.object(
            study_module,
            "registered_vocabulary_subset",
            side_effect=lambda _assets, rows, seed: VocabularySubset(
                requested_rows=rows,
                token_ids=(256,),
                sha256=f"{seed:064x}",
                algorithm="test",
            ),
        ),
        mock.patch.object(
            study_module,
            "fit_vocabulary_alignment",
            fit_mock,
        ),
    ):
        result = study_module.run_input_alignment_feasibility_study(
            study_path,
            root / "study",
            resume=resume,
        )
    return result, fit_mock


def test_checked_in_alignment_feasibility_studies_are_equal_prospective_contracts() -> None:
    root = Path(__file__).parents[1]
    paths = sorted(
        (root / "experiments/studies/input").glob(
            "*-alignment-feasibility.toml",
        ),
    )
    specs = [InputAlignmentFeasibilityStudySpec.load(path) for path in paths]
    experiments = [spec.load_experiment(path) for spec, path in zip(specs, paths, strict=True)]

    assert len(specs) == 2
    assert {spec.vocabulary_subset_sizes for spec in specs} == {
        INPUT_ALIGNMENT_FEASIBILITY_STAGES,
    }
    assert {spec.continuation_rule for spec in specs} == {
        INPUT_ALIGNMENT_CONTINUATION_RULE,
    }
    assert all(spec.prospective for spec in specs)
    assert all(spec.training_seeds == INPUT_ALIGNMENT_TRAINING_SEEDS for spec in specs)
    assert all(spec.subset_seed == INPUT_ALIGNMENT_SUBSET_SEED for spec in specs)
    assert len({spec.fingerprint() for spec in specs}) == 2
    assert {experiment.model.model_id for experiment in experiments} == {
        "Qwen/Qwen3.5-0.8B",
        "google/gemma-3-270m-it",
    }
    assert all(experiment.evidence_scope == "candidate" for experiment in experiments)
    assert all(experiment.stages == ("vocabulary",) for experiment in experiments)
    training = [experiment.training for experiment in experiments]
    assert all(isinstance(settings, InputTrainingSpec) for settings in training)
    input_training = [settings for settings in training if isinstance(settings, InputTrainingSpec)]
    assert all(settings.reconstruction_epochs == 0 for settings in input_training)
    assert all(settings.distillation_epochs == 0 for settings in input_training)


def test_alignment_feasibility_contract_rejects_historical_field_aliases() -> None:
    root = Path(__file__).parents[1]
    source = (root / "experiments/studies/input/qwen35-0.8b-alignment-feasibility.toml").read_text()
    source = source.replace("vocabulary_subset_sizes", "stages")
    with tempfile.TemporaryDirectory(
        dir=root / "experiments/studies/input",
    ) as directory:
        path = Path(directory) / "invalid.toml"
        path.write_text(source)
        with unittest.TestCase().assertRaisesRegex(ValueError, "unknown"):
            InputAlignmentFeasibilityStudySpec.load(path)


def test_alignment_feasibility_continues_after_each_passing_stage() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result, fit = _run_alignment_feasibility(
            Path(directory),
            [True] * 3,
        )
        report = (Path(directory) / "study/study-report.md").read_text()

    assert fit.call_count == 3
    assert [call.args[1].seed for call in fit.call_args_list] == [17] * 3
    assert [stage["status"] for stage in result["stages"]] == [
        "passed",
        "passed",
        "passed",
    ]
    assert all([seed["training_seed"] for seed in stage["seed_results"]] == [17] for stage in result["stages"])
    assert all({seed["subset_sha256"] for seed in stage["seed_results"]} == {stage["subset_sha256"]} for stage in result["stages"])
    assert result["training_seeds"] == [17]
    assert result["subset_seed"] == 17
    assert result["feasibility_passed"] is True
    assert result["full_model_evaluation_performed"] is False
    assert "Final-model verdict: **NONE" in report
    assert "Training seeds: `17`" in report
    assert report.count("Normalized RMSE") == 3


def test_alignment_feasibility_first_stage_failure_skips_remaining_work() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result, fit = _run_alignment_feasibility(
            Path(directory),
            [False],
        )

    assert fit.call_count == 1
    assert [stage["status"] for stage in result["stages"]] == [
        "failed_gate",
        "not_run_futility",
        "not_run_futility",
    ]
    assert result["stages"][0]["failed_gates"] == [
        {
            "training_seed": 17,
            "failed_gates": [
                {
                    "gate": "maximum_normalized_rmse",
                    "metric": "normalized_rmse",
                    "measured": 0.02,
                    "threshold": 0.01,
                    "comparison": "less_than_or_equal",
                },
            ],
        },
    ]
    assert all([seed["status"] for seed in stage["seed_results"]] == ["not_run_futility"] for stage in result["stages"][1:])
    assert result["feasibility_passed"] is False


def test_alignment_feasibility_later_failure_skips_only_later_work() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result, fit = _run_alignment_feasibility(
            Path(directory),
            [True, False],
        )

    assert fit.call_count == 2
    assert [stage["status"] for stage in result["stages"]] == [
        "passed",
        "failed_gate",
        "not_run_futility",
    ]
    assert result["stages"][2]["failed_prerequisite_stage"] == 256


def test_alignment_feasibility_resume_preserves_stage_decisions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial, _ = _run_alignment_feasibility(root, [False])
        (root / "study/result.json").unlink()
        (root / "study/manifest-final.json").unlink()
        study_path = Path(__file__).parents[1] / "experiments/studies/input/qwen35-0.8b-alignment-feasibility.toml"
        experiment = InputAlignmentFeasibilityStudySpec.load(
            study_path,
        ).load_experiment(study_path)
        assets = replace(
            pair_assets(root),
            model_id=experiment.model.model_id,
            revision=experiment.model.revision,
        )
        with (
            mock.patch.object(
                study_module,
                "load_model_assets",
                return_value=assets,
            ),
            mock.patch.object(
                study_module,
                "declared_device",
                return_value=torch.device("cpu"),
            ),
            mock.patch.object(
                study_module,
                "dependency_environment",
                return_value={"device": "cpu"},
            ),
            mock.patch.object(
                study_module,
                "registered_vocabulary_subset",
                side_effect=lambda _assets, rows, seed: VocabularySubset(
                    requested_rows=rows,
                    token_ids=(256,),
                    sha256=f"{seed:064x}",
                    algorithm="test",
                ),
            ),
            mock.patch.object(
                study_module,
                "fit_vocabulary_alignment",
            ) as fit,
        ):
            resumed = study_module.run_input_alignment_feasibility_study(
                study_path,
                root / "study",
                resume=True,
            )

    fit.assert_not_called()
    assert resumed["stages"] == initial["stages"]


def _compression_runtime(
    root: Path,
    *,
    resume: bool = False,
) -> study_module._CompressionRuntime:
    repository = Path(__file__).parents[1]
    study_path = repository / "experiments/studies/input/qwen35-0.8b-compression-feasibility.toml"
    study = InputCompressionFeasibilityStudySpec.load(study_path)
    experiment = study.load_experiment(study_path)
    assets = replace(
        pair_assets(root),
        model_id=experiment.model.model_id,
        revision=experiment.model.revision,
    )
    output = root / "compression-study"
    run = RunDirectory(output, resume=True) if output.exists() else RunDirectory(output)
    return study_module._CompressionRuntime(
        run=run,
        assets=assets,
        experiment=experiment,
        study=study,
        registered={
            "source_commit": "commit",
            "source_state_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
        },
        device=torch.device("cpu"),
        resume=resume,
    )


def _candidate_lengths(accepted: int) -> dict:
    metrics = {
        "2": {
            "accepted_spans": accepted,
            "reconstruction_fraction": float(accepted > 0),
            "candidates": 1,
            "bytes_per_position_with_atomic_fallback": 2.0 if accepted else 1.0,
        },
    }
    return {
        "candidate_lengths": [2, 8, 32],
        "source_dtype": "torch.float32",
        "vocabulary": {"metrics": metrics, "content_sha256": "c" * 64},
        "wikitext_validation": {"metrics": {}, "content_sha256": "d" * 64},
        "arbitrary_binary": {"metrics": {}, "content_sha256": "e" * 64},
    }


def _run_compression_mechanism(
    root: Path,
    *,
    round_trip: bool,
    accepted: int,
    resume: bool = False,
) -> tuple[dict, mock.Mock]:
    runtime = _compression_runtime(root, resume=resume)

    def train(_assets, options, *, device, resume_manager):
        del _assets, device, resume_manager
        options.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = options.output_dir / "large.pt"
        checkpoint.write_bytes(b"checkpoint")
        (options.output_dir / "training-result.json").write_text("{}\n")
        return TrainingResult(
            profile="large",
            optimizer={},
            checkpoint=str(checkpoint),
            compatibility_checkpoint=None,
            embedding_metrics=_alignment_metrics(passed=False),
            compatibility_embedding_metrics=_alignment_metrics(passed=False),
            candidate_state_bytes=1,
            reference_state_bytes=2,
            candidate_reference_state_ratio=0.5,
            native_tokens_per_continuous_token=1.2,
            round_trip=round_trip,
            compatibility_passed=False,
            alignment_preserved=False,
            passed=False,
            cache_metrics={},
        )

    train_mock = mock.Mock(side_effect=train)
    with (
        mock.patch.object(
            study_module,
            "registered_vocabulary_subset",
            return_value=VocabularySubset(
                requested_rows=INPUT_COMPRESSION_VOCABULARY_ROWS,
                token_ids=(256,),
                sha256="f" * 64,
                algorithm="test",
            ),
        ),
        mock.patch.object(study_module, "train_experiment", train_mock),
        mock.patch.object(
            study_module.InputEmbeddingAdapter,
            "from_checkpoint",
            return_value=SimpleNamespace(
                adapter=SimpleNamespace(codec=object()),
            ),
        ),
        mock.patch.object(
            study_module,
            "candidate_length_report",
            return_value=_candidate_lengths(accepted),
        ),
    ):
        result = study_module._run_compression_mechanism_stage(runtime)
    return result, train_mock


def test_checked_in_compression_feasibility_studies_are_equal_prospective_contracts() -> None:
    root = Path(__file__).parents[1]
    paths = sorted(
        (root / "experiments/studies/input").glob(
            "*-compression-feasibility.toml",
        ),
    )
    specs = [InputCompressionFeasibilityStudySpec.load(path) for path in paths]
    experiments = [spec.load_experiment(path) for spec, path in zip(specs, paths, strict=True)]

    assert len(specs) == 2
    assert all(spec.training_seeds == INPUT_COMPRESSION_TRAINING_SEEDS for spec in specs)
    assert all(spec.vocabulary_subset_size == INPUT_COMPRESSION_VOCABULARY_ROWS for spec in specs)
    assert all(spec.candidate_lengths == (2, 8, 32) for spec in specs)
    assert all(spec.binary_samples_per_length == 2 for spec in specs)
    assert all(spec.validation_bytes <= 512 for spec in specs)
    assert all(spec.continuation_rule == INPUT_COMPRESSION_CONTINUATION_RULE for spec in specs)
    assert all(spec.final_action == INPUT_COMPRESSION_FINAL_ACTION for spec in specs)
    assert all(spec.prospective for spec in specs)
    assert {experiment.model.model_id for experiment in experiments} == {
        "Qwen/Qwen3.5-0.8B",
        "google/gemma-3-270m-it",
    }
    assert all(experiment.training.profile == "large" for experiment in experiments)
    assert all(experiment.evidence_scope == "candidate" for experiment in experiments)
    assert all(experiment.stages == ("vocabulary", "reconstruction", "frozen_backbone_distillation") for experiment in experiments)


def test_compression_contract_rejects_historical_scaling_fields() -> None:
    root = Path(__file__).parents[1]
    source = (root / "experiments/studies/input/qwen35-0.8b-compression-feasibility.toml").read_text()
    source = source.replace("final_action", "selection_rule")
    with tempfile.TemporaryDirectory(
        dir=root / "experiments/studies/input",
    ) as directory:
        path = Path(directory) / "invalid.toml"
        path.write_text(source)
        with unittest.TestCase().assertRaisesRegex(ValueError, "unknown"):
            InputCompressionFeasibilityStudySpec.load(path)


def test_compression_alignment_failure_does_not_block_mechanism() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result, train = _run_compression_mechanism(
            Path(directory),
            round_trip=True,
            accepted=1,
        )

    assert train.call_count == 1
    assert result["status"] == "passed"
    assert all(seed["raw_metrics"]["alignment_failed_gates"] for seed in result["seed_results"])
    assert all(seed["raw_metrics"]["alignment_is_continuation_gate"] is False for seed in result["seed_results"])


def test_compression_exactness_failure_stops_after_all_mechanism_seeds() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result, train = _run_compression_mechanism(
            Path(directory),
            round_trip=False,
            accepted=1,
        )

    assert train.call_count == 1
    assert result["status"] == "failed_gate"
    assert all(seed["failed_gates"][0]["gate"] == "exact_byte_round_trip" for seed in result["seed_results"])


def test_compression_no_multibyte_acceptance_fails_mechanism() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result, _ = _run_compression_mechanism(
            Path(directory),
            round_trip=True,
            accepted=0,
        )

    assert result["status"] == "failed_gate"
    assert all(seed["failed_gates"][0]["gate"] == "minimum_multibyte_accepted_spans" for seed in result["seed_results"])


def _density_metrics(ratio: float, *, exact: bool = True) -> dict:
    return {
        "density": {
            "round_trip": exact,
            "native_tokens_per_continuous_token": ratio,
        },
        "density_strata": {
            "wikitext": {
                "windows": [{"empirical_round_trip": exact}],
            },
        },
        "embedding_fit": _alignment_metrics(passed=False),
    }


def test_compression_density_failure_aggregates_all_seeds_before_futility() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = _compression_runtime(root)
        seed_results = []
        for seed in INPUT_COMPRESSION_TRAINING_SEEDS:
            checkpoint = runtime.run.path(f"checkpoints/{seed}.pt")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            seed_results.append(
                {
                    "training_seed": seed,
                    "artifacts": {
                        "checkpoint": str(checkpoint.relative_to(runtime.run.root)),
                    },
                },
            )
        mechanism = {"seed_results": seed_results}
        metric_mock = mock.Mock(
            side_effect=(_density_metrics(1.0),),
        )
        with (
            mock.patch.object(
                study_module,
                "load_corpus_documents",
                return_value=[b"validation compression text"],
            ),
            mock.patch.object(
                study_module.InputEmbeddingAdapter,
                "from_checkpoint",
                return_value=SimpleNamespace(adapter=object()),
            ),
            mock.patch.object(study_module, "tokenizer_metrics", metric_mock),
        ):
            result = study_module._run_compression_density_stage(
                runtime,
                mechanism,
            )

    assert metric_mock.call_count == 1
    assert result["status"] == "failed_gate"
    assert result["failed_gates"][0]["training_seed"] == 17
    assert result["seed_results"][0]["failed_gates"][0]["gate"] == ("minimum_native_tokens_per_continuous_token")


def test_compression_behavior_persists_each_exact_failed_gate() -> None:
    failed, _ = study_module._compression_behavior_gates(
        {
            "teacher_forced": {
                "segmented": {
                    "mean_kl": 0.2,
                    "teacher_nll": 1.0,
                    "student_nll": 1.2,
                    "top1_agreement": 0.8,
                },
            },
            "generation": {
                "segmented_mean_byte_similarity": 0.4,
            },
        },
        study_module.InputGateSpec(),
    )

    assert {gate["gate"] for gate in failed} == {
        "maximum_segmented_mean_kl",
        "maximum_segmented_nll_delta",
        "minimum_segmented_top1_agreement",
        "minimum_segmented_generation_byte_similarity",
    }


def test_compression_ladder_continues_only_after_aggregate_passes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = _compression_runtime(Path(directory))
        mechanism = {"stage": "mechanism_exactness", "status": "passed"}
        density = {"stage": "held_out_density", "status": "passed"}
        behavior = {
            "stage": "candidate_behavior",
            "status": "passed",
            "selected_candidate": "reconstruction_only",
        }
        freeze = {"stage": "final_freeze_eligibility", "status": "passed"}
        with (
            mock.patch.object(
                study_module,
                "_run_compression_mechanism_stage",
                return_value=mechanism,
            ),
            mock.patch.object(
                study_module,
                "_run_compression_density_stage",
                return_value=density,
            ) as density_run,
            mock.patch.object(
                study_module,
                "load_frozen_causal_lm",
                return_value=object(),
            ),
            mock.patch.object(
                study_module,
                "_run_compression_behavior_stage",
                return_value=behavior,
            ) as behavior_run,
            mock.patch.object(
                study_module,
                "_compression_freeze_eligibility_stage",
                return_value=freeze,
            ) as freeze_run,
        ):
            stages = study_module._run_compression_ladder(runtime)

    density_run.assert_called_once()
    behavior_run.assert_called_once()
    freeze_run.assert_called_once()
    assert stages == [mechanism, density, behavior, freeze]


def test_compression_ladder_never_runs_behavior_after_density_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = _compression_runtime(Path(directory))
        mechanism = {"stage": "mechanism_exactness", "status": "passed"}
        density = {
            "stage": "held_out_density",
            "status": "failed_gate",
            "failed_gates": [{"gate": "density"}],
        }
        with (
            mock.patch.object(
                study_module,
                "_run_compression_mechanism_stage",
                return_value=mechanism,
            ),
            mock.patch.object(
                study_module,
                "_run_compression_density_stage",
                return_value=density,
            ),
            mock.patch.object(
                study_module,
                "_run_compression_behavior_stage",
            ) as behavior_run,
        ):
            stages = study_module._run_compression_ladder(runtime)

    behavior_run.assert_not_called()
    assert [stage["status"] for stage in stages] == [
        "passed",
        "failed_gate",
        "not_run_futility",
        "not_run_futility",
    ]


def test_compression_resume_reuses_hash_verified_seed_work() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial, _ = _run_compression_mechanism(
            root,
            round_trip=True,
            accepted=1,
        )
        resumed, train = _run_compression_mechanism(
            root,
            round_trip=True,
            accepted=1,
            resume=True,
        )

    train.assert_not_called()
    assert resumed == initial


def test_compression_result_records_eligibility_without_final_evidence() -> None:
    repository = Path(__file__).parents[1]
    study_path = repository / "experiments/studies/input/qwen35-0.8b-compression-feasibility.toml"
    experiment = InputCompressionFeasibilityStudySpec.load(
        study_path,
    ).load_experiment(study_path)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = replace(
            pair_assets(root),
            model_id=experiment.model.model_id,
            revision=experiment.model.revision,
        )

        def ladder(runtime):
            runtime.run.path("stages").mkdir()
            return [
                {
                    "stage": stage,
                    "status": "passed",
                    "reason": "passed",
                    "failed_gates": [],
                    "seed_results": [
                        {
                            "training_seed": seed,
                            "status": "passed",
                            "failed_gates": [],
                            "raw_metrics": {"passed": True},
                        }
                        for seed in INPUT_COMPRESSION_TRAINING_SEEDS
                    ],
                }
                for stage in study_module._COMPRESSION_STAGES
            ]

        with (
            mock.patch.object(
                study_module,
                "_registered_study",
                return_value={
                    "study": {},
                    "source_commit": "commit",
                    "source_dirty": True,
                    "source_state_sha256": "a" * 64,
                    "dependency_lock_sha256": "b" * 64,
                    "installed_package": {},
                },
            ),
            mock.patch.object(
                study_module,
                "load_model_assets",
                return_value=assets,
            ),
            mock.patch.object(
                study_module,
                "declared_device",
                return_value=torch.device("cpu"),
            ),
            mock.patch.object(
                study_module,
                "dependency_environment",
                return_value={"device": "cpu"},
            ),
            mock.patch.object(
                study_module,
                "_run_compression_ladder",
                side_effect=ladder,
            ),
        ):
            result = study_module.run_input_compression_feasibility_study(
                study_path,
                root / "result",
            )

        result_root = root / "result"
        assert result["feasibility_passed"] is True
        assert result["freeze_eligibility_recorded"] is True
        assert result["final_experiment_created"] is False
        assert result["freeze_performed"] is False
        assert result["final_claim_created"] is False
        assert result["final_evidence"] is False
        assert not (result_root / "final-experiment.toml").exists()
        assert "Final-model verdict: **NONE" in (result_root / "study-report.md").read_text()


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_checked_in_input_studies_are_strict_large_seed_17_contracts,
            test_input_study_contract_rejects_unregistered_candidate_lengths,
            test_vocabulary_and_binary_study_inputs_are_content_hashed,
            test_vocabulary_subset_excludes_aliases_and_stratifies_span_lengths,
            test_every_input_study_manifest_binds_source_and_dependency_provenance,
            test_candidate_length_report_covers_all_registered_sources_and_lengths,
            test_selection_uses_untouched_metrics_instead_of_aligned_construction,
            test_cli_registers_study_orchestration,
            test_input_study_resume_retains_hash_verified_completed_trial,
            test_checked_in_alignment_feasibility_studies_are_equal_prospective_contracts,
            test_alignment_feasibility_contract_rejects_historical_field_aliases,
            test_alignment_feasibility_continues_after_each_passing_stage,
            test_alignment_feasibility_first_stage_failure_skips_remaining_work,
            test_alignment_feasibility_later_failure_skips_only_later_work,
            test_alignment_feasibility_resume_preserves_stage_decisions,
            test_checked_in_compression_feasibility_studies_are_equal_prospective_contracts,
            test_compression_contract_rejects_historical_scaling_fields,
            test_compression_alignment_failure_does_not_block_mechanism,
            test_compression_exactness_failure_stops_after_all_mechanism_seeds,
            test_compression_no_multibyte_acceptance_fails_mechanism,
            test_compression_density_failure_aggregates_all_seeds_before_futility,
            test_compression_behavior_persists_each_exact_failed_gate,
            test_compression_ladder_continues_only_after_aggregate_passes,
            test_compression_ladder_never_runs_behavior_after_density_failure,
            test_compression_resume_reuses_hash_verified_seed_work,
            test_compression_result_records_eligibility_without_final_evidence,
        )
    )
