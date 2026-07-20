from __future__ import annotations

import argparse
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import torch
from input_training_fixtures import synthetic_assets

import continuous_tokenizer.campaigns.input as input_campaign_module
import continuous_tokenizer.commands.freeze as freeze_module
from continuous_tokenizer.artifacts.evidence import (
    _prospective_subset_run_errors,
    verify_artifact,
)
from continuous_tokenizer.artifacts.hashing import (
    installed_distribution_identity,
    sha256_file,
)
from continuous_tokenizer.artifacts.source import source_state
from continuous_tokenizer.artifacts.store import (
    load_json_object,
    write_json_atomic,
    write_text_atomic,
)
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.campaigns.dispatch import create_experiment_runner
from continuous_tokenizer.campaigns.input import InputExperimentRunner
from continuous_tokenizer.campaigns.lifecycle import (
    ProspectiveBudgetExhaustedError,
)
from continuous_tokenizer.campaigns.output import (
    OutputExperimentRunner,
    OutputRunnerOptions,
)
from continuous_tokenizer.contracts.claim_derivation import (
    FINAL_VERIFICATION_CHECKS,
)
from continuous_tokenizer.contracts.output import (
    OutputEvaluationSpec,
    OutputTrainingSpec,
)
from continuous_tokenizer.contracts.prospective import (
    PROSPECTIVE_NON_FINAL_FLAGS,
    CandidateOutcome,
    ProspectiveSpec,
    WallClockContract,
    futility_stages,
    prospective_budget,
    prospective_result_errors,
    select_smallest_candidate,
)
from continuous_tokenizer.contracts.prospective_subset import (
    PROSPECTIVE_INPUT_SUBSET_ALGORITHM,
    prospective_vocabulary_subset_errors,
)
from continuous_tokenizer.input.prospective import (
    ProspectiveExecutionPolicy,
    _bounded_experiment,
    _default_executor,
    _prospective_input_subset_request,
    _resume_state,
)
from continuous_tokenizer.input.prospective import (
    _outcome as _campaign_outcome,
)
from continuous_tokenizer.input.studies import (
    RegisteredVocabularySubsetRequest,
    registered_vocabulary_subset,
)
from continuous_tokenizer.input.training.run import TrainingResult
from continuous_tokenizer.input.training.vocabulary_batches import (
    build_vocabulary_batches,
    build_vocabulary_groups,
)
from continuous_tokenizer.reporting.discovery import (
    discover_artifact_runs,
    discover_prospective_artifacts,
)

ROOT = Path(__file__).parents[1]
PROSPECTIVE_ROOT = ROOT / "experiments/prospective"
_INPUT_SCREEN_ROWS = prospective_budget(
    "feasibility_screen",
    "input_only",
)["vocabulary_rows"]


def _outcome(**changes: object) -> CandidateOutcome:
    outcome = CandidateOutcome(
        name="smallest",
        kind="efficiency",
        operational_passed=True,
        invariant_passed=True,
        exactness_passed=True,
        density_passed=True,
        behavior_passed=True,
        compactness_passed=True,
    )
    return replace(outcome, **changes)


def _result(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "prospective_feasibility_screen",
        "tier": "feasibility_screen",
        "name": "screen",
        "mode": "input_only",
        "operational_status": "completed",
        "scientific_verdict": "unsupported",
        "spec_fingerprint": "a" * 64,
        "budget_exhausted": False,
        "wall_clock": {},
        "stages": [],
        "selection": {},
        **PROSPECTIVE_NON_FINAL_FLAGS,
    }
    result.update(updates)
    return result


def _assets_with_compatibility_rows(
    root: Path,
    rows: int = _INPUT_SCREEN_ROWS,
):
    assets = synthetic_assets(root)
    payloads = tuple(
        (index.to_bytes(2, "big") if index % 3 == 0 else (b"x" + index.to_bytes(2, "big") if index % 3 == 1 else b"yzz" + index.to_bytes(2, "big")))
        for index in range(rows)
    )
    token_bytes = (*assets.vocabulary.token_bytes, *payloads)
    token_ids = tuple(range(len(token_bytes)))
    return replace(
        assets,
        input_embeddings=torch.cat(
            (
                assets.input_embeddings,
                torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8),
            ),
        ),
        vocabulary=ByteVocabulary(
            token_bytes=token_bytes,
            ordinary_ids=token_ids,
            control_ids=(),
            byte_token_ids=assets.vocabulary.byte_token_ids,
            max_token_bytes=5,
            compatibility_ids=token_ids,
        ),
    )


def _subset_request(
    spec: ProspectiveSpec,
    subset_sha256: str,
) -> RegisteredVocabularySubsetRequest:
    return RegisteredVocabularySubsetRequest(
        requested_rows=int(spec.design["vocabulary_rows"]),
        subset_seed=int(spec.design["subset_seed"]),
        subset_sha256=subset_sha256,
        algorithm=PROSPECTIVE_INPUT_SUBSET_ALGORITHM,
        work_units=tuple(sorted(spec.wall_clock.work_units.items())),
    )


class ProspectiveTierTests(unittest.TestCase):
    def test_registered_wrappers_are_strict_and_bounded(self) -> None:
        paths = sorted(PROSPECTIVE_ROOT.glob("*/*/*.toml"))
        self.assertEqual(len(paths), 12)
        specs = [ProspectiveSpec.load(path) for path in paths]
        self.assertEqual(
            {spec.tier for spec in specs},
            {"mechanism_smoke", "feasibility_screen", "candidate_selection"},
        )
        self.assertEqual(
            {(spec.mode, spec.load_final_reference().model.model_id) for spec in specs},
            {
                ("input_only", "Qwen/Qwen3.5-0.8B"),
                ("output_only", "Qwen/Qwen3.5-0.8B"),
                ("input_only", "google/gemma-3-270m-it"),
                ("output_only", "google/gemma-3-270m-it"),
            },
        )
        screens = [spec for spec in specs if spec.tier == "feasibility_screen"]
        self.assertEqual(len(screens), 4)
        for spec in screens:
            budget = prospective_budget("feasibility_screen", spec.mode)
            self.assertTrue(
                all(spec.design[name] == value for name, value in budget.items()),
            )
            self.assertEqual(spec.load_experiment().seed, 17)
            self.assertEqual(spec.load_experiment().training.profile, "large")
        selections = [spec for spec in specs if spec.tier == "candidate_selection"]
        for spec in selections:
            self.assertEqual(spec.design["data_role"], "validation")
            self.assertFalse(spec.design["load_final_test"])
            self.assertEqual(
                sum(candidate.kind == "alignment" for candidate in spec.candidates()),
                1,
            )
            self.assertEqual(
                sum(candidate.kind == "efficiency" for candidate in spec.candidates()),
                2,
            )

    def test_unknown_wrapper_field_is_rejected(self) -> None:
        source = (PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml").read_text()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "unknown.toml"
            path.write_text(
                source.replace(
                    "schema_version = 1",
                    "schema_version = 1\nunknown = true",
                ),
            )
            with self.assertRaisesRegex(ValueError, "unknown prospective wrapper"):
                ProspectiveSpec.load(path)

    def test_suffixed_artifact_kind_is_rejected(self) -> None:
        source = (PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml").read_text()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "old-kind.toml"
            path.write_text(
                source.replace(
                    'artifact_kind = "prospective_feasibility_screen"',
                    'artifact_kind = "prospective_feasibility_screen_v1"',
                ),
            )
            with self.assertRaisesRegex(ValueError, "artifact_kind"):
                ProspectiveSpec.load(path)

    def test_prospective_contracts_have_no_compatibility_symbols(self) -> None:
        paths = (
            ROOT / "src/continuous_tokenizer/contracts/prospective.py",
            ROOT / "src/continuous_tokenizer/contracts/prospective_subset.py",
            ROOT / "src/continuous_tokenizer/input/prospective.py",
            ROOT / "src/continuous_tokenizer/commands/freeze.py",
            *PROSPECTIVE_ROOT.rglob("*.toml"),
            *PROSPECTIVE_ROOT.rglob("*.json"),
        )
        for path in paths:
            with self.subTest(path=path):
                contents = path.read_text()
                self.assertNotIn("_v1", contents)
                self.assertNotIn("legacy_batch_size", contents)
                self.assertNotIn("prospective/v1", contents)

    def test_screen_uses_exact_registered_subset_and_bounded_batches(self) -> None:
        wrapper = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        experiment = replace(_bounded_experiment(wrapper), device="cpu")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            assets = _assets_with_compatibility_rows(root)
            subset = registered_vocabulary_subset(
                assets,
                _INPUT_SCREEN_ROWS,
                17,
            )
            request = _subset_request(wrapper, subset.sha256)
            with mock.patch(
                "continuous_tokenizer.campaigns.lifecycle.require_storage",
            ):
                runner = create_experiment_runner(
                    experiment,
                    root / "run",
                    ROOT,
                    prospective_input_subset=request,
                )
            self.assertIsInstance(runner, InputExperimentRunner)
            runner = cast(InputExperimentRunner, runner)
            artifacts: dict[str, str] = {}
            runner._prepare_prospective_subset(assets, artifacts)

            options = runner._training_options()
            self.assertEqual(options.vocabulary_token_ids, subset.token_ids)
            self.assertEqual(
                len(options.vocabulary_token_ids or ()),
                _INPUT_SCREEN_ROWS,
            )
            with (
                mock.patch.object(
                    input_campaign_module,
                    "train_experiment",
                    side_effect=RuntimeError("captured training options"),
                ) as train,
                self.assertRaisesRegex(
                    RuntimeError,
                    "captured training options",
                ),
            ):
                runner._train_tokenizer(assets, artifacts)
            trained_options = train.call_args.args[1]
            self.assertEqual(
                trained_options.vocabulary_token_ids,
                subset.token_ids,
            )

            artifact = load_json_object(
                runner.run_directory.root / "prospective-vocabulary-subset.json",
            )
            groups = build_vocabulary_groups(assets, subset.token_ids)
            actual_batches = build_vocabulary_batches(
                groups,
                experiment.training.batch_size,
                torch.Generator().manual_seed(17),
            )
            expected_batches = sum((len(group.token_ids) + experiment.training.batch_size - 1) // experiment.training.batch_size for group in groups)
            self.assertEqual(len(actual_batches), expected_batches)
            self.assertEqual(
                artifact["vocabulary_batches_per_epoch"],
                expected_batches,
            )
            self.assertEqual(
                prospective_vocabulary_subset_errors(artifact),
                [],
            )
            write_json_atomic(
                runner.run_directory.root / "result.json",
                {
                    "prospective_vocabulary_subset": (runner.prospective_subset_descriptor),
                },
            )
            semantic_manifest = SimpleNamespace(
                mode="input_only",
                artifacts={
                    "prospective_vocabulary_subset": ("prospective-vocabulary-subset.json"),
                    "result": "result.json",
                },
                inputs=runner.inputs,
            )
            self.assertEqual(
                _prospective_subset_run_errors(
                    runner.run_directory.root / "manifest-final.json",
                    semantic_manifest,
                ),
                [],
            )
            tampered = dict(artifact)
            tampered["rows"] = [*artifact["rows"]]
            tampered["rows"][0] = {
                **tampered["rows"][0],
                "bytes": "ffff",
            }
            self.assertIn(
                "prospective vocabulary subset content hash mismatch",
                prospective_vocabulary_subset_errors(tampered),
            )

    def test_final_mechanism_and_output_paths_do_not_subset(self) -> None:
        screen = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        mechanism = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/mechanism-smoke.toml",
        )
        output = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "output/qwen35-0.8b/feasibility-screen.toml",
        )
        self.assertIsNone(_prospective_input_subset_request(mechanism))
        self.assertIsNone(
            _prospective_input_subset_request(
                replace(screen, tier="final_evidence"),
            ),
        )
        self.assertIsNone(_prospective_input_subset_request(output))
        experiment = replace(_bounded_experiment(screen), device="cpu")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            with mock.patch(
                "continuous_tokenizer.campaigns.lifecycle.require_storage",
            ):
                runner = create_experiment_runner(
                    experiment,
                    Path(directory) / "standard",
                    ROOT,
                )
            runner = cast(InputExperimentRunner, runner)
            self.assertIsNone(runner._training_options().vocabulary_token_ids)

    def test_prospective_executor_passes_only_declared_input_subset(self) -> None:
        screen = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        output = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "output/qwen35-0.8b/feasibility-screen.toml",
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            child = Path(directory) / "child"
            with mock.patch(
                "continuous_tokenizer.input.prospective.create_experiment_runner",
            ) as create:
                create.return_value.run.return_value = {}
                _default_executor(None, screen)(
                    _bounded_experiment(screen),
                    child,
                    False,
                )
                request = create.call_args.kwargs["prospective_input_subset"]
                self.assertIsInstance(
                    request,
                    RegisteredVocabularySubsetRequest,
                )
                self.assertEqual(
                    request.requested_rows,
                    _INPUT_SCREEN_ROWS,
                )
                policy = create.call_args.kwargs["prospective_execution_policy"]
                self.assertIsInstance(
                    policy,
                    ProspectiveExecutionPolicy,
                )
                self.assertTrue(policy.futility_enabled)
                self.assertEqual(
                    _bounded_experiment(screen).evaluation.batch_size,
                    8,
                )

                _default_executor(None, output)(
                    _bounded_experiment(output),
                    child,
                    False,
                )
                self.assertIsNone(
                    create.call_args.kwargs["prospective_input_subset"],
                )

    def test_qwen_zero_reconstruction_stops_before_distillation(self) -> None:
        wrapper = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        experiment = replace(_bounded_experiment(wrapper), device="cpu")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            assets = replace(
                _assets_with_compatibility_rows(root),
                model_id=experiment.model.model_id,
                revision=experiment.model.revision,
            )
            subset = registered_vocabulary_subset(
                assets,
                _INPUT_SCREEN_ROWS,
                17,
            )
            request = _subset_request(wrapper, subset.sha256)
            result_path = root / "run"
            policy = ProspectiveExecutionPolicy.from_spec(wrapper)
            training = TrainingResult(
                profile="large",
                optimizer={},
                checkpoint=str(result_path / "checkpoints/large.pt"),
                compatibility_checkpoint=None,
                embedding_metrics={"reconstruction_fraction": 0.0},
                compatibility_embedding_metrics={
                    "reconstruction_fraction": 0.0,
                },
                candidate_state_bytes=1,
                reference_state_bytes=2,
                candidate_reference_state_ratio=0.5,
                native_tokens_per_continuous_token=1.0,
                round_trip=False,
                compatibility_passed=False,
                alignment_preserved=False,
                passed=False,
                cache_metrics={},
            )

            def preflight(
                runner: InputExperimentRunner,
                _assets: object,
                *,
                load_full_model: bool = True,
            ):
                self.assertFalse(load_full_model)
                artifact = {"all_passed": True}
                write_json_atomic(
                    runner.run_directory.root / "preflight.json",
                    artifact,
                )
                return artifact, None

            with (
                mock.patch(
                    "continuous_tokenizer.campaigns.lifecycle.require_storage",
                ),
                mock.patch.object(
                    input_campaign_module,
                    "load_model_assets",
                    return_value=assets,
                ),
                mock.patch.object(
                    InputExperimentRunner,
                    "_run_preflight",
                    autospec=True,
                    side_effect=preflight,
                ),
                mock.patch.object(
                    input_campaign_module,
                    "train_experiment",
                    return_value=training,
                ),
                mock.patch.object(
                    input_campaign_module,
                    "load_frozen_causal_lm",
                ) as load_full_model,
                mock.patch.object(
                    input_campaign_module,
                    "distill_checkpoint",
                ) as distill,
                mock.patch.object(
                    input_campaign_module,
                    "benchmark_experiment",
                ) as benchmark,
                mock.patch.object(
                    input_campaign_module,
                    "evaluate_input_replacement",
                ) as evaluate,
            ):
                result = create_experiment_runner(
                    experiment,
                    result_path,
                    ROOT,
                    prospective_input_subset=request,
                    prospective_execution_policy=policy,
                ).run()

            load_full_model.assert_not_called()
            distill.assert_not_called()
            benchmark.assert_not_called()
            evaluate.assert_not_called()
            self.assertEqual(result["operational_status"], "completed")
            self.assertEqual(result["scientific_verdict"], "unsupported")
            self.assertFalse(result["gates_passed"])
            self.assertEqual(
                result["prospective_execution"]["stages"][2]["status"],
                "not_run_futility",
            )
            self.assertEqual(
                verify_artifact(result_path)["errors"],
                [],
            )
            self.assertEqual(
                [artifact.directory for artifact in discover_artifact_runs(root)],
                [result_path],
            )
            manifest = load_json_object(result_path / "manifest-final.json")
            self.assertEqual(manifest["status"], "passed")

    def test_budget_expires_at_epoch_boundary_and_final_is_unchanged(
        self,
    ) -> None:
        wrapper = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        policy = ProspectiveExecutionPolicy.from_spec(wrapper, started=10.0)
        with (
            mock.patch(
                "continuous_tokenizer.input.prospective.perf_counter",
                return_value=policy.deadline,
            ),
            self.assertRaises(ProspectiveBudgetExhaustedError) as raised,
        ):
            policy.enforce_boundary("epoch:input_vocabulary_alignment:1")
        self.assertEqual(
            raised.exception.boundary,
            "epoch:input_vocabulary_alignment:1",
        )

        final_policy = ProspectiveExecutionPolicy.from_spec(
            replace(wrapper, tier="final_evidence"),
            started=10.0,
        )
        self.assertFalse(final_policy.futility_enabled)
        self.assertEqual(wrapper.load_experiment().evaluation.batch_size, 8)
        with mock.patch(
            "continuous_tokenizer.input.prospective.perf_counter",
            return_value=final_policy.deadline + 1,
        ):
            final_policy.enforce_boundary("epoch:final:1")

    def test_density_failure_skips_input_behavior(self) -> None:
        wrapper = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        experiment = replace(_bounded_experiment(wrapper), device="cpu")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            assets = replace(
                synthetic_assets(root),
                model_id=experiment.model.model_id,
                revision=experiment.model.revision,
            )
            with mock.patch(
                "continuous_tokenizer.campaigns.lifecycle.require_storage",
            ):
                runner = InputExperimentRunner(
                    experiment,
                    root / "run",
                    ROOT,
                    prospective_policy=(ProspectiveExecutionPolicy.from_spec(wrapper)),
                )
            checkpoint = root / "run/checkpoints/large.pt"
            selection = input_campaign_module._SelectionState(
                checkpoint=checkpoint,
                trainable=(),
                distillation={},
                input_selection=None,
                ablations={},
            )
            tokenizer = {
                "acceptance": {
                    "embedding_fit": True,
                    "density": False,
                    "compactness": True,
                },
            }
            with (
                mock.patch.object(
                    input_campaign_module,
                    "benchmark_experiment",
                    return_value=tokenizer,
                ),
                mock.patch.object(
                    input_campaign_module,
                    "evaluate_input_replacement",
                ) as evaluate,
            ):
                _, llm, _, reason = runner._measure_selected(
                    assets,
                    None,
                    selection,
                    {},
                )
            self.assertIsNone(llm)
            self.assertEqual(reason, "density")
            evaluate.assert_not_called()

        outcome = _campaign_outcome(
            "candidate",
            "efficiency",
            {
                "operational_status": "completed",
                "gates": {
                    "exact_compatibility": {
                        "status": "passed",
                        "passed": True,
                    },
                    "embedding_alignment": {
                        "status": "passed",
                        "passed": True,
                    },
                    "exact_round_trip": {
                        "status": "passed",
                        "passed": True,
                    },
                    "held_out_density": {
                        "status": "failed",
                        "passed": False,
                    },
                    "behavioral_similarity": {
                        "status": "not_run_futility",
                        "passed": None,
                    },
                },
            },
            False,
        )
        self.assertTrue(outcome.exactness_passed)
        self.assertFalse(outcome.density_passed)
        self.assertIsNone(outcome.behavior_passed)
        self.assertEqual(
            futility_stages(outcome),
            {
                "exactness": "passed",
                "density": "failed",
                "behavior": "not_run_futility",
            },
        )

    def test_output_oracle_stop_seals_completed_child(self) -> None:
        wrapper = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "output/qwen35-0.8b/feasibility-screen.toml",
        )
        experiment = replace(_bounded_experiment(wrapper), device="cpu")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            assets = replace(
                synthetic_assets(root),
                model_id=experiment.model.model_id,
                revision=experiment.model.revision,
            )
            with mock.patch(
                "continuous_tokenizer.campaigns.lifecycle.require_storage",
            ):
                runner = OutputExperimentRunner(
                    experiment,
                    root / "run",
                    ROOT,
                    OutputRunnerOptions(
                        prospective_policy=(ProspectiveExecutionPolicy.from_spec(wrapper)),
                    ),
                )
            runner._write_experiment_contract()
            runner._write_start_manifest(assets)
            with mock.patch.object(
                runner,
                "_train_output_codec",
            ) as train:
                result = runner._publish_prospective_stop(
                    assets,
                    {},
                    stop_reason="oracle",
                    boundary="stage:prepare_output_trajectories",
                    output_metrics={
                        "oracle_feasible": False,
                        "native_head_oracle_ceilings": {},
                        "training_performed": False,
                    },
                )
            train.assert_not_called()
            self.assertEqual(
                result["prospective_execution"]["stages"][1]["status"],
                "not_run_futility",
            )
            self.assertEqual(
                verify_artifact(root / "run")["errors"],
                [],
            )

    def test_subset_hash_and_work_unit_tampering_fail_before_training(self) -> None:
        wrapper = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        experiment = replace(_bounded_experiment(wrapper), device="cpu")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            assets = _assets_with_compatibility_rows(root)
            assets = replace(
                assets,
                model_id=experiment.model.model_id,
                revision=experiment.model.revision,
            )
            bad_hash = _subset_request(wrapper, "0" * 64)
            with (
                mock.patch(
                    "continuous_tokenizer.campaigns.lifecycle.require_storage",
                ),
                mock.patch.object(
                    input_campaign_module,
                    "load_model_assets",
                    return_value=assets,
                ),
                mock.patch.object(
                    input_campaign_module,
                    "train_experiment",
                ) as train,
            ):
                runner = create_experiment_runner(
                    experiment,
                    root / "hash-run",
                    ROOT,
                    prospective_input_subset=bad_hash,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "differs from its registered contract",
                ):
                    runner.run()
            train.assert_not_called()

            subset = registered_vocabulary_subset(
                assets,
                _INPUT_SCREEN_ROWS,
                17,
            )
            request = _subset_request(wrapper, subset.sha256)
            wrong_units = replace(
                request,
                work_units=tuple((name, value - 1 if name == "vocabulary_rows" else value) for name, value in request.work_units),
            )
            with mock.patch(
                "continuous_tokenizer.campaigns.lifecycle.require_storage",
            ):
                bounded = create_experiment_runner(
                    experiment,
                    root / "units-run",
                    ROOT,
                    prospective_input_subset=wrong_units,
                )
            bounded = cast(InputExperimentRunner, bounded)
            with self.assertRaisesRegex(
                ValueError,
                "work units differ",
            ):
                bounded._prepare_prospective_subset(assets, {})

    def test_resume_identity_includes_prospective_subset(self) -> None:
        wrapper = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        experiment = replace(_bounded_experiment(wrapper), device="cpu")
        request = _prospective_input_subset_request(wrapper)
        if request is None:
            self.fail("expected a prospective subset request")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            with mock.patch(
                "continuous_tokenizer.campaigns.lifecycle.require_storage",
            ):
                standard = create_experiment_runner(
                    experiment,
                    root / "standard",
                    ROOT,
                )
                prospective = create_experiment_runner(
                    experiment,
                    root / "prospective",
                    ROOT,
                    prospective_input_subset=request,
                )
                changed = create_experiment_runner(
                    experiment,
                    root / "changed",
                    ROOT,
                    prospective_input_subset=replace(
                        request,
                        subset_sha256="0" * 64,
                    ),
                )
        self.assertEqual(
            standard.resume_manager.experiment_fingerprint,
            experiment.fingerprint(),
        )
        self.assertNotEqual(
            prospective.resume_manager.experiment_fingerprint,
            standard.resume_manager.experiment_fingerprint,
        )
        self.assertNotEqual(
            prospective.resume_manager.experiment_fingerprint,
            changed.resume_manager.experiment_fingerprint,
        )

    def test_all_futility_branches(self) -> None:
        self.assertEqual(
            set(
                futility_stages(
                    _outcome(operational_passed=False),
                ).values(),
            ),
            {"not_run_operational_failure"},
        )
        self.assertEqual(
            futility_stages(_outcome(exactness_passed=False)),
            {
                "exactness": "failed",
                "density": "not_run_futility",
                "behavior": "not_run_futility",
            },
        )
        self.assertEqual(
            futility_stages(_outcome(density_passed=False)),
            {
                "exactness": "passed",
                "density": "failed",
                "behavior": "not_run_futility",
            },
        )
        self.assertEqual(
            futility_stages(_outcome(behavior_passed=False))["behavior"],
            "failed",
        )

    def test_selection_is_ordered_smallest_and_alignment_is_independent(self) -> None:
        spec = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/candidate-selection.toml",
        )
        candidates = spec.candidates()
        alignment = replace(_outcome(), name=candidates[0].name, kind="alignment")
        first = replace(_outcome(), name="smallest", exactness_passed=False)
        second = replace(_outcome(), name="largest")
        selected, feasible = select_smallest_candidate(
            candidates,
            (alignment, first, second),
        )
        self.assertTrue(feasible)
        if selected is None:
            self.fail("expected the larger candidate")
        self.assertEqual(selected.name, "largest")
        selected, feasible = select_smallest_candidate(candidates, (alignment, first))
        self.assertIsNone(selected)
        self.assertFalse(feasible)

    def test_output_work_units_match_the_bounded_validation_only_child(self) -> None:
        spec = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "output/qwen35-0.8b/feasibility-screen.toml",
        )
        bounded = _bounded_experiment(spec)
        self.assertIsInstance(bounded.training, OutputTrainingSpec)
        self.assertIsInstance(bounded.evaluation, OutputEvaluationSpec)
        training = cast(OutputTrainingSpec, bounded.training)
        evaluation = cast(OutputEvaluationSpec, bounded.evaluation)
        self.assertEqual(bounded.runtime.corpus_max_rows, 256)
        self.assertEqual(training.epochs, 1)
        self.assertEqual(evaluation.samples, 2)
        self.assertEqual(evaluation.max_output_bytes, 256)
        tampered = replace(
            spec,
            wall_clock=replace(
                spec.wall_clock,
                work_units={**spec.wall_clock.work_units, "behavior_samples": 1},
            ),
        )
        with self.assertRaisesRegex(ValueError, "work units differ"):
            _bounded_experiment(tampered)

    def test_budget_stops_only_at_boundary_and_never_passes(self) -> None:
        spec = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml",
        )
        wall_clock: WallClockContract = spec.wall_clock
        self.assertFalse(
            wall_clock.exhausted_at_boundary(
                wall_clock.maximum_seconds + 1,
                at_boundary=False,
            ),
        )
        self.assertTrue(
            wall_clock.exhausted_at_boundary(
                wall_clock.maximum_seconds,
                at_boundary=True,
            ),
        )
        errors = prospective_result_errors(
            _result(
                budget_exhausted=True,
                scientific_verdict="supported",
            ),
        )
        self.assertIn("budget-exhausted prospective work can never pass", errors)

    def test_non_final_claim_promotion_is_rejected(self) -> None:
        errors = prospective_result_errors(
            _result(final_claims_allowed=True),
        )
        self.assertIn("non-final prospective result promotes a final claim", errors)

    def test_discovery_accepts_only_current_prospective_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "screen"
            artifact.mkdir()
            result_path = artifact / "prospective.json"
            write_json_atomic(result_path, _result())
            with (
                mock.patch(
                    "continuous_tokenizer.reporting.discovery.verify_artifact",
                    return_value={"valid": True},
                ),
                mock.patch(
                    "continuous_tokenizer.reporting.discovery.load_evidence_manifest",
                    return_value={
                        "artifact_kind": "prospective_feasibility_screen",
                        "model": {"id": "model"},
                    },
                ),
            ):
                discovered = discover_prospective_artifacts(root)
                self.assertEqual(
                    [value.directory.resolve() for value in discovered],
                    [artifact.resolve()],
                )
                write_json_atomic(
                    result_path,
                    _result(
                        artifact_kind="prospective_feasibility_screen_v1",
                    ),
                )
                self.assertEqual(discover_prospective_artifacts(root), ())

    def test_resume_is_bound_to_wrapper_fingerprint(self) -> None:
        spec = ProspectiveSpec.load(
            PROSPECTIVE_ROOT / "input/qwen35-0.8b/candidate-selection.toml",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_json_atomic(
                output / "prospective-resume.json",
                {
                    "schema_version": 1,
                    "spec_fingerprint": spec.fingerprint(),
                    "completed": ["alignment-reconstruction"],
                },
            )
            state = _resume_state(spec, output, resume=True)
            self.assertEqual(state["completed"], ["alignment-reconstruction"])
            write_json_atomic(
                output / "prospective-resume.json",
                {
                    "schema_version": 1,
                    "spec_fingerprint": "0" * 64,
                    "completed": [],
                },
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                _resume_state(spec, output, resume=True)

    def test_calibration_and_subset_tampering_are_rejected(self) -> None:
        source = (PROSPECTIVE_ROOT / "input/qwen35-0.8b/feasibility-screen.toml").read_text()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            calibration = root / "calibration.json"
            calibration.write_text("{}")
            tampered = source.replace(
                "../../calibration/qwen35-0.8b-input.json",
                "calibration.json",
            )
            path = root / "tampered.toml"
            path.write_text(tampered)
            with self.assertRaisesRegex(ValueError, "calibration"):
                ProspectiveSpec.load(path)

    def test_output_oracle_infeasibility_is_completed_unsupported(self) -> None:
        result = _result(
            artifact_kind="prospective_candidate_selection",
            tier="candidate_selection",
            mode="output_only",
            operational_status="completed",
            scientific_verdict="unsupported",
            selection={"selection_feasible": False},
        )
        self.assertEqual(prospective_result_errors(result), [])

    def test_freeze_generates_explicit_final_three_seed_specs(self) -> None:
        import tomli_w

        entries = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, direction in (
                ("input_only", "input"),
                ("output_only", "output"),
            ):
                for model_id, slug in (
                    ("Qwen/Qwen3.5-0.8B", "qwen35-0.8b"),
                    ("google/gemma-3-270m-it", "gemma3-270m"),
                ):
                    spec = ProspectiveSpec.load(
                        PROSPECTIVE_ROOT / direction / slug / "candidate-selection.toml",
                    )
                    template = replace(
                        spec.load_final_reference(),
                        device="cpu",
                        runtime=replace(
                            spec.load_final_reference().runtime,
                            projected_run_bytes=1,
                            storage_reserve_bytes=1,
                            inductor_cache_estimate_bytes=1,
                            minimum_mps_memory_bytes=1,
                        ),
                    )
                    template_path = root / f"{direction}-{slug}.experiment.toml"
                    write_text_atomic(
                        template_path,
                        tomli_w.dumps(template.to_toml_dict()),
                    )
                    candidate_path = root / f"{direction}-{slug}.candidate.toml"
                    candidate_path.write_bytes(spec.path.read_bytes())
                    calibration_source = (spec.path.parent / spec.wall_clock.calibration.locator).resolve()
                    calibration_path = root / f"{direction}-{slug}.calibration.json"
                    calibration_path.write_bytes(calibration_source.read_bytes())
                    selection_spec = replace(
                        spec,
                        path=candidate_path,
                        experiment=template_path.name,
                        final_reference=template_path.name,
                        wall_clock=replace(
                            spec.wall_clock,
                            calibration=replace(
                                spec.wall_clock.calibration,
                                locator=calibration_path.name,
                            ),
                        ),
                    )
                    result_path = root / f"{direction}-{slug}.json"
                    configuration = (
                        {
                            "strategy": "reconstruction_only",
                            "batch_size": 64,
                            "projection_multiplier": 8,
                            "muon_ns_steps": 3,
                        }
                        if mode == "input_only"
                        else {"max_span": 8, "batch_size": 64}
                    )
                    selection = {
                        "selection_feasible": True,
                        "selected_candidate": "selected",
                        "selected_strategy": ("reconstruction_only" if mode == "input_only" else None),
                        "selected_configuration": configuration,
                        "alignment": [],
                        "validation_only": True,
                        "final_test_loaded": False,
                    }
                    prospective_result = _result(
                        artifact_kind="prospective_candidate_selection",
                        tier="candidate_selection",
                        mode=mode,
                        operational_status="completed",
                        spec_fingerprint=selection_spec.fingerprint(),
                        budget_exhausted=False,
                        selection=selection,
                    )
                    write_json_atomic(result_path, prospective_result)
                    entries[(model_id, mode)] = (
                        result_path,
                        prospective_result,
                        selection_spec,
                    )
            output = root / "frozen"
            commit, _, state_sha256 = source_state(ROOT)
            identity = freeze_module._FreezeIdentity(
                commit,
                state_sha256,
                sha256_file(ROOT / "uv.lock"),
                installed_distribution_identity("continuous-byte-tokenizer"),
            )
            verification_log = root / "verification.log"
            verification_log.write_text("passed\n")
            verification_path = root / "verification.json"
            write_json_atomic(
                verification_path,
                {
                    "kind": "complete_verification",
                    "source_commit": commit,
                    "source_state_sha256": state_sha256,
                    "dependency_lock_sha256": sha256_file(ROOT / "uv.lock"),
                    "all_passed": True,
                    "checks": {
                        name: {
                            "passed": True,
                            "log": verification_log.name,
                            "log_sha256": sha256_file(verification_log),
                        }
                        for name in FINAL_VERIFICATION_CHECKS
                    },
                },
            )
            with mock.patch.object(
                freeze_module,
                "_prospective_candidate_directories",
                return_value=entries,
            ):
                result = freeze_module._freeze_prospective(
                    argparse.Namespace(artifacts=[], output_dir=output),
                    tomli_w.dumps,
                    identity,
                )

            paths = [Path(path) for path in result["specifications"]]
            self.assertEqual(len(paths), 12)
            wrappers = [ProspectiveSpec.load(path) for path in paths]
            self.assertEqual(
                {wrapper.tier for wrapper in wrappers},
                {"final_evidence"},
            )
            for mode in ("input_only", "output_only"):
                for model_id in (
                    "Qwen/Qwen3.5-0.8B",
                    "google/gemma-3-270m-it",
                ):
                    specs = [wrapper for wrapper in wrappers if wrapper.mode == mode and wrapper.load_experiment().model.model_id == model_id]
                    self.assertEqual(
                        [spec.load_experiment().seed for spec in specs],
                        [17, 23, 41],
                    )
                    self.assertTrue(
                        all(spec.load_experiment().evidence_scope == "final" for spec in specs),
                    )
            for wrapper in wrappers:
                if wrapper.load_experiment().seed != 17:
                    continue
                policy = ProspectiveExecutionPolicy.from_spec(wrapper)
                runner = create_experiment_runner(
                    wrapper.load_experiment(),
                    root / "runs" / wrapper.name,
                    ROOT,
                    verification_path,
                    prospective_execution_policy=policy,
                )
                runner._enforce_prospective_boundary("stage:preflight")
                self.assertFalse(policy.futility_enabled)


if __name__ == "__main__":
    unittest.main()
