from __future__ import annotations

import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

from torch import nn

from continuous_tokenizer.artifacts.hashing import (
    installed_distribution_identity,
    sha256_file,
    sha256_path,
)
from continuous_tokenizer.artifacts.manifest import load_artifact
from continuous_tokenizer.artifacts.source import source_state
from continuous_tokenizer.artifacts.store import RunDirectory, load_json_object
from continuous_tokenizer.backbone.assets import ModelAssets, load_frozen_causal_lm
from continuous_tokenizer.backbone.synthetic import SYNTHETIC_MODEL_ID
from continuous_tokenizer.codec.layers import gqa_metadata
from continuous_tokenizer.contracts.claim_derivation import (
    FINAL_VERIFICATION_CHECKS,
)
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import InputTrainingSpec
from continuous_tokenizer.contracts.manifest import RunManifest
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.contracts.profiles import (
    CAMPAIGN_PROFILE_NAME,
    profile_named,
)
from continuous_tokenizer.diagnostics.preflight import require_storage, run_preflight
from continuous_tokenizer.runtime.device import declared_device
from continuous_tokenizer.runtime.environment import dependency_environment, runtime_environment
from continuous_tokenizer.runtime.progress import log_event
from continuous_tokenizer.runtime.resume import ResumeManager
from continuous_tokenizer.runtime.tensors import tensor_fingerprint


def _load_selection_artifact(
    artifact: str,
    expected_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    path = Path(artifact)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} artifact hash does not match the experiment")
    return load_artifact(path)


type VerificationLog = tuple[Path, Path, str]


class ProspectiveBudgetExhaustedError(RuntimeError):
    def __init__(self, boundary: str, elapsed_seconds: float) -> None:
        super().__init__(
            f"prospective wall-clock budget exhausted at {boundary}",
        )
        self.boundary = boundary
        self.elapsed_seconds = elapsed_seconds


class ProspectivePolicy(Protocol):
    @property
    def tier(self) -> str: ...

    @property
    def futility_enabled(self) -> bool: ...

    def enforce_boundary(self, boundary: str) -> None: ...

    def execution_fingerprint(self, experiment_fingerprint: str) -> str: ...


def _verification_logs(
    source: Path,
    destination: Path,
    checks: Mapping[str, Any],
) -> tuple[VerificationLog, ...]:
    declared: list[VerificationLog] = []
    for name, raw_check in checks.items():
        if not isinstance(raw_check, Mapping):
            raise ValueError(f"verification check {name!r} is not canonical")
        relative = raw_check.get("log")
        expected = raw_check.get("log_sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"verification check {name!r} has no log")
        if not isinstance(expected, str):
            raise ValueError(f"verification check {name!r} has no log hash")
        source_log = source.parent / relative
        if sha256_file(source_log) != expected:
            raise ValueError(f"verification log hash mismatch for {name}")
        declared.append((source_log, destination / relative, expected))
    return tuple(declared)


def _validate_copied_verification(
    source: Path,
    target: Path,
    logs: tuple[VerificationLog, ...],
) -> None:
    if sha256_file(target) != sha256_file(source):
        raise ValueError("copied verification artifact changed before resume")
    if any(sha256_file(copied) != expected for _, copied, expected in logs):
        raise ValueError("copied verification log changed before resume")


def _copy_verification(
    source: Path,
    target: Path,
    logs: tuple[VerificationLog, ...],
) -> None:
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    for source_log, copied, _ in logs:
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_log, copied)


class ExperimentLifecycle:
    def __init__(  # noqa: PLR0913 - Lifecycle identity dependencies remain explicit.
        self,
        spec: ExperimentSpec,
        output_dir: Path,
        project_root: Path,
        verification_path: Path | None = None,
        *,
        resume: bool = False,
        resume_fingerprint: str | None = None,
        prospective_policy: ProspectivePolicy | None = None,
    ) -> None:
        self.spec = spec
        self.prospective_policy = prospective_policy
        if spec.evidence_scope == "final" and spec.prospective_selection is None:
            raise ValueError("final experiments must be materialized by tokenizer freeze")
        self.profile = profile_named(spec.training.profile)
        self.project_root = project_root
        self.device = declared_device(spec.device)
        self.source_state = source_state(project_root)
        self.dependency_lock_sha256 = sha256_file(project_root / "uv.lock")
        self.verification = self._load_verification(verification_path)
        self.installed_package = installed_distribution_identity("continuous-byte-tokenizer")
        selection_feasible = self._validate_selection_provenance()
        efficiency_feasible = self._validate_efficiency_pilot()
        self.provenance_feasible = selection_feasible and efficiency_feasible
        require_storage(
            output_dir.parent,
            spec,
            refusal_message="insufficient storage for the next run while preserving the registered reserve",
        )
        self.run_directory = RunDirectory(output_dir, resume=resume)
        self.verification = self._materialize_verification(
            verification_path,
            resume=resume,
        )
        self.inputs = self._input_identities()
        self.stage_timings: list[dict[str, Any]] = []
        _, _, source_sha256 = self.source_state
        execution_fingerprint = spec.fingerprint() if prospective_policy is None else prospective_policy.execution_fingerprint(spec.fingerprint())
        self.resume_manager = ResumeManager(
            output_dir,
            execution_fingerprint if resume_fingerprint is None else resume_fingerprint,
            self.source_state[0],
            source_sha256,
            self.dependency_lock_sha256,
            resume,
            spec.runtime.snapshot_interval,
        )
        if resume:
            self._validate_resume()

    @property
    def resuming(self) -> bool:
        return self.resume_manager.resuming

    def _validate_efficiency_pilot(self) -> bool:
        path_value = self.spec.efficiency_pilot
        expected_hash = self.spec.efficiency_pilot_sha256
        if path_value is None or expected_hash is None:
            return True
        path = Path(path_value)
        if sha256_file(path) != expected_hash:
            raise ValueError("efficiency pilot hash does not match the experiment")
        pilot = dict(load_artifact(path))
        selected_efficiency_passed = pilot.get(
            "selected_efficiency_passed",
        )
        selection_feasible = pilot.get("selection_feasible")
        if (
            pilot.get("evidence_scope") != "search"
            or pilot.get("operational_status") != "completed"
            or pilot.get("model_id") != self.spec.model.model_id
            or pilot.get("model_revision") != self.spec.model.revision
            or not isinstance(selected_efficiency_passed, bool)
            or selection_feasible is not selected_efficiency_passed
        ):
            raise ValueError("efficiency pilot provenance does not match the experiment")
        parameters = pilot.get("selected_parameters")
        training = self.spec.training
        if not isinstance(parameters, Mapping) or not isinstance(
            training,
            InputTrainingSpec,
        ):
            raise ValueError("efficiency pilot selection metadata is incomplete")
        expected = {
            "learning_rate": training.learning_rate,
            "batch_size": training.batch_size,
            "projection_multiplier": training.projection_multiplier,
            "muon_ns_steps": training.muon_ns_steps,
        }
        if any(parameters.get(name) != value for name, value in expected.items()):
            raise ValueError("experiment training settings differ from the efficiency selection")
        return selected_efficiency_passed

    def _validate_selection_provenance(self) -> bool:
        if self.spec.prospective_selection is not None:
            self._validate_prospective_selection()
            return True
        feasible = True
        for selection in self.spec.search_selections:
            label = f"{selection.search_kind} search"
            artifact = _load_selection_artifact(
                selection.artifact,
                selection.artifact_sha256,
                label,
            )
            expected = {
                "search_fingerprint": selection.search_fingerprint,
                "selected_trial": selection.selected_trial,
                "model_id": selection.model_id,
                "model_revision": selection.model_revision,
                "profile": selection.profile,
                "selected_parameters": dict(selection.selected_parameters),
                "selection_feasible": selection.feasible,
            }
            if artifact.get("operational_status") != "completed" or any(artifact.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{label} provenance does not match the experiment")
            feasible = feasible and selection.feasible
        for selection in self.spec.study_selections:
            label = f"{selection.study_kind} study"
            artifact = _load_selection_artifact(
                selection.artifact,
                selection.artifact_sha256,
                label,
            )
            selected = artifact.get("selection")
            if not isinstance(selected, Mapping):
                raise ValueError(f"{label} selection metadata is incomplete")
            expected = {
                "study_fingerprint": selection.study_fingerprint,
                "model_id": selection.model_id,
                "model_revision": selection.model_revision,
            }
            if (
                artifact.get("operational_status") != "completed"
                or any(artifact.get(name) != value for name, value in expected.items())
                or selected.get("selection_feasible") is not selection.feasible
                or any(selected.get(name) != value for name, value in selection.selected_parameters.items())
            ):
                raise ValueError(f"{label} provenance does not match the experiment")
            feasible = feasible and selection.feasible
        return feasible

    def _validate_prospective_selection(self) -> None:
        from continuous_tokenizer.contracts.prospective import ProspectiveSpec

        selection = self.spec.prospective_selection
        if selection is None:
            raise ValueError("prospective selection provenance is missing")
        for label, path_value, expected in (
            ("candidate-selection", selection.artifact, selection.artifact_sha256),
            (
                "candidate-selection TOML",
                selection.candidate_toml,
                selection.candidate_toml_sha256,
            ),
            ("timing calibration", selection.calibration, selection.calibration_sha256),
            ("frozen TOML", selection.frozen_toml, selection.frozen_toml_sha256),
        ):
            if sha256_file(Path(path_value)) != expected:
                raise ValueError(f"{label} hash does not match the frozen experiment")
        source_commit, _, source_state_sha256 = self.source_state
        if (
            selection.source_commit != source_commit
            or selection.source_state_sha256 != source_state_sha256
            or selection.dependency_lock_sha256 != self.dependency_lock_sha256
        ):
            raise ValueError("prospective selection source or dependency identity differs from the run")
        artifact = load_json_object(Path(selection.artifact))
        selected = artifact.get("selection")
        if (
            artifact.get("tier") != "candidate_selection"
            or artifact.get("mode") != self.spec.mode
            or artifact.get("operational_status") != "completed"
            or artifact.get("budget_exhausted") is not False
            or artifact.get("spec_fingerprint") != selection.spec_fingerprint
            or not isinstance(selected, Mapping)
            or selected.get("selection_feasible") is not True
            or selected.get("validation_only") is not True
            or selected.get("final_test_loaded") is not False
            or selected.get("selected_configuration") != dict(selection.selected_parameters)
            or (self.spec.mode == "input_only" and selected.get("selected_strategy") != selection.selected_strategy)
        ):
            raise ValueError("candidate-selection artifact does not match the frozen experiment")
        calibration = load_json_object(Path(selection.calibration))
        if (
            calibration.get("artifact_kind") != "timing_calibration"
            or calibration.get("model_id") != selection.model_id
            or calibration.get("model_revision") != selection.model_revision
            or calibration.get("mode") != self.spec.mode
        ):
            raise ValueError("timing calibration does not match the frozen experiment")
        frozen = ProspectiveSpec.load(Path(selection.frozen_toml))
        if (
            frozen.tier != "final_evidence"
            or frozen.mode != self.spec.mode
            or frozen.wall_clock.calibration.sha256 != selection.calibration_sha256
            or frozen.design.get("selection_artifact_sha256") != selection.artifact_sha256
            or frozen.design.get("selected_strategy") != selection.selected_strategy
            or frozen.design.get("selected_configuration") != dict(selection.selected_parameters)
            or frozen.load_experiment().fingerprint() != self.spec.fingerprint()
        ):
            raise ValueError("frozen prospective TOML does not match the experiment")

    def _validate_resume(self) -> None:
        final = self.run_directory.root / "manifest-final.json"
        if final.exists():
            raise ValueError("completed or failed runs cannot be resumed")
        start = dict(load_json_object(self.run_directory.root / "manifest-start.json"))
        source_commit, source_dirty, source_sha256 = self.source_state
        expected = {
            "experiment_fingerprint": self.spec.fingerprint(),
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "source_state_sha256": source_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "installed_package": self.installed_package,
        }
        if any(start.get(name) != value for name, value in expected.items()):
            raise ValueError("interrupted run does not match the current experiment contract")

    def _portable_identity(self, path: Path) -> dict[str, str]:
        resolved = path.resolve(strict=True)
        return {
            "locator": Path(os.path.relpath(resolved, start=self.run_directory.root.resolve())).as_posix(),
            "sha256": sha256_path(resolved),
        }

    def _input_identities(self) -> dict[str, dict[str, str]]:
        inputs: dict[str, dict[str, str]] = {}
        prospective = self.spec.prospective_selection
        if prospective is not None:
            for name, value in (
                ("prospective_selection", prospective.artifact),
                ("prospective_candidate_toml", prospective.candidate_toml),
                ("prospective_calibration", prospective.calibration),
                ("prospective_frozen_toml", prospective.frozen_toml),
            ):
                inputs[name] = self._portable_identity(Path(value))
        for selection in self.spec.search_selections:
            inputs[f"search_{selection.search_kind}"] = self._portable_identity(Path(selection.artifact))
        for selection in self.spec.study_selections:
            inputs[f"study_{selection.study_kind}"] = self._portable_identity(Path(selection.artifact))
        if self.spec.efficiency_pilot is not None:
            inputs["efficiency_pilot"] = self._portable_identity(Path(self.spec.efficiency_pilot))
        if self.verification.get("provided") is True:
            inputs["verification"] = self._portable_identity(self.run_directory.root / "verification/verification.json")
        return inputs

    def _materialize_verification(
        self,
        source: Path | None,
        *,
        resume: bool,
    ) -> dict[str, Any]:
        if source is None:
            return self.verification
        destination = self.run_directory.root / "verification"
        target = destination / "verification.json"
        checks = self.verification.get("checks")
        if not isinstance(checks, Mapping):
            raise ValueError("verification artifact has no check inventory")
        logs = _verification_logs(source, destination, checks)
        if resume:
            _validate_copied_verification(source, target, logs)
        else:
            _copy_verification(source, target, logs)
        return {
            **self.verification,
            "path": "verification/verification.json",
        }

    def _run_preflight(
        self,
        assets: ModelAssets,
        *,
        load_full_model: bool = True,
    ) -> tuple[dict[str, Any], nn.Module | None]:
        path = self.run_directory.root / "preflight.json"
        loader = None if self.spec.model.evaluation != "full" or not load_full_model else lambda: load_frozen_causal_lm(assets, self.device)
        identity = {
            "experiment_fingerprint": self.spec.fingerprint(),
            "source_commit": self.source_state[0],
            "source_dirty": self.source_state[1],
            "source_state_sha256": self.source_state[2],
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "installed_package": self.installed_package,
        }
        if self.resuming and path.is_file():
            artifact = dict(load_json_object(path))
            if artifact.get("all_passed") is not True or any(artifact.get(name) != value for name, value in identity.items()):
                raise ValueError("interrupted run has an invalid preflight artifact")
            return artifact, None if loader is None else loader()
        return run_preflight(
            self.spec,
            path,
            device=self.device,
            load_full_model=loader,
            identity=identity,
            representative_mps_verified=self._representative_mps_verified(),
        )

    def _representative_mps_verified(self) -> bool:
        checks = self.verification.get("checks")
        representative = checks.get("representative_mps") if isinstance(checks, Mapping) else None
        return (
            self.device.type == "mps"
            and self.verification.get("provided") is True
            and self.verification.get("kind") == "complete_verification"
            and isinstance(representative, Mapping)
            and representative.get("passed") is True
        )

    def _record_stage(
        self,
        name: str,
        status: Literal["completed", "failed", "stopped_budget"],
        started: float,
        before: dict[str, Any],
    ) -> float:
        elapsed = perf_counter() - started
        self.stage_timings.append(
            {
                "stage": name,
                "status": status,
                "seconds": elapsed,
                "before": before,
                "after": runtime_environment(self.device),
            }
        )
        return elapsed

    @contextmanager
    def _stage(self, name: str) -> Iterator[None]:
        context = {
            "experiment": self.spec.name,
            "mode": self.spec.mode,
            "stage": name,
        }
        started = perf_counter()
        before = runtime_environment(self.device)
        log_event("stage_started", **context)
        try:
            yield
        except ProspectiveBudgetExhaustedError as error:
            elapsed = self._record_stage(
                name,
                "stopped_budget",
                started,
                before,
            )
            log_event(
                "stage_stopped_budget",
                **context,
                elapsed_seconds=round(elapsed, 1),
                boundary=error.boundary,
            )
            raise
        except BaseException as error:
            elapsed = self._record_stage(name, "failed", started, before)
            log_event(
                "stage_failed",
                **context,
                elapsed_seconds=round(elapsed, 1),
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise
        elapsed = self._record_stage(name, "completed", started, before)
        log_event(
            "stage_completed",
            **context,
            elapsed_seconds=round(elapsed, 1),
        )
        self._enforce_prospective_boundary(f"stage:{name}")

    def _enforce_prospective_boundary(self, boundary: str) -> None:
        if self.prospective_policy is not None:
            self.prospective_policy.enforce_boundary(boundary)

    def _prospective_epoch_boundary(
        self,
        phase: str,
        epoch: int,
    ) -> None:
        self._enforce_prospective_boundary(f"epoch:{phase}:{epoch}")

    def _load_verification(self, path: Path | None) -> dict[str, Any]:
        if path is None:
            if self._requires_final_verification():
                raise ValueError("final experiments require a verification artifact")
            return {"provided": False}
        verification = dict(load_artifact(path))
        source_commit, _, source_sha256 = self.source_state
        if verification.get("source_commit") != source_commit:
            raise ValueError("verification source commit does not match the experiment source commit")
        if verification["source_state_sha256"] != source_sha256:
            raise ValueError("verification source state does not match the experiment source state")
        if verification["dependency_lock_sha256"] != self.dependency_lock_sha256:
            raise ValueError("verification dependency lock does not match the experiment lock")
        if not verification["all_passed"]:
            raise ValueError("verification artifact contains failed checks")
        if self._requires_final_verification():
            self._require_complete_verification(verification)
        return {"provided": True, **verification, "path": str(path)}

    @staticmethod
    def _require_complete_verification(
        verification: Mapping[str, Any],
    ) -> None:
        checks = verification.get("checks")
        if not isinstance(checks, Mapping):
            raise ValueError(
                "verification artifact lacks final-run checks: " + ", ".join(sorted(FINAL_VERIFICATION_CHECKS)),
            )
        check_names = set(checks)
        missing = sorted(FINAL_VERIFICATION_CHECKS - check_names)
        if missing:
            raise ValueError(
                f"verification artifact lacks final-run checks: {', '.join(missing)}",
            )
        if verification.get("kind") != "complete_verification" or check_names != FINAL_VERIFICATION_CHECKS:
            raise ValueError(
                "final experiments require a canonical complete verification artifact",
            )
        if any(not isinstance(checks[name], Mapping) or checks[name].get("passed") is not True for name in FINAL_VERIFICATION_CHECKS):
            raise ValueError("verification artifact contains a failed final-run check")

    def _requires_final_verification(self) -> bool:
        return (
            self.spec.evidence_scope == "final"
            and self.profile.name == CAMPAIGN_PROFILE_NAME
            and self.spec.model.evaluation == "full"
            and self.spec.model.model_id != SYNTHETIC_MODEL_ID
        )

    def _manifest(
        self,
        *,
        status: Literal["running", "passed", "failed"],
        artifacts: dict[str, str],
        trainable_parameters: tuple[str, ...] = (),
        assets: ModelAssets | None = None,
        frozen_backbone_fingerprint: str | None = None,
    ) -> RunManifest:
        spec = self.spec
        source_commit, source_dirty, source_state_sha256 = self.source_state
        return RunManifest(
            experiment_name=spec.name,
            mode=spec.mode,
            codec_direction=spec.mode,
            experiment_fingerprint=spec.fingerprint(),
            replication_fingerprint=spec.replication_fingerprint(),
            model_id=spec.model.model_id,
            model_revision=spec.model.revision,
            dataset_id=spec.dataset.dataset_id,
            dataset_revision=spec.dataset.revision,
            embedding_tensor=None if assets is None else assets.embedding_tensor_name,
            source_dtype=None if assets is None else str(assets.input_embeddings.dtype),
            seed=spec.seed,
            stages=spec.stages,
            source_commit=source_commit,
            source_dirty=source_dirty,
            source_state_sha256=source_state_sha256,
            dependency_lock_sha256=self.dependency_lock_sha256,
            installed_package=self.installed_package,
            claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
            source_assets=self._source_asset_identities(assets),
            inputs=self.inputs,
            codec_attention=(gqa_metadata(self.profile.query_heads) if spec.mode == "input_only" else {"type": "none"}),
            environment=dependency_environment(self.device),
            trainable_parameters=trainable_parameters,
            frozen_backbone_fingerprint=frozen_backbone_fingerprint,
            native_head_used=spec.mode == "input_only",
            feedback_policy=("native_output_tokens" if spec.mode == "input_only" else "longest_native_byte_match"),
            artifacts=artifacts,
            artifact_hashes=(self._artifact_hashes(artifacts) if status != "running" else {}),
            status=status,
            verification=self.verification,
        )

    def _source_asset_identities(
        self,
        assets: ModelAssets | None,
    ) -> dict[str, dict[str, str]]:
        if assets is None:
            return {}
        identities = {
            "model_config": {
                "locator": f"hf://{assets.model_id}@{assets.revision}/config.json",
                "sha256": mapping_fingerprint(assets.config),
            },
            "input_embedding_tensor": {
                "locator": (f"hf://{assets.model_id}@{assets.revision}#{assets.embedding_tensor_name}"),
                "sha256": tensor_fingerprint(((assets.embedding_tensor_name, assets.input_embeddings),)),
            },
            "tokenizer_vocabulary": {
                "locator": f"hf://{assets.model_id}@{assets.revision}/tokenizer",
                "sha256": mapping_fingerprint(assets.vocabulary.to_summary()),
            },
        }
        if assets.embedding_shard.is_file():
            identities["embedding_shard"] = {
                "locator": (f"hf://{assets.model_id}@{assets.revision}/{assets.embedding_shard.name}"),
                "sha256": sha256_file(assets.embedding_shard),
            }
        return identities

    def _artifact_hashes(self, artifacts: dict[str, str]) -> dict[str, str]:
        return {name: sha256_path(self.run_directory.root / relative) for name, relative in artifacts.items()}

    def _result_metadata(
        self,
        assets: ModelAssets,
        *,
        gates_passed: bool,
        search: bool = False,
    ) -> dict[str, str]:
        if search:
            evidence_scope = "search"
            scientific_verdict = "not_applicable_search"
        elif assets.model_id == SYNTHETIC_MODEL_ID:
            evidence_scope = "synthetic"
            scientific_verdict = "supported" if gates_passed else "unsupported"
        else:
            evidence_scope = self.spec.evidence_scope
            if evidence_scope == "diagnostic":
                scientific_verdict = "not_applicable_diagnostic"
            elif evidence_scope == "final":
                scientific_verdict = "supported" if gates_passed and self.provenance_feasible else "unsupported"
            else:
                scientific_verdict = f"not_applicable_{evidence_scope}"
        return {
            "mode": self.spec.mode,
            "evidence_scope": evidence_scope,
            "operational_status": "completed",
            "scientific_verdict": scientific_verdict,
        }

    def _reporting_context(self, assets: ModelAssets) -> dict[str, Any]:
        return {
            "vocabulary": assets.vocabulary.to_summary(),
            "verification": self.verification,
            "runtime": {
                "stages": list(self.stage_timings),
                "final": runtime_environment(self.device),
                "recovery_snapshots": self.resume_manager.telemetry(),
            },
        }

    def _write_manifest(
        self,
        name: str,
        manifest: RunManifest,
    ) -> None:
        self.run_directory.write_json(name, manifest.to_dict())

    def _write_start_manifest(self, assets: ModelAssets | None) -> None:
        if self.resuming:
            return
        self._write_manifest(
            "manifest-start.json",
            self._manifest(
                status="running",
                artifacts={},
                assets=assets,
            ),
        )

    def _write_experiment_contract(self) -> None:
        path = self.run_directory.root / "experiment.json"
        expected = self.spec.to_dict()
        if self.resuming:
            if dict(load_json_object(path)) != expected:
                raise ValueError("interrupted run has a different experiment contract")
            return
        self.run_directory.write_json("experiment.json", expected)

    def _final_artifacts(self, artifacts: dict[str, str]) -> dict[str, str]:
        sealed = {
            "experiment": "experiment.json",
            **({"verification": "verification"} if self.verification.get("provided") is True else {}),
            **artifacts,
        }
        if (self.run_directory.root / "phase-final").is_dir():
            sealed["phase_final"] = "phase-final"
        return sealed

    def _finalize_success(
        self,
        artifacts: dict[str, str],
        *,
        trainable_parameters: tuple[str, ...],
        assets: ModelAssets | None,
        frozen_backbone_fingerprint: str | None,
    ) -> None:
        artifacts = self._final_artifacts(artifacts)
        self._write_manifest(
            "manifest-final.json",
            self._manifest(
                status="passed",
                artifacts=artifacts,
                trainable_parameters=trainable_parameters,
                assets=assets,
                frozen_backbone_fingerprint=frozen_backbone_fingerprint,
            ),
        )
        self.resume_manager.cleanup()

    def _finalize_failure(
        self,
        error: BaseException,
        artifacts: dict[str, str],
        *,
        trainable_parameters: tuple[str, ...],
        assets: ModelAssets | None,
        frozen_backbone_fingerprint: str | None,
    ) -> None:
        self.run_directory.write_json(
            "failure.json",
            {
                "kind": "run_failure",
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        artifacts = {name: relative for name, relative in artifacts.items() if (self.run_directory.root / relative).exists()}
        artifacts["failure"] = "failure.json"
        artifacts = self._final_artifacts(artifacts)
        self._write_manifest(
            "manifest-final.json",
            self._manifest(
                status="failed",
                artifacts=artifacts,
                trainable_parameters=trainable_parameters,
                assets=assets,
                frozen_backbone_fingerprint=frozen_backbone_fingerprint,
            ),
        )
