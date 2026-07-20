from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from continuous_tokenizer.artifacts.evidence import (
    EVIDENCE_MANIFEST_FILENAME,
    load_evidence_manifest,
    verify_artifact,
)
from continuous_tokenizer.artifacts.hashing import (
    installed_distribution_identity,
    sha256_file,
)
from continuous_tokenizer.artifacts.source import find_project_root, source_state
from continuous_tokenizer.artifacts.store import load_json_object, write_text_atomic
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.prospective import (
    PROSPECTIVE_ARTIFACT_KINDS,
    PROSPECTIVE_RESULT_FILENAME,
    ProspectiveSpec,
)
from continuous_tokenizer.contracts.prospective_selection import (
    ProspectiveSelectionSpec,
)

_REPLICATION_SEEDS = (17, 23, 41)
_MODELS = {
    "Qwen/Qwen3.5-0.8B": "qwen35-0.8b",
    "google/gemma-3-270m-it": "gemma3-270m",
}


@dataclass(frozen=True, slots=True)
class _FreezeIdentity:
    source_commit: str
    source_state_sha256: str
    dependency_lock_sha256: str
    installed_package: Mapping[str, str]


def _portable(values: dict[str, Any], path: Path) -> dict[str, Any]:
    portable = dict(values)
    prospective = values.get("prospective_selection")
    if isinstance(prospective, Mapping):
        portable["prospective_selection"] = {
            **prospective,
            **{
                name: os.path.relpath(str(prospective[name]), path.parent)
                for name in (
                    "artifact",
                    "candidate_toml",
                    "calibration",
                    "frozen_toml",
                )
            },
        }
    for field in ("search_selections", "study_selections"):
        if field in values:
            portable[field] = [
                {
                    **selection,
                    "artifact": os.path.relpath(selection["artifact"], path.parent),
                }
                for selection in cast(Sequence[Mapping[str, Any]], values[field])
            ]
    pilot = values.get("efficiency_pilot")
    if isinstance(pilot, str):
        portable["efficiency_pilot"] = os.path.relpath(pilot, path.parent)
    return portable


def _prospective_candidate_directories(
    paths: Sequence[Path],
    identity: _FreezeIdentity,
) -> dict[tuple[str, str], tuple[Path, dict[str, Any], ProspectiveSpec]]:
    registered: dict[
        tuple[str, str],
        tuple[Path, dict[str, Any], ProspectiveSpec],
    ] = {}
    for supplied in paths:
        directory = supplied if supplied.is_dir() else supplied.parent
        verification = verify_artifact(directory)
        if verification["valid"] is not True:
            raise ValueError(f"prospective candidate evidence is invalid: {directory}")
        manifest = load_evidence_manifest(directory / EVIDENCE_MANIFEST_FILENAME)
        if manifest["artifact_kind"] != PROSPECTIVE_ARTIFACT_KINDS["candidate_selection"]:
            raise ValueError(
                "freeze accepts only current prospective candidate-selection artifacts",
            )
        source = cast(Mapping[str, Any], manifest["source"])
        if (
            source.get("commit") != identity.source_commit
            or source.get("dirty") is not False
            or source.get("state_sha256") != identity.source_state_sha256
            or manifest.get("dependency_lock_sha256") != identity.dependency_lock_sha256
            or manifest.get("installed_package") != identity.installed_package
        ):
            raise ValueError(
                "prospective selection provenance differs from the current clean source",
            )
        result_path = directory / PROSPECTIVE_RESULT_FILENAME
        result = dict(load_json_object(result_path))
        selection = result.get("selection")
        if (
            result.get("tier") != "candidate_selection"
            or result.get("operational_status") != "completed"
            or result.get("budget_exhausted") is not False
            or result.get("final_claims_allowed") is not False
            or not isinstance(selection, Mapping)
            or selection.get("selection_feasible") is not True
            or selection.get("validation_only") is not True
            or selection.get("final_test_loaded") is not False
        ):
            raise ValueError("prospective candidate selection is not freeze-eligible")
        spec_entry = cast(Mapping[str, Any], manifest["inputs"]).get("spec")
        if not isinstance(spec_entry, Mapping):
            raise ValueError("prospective candidate selection has no sealed wrapper")
        spec = ProspectiveSpec.load(
            (directory / str(spec_entry["locator"])).resolve(),
        )
        key = (str(cast(Mapping[str, Any], manifest["model"])["id"]), str(result["mode"]))
        if key in registered:
            raise ValueError(f"duplicate prospective candidate selection: {key}")
        registered[key] = (result_path, result, spec)
    expected = {(model_id, mode) for model_id in _MODELS for mode in ("input_only", "output_only")}
    if set(registered) != expected:
        raise ValueError(
            f"prospective freeze artifact set mismatch; missing={sorted(expected - set(registered))}",
        )
    return registered


def _freeze_prospective(
    args: argparse.Namespace,
    toml_dumps: Callable[[Mapping[str, Any]], str],
    identity: _FreezeIdentity,
) -> dict[str, Any]:
    artifacts = _prospective_candidate_directories(args.artifacts, identity)
    if args.output_dir.exists():
        raise FileExistsError(
            f"freeze output directory already exists: {args.output_dir}",
        )
    generated: list[Path] = []
    for (model_id, mode), (result_path, result, selection_spec) in artifacts.items():
        selection = cast(Mapping[str, Any], result["selection"])
        configuration = cast(
            Mapping[str, int | float | str],
            selection["selected_configuration"],
        )
        template = selection_spec.load_final_reference()
        selected_strategy = selection.get("selected_strategy")
        updates = dict(configuration)
        if mode == "input_only":
            if not isinstance(selected_strategy, str):
                raise ValueError("input candidate selection has no explicit strategy")
            updates["strategy"] = selected_strategy
        training = replace(
            template.training,
            **{name: value for name, value in updates.items() if name in getattr(template.training, "__dataclass_fields__", {})},
        )
        slug = _MODELS[model_id]
        direction = "input" if mode == "input_only" else "output"
        provenance = args.output_dir / "_provenance" / direction / slug
        provenance.mkdir(parents=True, exist_ok=True)
        copied_selection = provenance / "prospective.json"
        shutil.copy2(result_path, copied_selection)
        selection_sha256 = sha256_file(copied_selection)
        copied_candidate_toml = provenance / "candidate-selection.toml"
        shutil.copy2(selection_spec.path, copied_candidate_toml)
        candidate_toml_sha256 = sha256_file(copied_candidate_toml)
        calibration_source = (selection_spec.path.parent / selection_spec.wall_clock.calibration.locator).resolve()
        copied_calibration = provenance / "timing-calibration.json"
        shutil.copy2(calibration_source, copied_calibration)
        calibration_sha256 = sha256_file(copied_calibration)
        for seed in _REPLICATION_SEEDS:
            directory = args.output_dir / direction / slug
            experiment_path = directory / f"seed-{seed}.experiment.toml"
            wrapper_path = directory / f"seed-{seed}.toml"
            base_final = replace(
                template,
                name=f"{slug}-{direction}-only-seed-{seed}",
                seed=seed,
                training=training,
                evidence_scope="final",
                prospective_selection=None,
                search_selections=(),
                study_selections=(),
                efficiency_pilot=None,
                efficiency_pilot_sha256=None,
            )
            wrapper = {
                "schema_version": 1,
                "artifact_kind": PROSPECTIVE_ARTIFACT_KINDS["final_evidence"],
                "tier": "final_evidence",
                "name": base_final.name,
                "mode": mode,
                "experiment": os.path.relpath(
                    experiment_path,
                    wrapper_path.parent,
                ),
                "final_reference": os.path.relpath(
                    experiment_path,
                    wrapper_path.parent,
                ),
                "wall_clock": {
                    "calibration": {
                        "locator": os.path.relpath(
                            copied_calibration,
                            wrapper_path.parent,
                        ),
                        "sha256": calibration_sha256,
                    },
                    "expected_seconds": selection_spec.wall_clock.expected_seconds,
                    "maximum_seconds": selection_spec.wall_clock.maximum_seconds,
                    "stop_boundary": "epoch_or_stage",
                    "work_units": dict(selection_spec.wall_clock.work_units),
                },
                "design": {
                    "seed": seed,
                    "complete_vocabulary": True,
                    "full_final_evaluation": True,
                    "independent_retraining": True,
                    "reuse_study_weights": False,
                    "reuse_validation_metrics": False,
                    "selection_artifact": os.path.relpath(
                        copied_selection,
                        wrapper_path.parent,
                    ),
                    "selection_artifact_sha256": selection_sha256,
                    "selected_strategy": selected_strategy or "output_codec",
                    "selected_configuration": dict(configuration),
                    "selected_budget": {
                        name: selection_spec.design[name]
                        for name in (
                            "vocabulary_rows",
                            "vocabulary_epochs",
                            "patience",
                            "maximum_alignment_trials",
                            "maximum_efficiency_trials",
                        )
                    },
                },
            }
            experiment_path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(wrapper_path, toml_dumps(wrapper))
            frozen_toml_sha256 = sha256_file(wrapper_path)
            prospective_selection = ProspectiveSelectionSpec(
                artifact=str(copied_selection),
                artifact_sha256=selection_sha256,
                candidate_toml=str(copied_candidate_toml),
                candidate_toml_sha256=candidate_toml_sha256,
                calibration=str(copied_calibration),
                calibration_sha256=calibration_sha256,
                frozen_toml=str(wrapper_path),
                frozen_toml_sha256=frozen_toml_sha256,
                spec_fingerprint=selection_spec.fingerprint(),
                model_id=model_id,
                model_revision=template.model.revision,
                dataset_id=template.dataset.dataset_id,
                dataset_config=template.dataset.config,
                dataset_revision=template.dataset.revision,
                profile=training.profile,
                selected_strategy=str(selected_strategy or "output_codec"),
                selected_parameters=dict(configuration),
                source_commit=identity.source_commit,
                source_state_sha256=identity.source_state_sha256,
                dependency_lock_sha256=identity.dependency_lock_sha256,
            )
            final = replace(
                base_final,
                prospective_selection=prospective_selection,
            )
            write_text_atomic(
                experiment_path,
                toml_dumps(_portable(final.to_toml_dict(), experiment_path)),
            )
            ExperimentSpec.load(experiment_path)
            ProspectiveSpec.load(wrapper_path)
            generated.append(wrapper_path)
    return {
        "status": "completed",
        "specifications": [str(path) for path in generated],
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    project_root = find_project_root(Path.cwd())
    source_commit, source_dirty, source_state_sha256 = source_state(project_root)
    if source_dirty:
        raise ValueError("tokenizer freeze requires a clean source tree")
    dependency_lock_sha256 = sha256_file(project_root / "uv.lock")
    installed_package = installed_distribution_identity(
        "continuous-byte-tokenizer",
    )
    identity = _FreezeIdentity(
        source_commit,
        source_state_sha256,
        dependency_lock_sha256,
        installed_package,
    )
    try:
        import tomli_w
    except ModuleNotFoundError as error:
        raise RuntimeError("install the search dependency group with `uv sync --group search`") from error
    prospective_kind = PROSPECTIVE_ARTIFACT_KINDS["candidate_selection"]
    manifests = tuple(
        load_evidence_manifest(
            (path if path.is_dir() else path.parent) / EVIDENCE_MANIFEST_FILENAME,
        )
        for path in args.artifacts
    )
    if not manifests or any(manifest["artifact_kind"] != prospective_kind for manifest in manifests):
        raise ValueError(
            "freeze accepts only current prospective candidate-selection artifacts",
        )
    return _freeze_prospective(
        args,
        tomli_w.dumps,
        identity,
    )
