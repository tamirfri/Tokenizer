from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from continuous_tokenizer.artifacts.evidence import (
    EVIDENCE_MANIFEST_FILENAME,
    EvidenceIdentity,
    EvidenceManifest,
    load_evidence_manifest,
    verify_artifact,
    write_evidence_manifest,
)
from continuous_tokenizer.artifacts.manifest import load_artifact
from continuous_tokenizer.artifacts.software_validation import (
    load_software_validation_inputs,
    validation_input_directories,
)
from continuous_tokenizer.artifacts.store import RunDirectory
from continuous_tokenizer.contracts.claim_derivation import (
    GEMMA_MODEL,
    QWEN_MODEL,
    derive_deployment_claim_verdicts,
    derive_input_headline_verdict,
    derive_project_alignment_verdict,
    derive_project_deployment_verdicts,
)
from continuous_tokenizer.contracts.claims import (
    ClaimVerdict,
    claim_category_verdicts,
    claim_records,
    combine_claim_verdicts,
    directional_claims,
    project_claim_trace_records,
)
from continuous_tokenizer.contracts.experiment import TokenizerMode
from continuous_tokenizer.contracts.input_study import (
    alignment_feasibility_verdict,
)
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.contracts.profiles import CAMPAIGN_PROFILE_NAME
from continuous_tokenizer.contracts.statements import (
    SoftwareValidationInputs,
    SourceBinding,
    statement_trace_records,
)
from continuous_tokenizer.reporting.project_markdown import project_report

_PRIMARY_MODELS = (QWEN_MODEL, GEMMA_MODEL)


def _require_valid_parent(directory: Path, role: str) -> None:
    verification = verify_artifact(directory)
    if verification["valid"] is not True:
        errors = "; ".join(str(error) for error in verification["errors"])
        raise ValueError(f"{role} evidence failed semantic verification: {errors}")


def _replication_model(replication: Mapping[str, Any]) -> tuple[str, str]:
    model = replication.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("replication has no canonical model identity")
    return str(model.get("id")), str(model.get("revision"))


def _load_primary_replications(
    directories: Sequence[Path],
) -> tuple[TokenizerMode, list[dict[str, Any]]]:
    if len(directories) != 2:
        raise ValueError("project evidence requires exactly two primary-model replications")
    loaded: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for directory in directories:
        _require_valid_parent(directory, "primary-model replication")
        manifest = load_evidence_manifest(directory / EVIDENCE_MANIFEST_FILENAME)
        if manifest["artifact_kind"] != "replication":
            raise ValueError("project inputs must be sealed replication evidence")
        replication = dict(load_artifact(directory / "replication.json"))
        model = _replication_model(replication)
        if model in loaded:
            raise ValueError("project inputs contain duplicate primary models")
        loaded[model] = (directory, replication)
    if set(loaded) != set(_PRIMARY_MODELS):
        raise ValueError("project evidence requires pinned Qwen 0.8B and Gemma 270M replications")
    ordered = [loaded[model] for model in _PRIMARY_MODELS]
    modes = {str(replication["mode"]) for _, replication in ordered}
    if len(modes) != 1:
        raise ValueError("primary-model replications must use the same tokenizer mode")
    mode = cast(TokenizerMode, modes.pop())
    for _, replication in ordered:
        if (
            replication.get("replication_complete") is not True
            or replication.get("operational_status") != "completed"
            or replication.get("profile") != CAMPAIGN_PROFILE_NAME
            or len(cast(Sequence[object], replication.get("runs", ()))) != 3
        ):
            raise ValueError("each primary model requires a complete three-seed Large-profile replication")
    _require_shared_identity([replication for _, replication in ordered])
    return mode, [
        {
            "directory": str(directory),
            "model": dict(cast(Mapping[str, Any], replication["model"])),
            "quality_verdict": _quality_verdict(replication),
            "scientific_verdict": _replication_scientific_verdict(replication),
            "replication": replication,
        }
        for directory, replication in ordered
    ]


def _require_shared_identity(replications: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "source_commit",
        "source_dirty",
        "source_state_sha256",
        "dependency_lock_sha256",
        "installed_package",
        "claim_vocabulary_sha256",
    )
    first = replications[0]
    if any(any(replication.get(name) != first.get(name) for name in fields) for replication in replications[1:]):
        raise ValueError("primary-model replications must share source and dependency identity")


def _quality_verdict(replication: Mapping[str, Any]) -> ClaimVerdict:
    categories = replication.get("category_verdicts")
    verdict = categories.get("quality") if isinstance(categories, Mapping) else None
    if verdict not in {"supported", "unsupported", "incomplete", "inapplicable"}:
        raise ValueError("replication has an invalid quality verdict")
    return cast(ClaimVerdict, verdict)


def _replication_scientific_verdict(
    replication: Mapping[str, Any],
) -> ClaimVerdict:
    if replication.get("mode") == "input_only":
        return derive_input_headline_verdict(
            _claim_verdicts(replication.get("claims")),
        )
    return _quality_verdict(replication)


def _claim_verdicts(claims: object) -> dict[str, ClaimVerdict]:
    if not isinstance(claims, Sequence) or isinstance(claims, str | bytes):
        raise ValueError("replication has no canonical claims")
    records = (cast(Mapping[str, object], record) for record in claims if isinstance(record, Mapping))
    return {str(record["claim_id"]): cast(ClaimVerdict, record["verdict"]) for record in records}


def _alignment_study_verdicts(
    mode: TokenizerMode,
    directories: Sequence[Path],
) -> tuple[ClaimVerdict, list[dict[str, Any]]]:
    if mode == "output_only":
        if directories:
            raise ValueError("alignment-feasibility studies apply only to input-only projects")
        return "inapplicable", []
    if not directories:
        return "incomplete", []
    if len(directories) != 2:
        raise ValueError("input project alignment evidence requires one sealed study per primary model")
    by_model: dict[tuple[str, str], dict[str, Any]] = {}
    for directory in directories:
        _require_valid_parent(directory, "alignment-feasibility study")
        manifest = load_evidence_manifest(directory / EVIDENCE_MANIFEST_FILENAME)
        result = dict(load_artifact(directory / "result.json"))
        model = (str(result.get("model_id")), str(result.get("model_revision")))
        if (
            manifest["artifact_kind"] != "study"
            or result.get("artifact_kind") != "input_alignment_feasibility_study"
            or result.get("prospective") is not True
            or result.get("final_evidence") is not False
        ):
            raise ValueError("alignment evidence must be the prospective fixed-subset study")
        if model in by_model:
            raise ValueError("alignment evidence contains duplicate primary models")
        by_model[model] = {
            "directory": str(directory),
            "model": {"id": model[0], "revision": model[1]},
            "verdict": alignment_feasibility_verdict(result),
            "result": result,
        }
    if set(by_model) != set(_PRIMARY_MODELS):
        raise ValueError("alignment evidence requires pinned Qwen 0.8B and Gemma 270M studies")
    ordered = [by_model[model] for model in _PRIMARY_MODELS]
    return derive_project_alignment_verdict(ordered), ordered


def _deployment_claims(
    mode: TokenizerMode,
    directories: Sequence[Path],
) -> tuple[dict[str, ClaimVerdict], list[dict[str, Any]]]:
    if not directories:
        return derive_project_deployment_verdicts(mode, ()), []
    if len(directories) != 2:
        raise ValueError("deployment evidence requires one sealed artifact per primary model")
    by_model: dict[tuple[str, str], dict[str, Any]] = {}
    for directory in directories:
        _require_valid_parent(directory, "deployment")
        manifest = load_evidence_manifest(directory / EVIDENCE_MANIFEST_FILENAME)
        model = (str(manifest["model"]["id"]), str(manifest["model"]["revision"]))
        deployment = dict(load_artifact(directory / "deployment.json"))
        applicability = cast(Mapping[str, Any], deployment["applicability"])
        if manifest["artifact_kind"] != "deployment" or manifest["mode"] != mode:
            raise ValueError("deployment evidence has the wrong artifact role")
        omission, removability = derive_deployment_claim_verdicts(deployment)
        by_model[model] = {
            "directory": str(directory),
            "model": {"id": model[0], "revision": model[1]},
            "applicability": dict(applicability),
            "omission_verdict": omission,
            "removability_verdict": removability,
            "result": deployment,
        }
    if set(by_model) != set(_PRIMARY_MODELS):
        raise ValueError("deployment evidence requires pinned Qwen 0.8B and Gemma 270M artifacts")
    ordered = [by_model[model] for model in _PRIMARY_MODELS]
    return derive_project_deployment_verdicts(mode, ordered), ordered


def _project_claims(
    mode: TokenizerMode,
    models: Sequence[Mapping[str, Any]],
    *,
    alignment_verdict: ClaimVerdict,
    cross_model_verdict: ClaimVerdict,
    deployment_verdicts: Mapping[str, ClaimVerdict],
) -> list[dict[str, object]]:
    model_claims = [_claim_verdicts(cast(Mapping[str, Any], model["replication"]).get("claims")) for model in models]
    cross_model_id = f"{'input' if mode == 'input_only' else 'output'}.cross_model_confirmation"
    verdicts: dict[str, ClaimVerdict] = {}
    for definition in directional_claims(mode):
        claim_id = definition.claim_id
        if claim_id == "input.fixed_subset_alignment_feasibility":
            verdict = alignment_verdict
        elif claim_id == cross_model_id:
            verdict = cross_model_verdict
        elif claim_id in deployment_verdicts:
            verdict = deployment_verdicts[claim_id]
        else:
            verdict = combine_claim_verdicts(claims[claim_id] for claims in model_claims)
        verdicts[claim_id] = verdict
    return claim_records(mode, verdicts)


def _merged_source_assets(models: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        f"{index}_{name}": dict(value)
        for index, model in enumerate(models)
        for name, value in cast(Mapping[str, Mapping[str, str]], cast(Mapping[str, Any], model["replication"])["source_assets"]).items()
    }


def _validation_paths(
    paths: tuple[Path, Path, Path] | None,
) -> tuple[SoftwareValidationInputs | None, dict[str, Path]]:
    if paths is None:
        return None, {}
    verification, input_synthetic, output_synthetic = paths
    return (
        load_software_validation_inputs(
            verification,
            input_synthetic,
            output_synthetic,
        ),
        validation_input_directories(
            verification,
            input_synthetic,
            output_synthetic,
        ),
    )


def assemble_project_artifact(
    primary_replications: Sequence[Path],
    output_dir: Path,
    *,
    alignment_studies: Sequence[Path] = (),
    deployments: Sequence[Path] = (),
    software_validation_paths: tuple[Path, Path, Path] | None = None,
) -> dict[str, Any]:
    mode, models = _load_primary_replications(primary_replications)
    software_validation, validation_inputs = _validation_paths(software_validation_paths)
    alignment_verdict, alignment = _alignment_study_verdicts(mode, alignment_studies)
    deployment_verdicts, deployment = _deployment_claims(mode, deployments)
    cross_model_verdict = combine_claim_verdicts(cast(ClaimVerdict, model["scientific_verdict"]) for model in models)
    claims = _project_claims(
        mode,
        models,
        alignment_verdict=alignment_verdict,
        cross_model_verdict=cross_model_verdict,
        deployment_verdicts=deployment_verdicts,
    )
    scientific_verdict = derive_input_headline_verdict(_claim_verdicts(claims)) if mode == "input_only" else cross_model_verdict
    first = cast(Mapping[str, Any], models[0]["replication"])
    if software_validation is not None and software_validation.verification.source != (
        SourceBinding(
            str(first["source_state_sha256"]),
            str(first["dependency_lock_sha256"]),
        )
    ):
        raise ValueError("software validation bundle source identity differs from project evidence")
    verification = {
        "all_provided": all(cast(Mapping[str, Any], model["replication"]).get("verification", {}).get("all_provided") is True for model in models),
        "all_passed": all(cast(Mapping[str, Any], model["replication"]).get("verification", {}).get("all_passed") is True for model in models),
        "models": [
            {
                "model": model["model"],
                "verification": cast(Mapping[str, Any], model["replication"])["verification"],
            }
            for model in models
        ],
    }
    project = {
        "mode": mode,
        "evidence_scope": "project",
        "operational_status": "completed",
        "scientific_verdict": scientific_verdict,
        "cross_model_verdict": cross_model_verdict,
        "claims": claims,
        "claim_traces": project_claim_trace_records(claims, models),
        "statement_traces": statement_trace_records(software_validation),
        "category_verdicts": claim_category_verdicts(claims),
        "models": models,
        "alignment_feasibility": alignment,
        "deployment": deployment,
        "source_commit": first["source_commit"],
        "source_dirty": first["source_dirty"],
        "source_state_sha256": first["source_state_sha256"],
        "dependency_lock_sha256": first["dependency_lock_sha256"],
        "installed_package": first["installed_package"],
        "claim_vocabulary_sha256": first["claim_vocabulary_sha256"],
        "source_assets": _merged_source_assets(models),
        "verification": verification,
    }
    output = RunDirectory(output_dir)
    output.write_json("project.json", project)
    output.write_text("project-report.md", project_report(project))
    parent_paths = {f"replication_{index}": directory / EVIDENCE_MANIFEST_FILENAME for index, directory in enumerate(primary_replications)}
    parent_paths.update({f"alignment_study_{index}": directory / EVIDENCE_MANIFEST_FILENAME for index, directory in enumerate(alignment_studies)})
    parent_paths.update({f"deployment_{index}": directory / EVIDENCE_MANIFEST_FILENAME for index, directory in enumerate(deployments)})
    model_revision = mapping_fingerprint({"models": [{"id": model[0], "revision": model[1]} for model in _PRIMARY_MODELS]})
    write_evidence_manifest(
        output_dir,
        EvidenceManifest(
            artifact_kind="project",
            mode=mode,
            status="completed",
            identity=EvidenceIdentity(
                source_commit=str(first["source_commit"]),
                source_dirty=bool(first["source_dirty"]),
                source_state_sha256=str(first["source_state_sha256"]),
                dependency_lock_sha256=str(first["dependency_lock_sha256"]),
                installed_package=dict(first["installed_package"]),
                claim_vocabulary_sha256=str(first["claim_vocabulary_sha256"]),
                source_assets=project["source_assets"],
                verification=verification,
                model_id="continuous-tokenizer/cross-model-primary-pair",
                model_revision=model_revision,
            ),
            parents=parent_paths,
            inputs=validation_inputs,
            artifacts={
                "project": output_dir / "project.json",
                "report": output_dir / "project-report.md",
            },
        ),
    )
    return project
