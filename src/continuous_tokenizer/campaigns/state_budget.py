from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from continuous_tokenizer.artifacts.evidence import (
    EVIDENCE_MANIFEST_FILENAME,
    EvidenceIdentity,
    EvidenceManifest,
    load_evidence_manifest,
    verify_artifact,
    write_evidence_manifest,
)
from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.artifacts.manifest import load_verified_run_manifest
from continuous_tokenizer.artifacts.state_budget_verification import (
    load_checkpoint_inventory,
)
from continuous_tokenizer.artifacts.store import RunDirectory, load_json_object
from continuous_tokenizer.contracts.claim_derivation import (
    GEMMA_MODEL,
    QWEN_MODEL,
)
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.contracts.state_budget import (
    CONTROL_DEDUPLICATION_POLICY,
    REFERENCE_DEDUPLICATION_POLICY,
    STATE_BUDGET_CONCLUSION,
    STATE_BUDGET_MAXIMUM_RATIO,
    STATE_BUDGET_PROJECT_IDENTITY_FIELDS,
    STATE_BUDGET_RUN_IDENTITY_FIELDS,
    STATE_BUDGET_SCOPE,
    STATE_BUDGET_VERSION,
    StateBudgetConfig,
    StateBudgetIdentity,
    StateBudgetNonClaims,
    StateBudgetResult,
    StateBudgetSeedResult,
    StateBudgetTensor,
    derive_state_budget_arithmetic,
    inventory_sha256,
    reference_inventory,
)
from continuous_tokenizer.reporting.state_budget_markdown import (
    state_budget_report,
)

_PRIMARY_MODELS: Final = (QWEN_MODEL, GEMMA_MODEL)
_SEEDS: Final = (17, 23, 41)


def _require_project(directory: Path, mode: str) -> dict[str, Any]:
    verification = verify_artifact(directory)
    if verification["valid"] is not True:
        errors = "; ".join(str(error) for error in verification["errors"])
        raise ValueError(
            f"{mode} project failed semantic verification: {errors}",
        )
    manifest = load_evidence_manifest(
        directory / EVIDENCE_MANIFEST_FILENAME,
    )
    project = dict(load_json_object(directory / "project.json"))
    if (
        manifest["artifact_kind"] != "project"
        or manifest["mode"] != mode
        or manifest["status"] != "completed"
        or project.get("mode") != mode
        or project.get("evidence_scope") != "project"
        or project.get("operational_status") != "completed"
    ):
        raise ValueError(
            f"state budget requires sealed FINAL {mode} project evidence",
        )
    return project


def _project_identity(project: Mapping[str, Any]) -> tuple[object, ...]:
    return tuple(project.get(name) for name in STATE_BUDGET_PROJECT_IDENTITY_FIELDS)


def _project_models(
    project: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    raw_models = project.get("models")
    if not isinstance(raw_models, Sequence) or isinstance(
        raw_models,
        str | bytes,
    ):
        raise ValueError("state-budget project has no normalized models")
    models: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise ValueError("state-budget project model is not canonical")
        model = raw.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("state-budget project model identity is missing")
        identity = (str(model.get("id")), str(model.get("revision")))
        if identity in models:
            raise ValueError("state-budget project contains duplicate models")
        models[identity] = cast(Mapping[str, Any], raw)
    if set(models) != set(_PRIMARY_MODELS):
        raise ValueError(
            "state budget requires pinned Qwen and Gemma project evidence",
        )
    return models


def _replication_runs(
    model: Mapping[str, Any],
) -> dict[int, Path]:
    replication = model.get("replication")
    if not isinstance(replication, Mapping):
        raise ValueError("state-budget model has no replication evidence")
    if replication.get("replication_complete") is not True or replication.get("operational_status") != "completed" or replication.get("profile") != "large":
        raise ValueError(
            "state budget requires complete FINAL Large-profile replications",
        )
    raw_runs = replication.get("runs")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, str | bytes):
        raise ValueError("state-budget replication has no final runs")
    runs: dict[int, Path] = {}
    for raw in raw_runs:
        if not isinstance(raw, Mapping):
            raise ValueError("state-budget run entry is not canonical")
        seed = raw.get("seed")
        directory = raw.get("directory")
        if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(directory, str) or not directory or seed in runs:
            raise ValueError(
                "state-budget runs contain duplicate or invalid identities",
            )
        if raw.get("evidence_scope") != "final" or raw.get("operational_status") != "completed":
            raise ValueError("state budget refuses non-final run evidence")
        runs[seed] = Path(directory)
    if set(runs) != set(_SEEDS):
        raise ValueError(
            "state budget requires final seeds 17, 23, and 41",
        )
    return runs


def _run_checkpoint(
    directory: Path,
    *,
    mode: str,
    model: tuple[str, str],
    seed: int,
) -> tuple[Any, Path]:
    manifest = load_verified_run_manifest(
        directory / "manifest-final.json",
    )
    checkpoint_relative = manifest.artifacts.get("checkpoint")
    if (
        manifest.status != "passed"
        or manifest.mode != mode
        or (manifest.model_id, manifest.model_revision) != model
        or manifest.seed != seed
        or checkpoint_relative is None
    ):
        raise ValueError(
            "state-budget run manifest identity or checkpoint is invalid",
        )
    checkpoint = directory / checkpoint_relative
    return manifest, checkpoint


def _paired_identity(input_manifest: Any, output_manifest: Any) -> None:
    if any(getattr(input_manifest, name) != getattr(output_manifest, name) for name in STATE_BUDGET_RUN_IDENTITY_FIELDS):
        raise ValueError(
            "paired input/output project run identity mismatch",
        )


def _input_metrics(
    directory: Path,
    inventory: Sequence[StateBudgetTensor],
) -> Mapping[str, Any]:
    metrics = load_json_object(directory / "tokenizer-metrics.json")
    raw_inventory = metrics.get("deployment_tensors")
    expected = {
        row.name: {
            "shape": list(row.shape),
            "dtype": row.dtype,
            "bytes": row.bytes,
        }
        for row in inventory
    }
    if raw_inventory != expected:
        raise ValueError(
            "input checkpoint inventory differs from sealed raw metrics",
        )
    return metrics


def _asset_sha256(manifest: Any, name: str) -> str:
    asset = manifest.source_assets.get(name)
    if not isinstance(asset, Mapping):
        raise ValueError(f"paired run lacks source asset {name}")
    return str(asset["sha256"])


def _seed_result(
    model: tuple[str, str],
    seed: int,
    input_directory: Path,
    output_directory: Path,
) -> StateBudgetSeedResult:
    input_manifest, input_checkpoint = _run_checkpoint(
        input_directory,
        mode="input_only",
        model=model,
        seed=seed,
    )
    output_manifest, output_checkpoint = _run_checkpoint(
        output_directory,
        mode="output_only",
        model=model,
        seed=seed,
    )
    _paired_identity(input_manifest, output_manifest)
    input_inventory, input_metadata = load_checkpoint_inventory(
        input_checkpoint,
        "input_only",
    )
    output_inventory, output_metadata = load_checkpoint_inventory(
        output_checkpoint,
        "output_only",
    )
    expected_metadata = {
        "model_id": model[0],
        "model_revision": model[1],
    }
    if any(metadata.get(name) != value for metadata in (input_metadata, output_metadata) for name, value in expected_metadata.items()):
        raise ValueError("checkpoint model identity mismatch")

    input_metrics = _input_metrics(input_directory, input_inventory)
    output_metrics = load_json_object(
        output_directory / "output-metrics.json",
    )
    model_metrics = input_metrics.get("model")
    compactness = input_metrics.get("compactness")
    if not isinstance(model_metrics, Mapping) or not isinstance(
        compactness,
        Mapping,
    ):
        raise ValueError("input run lacks sealed compactness inventory")
    tied = input_metadata.get("tie_word_embeddings")
    if not isinstance(tied, bool) or model_metrics.get("tie_word_embeddings") is not tied:
        raise ValueError("paired model tied-table identity is invalid")
    input_reference_bytes = compactness.get("reference_state_bytes")
    output_reference_bytes = output_metrics.get("reference_state_bytes")
    references = reference_inventory(
        (input_manifest.model_id, input_manifest.model_revision),
        input_manifest.source_assets,
        input_metadata,
        input_bytes=input_reference_bytes,
        output_bytes=output_reference_bytes,
    )
    arithmetic = derive_state_budget_arithmetic(
        input_inventory,
        output_inventory,
        references,
        tied=tied,
    )
    identity = StateBudgetIdentity(
        source_commit=input_manifest.source_commit,
        source_dirty=input_manifest.source_dirty,
        source_state_sha256=input_manifest.source_state_sha256,
        dependency_lock_sha256=input_manifest.dependency_lock_sha256,
        installed_package_sha256=input_manifest.installed_package["content_sha256"],
        claim_vocabulary_sha256=input_manifest.claim_vocabulary_sha256,
        model_config_sha256=_asset_sha256(
            input_manifest,
            "model_config",
        ),
        input_embedding_sha256=_asset_sha256(
            input_manifest,
            "input_embedding_tensor",
        ),
        tokenizer_vocabulary_sha256=_asset_sha256(
            input_manifest,
            "tokenizer_vocabulary",
        ),
        input_contract_sha256=sha256_path(
            input_directory / "experiment.json",
        ),
        output_contract_sha256=sha256_path(
            output_directory / "experiment.json",
        ),
    )
    return StateBudgetSeedResult(
        model_id=model[0],
        model_revision=model[1],
        seed=seed,
        tie_word_embeddings=tied,
        identity=identity,
        input_checkpoint_sha256=sha256_path(input_checkpoint),
        output_checkpoint_sha256=sha256_path(output_checkpoint),
        input_inventory=input_inventory,
        output_inventory=output_inventory,
        reference_inventory=references,
        input_inventory_sha256=inventory_sha256(input_inventory),
        output_inventory_sha256=inventory_sha256(output_inventory),
        reference_inventory_sha256=inventory_sha256(
            references,
        ),
        reference_deduplication_policy=REFERENCE_DEDUPLICATION_POLICY,
        control_deduplication_policy=CONTROL_DEDUPLICATION_POLICY,
        arithmetic=arithmetic,
        ratio=(arithmetic.candidate_tensor_state_bytes / arithmetic.reference_tensor_state_bytes),
    )


def calculate_state_budget(
    input_project_directory: Path,
    output_project_directory: Path,
) -> StateBudgetResult:
    input_project = _require_project(
        input_project_directory,
        "input_only",
    )
    output_project = _require_project(
        output_project_directory,
        "output_only",
    )
    if _project_identity(input_project) != _project_identity(output_project):
        raise ValueError("input/output project identity mismatch")
    input_models = _project_models(input_project)
    output_models = _project_models(output_project)
    rows = []
    for model in _PRIMARY_MODELS:
        input_runs = _replication_runs(input_models[model])
        output_runs = _replication_runs(output_models[model])
        rows.extend(
            _seed_result(
                model,
                seed,
                input_runs[seed],
                output_runs[seed],
            )
            for seed in _SEEDS
        )
    worst = max(row.ratio for row in rows)
    verdict = "supported" if worst <= STATE_BUDGET_MAXIMUM_RATIO else "unsupported"
    return StateBudgetResult(
        version=STATE_BUDGET_VERSION,
        evidence_scope=STATE_BUDGET_SCOPE,
        operational_status="completed",
        config=StateBudgetConfig(),
        conclusion=STATE_BUDGET_CONCLUSION,
        verdict=verdict,
        non_claims=StateBudgetNonClaims(),
        per_seed=tuple(rows),
        worst_case_ratio=worst,
    )


def run_state_budget(
    input_project_directory: Path,
    output_project_directory: Path,
    output_dir: Path,
) -> dict[str, object]:
    result = calculate_state_budget(
        input_project_directory,
        output_project_directory,
    )
    output = RunDirectory(output_dir)
    output.write_json("joint-state-budget.json", result.to_dict())
    output.write_text(
        "joint-state-budget-report.md",
        state_budget_report(result),
    )
    input_manifest = load_evidence_manifest(
        input_project_directory / EVIDENCE_MANIFEST_FILENAME,
    )
    source = cast(Mapping[str, Any], input_manifest["source"])
    model_revision = mapping_fingerprint(
        {
            "models": [{"id": model_id, "revision": revision} for model_id, revision in _PRIMARY_MODELS],
            "seeds": list(_SEEDS),
        },
    )
    write_evidence_manifest(
        output_dir,
        EvidenceManifest(
            artifact_kind="state_budget",
            mode="cross_directional",
            status="completed",
            identity=EvidenceIdentity(
                source_commit=str(source["commit"]),
                source_dirty=bool(source["dirty"]),
                source_state_sha256=str(source["state_sha256"]),
                dependency_lock_sha256=str(
                    input_manifest["dependency_lock_sha256"],
                ),
                installed_package=cast(
                    Mapping[str, str],
                    input_manifest["installed_package"],
                ),
                claim_vocabulary_sha256=str(
                    input_manifest["claim_vocabulary_sha256"],
                ),
                source_assets=cast(
                    Mapping[str, Mapping[str, str]],
                    input_manifest["source_assets"],
                ),
                verification=cast(
                    Mapping[str, Any],
                    input_manifest["verification"],
                ),
                model_id=("continuous-tokenizer/cross-directional-primary-pair"),
                model_revision=model_revision,
            ),
            parents={
                "input_project": (input_project_directory / EVIDENCE_MANIFEST_FILENAME),
                "output_project": (output_project_directory / EVIDENCE_MANIFEST_FILENAME),
            },
            inputs={},
            artifacts={
                "state_budget": output_dir / "joint-state-budget.json",
                "report": (output_dir / "joint-state-budget-report.md"),
            },
        ),
    )
    return result.to_dict()
