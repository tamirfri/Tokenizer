from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.artifacts.manifest import load_verified_run_manifest
from continuous_tokenizer.artifacts.store import load_json_object
from continuous_tokenizer.contracts.state_budget import (
    STATE_BUDGET_PROJECT_IDENTITY_FIELDS,
    STATE_BUDGET_RUN_IDENTITY_FIELDS,
    StateBudgetResult,
    StateBudgetTensor,
    reference_inventory,
)


def _project_runs(
    project: Mapping[str, Any],
    expected_mode: str,
) -> dict[tuple[str, str, int], Path]:
    if project.get("mode") != expected_mode or project.get("evidence_scope") != "project" or project.get("operational_status") != "completed":
        raise ValueError(f"state-budget parent is not a FINAL {expected_mode} project")
    models = project.get("models")
    if not isinstance(models, Sequence) or isinstance(models, str | bytes):
        raise ValueError("state-budget parent project has no models")
    indexed: dict[tuple[str, str, int], Path] = {}
    for entry in models:
        if not isinstance(entry, Mapping):
            raise ValueError("state-budget parent model is not canonical")
        model = entry.get("model")
        replication = entry.get("replication")
        if not isinstance(model, Mapping) or not isinstance(replication, Mapping):
            raise ValueError("state-budget parent model evidence is incomplete")
        runs = replication.get("runs")
        if not isinstance(runs, Sequence) or isinstance(runs, str | bytes):
            raise ValueError("state-budget parent replication has no runs")
        for run in runs:
            if not isinstance(run, Mapping):
                raise ValueError("state-budget parent run is not canonical")
            identity = (
                str(model["id"]),
                str(model["revision"]),
                int(run["seed"]),
            )
            if identity in indexed:
                raise ValueError("state-budget parent contains duplicate runs")
            indexed[identity] = Path(str(run["directory"]))
    return indexed


def _asset_hash(manifest: Any, name: str) -> str:
    asset = manifest.source_assets.get(name)
    if not isinstance(asset, Mapping):
        raise ValueError(f"state-budget run lacks source asset {name!r}")
    return str(asset["sha256"])


def load_checkpoint_inventory(
    checkpoint: Path,
    direction: str,
) -> tuple[tuple[StateBudgetTensor, ...], Mapping[str, Any]]:
    import torch
    from torch import Tensor

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {
        "direction",
        "config",
        "metadata",
        "state_dict",
        "control_ids",
        *({"control_embeddings"} if direction == "input_only" else set()),
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("direction") != direction
        or not isinstance(payload.get("state_dict"), Mapping)
        or not isinstance(payload.get("metadata"), Mapping)
    ):
        raise ValueError(f"state-budget {direction} checkpoint is not canonical")
    tensors = [
        *(
            (f"codec.{name}", tensor)
            for name, tensor in sorted(
                cast(
                    Mapping[str, object],
                    payload["state_dict"],
                ).items(),
            )
        ),
        ("controls.ids", payload["control_ids"]),
        *((("controls.embeddings", payload["control_embeddings"]),) if direction == "input_only" else ()),
    ]
    rows: list[StateBudgetTensor] = []
    for name, value in tensors:
        if not isinstance(value, Tensor):
            raise ValueError(f"state-budget checkpoint row {name!r} is not a tensor")
        raw = value.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
        rows.append(
            StateBudgetTensor.from_mapping(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        )
    return tuple(rows), cast(Mapping[str, Any], payload["metadata"])


def _reference_inventory(
    manifest: Any,
    input_directory: Path,
    output_directory: Path,
    metadata: Mapping[str, Any],
) -> list[dict[str, object]]:
    input_metrics = load_json_object(input_directory / "tokenizer-metrics.json")
    output_metrics = load_json_object(output_directory / "output-metrics.json")
    compactness = input_metrics.get("compactness")
    model = input_metrics.get("model")
    tied = metadata.get("tie_word_embeddings")
    if not isinstance(compactness, Mapping) or not isinstance(model, Mapping) or model.get("tie_word_embeddings") is not tied:
        raise ValueError("state-budget reference tensor evidence is not canonical")
    input_bytes = compactness.get("reference_state_bytes")
    output_bytes = output_metrics.get("reference_state_bytes")
    return [
        row.to_dict()
        for row in reference_inventory(
            (manifest.model_id, manifest.model_revision),
            manifest.source_assets,
            metadata,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )
    ]


def _run_identity_errors(
    row: Any,
    input_directory: Path,
    output_directory: Path,
) -> list[str]:
    input_manifest = load_verified_run_manifest(
        input_directory / "manifest-final.json",
    )
    output_manifest = load_verified_run_manifest(
        output_directory / "manifest-final.json",
    )
    errors = []
    if any(getattr(input_manifest, name) != getattr(output_manifest, name) for name in STATE_BUDGET_RUN_IDENTITY_FIELDS):
        errors.append("state-budget paired run identity mismatch")
    identity = row.identity
    expected_identity = (
        input_manifest.source_commit,
        input_manifest.source_dirty,
        input_manifest.source_state_sha256,
        input_manifest.dependency_lock_sha256,
        input_manifest.installed_package["content_sha256"],
        input_manifest.claim_vocabulary_sha256,
        _asset_hash(input_manifest, "model_config"),
        _asset_hash(input_manifest, "input_embedding_tensor"),
        _asset_hash(input_manifest, "tokenizer_vocabulary"),
        sha256_path(input_directory / "experiment.json"),
        sha256_path(output_directory / "experiment.json"),
    )
    if (
        identity.source_commit,
        identity.source_dirty,
        identity.source_state_sha256,
        identity.dependency_lock_sha256,
        identity.installed_package_sha256,
        identity.claim_vocabulary_sha256,
        identity.model_config_sha256,
        identity.input_embedding_sha256,
        identity.tokenizer_vocabulary_sha256,
        identity.input_contract_sha256,
        identity.output_contract_sha256,
    ) != expected_identity:
        errors.append("state-budget sealed run identity differs from parent")
    input_checkpoint = input_directory / str(input_manifest.artifacts["checkpoint"])
    output_checkpoint = output_directory / str(output_manifest.artifacts["checkpoint"])
    if row.input_checkpoint_sha256 != sha256_path(input_checkpoint) or row.output_checkpoint_sha256 != sha256_path(output_checkpoint):
        errors.append("state-budget checkpoint hashes differ from parent runs")
    input_inventory, input_metadata = load_checkpoint_inventory(
        input_checkpoint,
        "input_only",
    )
    output_inventory, _ = load_checkpoint_inventory(
        output_checkpoint,
        "output_only",
    )
    if row.input_inventory != input_inventory:
        errors.append("state-budget input inventory differs from sealed checkpoint")
    if row.output_inventory != output_inventory:
        errors.append("state-budget output inventory differs from sealed checkpoint")
    reference_inventory = _reference_inventory(
        input_manifest,
        input_directory,
        output_directory,
        input_metadata,
    )
    if [tensor.to_dict() for tensor in row.reference_inventory] != reference_inventory:
        errors.append("state-budget reference inventory differs from sealed evidence")
    return errors


def _parent_errors(
    result: StateBudgetResult,
    input_project_directory: Path,
    output_project_directory: Path,
) -> list[str]:
    input_project = load_json_object(input_project_directory / "project.json")
    output_project = load_json_object(output_project_directory / "project.json")
    errors = []
    if any(input_project.get(name) != output_project.get(name) for name in STATE_BUDGET_PROJECT_IDENTITY_FIELDS):
        errors.append("state-budget project identity mismatch")
    input_runs = _project_runs(input_project, "input_only")
    output_runs = _project_runs(output_project, "output_only")
    expected = set(input_runs)
    actual = {(row.model_id, row.model_revision, row.seed) for row in result.per_seed}
    if expected != set(output_runs) or actual != expected:
        return [
            *errors,
            "state-budget model-seed identities differ from parent projects",
        ]
    for row in result.per_seed:
        identity = (row.model_id, row.model_revision, row.seed)
        errors.extend(
            _run_identity_errors(
                row,
                input_runs[identity],
                output_runs[identity],
            )
        )
    return errors


def verify_state_budget(
    budget: Mapping[str, Any],
    root: Path,
    manifest: Mapping[str, Any],
) -> list[str]:
    try:
        parsed = StateBudgetResult.from_mapping(budget)
    except (TypeError, ValueError) as error:
        return [str(error)]
    parents = cast(Mapping[str, Any], manifest["parents"])
    if set(parents) != {"input_project", "output_project"}:
        return ["state budget must declare exactly both directional projects"]
    try:
        input_manifest = (root / str(cast(Mapping[str, Any], parents["input_project"])["locator"])).resolve()
        output_manifest = (root / str(cast(Mapping[str, Any], parents["output_project"])["locator"])).resolve()
        return _parent_errors(
            parsed,
            input_manifest.parent,
            output_manifest.parent,
        )
    except (KeyError, TypeError, ValueError) as error:
        return [f"state budget cannot be recomputed from parents: {error}"]
