from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, cast, final

from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256
from continuous_tokenizer.contracts.experiment import (
    TRAINING_STAGES,
    TokenizerMode,
    TrainingStage,
)
from continuous_tokenizer.contracts.parsing import (
    is_lowercase_sha256,
    strict_fields,
    table,
)

type ManifestStatus = Literal["running", "passed", "failed"]


@dataclass(frozen=True, slots=True)
class _ManifestTables:
    environment: Mapping[str, Any]
    codec_attention: Mapping[str, Any]
    installed_package: Mapping[str, Any]
    source_assets: Mapping[str, Any]
    inputs: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    artifact_hashes: Mapping[str, Any]
    verification: Mapping[str, Any]

    @classmethod
    def parse(cls, values: Mapping[str, Any]) -> _ManifestTables:
        return cls(
            environment=table(values["environment"], "manifest.environment"),
            codec_attention=table(values["codec_attention"], "manifest.codec_attention"),
            installed_package=table(values["installed_package"], "manifest.installed_package"),
            source_assets=table(values["source_assets"], "manifest.source_assets"),
            inputs=table(values["inputs"], "manifest.inputs"),
            artifacts=table(values["artifacts"], "manifest.artifacts"),
            artifact_hashes=table(values["artifact_hashes"], "manifest.artifact_hashes"),
            verification=table(values["verification"], "manifest.verification"),
        )


def _manifest_status(values: Mapping[str, Any], path: Path) -> ManifestStatus:
    status = values.get("status")
    if status not in {"running", "passed", "failed"}:
        raise ValueError(f"invalid manifest status in {path}")
    return cast(ManifestStatus, status)


def _manifest_stages(values: Mapping[str, Any], path: Path) -> tuple[TrainingStage, ...]:
    stages = tuple(values["stages"])
    if any(stage not in TRAINING_STAGES for stage in stages):
        raise ValueError(f"invalid training stage in {path}")
    return cast(tuple[TrainingStage, ...], stages)


def _validate_manifest_header(
    values: Mapping[str, Any],
    path: Path,
) -> bool:
    if values["mode"] not in {"input_only", "output_only"}:
        raise ValueError(f"invalid tokenizer mode in {path}")
    if values["codec_direction"] != values["mode"]:
        raise ValueError(f"manifest codec direction does not match its mode in {path}")
    if not isinstance(values["native_head_used"], bool):
        raise ValueError(f"manifest native_head_used must be boolean in {path}")
    source_dirty = values["source_dirty"]
    if not isinstance(source_dirty, bool):
        raise ValueError(f"manifest source_dirty must be boolean in {path}")
    return source_dirty


_NON_EMPTY_MANIFEST_FIELDS = (
    "experiment_name",
    "experiment_fingerprint",
    "replication_fingerprint",
    "model_id",
    "model_revision",
    "dataset_id",
    "dataset_revision",
    "source_commit",
    "source_state_sha256",
    "dependency_lock_sha256",
    "claim_vocabulary_sha256",
    "feedback_policy",
)
_MANIFEST_HASH_FIELDS = (
    "experiment_fingerprint",
    "replication_fingerprint",
    "source_state_sha256",
    "dependency_lock_sha256",
    "claim_vocabulary_sha256",
)


def _validate_manifest_strings(
    values: Mapping[str, Any],
    path: Path,
) -> None:
    for name in _NON_EMPTY_MANIFEST_FIELDS:
        if not isinstance(values[name], str) or not values[name]:
            raise ValueError(f"manifest {name} must be a non-empty string in {path}")
    for name in _MANIFEST_HASH_FIELDS:
        if not is_lowercase_sha256(values[name]):
            raise ValueError(f"manifest {name} must be a lowercase SHA-256 digest in {path}")
    if values["claim_vocabulary_sha256"] != CLAIM_VOCABULARY_SHA256:
        raise ValueError(f"manifest claim vocabulary hash mismatch in {path}")


def _validate_installed_package(
    installed_package: Mapping[str, Any],
    path: Path,
) -> None:
    if set(installed_package) != {"name", "version", "content_sha256"}:
        raise ValueError(f"manifest installed package is not canonical in {path}")
    if not is_lowercase_sha256(installed_package["content_sha256"]):
        raise ValueError(f"manifest installed package hash is invalid in {path}")


def _validate_portable_entries(
    sections: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    for section_name, section in sections.items():
        for name, raw_entry in section.items():
            entry = table(raw_entry, f"manifest.{section_name}.{name}")
            if set(entry) != {"locator", "sha256"}:
                raise ValueError(f"manifest {section_name} entry {name!r} is not canonical in {path}")
            if not isinstance(entry["locator"], str) or not entry["locator"]:
                raise ValueError(f"manifest {section_name} entry {name!r} has no locator in {path}")
            if not is_lowercase_sha256(entry["sha256"]):
                raise ValueError(f"manifest {section_name} entry {name!r} has an invalid hash in {path}")


def _validate_sealed_artifacts(
    values: Mapping[str, Any],
    path: Path,
    *,
    status: ManifestStatus,
    tables: _ManifestTables,
) -> None:
    if set(tables.artifact_hashes) != set(tables.artifacts):
        raise ValueError(f"manifest artifact hashes do not match declared artifacts in {path}")
    if status != "running" and "experiment" not in tables.artifacts:
        raise ValueError(f"final manifest does not seal experiment.json in {path}")
    if status != "running" and tables.verification.get("provided") is True and "verification" not in tables.artifacts:
        raise ValueError(f"final manifest does not seal verification logs in {path}")
    if status != "running" and values["embedding_tensor"] is not None and not tables.source_assets:
        raise ValueError(f"final manifest does not seal source assets in {path}")


def _portable_entries(
    entries: Mapping[str, Any],
    section_name: str,
) -> dict[str, dict[str, str]]:
    return {
        str(name): {
            str(key): str(value)
            for key, value in table(
                entry,
                f"manifest.{section_name}.{name}",
            ).items()
        }
        for name, entry in entries.items()
    }


@final
@dataclass(frozen=True, slots=True)
class RunManifest:
    experiment_name: str
    mode: TokenizerMode
    codec_direction: TokenizerMode
    experiment_fingerprint: str
    replication_fingerprint: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    embedding_tensor: str | None
    source_dtype: str | None
    seed: int
    stages: tuple[TrainingStage, ...]
    source_commit: str
    source_dirty: bool
    source_state_sha256: str
    dependency_lock_sha256: str
    installed_package: Mapping[str, str]
    claim_vocabulary_sha256: str
    source_assets: Mapping[str, Mapping[str, str]]
    inputs: Mapping[str, Mapping[str, str]]
    codec_attention: Mapping[str, Any]
    environment: Mapping[str, str]
    trainable_parameters: tuple[str, ...]
    frozen_backbone_fingerprint: str | None
    native_head_used: bool
    feedback_policy: str
    artifacts: Mapping[str, str]
    artifact_hashes: Mapping[str, str]
    status: Literal["running", "passed", "failed"]
    verification: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"expected a JSON object in {path}")
        values = dict(cast(Mapping[str, Any], loaded))
        field_names = {definition.name for definition in fields(cast(Any, cls))}
        strict_fields(values, field_names, "manifest")
        missing = sorted(field_names - set(values))
        if missing:
            raise ValueError(f"missing manifest fields: {', '.join(missing)}")
        status = _manifest_status(values, path)
        stages = _manifest_stages(values, path)
        source_dirty = _validate_manifest_header(values, path)
        tables = _ManifestTables.parse(values)
        _validate_manifest_strings(values, path)
        _validate_installed_package(tables.installed_package, path)
        _validate_portable_entries(
            {
                "source_assets": tables.source_assets,
                "inputs": tables.inputs,
            },
            path,
        )
        _validate_sealed_artifacts(
            values,
            path,
            status=status,
            tables=tables,
        )
        return cls(
            experiment_name=str(values["experiment_name"]),
            mode=cast(TokenizerMode, values["mode"]),
            codec_direction=cast(TokenizerMode, values["codec_direction"]),
            experiment_fingerprint=str(values["experiment_fingerprint"]),
            replication_fingerprint=str(values["replication_fingerprint"]),
            model_id=str(values["model_id"]),
            model_revision=str(values["model_revision"]),
            dataset_id=str(values["dataset_id"]),
            dataset_revision=str(values["dataset_revision"]),
            embedding_tensor=(None if values["embedding_tensor"] is None else str(values["embedding_tensor"])),
            source_dtype=None if values["source_dtype"] is None else str(values["source_dtype"]),
            seed=int(values["seed"]),
            stages=stages,
            source_commit=str(values["source_commit"]),
            source_dirty=source_dirty,
            source_state_sha256=str(values["source_state_sha256"]),
            dependency_lock_sha256=str(values["dependency_lock_sha256"]),
            installed_package={str(key): str(value) for key, value in tables.installed_package.items()},
            claim_vocabulary_sha256=str(values["claim_vocabulary_sha256"]),
            source_assets=_portable_entries(tables.source_assets, "source_assets"),
            inputs=_portable_entries(tables.inputs, "inputs"),
            codec_attention=dict(tables.codec_attention),
            environment={str(key): str(value) for key, value in tables.environment.items()},
            trainable_parameters=tuple(values["trainable_parameters"]),
            frozen_backbone_fingerprint=(None if values["frozen_backbone_fingerprint"] is None else str(values["frozen_backbone_fingerprint"])),
            native_head_used=bool(values["native_head_used"]),
            feedback_policy=str(values["feedback_policy"]),
            artifacts={str(key): str(value) for key, value in tables.artifacts.items()},
            artifact_hashes={str(key): str(value) for key, value in tables.artifact_hashes.items()},
            status=status,
            verification=dict(tables.verification),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
