from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from continuous_tokenizer.artifacts.hashing import (
    directory_files,
    installed_distribution_identity,
    sha256_path,
)
from continuous_tokenizer.artifacts.manifest import load_verified_run_manifest
from continuous_tokenizer.artifacts.store import (
    load_json_object,
    write_json_atomic,
)
from continuous_tokenizer.contracts.claim_derivation import (
    FINAL_VERIFICATION_CHECKS,
    GEMMA_MODEL,
    QWEN_MODEL,
    WIKITEXT_DATASET,
    derive_input_claim_verdicts,
    derive_input_headline_verdict,
    derive_output_claim_verdicts,
    derive_project_alignment_verdict,
    derive_project_deployment_verdicts,
)
from continuous_tokenizer.contracts.claims import (
    CLAIM_RECORD_FIELDS,
    CLAIM_VERDICTS,
    CLAIM_VOCABULARY_SHA256,
    ClaimVerdict,
    claim_category_verdicts,
    claim_record,
    combine_claim_verdicts,
    directional_claims,
    project_claim_trace_records,
)
from continuous_tokenizer.contracts.input_study import (
    alignment_feasibility_verdict,
)
from continuous_tokenizer.contracts.parsing import is_lowercase_sha256
from continuous_tokenizer.contracts.performance import (
    output_performance_errors,
    performance_ablation_errors,
    performance_claim_context_errors,
    prefill_performance_errors,
    tokenizer_performance_errors,
)
from continuous_tokenizer.contracts.profiles import CAMPAIGN_PROFILE_NAME
from continuous_tokenizer.contracts.prospective import (
    PROSPECTIVE_ARTIFACT_KINDS,
    prospective_result_errors,
)
from continuous_tokenizer.contracts.prospective_subset import (
    PROSPECTIVE_INPUT_SUBSET_FILENAME,
    prospective_vocabulary_subset_errors,
)
from continuous_tokenizer.contracts.statements import (
    SourceBinding,
    statement_trace_records,
)

EVIDENCE_MANIFEST_FILENAME: Final = "evidence-manifest.json"
_STUDENT_T_95_DF_2: Final = 4.302652729911275
_REQUIRED_SOURCE_ASSETS: Final = frozenset(
    {
        "model_config",
        "input_embedding_tensor",
        "tokenizer_vocabulary",
    }
)
EvidenceKind = Literal[
    "deployment",
    "performance_ablation",
    "project",
    "replication",
    "search",
    "state_budget",
    "study",
    "prospective_mechanism_smoke",
    "prospective_feasibility_screen",
    "prospective_candidate_selection",
    "prospective_final_evidence",
]


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    source_commit: str
    source_dirty: bool
    source_state_sha256: str
    dependency_lock_sha256: str
    installed_package: Mapping[str, str]
    claim_vocabulary_sha256: str
    source_assets: Mapping[str, Mapping[str, str]]
    verification: Mapping[str, Any]
    model_id: str
    model_revision: str


@dataclass(frozen=True, slots=True)
class EvidenceManifest:
    artifact_kind: EvidenceKind
    mode: str
    status: str
    identity: EvidenceIdentity
    parents: Mapping[str, Path]
    inputs: Mapping[str, Path]
    artifacts: Mapping[str, Path]


def _artifact_entry(path: Path, base_directory: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {
        "locator": Path(os.path.relpath(resolved, start=base_directory.resolve())).as_posix(),
        "sha256": sha256_path(resolved),
    }


def write_evidence_manifest(
    output_dir: Path,
    declaration: EvidenceManifest,
) -> dict[str, Any]:
    identity = declaration.identity
    for name, value in (
        ("source_commit", identity.source_commit),
        ("source_state_sha256", identity.source_state_sha256),
        ("dependency_lock_sha256", identity.dependency_lock_sha256),
        ("claim_vocabulary_sha256", identity.claim_vocabulary_sha256),
        ("model_id", identity.model_id),
        ("model_revision", identity.model_revision),
    ):
        if not value:
            raise ValueError(f"evidence identity {name} must not be empty")
    for name, digest in (
        ("source_state_sha256", identity.source_state_sha256),
        ("dependency_lock_sha256", identity.dependency_lock_sha256),
        ("claim_vocabulary_sha256", identity.claim_vocabulary_sha256),
    ):
        if not is_lowercase_sha256(digest):
            raise ValueError(f"evidence identity {name} must be a lowercase SHA-256 digest")
    if set(identity.installed_package) != {"name", "version", "content_sha256"}:
        raise ValueError("evidence installed package identity is not canonical")
    if not is_lowercase_sha256(identity.installed_package["content_sha256"]):
        raise ValueError("evidence installed package hash is invalid")
    for name, path in (
        *declaration.parents.items(),
        *declaration.inputs.items(),
        *declaration.artifacts.items(),
    ):
        if not path.exists():
            raise FileNotFoundError(f"declared evidence path is missing: {name}: {path}")
    manifest = {
        "artifact_kind": declaration.artifact_kind,
        "mode": declaration.mode,
        "status": declaration.status,
        "source": {
            "commit": identity.source_commit,
            "dirty": identity.source_dirty,
            "state_sha256": identity.source_state_sha256,
        },
        "dependency_lock_sha256": identity.dependency_lock_sha256,
        "installed_package": dict(identity.installed_package),
        "claim_vocabulary_sha256": identity.claim_vocabulary_sha256,
        "source_assets": {name: dict(value) for name, value in sorted(identity.source_assets.items())},
        "verification": dict(identity.verification),
        "model": {
            "id": identity.model_id,
            "revision": identity.model_revision,
        },
        "parents": {name: _artifact_entry(path, output_dir) for name, path in sorted(declaration.parents.items())},
        "inputs": {name: _artifact_entry(path, output_dir) for name, path in sorted(declaration.inputs.items())},
        "artifacts": {name: _artifact_entry(path, output_dir) for name, path in sorted(declaration.artifacts.items())},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / EVIDENCE_MANIFEST_FILENAME, manifest)
    return manifest


def seal_generated_evidence(
    spec_path: Path,
    output_dir: Path,
    *,
    artifact_kind: Literal["search", "study"],
) -> dict[str, Any] | None:
    primary_name = "search.json" if artifact_kind == "search" else "manifest-final.json"
    primary_path = output_dir / primary_name
    if not primary_path.is_file():
        return None
    primary = load_json_object(primary_path)
    result_path = output_dir / "result.json"
    result = load_json_object(result_path) if result_path.is_file() else primary
    with spec_path.open("rb") as handle:
        registered = tomllib.load(handle)
    inputs = {"spec": spec_path}
    parents: dict[str, Path] = {}
    for name in ("experiment", "final_experiment"):
        relative = registered.get(name)
        if isinstance(relative, str):
            path = (spec_path.parent / relative).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"declared {name} contract is missing: {path}")
            inputs[name] = path
    oracle_study = primary.get("oracle_study")
    if isinstance(oracle_study, Mapping):
        artifact = oracle_study.get("artifact")
        if isinstance(artifact, str):
            oracle_path = Path(artifact)
            if not oracle_path.is_file():
                raise FileNotFoundError(f"declared oracle-study artifact is missing: {oracle_path}")
            inputs["oracle_study"] = oracle_path
            parent_manifest = oracle_path.parent / EVIDENCE_MANIFEST_FILENAME
            if parent_manifest.is_file():
                parents["oracle_study"] = parent_manifest
    model = result.get("model")
    if isinstance(model, Mapping):
        model_id = str(model["id"])
        model_revision = str(model["revision"])
    else:
        model_id = str(primary["model_id"])
        model_revision = str(primary["model_revision"])
    artifacts = {child.name: child for child in output_dir.iterdir() if child.name != EVIDENCE_MANIFEST_FILENAME}
    return write_evidence_manifest(
        output_dir,
        EvidenceManifest(
            artifact_kind=artifact_kind,
            mode=str(result["mode"]),
            status=str(result["operational_status"]),
            identity=EvidenceIdentity(
                source_commit=str(primary["source_commit"]),
                source_dirty=bool(primary["source_dirty"]),
                source_state_sha256=str(primary["source_state_sha256"]),
                dependency_lock_sha256=str(primary["dependency_lock_sha256"]),
                installed_package=installed_distribution_identity("continuous-byte-tokenizer"),
                claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
                source_assets={},
                verification=dict(cast(Mapping[str, Any], primary["verification"])),
                model_id=model_id,
                model_revision=model_revision,
            ),
            parents=parents,
            inputs=inputs,
            artifacts=artifacts,
        ),
    )


_EVIDENCE_MANIFEST_FIELDS: Final = {
    "artifact_kind",
    "mode",
    "status",
    "source",
    "dependency_lock_sha256",
    "installed_package",
    "claim_vocabulary_sha256",
    "source_assets",
    "verification",
    "model",
    "parents",
    "inputs",
    "artifacts",
}


def _canonical_mapping(
    value: object,
    expected_fields: set[str],
    message: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(message)
    return cast(Mapping[str, Any], value)


def _evidence_identity(
    manifest: Mapping[str, Any],
    path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    source = _canonical_mapping(
        manifest["source"],
        {"commit", "dirty", "state_sha256"},
        f"evidence manifest identity is not canonical in {path}",
    )
    model = _canonical_mapping(
        manifest["model"],
        {"id", "revision"},
        f"evidence manifest identity is not canonical in {path}",
    )
    installed_package = _canonical_mapping(
        manifest["installed_package"],
        {"name", "version", "content_sha256"},
        f"evidence installed package is not canonical in {path}",
    )
    if not isinstance(source["dirty"], bool):
        raise ValueError(f"evidence source dirty flag must be boolean in {path}")
    return source, model, installed_package


def _validate_evidence_identity(
    manifest: Mapping[str, Any],
    path: Path,
) -> None:
    source, model, installed_package = _evidence_identity(manifest, path)
    if manifest["claim_vocabulary_sha256"] != CLAIM_VOCABULARY_SHA256:
        raise ValueError(f"evidence claim vocabulary hash mismatch in {path}")
    for name, value in (
        ("source commit", source["commit"]),
        ("source state", source["state_sha256"]),
        ("dependency lock", manifest["dependency_lock_sha256"]),
        ("claim vocabulary", manifest["claim_vocabulary_sha256"]),
        ("installed package", installed_package["content_sha256"]),
        ("model ID", model["id"]),
        ("model revision", model["revision"]),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"evidence manifest {name} must be a non-empty string in {path}")


def _validate_evidence_entries(
    manifest: Mapping[str, Any],
    path: Path,
) -> None:
    for section in ("source_assets", "parents", "inputs", "artifacts"):
        entries = manifest[section]
        if not isinstance(entries, Mapping):
            raise ValueError(f"evidence manifest {section} must be an object in {path}")
        for name, value in entries.items():
            entry = _canonical_mapping(
                value,
                {"locator", "sha256"},
                f"evidence manifest {section} entry {name!r} is not canonical in {path}",
            )
            if not is_lowercase_sha256(entry.get("sha256")):
                raise ValueError(f"evidence manifest {section} entry {name!r} has an invalid hash in {path}")
            if section == "source_assets":
                continue
            artifact_path = (path.parent / str(entry["locator"])).resolve()
            if not artifact_path.exists():
                raise ValueError(f"evidence manifest references missing {section} artifact {name!r} in {path}")
            if entry["sha256"] != sha256_path(artifact_path):
                raise ValueError(f"evidence manifest hash mismatch for {section} artifact {name!r} in {path}")


def load_evidence_manifest(path: Path) -> dict[str, Any]:
    manifest = dict(load_json_object(path))
    if set(manifest) != _EVIDENCE_MANIFEST_FIELDS:
        raise ValueError(f"evidence manifest is not canonical in {path}")
    if manifest["artifact_kind"] not in {
        "deployment",
        "performance_ablation",
        "project",
        "replication",
        "search",
        "state_budget",
        "study",
        *PROSPECTIVE_ARTIFACT_KINDS.values(),
    }:
        raise ValueError(f"invalid evidence artifact kind in {path}")
    expected_modes = {"cross_directional"} if manifest["artifact_kind"] == "state_budget" else {"input_only", "output_only"}
    if manifest["mode"] not in expected_modes:
        raise ValueError(f"invalid evidence mode in {path}")
    _validate_evidence_identity(manifest, path)
    _validate_evidence_entries(manifest, path)
    return manifest


def student_t_95_interval(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 3:
        raise ValueError("Student-t replication intervals require exactly three values")
    mean = sum(values) / 3
    squared = sum((value - mean) ** 2 for value in values)
    standard_error = math.sqrt(squared / 2) / math.sqrt(3)
    half_width = _STUDENT_T_95_DF_2 * standard_error
    return mean - half_width, mean + half_width


def _claim_errors(
    claims: object,
    *,
    artifact_kind: str,
    mode: Literal["input_only", "output_only"] | None = None,
) -> list[str]:
    if not isinstance(claims, Sequence) or isinstance(claims, str | bytes):
        return [f"{artifact_kind} does not expose canonical claim records"]
    errors: list[str] = []
    identifiers: list[str] = []
    for value in claims:
        if not isinstance(value, Mapping) or set(value) != CLAIM_RECORD_FIELDS:
            errors.append(f"{artifact_kind} contains a non-canonical claim record")
            continue
        record = cast(Mapping[str, object], value)
        claim_id = str(record["claim_id"])
        verdict = record["verdict"]
        record_mode = record["mode"]
        reason = record["reason"]
        identifiers.append(claim_id)
        if (
            verdict not in CLAIM_VERDICTS
            or record_mode
            not in {
                "input_only",
                "output_only",
            }
            or not isinstance(reason, str)
            or not reason
        ):
            errors.append(f"{artifact_kind} claim {claim_id} has invalid mode or verdict")
            continue
        try:
            canonical = claim_record(
                cast(Literal["input_only", "output_only"], record_mode),
                claim_id,
                cast(
                    Literal[
                        "supported",
                        "unsupported",
                        "incomplete",
                        "inapplicable",
                    ],
                    verdict,
                ),
                reason=reason,
            ).to_dict()
        except ValueError as error:
            errors.append(str(error))
            continue
        if dict(record) != canonical:
            errors.append(f"{artifact_kind} claim {claim_id} differs from the vocabulary")
    if len(identifiers) != len(set(identifiers)):
        errors.append(f"{artifact_kind} contains duplicate claim records")
    if mode is not None:
        errors.extend(
            _exact_claim_set_errors(
                identifiers,
                mode=mode,
                artifact_kind=artifact_kind,
            )
        )
    return errors


def _exact_claim_set_errors(
    identifiers: Sequence[str],
    *,
    mode: Literal["input_only", "output_only"],
    artifact_kind: str,
) -> list[str]:
    expected = {claim.claim_id for claim in directional_claims(mode)}
    actual = set(identifiers)
    errors = []
    if missing := sorted(expected - actual):
        errors.append(f"{artifact_kind} does not separate required claims: " + ", ".join(missing))
    if unexpected := sorted(actual - expected):
        errors.append(f"{artifact_kind} contains unexpected claims: " + ", ".join(unexpected))
    return errors


def _replication_runs(
    replication: Mapping[str, Any],
) -> tuple[list[int], list[str]]:
    errors: list[str] = []
    runs = tuple(
        cast(Mapping[str, Any], value)
        for section in ("runs", "failed_runs")
        for value in cast(Sequence[Any], replication.get(section, ()))
        if isinstance(value, Mapping)
    )
    seeds = [int(run["seed"]) for run in runs]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        errors.append("replication must expose exactly three distinct seeds")
    if set(seeds) != {17, 23, 41}:
        errors.append("replication must expose registered seeds 17, 23, and 41")
    return seeds, errors


def _seed_evidence_errors(
    replication: Mapping[str, Any],
    expected_seeds: set[int],
) -> list[str]:
    errors: list[str] = []
    envelopes = replication.get("seed_evidence")
    envelope_seeds = (
        {
            int(row["seed"])
            for row in envelopes
            if isinstance(row, Mapping)
            and {
                "seed",
                "operational_status",
                "scientific_verdict",
                "metrics",
                "claims",
            }
            <= set(row)
        }
        if isinstance(envelopes, Sequence)
        else set()
    )
    if envelope_seeds != expected_seeds:
        errors.append("replication does not expose normalized per-seed evidence envelopes")
    if isinstance(envelopes, Sequence):
        for row in envelopes:
            if isinstance(row, Mapping):
                expected_fields = {
                    "seed",
                    "operational_status",
                    "scientific_verdict",
                    "metrics",
                    "thresholds",
                    "claims",
                }
                if row.get("operational_status") == "failed":
                    expected_fields.add("failure")
                if set(row) != expected_fields:
                    errors.append(f"seed {row.get('seed')} evidence envelope is not canonical")
                errors.extend(
                    _claim_errors(
                        row.get("claims"),
                        artifact_kind=f"seed {row.get('seed')}",
                        mode=cast(
                            Literal["input_only", "output_only"],
                            replication.get("mode"),
                        ),
                    )
                )
                errors.extend(
                    _seed_headline_errors(
                        row,
                        mode=replication.get("mode"),
                    ),
                )
    return errors


def _seed_headline_errors(
    row: Mapping[str, object],
    *,
    mode: object,
) -> list[str]:
    if mode != "input_only" or row.get("operational_status") != "completed":
        return []
    try:
        expected = derive_input_headline_verdict(
            _claim_verdict_map(row.get("claims")),
        )
    except ValueError:
        return [
            f"seed {row.get('seed')} headline cannot be derived from canonical claims",
        ]
    if row.get("scientific_verdict") == expected:
        return []
    return [
        f"seed {row.get('seed')} scientific verdict differs from canonical headline claims",
    ]


def _claim_verdict_map(records: object) -> dict[str, object]:
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        return {}
    verdicts: dict[str, object] = {}
    for value in records:
        if not isinstance(value, Mapping):
            continue
        record = cast(Mapping[str, object], value)
        if "claim_id" in record and "verdict" in record:
            verdicts[str(record["claim_id"])] = record["verdict"]
    return verdicts


def _derived_replication_claim_errors(
    replication: Mapping[str, Any],
) -> list[str]:
    mode = replication.get("mode")
    envelopes = replication.get("seed_evidence")
    if mode not in {"input_only", "output_only"} or not isinstance(
        envelopes,
        Sequence,
    ):
        return ["replication directional evidence cannot be recomputed"]
    completed = [cast(Mapping[str, Any], row) for row in envelopes if isinstance(row, Mapping) and row.get("operational_status") == "completed"]
    metrics = [cast(Mapping[str, object], row["metrics"]) for row in completed if isinstance(row.get("metrics"), Mapping)]
    thresholds = [cast(Mapping[str, object], row["thresholds"]) for row in completed if isinstance(row.get("thresholds"), Mapping)]
    complete = replication.get("replication_complete") is True
    if mode == "input_only":
        derived = derive_input_claim_verdicts(
            metrics,
            thresholds,
            complete=complete,
        )
    else:
        derived = derive_output_claim_verdicts(
            metrics,
            thresholds,
            complete=complete,
            structurally_unrepresentable=(replication.get("structurally_unrepresentable") is True),
        )
    errors: list[str] = []
    if _claim_verdict_map(replication.get("claims")) != derived:
        errors.append("replication claims differ from raw directional metrics")
    for row in completed:
        row_metrics = row.get("metrics")
        row_thresholds = row.get("thresholds")
        if not isinstance(row_metrics, Mapping) or not isinstance(
            row_thresholds,
            Mapping,
        ):
            errors.append(f"seed {row.get('seed')} lacks raw directional evidence")
            continue
        if mode == "input_only":
            expected = derive_input_claim_verdicts(
                (cast(Mapping[str, object], row_metrics),),
                (cast(Mapping[str, object], row_thresholds),),
                complete=True,
            )
        else:
            expected = derive_output_claim_verdicts(
                (cast(Mapping[str, object], row_metrics),),
                (cast(Mapping[str, object], row_thresholds),),
                complete=True,
                structurally_unrepresentable=(row_metrics.get("structurally_unrepresentable") is True),
            )
        if _claim_verdict_map(row.get("claims")) != expected:
            errors.append(f"seed {row.get('seed')} claims differ from raw directional metrics")
    return errors


def _verification_inventory_passes(
    verification: Mapping[str, Any],
) -> bool:
    checks = verification.get("checks")
    return (
        verification.get("provided") is True
        and verification.get("all_passed") is True
        and isinstance(checks, Mapping)
        and set(checks) == FINAL_VERIFICATION_CHECKS
        and all(isinstance(checks[name], Mapping) and checks[name].get("passed") is True for name in FINAL_VERIFICATION_CHECKS)
    )


def _verification_coverage_errors(
    replication: Mapping[str, Any],
    expected_seeds: set[int],
) -> list[str]:
    verification = replication.get("verification")
    if not isinstance(verification, Mapping):
        return ["replication does not propagate verification coverage"]
    coverage = verification.get("runs")
    coverage_rows = [cast(Mapping[str, Any], row) for row in coverage if isinstance(row, Mapping)] if isinstance(coverage, Sequence) else []
    coverage_seeds = {int(row["seed"]) for row in coverage_rows if "seed" in row}
    errors = [] if coverage_seeds == expected_seeds else ["replication verification coverage does not match raw seeds"]
    if verification.get("all_provided") is not True or verification.get("all_passed") is not True:
        errors.append("replication verification summary is not complete and passing")
    errors.extend(f"seed {row.get('seed')} verification inventory is not exact and passing" for row in coverage_rows if not _verification_inventory_passes(row))
    return errors


_CONFIDENCE_FIELDS: Final = (
    "confidence_method",
    "degrees_of_freedom",
    "confidence_95_low",
    "confidence_95_high",
)


def _confidence_errors(
    name: str,
    metric: Mapping[str, Any],
    replication: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    if metric.get("confidence_eligible") is not True:
        if any(metric.get(field) is not None for field in _CONFIDENCE_FIELDS):
            return [f"metric {name} has an ineligible confidence interval"]
        return []
    errors: list[str] = []
    model = replication.get("model")
    if (
        not isinstance(model, Mapping)
        or (model.get("id"), model.get("revision")) not in {QWEN_MODEL, GEMMA_MODEL}
        or replication.get("replication_complete") is not True
    ):
        errors.append(
            f"metric {name} has confidence outside completed final primary-model evidence",
        )
    values = [float(row["value"]) for row in raw_rows]
    raw_seeds = {int(row["seed"]) for row in raw_rows}
    if raw_seeds != {17, 23, 41} or len(values) != 3:
        errors.append(f"metric {name} is confidence-eligible without three registered seeds")
        return errors
    low, high = student_t_95_interval(values)
    if metric.get("confidence_method") != "student_t":
        errors.append(f"metric {name} does not use Student-t confidence")
    if metric.get("degrees_of_freedom") != 2:
        errors.append(f"metric {name} has invalid confidence degrees of freedom")
    if not math.isclose(float(metric["confidence_95_low"]), low):
        errors.append(f"metric {name} has an invalid lower confidence bound")
    if not math.isclose(float(metric["confidence_95_high"]), high):
        errors.append(f"metric {name} has an invalid upper confidence bound")
    return errors


def _metric_errors(
    replication: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    metrics = replication.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("replication does not expose canonical metrics")
        metrics = {}
    for name, value in metrics.items():
        if not isinstance(value, Mapping):
            continue
        raw = value.get("raw_values")
        if not isinstance(raw, Sequence):
            errors.append(f"metric {name} does not preserve raw per-seed values")
            continue
        raw_rows = [row for row in raw if isinstance(row, Mapping) and row.get("value") is not None]
        failed_seeds = {
            int(run["seed"])
            for run in cast(
                Sequence[Mapping[str, Any]],
                replication.get("failed_runs", ()),
            )
        }
        if any(int(row["seed"]) in failed_seeds for row in raw_rows):
            errors.append(f"metric {name} includes a failed run")
        errors.extend(
            _confidence_errors(
                str(name),
                value,
                replication,
                cast(Sequence[Mapping[str, Any]], raw_rows),
            )
        )
        if value.get("count") == 0 and any(value.get(field) is not None for field in ("mean", "minimum", "maximum")):
            errors.append(f"empty metric {name} must use null aggregates")
    return errors


def _pinned_source_errors(
    artifact: Mapping[str, Any],
    *,
    allowed_models: set[tuple[str, str]],
    label: str,
) -> list[str]:
    errors: list[str] = []
    model = artifact.get("model")
    dataset = artifact.get("dataset")
    model_role = (str(model.get("id")), str(model.get("revision"))) if isinstance(model, Mapping) else ("", "")
    dataset_role = (str(dataset.get("id")), str(dataset.get("revision"))) if isinstance(dataset, Mapping) else ("", "")
    if model_role not in allowed_models:
        errors.append(f"{label} does not use a pinned model role")
    if dataset_role != WIKITEXT_DATASET:
        errors.append(f"{label} does not use the pinned WikiText dataset")
    assets = artifact.get("source_assets")
    if not isinstance(assets, Mapping) or not _REQUIRED_SOURCE_ASSETS.issubset(assets):
        errors.append(f"{label} lacks the pinned source-asset inventory")
        return errors
    prefix = f"hf://{model_role[0]}@{model_role[1]}"
    if any(not isinstance(assets[name], Mapping) or not str(assets[name].get("locator", "")).startswith(prefix) for name in _REQUIRED_SOURCE_ASSETS):
        errors.append(f"{label} source assets do not match its pinned model")
    return errors


def _verify_replication(replication: Mapping[str, Any]) -> list[str]:
    seeds, errors = _replication_runs(replication)
    expected_seeds = set(seeds)
    errors.extend(_seed_evidence_errors(replication, expected_seeds))
    errors.extend(_verification_coverage_errors(replication, expected_seeds))
    errors.extend(_metric_errors(replication))
    errors.extend(
        _claim_errors(
            replication.get("claims"),
            artifact_kind="replication",
            mode=cast(
                Literal["input_only", "output_only"],
                replication.get("mode"),
            ),
        )
    )
    errors.extend(_derived_replication_claim_errors(replication))
    errors.extend(_replication_headline_errors(replication))
    if replication.get("evidence_scope") != "replication":
        errors.append("replication evidence scope is not canonical")
    if replication.get("profile") != CAMPAIGN_PROFILE_NAME:
        errors.append("replication evidence does not use the final campaign profile")
    errors.extend(
        _pinned_source_errors(
            replication,
            allowed_models={QWEN_MODEL, GEMMA_MODEL},
            label="replication",
        )
    )
    return errors


def _replication_headline_errors(
    replication: Mapping[str, Any],
) -> list[str]:
    mode = replication.get("mode")
    if mode == "input_only":
        try:
            expected = derive_input_headline_verdict(
                _claim_verdict_map(replication.get("claims")),
            )
        except ValueError:
            return ["replication headline cannot be derived from canonical claims"]
    elif mode == "output_only":
        claims = replication.get("claims")
        if not isinstance(claims, Sequence) or isinstance(claims, str | bytes):
            return ["replication headline cannot be derived from canonical claims"]
        categories = claim_category_verdicts(cast(Mapping[str, object], claim) for claim in claims if isinstance(claim, Mapping))
        expected = categories["quality"]
        if replication.get("structurally_unrepresentable") is True and replication.get("replication_complete") is True:
            expected = "unsupported"
    else:
        return ["replication headline has an invalid tokenizer mode"]
    return [] if replication.get("scientific_verdict") == expected else ["replication scientific verdict differs from canonical headline claims"]


def _project_category_errors(project: Mapping[str, Any]) -> list[str]:
    errors = []
    category_verdicts = project.get("category_verdicts")
    records = project.get("claims")
    if isinstance(records, Sequence):
        canonical = claim_category_verdicts(cast(Mapping[str, object], record) for record in records if isinstance(record, Mapping))
        for category, expected_category in canonical.items():
            if not isinstance(category_verdicts, Mapping) or category_verdicts.get(category) != expected_category:
                errors.append(f"project {category} verdict differs from canonical claims")
    return errors


def _verify_project(
    project: Mapping[str, Any],
    root: Path,
) -> list[str]:
    errors: list[str] = []
    models = project.get("models")
    model_entries = (
        [cast(Mapping[str, Any], value) for value in models if isinstance(value, Mapping)]
        if isinstance(models, Sequence) and not isinstance(models, str | bytes)
        else []
    )
    identities = {
        (
            cast(Mapping[str, Any], entry.get("model", {})).get("id"),
            cast(Mapping[str, Any], entry.get("model", {})).get("revision"),
        )
        for entry in model_entries
    }
    if len(model_entries) != 2 or identities != {QWEN_MODEL, GEMMA_MODEL}:
        errors.append("project must expose equal pinned Qwen and Gemma primary replications")
    model_verdicts, verdict_errors = _project_model_verdicts(model_entries)
    errors.extend(verdict_errors)
    cross_model = combine_claim_verdicts(model_verdicts)
    if project.get("cross_model_verdict") != cross_model:
        errors.append("project cross-model verdict differs from both primary replications")
    try:
        scientific_verdict = (
            derive_input_headline_verdict(
                _claim_verdict_map(project.get("claims")),
            )
            if project.get("mode") == "input_only"
            else cross_model
        )
    except ValueError:
        errors.append(
            "project scientific verdict cannot be derived from canonical headline claims",
        )
        scientific_verdict = None
    if project.get("scientific_verdict") != scientific_verdict:
        errors.append("project scientific verdict differs from canonical headline claims")
    errors.extend(_project_category_errors(project))
    errors.extend(
        _claim_errors(
            project.get("claims"),
            artifact_kind="project",
            mode=cast(
                Literal["input_only", "output_only"],
                project.get("mode"),
            ),
        )
    )
    errors.extend(_project_claim_lineage_errors(project, model_entries))
    errors.extend(_project_parent_errors(root, project, model_entries))
    errors.extend(_project_trace_errors(root, project, model_entries))
    return errors


def _project_model_verdicts(
    model_entries: Sequence[Mapping[str, Any]],
) -> tuple[list[ClaimVerdict], list[str]]:
    verdicts: list[ClaimVerdict] = []
    errors: list[str] = []
    for entry in model_entries:
        replication = entry.get("replication")
        if not isinstance(replication, Mapping):
            errors.append("project model entry has no normalized replication")
            continue
        categories = replication.get("category_verdicts")
        expected = categories.get("quality") if isinstance(categories, Mapping) else None
        if entry.get("quality_verdict") != expected or expected not in CLAIM_VERDICTS:
            errors.append("project model quality verdict differs from its replication")
            continue
        if replication.get("mode") == "input_only":
            try:
                scientific = derive_input_headline_verdict(
                    _claim_verdict_map(replication.get("claims")),
                )
            except ValueError:
                errors.append(
                    "project model scientific verdict cannot be derived from its replication",
                )
                continue
        else:
            scientific = cast(ClaimVerdict, expected)
        if entry.get("scientific_verdict") != scientific:
            errors.append(
                "project model scientific verdict differs from its replication",
            )
            continue
        verdicts.append(scientific)
    return verdicts, errors


def _project_trace_errors(
    root: Path,
    project: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    try:
        expected_claim_traces = project_claim_trace_records(
            cast(Sequence[Mapping[str, object]], project.get("claims", ())),
            cast(Sequence[Mapping[str, object]], models),
        )
    except KeyError, StopIteration, TypeError, ValueError:
        errors.append("project claim traces cannot be derived from parent model evidence")
    else:
        if project.get("claim_traces") != expected_claim_traces:
            errors.append("project claim traces differ from canonical parent model evidence")
    manifest = load_evidence_manifest(root / EVIDENCE_MANIFEST_FILENAME)
    inputs = cast(Mapping[str, Any], manifest["inputs"])
    names = {
        "software_verification",
        "software_input_synthetic",
        "software_output_synthetic",
    }
    supplied = names.intersection(inputs)
    software_validation = None
    if supplied and supplied != names:
        errors.append("project software validation inputs are incomplete")
    elif supplied:
        try:
            from continuous_tokenizer.artifacts.software_validation import (
                load_software_validation_inputs,
            )

            paths = {name: (root / str(cast(Mapping[str, Any], inputs[name])["locator"])).resolve() for name in names}
            software_validation = load_software_validation_inputs(
                paths["software_verification"],
                paths["software_input_synthetic"],
                paths["software_output_synthetic"],
            )
            if software_validation.verification.source != SourceBinding(
                str(project["source_state_sha256"]),
                str(project["dependency_lock_sha256"]),
            ):
                errors.append("project software validation source identity differs from project evidence")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"project software validation is invalid: {error}")
    if project.get("statement_traces") != statement_trace_records(
        software_validation,
    ):
        errors.append("project statement traces differ from sealed validation inputs")
    return errors


def _project_claim_lineage_errors(
    project: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
) -> list[str]:
    if len(models) != 2:
        return ["project has no normalized primary-model replications"]
    parent_claims = [_claim_verdict_map(cast(Mapping[str, Any], model["replication"]).get("claims")) for model in models]
    expected = {
        claim_id: combine_claim_verdicts(cast(ClaimVerdict, claims[claim_id]) for claims in parent_claims)
        for claim_id in set.intersection(*(set(claims) for claims in parent_claims))
    }
    cross_model_id = "input.cross_model_confirmation" if project.get("mode") == "input_only" else "output.cross_model_confirmation"
    expected[cross_model_id] = combine_claim_verdicts(cast(ClaimVerdict, model["scientific_verdict"]) for model in models)
    actual = _claim_verdict_map(project.get("claims"))
    if project.get("mode") == "input_only":
        expected["input.fixed_subset_alignment_feasibility"] = derive_project_alignment_verdict(
            project.get("alignment_feasibility"),
        )
    expected.update(
        derive_project_deployment_verdicts(
            cast(Literal["input_only", "output_only"], project.get("mode")),
            project.get("deployment"),
        ),
    )
    return [] if actual == expected else ["project claims differ from parent directional evidence"]


def _project_parent_errors(
    root: Path,
    project: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
) -> list[str]:
    manifest = load_evidence_manifest(root / EVIDENCE_MANIFEST_FILENAME)
    parents = cast(Mapping[str, Any], manifest["parents"])
    errors: list[str] = []
    for index, model in enumerate(models):
        parent_entry = parents.get(f"replication_{index}")
        if not isinstance(parent_entry, Mapping):
            errors.append("project does not declare both primary replication parents")
            continue
        parent_manifest = (root / str(parent_entry["locator"])).resolve()
        parent_replication = load_json_object(parent_manifest.parent / "replication.json")
        if model.get("replication") != parent_replication:
            errors.append("project embedded replication differs from its parent")
    for field, prefix, filename in (
        ("alignment_feasibility", "alignment_study", "result.json"),
        ("deployment", "deployment", "deployment.json"),
    ):
        project_values = project.get(field)
        if not isinstance(project_values, Sequence) or isinstance(
            project_values,
            str | bytes,
        ):
            continue
        for index, value in enumerate(project_values):
            if not isinstance(value, Mapping):
                errors.append(f"project {field} entry is not normalized")
                continue
            parent_entry = parents.get(f"{prefix}_{index}")
            if not isinstance(parent_entry, Mapping):
                errors.append(f"project does not declare its {field} parent")
                continue
            parent_manifest = (root / str(parent_entry["locator"])).resolve()
            parent_result = load_json_object(parent_manifest.parent / filename)
            if value.get("result") != parent_result:
                errors.append(f"project embedded {field} differs from its parent")
    return errors


def _identity_errors(
    parent: Mapping[str, Any],
    nested_directory: Path,
) -> list[str]:
    source = cast(Mapping[str, Any], parent.get("source", {}))
    expected = (
        source.get("commit"),
        source.get("state_sha256"),
        parent.get("dependency_lock_sha256"),
        cast(Mapping[str, Any], parent.get("installed_package", {})).get("content_sha256"),
        parent.get("claim_vocabulary_sha256"),
    )
    evidence_path = nested_directory / EVIDENCE_MANIFEST_FILENAME
    run_path = nested_directory / "manifest-final.json"
    if evidence_path.is_file():
        nested = load_json_object(evidence_path)
        nested_source = cast(Mapping[str, Any], nested.get("source", {}))
        actual = (
            nested_source.get("commit"),
            nested_source.get("state_sha256"),
            nested.get("dependency_lock_sha256"),
            cast(
                Mapping[str, Any],
                nested.get("installed_package", {}),
            ).get("content_sha256"),
            nested.get("claim_vocabulary_sha256"),
        )
    elif run_path.is_file():
        run = load_verified_run_manifest(run_path)
        actual = (
            run.source_commit,
            run.source_state_sha256,
            run.dependency_lock_sha256,
            run.installed_package.get("content_sha256"),
            run.claim_vocabulary_sha256,
        )
    else:
        return []
    return [] if actual == expected else [f"evidence identity mismatch in {nested_directory}"]


def _nested_manifest_directories(
    root: Path,
    artifact_directories: set[Path],
) -> set[Path]:
    nested_directories: set[Path] = set()
    walked_roots: list[Path] = []
    for artifact_directory in sorted(
        artifact_directories,
        key=lambda path: len(path.parts),
    ):
        if any(parent == artifact_directory or parent in artifact_directory.parents for parent in walked_roots):
            continue
        walked_roots.append(artifact_directory)
        for path in directory_files(artifact_directory):
            if path.name in {EVIDENCE_MANIFEST_FILENAME, "manifest-final.json"} and path.parent != root:
                nested_directories.add(path.parent)
    return nested_directories


def _nested_evidence_directories(
    root: Path,
    manifest: Mapping[str, Any],
) -> set[Path]:
    nested_directories: set[Path] = set()
    artifact_directories: set[Path] = set()
    for section in ("parents", "inputs", "artifacts"):
        entries = manifest.get(section)
        if not isinstance(entries, Mapping):
            continue
        for value in entries.values():
            if not isinstance(value, Mapping):
                continue
            artifact_path = (root / str(value.get("locator", ""))).resolve()
            parent = artifact_path if artifact_path.is_dir() else artifact_path.parent
            if parent != root and ((parent / EVIDENCE_MANIFEST_FILENAME).is_file() or (parent / "manifest-final.json").is_file()):
                nested_directories.add(parent)
            if artifact_path.is_dir():
                artifact_directories.add(artifact_path)
    return nested_directories | _nested_manifest_directories(
        root,
        artifact_directories,
    )


def _verify_evidence_tree(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    checked: list[str] = []
    for nested_directory in sorted(_nested_evidence_directories(root, manifest)):
        try:
            errors.extend(_identity_errors(manifest, nested_directory))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
        nested = verify_artifact(nested_directory)
        checked.extend(cast(Sequence[str], nested["checked"]))
        errors.extend(cast(Sequence[str], nested["errors"]))
    return errors, checked


def _verify_run_manifest(path: Path) -> tuple[list[str], list[str]]:
    try:
        manifest = load_verified_run_manifest(path)
    except (KeyError, TypeError, ValueError) as error:
        return [str(error)], []
    errors = _prospective_subset_run_errors(path, manifest)
    if manifest.status == "passed":
        result_relative = manifest.artifacts.get("result")
        if result_relative is None:
            errors.append("completed run manifest does not seal result.json")
        else:
            result = load_json_object(path.parent / result_relative)
            if result.get("mode") != manifest.mode:
                errors.append("completed run result mode differs from its manifest")
            if result.get("operational_status") != "completed":
                errors.append("completed run result has an invalid operational status")
            llm = result.get("llm")
            if isinstance(llm, Mapping):
                errors.extend(
                    prefill_performance_errors(llm.get("performance")),
                )
                errors.extend(
                    performance_claim_context_errors(
                        llm.get("performance_claim_context"),
                    ),
                )
            errors.extend(
                tokenizer_performance_errors(result.get("tokenizer")),
            )
            errors.extend(output_performance_errors(result.get("output")))
            if result.get("evidence_scope") == "final":
                errors.extend(_final_run_errors(manifest, result))
    return errors, [str(path)]


def _prospective_subset_run_errors(
    manifest_path: Path,
    manifest: Any,
) -> list[str]:
    relative = manifest.artifacts.get("prospective_vocabulary_subset")
    input_identity = manifest.inputs.get("prospective_vocabulary_subset")
    result_relative = manifest.artifacts.get("result")
    result = load_json_object(manifest_path.parent / result_relative) if result_relative is not None else {}
    result_descriptor = result.get("prospective_vocabulary_subset")
    if relative is None:
        if input_identity is not None or result_descriptor is not None:
            return ["prospective vocabulary subset identity is incomplete"]
        return []
    if manifest.mode != "input_only" or relative != PROSPECTIVE_INPUT_SUBSET_FILENAME or not isinstance(input_identity, Mapping):
        return ["prospective vocabulary subset manifest identity is invalid"]
    artifact_path = manifest_path.parent / relative
    artifact = load_json_object(artifact_path)
    errors = prospective_vocabulary_subset_errors(artifact)
    if input_identity.get("sha256") != sha256_path(artifact_path) or not isinstance(input_identity.get("locator"), str):
        errors.append("prospective vocabulary subset input identity mismatch")
    expected_descriptor = {key: value for key, value in artifact.items() if key != "rows"} | {"artifact": PROSPECTIVE_INPUT_SUBSET_FILENAME}
    if result_relative is not None and result_descriptor != expected_descriptor:
        errors.append("prospective vocabulary subset result descriptor mismatch")
    return errors


def _final_run_errors(
    manifest: Any,
    result: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if (manifest.model_id, manifest.model_revision) not in {
        QWEN_MODEL,
        GEMMA_MODEL,
    }:
        errors.append("final run does not use a pinned model role")
    if (manifest.dataset_id, manifest.dataset_revision) != WIKITEXT_DATASET:
        errors.append("final run does not use the pinned WikiText dataset")
    prefix = f"hf://{manifest.model_id}@{manifest.model_revision}"
    if not _REQUIRED_SOURCE_ASSETS.issubset(manifest.source_assets) or any(
        not str(manifest.source_assets[name].get("locator", "")).startswith(prefix) for name in _REQUIRED_SOURCE_ASSETS
    ):
        errors.append("final run lacks matching pinned source assets")
    if not _verification_inventory_passes(manifest.verification):
        errors.append("final run lacks the exact passing verification inventory")
    experiment = result.get("experiment")
    if (
        not isinstance(experiment, Mapping)
        or not isinstance(experiment.get("training"), Mapping)
        or experiment["training"].get("profile") != CAMPAIGN_PROFILE_NAME
    ):
        errors.append("final run does not use the final campaign profile")
    return errors


_PRIMARY_EVIDENCE_FILENAMES: Final = {
    "deployment": "deployment.json",
    "performance_ablation": "performance-ablation.json",
    "project": "project.json",
    "replication": "replication.json",
    "search": "search.json",
    "state_budget": "joint-state-budget.json",
    "study": "result.json",
    **dict.fromkeys(
        PROSPECTIVE_ARTIFACT_KINDS.values(),
        "prospective.json",
    ),
}
_EVIDENCE_SCOPES: Final = {
    "performance_ablation": "operational_secondary",
    "project": "project",
    "replication": "replication",
    "search": "search",
    "state_budget": "cross_directional_prerequisite",
    "study": "selection",
}


def _evidence_primary_errors(  # noqa: C901
    root: Path,
    manifest: Mapping[str, Any],
) -> list[str]:
    kind = str(manifest["artifact_kind"])
    primary_path = root / _PRIMARY_EVIDENCE_FILENAMES[kind]
    if not primary_path.is_file():
        return [f"sealed {kind} evidence has no {primary_path.name}"]
    sealed_paths = {
        (root / str(entry["locator"])).resolve()
        for value in cast(Mapping[str, Any], manifest["artifacts"]).values()
        if isinstance(value, Mapping)
        for entry in (cast(Mapping[str, Any], value),)
    }
    if primary_path.resolve() not in sealed_paths:
        return [f"evidence manifest does not seal {primary_path.name}"]
    primary = load_json_object(primary_path)
    errors = []
    if kind != "state_budget" and primary.get("mode") != manifest["mode"]:
        errors.append(f"{kind} mode differs from its evidence manifest")
    status = primary.get("operational_status", primary.get("status"))
    if status != manifest["status"]:
        errors.append(f"{kind} status differs from its evidence manifest")
    expected_scope = _EVIDENCE_SCOPES.get(kind)
    if expected_scope is not None and primary.get("evidence_scope") != expected_scope:
        errors.append(f"{kind} has an invalid evidence scope")
    if kind == "study" and primary.get("artifact_kind") == "input_alignment_feasibility_study":
        if primary.get("prospective") is not True or primary.get("final_evidence") is not False or primary.get("full_model_evaluation_performed") is not False:
            errors.append(
                "alignment feasibility study makes an invalid final-model claim",
            )
        try:
            alignment_feasibility_verdict(primary)
        except (TypeError, ValueError) as error:
            errors.append(str(error))
    if kind in PROSPECTIVE_ARTIFACT_KINDS.values():
        errors.extend(
            _prospective_primary_errors(
                root,
                manifest,
                primary,
            ),
        )
    if kind == "performance_ablation":
        errors.extend(performance_ablation_errors(primary))
        optimized = primary.get("optimized")
        source = manifest.get("source")
        model = manifest.get("model")
        if (
            not isinstance(optimized, Mapping)
            or not isinstance(source, Mapping)
            or not isinstance(model, Mapping)
            or optimized.get("source_state_sha256") != source.get("state_sha256")
            or optimized.get("dependency_lock_sha256") != manifest.get("dependency_lock_sha256")
            or optimized.get("model_id") != model.get("id")
            or optimized.get("model_revision") != model.get("revision")
        ):
            errors.append(
                "performance ablation optimized identity differs from its evidence manifest",
            )
    return errors


def _prospective_primary_errors(
    root: Path,
    manifest: Mapping[str, Any],
    primary: Mapping[str, Any],
) -> list[str]:
    from continuous_tokenizer.contracts.prospective import ProspectiveSpec

    errors = prospective_result_errors(primary)
    inputs = cast(Mapping[str, Any], manifest["inputs"])
    spec_entry = inputs.get("spec")
    if not isinstance(spec_entry, Mapping):
        return [*errors, "prospective evidence does not seal its wrapper"]
    spec_path = (root / str(spec_entry["locator"])).resolve()
    spec = ProspectiveSpec.load(spec_path)
    if primary.get("spec_fingerprint") != spec.fingerprint() or primary.get("artifact_kind") != spec.artifact_kind or primary.get("mode") != spec.mode:
        errors.append("prospective result differs from its sealed wrapper")
    return errors


def _verify_deployment(deployment: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    applicability = deployment.get("applicability")
    if not isinstance(applicability, Mapping):
        return ["deployment applicability is missing"]
    applicable = applicability.get("applicable")
    if not isinstance(applicable, bool):
        errors.append("deployment applicability must be boolean")
    raw = deployment.get("raw_repetitions")
    if not isinstance(raw, list):
        errors.append("deployment raw repetitions are missing")
    elif applicable and len(raw) != 3:
        errors.append("applicable deployment evidence requires three repetitions")
    elif not applicable and raw:
        errors.append("inapplicable deployment evidence must not run workers")
    if not applicable:
        errors.extend(
            f"inapplicable deployment field {name} must be null"
            for name in (
                "physical_reference_tensor_absent",
                "output_equivalent",
                "hidden_equivalent",
                "deployment_compactness_claimable",
            )
            if deployment.get(name) is not None
        )
    return errors


def _verify_state_budget(
    budget: Mapping[str, Any],
    root: Path,
) -> list[str]:
    from continuous_tokenizer.artifacts.state_budget_verification import (
        verify_state_budget,
    )

    manifest = load_evidence_manifest(root / EVIDENCE_MANIFEST_FILENAME)
    return verify_state_budget(budget, root, manifest)


def verify_artifact(directory: Path) -> dict[str, Any]:
    root = directory.resolve()
    manifest_path = root / EVIDENCE_MANIFEST_FILENAME
    run_manifest_path = root / "manifest-final.json"
    if manifest_path.is_file():
        try:
            manifest = load_evidence_manifest(manifest_path)
        except (KeyError, TypeError, ValueError) as error:
            errors = [str(error)]
            checked = []
        else:
            errors, checked = _verify_evidence_tree(root, manifest)
            checked.append(str(manifest_path))
            try:
                errors.extend(_evidence_primary_errors(root, manifest))
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
    elif run_manifest_path.is_file():
        errors, checked = _verify_run_manifest(run_manifest_path)
    else:
        errors = [f"no immutable manifest found in {root}"]
        checked = []

    for name, validator in (
        ("deployment.json", _verify_deployment),
        ("replication.json", _verify_replication),
    ):
        path = root / name
        if path.is_file():
            artifact = load_json_object(path)
            errors.extend(validator(artifact))
            checked.append(str(path))
    project_path = root / "project.json"
    if project_path.is_file():
        project = load_json_object(project_path)
        errors.extend(_verify_project(project, root))
        checked.append(str(project_path))
    state_budget_path = root / "joint-state-budget.json"
    if state_budget_path.is_file():
        state_budget = load_json_object(state_budget_path)
        errors.extend(_verify_state_budget(state_budget, root))
        checked.append(str(state_budget_path))
    return {
        "artifact": str(root),
        "valid": not errors,
        "checked": sorted(set(checked)),
        "errors": errors,
    }
