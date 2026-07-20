from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from continuous_tokenizer.artifacts.evidence import verify_artifact
from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.artifacts.manifest import load_verified_run_manifest
from continuous_tokenizer.artifacts.store import load_json_object
from continuous_tokenizer.contracts.claim_derivation import FINAL_VERIFICATION_CHECKS
from continuous_tokenizer.contracts.statements import (
    SoftwareValidationInputs,
    SourceBinding,
    SyntheticCampaignEvidence,
    SyntheticMode,
    VerificationEvidence,
)

_SYNTHETIC_MODEL = ("continuous-tokenizer/synthetic-model", "synthetic")
_SYNTHETIC_DATASET = ("continuous-tokenizer/synthetic-bytes", "synthetic")
_VERIFICATION_FIELDS = {
    "kind",
    "source_commit",
    "source_dirty",
    "source_state_sha256",
    "dependency_lock_sha256",
    "all_passed",
    "checks",
}
_CHECK_FIELDS = {
    "command",
    "passed",
    "return_code",
    "seconds",
    "log",
    "log_sha256",
}


def _artifact_file(path: Path, filename: str) -> Path:
    candidate = path / filename if path.is_dir() else path
    if not candidate.is_file() or candidate.name != filename:
        raise ValueError(f"expected {filename}: {path}")
    return candidate.resolve()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _validate_verification_check(
    artifact_path: Path,
    name: str,
    value: object,
) -> None:
    check = _mapping(value, f"verification check {name}")
    if set(check) != _CHECK_FIELDS:
        raise ValueError(f"verification check {name!r} is not canonical")
    command = check["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
        or type(check["return_code"]) is not int
        or not isinstance(check["seconds"], int | float)
    ):
        raise ValueError(f"verification check {name!r} metadata is invalid")
    if check["passed"] is not True or check["return_code"] != 0:
        raise ValueError(f"verification check {name!r} did not pass")
    log = check["log"]
    if not isinstance(log, str) or not log:
        raise ValueError(f"verification check {name!r} has no canonical log")
    log_path = (artifact_path.parent / log).resolve()
    if not log_path.is_file():
        raise ValueError(f"verification check {name!r} log is missing")
    if sha256_path(log_path) != check["log_sha256"]:
        raise ValueError(f"verification check {name!r} log hash mismatch")


def _verification_evidence(path: Path) -> VerificationEvidence:
    artifact_path = _artifact_file(path, "verification.json")
    artifact = load_json_object(artifact_path)
    if set(artifact) != _VERIFICATION_FIELDS:
        raise ValueError("complete verification artifact is not canonical")
    if artifact["kind"] != "complete_verification":
        raise ValueError("validation requires a complete verification artifact")
    if not isinstance(artifact["source_commit"], str) or not artifact["source_commit"]:
        raise ValueError("verification source commit must be a non-empty string")
    if type(artifact["source_dirty"]) is not bool or type(artifact["all_passed"]) is not bool:
        raise ValueError("verification status fields must be booleans")
    checks = _mapping(artifact["checks"], "verification checks")
    if set(checks) != FINAL_VERIFICATION_CHECKS:
        raise ValueError("complete verification check inventory is not exact")
    for name, value in checks.items():
        _validate_verification_check(artifact_path, str(name), value)
    if artifact["all_passed"] is not True:
        raise ValueError("complete verification artifact contains failed checks")
    source = SourceBinding(
        str(artifact["source_state_sha256"]),
        str(artifact["dependency_lock_sha256"]),
    )
    return VerificationEvidence(
        source=source,
        all_passed=True,
        pointer=f"{artifact_path}#",
    )


def _input_software_checks(result: Mapping[str, Any]) -> bool:
    tokenizer = _mapping(result.get("tokenizer"), "input synthetic tokenizer evidence")
    acceptance = _mapping(tokenizer.get("acceptance"), "input synthetic acceptance")
    vocabulary = _mapping(result.get("vocabulary"), "input synthetic vocabulary")
    gates = _mapping(result.get("gates"), "input synthetic gates")
    return acceptance.get("overall") is True and gates.get("tokenizer") is True and vocabulary.get("atomic_bytes") == 256


def _output_software_checks(result: Mapping[str, Any], *, native_head_used: bool) -> bool:
    output = _mapping(result.get("output"), "output synthetic evidence")
    gates = _mapping(result.get("gates"), "output synthetic gates")
    return (
        native_head_used is False
        and output.get("native_head_invocations") == 0
        and gates.get("direct_feedback") is True
        and gates.get("invalid_events") is True
        and gates.get("valid_non_empty_termination") is True
    )


def _synthetic_campaign_evidence(
    path: Path,
    expected_mode: SyntheticMode,
) -> SyntheticCampaignEvidence:
    manifest_path = _artifact_file(path, "manifest-final.json")
    directory = manifest_path.parent
    verification = verify_artifact(directory)
    if verification["valid"] is not True:
        raise ValueError(f"{expected_mode} synthetic campaign failed semantic verification: " + "; ".join(str(error) for error in verification["errors"]))
    manifest = load_verified_run_manifest(manifest_path)
    if (
        manifest.mode != expected_mode
        or manifest.status != "passed"
        or (manifest.model_id, manifest.model_revision) != _SYNTHETIC_MODEL
        or (manifest.dataset_id, manifest.dataset_revision) != _SYNTHETIC_DATASET
        or manifest.seed != 17
    ):
        raise ValueError(f"{expected_mode} validation input is not the completed deterministic synthetic campaign")
    result_relative = manifest.artifacts.get("result")
    if result_relative is None:
        raise ValueError(f"{expected_mode} synthetic campaign does not seal result.json")
    result_path = directory / result_relative
    if result_path.name != "result.json":
        raise ValueError(f"{expected_mode} synthetic result path is not canonical")
    result = load_json_object(result_path)
    if result.get("mode") != expected_mode or result.get("evidence_scope") != "synthetic" or result.get("operational_status") != "completed":
        raise ValueError(f"{expected_mode} synthetic result has invalid lifecycle metadata")
    passed = _input_software_checks(result) if expected_mode == "input_only" else _output_software_checks(result, native_head_used=manifest.native_head_used)
    if not passed:
        raise ValueError(f"{expected_mode} synthetic campaign did not pass the registered software checks")
    return SyntheticCampaignEvidence(
        mode=expected_mode,
        source=SourceBinding(
            manifest.source_state_sha256,
            manifest.dependency_lock_sha256,
        ),
        operational_status="completed",
        software_checks_passed=True,
        pointer=f"{result_path.resolve()}#",
    )


def load_software_validation_inputs(
    verification: Path,
    input_synthetic: Path,
    output_synthetic: Path,
) -> SoftwareValidationInputs:
    verification_evidence = _verification_evidence(verification)
    campaigns = (
        _synthetic_campaign_evidence(input_synthetic, "input_only"),
        _synthetic_campaign_evidence(output_synthetic, "output_only"),
    )
    if any(campaign.source != verification_evidence.source for campaign in campaigns):
        raise ValueError("verification and deterministic synthetic campaigns have different source_state or dependency_lock identities")
    return SoftwareValidationInputs(
        verification=verification_evidence,
        synthetic_campaigns=campaigns,
    )


def validation_input_directories(
    verification: Path,
    input_synthetic: Path,
    output_synthetic: Path,
) -> dict[str, Path]:
    return {
        "software_verification": _artifact_file(
            verification,
            "verification.json",
        ).parent,
        "software_input_synthetic": _artifact_file(
            input_synthetic,
            "manifest-final.json",
        ).parent,
        "software_output_synthetic": _artifact_file(
            output_synthetic,
            "manifest-final.json",
        ).parent,
    }
