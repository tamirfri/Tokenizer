from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, final

from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.artifacts.store import load_json_object, write_json_atomic
from continuous_tokenizer.contracts.parsing import is_lowercase_sha256, mapping_fingerprint

CALIBRATION_ARTIFACT_KIND: Final = "input_evaluation_calibration"
UNIQUE_SAMPLE_COUNT: Final = 2
CALIBRATION_ROWS: Final = 8
PRODUCTION_BATCH_SIZE: Final = 8
_COUNT_FIELDS: Final = frozenset(
    {
        "unique_sample_count",
        "calibration_rows",
        "production_batch_size",
        "paths",
        "tokens",
        "scalar_model_forwards",
        "batched_model_forwards",
    }
)
_MEASUREMENT_FIELDS: Final = _COUNT_FIELDS | {
    "maximum_kl",
    "maximum_nll_delta",
    "top1_agreement",
    "maximum_logit_error",
}


@final
@dataclass(frozen=True, slots=True)
class CalibrationTolerance:
    maximum_kl: float = 1e-4
    maximum_nll_delta: float = 1e-3
    minimum_top1_agreement: float = 1.0
    maximum_logit_error: float = 1e-2

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_kl,
                self.maximum_nll_delta,
                self.minimum_top1_agreement,
                self.maximum_logit_error,
            )
            < 0
        ):
            raise ValueError("evaluation calibration tolerances must be non-negative")
        if self.minimum_top1_agreement > 1:
            raise ValueError("evaluation calibration top-1 agreement must not exceed one")


@final
@dataclass(frozen=True, slots=True)
class CalibrationIdentity:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    model_fingerprint: str
    codec_checkpoint_fingerprint: str
    segmentation_alignment: str
    source_commit: str
    source_state_sha256: str
    dtype: str
    device: str
    production_batch_size: int
    unique_sample_count: int
    calibration_rows: int
    implementation_sha256: str
    unique_samples_sha256: str
    calibration_rows_sha256: str
    tolerance: CalibrationTolerance
    dependency_lock_sha256: str

    def __post_init__(self) -> None:
        if not self.source_commit:
            raise ValueError("evaluation calibration source commit must not be empty")
        if self.production_batch_size != PRODUCTION_BATCH_SIZE:
            raise ValueError("evaluation calibration requires production batch size eight")
        if self.unique_sample_count != UNIQUE_SAMPLE_COUNT:
            raise ValueError("evaluation calibration requires exactly two registered samples")
        if self.calibration_rows != CALIBRATION_ROWS:
            raise ValueError("evaluation calibration requires exactly eight calibration rows")
        for name in (
            "model_fingerprint",
            "codec_checkpoint_fingerprint",
            "source_state_sha256",
            "implementation_sha256",
            "unique_samples_sha256",
            "calibration_rows_sha256",
            "dependency_lock_sha256",
        ):
            if not is_lowercase_sha256(getattr(self, name)):
                raise ValueError(f"evaluation calibration {name} is not a SHA-256 digest")

    @property
    def key(self) -> str:
        return mapping_fingerprint(asdict(self))


@final
@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    artifact: Mapping[str, Any]
    cache_path: Path
    sha256: str
    built: bool

    def materialize(self, destination: Path) -> dict[str, Any]:
        write_json_atomic(
            destination,
            {
                "artifact_kind": "input_evaluation_calibration_record",
                "calibration": dict(self.artifact),
                "cache": {
                    "locator": str(self.cache_path),
                    "sha256": self.sha256,
                    "built": self.built,
                },
            },
        )
        digest = sha256_file(destination)
        return {
            "locator": destination.name,
            "sha256": digest,
            "key": str(self.artifact["key"]),
            "built": self.built,
        }


def _validated_measurements(
    measurements: Mapping[str, Any],
) -> dict[str, float | int]:
    if set(measurements) != _MEASUREMENT_FIELDS:
        raise ValueError("evaluation calibration measurement fields are not canonical")
    for value in measurements.values():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError("evaluation calibration measurements must be finite and non-negative")
    if any(not isinstance(measurements[name], int) for name in _COUNT_FIELDS):
        raise ValueError("evaluation calibration counts must be integers")
    if measurements["top1_agreement"] > 1:
        raise ValueError("evaluation calibration top-1 agreement must not exceed one")
    return dict(measurements)


def _artifact_content(
    identity: CalibrationIdentity,
    raw_measurements: Mapping[str, Any],
) -> dict[str, Any]:
    measurements = _validated_measurements(raw_measurements)
    for name in (
        "unique_sample_count",
        "calibration_rows",
        "production_batch_size",
    ):
        if measurements[name] != getattr(identity, name):
            raise ValueError(f"evaluation calibration {name} differs from its identity")
    passed = (
        float(measurements["maximum_kl"]) <= identity.tolerance.maximum_kl
        and float(measurements["maximum_nll_delta"]) <= identity.tolerance.maximum_nll_delta
        and float(measurements["top1_agreement"]) >= identity.tolerance.minimum_top1_agreement
        and float(measurements["maximum_logit_error"]) <= identity.tolerance.maximum_logit_error
    )
    return {
        "artifact_kind": CALIBRATION_ARTIFACT_KIND,
        "identity": asdict(identity),
        "key": identity.key,
        "measurements": measurements,
        "passed": passed,
        "scientific_evidence": False,
    }


def _sealed_artifact(content: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **content,
        "content_sha256": mapping_fingerprint(content),
    }


def _verify_artifact(
    raw: Mapping[str, Any],
    identity: CalibrationIdentity,
) -> dict[str, Any]:
    expected_fields = {
        "artifact_kind",
        "identity",
        "key",
        "measurements",
        "passed",
        "scientific_evidence",
        "content_sha256",
    }
    if set(raw) != expected_fields:
        raise ValueError("evaluation calibration fields are not canonical")
    content = {name: value for name, value in raw.items() if name != "content_sha256"}
    if raw["content_sha256"] != mapping_fingerprint(content):
        raise ValueError("evaluation calibration content was modified")
    if (
        raw["artifact_kind"] != CALIBRATION_ARTIFACT_KIND
        or raw["identity"] != asdict(identity)
        or raw["key"] != identity.key
        or raw["scientific_evidence"] is not False
    ):
        raise ValueError("evaluation calibration identity does not match")
    measurements = raw["measurements"]
    if not isinstance(measurements, Mapping):
        raise ValueError("evaluation calibration measurements are invalid")
    expected = _artifact_content(identity, measurements)
    if any(content.get(name) != value for name, value in expected.items()):
        raise ValueError("evaluation calibration verdict does not match its measurements")
    if raw["passed"] is not True:
        raise ValueError("evaluation calibration failed")
    return dict(raw)


def load_or_build_calibration(
    cache_directory: Path,
    identity: CalibrationIdentity,
    build: Callable[[], Mapping[str, float | int]],
) -> CalibrationRecord:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / f"{identity.key}.json"
    if path.exists():
        artifact = _verify_artifact(load_json_object(path), identity)
        return CalibrationRecord(
            artifact=artifact,
            cache_path=path,
            sha256=sha256_file(path),
            built=False,
        )
    artifact = _sealed_artifact(_artifact_content(identity, build()))
    _verify_artifact(artifact, identity)
    write_json_atomic(path, artifact)
    verified = _verify_artifact(load_json_object(path), identity)
    return CalibrationRecord(
        artifact=verified,
        cache_path=path,
        sha256=sha256_file(path),
        built=True,
    )
