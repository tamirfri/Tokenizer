from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self, final

from continuous_tokenizer.contracts.parsing import (
    exact_fields,
    non_empty_string,
    scalar_mapping,
    sha256_string,
    table,
)


@final
@dataclass(frozen=True, slots=True)
class ProspectiveSelectionSpec:
    artifact: str
    artifact_sha256: str
    candidate_toml: str
    candidate_toml_sha256: str
    calibration: str
    calibration_sha256: str
    frozen_toml: str
    frozen_toml_sha256: str
    spec_fingerprint: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_config: str
    dataset_revision: str
    profile: str
    selected_strategy: str
    selected_parameters: Mapping[str, int | float | str]
    source_commit: str
    source_state_sha256: str
    dependency_lock_sha256: str

    @classmethod
    def parse(cls, value: object, base_directory: Path) -> Self:
        values = table(value, "prospective selection")
        expected = {
            "artifact",
            "artifact_sha256",
            "candidate_toml",
            "candidate_toml_sha256",
            "calibration",
            "calibration_sha256",
            "frozen_toml",
            "frozen_toml_sha256",
            "spec_fingerprint",
            "model_id",
            "model_revision",
            "dataset_id",
            "dataset_config",
            "dataset_revision",
            "profile",
            "selected_strategy",
            "selected_parameters",
            "source_commit",
            "source_state_sha256",
            "dependency_lock_sha256",
        }
        exact_fields(values, expected, "prospective selection")

        def path(name: str) -> str:
            relative = non_empty_string(values, name, "prospective selection")
            return str((base_directory / relative).resolve())

        return cls(
            artifact=path("artifact"),
            artifact_sha256=sha256_string(
                values,
                "artifact_sha256",
                "prospective selection",
            ),
            candidate_toml=path("candidate_toml"),
            candidate_toml_sha256=sha256_string(
                values,
                "candidate_toml_sha256",
                "prospective selection",
            ),
            calibration=path("calibration"),
            calibration_sha256=sha256_string(
                values,
                "calibration_sha256",
                "prospective selection",
            ),
            frozen_toml=path("frozen_toml"),
            frozen_toml_sha256=sha256_string(
                values,
                "frozen_toml_sha256",
                "prospective selection",
            ),
            spec_fingerprint=sha256_string(
                values,
                "spec_fingerprint",
                "prospective selection",
            ),
            model_id=non_empty_string(values, "model_id", "prospective selection"),
            model_revision=non_empty_string(values, "model_revision", "prospective selection"),
            dataset_id=non_empty_string(values, "dataset_id", "prospective selection"),
            dataset_config=non_empty_string(values, "dataset_config", "prospective selection"),
            dataset_revision=non_empty_string(values, "dataset_revision", "prospective selection"),
            profile=non_empty_string(values, "profile", "prospective selection"),
            selected_strategy=non_empty_string(values, "selected_strategy", "prospective selection"),
            selected_parameters=scalar_mapping(
                values,
                "selected_parameters",
                "prospective selection",
            ),
            source_commit=non_empty_string(values, "source_commit", "prospective selection"),
            source_state_sha256=sha256_string(
                values,
                "source_state_sha256",
                "prospective selection",
            ),
            dependency_lock_sha256=sha256_string(
                values,
                "dependency_lock_sha256",
                "prospective selection",
            ),
        )
