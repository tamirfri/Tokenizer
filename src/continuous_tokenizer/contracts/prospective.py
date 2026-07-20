from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self, cast, final

from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.parsing import (
    exact_fields,
    is_lowercase_sha256,
    mapping_fingerprint,
    non_empty_string,
    non_negative_int,
    positive_int,
    sha256_string,
    table,
)

type ProspectiveTier = Literal[
    "mechanism_smoke",
    "feasibility_screen",
    "candidate_selection",
    "final_evidence",
]
type ProspectiveMode = Literal["input_only", "output_only"]
type CandidateKind = Literal["alignment", "efficiency"]

PROSPECTIVE_ARTIFACT_KINDS: Final = {
    "mechanism_smoke": "prospective_mechanism_smoke",
    "feasibility_screen": "prospective_feasibility_screen",
    "candidate_selection": "prospective_candidate_selection",
    "final_evidence": "prospective_final_evidence",
}
PROSPECTIVE_RESULT_FILENAME: Final = "prospective.json"
PROSPECTIVE_REPORT_FILENAME: Final = "prospective-report.md"
PROSPECTIVE_RESUME_FILENAME: Final = "prospective-resume.json"
PROSPECTIVE_NON_FINAL_FLAGS: Final = {
    "final_claims_allowed": False,
    "project_evidence_eligible": False,
    "replication_eligible": False,
}
_PROSPECTIVE_BUDGETS: Final = {
    "feasibility_screen": {
        "vocabulary_rows": 256,
        "vocabulary_epochs": 1,
        "reconstruction_epochs": 1,
        "reconstruction_samples": 64,
        "distillation_windows": 0,
        "validation_bytes": 256,
        "behavior_samples": 2,
        "generation_samples": 0,
    },
    "candidate_selection": {
        "vocabulary_rows": 512,
        "vocabulary_epochs": 2,
        "reconstruction_epochs": 1,
        "reconstruction_samples": 64,
        "distillation_windows": 0,
        "validation_bytes": 256,
        "behavior_samples": 2,
        "generation_samples": 0,
        "patience": 1,
        "maximum_alignment_trials": 1,
        "maximum_efficiency_trials": 2,
    },
}
_OUTPUT_DISABLED_BUDGET: Final = {
    "reconstruction_epochs": 0,
    "reconstruction_samples": 0,
}
_SPEC_FIELDS: Final = {
    "schema_version",
    "artifact_kind",
    "tier",
    "name",
    "mode",
    "experiment",
    "final_reference",
    "wall_clock",
    "design",
}
_CALIBRATION_FIELDS: Final = {
    "schema_version",
    "artifact_kind",
    "model_id",
    "model_revision",
    "mode",
    "device",
    "work_units",
    "observed_seconds",
}


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def prospective_budget(
    tier: Literal["feasibility_screen", "candidate_selection"],
    mode: ProspectiveMode,
) -> dict[str, int]:
    budget = dict(_PROSPECTIVE_BUDGETS[tier])
    if mode == "output_only":
        budget.update(_OUTPUT_DISABLED_BUDGET)
    return budget


@final
@dataclass(frozen=True, slots=True)
class TimingCalibration:
    locator: str
    sha256: str
    work_units: Mapping[str, int]
    observed_seconds: float

    @classmethod
    def load(cls, value: object, base_directory: Path) -> Self:
        values = table(value, "wall_clock.calibration")
        expected = {"locator", "sha256"}
        exact_fields(
            values,
            expected,
            "wall_clock.calibration",
            incomplete_message="wall_clock.calibration fields are incomplete",
        )
        locator = non_empty_string(values, "locator", "wall_clock.calibration")
        digest = sha256_string(values, "sha256", "wall_clock.calibration")
        path = (base_directory / locator).resolve()
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError("timing calibration locator or SHA-256 is invalid")
        artifact = _load_json_object(path)
        if set(artifact) != _CALIBRATION_FIELDS:
            raise ValueError("timing calibration fields are not canonical")
        if artifact.get("schema_version") != 1 or artifact.get("artifact_kind") != "timing_calibration":
            raise ValueError("timing calibration schema is unsupported")
        raw_units = table(artifact.get("work_units"), "timing calibration work_units")
        work_units = {str(name): non_negative_int(item, f"timing calibration work_units.{name}") for name, item in raw_units.items()}
        observed = artifact.get("observed_seconds")
        if not isinstance(observed, int | float) or isinstance(observed, bool) or observed <= 0:
            raise ValueError("timing calibration observed_seconds must be positive")
        return cls(locator, digest, work_units, float(observed))


@final
@dataclass(frozen=True, slots=True)
class WallClockContract:
    calibration: TimingCalibration
    expected_seconds: int
    maximum_seconds: int
    stop_boundary: str
    work_units: Mapping[str, int]

    @classmethod
    def parse(cls, value: object, base_directory: Path) -> Self:
        values = table(value, "wall_clock")
        expected = {
            "calibration",
            "expected_seconds",
            "maximum_seconds",
            "stop_boundary",
            "work_units",
        }
        exact_fields(
            values,
            expected,
            "wall_clock",
            incomplete_message="wall_clock fields are incomplete",
        )
        units = table(values.get("work_units"), "wall_clock.work_units")
        work_units = {str(name): non_negative_int(item, f"wall_clock.work_units.{name}") for name, item in units.items()}
        contract = cls(
            calibration=TimingCalibration.load(values.get("calibration"), base_directory),
            expected_seconds=positive_int(
                values.get("expected_seconds"),
                "wall_clock.expected_seconds",
            ),
            maximum_seconds=positive_int(
                values.get("maximum_seconds"),
                "wall_clock.maximum_seconds",
            ),
            stop_boundary=non_empty_string(values, "stop_boundary", "wall_clock"),
            work_units=work_units,
        )
        if contract.maximum_seconds < contract.expected_seconds:
            raise ValueError("wall_clock.maximum_seconds must be at least expected_seconds")
        if contract.stop_boundary != "epoch_or_stage":
            raise ValueError("wall-clock enforcement is allowed only at epoch or stage boundaries")
        if contract.work_units != contract.calibration.work_units:
            raise ValueError("wall-clock work units differ from the sealed timing calibration")
        return contract

    def exhausted_at_boundary(self, elapsed_seconds: float, *, at_boundary: bool) -> bool:
        return at_boundary and elapsed_seconds >= self.maximum_seconds


@final
@dataclass(frozen=True, slots=True)
class ProspectiveCandidate:
    name: str
    kind: CandidateKind
    parameters: Mapping[str, int | float | str]

    @classmethod
    def parse_many(cls, value: object) -> tuple[Self, ...]:
        if not isinstance(value, list) or not value:
            raise ValueError("candidate selection requires ordered budget candidates")
        candidates: list[Self] = []
        for index, raw in enumerate(value):
            values = table(raw, f"design.candidates[{index}]")
            expected = {"name", "kind", "parameters"}
            exact_fields(
                values,
                expected,
                f"design.candidates[{index}]",
                incomplete_message="candidate fields are incomplete",
            )
            kind = non_empty_string(values, "kind", f"design.candidates[{index}]")
            if kind not in {"alignment", "efficiency"}:
                raise ValueError("candidate kind must be alignment or efficiency")
            parameters = table(values.get("parameters"), f"design.candidates[{index}].parameters")
            if not parameters or any(isinstance(item, bool) or not isinstance(item, int | float | str) for item in parameters.values()):
                raise ValueError("candidate parameters must contain scalar configuration values")
            candidates.append(
                cls(
                    non_empty_string(values, "name", f"design.candidates[{index}]"),
                    kind,
                    dict(parameters),
                ),
            )
        if len({candidate.name for candidate in candidates}) != len(candidates):
            raise ValueError("candidate names must be unique")
        if tuple(candidate.kind for candidate in candidates) != (
            "alignment",
            "efficiency",
            "efficiency",
        ):
            raise ValueError(
                "candidate selection requires one alignment candidate followed by two efficiency candidates",
            )
        return tuple(candidates)


@final
@dataclass(frozen=True, slots=True)
class ProspectiveSpec:
    schema_version: int
    artifact_kind: str
    tier: ProspectiveTier
    name: str
    mode: ProspectiveMode
    experiment: str
    final_reference: str
    wall_clock: WallClockContract
    design: Mapping[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        exact_fields(
            values,
            _SPEC_FIELDS,
            "prospective wrapper",
            incomplete_message="prospective wrapper fields are incomplete",
        )
        if values.get("schema_version") != 1:
            raise ValueError("prospective wrapper schema_version must be 1")
        tier = non_empty_string(values, "tier", "prospective wrapper")
        if tier not in PROSPECTIVE_ARTIFACT_KINDS:
            raise ValueError("prospective wrapper tier is invalid")
        if values.get("artifact_kind") != PROSPECTIVE_ARTIFACT_KINDS[tier]:
            raise ValueError("prospective wrapper artifact_kind does not match its tier")
        mode = non_empty_string(values, "mode", "prospective wrapper")
        if mode not in {"input_only", "output_only"}:
            raise ValueError("prospective wrapper mode is invalid")
        spec = cls(
            schema_version=1,
            artifact_kind=str(values["artifact_kind"]),
            tier=cast(ProspectiveTier, tier),
            name=non_empty_string(values, "name", "prospective wrapper"),
            mode=mode,
            experiment=non_empty_string(values, "experiment", "prospective wrapper"),
            final_reference=non_empty_string(values, "final_reference", "prospective wrapper"),
            wall_clock=WallClockContract.parse(values.get("wall_clock"), path.parent),
            design=dict(table(values.get("design"), "design")),
            path=path.resolve(),
        )
        spec._validate()
        return spec

    def _validate(self) -> None:
        experiment = self.load_experiment()
        final_reference = self.load_final_reference()
        if experiment.mode != self.mode or final_reference.mode != self.mode:
            raise ValueError("prospective wrapper mode differs from its experiments")
        if self.tier == "mechanism_smoke":
            self._validate_mechanism(experiment)
            return
        if experiment.model != final_reference.model or experiment.dataset != final_reference.dataset:
            raise ValueError("prospective experiment source identity differs from its final reference")
        if asdict(experiment.gates) != asdict(final_reference.gates):
            raise ValueError("prospective gates must remain numerically identical to the final gate set")
        if self.tier == "feasibility_screen":
            self._validate_screen(experiment)
        elif self.tier == "candidate_selection":
            self._validate_selection()
        else:
            self._validate_final(experiment)

    def _validate_mechanism(self, experiment: ExperimentSpec) -> None:
        expected = {"proof_role"}
        if set(self.design) != expected or self.design.get("proof_role") not in {
            "synthetic",
            "diagnostic",
        }:
            raise ValueError("mechanism smoke must consume synthetic or diagnostic proof only")
        if experiment.evidence_scope not in {"synthetic", "diagnostic"}:
            raise ValueError("mechanism smoke experiment is not synthetic or diagnostic")

    def _validate_screen(self, experiment: ExperimentSpec) -> None:
        budget = prospective_budget("feasibility_screen", self.mode)
        expected = {
            *budget,
            "load_final_test",
        }
        if self.mode == "input_only":
            expected.update(
                {
                    "subset_seed",
                    "subset_sha256",
                    "subset_content_hashed",
                },
            )
        if set(self.design) != expected:
            raise ValueError("feasibility screen design fields are not canonical")
        if any(self.design[name] != value for name, value in budget.items()):
            raise ValueError("feasibility screen budget differs from the registered design")
        if experiment.seed != 17 or getattr(experiment.training, "profile", None) != "large" or self.design.get("load_final_test") is not False:
            raise ValueError("feasibility screen requires Large, seed 17, and validation-only data")
        if self.mode == "input_only" and (
            self.design.get("subset_seed") != 17
            or not is_lowercase_sha256(self.design.get("subset_sha256"))
            or self.design.get("subset_content_hashed") is not True
        ):
            raise ValueError("input feasibility screen requires a sealed validation subset")

    def _validate_selection(self) -> None:
        budget = prospective_budget("candidate_selection", self.mode)
        expected = {
            *budget,
            "sampler_seed",
            "data_role",
            "load_final_test",
            "candidates",
        }
        if self.mode == "input_only":
            expected.update({"subset_seed", "subset_sha256"})
        if set(self.design) != expected:
            raise ValueError("candidate selection design fields are not canonical")
        if any(self.design[name] != value for name, value in budget.items()):
            raise ValueError("candidate selection budget differs from the registered design")
        if self.mode == "input_only" and not is_lowercase_sha256(
            self.design.get("subset_sha256"),
        ):
            raise ValueError("candidate selection subset_sha256 is invalid")
        if self.design.get("data_role") != "validation" or self.design.get("load_final_test") is not False:
            raise ValueError("candidate selection must remain validation-only")
        ProspectiveCandidate.parse_many(self.design.get("candidates"))

    def _validate_final(self, experiment: ExperimentSpec) -> None:
        expected = {
            "seed",
            "complete_vocabulary",
            "full_final_evaluation",
            "independent_retraining",
            "reuse_study_weights",
            "reuse_validation_metrics",
            "selection_artifact",
            "selection_artifact_sha256",
            "selected_strategy",
            "selected_configuration",
            "selected_budget",
        }
        if set(self.design) != expected:
            raise ValueError("final prospective design fields are not canonical")
        if (
            self.design.get("seed") not in {17, 23, 41}
            or experiment.seed != self.design.get("seed")
            or experiment.evidence_scope != "final"
            or self.design.get("complete_vocabulary") is not True
            or self.design.get("full_final_evaluation") is not True
            or self.design.get("independent_retraining") is not True
            or self.design.get("reuse_study_weights") is not False
            or self.design.get("reuse_validation_metrics") is not False
        ):
            raise ValueError("final prospective execution is not an independent complete final run")
        artifact = (self.path.parent / str(self.design["selection_artifact"])).resolve()
        digest = self.design.get("selection_artifact_sha256")
        if not artifact.is_file() or not is_lowercase_sha256(digest) or _sha256_file(artifact) != digest:
            raise ValueError("final selection provenance is missing or tampered")

    def load_experiment(self) -> ExperimentSpec:
        return ExperimentSpec.load((self.path.parent / self.experiment).resolve())

    def load_final_reference(self) -> ExperimentSpec:
        return ExperimentSpec.load((self.path.parent / self.final_reference).resolve())

    def candidates(self) -> tuple[ProspectiveCandidate, ...]:
        return ProspectiveCandidate.parse_many(self.design.get("candidates"))

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "tier": self.tier,
            "name": self.name,
            "mode": self.mode,
            "experiment": self.experiment,
            "final_reference": self.final_reference,
            "wall_clock": {
                "calibration": {
                    "locator": self.wall_clock.calibration.locator,
                    "sha256": self.wall_clock.calibration.sha256,
                },
                "expected_seconds": self.wall_clock.expected_seconds,
                "maximum_seconds": self.wall_clock.maximum_seconds,
                "stop_boundary": self.wall_clock.stop_boundary,
                "work_units": dict(self.wall_clock.work_units),
            },
            "design": dict(self.design),
        }


@final
@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    name: str
    kind: CandidateKind
    operational_passed: bool
    invariant_passed: bool
    exactness_passed: bool | None
    density_passed: bool | None
    behavior_passed: bool | None
    compactness_passed: bool | None
    alignment_passed: bool | None = None
    budget_exhausted: bool = False

    @property
    def selectable(self) -> bool:
        return (
            self.kind == "efficiency"
            and self.operational_passed
            and self.invariant_passed
            and self.exactness_passed is True
            and self.density_passed is True
            and self.behavior_passed is True
            and self.compactness_passed is True
            and not self.budget_exhausted
        )


def _stage_status(value: bool | None, *, not_run: str) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return not_run


def prospective_stage_records(
    statuses: Mapping[str, str],
) -> list[dict[str, str]]:
    return [{"name": name, "status": status} for name, status in statuses.items()]


def futility_stages(outcome: CandidateOutcome) -> dict[str, str]:
    if not outcome.operational_passed or not outcome.invariant_passed:
        return {
            "exactness": "not_run_operational_failure",
            "density": "not_run_operational_failure",
            "behavior": "not_run_operational_failure",
        }
    if outcome.budget_exhausted:
        values = {
            "exactness": outcome.exactness_passed,
            "density": outcome.density_passed,
            "behavior": outcome.behavior_passed,
        }
        return {name: _stage_status(value, not_run="not_run_budget") for name, value in values.items()}
    if outcome.exactness_passed is not True:
        return {
            "exactness": _stage_status(
                outcome.exactness_passed,
                not_run="not_run_futility",
            ),
            "density": "not_run_futility",
            "behavior": "not_run_futility",
        }
    if outcome.density_passed is not True:
        return {
            "exactness": "passed",
            "density": _stage_status(
                outcome.density_passed,
                not_run="not_run_futility",
            ),
            "behavior": "not_run_futility",
        }
    return {
        "exactness": "passed",
        "density": "passed",
        "behavior": _stage_status(
            outcome.behavior_passed,
            not_run="not_run_futility",
        ),
    }


def select_smallest_candidate(
    candidates: Sequence[ProspectiveCandidate],
    outcomes: Sequence[CandidateOutcome],
) -> tuple[ProspectiveCandidate | None, bool]:
    by_name = {outcome.name: outcome for outcome in outcomes}
    for candidate in candidates:
        outcome = by_name.get(candidate.name)
        if outcome is not None and outcome.selectable:
            return candidate, True
    return None, False


def prospective_result_errors(result: Mapping[str, Any]) -> list[str]:
    required = {
        "schema_version",
        "artifact_kind",
        "tier",
        "name",
        "mode",
        "operational_status",
        "scientific_verdict",
        "spec_fingerprint",
        "budget_exhausted",
        "wall_clock",
        "stages",
        "selection",
        "final_claims_allowed",
        "project_evidence_eligible",
        "replication_eligible",
    }
    errors = []
    if set(result) != required:
        errors.append("prospective result fields are not canonical")
        return errors
    tier = result.get("tier")
    if result.get("schema_version") != 1 or tier not in PROSPECTIVE_ARTIFACT_KINDS:
        errors.append("prospective result schema is unsupported")
    elif result.get("artifact_kind") != PROSPECTIVE_ARTIFACT_KINDS[str(tier)]:
        errors.append("prospective result artifact kind differs from its tier")
    if tier != "final_evidence" and any(result.get(name) is not False for name in PROSPECTIVE_NON_FINAL_FLAGS):
        errors.append("non-final prospective result promotes a final claim")
    if result.get("budget_exhausted") is True and result.get("scientific_verdict") in {
        "supported",
        "passed",
    }:
        errors.append("budget-exhausted prospective work can never pass")
    if tier == "candidate_selection":
        selection = result.get("selection")
        if not isinstance(selection, Mapping) or not isinstance(selection.get("selection_feasible"), bool):
            errors.append("candidate selection result has no explicit feasibility outcome")
    return errors
