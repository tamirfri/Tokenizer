from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Self, final

from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.output import (
    OUTPUT_ORACLE_SPAN_LIMITS,
    OutputEvaluationSpec,
    OutputGateSpec,
    OutputTrainingSpec,
)
from continuous_tokenizer.contracts.parsing import (
    exact_fields,
    mapping_fingerprint,
    non_empty_string,
)

OUTPUT_ORACLE_SELECTION_RULE: Final = "largest_feasible_span"


@final
@dataclass(frozen=True, slots=True)
class OutputOracleStudySpec:
    name: str
    experiment: str
    corpus_role: str
    span_limits: tuple[int, ...]
    selection_rule: str
    study: str = "output_oracle"

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        expected = {
            "study",
            "name",
            "experiment",
            "corpus_role",
            "span_limits",
            "selection_rule",
        }
        exact_fields(
            values,
            expected,
            "output oracle study",
            incomplete_message="output oracle study fields are incomplete",
        )
        if values.get("study") != "output_oracle":
            raise ValueError("study must be 'output_oracle'")
        raw_limits = values.get("span_limits")
        if not isinstance(raw_limits, list) or tuple(raw_limits) != OUTPUT_ORACLE_SPAN_LIMITS:
            raise ValueError("output oracle studies require exact span limits 1, 2, 4, and 8")
        spec = cls(
            name=non_empty_string(values, "name", "output oracle study"),
            experiment=non_empty_string(values, "experiment", "output oracle study"),
            corpus_role=non_empty_string(values, "corpus_role", "output oracle study"),
            span_limits=OUTPUT_ORACLE_SPAN_LIMITS,
            selection_rule=non_empty_string(values, "selection_rule", "output oracle study"),
        )
        if spec.selection_rule != OUTPUT_ORACLE_SELECTION_RULE:
            raise ValueError(f"output oracle studies require selection_rule = {OUTPUT_ORACLE_SELECTION_RULE!r}")
        if spec.corpus_role != "oracle_validation":
            raise ValueError("output oracle studies require corpus_role = 'oracle_validation'")
        experiment = spec.load_experiment(path)
        if (
            experiment.mode != "output_only"
            or not isinstance(experiment.training, OutputTrainingSpec)
            or not isinstance(experiment.evaluation, OutputEvaluationSpec)
            or not isinstance(experiment.gates, OutputGateSpec)
        ):
            raise ValueError("output oracle studies require an output-only experiment")
        if experiment.training.profile != "large" or experiment.seed != 17:
            raise ValueError("output oracle studies require the Large profile and seed 17")
        if experiment.model.evaluation != "full" or experiment.evidence_scope == "final":
            raise ValueError("output oracle studies require a non-final full-model experiment")
        return spec

    def load_experiment(self, study_path: Path) -> ExperimentSpec:
        return ExperimentSpec.load((study_path.parent / self.experiment).resolve())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())
