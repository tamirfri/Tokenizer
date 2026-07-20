from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self, cast, final

from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import (
    InputEvaluationSpec,
    InputTrainingSpec,
)
from continuous_tokenizer.contracts.parsing import (
    exact_fields,
    mapping_fingerprint,
    non_empty_string,
    positive_int,
)

INPUT_STUDY_CANDIDATE_LENGTHS: Final = (2, 8, 32)
INPUT_SELECTION_RULE: Final = "feasible_min_kl_js_nll_max_top1_density"
INPUT_SELECTION_CANDIDATES: Final = (
    "reconstruction_only",
    "token_aligned_distillation",
    "arbitrary_boundary_distillation",
)
INPUT_ALIGNMENT_FEASIBILITY_STAGES: Final = (128, 256, 512)
INPUT_ALIGNMENT_CONTINUATION_RULE: Final = "continue_only_while_alignment_gates_pass"
INPUT_ALIGNMENT_TRAINING_SEEDS: Final = (17,)
INPUT_ALIGNMENT_SUBSET_SEED: Final = 17
INPUT_COMPRESSION_TRAINING_SEEDS: Final = (17,)
INPUT_COMPRESSION_SUBSET_SEED: Final = 17
INPUT_COMPRESSION_VOCABULARY_ROWS: Final = 512
INPUT_COMPRESSION_CONTINUATION_RULE: Final = "continue_only_after_all_seed_exactness_density_and_behavior_pass"
INPUT_COMPRESSION_FINAL_ACTION: Final = "record_freeze_eligibility_only"

type InputStudyKind = Literal["scaling"]
type InputAlignmentFeasibilityVerdict = Literal["supported", "unsupported"]


def _load_study_values(
    path: Path,
    *,
    table_name: str,
    study: str,
    fields: set[str],
) -> dict[str, Any]:
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    expected = {"study", *fields}
    exact_fields(values, expected, table_name)
    if values.get("study") != study:
        raise ValueError(f"study must be {study!r}")
    return values


_ALIGNMENT_GATES = (
    (
        "maximum_normalized_rmse",
        "normalized_rmse",
        "less_than_or_equal",
    ),
    (
        "minimum_cosine_p01",
        "cosine_similarity_p01",
        "greater_than_or_equal",
    ),
    (
        "minimum_cosine_p50",
        "cosine_similarity_p50",
        "greater_than_or_equal",
    ),
)


def failed_alignment_gates(
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> list[dict[str, object]]:
    failed: list[dict[str, object]] = []
    for threshold_name, metric_name, comparison in _ALIGNMENT_GATES:
        measured = metrics.get(metric_name)
        threshold = gates.get(threshold_name)
        if isinstance(measured, bool) or not isinstance(measured, int | float) or isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ValueError("alignment feasibility metrics and gates must be numeric")
        passed = measured <= threshold if comparison == "less_than_or_equal" else measured >= threshold
        if not passed:
            failed.append(
                {
                    "gate": threshold_name,
                    "metric": metric_name,
                    "measured": float(measured),
                    "threshold": float(threshold),
                    "comparison": comparison,
                },
            )
    return failed


def _fixed_seed_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes) and tuple(value) == INPUT_ALIGNMENT_TRAINING_SEEDS


def _alignment_stage_seed_rows(
    stage: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    subset = stage.get("vocabulary_subset")
    seed_results = stage.get("seed_results")
    if (
        not isinstance(subset, Mapping)
        or stage.get("subset_sha256") != subset.get("sha256")
        or not _fixed_seed_sequence(stage.get("training_seeds"))
        or not isinstance(seed_results, Sequence)
        or isinstance(seed_results, str | bytes)
    ):
        raise ValueError("alignment stage does not expose its fixed subset and seeds")
    per_seed = [cast(Mapping[str, Any], seed_result) for seed_result in seed_results if isinstance(seed_result, Mapping)]
    if len(per_seed) != len(INPUT_ALIGNMENT_TRAINING_SEEDS) or tuple(seed.get("training_seed") for seed in per_seed) != INPUT_ALIGNMENT_TRAINING_SEEDS:
        raise ValueError("alignment stage does not expose every registered training seed")
    return per_seed


def _alignment_stage_failed_seeds(
    stage: Mapping[str, Any],
    per_seed: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> list[dict[str, object]]:
    failed_seeds: list[dict[str, object]] = []
    for seed in per_seed:
        alignment = seed.get("alignment")
        metrics = alignment.get("embedding_metrics") if isinstance(alignment, Mapping) else None
        if not isinstance(metrics, Mapping) or seed.get("subset_sha256") != stage["subset_sha256"]:
            raise ValueError("alignment seed result lacks fixed-subset metrics")
        failed_gates = failed_alignment_gates(metrics, gates)
        expected_status = "failed_gate" if failed_gates else "passed"
        if seed.get("status") != expected_status or seed.get("failed_gates") != failed_gates:
            raise ValueError("alignment seed status differs from its raw metrics")
        if failed_gates:
            failed_seeds.append(
                {
                    "training_seed": seed["training_seed"],
                    "failed_gates": failed_gates,
                },
            )
    return failed_seeds


def _validate_futility_stage(
    stage: Mapping[str, Any],
    per_seed: Sequence[Mapping[str, Any]],
    failed_stage: Mapping[str, Any],
) -> None:
    if (
        stage.get("status") != "not_run_futility"
        or stage.get("failed_prerequisite_stage") != failed_stage.get("vocabulary_subset_size")
        or stage.get("failed_gates") != failed_stage.get("failed_gates")
        or any(seed.get("status") != "not_run_futility" for seed in per_seed)
    ):
        raise ValueError("alignment futility status is inconsistent")


def alignment_feasibility_verdict(
    result: Mapping[str, Any],
) -> InputAlignmentFeasibilityVerdict:
    study = result.get("study")
    stages = result.get("stages")
    fixed_design = (
        isinstance(study, Mapping)
        and _fixed_seed_sequence(study.get("training_seeds"))
        and study.get("subset_seed") == INPUT_ALIGNMENT_SUBSET_SEED
        and _fixed_seed_sequence(result.get("training_seeds"))
        and result.get("subset_seed") == INPUT_ALIGNMENT_SUBSET_SEED
    )
    if not fixed_design or not isinstance(stages, Sequence) or isinstance(stages, str | bytes):
        raise ValueError("alignment evidence does not declare the fixed seed design")
    stage_rows = [cast(Mapping[str, Any], stage) for stage in stages if isinstance(stage, Mapping)]
    sizes = tuple(stage.get("vocabulary_subset_size") for stage in stage_rows)
    if len(stage_rows) != len(INPUT_ALIGNMENT_FEASIBILITY_STAGES) or sizes != INPUT_ALIGNMENT_FEASIBILITY_STAGES:
        raise ValueError("alignment evidence stages do not match the fixed prospective design")
    gates = result.get("acceptance_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("alignment evidence does not expose its registered gates")

    failed_stage: Mapping[str, Any] | None = None
    for stage in stage_rows:
        per_seed = _alignment_stage_seed_rows(stage)
        if failed_stage is not None:
            _validate_futility_stage(stage, per_seed, failed_stage)
            continue
        failed_seeds = _alignment_stage_failed_seeds(stage, per_seed, gates)
        expected_status = "failed_gate" if failed_seeds else "passed"
        if stage.get("status") != expected_status or stage.get("failed_gates") != failed_seeds:
            raise ValueError("alignment stage status differs from its seed results")
        if failed_seeds:
            failed_stage = stage

    all_passed = failed_stage is None
    if result.get("feasibility_passed") is not all_passed:
        raise ValueError("alignment feasibility outcome differs from its stages")
    return "supported" if all_passed else "unsupported"


def _integer_array(value: object, name: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain only integers")
    result = tuple(cast(int, item) for item in value)
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values of at least {minimum}")
    return result


@final
@dataclass(frozen=True, slots=True)
class InputSelectionStudySpec:
    name: str
    kind: InputStudyKind
    experiment: str
    vocabulary_subset_sizes: tuple[int, ...]
    candidate_lengths: tuple[int, ...]
    binary_samples_per_length: int
    validation_bytes: int
    run_selection: bool
    selection_rule: str
    study: str = "input_selection"

    @classmethod
    def load(cls, path: Path) -> Self:
        table_name = "input selection study"
        values = _load_study_values(
            path,
            table_name=table_name,
            study="input_selection",
            fields={
                "name",
                "kind",
                "experiment",
                "vocabulary_subset_sizes",
                "candidate_lengths",
                "binary_samples_per_length",
                "validation_bytes",
                "run_selection",
                "selection_rule",
            },
        )
        raw_kind = non_empty_string(values, "kind", table_name)
        if raw_kind != "scaling":
            raise ValueError("input selection study.kind is invalid")
        run_selection = values.get("run_selection")
        if not isinstance(run_selection, bool):
            raise ValueError("input selection study.run_selection must be boolean")
        spec = cls(
            name=non_empty_string(values, "name", table_name),
            kind=raw_kind,
            experiment=non_empty_string(values, "experiment", table_name),
            vocabulary_subset_sizes=_integer_array(
                values.get("vocabulary_subset_sizes"),
                "input selection study.vocabulary_subset_sizes",
                allow_zero=True,
            ),
            candidate_lengths=_integer_array(
                values.get("candidate_lengths"),
                "input selection study.candidate_lengths",
            ),
            binary_samples_per_length=positive_int(
                values.get("binary_samples_per_length"),
                "input selection study.binary_samples_per_length",
            ),
            validation_bytes=positive_int(
                values.get("validation_bytes"),
                "input selection study.validation_bytes",
            ),
            run_selection=run_selection,
            selection_rule=non_empty_string(
                values,
                "selection_rule",
                "input selection study",
            ),
        )
        spec._validate(path)
        return spec

    def _validate(self, path: Path) -> None:
        self._validate_shape()
        self._validate_experiment(self.load_experiment(path))

    def _validate_shape(self) -> None:
        if self.candidate_lengths != INPUT_STUDY_CANDIDATE_LENGTHS:
            raise ValueError(
                "input selection studies require exact candidate lengths 2, 8, and 32",
            )
        if self.selection_rule != INPUT_SELECTION_RULE:
            raise ValueError(f"input selection studies require selection_rule = {INPUT_SELECTION_RULE!r}")
        if tuple(sorted(self.vocabulary_subset_sizes)) != self.vocabulary_subset_sizes:
            raise ValueError("vocabulary subset sizes must be strictly increasing")
        if self.run_selection or 0 in self.vocabulary_subset_sizes:
            raise ValueError("scaling studies require non-zero subsets and no candidate selection")

    @staticmethod
    def _validate_experiment(experiment: ExperimentSpec) -> None:
        training = experiment.training
        if not isinstance(training, InputTrainingSpec) or experiment.mode != "input_only":
            raise ValueError("input selection studies require an input-only experiment")
        if training.profile != "large":
            raise ValueError("input selection studies require the Large profile")
        if experiment.seed != 17:
            raise ValueError("input selection studies require seed 17")
        if experiment.model.evaluation != "full":
            raise ValueError("input selection studies require full model evaluation")
        required = {"vocabulary", "reconstruction"}
        if not required.issubset(experiment.stages):
            raise ValueError("input selection studies require vocabulary and reconstruction stages")
        if experiment.evidence_scope == "final":
            raise ValueError("selection studies must not reference a final-evidence experiment")

    def load_experiment(self, study_path: Path) -> ExperimentSpec:
        return ExperimentSpec.load((study_path.parent / self.experiment).resolve())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())


@final
@dataclass(frozen=True, slots=True)
class InputAlignmentFeasibilityStudySpec:
    name: str
    experiment: str
    vocabulary_subset_sizes: tuple[int, ...]
    training_seeds: tuple[int, ...]
    subset_seed: int
    continuation_rule: str
    prospective: bool
    study: str = "input_alignment_feasibility"

    @classmethod
    def load(cls, path: Path) -> Self:
        table_name = "input alignment feasibility study"
        values = _load_study_values(
            path,
            table_name=table_name,
            study="input_alignment_feasibility",
            fields={
                "name",
                "experiment",
                "vocabulary_subset_sizes",
                "training_seeds",
                "subset_seed",
                "continuation_rule",
                "prospective",
            },
        )
        prospective = values.get("prospective")
        if prospective is not True:
            raise ValueError("input alignment feasibility studies must be prospective")
        spec = cls(
            name=non_empty_string(values, "name", table_name),
            experiment=non_empty_string(
                values,
                "experiment",
                table_name,
            ),
            vocabulary_subset_sizes=_integer_array(
                values.get("vocabulary_subset_sizes"),
                f"{table_name}.vocabulary_subset_sizes",
            ),
            training_seeds=_integer_array(
                values.get("training_seeds"),
                f"{table_name}.training_seeds",
            ),
            subset_seed=positive_int(
                values.get("subset_seed"),
                f"{table_name}.subset_seed",
            ),
            continuation_rule=non_empty_string(
                values,
                "continuation_rule",
                table_name,
            ),
            prospective=prospective,
        )
        spec._validate(path)
        return spec

    def _validate(self, path: Path) -> None:
        self._validate_shape()
        self._validate_experiment(self.load_experiment(path))

    def _validate_shape(self) -> None:
        if self.vocabulary_subset_sizes != INPUT_ALIGNMENT_FEASIBILITY_STAGES:
            raise ValueError(
                "input alignment feasibility studies require exact stages 128, 256, and 512",
            )
        if self.training_seeds != INPUT_ALIGNMENT_TRAINING_SEEDS:
            raise ValueError(
                "input alignment feasibility studies require training seed 17",
            )
        if self.subset_seed != INPUT_ALIGNMENT_SUBSET_SEED:
            raise ValueError(
                "input alignment feasibility studies require fixed subset seed 17",
            )
        if self.continuation_rule != INPUT_ALIGNMENT_CONTINUATION_RULE:
            raise ValueError(
                f"input alignment feasibility studies require continuation_rule = {INPUT_ALIGNMENT_CONTINUATION_RULE!r}",
            )
        if "prospective" not in self.name:
            raise ValueError("input alignment feasibility study names must identify prospective evidence")

    def _validate_experiment(self, experiment: ExperimentSpec) -> None:
        training = experiment.training
        if not isinstance(training, InputTrainingSpec) or experiment.mode != "input_only":
            raise ValueError("input alignment feasibility studies require an input-only experiment")
        if experiment.stages != ("vocabulary",):
            raise ValueError("input alignment feasibility studies require vocabulary-only experiments")
        if training.profile != "large" or experiment.seed != self.subset_seed:
            raise ValueError(
                "input alignment feasibility studies require the Large profile and a base experiment seed matching subset_seed",
            )
        if training.reconstruction_epochs != 0 or training.reconstruction_samples != 0 or training.distillation_epochs != 0:
            raise ValueError(
                "input alignment feasibility studies prohibit reconstruction and distillation",
            )
        if experiment.evidence_scope != "candidate":
            raise ValueError(
                "input alignment feasibility studies require candidate experiment evidence",
            )

    def load_experiment(self, study_path: Path) -> ExperimentSpec:
        return ExperimentSpec.load((study_path.parent / self.experiment).resolve())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())


@final
@dataclass(frozen=True, slots=True)
class InputCompressionFeasibilityStudySpec:
    name: str
    experiment: str
    training_seeds: tuple[int, ...]
    vocabulary_subset_size: int
    subset_seed: int
    candidate_lengths: tuple[int, ...]
    binary_samples_per_length: int
    validation_bytes: int
    continuation_rule: str
    final_action: str
    prospective: bool
    study: str = "input_compression_feasibility"

    @classmethod
    def load(cls, path: Path) -> Self:
        table_name = "input compression feasibility study"
        values = _load_study_values(
            path,
            table_name=table_name,
            study="input_compression_feasibility",
            fields={
                "name",
                "experiment",
                "training_seeds",
                "vocabulary_subset_size",
                "subset_seed",
                "candidate_lengths",
                "binary_samples_per_length",
                "validation_bytes",
                "continuation_rule",
                "final_action",
                "prospective",
            },
        )
        prospective = values.get("prospective")
        if prospective is not True:
            raise ValueError("input compression feasibility studies must be prospective")
        spec = cls(
            name=non_empty_string(
                values,
                "name",
                table_name,
            ),
            experiment=non_empty_string(
                values,
                "experiment",
                table_name,
            ),
            training_seeds=_integer_array(
                values.get("training_seeds"),
                f"{table_name}.training_seeds",
            ),
            vocabulary_subset_size=positive_int(
                values.get("vocabulary_subset_size"),
                f"{table_name}.vocabulary_subset_size",
            ),
            subset_seed=positive_int(
                values.get("subset_seed"),
                f"{table_name}.subset_seed",
            ),
            candidate_lengths=_integer_array(
                values.get("candidate_lengths"),
                f"{table_name}.candidate_lengths",
            ),
            binary_samples_per_length=positive_int(
                values.get("binary_samples_per_length"),
                f"{table_name}.binary_samples_per_length",
            ),
            validation_bytes=positive_int(
                values.get("validation_bytes"),
                f"{table_name}.validation_bytes",
            ),
            continuation_rule=non_empty_string(
                values,
                "continuation_rule",
                table_name,
            ),
            final_action=non_empty_string(
                values,
                "final_action",
                table_name,
            ),
            prospective=prospective,
        )
        spec._validate(path)
        return spec

    def _validate(self, path: Path) -> None:
        if self.training_seeds != INPUT_COMPRESSION_TRAINING_SEEDS:
            raise ValueError(
                "input compression feasibility studies require training seed 17",
            )
        if self.vocabulary_subset_size != INPUT_COMPRESSION_VOCABULARY_ROWS:
            raise ValueError(
                "input compression feasibility studies require exactly 512 vocabulary rows",
            )
        if self.subset_seed != INPUT_COMPRESSION_SUBSET_SEED:
            raise ValueError(
                "input compression feasibility studies require fixed subset seed 17",
            )
        if self.candidate_lengths != INPUT_STUDY_CANDIDATE_LENGTHS:
            raise ValueError(
                "input compression feasibility studies require exact candidate lengths 2, 8, and 32",
            )
        if self.continuation_rule != INPUT_COMPRESSION_CONTINUATION_RULE:
            raise ValueError(
                "input compression feasibility studies require the fixed all-seed continuation rule",
            )
        if self.final_action != INPUT_COMPRESSION_FINAL_ACTION:
            raise ValueError(
                "input compression feasibility studies may only record freeze eligibility",
            )
        if "prospective" not in self.name:
            raise ValueError(
                "input compression feasibility study names must identify prospective evidence",
            )
        experiment = self.load_experiment(path)
        self._validate_experiment(experiment)
        training = cast(InputTrainingSpec, experiment.training)
        evaluation = cast(InputEvaluationSpec, experiment.evaluation)
        if training.validation_bytes != self.validation_bytes or evaluation.max_test_bytes != self.validation_bytes:
            raise ValueError(
                "input compression feasibility validation byte budgets must match",
            )
        if evaluation.generation_samples < 1:
            raise ValueError(
                "input compression feasibility studies require generation behavior samples",
            )

    @staticmethod
    def _validate_experiment(experiment: ExperimentSpec) -> None:
        training = experiment.training
        if not isinstance(training, InputTrainingSpec) or experiment.mode != "input_only":
            raise ValueError(
                "input compression feasibility studies require an input-only experiment",
            )
        if experiment.stages != (
            "vocabulary",
            "reconstruction",
            "frozen_backbone_distillation",
        ):
            raise ValueError(
                "input compression feasibility studies require vocabulary, reconstruction, and distillation stages",
            )
        if training.profile != "large":
            raise ValueError(
                "input compression feasibility studies require the Large profile",
            )
        if experiment.seed != INPUT_COMPRESSION_SUBSET_SEED:
            raise ValueError(
                "input compression feasibility base experiments require seed 17",
            )
        if training.reconstruction_epochs < 1 or training.reconstruction_samples < 1:
            raise ValueError(
                "input compression feasibility studies require explicit reconstruction work",
            )
        if training.distillation_epochs < 1:
            raise ValueError(
                "input compression feasibility studies require explicit candidate distillation work",
            )
        if experiment.model.evaluation != "full":
            raise ValueError(
                "input compression feasibility studies require full model evaluation",
            )
        if experiment.evidence_scope != "candidate":
            raise ValueError(
                "input compression feasibility studies require candidate experiment evidence",
            )

    def load_experiment(self, study_path: Path) -> ExperimentSpec:
        return ExperimentSpec.load((study_path.parent / self.experiment).resolve())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())
