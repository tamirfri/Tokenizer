from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Self, final

from continuous_tokenizer.contracts.parsing import (
    exact_fields,
    float_list,
    mapping_fingerprint,
    non_empty_string,
    non_negative_int,
    positive_int,
    table,
)
from continuous_tokenizer.contracts.profiles import (
    EFFICIENCY_BATCH_SIZES,
    EFFICIENCY_MUON_NS_STEPS,
    EFFICIENCY_PROJECTION_MULTIPLIERS,
)

MAXIMUM_SEARCH_TRIALS: Final = 2
MAXIMUM_SEARCH_VOCABULARY_ROWS: Final = 512
MAXIMUM_SEARCH_EPOCHS: Final = 3
MAXIMUM_OUTPUT_PILOT_DOCUMENTS: Final = 4


@final
@dataclass(frozen=True, slots=True)
class EfficiencySearchSpace:
    learning_rate_min: float
    learning_rate_max: float
    batch_sizes: tuple[int, ...]
    projection_multipliers: tuple[int, ...]
    muon_ns_steps: tuple[int, ...]

    @classmethod
    def parse(cls, value: object) -> Self:
        values = table(value, "efficiency space")
        exact_fields(
            values,
            {
                "learning_rate_min",
                "learning_rate_max",
                "batch_sizes",
                "projection_multipliers",
                "muon_ns_steps",
            },
            "efficiency space",
        )
        minimum = float(values.get("learning_rate_min", 0))
        maximum = float(values.get("learning_rate_max", 0))
        if minimum <= 0 or maximum < minimum:
            raise ValueError("efficiency learning-rate bounds must be positive and ordered")

        def choices(name: str, allowed: set[int]) -> tuple[int, ...]:
            raw = values.get(name)
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"efficiency space.{name} must be a non-empty array")
            parsed = tuple(item for item in raw if isinstance(item, int) and not isinstance(item, bool))
            if len(parsed) != len(raw) or len(parsed) != len(set(parsed)) or not set(parsed).issubset(allowed):
                raise ValueError(f"efficiency space.{name} contains invalid choices")
            return parsed

        return cls(
            minimum,
            maximum,
            choices("batch_sizes", set(EFFICIENCY_BATCH_SIZES)),
            choices("projection_multipliers", set(EFFICIENCY_PROJECTION_MULTIPLIERS)),
            choices("muon_ns_steps", set(EFFICIENCY_MUON_NS_STEPS)),
        )


@final
@dataclass(frozen=True, slots=True)
class EfficiencyPilotSpec:
    name: str
    experiment: str
    final_experiment: str
    trials: int
    sampler_seed: int
    vocabulary_rows: int
    vocabulary_epochs: int
    patience: int
    evaluation_interval: int
    minimum_runtime_improvement: float
    space: EfficiencySearchSpace

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        exact_fields(
            values,
            {
                "mode",
                "study",
                "name",
                "experiment",
                "final_experiment",
                "trials",
                "sampler_seed",
                "vocabulary_rows",
                "vocabulary_epochs",
                "patience",
                "evaluation_interval",
                "minimum_runtime_improvement",
                "space",
            },
            "efficiency pilot",
        )
        if values.get("mode") != "input_only" or values.get("study") != "efficiency":
            raise ValueError("invalid efficiency pilot contract")
        name = non_empty_string(values, "name", "efficiency pilot")
        experiment = non_empty_string(values, "experiment", "efficiency pilot")
        improvement = float(values.get("minimum_runtime_improvement", 0))
        if not 0.05 <= improvement < 1:
            raise ValueError("efficiency pilot runtime improvement must be at least 5%")
        spec = cls(
            name=name,
            experiment=experiment,
            final_experiment=non_empty_string(values, "final_experiment", "efficiency pilot"),
            trials=positive_int(values.get("trials"), "efficiency pilot.trials"),
            sampler_seed=non_negative_int(
                values.get("sampler_seed"),
                "efficiency pilot.sampler_seed",
            ),
            vocabulary_rows=positive_int(
                values.get("vocabulary_rows"),
                "efficiency pilot.vocabulary_rows",
            ),
            vocabulary_epochs=positive_int(
                values.get("vocabulary_epochs"),
                "efficiency pilot.vocabulary_epochs",
            ),
            patience=positive_int(values.get("patience"), "efficiency pilot.patience"),
            evaluation_interval=positive_int(
                values.get("evaluation_interval"),
                "efficiency pilot.evaluation_interval",
            ),
            minimum_runtime_improvement=improvement,
            space=EfficiencySearchSpace.parse(values.get("space")),
        )
        if spec.trials > MAXIMUM_SEARCH_TRIALS or spec.vocabulary_rows > MAXIMUM_SEARCH_VOCABULARY_ROWS or spec.vocabulary_epochs > MAXIMUM_SEARCH_EPOCHS:
            raise ValueError("efficiency pilot exceeds the current search budget")
        return spec

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update(
            mode="input_only",
            study="efficiency",
        )
        return values

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())


@final
@dataclass(frozen=True, slots=True)
class SearchSpace:
    learning_rate_min: float
    learning_rate_max: float
    weight_decays: tuple[float, ...]
    batch_sizes: tuple[int, ...]

    @classmethod
    def parse(cls, value: object) -> Self:
        values = table(value, "space")
        exact_fields(
            values,
            {"learning_rate_min", "learning_rate_max", "weight_decays", "batch_sizes"},
            "space",
        )
        minimum = float(values.get("learning_rate_min", 0))
        maximum = float(values.get("learning_rate_max", 0))
        if minimum <= 0 or maximum < minimum:
            raise ValueError("space learning-rate bounds must be positive and ordered")
        raw_batch_sizes = values.get("batch_sizes")
        if not isinstance(raw_batch_sizes, list) or not raw_batch_sizes:
            raise ValueError("space.batch_sizes must be a non-empty array")
        batch_sizes = tuple(positive_int(item, "space.batch_sizes item") for item in raw_batch_sizes)
        if len(batch_sizes) != len(set(batch_sizes)):
            raise ValueError("space.batch_sizes must contain unique values")
        return cls(
            minimum,
            maximum,
            float_list(values.get("weight_decays"), "space.weight_decays"),
            batch_sizes,
        )


@final
@dataclass(frozen=True, slots=True)
class SearchSpec:
    name: str
    experiment: str
    final_experiment: str
    trials: int
    sampler_seed: int
    vocabulary_rows: int
    vocabulary_epochs: int
    patience: int
    evaluation_interval: int
    space: SearchSpace

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        exact_fields(
            values,
            {
                "name",
                "experiment",
                "final_experiment",
                "trials",
                "sampler_seed",
                "vocabulary_rows",
                "vocabulary_epochs",
                "patience",
                "evaluation_interval",
                "space",
            },
            "search",
        )
        spec = cls(
            name=non_empty_string(values, "name", "search"),
            experiment=non_empty_string(values, "experiment", "search"),
            final_experiment=non_empty_string(values, "final_experiment", "search"),
            trials=positive_int(values.get("trials"), "search.trials"),
            sampler_seed=non_negative_int(
                values.get("sampler_seed"),
                "search.sampler_seed",
            ),
            vocabulary_rows=positive_int(values.get("vocabulary_rows"), "search.vocabulary_rows"),
            vocabulary_epochs=positive_int(
                values.get("vocabulary_epochs"),
                "search.vocabulary_epochs",
            ),
            patience=positive_int(values.get("patience"), "search.patience"),
            evaluation_interval=positive_int(
                values.get("evaluation_interval"),
                "search.evaluation_interval",
            ),
            space=SearchSpace.parse(values.get("space")),
        )
        if spec.trials > MAXIMUM_SEARCH_TRIALS or spec.vocabulary_rows > MAXIMUM_SEARCH_VOCABULARY_ROWS or spec.vocabulary_epochs > MAXIMUM_SEARCH_EPOCHS:
            raise ValueError("search exceeds the current trial, row, or epoch budget")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())


@final
@dataclass(frozen=True, slots=True)
class OutputSearchSpec:
    name: str
    experiment: str
    final_experiment: str
    trials: int
    sampler_seed: int
    fallback_rule: str
    pilot_documents: int
    learning_rate_min: float
    learning_rate_max: float
    weight_decays: tuple[float, ...]
    batch_sizes: tuple[int, ...]

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        expected = {
            "mode",
            "name",
            "experiment",
            "final_experiment",
            "trials",
            "sampler_seed",
            "fallback_rule",
            "pilot_documents",
            "space",
        }
        exact_fields(values, expected, "output search")
        space = table(values.get("space"), "output search space")
        exact_fields(
            space,
            {
                "learning_rate_min",
                "learning_rate_max",
                "weight_decays",
                "batch_sizes",
            },
            "output search space",
        )
        if values.get("mode") != "output_only":
            raise ValueError("invalid output search contract")
        batch_sizes = space.get("batch_sizes")
        if not isinstance(batch_sizes, list) or not batch_sizes:
            raise ValueError("output search batch_sizes must be a non-empty array")
        spec = cls(
            name=non_empty_string(values, "name", "output search"),
            experiment=non_empty_string(values, "experiment", "output search"),
            final_experiment=non_empty_string(values, "final_experiment", "output search"),
            trials=positive_int(values.get("trials"), "output search trials"),
            sampler_seed=non_negative_int(
                values.get("sampler_seed"),
                "output search.sampler_seed",
            ),
            fallback_rule=non_empty_string(values, "fallback_rule", "output search"),
            pilot_documents=positive_int(
                values.get("pilot_documents"),
                "output search pilot_documents",
            ),
            learning_rate_min=float(space["learning_rate_min"]),
            learning_rate_max=float(space["learning_rate_max"]),
            weight_decays=float_list(
                space.get("weight_decays"),
                "output search weight_decays",
            ),
            batch_sizes=tuple(positive_int(value, "output search batch size") for value in batch_sizes),
        )
        if not 0 < spec.learning_rate_min <= spec.learning_rate_max:
            raise ValueError("output search learning-rate bounds are invalid")
        if spec.fallback_rule != "best_exact_full_sequence_rate":
            raise ValueError(
                "output search fallback_rule must be 'best_exact_full_sequence_rate'",
            )
        if spec.trials > MAXIMUM_SEARCH_TRIALS or spec.pilot_documents > MAXIMUM_OUTPUT_PILOT_DOCUMENTS:
            raise ValueError("output search exceeds the current trial or corpus budget")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "output_only",
            **asdict(self),
        }

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())
