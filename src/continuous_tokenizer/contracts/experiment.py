from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self, cast, final

from continuous_tokenizer.contracts.input import (
    InputEvaluationSpec,
    InputGateSpec,
    InputTrainingSpec,
)
from continuous_tokenizer.contracts.output import (
    OutputEvaluationSpec,
    OutputGateSpec,
    OutputTrainingSpec,
)
from continuous_tokenizer.contracts.parsing import (
    exact_fields,
    mapping_fingerprint,
    non_empty_string,
    non_negative_int,
    positive_int,
    scalar_mapping,
    sha256_string,
    strict_fields,
    table,
)
from continuous_tokenizer.contracts.profiles import (
    CAMPAIGN_PROFILE_NAME,
    EFFICIENCY_BATCH_SIZES,
    EFFICIENCY_MUON_NS_STEPS,
    profile_named,
)
from continuous_tokenizer.contracts.prospective_selection import (
    ProspectiveSelectionSpec,
)

INPUT_TRAINING_STAGES: Final = (
    "vocabulary",
    "reconstruction",
    "frozen_backbone_distillation",
)
OUTPUT_TRAINING_STAGES: Final = ("output_codec",)
TRAINING_STAGES: Final = (*INPUT_TRAINING_STAGES, *OUTPUT_TRAINING_STAGES)

type TrainingStage = Literal[
    "vocabulary",
    "reconstruction",
    "frozen_backbone_distillation",
    "output_codec",
]
type TokenizerMode = Literal["input_only", "output_only"]
type EvaluationCapability = Literal["full"]
type DeviceType = Literal["cpu", "mps", "cuda"]
type EvidenceScope = Literal["candidate", "diagnostic", "final", "search", "synthetic"]
type SearchKind = Literal["alignment", "efficiency", "output"]
type StudyKind = Literal["input_selection", "output_oracle"]


@final
@dataclass(frozen=True, slots=True)
class RuntimePolicySpec:
    corpus_max_rows: int = 512
    cache_chunk_rows: int = 32
    snapshot_interval: int = 100
    projected_run_bytes: int = 16 * 1024**3
    storage_reserve_bytes: int = 10 * 1024**3
    inductor_cache_estimate_bytes: int = 8 * 1024**3
    minimum_mps_memory_bytes: int = 32 * 1024**3

    @classmethod
    def parse(cls, value: object) -> Self:
        values = table(value, "runtime")
        expected = {
            "corpus_max_rows",
            "cache_chunk_rows",
            "snapshot_interval",
            "projected_run_bytes",
            "storage_reserve_bytes",
            "inductor_cache_estimate_bytes",
            "minimum_mps_memory_bytes",
        }
        exact_fields(values, expected, "runtime")
        return cls(
            corpus_max_rows=positive_int(values.get("corpus_max_rows"), "runtime.corpus_max_rows"),
            cache_chunk_rows=positive_int(values.get("cache_chunk_rows"), "runtime.cache_chunk_rows"),
            snapshot_interval=positive_int(values.get("snapshot_interval"), "runtime.snapshot_interval"),
            projected_run_bytes=positive_int(values.get("projected_run_bytes"), "runtime.projected_run_bytes"),
            storage_reserve_bytes=positive_int(values.get("storage_reserve_bytes"), "runtime.storage_reserve_bytes"),
            inductor_cache_estimate_bytes=positive_int(
                values.get("inductor_cache_estimate_bytes"),
                "runtime.inductor_cache_estimate_bytes",
            ),
            minimum_mps_memory_bytes=positive_int(
                values.get("minimum_mps_memory_bytes"),
                "runtime.minimum_mps_memory_bytes",
            ),
        )


@final
@dataclass(frozen=True, slots=True)
class SearchSelectionSpec:
    search_kind: SearchKind
    artifact: str
    artifact_sha256: str
    selected_trial: int
    search_fingerprint: str
    model_id: str
    model_revision: str
    profile: str
    selected_parameters: Mapping[str, int | float | str]
    feasible: bool

    @classmethod
    def parse(cls, value: object, base_directory: Path) -> Self:
        values = table(value, "search selection")
        exact_fields(
            values,
            {
                "search_kind",
                "artifact",
                "artifact_sha256",
                "selected_trial",
                "search_fingerprint",
                "model_id",
                "model_revision",
                "profile",
                "selected_parameters",
                "feasible",
            },
            "search selection",
        )
        search_kind = non_empty_string(values, "search_kind", "search selection")
        if search_kind not in {"alignment", "efficiency", "output"}:
            raise ValueError("search selection.search_kind is invalid")
        selected_trial = non_negative_int(
            values.get("selected_trial"),
            "search selection.selected_trial",
        )
        feasible = values.get("feasible")
        if not isinstance(feasible, bool):
            raise ValueError("search selection.feasible must be boolean")
        return cls(
            search_kind=search_kind,
            artifact=str((base_directory / non_empty_string(values, "artifact", "search selection")).resolve()),
            artifact_sha256=sha256_string(
                values,
                "artifact_sha256",
                "search selection",
            ),
            selected_trial=selected_trial,
            search_fingerprint=sha256_string(
                values,
                "search_fingerprint",
                "search selection",
            ),
            model_id=non_empty_string(values, "model_id", "search selection"),
            model_revision=non_empty_string(values, "model_revision", "search selection"),
            profile=non_empty_string(values, "profile", "search selection"),
            selected_parameters=scalar_mapping(
                values,
                "selected_parameters",
                "search selection",
            ),
            feasible=feasible,
        )


@final
@dataclass(frozen=True, slots=True)
class StudySelectionSpec:
    study_kind: StudyKind
    artifact: str
    artifact_sha256: str
    study_fingerprint: str
    model_id: str
    model_revision: str
    selected_parameters: Mapping[str, int | float | str]
    feasible: bool

    @classmethod
    def parse(cls, value: object, base_directory: Path) -> Self:
        values = table(value, "study selection")
        expected = {
            "study_kind",
            "artifact",
            "artifact_sha256",
            "study_fingerprint",
            "model_id",
            "model_revision",
            "selected_parameters",
            "feasible",
        }
        exact_fields(
            values,
            expected,
            "study selection",
            incomplete_message="study selection fields are incomplete",
        )
        study_kind = non_empty_string(values, "study_kind", "study selection")
        if study_kind not in {"input_selection", "output_oracle"}:
            raise ValueError("study selection.study_kind is invalid")
        feasible = values.get("feasible")
        if not isinstance(feasible, bool):
            raise ValueError("study selection.feasible must be boolean")
        return cls(
            study_kind=study_kind,
            artifact=str((base_directory / non_empty_string(values, "artifact", "study selection")).resolve()),
            artifact_sha256=sha256_string(
                values,
                "artifact_sha256",
                "study selection",
            ),
            study_fingerprint=sha256_string(
                values,
                "study_fingerprint",
                "study selection",
            ),
            model_id=non_empty_string(values, "model_id", "study selection"),
            model_revision=non_empty_string(values, "model_revision", "study selection"),
            selected_parameters=scalar_mapping(
                values,
                "selected_parameters",
                "study selection",
            ),
            feasible=feasible,
        )


@final
@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    revision: str
    evaluation: EvaluationCapability

    @classmethod
    def parse(cls, value: object) -> Self:
        values = table(value, "model")
        exact_fields(values, {"id", "revision", "evaluation"}, "model")
        evaluation = non_empty_string(values, "evaluation", "model")
        if evaluation != "full":
            raise ValueError("model.evaluation must be 'full'")
        return cls(
            non_empty_string(values, "id", "model"),
            non_empty_string(values, "revision", "model"),
            evaluation,
        )


@final
@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: str
    config: str
    revision: str

    @classmethod
    def parse(cls, value: object) -> Self:
        values = table(value, "dataset")
        exact_fields(values, {"id", "config", "revision"}, "dataset")
        return cls(
            non_empty_string(values, "id", "dataset"),
            non_empty_string(values, "config", "dataset"),
            non_empty_string(values, "revision", "dataset"),
        )


@final
@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    mode: TokenizerMode
    device: DeviceType
    model: ModelSpec
    dataset: DatasetSpec
    stages: tuple[TrainingStage, ...]
    seed: int
    training: InputTrainingSpec | OutputTrainingSpec
    evaluation: InputEvaluationSpec | OutputEvaluationSpec
    gates: InputGateSpec | OutputGateSpec
    runtime: RuntimePolicySpec = RuntimePolicySpec()
    evidence_scope: EvidenceScope = "candidate"
    prospective_selection: ProspectiveSelectionSpec | None = None
    search_selections: tuple[SearchSelectionSpec, ...] = ()
    study_selections: tuple[StudySelectionSpec, ...] = ()
    efficiency_pilot: str | None = None
    efficiency_pilot_sha256: str | None = None

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        strict_fields(
            values,
            {
                "name",
                "mode",
                "device",
                "model",
                "dataset",
                "stages",
                "seed",
                "training",
                "evaluation",
                "gates",
                "runtime",
                "evidence_scope",
                "prospective_selection",
                "search_selections",
                "study_selections",
                "efficiency_pilot",
                "efficiency_pilot_sha256",
            },
            "experiment",
        )
        raw_mode = values.get("mode")
        if raw_mode not in {"input_only", "output_only"}:
            raise ValueError("mode must be 'input_only' or 'output_only'")
        mode = cast(TokenizerMode, raw_mode)
        allowed_stages = INPUT_TRAINING_STAGES if mode == "input_only" else OUTPUT_TRAINING_STAGES
        raw_stages = values.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages or any(not isinstance(stage, str) for stage in raw_stages):
            raise ValueError("stages must be a non-empty TOML array")
        stages = cast(tuple[TrainingStage, ...], tuple(str(stage) for stage in raw_stages))
        if len(stages) != len(set(stages)) or any(stage not in allowed_stages for stage in stages):
            raise ValueError(f"stages must be unique values from {allowed_stages}")
        expected_order = tuple(stage for stage in allowed_stages if stage in stages)
        required_first = "vocabulary" if mode == "input_only" else "output_codec"
        if stages != expected_order or stages[0] != required_first:
            raise ValueError(f"stages must follow this order: {allowed_stages}")
        seed_value = values.get("seed")
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            raise ValueError("seed must be an integer")
        if seed_value < 0:
            raise ValueError("seed must be non-negative")
        raw_scope = non_empty_string(values, "evidence_scope", "experiment")
        if raw_scope not in {"candidate", "diagnostic", "final", "search", "synthetic"}:
            raise ValueError("experiment.evidence_scope is invalid")
        evidence_scope = raw_scope
        raw_selections = values.get("search_selections", [])
        if not isinstance(raw_selections, list):
            raise ValueError("experiment.search_selections must be an array of tables")
        search_selections = tuple(SearchSelectionSpec.parse(item, path.parent) for item in raw_selections)
        raw_studies = values.get("study_selections", [])
        if not isinstance(raw_studies, list):
            raise ValueError("experiment.study_selections must be an array of tables")
        study_selections = tuple(StudySelectionSpec.parse(item, path.parent) for item in raw_studies)
        runtime = RuntimePolicySpec.parse(values.get("runtime"))
        spec = cls(
            name=non_empty_string(values, "name", "experiment"),
            mode=mode,
            device=cast(DeviceType, non_empty_string(values, "device", "experiment")),
            model=ModelSpec.parse(values["model"]),
            dataset=DatasetSpec.parse(values["dataset"]),
            stages=stages,
            seed=seed_value,
            training=(InputTrainingSpec.parse(values["training"]) if mode == "input_only" else OutputTrainingSpec.parse(values["training"])),
            evaluation=(InputEvaluationSpec.parse(values["evaluation"]) if mode == "input_only" else OutputEvaluationSpec.parse(values["evaluation"])),
            gates=(InputGateSpec.parse(values["gates"]) if mode == "input_only" else OutputGateSpec.parse(values["gates"])),
            runtime=runtime,
            evidence_scope=evidence_scope,
            prospective_selection=(
                None
                if values.get("prospective_selection") is None
                else ProspectiveSelectionSpec.parse(
                    values["prospective_selection"],
                    path.parent,
                )
            ),
            search_selections=search_selections,
            study_selections=study_selections,
            efficiency_pilot=(
                None if values.get("efficiency_pilot") is None else str((path.parent / non_empty_string(values, "efficiency_pilot", "experiment")).resolve())
            ),
            efficiency_pilot_sha256=(
                None
                if values.get("efficiency_pilot_sha256") is None
                else non_empty_string(
                    values,
                    "efficiency_pilot_sha256",
                    "experiment",
                )
            ),
        )
        spec._validate_stage_settings()
        return spec

    def _validate_stage_settings(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be 'cpu', 'mps', or 'cuda'")
        self._validate_search_selections()
        self._validate_study_selections()
        self._validate_prospective_selection()
        if self.mode == "output_only":
            self._validate_output_settings()
            return
        self._validate_input_settings()

    def _validate_prospective_selection(self) -> None:
        selection = self.prospective_selection
        if selection is None:
            return
        if self.evidence_scope != "final":
            raise ValueError("only final experiments may declare prospective selection provenance")
        if self.search_selections or self.study_selections or self.efficiency_pilot is not None:
            raise ValueError("prospective final experiments cannot declare legacy selection provenance")
        expected_identity = (
            self.model.model_id,
            self.model.revision,
            self.dataset.dataset_id,
            self.dataset.config,
            self.dataset.revision,
            self.training.profile,
        )
        actual_identity = (
            selection.model_id,
            selection.model_revision,
            selection.dataset_id,
            selection.dataset_config,
            selection.dataset_revision,
            selection.profile,
        )
        if actual_identity != expected_identity:
            raise ValueError("prospective selection model, data, or profile differs from the experiment")
        for name, selected_value in selection.selected_parameters.items():
            if not hasattr(self.training, name) or getattr(self.training, name) != selected_value:
                raise ValueError(f"experiment training.{name} differs from its prospective selection")
        if isinstance(self.training, InputTrainingSpec) and self.training.strategy != selection.selected_strategy:
            raise ValueError("experiment training.strategy differs from its prospective selection")

    def _validate_output_settings(self) -> None:
        if not isinstance(self.training, OutputTrainingSpec):
            raise ValueError("output-only experiment requires output training settings")
        if not isinstance(self.evaluation, OutputEvaluationSpec):
            raise ValueError("output-only experiment requires output evaluation settings")
        if self.training.max_span not in self.evaluation.oracle_span_limits:
            raise ValueError("output training.max_span must be one of the registered oracle span limits")

    def _validate_input_settings(self) -> None:
        if not isinstance(self.training, InputTrainingSpec):
            raise ValueError("input-only experiment requires input training settings")
        if self.study_selections and self.training.strategy == "candidate_selection":
            raise ValueError("final input experiments require a frozen training.strategy")
        if self.evidence_scope != "final" and self.training.strategy != "candidate_selection":
            raise ValueError("non-final input experiments must retain candidate_selection strategy")
        self._validate_efficiency_settings()
        self._validate_reconstruction_settings()
        self._validate_distillation_settings()

    def _validate_efficiency_settings(self) -> None:
        if not isinstance(self.training, InputTrainingSpec):
            raise TypeError("input efficiency settings require input training")
        profile = profile_named(self.training.profile)
        efficiency_changed = profile.name == CAMPAIGN_PROFILE_NAME and (
            self.training.batch_size >= min(EFFICIENCY_BATCH_SIZES)
            or self.training.projection_multiplier not in {0, profile.projection_multiplier}
            or self.training.muon_ns_steps != max(EFFICIENCY_MUON_NS_STEPS)
        )
        if efficiency_changed and self.prospective_selection is None and (self.efficiency_pilot is None or self.efficiency_pilot_sha256 is None):
            raise ValueError("optimized input experiments require a pinned efficiency pilot")
        if (self.efficiency_pilot is None) != (self.efficiency_pilot_sha256 is None):
            raise ValueError("efficiency pilot path and hash must be declared together")

    def _validate_reconstruction_settings(self) -> None:
        if not isinstance(self.training, InputTrainingSpec):
            raise TypeError("input reconstruction settings require input training")
        if self.training.vocabulary_epochs == 0:
            raise ValueError("vocabulary stage requires training.vocabulary_epochs")
        if "reconstruction" in self.stages and (self.training.reconstruction_epochs == 0 or self.training.reconstruction_samples == 0):
            raise ValueError("reconstruction stage requires epochs and samples")
        if "reconstruction" not in self.stages and (self.training.reconstruction_epochs != 0 or self.training.reconstruction_samples != 0):
            raise ValueError("unused reconstruction settings must be zero")

    def _validate_distillation_settings(self) -> None:
        if not isinstance(self.training, InputTrainingSpec):
            raise TypeError("input distillation settings require input training")
        if "frozen_backbone_distillation" in self.stages:
            if self.model.evaluation != "full":
                raise ValueError("frozen-backbone distillation requires full model evaluation")
            if self.training.distillation_epochs == 0:
                raise ValueError("distillation stage requires training.distillation_epochs")
        elif self.training.distillation_epochs != 0:
            raise ValueError("unused distillation epochs must be zero")

    def _validate_search_selections(self) -> None:
        kinds = tuple(selection.search_kind for selection in self.search_selections)
        if len(kinds) != len(set(kinds)):
            raise ValueError("experiment search selections must have unique search kinds")
        if self.evidence_scope == "final" and self.prospective_selection is None and not self.search_selections:
            raise ValueError("final experiments require search-selection provenance")
        if self.evidence_scope != "final" and self.search_selections:
            raise ValueError("only final experiments may declare search-selection provenance")
        for selection in self.search_selections:
            self._validate_search_selection(selection)
        effective_parameters: dict[str, int | float | str] = {}
        for selection in self.search_selections:
            effective_parameters.update(selection.selected_parameters)
        for name, selected_value in effective_parameters.items():
            if not hasattr(self.training, name) or getattr(self.training, name) != selected_value:
                raise ValueError(f"experiment training.{name} differs from its effective search selection")

    def _validate_search_selection(self, selection: SearchSelectionSpec) -> None:
        if selection.model_id != self.model.model_id or selection.model_revision != self.model.revision:
            raise ValueError("search selection model does not match the experiment")
        if selection.profile != self.training.profile:
            raise ValueError("search selection profile does not match the experiment")
        if self.mode == "input_only" and selection.search_kind == "output":
            raise ValueError("input-only experiments cannot use output search provenance")
        if self.mode == "output_only" and selection.search_kind != "output":
            raise ValueError("output-only experiments require output search provenance")

    def _validate_study_selections(self) -> None:
        kinds = tuple(selection.study_kind for selection in self.study_selections)
        if len(kinds) != len(set(kinds)):
            raise ValueError("experiment study selections must have unique study kinds")
        if self.evidence_scope != "final" and self.study_selections:
            raise ValueError("only final experiments may declare study-selection provenance")
        expected_kind: StudyKind = "input_selection" if self.mode == "input_only" else "output_oracle"
        if self.study_selections and set(kinds) != {expected_kind}:
            raise ValueError(f"{self.mode} experiments require {expected_kind} study provenance")
        for selection in self.study_selections:
            if selection.model_id != self.model.model_id or selection.model_revision != self.model.revision:
                raise ValueError("study selection model does not match the experiment")
            if selection.study_kind != expected_kind:
                raise ValueError(f"{self.mode} experiments require {expected_kind} study provenance")
            for name, selected_value in selection.selected_parameters.items():
                if not hasattr(self.training, name) or getattr(self.training, name) != selected_value:
                    raise ValueError(f"experiment training.{name} differs from its study selection")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_toml_dict(self) -> dict[str, Any]:
        values = self.to_dict()
        if values["prospective_selection"] is None:
            del values["prospective_selection"]
        if values["efficiency_pilot"] is None:
            del values["efficiency_pilot"]
            del values["efficiency_pilot_sha256"]
        model = cast(dict[str, Any], values["model"])
        model["id"] = model.pop("model_id")
        dataset = cast(dict[str, Any], values["dataset"])
        dataset["id"] = dataset.pop("dataset_id")
        return values

    def fingerprint(self) -> str:
        return mapping_fingerprint(self.to_dict())

    def replication_fingerprint(self) -> str:
        values = self.to_dict()
        values["name"] = None
        values["seed"] = None
        return mapping_fingerprint(values)
