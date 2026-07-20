from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import torch

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.backbone.config import input_table_is_removable
from continuous_tokenizer.codec.checkpoints import save_checkpoint
from continuous_tokenizer.codec.compilation import DYNAMIC_SEGMENTATION_MAX_BYTES
from continuous_tokenizer.contracts.experiment import ExperimentSpec, TrainingStage
from continuous_tokenizer.contracts.input import InputGateSpec, InputTrainingSpec
from continuous_tokenizer.contracts.profiles import PROFILES, Profile, profile_named
from continuous_tokenizer.data.corpus import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    joined_prefix,
    load_corpus_documents,
    sample_spans,
)
from continuous_tokenizer.input.alignment import (
    DEFAULT_EMBEDDING_FIT_TARGETS,
    EmbeddingFitTargets,
)
from continuous_tokenizer.input.training.reconstruction import ReconstructionFitter
from continuous_tokenizer.input.training.runtime import TrainingRuntime
from continuous_tokenizer.input.training.vocabulary import VocabularyFitter
from continuous_tokenizer.runtime.device import default_device
from continuous_tokenizer.training.optimizers import (
    DEFAULT_MUON_NS_STEPS,
    optimizer_metadata,
)

if TYPE_CHECKING:
    from continuous_tokenizer.runtime.resume import ResumeManager


@final
@dataclass(frozen=True, slots=True)
class TrainingOptions:
    output_dir: Path
    stages: tuple[TrainingStage, ...] = ("vocabulary", "reconstruction")
    profile: Profile = PROFILES[0]
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    muon_ns_steps: int = DEFAULT_MUON_NS_STEPS
    vocabulary_epochs: int = 100
    reconstruction_epochs: int = 10
    reconstruction_samples: int = 250_000
    reconstruction_vocabulary_fraction: float = 0.75
    validation_bytes: int = 4096
    patience: int = 5
    evaluation_interval: int = 5
    seed: int = 17
    dataset_id: str = DATASET_ID
    dataset_config: str = DATASET_CONFIG
    dataset_revision: str = DATASET_REVISION
    embedding_targets: EmbeddingFitTargets = DEFAULT_EMBEDDING_FIT_TARGETS
    minimum_native_tokens_per_continuous_token: float = 1.1
    maximum_candidate_reference_state_ratio: float = 0.5
    corpus_max_rows: int = 4096
    cache_chunk_rows: int = 64
    vocabulary_token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.reconstruction_vocabulary_fraction < 1.0:
            raise ValueError("reconstruction vocabulary fraction must be between zero and one")
        if self.muon_ns_steps < 1:
            raise ValueError("Muon Newton-Schulz steps must be positive")
        if self.vocabulary_token_ids is not None and (
            not self.vocabulary_token_ids
            or len(self.vocabulary_token_ids) != len(set(self.vocabulary_token_ids))
            or any(token_id < 0 for token_id in self.vocabulary_token_ids)
        ):
            raise ValueError("vocabulary training IDs must be unique non-negative rows")

    @property
    def reconstruction_enabled(self) -> bool:
        return "reconstruction" in self.stages and self.reconstruction_epochs > 0


def training_options_from_spec(spec: ExperimentSpec, output_dir: Path) -> TrainingOptions:
    training = spec.training
    gates = spec.gates
    if not isinstance(training, InputTrainingSpec) or not isinstance(gates, InputGateSpec):
        raise ValueError("input runner requires input-only training and gate settings")
    profile = profile_named(training.profile)
    if training.projection_multiplier:
        profile = replace(profile, projection_multiplier=training.projection_multiplier)
    return TrainingOptions(
        output_dir=output_dir,
        stages=spec.stages,
        profile=profile,
        batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        muon_ns_steps=training.muon_ns_steps,
        vocabulary_epochs=training.vocabulary_epochs,
        reconstruction_epochs=training.reconstruction_epochs,
        reconstruction_samples=training.reconstruction_samples,
        reconstruction_vocabulary_fraction=training.reconstruction_vocabulary_fraction,
        validation_bytes=training.validation_bytes,
        patience=training.patience,
        evaluation_interval=training.evaluation_interval,
        seed=spec.seed,
        dataset_id=spec.dataset.dataset_id,
        dataset_config=spec.dataset.config,
        dataset_revision=spec.dataset.revision,
        embedding_targets=EmbeddingFitTargets(
            gates.maximum_normalized_rmse,
            gates.minimum_cosine_p01,
            gates.minimum_cosine_p50,
        ),
        minimum_native_tokens_per_continuous_token=gates.minimum_native_tokens_per_continuous_token,
        maximum_candidate_reference_state_ratio=gates.maximum_candidate_reference_state_ratio,
        corpus_max_rows=spec.runtime.corpus_max_rows,
        cache_chunk_rows=spec.runtime.cache_chunk_rows,
    )


@final
@dataclass(frozen=True, slots=True)
class TrainingResult:
    profile: str
    optimizer: dict[str, str | int]
    checkpoint: str
    compatibility_checkpoint: str | None
    embedding_metrics: dict[str, int | float]
    compatibility_embedding_metrics: dict[str, int | float]
    candidate_state_bytes: int
    reference_state_bytes: int
    candidate_reference_state_ratio: float
    native_tokens_per_continuous_token: float
    round_trip: bool
    compatibility_passed: bool
    alignment_preserved: bool
    passed: bool
    cache_metrics: dict[str, dict[str, Any]]


def _load_data(options: TrainingOptions) -> tuple[list[bytes], bytes]:
    dataset = {
        "dataset_id": options.dataset_id,
        "config": options.dataset_config,
        "revision": options.dataset_revision,
    }
    train_documents = (
        load_corpus_documents(
            "train",
            **dataset,
            max_rows=options.corpus_max_rows,
        )
        if options.reconstruction_enabled
        else []
    )
    corpus_spans = (
        sample_spans(
            train_documents,
            count=options.reconstruction_samples,
            seed=options.seed,
            maximum=DYNAMIC_SEGMENTATION_MAX_BYTES,
        )
        if options.reconstruction_enabled
        else []
    )
    validation_data = joined_prefix(
        load_corpus_documents(
            "validation",
            **dataset,
            max_rows=options.corpus_max_rows,
        ),
        max_bytes=options.validation_bytes,
    )
    return corpus_spans, validation_data


def _train_tokenizer(
    runtime: TrainingRuntime,
    corpus_spans: list[bytes],
    validation_data: bytes,
    generator: torch.Generator,
    randomizer: random.Random,
) -> TrainingResult:
    options = runtime.options
    profile = options.profile
    codec = runtime.build_codec(profile)
    vocabulary_cache_metrics = VocabularyFitter(runtime).fit(codec, profile, generator)
    deployment_codec = runtime.deployment_evaluator(codec)
    compatibility_metrics = runtime.evaluate_deployment(
        codec,
        deployment_codec,
        token_ids=options.vocabulary_token_ids,
    )
    control_ids = torch.tensor(runtime.assets.vocabulary.control_ids, dtype=torch.long)
    control_embeddings = runtime.assets.input_embeddings[control_ids]
    candidate_state_bytes, reference_state_bytes = runtime.candidate_reference_state_bytes(deployment_codec)
    candidate_reference_state_ratio = candidate_state_bytes / reference_state_bytes
    compatibility_passed = options.embedding_targets.accepts(compatibility_metrics)
    compatibility_path: Path | None = None
    if compatibility_passed:
        compatibility_path = options.output_dir / f"{profile.name}-compatibility.pt"
        save_checkpoint(
            compatibility_path,
            codec,
            runtime.checkpoint_metadata(profile, checkpoint_stage="vocabulary"),
            control_ids=control_ids,
            control_embeddings=control_embeddings,
        )

    reconstruction_cache_metrics: dict[str, Any] = {}
    if options.reconstruction_enabled:
        reconstruction = ReconstructionFitter(runtime).fit(
            codec,
            profile,
            corpus_spans,
            validation_data,
            randomizer,
        )
        reconstruction_cache_metrics = reconstruction.cache_metrics
        metrics = reconstruction.embedding_metrics
        native_tokens_per_continuous_token = reconstruction.native_tokens_per_continuous_token
        round_trip = reconstruction.round_trip
        measurement_seconds = 0.0
        if reconstruction.density_identity != runtime.density_identity(deployment_codec, validation_data):
            raise RuntimeError("selected reconstruction density identity does not match the final checkpoint")
    else:
        metrics = compatibility_metrics
        (
            (
                native_tokens_per_continuous_token,
                round_trip,
            ),
            measurement_seconds,
        ) = runtime.timed(lambda: runtime.density_metrics(deployment_codec, validation_data))
    runtime.write_json(
        f"progress/{profile.name}-density.json",
        {
            "profile": profile.name,
            "native_tokens_per_continuous_token": native_tokens_per_continuous_token,
            "round_trip": round_trip,
            "seconds": measurement_seconds,
            "reused_selected_reconstruction_measurement": (options.reconstruction_enabled),
            "identity": runtime.density_identity(
                deployment_codec,
                validation_data,
            ),
        },
    )
    checkpoint_path = options.output_dir / f"{profile.name}.pt"
    save_checkpoint(
        checkpoint_path,
        codec,
        runtime.checkpoint_metadata(
            profile,
            checkpoint_stage="reconstruction" if options.reconstruction_enabled else "vocabulary",
        ),
        control_ids=control_ids,
        control_embeddings=control_embeddings,
    )
    alignment_preserved = options.embedding_targets.accepts(metrics)
    compactness_passed = candidate_reference_state_ratio <= options.maximum_candidate_reference_state_ratio
    passed = (
        compatibility_passed
        and alignment_preserved
        and round_trip
        and native_tokens_per_continuous_token >= options.minimum_native_tokens_per_continuous_token
        and (compactness_passed or not input_table_is_removable(runtime.assets.config))
    )
    return TrainingResult(
        profile=profile.name,
        optimizer=optimizer_metadata(options.muon_ns_steps),
        checkpoint=str(checkpoint_path),
        compatibility_checkpoint=(None if compatibility_path is None else str(compatibility_path)),
        embedding_metrics=metrics.to_dict(),
        compatibility_embedding_metrics=compatibility_metrics.to_dict(),
        candidate_state_bytes=candidate_state_bytes,
        reference_state_bytes=reference_state_bytes,
        candidate_reference_state_ratio=candidate_reference_state_ratio,
        native_tokens_per_continuous_token=native_tokens_per_continuous_token,
        round_trip=round_trip,
        compatibility_passed=compatibility_passed,
        alignment_preserved=alignment_preserved,
        passed=passed,
        cache_metrics={
            "vocabulary": vocabulary_cache_metrics,
            "reconstruction": reconstruction_cache_metrics,
        },
    )


def train_experiment(
    assets: ModelAssets,
    options: TrainingOptions,
    *,
    device: torch.device | None = None,
    resume_manager: ResumeManager | None = None,
    epoch_boundary: Callable[[str, int], None] | None = None,
) -> TrainingResult:
    selected_device = default_device() if device is None else device
    runtime = TrainingRuntime(
        assets,
        options,
        selected_device,
        resume_manager,
        epoch_boundary,
    )
    torch.manual_seed(options.seed)
    generator = torch.Generator().manual_seed(options.seed)
    randomizer = random.Random(options.seed)
    corpus_spans, validation_data = _load_data(options)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    if resume_manager is None or not resume_manager.resuming:
        runtime.write_json("run-manifest.json", runtime.run_manifest())
    result = _train_tokenizer(
        runtime,
        corpus_spans,
        validation_data,
        generator,
        randomizer,
    )
    runtime.write_json("training-result.json", asdict(result))
    return result
