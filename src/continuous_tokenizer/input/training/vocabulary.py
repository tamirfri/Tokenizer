from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.codec.batches import byte_reconstruction_loss
from continuous_tokenizer.codec.input import InputByteCodec
from continuous_tokenizer.contracts.profiles import Profile
from continuous_tokenizer.input.alignment import (
    EmbeddingMetrics,
    embedding_alignment_loss,
)
from continuous_tokenizer.input.training.cache import FrozenSpanCache
from continuous_tokenizer.input.training.runtime import TrainingRuntime
from continuous_tokenizer.input.training.vocabulary_batches import (
    VocabularyBucket,
    build_vocabulary_batches,
    build_vocabulary_groups,
    stage_vocabulary_groups,
    vocabulary_bucket_offsets,
    vocabulary_bucket_tensor_bytes,
)
from continuous_tokenizer.runtime.device import default_device
from continuous_tokenizer.runtime.progress import ProgressTracker, log_event
from continuous_tokenizer.runtime.resume import capture_torch_rng_state, restore_torch_rng_state
from continuous_tokenizer.runtime.tensors import module_state_snapshot
from continuous_tokenizer.training.optimizers import TokenizerOptimizers, optimizer_metadata

if TYPE_CHECKING:
    from continuous_tokenizer.input.training.run import TrainingOptions


@final
@dataclass(frozen=True, slots=True)
class AlignmentResult:
    profile: str
    optimizer: dict[str, str | int]
    embedding_metrics: dict[str, int | float]
    candidate_state_bytes: int
    reference_state_bytes: int
    candidate_reference_state_ratio: float


@final
@dataclass(frozen=True, slots=True)
class EmbeddingEpochRequest:
    codec: InputByteCodec
    optimizers: TokenizerOptimizers
    generator: torch.Generator
    vocabulary_groups: tuple[VocabularyBucket, ...]
    ordering_groups: tuple[VocabularyBucket, ...] | None = None
    epoch: int | None = None


@final
@dataclass(frozen=True, slots=True)
class DecoderEpochRequest:
    codec: InputByteCodec
    optimizers: TokenizerOptimizers
    generator: torch.Generator
    vocabulary_groups: tuple[VocabularyBucket, ...]
    latent_cache: FrozenSpanCache | None = None
    validate_cache: bool = True
    epoch: int | None = None


@final
@dataclass(frozen=True, slots=True)
class VocabularyProgress:
    profile: Profile
    phase: str
    epoch: int
    loss: float
    stale_epochs: int
    training_seconds: float
    evaluation_seconds: float
    metrics: EmbeddingMetrics


@final
@dataclass(frozen=True, slots=True)
class VocabularyFitter:
    runtime: TrainingRuntime

    def fit(
        self,
        codec: InputByteCodec,
        profile: Profile,
        generator: torch.Generator,
    ) -> dict[str, int | float | str]:
        token_ids = self.runtime.training_vocabulary_ids
        vocabulary_groups, preparation_seconds = self.runtime.timed(
            lambda: build_vocabulary_groups(self.runtime.assets, token_ids),
        )
        alignment_cache_metrics = self.fit_encoder(
            codec,
            profile,
            generator,
            vocabulary_groups=vocabulary_groups,
            token_ids=token_ids,
        )
        cache_metrics = self._fit_decoder(
            codec,
            profile,
            generator,
            vocabulary_groups,
            token_ids,
        )
        metrics: dict[str, int | float | str] = {
            "batch_preparation_seconds": preparation_seconds,
            "bucket_tensor_bytes": vocabulary_bucket_tensor_bytes(vocabulary_groups),
            "transfer_policy": "full_device",
            **alignment_cache_metrics,
            **cache_metrics,
            **self.runtime.cache_telemetry(),
        }
        log_event("input_training_cache_prepared", phase="vocabulary", **metrics)
        suffix = "-resume" if self.runtime.resume_manager is not None and self.runtime.resume_manager.resuming else ""
        self.runtime.write_json(
            f"progress/{profile.name}-vocabulary-cache{suffix}.json",
            metrics,
        )
        return metrics

    def fit_encoder(  # noqa: PLR0915 - Epoch selection and resume state stay colocated.
        self,
        codec: InputByteCodec,
        profile: Profile,
        generator: torch.Generator,
        token_ids: Sequence[int] | None = None,
        *,
        vocabulary_groups: tuple[VocabularyBucket, ...] | None = None,
    ) -> dict[str, int | float]:
        groups = build_vocabulary_groups(self.runtime.assets, token_ids) if vocabulary_groups is None else vocabulary_groups
        deployment_codec = self.runtime.deployment_evaluator(codec)
        encoder = codec.set_trainable_components(encoder=True, decoder=False)
        optimizers = self.runtime.optimizers(codec, encoder)
        best_state = module_state_snapshot(codec)
        best_score: tuple[float, float, float] | None = None
        stale_epochs = 0
        start_epoch = 0
        resume = self.runtime.resume_manager
        state = None if resume is None else resume.latest("input-alignment")
        if state is not None:
            codec.load_state_dict(state["codec"])
            optimizers.load_resume_state(state["optimizers"])
            generator.set_state(state["generator"])
            restore_torch_rng_state(self.runtime.device, state["torch_rng"])
            best_state = state["best_codec"]
            best_score = state["best_score"]
            stale_epochs = int(state["stale_epochs"])
            start_epoch = int(state["epoch"])
            if state["completed"]:
                codec.load_state_dict(best_state)
                return {
                    "device_staging_seconds": 0.0,
                    "device_staging_bytes": 0,
                }
        staged_groups, staging_seconds = self.runtime.timed(lambda: stage_vocabulary_groups(groups, self.runtime.device))
        for epoch in range(start_epoch, self.runtime.options.vocabulary_epochs):
            epoch_number = epoch + 1
            log_event(
                "epoch_started",
                phase="input_vocabulary_alignment",
                epoch=epoch_number,
                total_epochs=self.runtime.options.vocabulary_epochs,
            )
            loss, training_seconds = self.runtime.timed(
                lambda epoch_number=epoch_number: self.train_embedding_epoch(
                    EmbeddingEpochRequest(
                        codec=codec,
                        optimizers=optimizers,
                        generator=generator,
                        vocabulary_groups=staged_groups,
                        ordering_groups=groups,
                        epoch=epoch_number,
                    )
                )
            )
            self.runtime.write_epoch_telemetry(
                "vocabulary-alignment",
                epoch_number,
                wall_seconds=training_seconds,
                component_losses={"embedding_alignment": loss},
                optimizer=optimizers.epoch_telemetry(reset=True),
            )
            if not self._should_evaluate(epoch):
                completed = epoch_number == self.runtime.options.vocabulary_epochs
                if resume is not None and (completed or resume.should_snapshot(epoch_number)):
                    resume.save(
                        "input-alignment",
                        epoch_number,
                        {
                            "completed": completed,
                            "codec": module_state_snapshot(codec),
                            "optimizers": optimizers.resume_state(),
                            "generator": generator.get_state(),
                            "torch_rng": capture_torch_rng_state(self.runtime.device),
                            "best_codec": best_state,
                            "best_score": best_score,
                            "stale_epochs": stale_epochs,
                        },
                    )
                self.runtime.complete_epoch(
                    "input_vocabulary_alignment",
                    epoch_number,
                )
                continue
            log_event(
                "evaluation_started",
                phase="input_vocabulary_alignment",
                epoch=epoch_number,
            )
            metrics, evaluation_seconds = self.runtime.timed(
                lambda: self.runtime.evaluate_deployment(
                    codec,
                    deployment_codec,
                    reconstruction=False,
                    token_ids=token_ids,
                )
            )
            score = self.runtime.alignment_score(metrics)
            selected = best_score is None or score > best_score
            if selected:
                best_score = score
                best_state = module_state_snapshot(codec)
                stale_epochs = 0
            else:
                stale_epochs += self.runtime.options.evaluation_interval
            self._write_progress(
                VocabularyProgress(
                    profile=profile,
                    phase="alignment",
                    epoch=epoch + 1,
                    loss=loss,
                    stale_epochs=stale_epochs,
                    training_seconds=training_seconds,
                    evaluation_seconds=evaluation_seconds,
                    metrics=metrics,
                )
            )
            log_event(
                "evaluation_completed",
                phase="input_vocabulary_alignment",
                epoch=epoch_number,
                training_loss=loss,
                normalized_rmse=metrics.normalized_rmse,
                cosine_p01=metrics.cosine_similarity_p01,
                cosine_p50=metrics.cosine_similarity_p50,
                selected=selected,
                stale_epochs=stale_epochs,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
            )
            completed = (
                self.runtime.options.embedding_targets.accepts_alignment(metrics)
                or stale_epochs >= self.runtime.options.patience
                or epoch_number == self.runtime.options.vocabulary_epochs
            )
            if resume is not None and (completed or resume.should_snapshot(epoch_number)):
                resume.save(
                    "input-alignment",
                    epoch_number,
                    {
                        "completed": completed,
                        "codec": module_state_snapshot(codec),
                        "optimizers": optimizers.resume_state(),
                        "generator": generator.get_state(),
                        "torch_rng": capture_torch_rng_state(self.runtime.device),
                        "best_codec": best_state,
                        "best_score": best_score,
                        "stale_epochs": stale_epochs,
                    },
                )
            self.runtime.complete_epoch(
                "input_vocabulary_alignment",
                epoch_number,
            )
            if completed:
                break
        codec.load_state_dict(best_state)
        return {
            "device_staging_seconds": staging_seconds,
            "device_staging_bytes": vocabulary_bucket_tensor_bytes(staged_groups),
        }

    def _fit_decoder(
        self,
        codec: InputByteCodec,
        profile: Profile,
        generator: torch.Generator,
        vocabulary_groups: tuple[VocabularyBucket, ...],
        token_ids: Sequence[int],
    ) -> dict[str, int | float]:
        deployment_codec = self.runtime.deployment_evaluator(codec)
        decoder = codec.set_trainable_components(encoder=False, decoder=True)
        optimizers = self.runtime.optimizers(codec, decoder)
        resume = self.runtime.resume_manager
        state = None if resume is None else resume.latest("input-reconstruction")
        if state is not None:
            codec.load_state_dict(state["codec"])
            optimizers.load_resume_state(state["optimizers"])
            generator.set_state(state["generator"])
            restore_torch_rng_state(self.runtime.device, state["torch_rng"])
        latent_cache, latent_cache_seconds = self.runtime.timed(lambda: self._vocabulary_latents(codec, vocabulary_groups) if vocabulary_groups else None)
        deployment_cache, deployment_cache_seconds = self.runtime.timed(
            lambda: self.runtime.cached_deployment_evaluation(
                codec,
                deployment_codec,
                token_ids=token_ids,
            )
        )
        initial_metrics = self.runtime.evaluate_cached_deployment(
            codec,
            deployment_codec,
            deployment_cache,
        )
        best_state = module_state_snapshot(codec) if state is None else state["best_codec"]
        best_compatibility_score = self.runtime.compatibility_score(initial_metrics) if state is None else tuple(state["best_score"])
        stale_epochs = 0 if state is None else int(state["stale_epochs"])
        start_epoch = 0 if state is None else int(state["epoch"])
        if state is not None and state["completed"]:
            codec.load_state_dict(best_state)
            codec.set_trainable_components(encoder=True, decoder=True)
            return {
                "fp32_latent_cache_seconds": latent_cache_seconds,
                "fp32_latent_cache_bytes": (0 if latent_cache is None else latent_cache.tensor_bytes),
                "source_dtype_cache_seconds": deployment_cache_seconds,
                "source_dtype_cache_bytes": deployment_cache.tensor_bytes,
            }
        for epoch in range(start_epoch, self.runtime.options.vocabulary_epochs):
            epoch_number = epoch + 1
            log_event(
                "epoch_started",
                phase="input_vocabulary_reconstruction",
                epoch=epoch_number,
                total_epochs=self.runtime.options.vocabulary_epochs,
            )
            loss, training_seconds = self.runtime.timed(
                lambda epoch_number=epoch_number: self.train_decoder_epoch(
                    DecoderEpochRequest(
                        codec=codec,
                        optimizers=optimizers,
                        generator=generator,
                        vocabulary_groups=vocabulary_groups,
                        latent_cache=latent_cache,
                        validate_cache=False,
                        epoch=epoch_number,
                    )
                )
            )
            self.runtime.write_epoch_telemetry(
                "vocabulary-reconstruction",
                epoch_number,
                wall_seconds=training_seconds,
                component_losses={"byte_reconstruction": loss},
                optimizer=optimizers.epoch_telemetry(reset=True),
            )
            if not self._should_evaluate(epoch):
                if resume is not None and resume.should_snapshot(epoch_number):
                    resume.save(
                        "input-reconstruction",
                        epoch_number,
                        {
                            "completed": False,
                            "codec": module_state_snapshot(codec),
                            "optimizers": optimizers.resume_state(),
                            "generator": generator.get_state(),
                            "torch_rng": capture_torch_rng_state(self.runtime.device),
                            "best_codec": best_state,
                            "best_score": best_compatibility_score,
                            "stale_epochs": stale_epochs,
                        },
                    )
                self.runtime.complete_epoch(
                    "input_vocabulary_reconstruction",
                    epoch_number,
                )
                continue
            log_event(
                "evaluation_started",
                phase="input_vocabulary_reconstruction",
                epoch=epoch_number,
            )
            metrics, evaluation_seconds = self.runtime.timed(
                lambda: self.runtime.evaluate_cached_deployment(
                    codec,
                    deployment_codec,
                    deployment_cache,
                )
            )
            score = self.runtime.compatibility_score(metrics)
            selected = score > best_compatibility_score
            if selected:
                best_compatibility_score = score
                best_state = module_state_snapshot(codec)
                stale_epochs = 0
            else:
                stale_epochs += self.runtime.options.evaluation_interval
            self._write_progress(
                VocabularyProgress(
                    profile=profile,
                    phase="reconstruction",
                    epoch=epoch + 1,
                    loss=loss,
                    stale_epochs=stale_epochs,
                    training_seconds=training_seconds,
                    evaluation_seconds=evaluation_seconds,
                    metrics=metrics,
                )
            )
            log_event(
                "evaluation_completed",
                phase="input_vocabulary_reconstruction",
                epoch=epoch_number,
                training_loss=loss,
                reconstruction_fraction=metrics.reconstruction_fraction,
                selected=selected,
                stale_epochs=stale_epochs,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
            )
            completed = (
                metrics.reconstruction_fraction == 1.0
                or stale_epochs >= self.runtime.options.patience
                or epoch_number == self.runtime.options.vocabulary_epochs
            )
            if resume is not None and (completed or resume.should_snapshot(epoch_number)):
                resume.save(
                    "input-reconstruction",
                    epoch_number,
                    {
                        "completed": completed,
                        "codec": module_state_snapshot(codec),
                        "optimizers": optimizers.resume_state(),
                        "generator": generator.get_state(),
                        "torch_rng": capture_torch_rng_state(self.runtime.device),
                        "best_codec": best_state,
                        "best_score": best_compatibility_score,
                        "stale_epochs": stale_epochs,
                    },
                )
            self.runtime.complete_epoch(
                "input_vocabulary_reconstruction",
                epoch_number,
            )
            if completed:
                break
        codec.load_state_dict(best_state)
        codec.set_trainable_components(encoder=True, decoder=True)
        return {
            "fp32_latent_cache_seconds": latent_cache_seconds,
            "fp32_latent_cache_bytes": (0 if latent_cache is None else latent_cache.tensor_bytes),
            "source_dtype_cache_seconds": deployment_cache_seconds,
            "source_dtype_cache_bytes": deployment_cache.tensor_bytes,
        }

    def train_embedding_epoch(self, request: EmbeddingEpochRequest) -> float:
        codec = request.codec
        optimizers = request.optimizers
        codec.train()
        total: Tensor | None = None
        batches = 0
        epoch_batches = build_vocabulary_batches(
            request.vocabulary_groups if request.ordering_groups is None else request.ordering_groups,
            self.runtime.options.batch_size,
            request.generator,
        )
        progress = (
            None
            if request.epoch is None or not epoch_batches
            else ProgressTracker(
                "input_vocabulary_alignment_batches",
                len(epoch_batches),
                {"epoch": request.epoch},
            )
        )
        for reference in epoch_batches:
            bucket = request.vocabulary_groups[reference.bucket]
            rows = reference.rows.to(self.runtime.device)
            latent = codec.encode(
                bucket.byte_values.index_select(0, rows),
                bucket.valid_mask.index_select(0, rows),
            )
            target = bucket.source_targets.index_select(0, rows).to(dtype=latent.dtype)
            loss = embedding_alignment_loss(
                latent[: reference.logical_rows],
                target[: reference.logical_rows],
            )
            optimizers.optimize(loss)
            total = loss.detach() if total is None else total + loss.detach()
            batches += 1
            if progress is not None:
                progress.update(batches)
        return self.runtime.mean_epoch_loss(total, batches)

    def train_decoder_epoch(self, request: DecoderEpochRequest) -> float:
        codec = request.codec
        optimizers = request.optimizers
        vocabulary_groups = request.vocabulary_groups
        codec.train()
        total: Tensor | None = None
        batches = 0
        epoch_batches = build_vocabulary_batches(
            vocabulary_groups,
            self.runtime.options.batch_size,
            request.generator,
        )
        if not epoch_batches:
            return self.runtime.mean_epoch_loss(total, batches)
        cached = self._vocabulary_latents(codec, vocabulary_groups) if request.latent_cache is None else request.latent_cache
        if request.validate_cache:
            cached.validate(codec)
        offsets = vocabulary_bucket_offsets(vocabulary_groups)
        progress = (
            None
            if request.epoch is None or not epoch_batches
            else ProgressTracker(
                "input_vocabulary_reconstruction_batches",
                len(epoch_batches),
                {"epoch": request.epoch},
            )
        )
        for reference in epoch_batches:
            latent, targets, target_mask, positions = cached.select(
                reference.rows + offsets[reference.bucket],
                device=self.runtime.device,
            )
            logits = codec.decode_logits(latent, positions)
            loss = byte_reconstruction_loss(
                logits[: reference.logical_rows],
                targets[: reference.logical_rows],
                target_mask[: reference.logical_rows],
            )
            optimizers.optimize(loss)
            total = loss.detach() if total is None else total + loss.detach()
            batches += 1
            if progress is not None:
                progress.update(batches)
        return self.runtime.mean_epoch_loss(total, batches)

    def _vocabulary_latents(
        self,
        codec: InputByteCodec,
        groups: tuple[VocabularyBucket, ...],
    ) -> FrozenSpanCache:
        vocabulary = self.runtime.assets.vocabulary
        cache, _ = self.runtime.frozen_span_cache(
            codec,
            tuple(vocabulary.bytes_for(token_id) for group in groups for token_id in group.token_ids.tolist()),
            batch_size=self.runtime.evaluation_batch_size,
        )
        return cache

    def _should_evaluate(self, epoch: int) -> bool:
        epoch_number = epoch + 1
        return epoch_number % self.runtime.options.evaluation_interval == 0 or epoch_number == self.runtime.options.vocabulary_epochs

    def _write_progress(self, progress: VocabularyProgress) -> None:
        self.runtime.write_json(
            f"progress/{progress.profile.name}-vocabulary-{progress.phase}-{progress.epoch:03d}.json",
            {
                "epoch": progress.epoch,
                "phase": progress.phase,
                "profile": progress.profile.name,
                "stale_epochs": progress.stale_epochs,
                "training_loss": progress.loss,
                "training_seconds": progress.training_seconds,
                "evaluation_seconds": progress.evaluation_seconds,
                "embedding_metrics": progress.metrics.to_dict(),
            },
        )


def fit_vocabulary_alignment(
    assets: ModelAssets,
    options: TrainingOptions,
    *,
    device: torch.device | None = None,
    token_ids: Sequence[int] | None = None,
) -> AlignmentResult:
    selected_device = default_device() if device is None else device
    runtime = TrainingRuntime(assets, options, selected_device)
    torch.manual_seed(options.seed)
    generator = torch.Generator().manual_seed(options.seed)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    runtime.write_json("run-manifest.json", runtime.run_manifest())
    codec = runtime.build_codec(options.profile)
    VocabularyFitter(runtime).fit_encoder(codec, options.profile, generator, token_ids)
    runtime.prepare_deployment(codec)
    metrics = runtime.evaluate(codec, reconstruction=False, token_ids=token_ids)
    candidate_state_bytes, reference_state_bytes = runtime.candidate_reference_state_bytes(codec)
    result = AlignmentResult(
        profile=options.profile.name,
        optimizer=optimizer_metadata(options.muon_ns_steps),
        embedding_metrics=metrics.to_dict(),
        candidate_state_bytes=candidate_state_bytes,
        reference_state_bytes=reference_state_bytes,
        candidate_reference_state_ratio=(candidate_state_bytes / reference_state_bytes),
    )
    runtime.write_json("alignment-result.json", asdict(result))
    return result
