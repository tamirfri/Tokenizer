from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, final

import torch
from torch import Tensor

from continuous_tokenizer.codec.batches import byte_reconstruction_loss, span_bucket_width
from continuous_tokenizer.codec.input import InputByteCodec
from continuous_tokenizer.contracts.profiles import Profile
from continuous_tokenizer.input.alignment import (
    CachedEmbeddingEvaluation,
    EmbeddingMetrics,
)
from continuous_tokenizer.input.training.cache import FrozenSpanCache
from continuous_tokenizer.input.training.runtime import TrainingRuntime
from continuous_tokenizer.runtime.progress import ProgressTracker, log_event
from continuous_tokenizer.runtime.resume import capture_torch_rng_state, restore_torch_rng_state
from continuous_tokenizer.runtime.tensors import module_state_snapshot
from continuous_tokenizer.training.optimizers import TokenizerOptimizers


@final
@dataclass(frozen=True, slots=True)
class MixedEpochRequest:
    codec: InputByteCodec
    corpus_spans: list[bytes]
    optimizers: TokenizerOptimizers
    randomizer: random.Random
    vocabulary_cache: FrozenSpanCache | None = None
    dynamic_cache: FrozenSpanCache | None = None
    dynamic_order: list[int] | None = None
    validate_cache: bool = True
    epoch: int | None = None


@final
@dataclass(frozen=True, slots=True)
class ReconstructionFitResult:
    cache_metrics: dict[str, int | float]
    embedding_metrics: EmbeddingMetrics
    native_tokens_per_continuous_token: float
    round_trip: bool
    density_identity: dict[str, str | int]


@final
@dataclass(frozen=True, slots=True)
class ReconstructionFitter:
    runtime: TrainingRuntime

    def _selection_baseline(
        self,
        codec: InputByteCodec,
        deployment_codec: InputByteCodec,
        deployment_cache: CachedEmbeddingEvaluation,
        validation_data: bytes,
        state: dict[str, Any] | None,
    ) -> tuple[dict[str, Tensor], EmbeddingMetrics, float, bool]:
        if state is not None:
            return (
                state["best_codec"],
                EmbeddingMetrics(**state["best_metrics"]),
                float(state["best_density"]),
                bool(state["best_round_trip"]),
            )
        metrics = self.runtime.evaluate_cached_deployment(
            codec,
            deployment_codec,
            deployment_cache,
        )
        density, round_trip = self.runtime.density_metrics(
            deployment_codec,
            validation_data,
        )
        return module_state_snapshot(codec), metrics, density, round_trip

    def fit(
        self,
        codec: InputByteCodec,
        profile: Profile,
        corpus_spans: list[bytes],
        validation_data: bytes,
        randomizer: random.Random,
    ) -> ReconstructionFitResult:
        parameters = codec.set_trainable_components(encoder=False, decoder=True)
        optimizers = self.runtime.optimizers(codec, parameters)
        resume = self.runtime.resume_manager
        state = None if resume is None else resume.latest("input-dynamic-reconstruction")
        if state is not None:
            codec.load_state_dict(state["codec"])
            optimizers.load_resume_state(state["optimizers"])
            randomizer.setstate(state["randomizer"])
            restore_torch_rng_state(self.runtime.device, state["torch_rng"])
        deployment_codec = self.runtime.deployment_evaluator(codec)
        deployment_cache, deployment_cache_seconds = self.runtime.timed(
            lambda: self.runtime.cached_deployment_evaluation(
                codec,
                deployment_codec,
            )
        )
        (
            best_state,
            best_metrics,
            best_native_tokens_per_continuous_token,
            best_round_trip,
        ) = self._selection_baseline(
            codec,
            deployment_codec,
            deployment_cache,
            validation_data,
            state,
        )
        alignment_required = self.runtime.options.embedding_targets.accepts(best_metrics)
        best_score = (*self.runtime.compatibility_score(best_metrics), best_native_tokens_per_continuous_token) if state is None else tuple(state["best_score"])
        (vocabulary_cache, dynamic_cache), replay_cache_seconds = self.runtime.timed(
            lambda: (
                self.runtime.frozen_span_cache(
                    codec,
                    tuple(self.runtime.assets.vocabulary.bytes_for(token_id) for token_id in self.runtime.training_vocabulary_ids),
                    batch_size=self.runtime.evaluation_batch_size,
                )[0],
                (
                    self.runtime.frozen_span_cache(
                        codec,
                        tuple(corpus_spans),
                        batch_size=self.runtime.evaluation_batch_size,
                    )[0]
                    if corpus_spans
                    else None
                ),
            )
        )
        dynamic_order = list(range(len(corpus_spans))) if state is None else list(state["dynamic_order"])
        start_epoch = 0 if state is None else int(state["epoch"])
        cache_metrics = {
            "replay_cache_seconds": replay_cache_seconds,
            "replay_cache_bytes": vocabulary_cache.tensor_bytes + (0 if dynamic_cache is None else dynamic_cache.tensor_bytes),
            "source_dtype_cache_seconds": deployment_cache_seconds,
            "source_dtype_cache_bytes": deployment_cache.tensor_bytes,
            **self.runtime.cache_telemetry(),
        }
        if state is not None and state["completed"]:
            start_epoch = self.runtime.options.reconstruction_epochs
        for epoch in range(start_epoch, self.runtime.options.reconstruction_epochs):
            epoch_number = epoch + 1
            log_event(
                "epoch_started",
                phase="input_dynamic_reconstruction",
                epoch=epoch_number,
                total_epochs=self.runtime.options.reconstruction_epochs,
            )
            loss, training_seconds = self.runtime.timed(
                lambda epoch_number=epoch_number: self.train_mixed_epoch(
                    MixedEpochRequest(
                        codec=codec,
                        corpus_spans=corpus_spans,
                        optimizers=optimizers,
                        randomizer=randomizer,
                        vocabulary_cache=vocabulary_cache,
                        dynamic_cache=dynamic_cache,
                        dynamic_order=dynamic_order,
                        validate_cache=False,
                        epoch=epoch_number,
                    )
                )
            )
            self.runtime.write_epoch_telemetry(
                "dynamic-reconstruction",
                epoch_number,
                wall_seconds=training_seconds,
                component_losses={"byte_reconstruction": loss},
                optimizer=optimizers.epoch_telemetry(reset=True),
            )
            log_event(
                "evaluation_started",
                phase="input_dynamic_reconstruction",
                epoch=epoch_number,
            )
            (metrics, native_tokens_per_continuous_token, round_trip), evaluation_seconds = self.runtime.timed(
                lambda: (
                    self.runtime.evaluate_cached_deployment(
                        codec,
                        deployment_codec,
                        deployment_cache,
                    ),
                    *self.runtime.density_metrics(deployment_codec, validation_data),
                )
            )
            score = (*self.runtime.compatibility_score(metrics), native_tokens_per_continuous_token)
            selected = round_trip and (not alignment_required or self.runtime.options.embedding_targets.accepts(metrics)) and score > best_score
            if selected:
                best_score = score
                best_native_tokens_per_continuous_token = native_tokens_per_continuous_token
                best_round_trip = round_trip
                best_metrics = metrics
                best_state = module_state_snapshot(codec)
            self.runtime.write_json(
                f"progress/{profile.name}-dynamic-reconstruction-{epoch + 1:03d}.json",
                {
                    "phase": "dynamic_reconstruction",
                    "profile": profile.name,
                    "epoch": epoch + 1,
                    "training_loss": loss,
                    "training_seconds": training_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "native_tokens_per_continuous_token": native_tokens_per_continuous_token,
                    "round_trip": round_trip,
                    "selected": selected,
                    "embedding_metrics": metrics.to_dict(),
                },
            )
            log_event(
                "evaluation_completed",
                phase="input_dynamic_reconstruction",
                epoch=epoch_number,
                training_loss=loss,
                reconstruction_fraction=metrics.reconstruction_fraction,
                native_tokens_per_continuous_token=native_tokens_per_continuous_token,
                round_trip=round_trip,
                selected=selected,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
            )
            completed = epoch_number == self.runtime.options.reconstruction_epochs
            if resume is not None and (completed or resume.should_snapshot(epoch_number)):
                resume.save(
                    "input-dynamic-reconstruction",
                    epoch_number,
                    {
                        "completed": completed,
                        "codec": module_state_snapshot(codec),
                        "optimizers": optimizers.resume_state(),
                        "randomizer": randomizer.getstate(),
                        "torch_rng": capture_torch_rng_state(self.runtime.device),
                        "dynamic_order": dynamic_order,
                        "best_codec": best_state,
                        "best_score": best_score,
                        "best_metrics": asdict(best_metrics),
                        "best_density": best_native_tokens_per_continuous_token,
                        "best_round_trip": best_round_trip,
                    },
                )
            self.runtime.complete_epoch(
                "input_dynamic_reconstruction",
                epoch_number,
            )
        codec.load_state_dict(best_state)
        self.runtime.write_json(
            f"progress/{profile.name}-dynamic-reconstruction-selected.json",
            {
                "phase": "dynamic_reconstruction",
                "profile": profile.name,
                "native_tokens_per_continuous_token": best_native_tokens_per_continuous_token,
                "round_trip": best_round_trip,
                "embedding_metrics": best_metrics.to_dict(),
            },
        )
        log_event(
            "input_training_cache_prepared",
            phase="dynamic_reconstruction",
            **cache_metrics,
        )
        suffix = "-resume" if self.runtime.resume_manager is not None and self.runtime.resume_manager.resuming else ""
        self.runtime.write_json(
            f"progress/{profile.name}-dynamic-reconstruction-cache{suffix}.json",
            cache_metrics,
        )
        if deployment_codec is not codec:
            deployment_codec.load_state_dict(codec.state_dict())
        return ReconstructionFitResult(
            cache_metrics=cache_metrics,
            embedding_metrics=best_metrics,
            native_tokens_per_continuous_token=(best_native_tokens_per_continuous_token),
            round_trip=best_round_trip,
            density_identity=self.runtime.density_identity(
                deployment_codec,
                validation_data,
            ),
        )

    def train_mixed_epoch(self, request: MixedEpochRequest) -> float:
        codec = request.codec
        corpus_spans = request.corpus_spans
        optimizers = request.optimizers
        randomizer = request.randomizer
        codec.train()
        if not corpus_spans:
            return 0.0
        compatibility_ids = self.runtime.training_vocabulary_ids
        cached_vocabulary = (
            self.runtime.frozen_span_cache(
                codec,
                tuple(self.runtime.assets.vocabulary.bytes_for(token_id) for token_id in compatibility_ids),
                batch_size=self.runtime.evaluation_batch_size,
            )[0]
            if request.vocabulary_cache is None
            else request.vocabulary_cache
        )
        cached_dynamic = (
            self.runtime.frozen_span_cache(
                codec,
                tuple(corpus_spans),
                batch_size=self.runtime.evaluation_batch_size,
            )[0]
            if request.dynamic_cache is None
            else request.dynamic_cache
        )
        if request.validate_cache:
            for cache in (cached_vocabulary, cached_dynamic):
                cache.validate(codec)
        total: Tensor | None = None
        batches = 0
        order = list(range(len(corpus_spans))) if request.dynamic_order is None else request.dynamic_order
        randomizer.shuffle(order)
        vocabulary_rows = {token_id: row for row, token_id in enumerate(compatibility_ids)}
        dynamic_batch_size = max(
            1,
            int(self.runtime.options.batch_size * (1.0 - self.runtime.options.reconstruction_vocabulary_fraction)),
        )
        total_batches = (len(corpus_spans) + dynamic_batch_size - 1) // dynamic_batch_size
        progress = (
            None
            if request.epoch is None or total_batches == 0
            else ProgressTracker(
                "input_dynamic_reconstruction_batches",
                total_batches,
                {"epoch": request.epoch},
            )
        )
        for start in range(0, len(order), dynamic_batch_size):
            dynamic = torch.tensor(
                order[start : start + dynamic_batch_size],
                dtype=torch.long,
            )
            vocabulary_count = self.runtime.options.batch_size - len(dynamic)
            vocab_ids = randomizer.choices(compatibility_ids, k=vocabulary_count)
            vocabulary = torch.tensor(
                [vocabulary_rows[token_id] for token_id in vocab_ids],
                dtype=torch.long,
            )
            maximum_length = max(
                cached_vocabulary.maximum_length(vocabulary),
                cached_dynamic.maximum_length(dynamic),
            )
            positions = span_bucket_width(maximum_length, max_span=codec.max_span) + 1
            vocab_latent, vocab_targets, vocab_mask, _ = cached_vocabulary.select(
                vocabulary,
                device=self.runtime.device,
                target_width=positions,
            )
            dynamic_latent, dynamic_targets, dynamic_mask, _ = cached_dynamic.select(
                dynamic,
                device=self.runtime.device,
                target_width=positions,
            )
            latent = torch.cat((vocab_latent, dynamic_latent))
            targets = torch.cat((vocab_targets, dynamic_targets))
            target_mask = torch.cat((vocab_mask, dynamic_mask))
            logits = codec.decode_logits(latent, positions)
            loss = byte_reconstruction_loss(
                logits,
                targets,
                target_mask,
            )
            optimizers.optimize(loss)
            total = loss.detach() if total is None else total + loss.detach()
            batches += 1
            if progress is not None:
                progress.update(batches)
        return self.runtime.mean_epoch_loss(total, batches)
