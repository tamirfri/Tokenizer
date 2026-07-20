from __future__ import annotations

import hashlib
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, final

import torch
from torch import Tensor
from torch.nn import Parameter

from continuous_tokenizer.artifacts.store import write_json_atomic
from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.backbone.config import tie_word_embeddings
from continuous_tokenizer.codec.compilation import (
    DYNAMIC_SEGMENTATION_MAX_BYTES,
    TOKENIZER_RECOMPILE_LIMIT,
)
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.codec.layers import gqa_metadata
from continuous_tokenizer.contracts.profiles import Profile
from continuous_tokenizer.input.alignment import (
    CachedEmbeddingEvaluation,
    CachedEmbeddingRequest,
    EmbeddingEvaluationRequest,
    EmbeddingMetrics,
    build_cached_embedding_evaluation,
    evaluate_cached_embeddings,
    evaluate_embeddings,
)
from continuous_tokenizer.input.evidence import input_source_identity
from continuous_tokenizer.input.segmentation import greedy_segment, reconstruct
from continuous_tokenizer.input.training.cache import (
    FrozenSpanCache,
    build_frozen_span_cache,
    ordered_spans_digest,
)
from continuous_tokenizer.runtime.environment import runtime_environment
from continuous_tokenizer.runtime.tensors import (
    module_bytes,
    parameter_fingerprint,
    tensor_bytes,
)
from continuous_tokenizer.runtime.timing import timed_call
from continuous_tokenizer.training.optimizers import (
    TokenizerOptimizers,
    build_tokenizer_optimizers,
    optimizer_metadata,
)

if TYPE_CHECKING:
    from continuous_tokenizer.input.training.run import TrainingOptions
    from continuous_tokenizer.runtime.resume import ResumeManager


type EpochBoundary = Callable[[str, int], None]


@final
@dataclass(slots=True)
class TrainingRuntime:
    assets: ModelAssets
    options: TrainingOptions
    device: torch.device
    resume_manager: ResumeManager | None = None
    epoch_boundary: EpochBoundary | None = None
    _deployment_evaluators: dict[tuple[Any, ...], InputByteCodec] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _deployment_caches: dict[tuple[Any, ...], CachedEmbeddingEvaluation] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _span_caches: dict[tuple[Any, ...], FrozenSpanCache] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _training_codecs: list[InputByteCodec] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _cache_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)

    def complete_epoch(self, phase: str, epoch: int) -> None:
        if self.epoch_boundary is not None:
            self.epoch_boundary(phase, epoch)

    @property
    def training_vocabulary_ids(self) -> tuple[int, ...]:
        selected = self.options.vocabulary_token_ids
        return self.assets.vocabulary.compatibility_ids if selected is None else selected

    @property
    def evaluation_batch_size(self) -> int:
        return min(self.options.batch_size, self.options.cache_chunk_rows)

    def build_codec(self, profile: Profile) -> InputByteCodec:
        vocabulary = self.assets.vocabulary
        embedding_dim = self.assets.input_embeddings.shape[1]
        byte_ids = torch.tensor(vocabulary.byte_token_ids, dtype=torch.long)
        byte_embeddings = self.assets.input_embeddings[byte_ids]
        config = InputByteCodecConfig(
            embedding_dim=embedding_dim,
            local_dim=profile.local_dim,
            projection_dim=profile.projection_dim(embedding_dim),
            max_span=max(
                vocabulary.max_token_bytes,
                DYNAMIC_SEGMENTATION_MAX_BYTES,
            ),
            query_heads=profile.query_heads,
            feedforward_dim=profile.feedforward_dim,
            encoder_layers=profile.encoder_layers,
            decoder_layers=profile.decoder_layers,
        )
        codec = InputByteCodec(config, byte_embeddings).to(self.device)
        if self.device.type == "mps":
            codec.compile_neural_paths(static_rows=self._compiled_static_rows())
        self._training_codecs.append(codec)
        return codec

    def _compiled_static_rows(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    self.options.batch_size,
                    self.evaluation_batch_size,
                }
            )
        )

    def optimizers(
        self,
        codec: InputByteCodec,
        parameters: tuple[Parameter, ...],
    ) -> TokenizerOptimizers:
        muon_parameters, adamw_parameters = codec.optimizer_parameter_groups(parameters)
        return build_tokenizer_optimizers(
            muon_parameters,
            adamw_parameters,
            learning_rate=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
            muon_ns_steps=self.options.muon_ns_steps,
        )

    @staticmethod
    def mean_epoch_loss(total: Tensor | None, batches: int) -> float:
        return 0.0 if total is None else float((total / batches).item())

    @staticmethod
    def alignment_score(metrics: EmbeddingMetrics) -> tuple[float, float, float]:
        return (
            -metrics.normalized_rmse,
            metrics.cosine_similarity_p01,
            metrics.cosine_similarity_p50,
        )

    @classmethod
    def compatibility_score(
        cls,
        metrics: EmbeddingMetrics,
    ) -> tuple[bool, int, float, float, float]:
        return (
            metrics.reconstruction_fraction == 1.0,
            metrics.reconstruction_rows,
            *cls.alignment_score(metrics),
        )

    def evaluate(
        self,
        codec: InputByteCodec,
        *,
        reconstruction: bool = True,
        token_ids: Sequence[int] | None = None,
    ) -> EmbeddingMetrics:
        return evaluate_embeddings(
            codec,
            self.assets.vocabulary,
            self.assets.input_embeddings,
            EmbeddingEvaluationRequest(
                batch_size=self.evaluation_batch_size,
                device=self.device,
                reconstruction=reconstruction,
                token_ids=token_ids,
            ),
        )

    def deployment_evaluator(self, codec: InputByteCodec) -> InputByteCodec:
        source_dtype = self.assets.input_embeddings.dtype
        key = (
            tuple(sorted(codec.config.to_dict().items())),
            str(source_dtype),
            str(self.device),
        )
        if codec.dtype == source_dtype:
            self._cache_counts["deployment_evaluator_reuses"] += 1
            return codec
        evaluator = self._deployment_evaluators.get(key)
        if evaluator is None:
            evaluator = InputByteCodec(
                codec.config,
                codec.byte_embeddings.detach().to(dtype=source_dtype),
            ).to(device=self.device, dtype=source_dtype)
            for parameter in evaluator.parameters():
                parameter.requires_grad_(False)
            if self.device.type == "mps":
                evaluator.compile_neural_paths(
                    static_rows=self._compiled_static_rows(),
                )
            evaluator.eval()
            self._deployment_evaluators[key] = evaluator
            self._cache_counts["deployment_evaluator_builds"] += 1
        else:
            self._cache_counts["deployment_evaluator_reuses"] += 1
        return evaluator

    def prepare_deployment(self, codec: InputByteCodec) -> None:
        codec.to(dtype=self.assets.input_embeddings.dtype)
        if self.device.type == "mps":
            codec.compile_neural_paths(static_rows=self._compiled_static_rows())
        codec.eval()

    def evaluate_deployment(
        self,
        codec: InputByteCodec,
        evaluator: InputByteCodec,
        *,
        reconstruction: bool = True,
        token_ids: Sequence[int] | None = None,
    ) -> EmbeddingMetrics:
        if evaluator is not codec:
            evaluator.load_state_dict(codec.state_dict())
        return self.evaluate(
            evaluator,
            reconstruction=reconstruction,
            token_ids=token_ids,
        )

    def cached_deployment_evaluation(
        self,
        codec: InputByteCodec,
        evaluator: InputByteCodec,
        *,
        token_ids: Sequence[int] | None = None,
    ) -> CachedEmbeddingEvaluation:
        selected_ids = self.training_vocabulary_ids if token_ids is None else tuple(token_ids)
        key = (
            codec.encoder_fingerprint(),
            selected_ids,
            str(self.assets.input_embeddings.dtype),
            str(self.device),
            tuple(self.assets.input_embeddings.shape),
            self.evaluation_batch_size,
            "cached-embedding-evaluation-v1",
        )
        cached = self._deployment_caches.get(key)
        if cached is not None:
            self._cache_counts["source_dtype_cache_reuses"] += 1
            return cached
        if evaluator is not codec:
            evaluator.load_state_dict(codec.state_dict())
        cached = build_cached_embedding_evaluation(
            evaluator,
            self.assets.vocabulary,
            self.assets.input_embeddings,
            CachedEmbeddingRequest(
                batch_size=self.evaluation_batch_size,
                device=self.device,
                token_ids=selected_ids,
            ),
        )
        self._deployment_caches[key] = cached
        self._cache_counts["source_dtype_cache_builds"] += 1
        return cached

    def frozen_span_cache(
        self,
        codec: InputByteCodec,
        spans: tuple[bytes, ...],
        *,
        batch_size: int,
    ) -> tuple[FrozenSpanCache, bool]:
        key = (
            codec.encoder_fingerprint(),
            ordered_spans_digest(spans),
            str(codec.dtype),
            str(self.device),
            codec.config.embedding_dim,
            codec.max_span,
            batch_size,
            "compact-frozen-span-v1",
        )
        cached = self._span_caches.get(key)
        if cached is not None:
            self._cache_counts["frozen_span_cache_reuses"] += 1
            return cached, True
        cached = build_frozen_span_cache(
            codec,
            spans,
            batch_size=batch_size,
            device=self.device,
        )
        self._span_caches[key] = cached
        self._cache_counts["frozen_span_cache_builds"] += 1
        return cached, False

    def cache_telemetry(self) -> dict[str, int]:
        telemetry = {
            name: self._cache_counts.get(name, 0)
            for name in (
                "deployment_evaluator_builds",
                "deployment_evaluator_reuses",
                "source_dtype_cache_builds",
                "source_dtype_cache_reuses",
                "frozen_span_cache_builds",
                "frozen_span_cache_reuses",
            )
        }
        graph_telemetry = [
            codec.graph_signature_telemetry()
            for codec in (
                *self._training_codecs,
                *self._deployment_evaluators.values(),
            )
        ]
        planned = [count for item in graph_telemetry for count in item["planned"].values()]
        encountered = [len(signatures) for item in graph_telemetry for signatures in item["encountered"].values()]
        return telemetry | {
            "accelerator_length_synchronizations": 0,
            "compiled_graph_signature_limit": TOKENIZER_RECOMPILE_LIMIT,
            "maximum_planned_graph_signatures": max(planned, default=0),
            "maximum_encountered_graph_signatures": max(encountered, default=0),
        }

    def graph_signature_telemetry(self) -> dict[str, dict[str, Any]]:
        return {
            **{f"training_{index}": codec.graph_signature_telemetry() for index, codec in enumerate(self._training_codecs)},
            **{f"deployment_{index}": codec.graph_signature_telemetry() for index, codec in enumerate(self._deployment_evaluators.values())},
        }

    def evaluate_cached_deployment(
        self,
        codec: InputByteCodec,
        evaluator: InputByteCodec,
        cache: CachedEmbeddingEvaluation,
    ) -> EmbeddingMetrics:
        if evaluator is not codec:
            evaluator.load_decoder_state(codec)
        return evaluate_cached_embeddings(evaluator, cache, validate=False)

    def candidate_reference_state_bytes(
        self,
        codec: InputByteCodec,
    ) -> tuple[int, int]:
        control_ids = torch.tensor(self.assets.vocabulary.control_ids, dtype=torch.long)
        source_embeddings = self.assets.input_embeddings
        control_embedding_bytes = control_ids.numel() * source_embeddings.shape[1] * source_embeddings.element_size()
        deployed = module_bytes(codec) + tensor_bytes(control_ids) + control_embedding_bytes
        return deployed, tensor_bytes(source_embeddings)

    @torch.inference_mode()
    def density_metrics(self, codec: InputByteCodec, data: bytes) -> tuple[float, bool]:
        codec.eval()
        text = data.decode("utf-8", errors="strict")
        native_ids = self.assets.tokenizer.encode(text, add_special_tokens=False)
        segments = greedy_segment(codec, data, namespace=self.assets.revision)
        ratio = len(native_ids) / max(len(segments), 1)
        return ratio, reconstruct(segments) == data

    def density_identity(
        self,
        codec: InputByteCodec,
        data: bytes,
    ) -> dict[str, str | int]:
        return {
            "checkpoint_sha256": parameter_fingerprint(codec),
            "corpus_sha256": hashlib.sha256(data).hexdigest(),
            "candidate_limit": codec.max_span,
            "dtype": str(codec.dtype),
            "device": str(codec.device),
            "implementation": "exhaustive-greedy-density-v1",
        }

    def timed[T](self, operation: Callable[[], T]) -> tuple[T, float]:
        return timed_call(operation, self.device)

    def write_epoch_telemetry(
        self,
        phase: str,
        epoch: int,
        *,
        wall_seconds: float,
        component_losses: dict[str, float],
        optimizer: dict[str, int | float],
    ) -> dict[str, Any]:
        environment = runtime_environment(self.device)
        telemetry = {
            "phase": phase,
            "epoch": epoch,
            "epoch_wall_seconds": wall_seconds,
            "component_losses": component_losses,
            "gradient_norms": {name: value for name, value in optimizer.items() if "gradient_norm" in name},
            "optimizer_steps": optimizer["optimizer_steps"],
            "peak_memory": {
                "cpu_rss_bytes": optimizer["peak_cpu_rss_bytes"],
                "mps_allocated_bytes": optimizer["peak_mps_allocated_bytes"],
                "mps_driver_allocated_bytes": optimizer["peak_mps_driver_allocated_bytes"],
                "process_peak_rss_bytes": environment["peak_rss_bytes"],
            },
            "optimization_dtype": "torch.float32",
            "selection_dtype": str(self.assets.input_embeddings.dtype),
            "compiled_graph_signatures": self.graph_signature_telemetry(),
        }
        self.write_json(
            f"progress/{self.options.profile.name}-{phase}-telemetry-{epoch:03d}.json",
            telemetry,
        )
        return telemetry

    def serialized_options(self) -> dict[str, Any]:
        values = asdict(self.options)
        del values["output_dir"]
        if values["vocabulary_token_ids"] is not None:
            values["vocabulary_token_ids"] = list(values["vocabulary_token_ids"])
        values["profile"] = self.options.profile.name
        values["profile_config"] = asdict(self.options.profile)
        return values

    def experiment_metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.assets.model_id,
            "model_revision": self.assets.revision,
            "embedding_tensor_name": self.assets.embedding_tensor_name,
            "source_dtype": str(self.assets.input_embeddings.dtype),
            "source_shape": list(self.assets.input_embeddings.shape),
            "source_identity": input_source_identity(self.assets),
            "tie_word_embeddings": tie_word_embeddings(self.assets.config),
            "dataset": {
                "id": self.options.dataset_id,
                "config": self.options.dataset_config,
                "revision": self.options.dataset_revision,
            },
            "embedding_targets": self.options.embedding_targets.to_dict(),
            "minimum_native_tokens_per_continuous_token": self.options.minimum_native_tokens_per_continuous_token,
            "candidate_reference_state_target": (self.options.maximum_candidate_reference_state_ratio),
            "codec_compilation": {
                "enabled": self.device.type == "mps",
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": False,
                "shape_policy": "bounded_static_specialization",
                "recompile_limit": TOKENIZER_RECOMPILE_LIMIT,
                "scope": "standalone_and_complete_tokenizer_workloads",
            },
            "vocabulary_training_policy": {
                "alignment_loss": "mean_per_row_relative_mse_plus_cosine_error",
                "checkpoint_unit": "complete_codec",
                "encoder_condition": "source_embedding",
                "decoder_condition": "encoded_latent",
                "dynamic_reconstruction": "frozen_encoder_with_vocabulary_replay",
                "selection_dtype": str(self.assets.input_embeddings.dtype),
                "epoch_budget_per_phase": self.options.vocabulary_epochs,
                "phases": ["alignment", "reconstruction"],
            },
            "optimizer": optimizer_metadata(self.options.muon_ns_steps),
        }

    def checkpoint_metadata(self, profile: Profile, *, checkpoint_stage: str) -> dict[str, Any]:
        return {
            **self.experiment_metadata(),
            "profile": asdict(profile),
            "codec_attention": gqa_metadata(profile.query_heads),
            "checkpoint_stage": checkpoint_stage,
            "training": {
                **self.serialized_options(),
                "python": sys.version,
                "torch": str(torch.__version__),
            },
        }

    def run_manifest(self) -> dict[str, Any]:
        profile = self.options.profile
        return {
            **self.experiment_metadata(),
            "codec_attention": gqa_metadata(profile.query_heads),
            "embedding_shard": self.assets.embedding_shard.name,
            "reference_state_bytes": tensor_bytes(self.assets.input_embeddings),
            "options": self.serialized_options(),
            "environment": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "transformers": version("transformers"),
                "huggingface_hub": version("huggingface-hub"),
                "datasets": version("datasets"),
            },
        }

    def write_json(self, name: str, value: Any) -> None:
        write_json_atomic(self.options.output_dir / name, value)
