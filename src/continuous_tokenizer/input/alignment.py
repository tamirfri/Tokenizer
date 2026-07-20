from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Final, final

import torch
from torch import Tensor
from torch.nn import functional as F

from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.batches import build_span_batch
from continuous_tokenizer.codec.input import InputByteCodec
from continuous_tokenizer.runtime.tensors import tensor_bytes


@final
@dataclass(frozen=True, slots=True)
class EmbeddingMetrics:
    rows: int
    exact_rows: int
    exact_fraction: float
    reconstruction_rows: int
    reconstruction_fraction: float
    normalized_rmse: float
    mean_cosine_similarity: float
    cosine_similarity_p01: float
    cosine_similarity_p50: float
    cosine_similarity_p99: float
    maximum_absolute_error: float
    relative_l2_p50: float
    relative_l2_p95: float
    relative_l2_p99: float
    retrieval_top1_fraction: float
    retrieval_top5_fraction: float
    retrieval_queries: int
    retrieval_candidates: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@final
@dataclass(frozen=True, slots=True)
class EmbeddingFitTargets:
    maximum_normalized_rmse: float = 1e-2
    minimum_cosine_p01: float = 0.999
    minimum_cosine_p50: float = 0.9999

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> EmbeddingFitTargets:
        if values is None:
            return cls()
        defaults = cls()
        return cls(
            maximum_normalized_rmse=float(values.get("maximum_normalized_rmse", defaults.maximum_normalized_rmse)),
            minimum_cosine_p01=float(values.get("minimum_cosine_p01", defaults.minimum_cosine_p01)),
            minimum_cosine_p50=float(values.get("minimum_cosine_p50", defaults.minimum_cosine_p50)),
        )

    def accepts_alignment(self, metrics: EmbeddingMetrics) -> bool:
        return (
            metrics.normalized_rmse <= self.maximum_normalized_rmse
            and metrics.cosine_similarity_p01 >= self.minimum_cosine_p01
            and metrics.cosine_similarity_p50 >= self.minimum_cosine_p50
        )

    def accepts(self, metrics: EmbeddingMetrics) -> bool:
        return self.accepts_alignment(metrics) and metrics.reconstruction_fraction == 1.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_EMBEDDING_FIT_TARGETS: Final = EmbeddingFitTargets()


@final
@dataclass(frozen=True, slots=True)
class CachedEmbeddingBatch:
    latent: Tensor
    byte_values: Tensor
    valid_mask: Tensor


@final
@dataclass(frozen=True, slots=True)
class CachedEmbeddingEvaluation:
    encoder_fingerprint: str
    dtype: torch.dtype
    static_rows: int
    batches: tuple[CachedEmbeddingBatch, ...]
    metrics: EmbeddingMetrics

    @property
    def tensor_bytes(self) -> int:
        return sum(tensor_bytes(value) for batch in self.batches for value in (batch.latent, batch.byte_values, batch.valid_mask))


@final
@dataclass(frozen=True, slots=True)
class CachedEmbeddingRequest:
    batch_size: int
    device: torch.device
    token_ids: Sequence[int] | None = None


@final
@dataclass(frozen=True, slots=True)
class EmbeddingEvaluationRequest:
    batch_size: int
    device: torch.device
    retrieval: bool = False
    retrieval_rows: int | None = 4096
    retrieval_candidate_batch_size: int = 4096
    reconstruction: bool = True
    token_ids: Sequence[int] | None = None


@final
@dataclass(frozen=True, slots=True)
class _RetrievalRequest:
    query_batch_size: int
    candidate_batch_size: int
    device: torch.device


def tensor_quantile(values: Tensor, probability: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), probability).item())


def _embedding_ids(
    vocabulary: ByteVocabulary,
    token_ids: Sequence[int] | None,
) -> list[int]:
    ids = list(vocabulary.compatibility_ids if token_ids is None else token_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(vocabulary.compatibility_ids):
        raise ValueError("embedding evaluation IDs must be unique compatibility vocabulary rows")
    ids.sort(key=lambda token_id: len(vocabulary.bytes_for(token_id)))
    return ids


def _pad_spans(spans: list[bytes], static_rows: int) -> list[bytes]:
    if not spans or static_rows < len(spans):
        raise ValueError("static evaluation rows must cover a non-empty span batch")
    return spans + [spans[-1]] * (static_rows - len(spans))


def _pad_tensor_rows(value: Tensor, static_rows: int) -> Tensor:
    if not 0 < value.shape[0] <= static_rows:
        raise ValueError("static evaluation rows must cover a non-empty tensor batch")
    padding = static_rows - value.shape[0]
    return torch.cat((value, value[-1:].expand(padding, *value.shape[1:]))) if padding else value


@torch.inference_mode()
def build_cached_embedding_evaluation(
    codec: InputByteCodec,
    vocabulary: ByteVocabulary,
    source_embeddings: Tensor,
    request: CachedEmbeddingRequest,
) -> CachedEmbeddingEvaluation:
    if codec.dtype != source_embeddings.dtype:
        raise ValueError("cached embedding evaluation requires the source deployment dtype")
    if request.batch_size < 1:
        raise ValueError("cached embedding evaluation batch size must be positive")
    ids = _embedding_ids(vocabulary, request.token_ids)
    batches = []
    latents = []
    targets = []
    for start in range(0, len(ids), request.batch_size):
        batch_ids = ids[start : start + request.batch_size]
        logical_rows = len(batch_ids)
        batch = build_span_batch(
            _pad_spans(
                [vocabulary.bytes_for(token_id) for token_id in batch_ids],
                request.batch_size,
            ),
            max_span=codec.max_span,
            device=request.device,
        )
        latent = codec.encode(batch.byte_values, batch.valid_mask)[:logical_rows]
        batches.append(
            CachedEmbeddingBatch(
                latent.to("cpu"),
                batch.byte_values[:logical_rows].to("cpu"),
                batch.valid_mask[:logical_rows].to("cpu"),
            )
        )
        latents.append(latent)
        targets.append(source_embeddings[batch_ids].to(device=request.device, dtype=latent.dtype))
    metrics = _summarize_cached_embeddings(latents, targets)
    return CachedEmbeddingEvaluation(
        encoder_fingerprint=codec.encoder_fingerprint(),
        dtype=codec.dtype,
        static_rows=request.batch_size,
        batches=tuple(batches),
        metrics=metrics,
    )


@torch.inference_mode()
def evaluate_cached_embeddings(
    codec: InputByteCodec,
    cache: CachedEmbeddingEvaluation,
    *,
    validate: bool = True,
) -> EmbeddingMetrics:
    if validate and (cache.dtype != codec.dtype or cache.encoder_fingerprint != codec.encoder_fingerprint()):
        raise ValueError("cached embedding evaluation does not match the encoder state and dtype")
    reconstructed = torch.zeros((), dtype=torch.long, device=codec.device)
    for batch in cache.batches:
        latent = batch.latent.to(codec.device)
        byte_values = batch.byte_values.to(codec.device)
        valid_mask = batch.valid_mask.to(codec.device)
        logical_rows = latent.shape[0]
        matches = codec.reconstruction_matches(
            _pad_tensor_rows(latent, cache.static_rows),
            _pad_tensor_rows(byte_values, cache.static_rows),
            _pad_tensor_rows(valid_mask, cache.static_rows),
        )[:logical_rows]
        reconstructed += (matches | (valid_mask.sum(dim=1) == 1)).sum()
    rows = cache.metrics.rows
    reconstructed_rows = int(reconstructed.cpu().item())
    return replace(
        cache.metrics,
        reconstruction_rows=reconstructed_rows,
        reconstruction_fraction=reconstructed_rows / max(rows, 1),
    )


def _summarize_cached_embeddings(
    latents: list[Tensor],
    targets: list[Tensor],
) -> EmbeddingMetrics:
    exact_rows = torch.zeros((), dtype=torch.long, device=latents[0].device)
    squared_error = torch.zeros((), device=latents[0].device)
    squared_target = torch.zeros((), device=latents[0].device)
    maximum_error = torch.zeros((), device=latents[0].device)
    relative_l2 = []
    cosine_values = []
    for latent, target in zip(latents, targets, strict=True):
        exact_rows += (latent.to(target.dtype) == target).all(dim=1).sum()
        difference = latent.float() - target.float()
        squared_error += difference.square().sum()
        squared_target += target.float().square().sum()
        maximum_error = torch.maximum(maximum_error, difference.abs().max())
        relative_l2.append(difference.norm(dim=1) / target.float().norm(dim=1).clamp_min(1e-12))
        cosine_values.append(F.cosine_similarity(latent.float(), target.float()))
    summary = torch.stack((exact_rows.float(), squared_error, squared_target, maximum_error)).cpu()
    exact = int(summary[0].item())
    squared_error_value = float(summary[1].item())
    squared_target_value = float(summary[2].item())
    relative = torch.cat(relative_l2).cpu()
    cosines = torch.cat(cosine_values).cpu()
    rows = sum(value.shape[0] for value in latents)
    return EmbeddingMetrics(
        rows=rows,
        exact_rows=exact,
        exact_fraction=exact / rows,
        reconstruction_rows=0,
        reconstruction_fraction=0.0,
        normalized_rmse=math.sqrt(squared_error_value / max(squared_target_value, 1e-12)),
        mean_cosine_similarity=float(cosines.mean().item()),
        cosine_similarity_p01=tensor_quantile(cosines, 0.01),
        cosine_similarity_p50=tensor_quantile(cosines, 0.50),
        cosine_similarity_p99=tensor_quantile(cosines, 0.99),
        maximum_absolute_error=float(summary[3].item()),
        relative_l2_p50=tensor_quantile(relative, 0.50),
        relative_l2_p95=tensor_quantile(relative, 0.95),
        relative_l2_p99=tensor_quantile(relative, 0.99),
        retrieval_top1_fraction=0.0,
        retrieval_top5_fraction=0.0,
        retrieval_queries=0,
        retrieval_candidates=0,
    )


@torch.inference_mode()
def evaluate_embeddings(
    codec: InputByteCodec,
    vocabulary: ByteVocabulary,
    source_embeddings: Tensor,
    request: EmbeddingEvaluationRequest,
) -> EmbeddingMetrics:
    if request.batch_size < 1:
        raise ValueError("embedding evaluation batch size must be positive")
    codec.eval()
    exact_rows = torch.zeros((), dtype=torch.long, device=request.device)
    reconstructed = torch.zeros((), dtype=torch.long, device=request.device)
    squared_error = torch.zeros((), device=request.device)
    squared_target = torch.zeros((), device=request.device)
    maximum_error = torch.zeros((), device=request.device)
    relative_l2: list[Tensor] = []
    cosine_values: list[Tensor] = []
    ids = _embedding_ids(vocabulary, request.token_ids)

    for start in range(0, len(ids), request.batch_size):
        batch_ids = ids[start : start + request.batch_size]
        spans = [vocabulary.bytes_for(token_id) for token_id in batch_ids]
        logical_rows = len(batch_ids)
        batch = build_span_batch(
            _pad_spans(spans, request.batch_size),
            max_span=codec.max_span,
            device=request.device,
        )
        if request.reconstruction:
            latent, matches = codec.encode_and_reconstruction_matches(
                batch.byte_values,
                batch.valid_mask,
            )
            matches = matches[:logical_rows]
        else:
            latent = codec.encode(batch.byte_values, batch.valid_mask)
        latent = latent[:logical_rows]
        valid_mask = batch.valid_mask[:logical_rows]
        source_rows = source_embeddings[batch_ids].to(device=request.device)
        targets = source_rows.to(latent.dtype)
        exact_rows += (latent.to(source_rows.dtype) == source_rows).all(dim=1).sum()
        if request.reconstruction:
            reconstructed += (matches | (valid_mask.sum(dim=1) == 1)).sum()

        difference = latent.float() - targets.float()
        squared_error += difference.square().sum()
        squared_target += targets.float().square().sum()
        cosine = F.cosine_similarity(latent.float(), targets.float())
        cosine_values.append(cosine)
        maximum_error = torch.maximum(maximum_error, difference.abs().max())
        row_relative = difference.norm(dim=1) / targets.float().norm(dim=1).clamp_min(1e-12)
        relative_l2.append(row_relative)

    rows = len(ids)
    summary = torch.stack(
        (
            exact_rows.float(),
            reconstructed.float(),
            squared_error,
            squared_target,
            maximum_error,
        )
    ).cpu()
    exact_rows_value = int(summary[0].item())
    reconstructed_value = int(summary[1].item())
    squared_error_value = float(summary[2].item())
    squared_target_value = float(summary[3].item())
    maximum_error_value = float(summary[4].item())
    relative = torch.cat(relative_l2).cpu() if relative_l2 else torch.empty(0)
    cosines = torch.cat(cosine_values).cpu() if cosine_values else torch.empty(0)
    normalized_rmse = math.sqrt(squared_error_value / max(squared_target_value, 1e-12))
    selected_positions = _retrieval_positions(len(ids), request.retrieval_rows) if request.retrieval else []
    selected_ids = [ids[position] for position in selected_positions]
    retrieval_latents = _encode_retrieval_latents(
        codec,
        vocabulary,
        selected_ids,
        request,
    )
    top1, top5 = _retrieval_accuracy(
        torch.cat(retrieval_latents) if retrieval_latents else None,
        source_embeddings[ids] if selected_ids else None,
        selected_positions,
        _RetrievalRequest(
            query_batch_size=request.batch_size,
            candidate_batch_size=request.retrieval_candidate_batch_size,
            device=request.device,
        ),
    )
    return EmbeddingMetrics(
        rows=rows,
        exact_rows=exact_rows_value,
        exact_fraction=exact_rows_value / max(rows, 1),
        reconstruction_rows=reconstructed_value,
        reconstruction_fraction=reconstructed_value / max(rows, 1),
        normalized_rmse=normalized_rmse,
        mean_cosine_similarity=float(cosines.mean().item()) if cosines.numel() else 0.0,
        cosine_similarity_p01=tensor_quantile(cosines, 0.01),
        cosine_similarity_p50=tensor_quantile(cosines, 0.50),
        cosine_similarity_p99=tensor_quantile(cosines, 0.99),
        maximum_absolute_error=maximum_error_value,
        relative_l2_p50=tensor_quantile(relative, 0.50),
        relative_l2_p95=tensor_quantile(relative, 0.95),
        relative_l2_p99=tensor_quantile(relative, 0.99),
        retrieval_top1_fraction=top1,
        retrieval_top5_fraction=top5,
        retrieval_queries=len(selected_ids),
        retrieval_candidates=len(ids) if selected_ids else 0,
    )


def _encode_retrieval_latents(
    codec: InputByteCodec,
    vocabulary: ByteVocabulary,
    selected_ids: list[int],
    request: EmbeddingEvaluationRequest,
) -> list[Tensor]:
    latents = []
    for start in range(0, len(selected_ids), request.batch_size):
        batch_ids = selected_ids[start : start + request.batch_size]
        spans = [vocabulary.bytes_for(token_id) for token_id in batch_ids]
        batch = build_span_batch(
            _pad_spans(spans, request.batch_size),
            max_span=codec.max_span,
            device=request.device,
        )
        latents.append(codec.encode(batch.byte_values, batch.valid_mask)[: len(batch_ids)].cpu())
    return latents


def _retrieval_positions(size: int, limit: int | None) -> list[int]:
    if limit is None or size <= limit:
        return list(range(size))
    if limit <= 0:
        raise ValueError("retrieval row limit must be positive or None")
    return torch.linspace(0, size - 1, steps=limit).round().long().unique().tolist()


def _retrieval_accuracy(
    latents: Tensor | None,
    targets: Tensor | None,
    expected_indices: list[int],
    request: _RetrievalRequest,
) -> tuple[float, float]:
    if latents is None or targets is None or latents.numel() == 0:
        return 0.0, 0.0
    query_count = latents.shape[0]
    candidate_count = targets.shape[0]
    top_k = min(5, candidate_count)
    top1_matches = torch.zeros((), dtype=torch.long, device=request.device)
    top5_matches = torch.zeros((), dtype=torch.long, device=request.device)

    for query_start in range(0, query_count, request.query_batch_size):
        query_end = min(query_start + request.query_batch_size, query_count)
        queries = F.normalize(latents[query_start:query_end].to(request.device).float(), dim=1)
        best_scores = torch.full((queries.shape[0], top_k), -torch.inf, device=request.device)
        best_indices = torch.full((queries.shape[0], top_k), -1, dtype=torch.long, device=request.device)

        for candidate_start in range(0, candidate_count, request.candidate_batch_size):
            candidate_end = min(candidate_start + request.candidate_batch_size, candidate_count)
            candidates = F.normalize(targets[candidate_start:candidate_end].to(request.device).float(), dim=1)
            scores = queries @ candidates.transpose(0, 1)
            local_k = min(top_k, scores.shape[1])
            local_scores, local_indices = scores.topk(local_k, dim=1)
            local_indices += candidate_start
            combined_scores = torch.cat((best_scores, local_scores), dim=1)
            combined_indices = torch.cat((best_indices, local_indices), dim=1)
            best_scores, selected = combined_scores.topk(top_k, dim=1)
            best_indices = combined_indices.gather(1, selected)

        expected = torch.tensor(expected_indices[query_start:query_end], device=request.device)
        top1_matches += (best_indices[:, 0] == expected).sum()
        top5_matches += (best_indices == expected[:, None]).any(dim=1).sum()

    matches = torch.stack((top1_matches, top5_matches)).cpu()
    return float(matches[0].item()) / query_count, float(matches[1].item()) / query_count


def embedding_alignment_loss(latent: Tensor, target: Tensor) -> Tensor:
    latent_float = latent.float()
    target_float = target.float()
    relative_mse = (latent_float - target_float).square().mean(dim=1) / (target_float.square().mean(dim=1).clamp_min(1e-12))
    cosine_error = 1.0 - F.cosine_similarity(latent_float, target_float, dim=1)
    return (relative_mse + cosine_error).mean()
