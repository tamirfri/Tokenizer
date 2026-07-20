from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from continuous_tokenizer.batching import build_span_batch
from continuous_tokenizer.codec import ContinuousByteCodec
from continuous_tokenizer.vocabulary import ByteVocabulary


@dataclass(frozen=True, slots=True)
class EmbeddingMetrics:
    rows: int
    exact_rows: int
    exact_fraction: float
    reconstruction_rows: int
    reconstruction_fraction: float
    normalized_rmse: float
    mean_cosine_similarity: float
    maximum_absolute_error: float
    relative_l2_p50: float
    relative_l2_p95: float
    relative_l2_p99: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _quantile(values: Tensor, probability: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), probability).item())


@torch.inference_mode()
def evaluate_embeddings(
    codec: ContinuousByteCodec,
    vocabulary: ByteVocabulary,
    source_embeddings: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> EmbeddingMetrics:
    codec.eval()
    exact_rows = 0
    reconstructed = 0
    squared_error = 0.0
    squared_target = 0.0
    cosine_sum = 0.0
    maximum_error = 0.0
    relative_l2: list[Tensor] = []
    ids = list(vocabulary.ordinary_ids)

    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        spans = [vocabulary.bytes_for(token_id) for token_id in batch_ids]
        batch = build_span_batch(spans, max_span=codec.max_span, device=device)
        latent = codec.encode(batch.byte_values, batch.valid_mask)
        targets = source_embeddings[batch_ids].to(device=device, dtype=latent.dtype)
        source_dtype_latent = latent.to(source_embeddings.dtype).cpu()
        source_rows = source_embeddings[batch_ids].cpu()
        exact_rows += int((source_dtype_latent == source_rows).all(dim=1).sum().item())
        decoded = codec.decode_greedy(latent)
        reconstructed += sum(
            actual == expected for actual, expected in zip(decoded, spans, strict=True)
        )

        difference = latent.float() - targets.float()
        squared_error += float(difference.square().sum().item())
        squared_target += float(targets.float().square().sum().item())
        cosine_sum += float(F.cosine_similarity(latent.float(), targets.float()).sum().item())
        maximum_error = max(maximum_error, float(difference.abs().max().item()))
        row_relative = difference.norm(dim=1) / targets.float().norm(dim=1).clamp_min(1e-12)
        relative_l2.append(row_relative.cpu())

    rows = len(ids)
    relative = torch.cat(relative_l2) if relative_l2 else torch.empty(0)
    normalized_rmse = math.sqrt(squared_error / max(squared_target, 1e-12))
    return EmbeddingMetrics(
        rows=rows,
        exact_rows=exact_rows,
        exact_fraction=exact_rows / max(rows, 1),
        reconstruction_rows=reconstructed,
        reconstruction_fraction=reconstructed / max(rows, 1),
        normalized_rmse=normalized_rmse,
        mean_cosine_similarity=cosine_sum / max(rows, 1),
        maximum_absolute_error=maximum_error,
        relative_l2_p50=_quantile(relative, 0.50),
        relative_l2_p95=_quantile(relative, 0.95),
        relative_l2_p99=_quantile(relative, 0.99),
    )


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def module_bytes(codec: ContinuousByteCodec) -> int:
    return sum(tensor_bytes(tensor) for tensor in codec.state_dict().values())
