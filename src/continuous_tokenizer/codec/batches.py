from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
from torch import Tensor
from torch.nn import functional as F

from continuous_tokenizer.codec.constants import CODEC_EOS


@final
@dataclass(frozen=True, slots=True)
class ByteBatch:
    byte_values: Tensor
    valid_mask: Tensor


@final
@dataclass(frozen=True, slots=True)
class SpanBatch:
    byte_values: Tensor
    valid_mask: Tensor
    framed_targets: Tensor
    target_mask: Tensor


def span_bucket_width(length: int, *, max_span: int) -> int:
    if not 1 <= length <= max_span:
        raise ValueError(f"span length must be between 1 and {max_span}")
    return min(1 << (length - 1).bit_length(), max_span)


def build_byte_batch(
    spans: list[bytes],
    *,
    max_span: int,
    device: torch.device,
    batch_span: int | None = None,
) -> ByteBatch:
    if not spans:
        raise ValueError("spans must not be empty")
    if any(not 1 <= len(span) <= max_span for span in spans):
        raise ValueError(f"every span length must be between 1 and {max_span}")

    minimum_span = span_bucket_width(max(map(len, spans)), max_span=max_span)
    if batch_span is None:
        batch_span = minimum_span
    elif not minimum_span <= batch_span <= max_span:
        raise ValueError("batch span must cover every span and fit the codec")
    lengths = torch.tensor([len(span) for span in spans], dtype=torch.long)
    packed = bytearray(len(spans) * batch_span)
    for row, span in enumerate(spans):
        start = row * batch_span
        packed[start : start + len(span)] = span
    inputs = torch.frombuffer(packed, dtype=torch.uint8).reshape(len(spans), batch_span).long()
    valid_mask = torch.arange(batch_span)[None, :] < lengths[:, None]
    return ByteBatch(inputs.to(device), valid_mask.to(device))


def build_span_batch(spans: list[bytes], *, max_span: int, device: torch.device) -> SpanBatch:
    batch = build_byte_batch(spans, max_span=max_span, device=torch.device("cpu"))
    lengths = batch.valid_mask.sum(dim=1)
    batch_span = batch.byte_values.shape[1]
    targets = torch.zeros((len(spans), batch_span + 1), dtype=torch.long)
    targets[:, :batch_span] = batch.byte_values
    targets.scatter_(1, lengths[:, None], CODEC_EOS)
    target_mask = torch.arange(batch_span + 1)[None, :] <= lengths[:, None]
    return SpanBatch(
        batch.byte_values.to(device),
        batch.valid_mask.to(device),
        targets.to(device),
        target_mask.to(device),
    )


def byte_reconstruction_loss(logits: Tensor, targets: Tensor, target_mask: Tensor) -> Tensor:
    if logits.shape[1] < targets.shape[1]:
        raise ValueError("decoder logits do not cover every reconstruction target")
    logits = logits[:, : targets.shape[1]]
    per_position = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    weights = target_mask.to(per_position.dtype)
    return (per_position * weights).sum() / weights.sum().clamp_min(1)


def decode_span_rows(generated: Tensor, *, max_span: int) -> list[bytes | None]:
    spans: list[bytes | None] = []
    for row in generated:
        eos_positions = (row == CODEC_EOS).nonzero(as_tuple=False)
        if eos_positions.numel() == 0:
            spans.append(None)
            continue
        length = int(eos_positions[0].item())
        spans.append(bytes(row[:length].tolist()) if 1 <= length <= max_span else None)
    return spans
