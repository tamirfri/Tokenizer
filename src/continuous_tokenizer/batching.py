from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SpanBatch:
    byte_values: Tensor
    valid_mask: Tensor
    framed_targets: Tensor
    target_mask: Tensor


def build_span_batch(spans: list[bytes], *, max_span: int, device: torch.device) -> SpanBatch:
    if not spans:
        raise ValueError("spans must not be empty")
    if any(not 1 <= len(span) <= max_span for span in spans):
        raise ValueError(f"every span length must be between 1 and {max_span}")

    inputs = torch.zeros((len(spans), max_span), dtype=torch.long, device=device)
    valid_mask = torch.zeros((len(spans), max_span), dtype=torch.bool, device=device)
    targets = torch.zeros((len(spans), max_span + 1), dtype=torch.long, device=device)
    target_mask = torch.zeros((len(spans), max_span + 1), dtype=torch.bool, device=device)

    for row, span in enumerate(spans):
        values = torch.tensor(list(span), dtype=torch.long, device=device)
        length = len(span)
        inputs[row, :length] = values
        valid_mask[row, :length] = True
        targets[row, 0] = length
        targets[row, 1 : length + 1] = values
        target_mask[row, : length + 1] = True
    return SpanBatch(inputs, valid_mask, targets, target_mask)


def byte_reconstruction_loss(logits: Tensor, targets: Tensor, target_mask: Tensor) -> Tensor:
    per_position = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    weights = target_mask.to(per_position.dtype)
    return (per_position * weights).sum() / weights.sum().clamp_min(1)
