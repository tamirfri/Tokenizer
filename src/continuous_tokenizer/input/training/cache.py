from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import final

import torch
from torch import Tensor

from continuous_tokenizer.codec.batches import build_byte_batch, span_bucket_width
from continuous_tokenizer.codec.constants import CODEC_EOS
from continuous_tokenizer.codec.input import InputByteCodec
from continuous_tokenizer.runtime.tensors import tensor_bytes


@final
@dataclass(frozen=True, slots=True)
class FrozenSpanCache:
    encoder_fingerprint: str
    dtype: torch.dtype
    latents: Tensor
    payload_bytes: Tensor
    lengths: Tensor
    occurrence_to_unique: Tensor
    maximum_span: int

    @property
    def tensor_bytes(self) -> int:
        return sum(
            tensor_bytes(value)
            for value in (
                self.latents,
                self.payload_bytes,
                self.lengths,
                self.occurrence_to_unique,
            )
        )

    def validate(self, codec: InputByteCodec) -> None:
        if self.dtype != codec.dtype or self.encoder_fingerprint != codec.encoder_fingerprint():
            raise ValueError("frozen span cache does not match the encoder state and dtype")

    def _unique_rows(self, indices: Tensor) -> Tensor:
        rows = indices.to(device="cpu", dtype=torch.long)
        return self.occurrence_to_unique.index_select(0, rows).to(dtype=torch.long)

    def select(
        self,
        indices: Tensor,
        *,
        device: torch.device | None = None,
        target_width: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        unique_rows = self._unique_rows(indices)
        lengths = self.lengths.index_select(0, unique_rows)
        maximum_length = int(lengths.max().item())
        required_width = span_bucket_width(maximum_length, max_span=self.maximum_span) + 1
        if target_width is None:
            target_width = required_width
        elif target_width < required_width or target_width > self.maximum_span + 1:
            raise ValueError("frozen span cache target width does not cover the selected payloads")
        latents = self.latents.index_select(0, unique_rows)
        payloads = self.payload_bytes.index_select(0, unique_rows)[:, : target_width - 1]
        targets = torch.zeros((len(unique_rows), target_width), dtype=torch.long)
        targets[:, : payloads.shape[1]] = payloads
        targets.scatter_(1, lengths[:, None], CODEC_EOS)
        positions = torch.arange(target_width)
        target_mask = positions[None, :] <= lengths[:, None]
        if device is not None and device.type != "cpu":
            latents = latents.to(device)
            targets = targets.to(device)
            target_mask = target_mask.to(device)
        return (
            latents,
            targets,
            target_mask,
            target_width,
        )

    def maximum_length(self, indices: Tensor) -> int:
        unique_rows = self._unique_rows(indices)
        return int(self.lengths.index_select(0, unique_rows).max().item())


def ordered_spans_digest(spans: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    for span in spans:
        digest.update(len(span).to_bytes(8, "big"))
        digest.update(span)
    return digest.hexdigest()


@torch.no_grad()
def build_frozen_span_cache(
    codec: InputByteCodec,
    spans: tuple[bytes, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> FrozenSpanCache:
    if not spans or batch_size < 1:
        raise ValueError("frozen span caches require spans and a positive batch size")
    if any(not 1 <= len(span) <= codec.max_span for span in spans):
        raise ValueError("frozen span cache payload exceeds the codec span limit")
    unique_spans = tuple(dict.fromkeys(spans))
    unique_rows = {span: index for index, span in enumerate(unique_spans)}
    occurrence_to_unique = torch.tensor(
        [unique_rows[span] for span in spans],
        dtype=torch.int32,
    )
    by_width: dict[int, list[int]] = {}
    for index, span in enumerate(unique_spans):
        width = span_bucket_width(len(span), max_span=codec.max_span)
        by_width.setdefault(width, []).append(index)
    latents = torch.empty(
        (len(unique_spans), codec.config.embedding_dim),
        dtype=codec.dtype,
        device="cpu",
    )
    for indices in by_width.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            padded = selected + [selected[-1]] * (batch_size - len(selected))
            batch = build_byte_batch(
                [unique_spans[index] for index in padded],
                max_span=codec.max_span,
                device=device,
            )
            rows = torch.tensor(selected, dtype=torch.long)
            encoded = codec.encode(batch.byte_values, batch.valid_mask)[: len(selected)].detach().to("cpu")
            latents.index_copy_(0, rows, encoded)
    payload_bytes = torch.zeros((len(unique_spans), codec.max_span), dtype=torch.uint8)
    lengths = torch.tensor([len(span) for span in unique_spans], dtype=torch.int32)
    for row, span in enumerate(unique_spans):
        payload_bytes[row, : len(span)] = torch.tensor(tuple(span), dtype=torch.uint8)
    return FrozenSpanCache(
        encoder_fingerprint=codec.encoder_fingerprint(),
        dtype=codec.dtype,
        latents=latents,
        payload_bytes=payload_bytes,
        lengths=lengths,
        occurrence_to_unique=occurrence_to_unique,
        maximum_span=codec.max_span,
    )
