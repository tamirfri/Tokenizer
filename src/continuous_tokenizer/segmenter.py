from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from continuous_tokenizer.batching import build_span_batch
from continuous_tokenizer.cache import CacheKey, EncodingCache


class SpanCodec(Protocol):
    training: bool

    @property
    def max_span(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def encode(self, byte_values: Tensor, valid_mask: Tensor) -> Tensor: ...

    def decode_greedy(self, latent: Tensor) -> list[bytes | None]: ...


@dataclass(frozen=True, slots=True)
class EncodedSpan:
    data: bytes
    latent: Tensor
    atomic: bool


def _cache_is_safe(codec: SpanCodec, cache: EncodingCache | None) -> bool:
    return (
        cache is not None
        and not codec.training
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled()
    )


def encode_spans(
    codec: SpanCodec,
    spans: list[bytes],
    *,
    cache: EncodingCache | None = None,
    namespace: str = "uncached",
) -> Tensor:
    if not spans:
        raise ValueError("spans must not be empty")
    use_cache = _cache_is_safe(codec, cache)
    dtype = codec.dtype
    results: list[Tensor | None] = [None] * len(spans)
    misses: list[tuple[int, bytes, CacheKey]] = []

    for index, span in enumerate(spans):
        key = CacheKey(namespace, str(dtype), span)
        value = cache.get(key, device=codec.device, dtype=dtype) if use_cache and cache else None
        if value is None:
            misses.append((index, span, key))
        else:
            results[index] = value

    if misses:
        batch = build_span_batch(
            [span for _, span, _ in misses], max_span=codec.max_span, device=codec.device
        )
        encoded = codec.encode(batch.byte_values, batch.valid_mask)
        for row, (index, _, key) in enumerate(misses):
            value = encoded[row]
            results[index] = value
            if use_cache and cache:
                cache.put(key, value)

    if any(value is None for value in results):
        raise RuntimeError("internal error: missing encoded span")
    return torch.stack([value for value in results if value is not None])


def greedy_segment(
    codec: SpanCodec,
    data: bytes,
    *,
    max_candidate_bytes: int = 64,
    cache: EncodingCache | None = None,
    namespace: str = "uncached",
) -> list[EncodedSpan]:
    if max_candidate_bytes < 2:
        raise ValueError("max_candidate_bytes must be at least 2")
    result: list[EncodedSpan] = []
    position = 0
    candidate_limit = min(max_candidate_bytes, codec.max_span)

    while position < len(data):
        remaining = len(data) - position
        lengths = range(2, min(candidate_limit, remaining) + 1)
        candidates = [data[position : position + length] for length in lengths]
        selected: EncodedSpan | None = None

        if candidates:
            latents = encode_spans(codec, candidates, cache=cache, namespace=namespace)
            decoded = codec.decode_greedy(latents)
            for candidate, latent, reconstructed in zip(candidates, latents, decoded, strict=True):
                if reconstructed == candidate:
                    selected = EncodedSpan(candidate, latent, atomic=False)

        if selected is None:
            atomic = data[position : position + 1]
            latent = encode_spans(codec, [atomic], cache=cache, namespace=namespace)[0]
            selected = EncodedSpan(atomic, latent, atomic=True)

        result.append(selected)
        position += len(selected.data)
    return result


def reconstruct(segments: list[EncodedSpan]) -> bytes:
    return b"".join(segment.data for segment in segments)
