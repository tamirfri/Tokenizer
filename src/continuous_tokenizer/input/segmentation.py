from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, final

import torch
from torch import Tensor

from continuous_tokenizer.codec.batches import build_byte_batch, span_bucket_width
from continuous_tokenizer.codec.compilation import DYNAMIC_SEGMENTATION_MAX_BYTES
from continuous_tokenizer.codec.encoding_cache import CacheKey, EncodingCache

SEGMENTATION_FRONTIERS: Final = (1, 2, 4, 8)
LONG_ACCEPTED_SPAN: Final = 8


class SpanCodec(Protocol):
    training: bool

    @property
    def max_span(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def encode(self, byte_values: Tensor, valid_mask: Tensor) -> Tensor: ...

    def atomic_latent(self, value: int) -> Tensor: ...

    def reconstruction_matches(
        self,
        latent: Tensor,
        byte_values: Tensor,
        valid_mask: Tensor,
    ) -> Tensor: ...

    def encode_and_reconstruction_matches(
        self,
        byte_values: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


@final
@dataclass(frozen=True, slots=True)
class EncodedSpan:
    data: bytes
    latent: Tensor
    atomic: bool


@final
@dataclass(frozen=True, slots=True)
class SegmentationStats:
    candidates: int
    candidate_lengths: dict[int, int]
    valid_candidates: int
    invalid_by_length: dict[int, int]
    span_lengths: dict[int, int]
    atomic_spans: int


@final
@dataclass(frozen=True, slots=True)
class SegmentationWork:
    logical_candidates: int
    neural_candidate_rows: int
    speculative_discarded_rows: int

    @property
    def logical_candidate_rows(self) -> int:
        return self.logical_candidates

    @property
    def neural_padded_rows(self) -> int:
        return self.neural_candidate_rows


@final
@dataclass(frozen=True, slots=True)
class SegmentationResult:
    spans: tuple[EncodedSpan, ...]
    stats: SegmentationStats
    work: SegmentationWork


@dataclass(frozen=True, slots=True)
class _SegmentationContext:
    codec: SpanCodec
    data: bytes
    data_values: Tensor
    candidate_limit: int
    cache: EncodingCache | None
    namespace: str


@dataclass(frozen=True, slots=True)
class _CachedEncodingShape:
    padded_rows: int
    width: int

    def pad(self, spans: list[bytes]) -> list[bytes]:
        padded = list(spans)
        if len(padded) > self.padded_rows:
            raise ValueError("padded rows must cover every missing span")
        padded.extend([padded[-1]] * (self.padded_rows - len(padded)))
        return padded


@dataclass(frozen=True, slots=True)
class _CachedEncodingResult:
    latents: Tensor
    neural_rows: int


@dataclass(frozen=True, slots=True)
class _WindowValidation:
    valid_spans: dict[int, dict[int, Tensor]]
    candidate_rows: int
    neural_rows: int


def _usable_cache(codec: SpanCodec, cache: EncodingCache | None) -> EncodingCache | None:
    if cache is None or codec.training or torch.is_grad_enabled() or torch.is_autocast_enabled():
        return None
    return cache


def encode_spans(
    codec: SpanCodec,
    spans: list[bytes],
    *,
    cache: EncodingCache | None = None,
    namespace: str = "uncached",
) -> Tensor:
    if not spans:
        raise ValueError("spans must not be empty")
    usable_cache = _usable_cache(codec, cache)
    if usable_cache is not None:
        return _encode_cached_spans(codec, spans, cache=usable_cache, namespace=namespace)
    batch = build_byte_batch(spans, max_span=codec.max_span, device=codec.device)
    return codec.encode(batch.byte_values, batch.valid_mask)


def _encode_cached_spans(
    codec: SpanCodec,
    spans: list[bytes],
    *,
    cache: EncodingCache,
    namespace: str,
) -> Tensor:
    if any(not 1 <= len(span) <= codec.max_span for span in spans):
        raise ValueError(f"every span length must be between 1 and {codec.max_span}")
    computed_indices = [index for index, span in enumerate(spans) if len(span) > 1]
    atomic_indices = [index for index, span in enumerate(spans) if len(span) == 1]
    if not computed_indices:
        return torch.stack([codec.atomic_latent(spans[index][0]) for index in atomic_indices])
    computed_spans = [spans[index] for index in computed_indices]
    computed = _encode_cached_computed_spans(
        codec,
        computed_spans,
        cache=cache,
        namespace=namespace,
        shape=_CachedEncodingShape(
            padded_rows=1 << (len(computed_spans) - 1).bit_length(),
            width=span_bucket_width(
                max(map(len, computed_spans)),
                max_span=codec.max_span,
            ),
        ),
    )
    if not atomic_indices:
        return computed.latents
    atomic = torch.stack([codec.atomic_latent(spans[index][0]) for index in atomic_indices])
    output = torch.empty(
        (len(spans), computed.latents.shape[1]),
        device=codec.device,
        dtype=codec.dtype,
    )
    output.index_copy_(
        0,
        torch.tensor(computed_indices, dtype=torch.long, device=codec.device),
        computed.latents,
    )
    output.index_copy_(
        0,
        torch.tensor(atomic_indices, dtype=torch.long, device=codec.device),
        atomic,
    )
    return output


def _encode_cached_computed_spans(
    codec: SpanCodec,
    spans: list[bytes],
    *,
    cache: EncodingCache,
    namespace: str,
    shape: _CachedEncodingShape,
) -> _CachedEncodingResult:
    dtype = codec.dtype
    dtype_name = str(dtype)
    keys = [
        CacheKey(
            namespace,
            dtype_name,
            span,
            shape.padded_rows,
            shape.width,
        )
        for span in spans
    ]
    cached = cache.get_many_cpu(keys)
    missing_rows: dict[CacheKey, int] = {}
    for key, value in zip(keys, cached, strict=True):
        if value is None and key not in missing_rows:
            missing_rows[key] = len(missing_rows)

    missing_keys = tuple(missing_rows)
    if missing_keys:
        missing_spans = shape.pad([key.span for key in missing_keys])
        batch = build_byte_batch(
            missing_spans,
            max_span=codec.max_span,
            device=codec.device,
            batch_span=shape.width,
        )
        encoded_batch = codec.encode(batch.byte_values, batch.valid_mask)
        encoded = encoded_batch[: len(missing_keys)]
        cache.put_many(missing_keys, encoded)
        neural_rows = len(missing_spans)
    else:
        encoded = None
        neural_rows = 0

    hit_indices = [index for index, value in enumerate(cached) if value is not None]
    if encoded is None:
        hit_values = torch.stack([value for value in cached if value is not None])
        return _CachedEncodingResult(
            hit_values.to(device=codec.device, dtype=dtype),
            neural_rows,
        )

    output = torch.empty(
        (len(spans), encoded.shape[1]),
        device=codec.device,
        dtype=dtype,
    )
    if hit_indices:
        hit_values = torch.stack([value for value in cached if value is not None])
        output.index_copy_(
            0,
            torch.tensor(hit_indices, dtype=torch.long, device=codec.device),
            hit_values.to(device=codec.device, dtype=dtype),
        )
    miss_indices = [index for index, value in enumerate(cached) if value is None]
    if miss_indices:
        source_indices = [missing_rows[keys[index]] for index in miss_indices]
        output.index_copy_(
            0,
            torch.tensor(miss_indices, dtype=torch.long, device=codec.device),
            encoded.index_select(
                0,
                torch.tensor(source_indices, dtype=torch.long, device=codec.device),
            ),
        )
    return _CachedEncodingResult(output, neural_rows)


def validate_spans(
    codec: SpanCodec,
    spans: list[bytes],
    *,
    static_size: int,
    cache: EncodingCache | None = None,
    namespace: str = "uncached",
) -> tuple[Tensor, Tensor]:
    """Validate spans at fixed width with bounded power-of-two rows."""
    if not spans:
        raise ValueError("spans must not be empty")
    if not 1 <= static_size <= codec.max_span:
        raise ValueError("static size must fit the codec")
    if len(spans) > static_size or any(not 1 <= len(span) <= static_size for span in spans):
        raise ValueError("static size must cover every span and byte")

    target_rows = 1 << (len(spans) - 1).bit_length()
    padding = target_rows - len(spans)
    padded_spans = spans + [bytes(static_size)] * padding
    batch = build_byte_batch(
        padded_spans,
        max_span=codec.max_span,
        device=codec.device,
        batch_span=static_size,
    )
    usable_cache = _usable_cache(codec, cache)
    if usable_cache is not None:
        encoded = _encode_cached_computed_spans(
            codec,
            spans,
            cache=usable_cache,
            namespace=namespace,
            shape=_CachedEncodingShape(
                padded_rows=target_rows,
                width=static_size,
            ),
        )
        latents = encoded.latents
        padded_latents = torch.cat((latents, latents[-1:].expand(padding, -1))) if padding else latents
    else:
        padded_latents = codec.encode(batch.byte_values, batch.valid_mask)
        latents = padded_latents[: len(spans)]
    matches = codec.reconstruction_matches(
        padded_latents,
        batch.byte_values,
        batch.valid_mask,
    )
    return latents, matches[: len(spans)]


def _candidate_batch(
    data: Tensor,
    offsets: list[int],
    lengths: list[int],
    *,
    max_span: int,
) -> tuple[Tensor, Tensor]:
    offset_values = torch.tensor(offsets, dtype=torch.long, device=data.device)
    length_values = torch.tensor(lengths, dtype=torch.long, device=data.device)
    width = span_bucket_width(max(lengths), max_span=max_span)
    positions = torch.arange(width, dtype=torch.long, device=data.device)
    valid_mask = positions[None, :] < length_values[:, None]
    indices = offset_values[:, None] + positions[None, :]
    byte_values = data[indices.clamp_max(data.numel() - 1)]
    byte_values.masked_fill_(~valid_mask, 0)
    return byte_values, valid_mask


def _candidate_groups(lengths: list[int], *, max_span: int) -> tuple[list[int], ...]:
    by_width: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        width = span_bucket_width(length, max_span=max_span)
        by_width.setdefault(width, []).append(index)
    return tuple(by_width.values())


def candidate_group_rows(width: int, frontier: int) -> int:
    return frontier * max(1, width // 2)


def _pad_candidate_group(
    offsets: list[int],
    lengths: list[int],
    *,
    max_span: int,
    frontier: int,
) -> tuple[list[int], list[int]]:
    width = span_bucket_width(max(lengths), max_span=max_span)
    target_size = candidate_group_rows(width, frontier)
    padding = target_size - len(lengths)
    if padding < 0:
        raise RuntimeError("candidate group exceeds its bounded batch size")
    return offsets + [offsets[-1]] * padding, lengths + [lengths[-1]] * padding


def _window_valid_spans(
    context: _SegmentationContext,
    position: int,
    window_end: int,
    *,
    frontier: int,
) -> _WindowValidation:
    offsets: list[int] = []
    lengths: list[int] = []
    for offset in range(position, window_end):
        maximum_length = min(context.candidate_limit, len(context.data) - offset)
        for length in range(2, maximum_length + 1):
            offsets.append(offset)
            lengths.append(length)
    if not lengths:
        return _WindowValidation({}, 0, 0)

    group_results: list[tuple[list[int], Tensor]] = []
    match_batches: list[Tensor] = []
    neural_rows = 0
    for group in _candidate_groups(lengths, max_span=context.candidate_limit):
        group_offsets = [offsets[index] for index in group]
        group_lengths = [lengths[index] for index in group]
        padded_offsets, padded_lengths = _pad_candidate_group(
            group_offsets,
            group_lengths,
            max_span=context.candidate_limit,
            frontier=frontier,
        )
        byte_values, valid_mask = _candidate_batch(
            context.data_values,
            padded_offsets,
            padded_lengths,
            max_span=context.candidate_limit,
        )
        if context.cache is None:
            latents, matches = context.codec.encode_and_reconstruction_matches(
                byte_values,
                valid_mask,
            )
            neural_rows += len(padded_lengths)
        else:
            candidates = [
                context.data[offset : offset + length]
                for offset, length in zip(
                    group_offsets,
                    group_lengths,
                    strict=True,
                )
            ]
            encoded = _encode_cached_computed_spans(
                context.codec,
                candidates,
                cache=context.cache,
                namespace=context.namespace,
                shape=_CachedEncodingShape(
                    padded_rows=len(padded_lengths),
                    width=byte_values.shape[1],
                ),
            )
            latents = encoded.latents
            padded_latents = torch.cat(
                (
                    latents,
                    latents[-1:].expand(len(padded_lengths) - len(group), -1),
                ),
            )
            matches = context.codec.reconstruction_matches(
                padded_latents,
                byte_values,
                valid_mask,
            )
            neural_rows += encoded.neural_rows + len(padded_lengths)
        group_results.append((group, latents[: len(group)]))
        match_batches.append(matches[: len(group)])

    valid_spans: dict[int, dict[int, Tensor]] = {}
    matches = torch.cat(match_batches).tolist()
    match_index = 0
    for group, latents in group_results:
        for candidate_index, latent in zip(group, latents, strict=True):
            if matches[match_index]:
                valid_spans.setdefault(offsets[candidate_index], {})[lengths[candidate_index]] = latent
            match_index += 1
    return _WindowValidation(valid_spans, len(lengths), neural_rows)


def _next_frontier(frontier: int, accepted_length: int) -> int:
    if accepted_length <= 2:
        return min(frontier * 2, SEGMENTATION_FRONTIERS[-1])
    if accepted_length >= LONG_ACCEPTED_SPAN:
        return SEGMENTATION_FRONTIERS[0]
    return frontier


def segment_bytes(
    codec: SpanCodec,
    data: bytes,
    *,
    max_candidate_bytes: int = DYNAMIC_SEGMENTATION_MAX_BYTES,
    cache: EncodingCache | None = None,
    namespace: str = "uncached",
) -> SegmentationResult:
    if max_candidate_bytes < 2:
        raise ValueError("max_candidate_bytes must be at least 2")
    result: list[EncodedSpan] = []
    invalid_by_length: Counter[int] = Counter()
    candidate_lengths: Counter[int] = Counter()
    candidate_count = 0
    valid_count = 0
    neural_rows = 0
    speculative_rows = 0
    position = 0
    frontier = SEGMENTATION_FRONTIERS[0]
    candidate_limit = min(max_candidate_bytes, codec.max_span)
    usable_cache = _usable_cache(codec, cache)
    data_values = (
        torch.frombuffer(bytearray(data), dtype=torch.uint8).to(
            device=codec.device,
            dtype=torch.long,
        )
        if data
        else torch.empty(0, dtype=torch.long, device=codec.device)
    )
    context = _SegmentationContext(
        codec,
        data,
        data_values,
        candidate_limit,
        usable_cache,
        namespace,
    )

    while position < len(data):
        window_end = min(position + frontier, len(data))
        validation = _window_valid_spans(
            context,
            position,
            window_end,
            frontier=frontier,
        )
        neural_rows += validation.neural_rows
        reached_rows = 0
        last_accepted_length = 0

        while position < window_end:
            maximum_length = min(candidate_limit, len(data) - position)
            spans_at_position = validation.valid_spans.get(position, {})
            candidates_at_position = max(maximum_length - 1, 0)
            candidate_count += candidates_at_position
            reached_rows += candidates_at_position
            valid_count += len(spans_at_position)
            for length in range(2, maximum_length + 1):
                candidate_lengths[length] += 1
                if length not in spans_at_position:
                    invalid_by_length[length] += 1

            if spans_at_position:
                selected_length = max(spans_at_position)
                selected = EncodedSpan(
                    data[position : position + selected_length],
                    spans_at_position[selected_length].clone(),
                    atomic=False,
                )
            else:
                atomic = data[position : position + 1]
                selected = EncodedSpan(
                    atomic,
                    codec.atomic_latent(atomic[0]),
                    atomic=True,
                )
            result.append(selected)
            position += len(selected.data)
            last_accepted_length = len(selected.data)
        frontier = _next_frontier(frontier, last_accepted_length)
        speculative_rows += validation.candidate_rows - reached_rows
    span_lengths = Counter(len(span.data) for span in result)
    return SegmentationResult(
        tuple(result),
        SegmentationStats(
            candidates=candidate_count,
            candidate_lengths=dict(sorted(candidate_lengths.items())),
            valid_candidates=valid_count,
            invalid_by_length=dict(sorted(invalid_by_length.items())),
            span_lengths=dict(sorted(span_lengths.items())),
            atomic_spans=sum(span.atomic for span in result),
        ),
        SegmentationWork(
            logical_candidates=candidate_count,
            neural_candidate_rows=neural_rows,
            speculative_discarded_rows=speculative_rows,
        ),
    )


def greedy_segment(
    codec: SpanCodec,
    data: bytes,
    *,
    max_candidate_bytes: int = DYNAMIC_SEGMENTATION_MAX_BYTES,
    cache: EncodingCache | None = None,
    namespace: str = "uncached",
) -> list[EncodedSpan]:
    return list(
        segment_bytes(
            codec,
            data,
            max_candidate_bytes=max_candidate_bytes,
            cache=cache,
            namespace=namespace,
        ).spans
    )


def reconstruct(segments: Sequence[EncodedSpan]) -> bytes:
    return b"".join(segment.data for segment in segments)
