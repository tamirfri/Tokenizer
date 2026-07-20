from __future__ import annotations

import unittest
from collections import Counter
from random import Random
from zlib import crc32

import torch

from continuous_tokenizer.codec.constants import CODEC_EOS
from continuous_tokenizer.codec.encoding_cache import EncodingCache
from continuous_tokenizer.input.evidence import segmentation_evidence
from continuous_tokenizer.input.segmentation import (
    SegmentationStats,
    encode_spans,
    greedy_segment,
    reconstruct,
    segment_bytes,
    validate_spans,
)


class MappingCodec:
    def __init__(self, valid: set[bytes]) -> None:
        self.max_span = 4
        self.training = False
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.valid = valid
        self._values: dict[int, bytes] = {}
        self.encode_calls = 0
        self.encoded_rows = 0
        self.encode_widths: list[int] = []
        self.encode_batch_sizes: list[int] = []

    def encode(self, byte_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        self.encoded_rows += byte_values.shape[0]
        self.encode_widths.append(byte_values.shape[1])
        self.encode_batch_sizes.append(byte_values.shape[0])
        rows: list[list[float]] = []
        for values, mask in zip(byte_values, valid_mask, strict=True):
            span = bytes(values[mask].tolist())
            identifier = (len(span) << 32) | crc32(span)
            self._values[identifier] = span
            rows.append([float(identifier)])
        return torch.tensor(rows, dtype=self.dtype)

    def atomic_latent(self, value: int) -> torch.Tensor:
        return torch.tensor([float(value)], dtype=self.dtype)

    def reconstruction_matches(
        self,
        latent: torch.Tensor,
        byte_values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        matches = []
        for row, values, mask in zip(latent, byte_values, valid_mask, strict=True):
            encoded = self._values[int(row[0].item())]
            expected = bytes(values[mask].tolist())
            matches.append(encoded == expected and encoded in self.valid)
        return torch.tensor(matches)

    def encode_and_reconstruction_matches(
        self,
        byte_values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(byte_values, valid_mask)
        return latent, self.reconstruction_matches(latent, byte_values, valid_mask)


class BatchSensitiveCodec(MappingCodec):
    byte_embeddings = torch.empty(0, dtype=torch.float64)

    def encode(self, byte_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        latent = super().encode(byte_values, valid_mask)
        return latent + byte_values.shape[0] / 1_000

    def decode_logits(
        self,
        latent: torch.Tensor,
        output_positions: int | None = None,
    ) -> torch.Tensor:
        positions = self.max_span + 1 if output_positions is None else output_positions
        logits = torch.full(
            (latent.shape[0], positions, CODEC_EOS + 1),
            -1.0,
            dtype=self.dtype,
        )
        for row, value in enumerate(latent):
            span = self._values[int(value[0].item())]
            targets = (*span, CODEC_EOS)
            for position, target in enumerate(targets):
                logits[row, position, target] = 1.0
        return logits


def test_longer_valid_span_survives_shorter_invalid_span() -> None:
    codec = MappingCodec({b"abc"})

    segments = greedy_segment(codec, b"abcd")

    assert [segment.data for segment in segments] == [b"abc", b"d"]
    assert reconstruct(segments) == b"abcd"


def test_longest_of_all_valid_candidates_is_selected() -> None:
    codec = MappingCodec({b"ab", b"abc", b"abcd"})

    segments = greedy_segment(codec, b"abcd")

    assert [segment.data for segment in segments] == [b"abcd"]


def test_atomic_fallback_round_trips_all_byte_values() -> None:
    data = bytes(range(256))
    codec = MappingCodec(set())

    segments = greedy_segment(codec, data)

    assert all(segment.atomic for segment in segments)
    assert [int(segment.latent.item()) for segment in segments] == list(range(256))
    assert reconstruct(segments) == data


def test_cache_does_not_change_segmentation() -> None:
    data = b"abcdabcd\x00\xff"
    valid = {b"abc"}

    with torch.inference_mode():
        uncached = segment_bytes(MappingCodec(valid), data)
        codec = MappingCodec(valid)
        cache = EncodingCache()
        cold = segment_bytes(codec, data, cache=cache, namespace="test")
        warm = segment_bytes(codec, data, cache=cache, namespace="test")

    expected = [(span.data, span.atomic) for span in uncached.spans]
    assert [(span.data, span.atomic) for span in cold.spans] == expected
    assert [(span.data, span.atomic) for span in warm.spans] == expected
    assert cold.stats == uncached.stats == warm.stats
    assert all(torch.equal(actual.latent, expected.latent) for actual, expected in zip(cold.spans, uncached.spans, strict=True))
    assert all(torch.equal(actual.latent, expected.latent) for actual, expected in zip(warm.spans, uncached.spans, strict=True))


def test_cache_keys_include_the_numerical_encoding_shape() -> None:
    windows = (b"abcdefghde", b"abcdefde")
    valid = {data[offset : offset + 2] for data in windows for offset in range(len(data) - 1)}
    codec = BatchSensitiveCodec(valid)

    with torch.inference_mode():
        disabled = tuple(segment_bytes(codec, data) for data in windows)
        cache = EncodingCache()
        cold = tuple(segment_bytes(codec, data, cache=cache, namespace="test") for data in windows)
        warm = tuple(segment_bytes(codec, data, cache=cache, namespace="test") for data in windows)

    for data, expected, cold_result, warm_result in zip(
        windows,
        disabled,
        cold,
        warm,
        strict=True,
    ):
        for actual in (cold_result, warm_result):
            assert actual.stats == expected.stats
            assert [span.data for span in actual.spans] == [span.data for span in expected.spans]
            assert all(
                torch.equal(actual_span.latent, expected_span.latent)
                for actual_span, expected_span in zip(
                    actual.spans,
                    expected.spans,
                    strict=True,
                )
            )
            assert (
                segmentation_evidence(
                    codec,
                    actual.spans,
                    data,
                    source_dtype=codec.dtype,
                ).semantic_sha256
                == segmentation_evidence(
                    codec,
                    expected.spans,
                    data,
                    source_dtype=codec.dtype,
                ).semantic_sha256
            )


def test_cold_cache_coalesces_repeated_candidate_entries_per_shape() -> None:
    data = b"aaaaaaaa"
    codec = MappingCodec(set())
    codec.max_span = 4

    with torch.inference_mode():
        cache = EncodingCache()
        cached = segment_bytes(
            codec,
            data,
            cache=cache,
            namespace="test",
        )

    assert reconstruct(cached.spans) == data
    assert cache.info().entries == 9
    assert cache.info().coalesced == 9


def test_cached_encode_uses_power_of_two_rows_without_caching_padding() -> None:
    codec = MappingCodec(set())
    cache = EncodingCache()

    with torch.inference_mode():
        encoded = encode_spans(
            codec,
            [b"ab", b"abc", b"abcd"],
            cache=cache,
            namespace="test",
        )

    assert encoded.shape == (3, 1)
    assert codec.encode_batch_sizes == [4]
    assert cache.info().entries == 3


def test_cached_validation_keeps_bounded_rows_and_fixed_width() -> None:
    codec = MappingCodec({b"ab"})
    cache = EncodingCache()

    with torch.inference_mode():
        latents, matches = validate_spans(
            codec,
            [b"ab", b"abc"],
            static_size=4,
            cache=cache,
            namespace="test",
        )

    assert latents.shape == (2, 1)
    assert matches.tolist() == [True, False]
    assert codec.encode_batch_sizes == [2]
    assert codec.encode_widths == [4]
    assert cache.info().entries == 2


def test_aligned_validation_uses_bounded_rows_and_fixed_width() -> None:
    codec = MappingCodec(set())
    codec.max_span = 64

    for count in (1, 2, 3, 5, 9, 17, 33, 64):
        spans = [bytes([index]) * (index % 63 + 1) for index in range(count)]
        validate_spans(codec, spans, static_size=64)
        expected_rows = 1 << (count - 1).bit_length()
        assert codec.encode_batch_sizes[-1] == expected_rows
        assert codec.encode_widths[-1] == 64


def test_segmentation_reports_candidates_without_affecting_selection() -> None:
    codec = MappingCodec({b"ab", b"abcd"})

    result = segment_bytes(codec, b"abcd")

    assert [span.data for span in result.spans] == [b"abcd"]
    assert result.stats.candidates == 3
    assert result.stats.valid_candidates == 2
    assert result.stats.invalid_by_length == {3: 1}
    assert result.stats.span_lengths == {4: 1}
    assert result.work.logical_candidate_rows == result.stats.candidates
    assert result.work.neural_padded_rows == 3
    assert result.work.speculative_discarded_rows == 0


def test_segmentation_batches_candidates_by_bounded_width() -> None:
    codec = MappingCodec(set())
    codec.max_span = 64

    segment_bytes(codec, bytes(range(64)))

    assert codec.encode_widths[:5] == [2, 4, 8, 16, 32]
    assert codec.encode_batch_sizes[:5] == [1, 2, 4, 8, 16]
    assert codec.encode_batch_sizes[5:10] == [2, 4, 8, 16, 32]
    assert [rows for width, rows in zip(codec.encode_widths, codec.encode_batch_sizes, strict=True) if width == 2][:4] == [1, 2, 4, 8]
    codec.encode_widths.clear()
    codec.encode_batch_sizes.clear()

    with torch.inference_mode():
        segment_bytes(codec, bytes(range(64)), cache=EncodingCache(), namespace="test")

    assert codec.encode_widths[:5] == [2, 4, 8, 16, 32]
    assert codec.encode_batch_sizes[:5] == [1, 2, 4, 8, 16]
    assert codec.encode_batch_sizes[5:10] == [2, 4, 8, 16, 32]


def _scalar_segmentation(data: bytes, valid: set[bytes], max_span: int) -> tuple[list[bytes], SegmentationStats]:
    spans: list[bytes] = []
    candidates = 0
    valid_candidates = 0
    atomic_spans = 0
    candidate_lengths: Counter[int] = Counter()
    invalid_by_length: Counter[int] = Counter()
    position = 0
    while position < len(data):
        maximum_length = min(max_span, len(data) - position)
        selected = data[position : position + 1]
        for length in range(2, maximum_length + 1):
            candidate = data[position : position + length]
            candidates += 1
            candidate_lengths[length] += 1
            if candidate in valid:
                valid_candidates += 1
                selected = candidate
            else:
                invalid_by_length[length] += 1
        atomic_spans += len(selected) == 1
        spans.append(selected)
        position += len(selected)
    return spans, SegmentationStats(
        candidates=candidates,
        candidate_lengths=dict(sorted(candidate_lengths.items())),
        valid_candidates=valid_candidates,
        invalid_by_length=dict(sorted(invalid_by_length.items())),
        span_lengths=dict(sorted(Counter(map(len, spans)).items())),
        atomic_spans=atomic_spans,
    )


def test_adaptive_frontier_matches_scalar_exhaustive_oracle() -> None:
    data = bytes(range(24))
    random = Random(17)
    validity_sets = [
        set(),
        {data[:8]},
        {data[offset : offset + length] for offset in range(len(data)) for length in range(2, min(8, len(data) - offset) + 1) if (offset + length) % 2 == 0},
        {data[offset : offset + length] for offset in range(len(data)) for length in range(2, min(8, len(data) - offset) + 1) if random.choice((False, True))},
    ]

    for valid in validity_sets:
        codec = MappingCodec(valid)
        codec.max_span = 8
        result = segment_bytes(codec, data, max_candidate_bytes=8)
        expected_spans, expected_stats = _scalar_segmentation(data, valid, 8)
        assert [span.data for span in result.spans] == expected_spans
        assert result.stats == expected_stats
        assert reconstruct(result.spans) == data


def test_default_segmentation_selects_longest_valid_span_up_to_32_bytes() -> None:
    data = bytes(range(40))
    codec = MappingCodec({data[:31], data[:32], data[:33]})
    codec.max_span = 64

    result = segment_bytes(codec, data)

    assert result.spans[0].data == data[:32]
    assert reconstruct(result.spans) == data
    assert max(result.stats.candidate_lengths) == 32
    assert 33 not in result.stats.candidate_lengths


def test_frontier_expands_then_discards_unreachable_speculation() -> None:
    data = bytes(range(16))
    codec = MappingCodec({data[1:9]})
    codec.max_span = 8

    result = segment_bytes(codec, data, max_candidate_bytes=8)

    assert [span.data for span in result.spans[:2]] == [data[:1], data[1:9]]
    assert result.work.speculative_discarded_rows == 7
    assert result.work.logical_candidate_rows == result.stats.candidates


def test_selected_latents_do_not_retain_candidate_batches() -> None:
    data = b"abcdefgh"
    valid = {data[offset : offset + length] for offset in range(len(data)) for length in range(2, min(4, len(data) - offset) + 1)}
    codec = MappingCodec(valid)

    result = segment_bytes(codec, data)

    assert [span.data for span in result.spans] == [b"abcd", b"efgh"]
    assert all(span.latent.untyped_storage().nbytes() == span.latent.numel() * span.latent.element_size() for span in result.spans)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_longer_valid_span_survives_shorter_invalid_span,
            test_longest_of_all_valid_candidates_is_selected,
            test_atomic_fallback_round_trips_all_byte_values,
            test_cache_does_not_change_segmentation,
            test_cache_keys_include_the_numerical_encoding_shape,
            test_cold_cache_coalesces_repeated_candidate_entries_per_shape,
            test_cached_encode_uses_power_of_two_rows_without_caching_padding,
            test_cached_validation_keeps_bounded_rows_and_fixed_width,
            test_aligned_validation_uses_bounded_rows_and_fixed_width,
            test_segmentation_reports_candidates_without_affecting_selection,
            test_segmentation_batches_candidates_by_bounded_width,
            test_adaptive_frontier_matches_scalar_exhaustive_oracle,
            test_default_segmentation_selects_longest_valid_span_up_to_32_bytes,
            test_frontier_expands_then_discards_unreachable_speculation,
            test_selected_latents_do_not_retain_candidate_batches,
        )
    )
