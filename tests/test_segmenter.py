from __future__ import annotations

import torch

from continuous_tokenizer.cache import EncodingCache
from continuous_tokenizer.segmenter import greedy_segment, reconstruct


class MappingCodec:
    def __init__(self, valid: set[bytes]) -> None:
        self.max_span = 4
        self.training = False
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.valid = valid
        self._values: dict[int, bytes] = {}
        self.encode_calls = 0

    def encode(self, byte_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        rows: list[list[float]] = []
        for values, mask in zip(byte_values, valid_mask, strict=True):
            span = bytes(values[mask].tolist())
            identifier = len(self._values) + 1
            self._values[identifier] = span
            rows.append([float(identifier)])
        return torch.tensor(rows)

    def decode_greedy(self, latent: torch.Tensor) -> list[bytes | None]:
        results: list[bytes | None] = []
        for row in latent:
            span = self._values[int(row[0].item())]
            results.append(span if span in self.valid else None)
        return results


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
    assert reconstruct(segments) == data


def test_cache_does_not_change_segmentation() -> None:
    data = b"abcdabcd"
    codec = MappingCodec({b"abc"})

    with torch.inference_mode():
        uncached = greedy_segment(codec, data)
        cache = EncodingCache()
        cold = greedy_segment(codec, data, cache=cache, namespace="test")
        warm = greedy_segment(codec, data, cache=cache, namespace="test")

    expected = [segment.data for segment in uncached]
    assert [segment.data for segment in cold] == expected
    assert [segment.data for segment in warm] == expected
