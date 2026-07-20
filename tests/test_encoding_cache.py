from __future__ import annotations

import unittest

import torch

from continuous_tokenizer.codec.batches import (
    build_span_batch,
    span_bucket_width,
)
from continuous_tokenizer.codec.encoding_cache import CacheKey, EncodingCache
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.input.segmentation import encode_spans


class CountingCodec(InputByteCodec):
    def __init__(self) -> None:
        config = InputByteCodecConfig(8, 8, 8, 4, 4, 16, 1, 1)
        super().__init__(config, torch.randn((256, 8)))
        self.encode_calls = 0
        self.encoded_rows = 0

    def encode(self, byte_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        self.encoded_rows += byte_values.shape[0]
        return super().encode(byte_values, valid_mask)


def test_cache_returns_bit_identical_latents_without_reencoding() -> None:
    codec = CountingCodec().eval()
    cache = EncodingCache(1024 * 1024)

    with torch.inference_mode():
        first = encode_spans(codec, [b"ab", b"cd"], cache=cache, namespace="test")
        second = encode_spans(codec, [b"ab", b"cd"], cache=cache, namespace="test")

    assert torch.equal(first, second)
    assert codec.encode_calls == 1
    assert cache.info().hits == 2


def test_atomic_bytes_bypass_encoding_cache() -> None:
    codec = CountingCodec().eval()
    cache = EncodingCache(1024 * 1024)

    with torch.inference_mode():
        first = encode_spans(codec, [b"a", b"ab"], cache=cache, namespace="test")
        second = encode_spans(codec, [b"a", b"ab"], cache=cache, namespace="test")

    assert torch.equal(first, second)
    assert torch.equal(first[0], codec.atomic_latent(ord("a")))
    assert codec.encode_calls == 1
    assert cache.info().entries == 1
    assert cache.info().hits == 1


def test_cache_isolates_different_numerical_batch_shapes() -> None:
    codec = CountingCodec().eval()
    cache = EncodingCache(1024 * 1024)

    with torch.inference_mode():
        first = encode_spans(codec, [b"ab"], cache=cache, namespace="test")
        mixed = encode_spans(codec, [b"ab", b"cd"], cache=cache, namespace="test")
        expected = encode_spans(codec, [b"ab", b"cd"])

    assert codec.encode_calls == 3
    assert torch.equal(mixed, expected)
    assert torch.equal(mixed[0], first[0])
    assert cache.info().hits == 0
    assert cache.info().entries == 3


def test_cache_coalesces_repeated_batch_misses() -> None:
    codec = CountingCodec().eval()
    cache = EncodingCache(1024 * 1024)

    with torch.inference_mode():
        encoded = encode_spans(
            codec,
            [b"ab", b"ab", b"cd", b"ab"],
            cache=cache,
            namespace="test",
        )

    stats = cache.info()
    assert codec.encode_calls == 1
    assert codec.encoded_rows == 4
    assert torch.equal(encoded[0], encoded[1])
    assert torch.equal(encoded[0], encoded[3])
    assert stats.misses == 2
    assert stats.coalesced == 2
    assert stats.stores == 2


def test_codec_state_changes_clear_its_cache() -> None:
    codec = CountingCodec().eval()
    key = CacheKey("test", str(codec.byte_embeddings.dtype), b"ab", 1, 2)
    cache = codec.encoding_cache
    cache.put(key, torch.ones(8))

    codec.train()
    assert cache.info().entries == 0
    assert cache.info().stores == 0

    codec.eval()
    cache.put(key, torch.ones(8))
    codec.to(dtype=torch.float64)
    assert cache.info().entries == 0

    cache.put(CacheKey("test", str(torch.float64), b"ab", 1, 2), torch.ones(8))
    codec.load_state_dict(codec.state_dict())
    assert cache.info().entries == 0


def test_projected_byte_table_is_derived_exact_and_invalidated() -> None:
    codec = CountingCodec().eval()
    batch = build_span_batch([b"ab", b"abc"], max_span=codec.max_span, device=codec.device)
    state_keys = tuple(codec.state_dict())
    fingerprint = codec.encoder_fingerprint()
    projection_calls = 0

    def count_projection(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal projection_calls
        projection_calls += 1

    handle = codec.input_projection.register_forward_hook(count_projection)
    try:
        with torch.inference_mode():
            first = codec.encode(batch.byte_values, batch.valid_mask)
            second = codec.encode(batch.byte_values, batch.valid_mask)
        assert torch.equal(first, second)
        assert projection_calls == 1
        assert codec.projected_byte_table_cached
        assert tuple(codec.state_dict()) == state_keys
        assert codec.encoder_fingerprint() == fingerprint

        with torch.no_grad():
            codec.input_projection.weight.add_(1)
        with torch.inference_mode():
            codec.encode(batch.byte_values, batch.valid_mask)
        assert projection_calls == 2

        codec.to(dtype=torch.float64)
        assert not codec.projected_byte_table_cached
        with torch.inference_mode():
            codec.encode(batch.byte_values, batch.valid_mask)
        assert codec.projected_byte_table_cached

        codec.load_state_dict(codec.state_dict())
        assert not codec.projected_byte_table_cached
        with torch.inference_mode():
            codec.encode(batch.byte_values, batch.valid_mask)
        codec.set_trainable_components(encoder=False, decoder=False)
        assert not codec.projected_byte_table_cached

        with torch.inference_mode():
            codec.encode(batch.byte_values, batch.valid_mask)
        codec.clear_runtime_caches()
        assert not codec.projected_byte_table_cached
    finally:
        handle.remove()


def test_training_recomputes_projected_byte_table_with_gradients() -> None:
    codec = CountingCodec().train()
    batch = build_span_batch([b"ab", b"abc"], max_span=codec.max_span, device=codec.device)
    projection_calls = 0

    def count_projection(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal projection_calls
        projection_calls += 1

    handle = codec.input_projection.register_forward_hook(count_projection)
    try:
        codec.encode(batch.byte_values, batch.valid_mask).sum().backward()
        codec.zero_grad(set_to_none=True)
        codec.encode(batch.byte_values, batch.valid_mask).sum().backward()
    finally:
        handle.remove()

    assert projection_calls == 2
    assert codec.input_projection.weight.grad is not None
    assert not codec.projected_byte_table_cached


def test_cache_respects_byte_budget() -> None:
    cache = EncodingCache(max_bytes=16)
    first = CacheKey("n", "torch.float32", b"a", 1, 1)
    second = CacheKey("n", "torch.float32", b"b", 1, 1)
    cache.put(first, torch.ones(4))
    cache.put(second, torch.ones(4))

    stats = cache.info()
    assert stats.entries == 1
    assert stats.tensor_bytes == 16
    assert stats.stores == 2
    assert stats.evictions == 1
    assert cache.get(first, device=torch.device("cpu"), dtype=torch.float32) is None
    assert cache.get(second, device=torch.device("cpu"), dtype=torch.float32) is not None


def test_cache_evicts_the_least_recently_used_entry() -> None:
    cache = EncodingCache(max_bytes=32)
    keys = [CacheKey("n", "torch.float32", bytes([value]), 1, 1) for value in range(3)]
    for key in keys[:2]:
        cache.put(key, torch.ones(4))
    assert cache.get(keys[0], device=torch.device("cpu"), dtype=torch.float32) is not None

    before = cache.info()
    cache.put(keys[2], torch.ones(4))
    activity = cache.info().since(before)

    assert cache.get(keys[0], device=torch.device("cpu"), dtype=torch.float32) is not None
    assert cache.get(keys[1], device=torch.device("cpu"), dtype=torch.float32) is None
    assert cache.get(keys[2], device=torch.device("cpu"), dtype=torch.float32) is not None
    assert activity.entries == 2
    assert activity.tensor_bytes == 32
    assert activity.stores == 1
    assert activity.evictions == 1


def test_build_batch_keeps_input_immutable() -> None:
    spans = [b"abc"]
    original = spans.copy()
    build_span_batch(spans, max_span=4, device=torch.device("cpu"))
    assert spans == original


def test_span_batches_use_bounded_power_of_two_widths() -> None:
    widths = {
        build_span_batch(
            [bytes(length)],
            max_span=64,
            device=torch.device("cpu"),
        ).byte_values.shape[1]
        for length in range(1, 65)
    }

    assert widths == {1, 2, 4, 8, 16, 32, 64}
    assert span_bucket_width(64, max_span=64) == 64


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_cache_returns_bit_identical_latents_without_reencoding,
            test_atomic_bytes_bypass_encoding_cache,
            test_cache_isolates_different_numerical_batch_shapes,
            test_cache_coalesces_repeated_batch_misses,
            test_codec_state_changes_clear_its_cache,
            test_projected_byte_table_is_derived_exact_and_invalidated,
            test_training_recomputes_projected_byte_table_with_gradients,
            test_cache_respects_byte_budget,
            test_cache_evicts_the_least_recently_used_entry,
            test_build_batch_keeps_input_immutable,
            test_span_batches_use_bounded_power_of_two_widths,
        )
    )
