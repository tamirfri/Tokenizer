from __future__ import annotations

import torch

from continuous_tokenizer.batching import build_span_batch
from continuous_tokenizer.cache import CacheKey, EncodingCache
from continuous_tokenizer.codec import CodecConfig, ContinuousByteCodec
from continuous_tokenizer.segmenter import encode_spans


class CountingCodec(ContinuousByteCodec):
    def __init__(self) -> None:
        config = CodecConfig(8, 8, 4, 2, 16, 1, 1)
        super().__init__(config, torch.randn((256, 8)))
        self.encode_calls = 0

    def encode(self, byte_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
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


def test_codec_state_changes_clear_its_cache() -> None:
    codec = CountingCodec().eval()
    key = CacheKey("test", str(codec.byte_embeddings.dtype), b"ab")
    cache = codec.encoding_cache
    cache.put(key, torch.ones(8))

    codec.train()
    assert cache.info().entries == 0

    codec.eval()
    cache.put(key, torch.ones(8))
    codec.to(dtype=torch.float64)
    assert cache.info().entries == 0

    cache.put(CacheKey("test", str(torch.float64), b"ab"), torch.ones(8))
    codec.load_state_dict(codec.state_dict())
    assert cache.info().entries == 0


def test_cache_respects_byte_budget() -> None:
    cache = EncodingCache(max_bytes=16)
    first = CacheKey("n", "torch.float32", b"a")
    second = CacheKey("n", "torch.float32", b"b")
    cache.put(first, torch.ones(4))
    cache.put(second, torch.ones(4))

    assert cache.info().entries == 1
    assert cache.info().bytes == 16


def test_build_batch_keeps_input_immutable() -> None:
    spans = [b"abc"]
    original = spans.copy()
    build_span_batch(spans, max_span=4, device=torch.device("cpu"))
    assert spans == original
