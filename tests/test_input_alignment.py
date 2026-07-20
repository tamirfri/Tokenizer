from __future__ import annotations

import unittest

import torch
from input_model_fixtures import InvalidSpanCodec, make_adapter

from continuous_tokenizer.codec.encoding_cache import EncodingCache
from continuous_tokenizer.codec.input import InputByteCodecConfig
from continuous_tokenizer.input.adapter import (
    ByteRun,
    ControlToken,
    InputEmbeddingAdapter,
)
from continuous_tokenizer.input.segmentation import EncodedSpan, reconstruct


class MergeCodec(InvalidSpanCodec):
    def encode(self, byte_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        latent = torch.zeros((byte_values.shape[0], 8), device=byte_values.device)
        lengths = valid_mask.sum(dim=1)
        latent[:, 0] = lengths
        latent[:, 1:3] = byte_values[:, :2]
        single = lengths == 1
        return torch.where(
            single[:, None],
            self.byte_embeddings[byte_values[:, 0]],
            latent,
        )

    def reconstruction_matches(
        self,
        latent: torch.Tensor,
        byte_values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del byte_values, valid_mask
        return (latent[:, 0] == 2) & (latent[:, 1] == 65) & (latent[:, 2] == 66)

    def decode_greedy(
        self,
        latent: torch.Tensor,
        *,
        maximum_length: int | None = None,
    ) -> list[bytes | None]:
        del maximum_length
        return [b"AB" if int(row[0].item()) == 2 and row[1:3].tolist() == [65, 66] else None for row in latent]


class ShapeTrackingMergeCodec(MergeCodec):
    def __init__(self, config: InputByteCodecConfig, byte_embeddings: torch.Tensor) -> None:
        super().__init__(config, byte_embeddings)
        self.encode_shapes: list[tuple[int, ...]] = []
        self.validation_shapes: list[tuple[int, ...]] = []

    def encode(self, byte_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        self.encode_shapes.append(tuple(byte_values.shape))
        return super().encode(byte_values, valid_mask)

    def reconstruction_matches(
        self,
        latent: torch.Tensor,
        byte_values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.validation_shapes.append(tuple(byte_values.shape))
        return super().reconstruction_matches(latent, byte_values, valid_mask)


def test_control_tokens_bypass_codec_with_exact_embedding() -> None:
    adapter = make_adapter()

    with torch.inference_mode():
        encoding = adapter.encode_compatibility((65, 256, 66))

    assert isinstance(encoding.positions[0], EncodedSpan)
    assert encoding.positions[1] == ControlToken(256)
    assert torch.equal(encoding.embeddings[0], adapter.codec.byte_embeddings[65])
    assert torch.equal(encoding.embeddings[1], adapter.control_embeddings[0])
    assert torch.equal(encoding.embeddings[2], adapter.codec.byte_embeddings[66])


def test_span_metadata_reuses_model_embedding_storage() -> None:
    adapter = make_adapter()

    with torch.inference_mode():
        encoding = adapter.encode_compatibility((65, 256, 66))

    storage = encoding.embeddings.untyped_storage().data_ptr()
    assert all(position.latent.untyped_storage().data_ptr() == storage for position in encoding.positions if isinstance(position, EncodedSpan))


def test_token_ids_become_byte_runs_separated_by_controls() -> None:
    adapter = make_adapter()

    pieces = adapter.pieces_from_token_ids((65, 66, 256, 67))

    assert pieces == (ByteRun(b"AB"), ControlToken(256), ByteRun(b"C"))


def test_segmented_position_ids_preserve_native_lineage() -> None:
    adapter = make_adapter()

    with torch.inference_mode():
        compatibility = adapter.encode_token_ids((65, 256, 66), mode="compatibility")
        segmented = adapter.encode_token_ids((65, 256, 66), mode="segmented", alignment="arbitrary")

    assert compatibility.position_ids.tolist() == [0, 1, 2]
    assert segmented.position_ids.tolist() == [0, 1, 2]


def test_aligned_merge_uses_final_native_position() -> None:
    original = make_adapter()
    codec = MergeCodec(original.codec.config, original.codec.byte_embeddings).eval()
    adapter = InputEmbeddingAdapter(
        codec,
        original.vocabulary,
        original.control_ids,
        original.control_embeddings,
        namespace="aligned-test",
    ).eval()

    with torch.inference_mode():
        encoding = adapter.encode_token_ids((65, 66, 256, 67), mode="segmented", alignment="aligned")

    assert encoding.position_ids.tolist() == [1, 2, 3]
    assert isinstance(encoding.positions[0], EncodedSpan)
    assert encoding.positions[0].data == b"AB"


def test_aligned_validation_uses_static_shapes_and_warm_cache() -> None:
    original = make_adapter()
    codec = ShapeTrackingMergeCodec(
        original.codec.config,
        original.codec.byte_embeddings,
    ).eval()
    adapter = InputEmbeddingAdapter(
        codec,
        original.vocabulary,
        original.control_ids,
        original.control_embeddings,
        namespace="aligned-static-test",
    ).eval()
    cache = EncodingCache()

    with torch.inference_mode():
        first = adapter.encode_token_ids(
            (65, 66, 67),
            mode="segmented",
            alignment="aligned",
            cache=cache,
        )
        second = adapter.encode_token_ids(
            (65, 66, 67),
            mode="segmented",
            alignment="aligned",
            cache=cache,
        )

    assert isinstance(first.positions[0], EncodedSpan)
    assert isinstance(second.positions[0], EncodedSpan)
    assert first.positions[0].data == b"AB"
    assert second.positions[0].data == b"AB"
    assert codec.encode_shapes == [(2, codec.max_span)]
    assert codec.validation_shapes == [(2, codec.max_span), (2, codec.max_span)]


def test_arbitrary_bytes_round_trip_through_atomic_fallback() -> None:
    adapter = make_adapter()
    fixtures = (
        bytes(range(256)),
        "שלום".encode(),
        "hello".encode("utf-16-le"),
        "hello".encode("utf-32-be"),
        b"\x00\xff\x80invalid",
    )

    with torch.inference_mode():
        for fixture in fixtures:
            encoding = adapter.encode_bytes(fixture)
            spans = [position for position in encoding.positions if isinstance(position, EncodedSpan)]
            assert reconstruct(spans) == fixture
            assert encoding.atomic_spans == len(fixture)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_control_tokens_bypass_codec_with_exact_embedding,
            test_span_metadata_reuses_model_embedding_storage,
            test_token_ids_become_byte_runs_separated_by_controls,
            test_segmented_position_ids_preserve_native_lineage,
            test_aligned_merge_uses_final_native_position,
            test_aligned_validation_uses_static_shapes_and_warm_cache,
            test_arbitrary_bytes_round_trip_through_atomic_fallback,
        )
    )
