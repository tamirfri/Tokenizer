from __future__ import annotations

import unittest

import torch
from codec_fixtures import make_codec
from torch import nn

from continuous_tokenizer.codec.batches import (
    build_span_batch,
    byte_reconstruction_loss,
)
from continuous_tokenizer.codec.compute import (
    input_encode_flops,
    input_validation_flops,
)
from continuous_tokenizer.codec.constants import CODEC_EOS
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.codec.layers import KEY_VALUE_HEADS, GroupedQuerySelfAttention
from continuous_tokenizer.contracts.profiles import PROFILES, PROJECTION_DIMENSION_CAP
from continuous_tokenizer.runtime.tensors import module_bytes


def test_single_bytes_use_exact_frozen_embeddings() -> None:
    codec = make_codec()
    spans = [bytes([value]) for value in range(256)]
    batch = build_span_batch(spans, max_span=codec.max_span, device=torch.device("cpu"))

    latent = codec.encode(batch.byte_values, batch.valid_mask)

    assert torch.equal(latent, codec.byte_embeddings)


def test_source_dtype_byte_rows_stay_exact() -> None:
    embeddings = torch.randn((256, 16)).to(torch.bfloat16)
    codec = InputByteCodec(
        InputByteCodecConfig(16, 8, 16, 4, 4, 16, 1, 1),
        embeddings,
    )
    batch = build_span_batch([b"a"], max_span=codec.max_span, device=codec.device)

    latent = codec.encode(batch.byte_values, batch.valid_mask)

    assert codec.byte_embeddings.dtype == torch.bfloat16
    assert latent.dtype == torch.float32
    assert torch.equal(latent.to(torch.bfloat16), embeddings[ord("a")].unsqueeze(0))


def test_forward_and_reconstruction_loss_are_finite() -> None:
    codec = make_codec()
    batch = build_span_batch([b"a", b"abc"], max_span=codec.max_span, device=codec.device)

    latent, logits = codec(batch.byte_values, batch.valid_mask)
    loss = byte_reconstruction_loss(logits, batch.framed_targets, batch.target_mask)

    assert latent.shape == (2, 16)
    assert logits.shape == (2, 5, 257)
    assert torch.isfinite(loss)
    assert batch.framed_targets[0, :2].tolist() == [ord("a"), CODEC_EOS]
    assert batch.framed_targets[1, :4].tolist() == [ord("a"), ord("b"), ord("c"), CODEC_EOS]


def test_codec_uses_mandatory_gqa_and_separate_projections() -> None:
    codec = make_codec()
    batch = build_span_batch([b"a", b"abc"], max_span=codec.max_span, device=codec.device)

    latent, logits = codec(batch.byte_values, batch.valid_mask)
    matches = codec.reconstruction_matches(latent, batch.byte_values, batch.valid_mask)
    validated_latent, validated_matches = codec.encode_and_reconstruction_matches(
        batch.byte_values,
        batch.valid_mask,
    )

    assert latent.shape == (2, 16)
    assert logits.shape == (2, 5, 257)
    assert matches.shape == (2,)
    assert matches.dtype == torch.bool
    assert torch.equal(validated_latent, latent)
    assert torch.equal(validated_matches, matches)
    attention = codec.encoder.layers[0].attention
    assert isinstance(attention, GroupedQuerySelfAttention)
    assert attention.query_heads == codec.config.query_heads
    assert attention.key_value_heads == KEY_VALUE_HEADS
    key_value_dimension = KEY_VALUE_HEADS * attention.head_dimension
    assert attention.query.weight.shape == (codec.config.local_dim, codec.config.local_dim)
    assert attention.key.weight.shape == (key_value_dimension, codec.config.local_dim)
    assert attention.value.weight.shape == (key_value_dimension, codec.config.local_dim)
    assert attention.output.weight.shape == (codec.config.local_dim, codec.config.local_dim)


def test_gqa_matches_explicit_key_value_repetition_and_gradients() -> None:
    attention = GroupedQuerySelfAttention(dimension=16, query_heads=4)
    value = torch.randn((2, 5, 16), requires_grad=True)
    valid_mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    actual = attention(value, valid_mask)
    batch_size, positions, dimension = value.shape
    query = attention.query(value).view(
        batch_size,
        positions,
        attention.query_heads,
        attention.head_dimension,
    )
    key = attention.key(value).view(
        batch_size,
        positions,
        attention.key_value_heads,
        attention.head_dimension,
    )
    projected_value = attention.value(value).view(
        batch_size,
        positions,
        attention.key_value_heads,
        attention.head_dimension,
    )
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    projected_value = projected_value.transpose(1, 2)
    groups = attention.query_heads // attention.key_value_heads
    key = key.repeat_interleave(groups, dim=1)
    projected_value = projected_value.repeat_interleave(groups, dim=1)
    expected = nn.functional.scaled_dot_product_attention(
        query,
        key,
        projected_value,
        attn_mask=valid_mask[:, None, None, :],
        dropout_p=0.0,
    )
    expected = attention.output(expected.transpose(1, 2).contiguous().view(batch_size, positions, dimension))

    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.square().sum(), value, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), value)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_encoder_baseline_matches_position_weighted_byte_embeddings() -> None:
    codec = make_codec()
    batch = build_span_batch([b"ab", b"abc"], max_span=codec.max_span, device=codec.device)
    embedded = nn.functional.embedding(batch.byte_values, codec.byte_embeddings)
    weights = torch.arange(1, batch.byte_values.shape[1] + 1) * batch.valid_mask
    expected = (embedded * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
        dim=1,
        keepdim=True,
    )

    actual = codec.encode(batch.byte_values, batch.valid_mask)

    torch.testing.assert_close(actual, expected)


def test_decoder_prefix_is_independent_of_later_queries() -> None:
    codec = make_codec()
    codec.eval()
    latent = torch.randn((2, codec.config.embedding_dim))

    full = codec.decode_logits(latent)
    prefix = codec.decode_logits(latent, 3)

    torch.testing.assert_close(prefix, full[:, :3])


def test_profiles_fit_target_deployment_budgets_without_allocating_weights() -> None:
    targets = (
        (1_024, 248_320, 276, 128),
        (640, 262_144, 1_000, 128),
    )
    for embedding_dim, vocabulary_size, controls, max_span in targets:
        source_bytes = vocabulary_size * embedding_dim * 2
        control_bytes = controls * (8 + embedding_dim * 2)
        candidate_reference_state_ratios: list[float] = []
        candidate_state_bytes: list[int] = []
        for profile in PROFILES:
            with torch.device("meta"):
                codec = InputByteCodec(
                    InputByteCodecConfig(
                        embedding_dim=embedding_dim,
                        local_dim=profile.local_dim,
                        projection_dim=profile.projection_dim(embedding_dim),
                        max_span=max(64, max_span),
                        query_heads=profile.query_heads,
                        feedforward_dim=profile.feedforward_dim,
                        encoder_layers=profile.encoder_layers,
                        decoder_layers=profile.decoder_layers,
                    ),
                    torch.empty((256, embedding_dim)),
                ).to(dtype=torch.bfloat16)
            candidate_state_bytes.append(module_bytes(codec) + control_bytes)
            candidate_reference_state_ratios.append(
                candidate_state_bytes[-1] / source_bytes,
            )

        assert candidate_reference_state_ratios[0] < candidate_reference_state_ratios[1] < 0.5
        assert candidate_state_bytes[1] >= candidate_state_bytes[0] * 3 // 2


def test_profile_shrink_reduces_parameters_state_and_analytical_flops() -> None:
    embedding_dim = 1_024
    old_profiles = (
        (320, 8, 3, 2, 10, 640),
        (1_024, 32, 4, 2, 16, 2_048),
    )
    for profile, old in zip(PROFILES, old_profiles, strict=True):
        old_local, old_multiplier, old_encoder_layers, old_decoder_layers, old_heads, old_feedforward = old
        old_config = InputByteCodecConfig(
            embedding_dim=embedding_dim,
            local_dim=old_local,
            projection_dim=min(PROJECTION_DIMENSION_CAP, old_multiplier * embedding_dim),
            max_span=64,
            query_heads=old_heads,
            feedforward_dim=old_feedforward,
            encoder_layers=old_encoder_layers,
            decoder_layers=old_decoder_layers,
        )
        new_config = InputByteCodecConfig(
            embedding_dim=embedding_dim,
            local_dim=profile.local_dim,
            projection_dim=profile.projection_dim(embedding_dim),
            max_span=32,
            query_heads=profile.query_heads,
            feedforward_dim=profile.feedforward_dim,
            encoder_layers=profile.encoder_layers,
            decoder_layers=profile.decoder_layers,
        )
        with torch.device("meta"):
            old_codec = InputByteCodec(old_config, torch.empty((256, embedding_dim)))
            new_codec = InputByteCodec(new_config, torch.empty((256, embedding_dim)))

        assert sum(parameter.numel() for parameter in new_codec.parameters()) < sum(parameter.numel() for parameter in old_codec.parameters())
        assert module_bytes(new_codec) < module_bytes(old_codec)
        assert input_encode_flops(new_config, 32) + input_validation_flops(
            new_config,
            32,
        ) < input_encode_flops(old_config, 64) + input_validation_flops(
            old_config,
            64,
        )


def test_decoder_trains_only_its_model_to_local_projection() -> None:
    codec = make_codec()

    codec.decode_logits(torch.randn((2, 16))).sum().backward()

    assert codec.decoder_projection.weight.grad is not None
    assert codec.decoder_projection.bias.grad is not None
    assert codec.input_projection.weight.grad is None
    assert codec.input_projection.bias.grad is None


def test_config_requires_real_two_key_value_head_grouping() -> None:
    cases = (
        ((16, 8, 16, 4, 2, 16, 1, 1), "greater than"),
        ((16, 8, 16, 4, 3, 16, 1, 1), "divisible by 2"),
        ((16, 8, 16, 4, 6, 16, 1, 1), "local_dim"),
    )
    for arguments, message in cases:
        with unittest.TestCase().assertRaisesRegex(ValueError, message):
            InputByteCodecConfig(*arguments)


class FixedDecoderCodec(InputByteCodec):
    def __init__(self, generated: torch.Tensor) -> None:
        super().__init__(
            InputByteCodecConfig(8, 8, 8, 4, 4, 16, 1, 1),
            torch.randn((256, 8)),
        )
        self.generated = generated

    def decode_logits(
        self,
        latent: torch.Tensor,
        output_positions: int | None = None,
    ) -> torch.Tensor:
        positions = self.max_span + 1 if output_positions is None else output_positions
        logits = torch.full((latent.shape[0], positions, 257), -1_000.0)
        generated = self.generated[: latent.shape[0], :positions]
        logits.scatter_(2, generated[:, :, None], 1_000.0)
        return logits


def test_private_eos_terminates_payload_and_rejects_invalid_frames() -> None:
    generated = torch.tensor(
        [
            [ord("a"), ord("b"), CODEC_EOS, ord("x"), ord("x")],
            [CODEC_EOS, ord("a"), ord("b"), ord("c"), ord("d")],
            [ord("a"), ord("b"), ord("c"), ord("d"), ord("e")],
        ]
    )
    codec = FixedDecoderCodec(generated)

    decoded = codec.decode_greedy(torch.zeros((3, 8)))

    assert decoded == [b"ab", None, None]


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_single_bytes_use_exact_frozen_embeddings,
            test_source_dtype_byte_rows_stay_exact,
            test_forward_and_reconstruction_loss_are_finite,
            test_codec_uses_mandatory_gqa_and_separate_projections,
            test_gqa_matches_explicit_key_value_repetition_and_gradients,
            test_encoder_baseline_matches_position_weighted_byte_embeddings,
            test_decoder_prefix_is_independent_of_later_queries,
            test_profiles_fit_target_deployment_budgets_without_allocating_weights,
            test_profile_shrink_reduces_parameters_state_and_analytical_flops,
            test_decoder_trains_only_its_model_to_local_projection,
            test_config_requires_real_two_key_value_head_grouping,
            test_private_eos_terminates_payload_and_rejects_invalid_frames,
        )
    )
