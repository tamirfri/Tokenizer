from __future__ import annotations

import unittest
from collections.abc import Callable
from unittest import mock

import torch
from input_model_fixtures import make_adapter
from transformers import (
    Gemma3TextConfig,
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from continuous_tokenizer.backbone.config import build_model_from_config
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.codec.output import OutputByteCodec, OutputByteCodecConfig
from continuous_tokenizer.contracts.profiles import profile_named
from continuous_tokenizer.input.evaluation import (
    PromptSample,
    _calibration_measurements,
)


def _gemma_config() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        vocab_size=258,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        query_pre_attn_scalar=8,
        sliding_window=16,
        layer_types=["full_attention"],
    )


def _qwen_config() -> Qwen3_5Config:
    text = Qwen3_5TextConfig(
        vocab_size=258,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        tie_word_embeddings=True,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=16,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
    )
    return Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        tie_word_embeddings=True,
    )


def test_supported_backbones_accept_ids_and_input_embeddings_without_native_head() -> None:
    cases: tuple[tuple[str, Callable[[], object], str], ...] = (
        ("qwen", _qwen_config, "Qwen3_5TextModel"),
        ("gemma", _gemma_config, "Gemma3TextModel"),
    )
    token_ids = torch.tensor([[2, 3, 4]])

    for name, config_factory, expected_backbone in cases:
        with unittest.TestCase().subTest(model=name):
            model = build_model_from_config(config_factory())
            backbone = FrozenBackbone(model)
            head = model.get_output_embeddings()
            assert head is not None

            with mock.patch.object(
                head,
                "forward",
                side_effect=AssertionError("native vocabulary head was called"),
            ):
                native = backbone.forward(input_ids=token_ids, use_cache=True)
                embedded = backbone.forward(
                    inputs_embeds=backbone.input_embeddings(token_ids),
                    use_cache=False,
                )

            assert type(backbone.model).__name__ == expected_backbone
            assert native.last_hidden_state.shape == (1, 3, 16)
            assert native.past_key_values is not None
            assert embedded.last_hidden_state.shape == (1, 3, 16)
            assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


def test_small_profile_constructs_both_directional_codecs_for_supported_widths() -> None:
    profile = profile_named("small")
    for embedding_dim in (640, 1_024):
        with unittest.TestCase().subTest(embedding_dim=embedding_dim), torch.device("meta"):
            input_codec = InputByteCodec(
                InputByteCodecConfig(
                    embedding_dim=embedding_dim,
                    local_dim=profile.local_dim,
                    projection_dim=profile.projection_dim(embedding_dim),
                    max_span=128,
                    query_heads=profile.query_heads,
                    feedforward_dim=profile.feedforward_dim,
                    encoder_layers=profile.encoder_layers,
                    decoder_layers=profile.decoder_layers,
                ),
                torch.empty((256, embedding_dim)),
            )
            output_codec = OutputByteCodec(
                OutputByteCodecConfig(
                    embedding_dim=embedding_dim,
                    local_dim=profile.local_dim,
                    max_span=8,
                    feedforward_dim=profile.feedforward_dim,
                    decoder_layers=profile.decoder_layers,
                    control_count=1_000,
                )
            )

        assert input_codec.config.projection_dim == 4 * embedding_dim
        assert output_codec.config.embedding_dim == embedding_dim


def test_supported_model_families_pass_scalar_batch_calibration() -> None:
    samples = [
        PromptSample((2, 3), (4,)),
        PromptSample((5, 6, 7), (8, 9)),
    ]
    for name, config_factory in (
        ("qwen", _qwen_config),
        ("gemma", _gemma_config),
    ):
        with unittest.TestCase().subTest(model=name):
            model = build_model_from_config(config_factory()).eval()
            embeddings = model.get_input_embeddings().weight.detach()
            adapter = make_adapter(embeddings)
            measurements = _calibration_measurements(
                model,
                adapter,
                samples,
                "arbitrary",
                embeddings,
            )

            assert measurements["maximum_kl"] <= 1e-4
            assert measurements["maximum_nll_delta"] <= 1e-3
            assert measurements["top1_agreement"] == 1.0
            assert measurements["maximum_logit_error"] <= 1e-2


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_supported_backbones_accept_ids_and_input_embeddings_without_native_head,
            test_small_profile_constructs_both_directional_codecs_for_supported_widths,
            test_supported_model_families_pass_scalar_batch_calibration,
        )
    )
