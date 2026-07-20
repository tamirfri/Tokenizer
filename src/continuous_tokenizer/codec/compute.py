from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from continuous_tokenizer.backbone.config import text_config
from continuous_tokenizer.codec.input import InputByteCodecConfig
from continuous_tokenizer.codec.layers import KEY_VALUE_HEADS
from continuous_tokenizer.codec.output import OutputByteCodecConfig


def backbone_forward_flops(
    config: Any,
    query_positions: int,
    context_positions: int | None = None,
) -> int:
    """Estimate multiply-add FLOPs for one frozen-backbone forward."""
    config = text_config(config)
    context_positions = query_positions if context_positions is None else context_positions
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    heads = int(config.num_attention_heads)
    key_value_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    query_dim = heads * head_dim
    key_value_dim = key_value_heads * head_dim
    layer_types = tuple(
        getattr(
            config,
            "layer_types",
            ("full_attention",) * int(config.num_hidden_layers),
        ),
    )
    active_experts = int(getattr(config, "num_experts_per_tok", 1))
    feedforward = 6 * query_positions * hidden * intermediate * active_experts
    total = len(layer_types) * feedforward
    projections = 6 * query_positions * hidden * query_dim
    projections += 4 * query_positions * hidden * key_value_dim

    full_layers = layer_types.count("full_attention")
    total += full_layers * (projections + 4 * query_positions * context_positions * query_dim)

    sliding_layers = layer_types.count("sliding_attention")
    sliding_context = min(
        context_positions,
        int(getattr(config, "sliding_window", context_positions)),
    )
    total += sliding_layers * (projections + 4 * query_positions * sliding_context * query_dim)

    linear_layers = layer_types.count("linear_attention")
    if linear_layers:
        key_dim = int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
        value_dim = int(config.linear_num_value_heads) * int(
            config.linear_value_head_dim,
        )
        value_heads = int(config.linear_num_value_heads)
        convolution = 2 * query_positions * (2 * key_dim + value_dim) * int(config.linear_conv_kernel_dim)
        linear_projections = 2 * query_positions * hidden * (2 * key_dim + value_dim)
        linear_projections += 2 * query_positions * hidden * value_dim
        linear_projections += 4 * query_positions * hidden * value_heads
        linear_projections += 2 * query_positions * value_dim * hidden
        recurrence = 8 * query_positions * value_heads * int(config.linear_key_head_dim) * int(config.linear_value_head_dim)
        total += linear_layers * (convolution + linear_projections + recurrence)
    unsupported = set(layer_types) - {
        "full_attention",
        "linear_attention",
        "sliding_attention",
    }
    if unsupported:
        raise ValueError(
            f"unsupported layer types for FLOP estimation: {sorted(unsupported)}",
        )
    return total


def input_table_projection_flops(config: InputByteCodecConfig) -> int:
    return 2 * 256 * config.embedding_dim * config.local_dim


def input_encode_row_flops(config: InputByteCodecConfig, length: int) -> int:
    positions = length + 1
    local = config.local_dim
    head_dim = local // config.query_heads
    key_value_dim = KEY_VALUE_HEADS * head_dim
    baseline = 2 * length * config.embedding_dim
    attention_projection = 4 * positions * local * local
    attention_projection += 4 * positions * local * key_value_dim
    attention = 4 * positions * positions * local
    feedforward = 4 * positions * local * config.feedforward_dim
    encoder = config.encoder_layers * (attention_projection + attention + feedforward)
    output = 2 * local * config.projection_dim
    output += 2 * config.projection_dim * config.embedding_dim
    return baseline + encoder + output


def input_encode_flops(config: InputByteCodecConfig, length: int) -> int:
    return input_table_projection_flops(config) + input_encode_row_flops(
        config,
        length,
    )


def input_encode_batch_flops(
    config: InputByteCodecConfig,
    candidate_lengths: Mapping[int, int],
    *,
    neural_invocations: int,
) -> int:
    if neural_invocations < 0:
        raise ValueError("neural invocation count must be non-negative")
    rows = sum(count * input_encode_row_flops(config, length) for length, count in candidate_lengths.items())
    return neural_invocations * input_table_projection_flops(config) + rows


def _decoder_flops(
    config: InputByteCodecConfig | OutputByteCodecConfig,
    positions: int,
) -> int:
    local = config.local_dim
    projection = 2 * config.embedding_dim * local
    layer = 2 * local * local + 4 * positions * local * config.feedforward_dim
    byte_head = 2 * positions * local * 257
    return projection + config.decoder_layers * layer + byte_head


def input_validation_flops(config: InputByteCodecConfig, length: int) -> int:
    return _decoder_flops(config, length + 1)


def input_codec_flops(
    config: InputByteCodecConfig,
    candidate_lengths: Mapping[int, int],
    *,
    neural_invocations: int,
) -> tuple[int, int]:
    encode = input_encode_batch_flops(
        config,
        candidate_lengths,
        neural_invocations=neural_invocations,
    )
    validation = sum(count * input_validation_flops(config, length) for length, count in candidate_lengths.items())
    return encode, validation


def output_decode_flops(config: OutputByteCodecConfig) -> int:
    controls = 2 * config.embedding_dim * (config.control_count + 1)
    return _decoder_flops(config, config.max_span + 1) + controls
