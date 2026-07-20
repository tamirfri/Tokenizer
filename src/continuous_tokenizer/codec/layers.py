from __future__ import annotations

from typing import Final, final

from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function

KEY_VALUE_HEADS: Final = 2


def gqa_metadata(query_heads: int) -> dict[str, int | bool]:
    return {
        "query_heads": query_heads,
        "key_value_heads": KEY_VALUE_HEADS,
        "enable_gqa": True,
    }


@final
class GroupedQuerySelfAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        query_heads: int,
    ) -> None:
        super().__init__()
        if query_heads <= KEY_VALUE_HEADS or query_heads % KEY_VALUE_HEADS:
            raise ValueError(f"query_heads must be greater than and divisible by {KEY_VALUE_HEADS}")
        if dimension % query_heads:
            raise ValueError("dimension must be divisible by query_heads")

        self.query_heads = query_heads
        self.key_value_heads = KEY_VALUE_HEADS
        self.head_dimension = dimension // query_heads
        key_value_dimension = KEY_VALUE_HEADS * self.head_dimension
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, key_value_dimension)
        self.value = nn.Linear(dimension, key_value_dimension)
        self.output = nn.Linear(dimension, dimension)

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        batch_size, positions, dimension = value.shape
        query = self.query(value).view(
            batch_size,
            positions,
            self.query_heads,
            self.head_dimension,
        )
        key = self.key(value).view(
            batch_size,
            positions,
            self.key_value_heads,
            self.head_dimension,
        )
        projected_value = self.value(value).view(
            batch_size,
            positions,
            self.key_value_heads,
            self.head_dimension,
        )
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            projected_value.transpose(1, 2),
            attn_mask=valid_mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, positions, dimension)
        return self.output(attended)


@final
class _EncoderLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        query_heads: int,
        feedforward_dimension: int,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = GroupedQuerySelfAttention(dimension, query_heads)
        self.feedforward_norm = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension),
            nn.GELU(),
            nn.Linear(feedforward_dimension, dimension),
        )

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        with record_function("tokenizer.encoder_attention"):
            value = value + self.attention(self.attention_norm(value), valid_mask)
        with record_function("tokenizer.encoder_feedforward"):
            return value + self.feedforward(self.feedforward_norm(value))


@final
class ByteSpanEncoder(nn.Module):
    def __init__(
        self,
        dimension: int,
        query_heads: int,
        feedforward_dimension: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_EncoderLayer(dimension, query_heads, feedforward_dimension) for _ in range(layers))
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, value: Tensor, valid_mask: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value, valid_mask)
        return self.output_norm(value)


@final
class _DecoderLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        feedforward_dimension: int,
    ) -> None:
        super().__init__()
        self.condition = nn.Linear(dimension, dimension, bias=False)
        self.feedforward_norm = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension, bias=False),
            nn.GELU(),
            nn.Linear(feedforward_dimension, dimension, bias=False),
        )

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        value = value + self.condition(condition).unsqueeze(1)
        return value + self.feedforward(self.feedforward_norm(value))


@final
class ByteSpanDecoder(nn.Module):
    def __init__(
        self,
        dimension: int,
        feedforward_dimension: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _DecoderLayer(
                dimension,
                feedforward_dimension,
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, queries: Tensor, condition: Tensor) -> Tensor:
        for layer in self.layers:
            queries = layer(queries, condition)
        return self.output_norm(queries)
