from __future__ import annotations

import torch

from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.input.adapter import (
    InputEmbeddingAdapter,
)


class InvalidSpanCodec(InputByteCodec):
    def reconstruction_matches(
        self,
        latent: torch.Tensor,
        byte_values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del byte_values, valid_mask
        return torch.zeros(latent.shape[0], dtype=torch.bool, device=latent.device)

    def decode_greedy(
        self,
        latent: torch.Tensor,
        *,
        maximum_length: int | None = None,
    ) -> list[bytes | None]:
        del maximum_length
        return [None] * latent.shape[0]


def make_adapter(source: torch.Tensor | None = None) -> InputEmbeddingAdapter:
    if source is None:
        generator = torch.Generator().manual_seed(19)
        source = torch.randn((258, 8), generator=generator)
    width = source.shape[1]
    token_bytes: tuple[bytes | None, ...] = (
        *(bytes([value]) for value in range(256)),
        None,
        b"ab",
    )
    vocabulary = ByteVocabulary(
        token_bytes=token_bytes,
        ordinary_ids=(*range(256), 257),
        control_ids=(256,),
        byte_token_ids=tuple(range(256)),
        max_token_bytes=2,
    )
    codec = InvalidSpanCodec(
        InputByteCodecConfig(width, 8, width, 4, 4, 16, 1, 1),
        source[:256],
    ).eval()
    return InputEmbeddingAdapter(
        codec,
        vocabulary,
        torch.tensor([256]),
        source[256:257],
        namespace="test",
    ).eval()
