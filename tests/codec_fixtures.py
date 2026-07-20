from __future__ import annotations

import torch

from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig


def make_codec(max_span: int = 4) -> InputByteCodec:
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn((256, 16), generator=generator)
    config = InputByteCodecConfig(
        embedding_dim=16,
        local_dim=8,
        projection_dim=16,
        max_span=max_span,
        query_heads=4,
        feedforward_dim=16,
        encoder_layers=1,
        decoder_layers=1,
    )
    return InputByteCodec(config, embeddings)
