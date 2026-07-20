from __future__ import annotations

import torch

from continuous_tokenizer.batching import build_span_batch, byte_reconstruction_loss
from continuous_tokenizer.codec import CodecConfig, ContinuousByteCodec


def make_codec(max_span: int = 4) -> ContinuousByteCodec:
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn((256, 16), generator=generator)
    config = CodecConfig(
        embedding_dim=16,
        local_dim=8,
        max_span=max_span,
        heads=2,
        feedforward_dim=16,
        encoder_layers=1,
        decoder_layers=1,
    )
    return ContinuousByteCodec(config, embeddings)


def test_single_bytes_use_exact_frozen_embeddings() -> None:
    codec = make_codec()
    spans = [bytes([value]) for value in range(256)]
    batch = build_span_batch(spans, max_span=codec.max_span, device=torch.device("cpu"))

    latent = codec.encode(batch.byte_values, batch.valid_mask)

    assert torch.equal(latent, codec.byte_embeddings)


def test_forward_and_reconstruction_loss_are_finite() -> None:
    codec = make_codec()
    batch = build_span_batch([b"a", b"abc"], max_span=codec.max_span, device=codec.device)

    latent, logits = codec(batch.byte_values, batch.valid_mask)
    loss = byte_reconstruction_loss(logits, batch.framed_targets, batch.target_mask)

    assert latent.shape == (2, 16)
    assert logits.shape == (2, 5, 256)
    assert torch.isfinite(loss)


def test_control_embeddings_remain_frozen_buffers() -> None:
    embeddings = torch.randn((256, 16))
    controls = torch.randn((2, 16))
    codec = ContinuousByteCodec(
        make_codec().config,
        embeddings,
        torch.tensor([300, 301]),
        controls,
    )

    assert torch.equal(codec.control_embeddings, controls)
    assert "control_embeddings" not in dict(codec.named_parameters())
