from __future__ import annotations

from pathlib import Path

import torch

from continuous_tokenizer.checkpoint import load_checkpoint, save_checkpoint
from continuous_tokenizer.codec import CodecConfig, ContinuousByteCodec


def test_checkpoint_contains_codec_but_not_source_table(tmp_path: Path) -> None:
    codec = ContinuousByteCodec(
        CodecConfig(8, 8, 4, 2, 16, 1, 1),
        torch.randn((256, 8)),
    )
    path = tmp_path / "codec.pt"

    save_checkpoint(path, codec, {"model_id": "synthetic"})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    restored, metadata = load_checkpoint(path)

    assert payload["format_version"] == 1
    assert not any("original" in key for key in payload["state_dict"])
    assert metadata["model_id"] == "synthetic"
    assert torch.equal(restored.byte_embeddings, codec.byte_embeddings)
