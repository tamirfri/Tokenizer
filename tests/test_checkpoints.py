from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from codec_fixtures import make_codec

from continuous_tokenizer.codec.checkpoints import load_checkpoint, save_checkpoint
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig


def test_checkpoint_contains_codec_but_not_source_table() -> None:
    with tempfile.TemporaryDirectory() as directory:
        codec = InputByteCodec(
            InputByteCodecConfig(8, 8, 8, 4, 4, 16, 1, 1),
            torch.randn((256, 8)),
        )
        path = Path(directory) / "codec.pt"
        control_ids = torch.tensor([300, 301])
        control_embeddings = torch.randn((2, 8))

        save_checkpoint(
            path,
            codec,
            {"model_id": "synthetic"},
            control_ids=control_ids,
            control_embeddings=control_embeddings,
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        loaded = load_checkpoint(path)
        restored = loaded.codec
        metadata = loaded.metadata
        controls = loaded.controls

        assert set(payload) == {
            "direction",
            "config",
            "metadata",
            "state_dict",
            "control_ids",
            "control_embeddings",
        }
        assert payload["direction"] == "input_only"
        assert not any("original" in key for key in payload["state_dict"])
        assert not any("control" in key for key in payload["state_dict"])
        assert metadata["model_id"] == "synthetic"
        assert torch.equal(restored.byte_embeddings, codec.byte_embeddings)
        assert torch.equal(controls.ids, control_ids)
        assert torch.equal(controls.embeddings, control_embeddings)


def test_checkpoint_round_trip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        codec = make_codec()
        path = Path(directory) / "codec.pt"

        save_checkpoint(path, codec, {"model_id": "synthetic"})
        restored = load_checkpoint(path).codec
        restored_state = restored.state_dict()

        assert restored.config == codec.config
        assert all(torch.equal(restored_state[name], value) for name, value in codec.state_dict().items())


def test_checkpoint_rejects_noncanonical_shape() -> None:
    with tempfile.TemporaryDirectory() as directory:
        codec = make_codec()
        current = Path(directory) / "current.pt"
        invalid = Path(directory) / "invalid.pt"
        save_checkpoint(current, codec, {"model_id": "synthetic"})
        payload = torch.load(current, map_location="cpu", weights_only=True)
        payload.pop("direction")
        torch.save(payload, invalid)

        with unittest.TestCase().assertRaisesRegex(ValueError, "current input codec"):
            load_checkpoint(invalid)


def test_checkpoint_uses_frozen_byte_embedding_dtype() -> None:
    with tempfile.TemporaryDirectory() as directory:
        codec = InputByteCodec(
            InputByteCodecConfig(8, 8, 8, 4, 4, 16, 1, 1),
            torch.randn((256, 8), dtype=torch.bfloat16),
        )
        path = Path(directory) / "codec.pt"

        save_checkpoint(path, codec, {"model_id": "synthetic"})
        restored = load_checkpoint(path).codec

        assert restored.dtype == torch.bfloat16
        assert all(not value.is_floating_point() or value.dtype == torch.bfloat16 for value in restored.state_dict().values())


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_checkpoint_contains_codec_but_not_source_table,
            test_checkpoint_round_trip,
            test_checkpoint_rejects_noncanonical_shape,
            test_checkpoint_uses_frozen_byte_embedding_dtype,
        )
    )
