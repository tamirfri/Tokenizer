from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.backbone.synthetic import synthetic_model_assets
from continuous_tokenizer.codec.checkpoints import save_checkpoint
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.diagnostics.attention import (
    AttentionOptions,
    AttentionRuntime,
    bertviz_html,
    capture_attention_artifact,
)
from continuous_tokenizer.input.evidence import input_source_identity
from continuous_tokenizer.runtime.tensors import parameter_fingerprint


class FrozenAttentionModel(nn.Module):
    def __init__(self, embeddings: torch.Tensor) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings.clone())

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        scores = hidden @ hidden.transpose(-1, -2) / math.sqrt(hidden.shape[-1])
        causal = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        weights = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
        attention = weights[:, None].repeat(1, 2, 1, 1)
        return SimpleNamespace(attentions=(attention,))


def _checkpoint(tmp_path: Path) -> tuple[ModelAssets, Path]:
    assets = synthetic_model_assets()
    byte_embeddings = assets.input_embeddings[:256]
    codec = InputByteCodec(
        InputByteCodecConfig(
            embedding_dim=16,
            local_dim=16,
            projection_dim=16,
            max_span=64,
            query_heads=4,
            feedforward_dim=32,
            encoder_layers=1,
            decoder_layers=1,
        ),
        byte_embeddings,
    ).eval()
    path = tmp_path / "codec.pt"
    save_checkpoint(
        path,
        codec,
        {
            "model_id": assets.model_id,
            "model_revision": assets.revision,
            "source_identity": input_source_identity(assets),
        },
        control_ids=torch.empty(0, dtype=torch.long),
        control_embeddings=torch.empty((0, 16)),
    )
    return assets, path


def test_attention_artifact_contains_native_and_segmented_views() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets, path = _checkpoint(root)
        model = FrozenAttentionModel(assets.input_embeddings)
        before = parameter_fingerprint(model)

        metadata = capture_attention_artifact(
            assets,
            path,
            AttentionOptions(output_dir=root / "artifact", text="hello", max_tokens=8),
            AttentionRuntime(
                frozen_model=model,
                html_renderer=lambda attention, labels: f"<html>{len(attention)} layers, {len(labels)} positions</html>",
            ),
        )

        artifact_dir = root / "artifact" / "attention"
        assert metadata["diagnostic_only"]
        assert not metadata["performance_comparable"]
        assert metadata["attention_backend"] == "provided"
        assert set(metadata["modes"]) == {"native", "segmented"}
        assert metadata["model"]["parameter_fingerprint"] == before
        assert parameter_fingerprint(model) == before
        assert (artifact_dir / "metadata.json").is_file()
        assert "positions" in (artifact_dir / "native.html").read_text(encoding="utf-8")
        report = (artifact_dir / "report.md").read_text(encoding="utf-8")
        assert "# BertViz Attention Diagnostic" in report
        assert "[Open BertViz](native.html)" in report
        assert "[Open BertViz](segmented.html)" in report
        assert "diagnostic only" in report

        for mode in ("native", "segmented"):
            mode_metadata = metadata["modes"][mode]
            with safe_open(artifact_dir / f"{mode}.safetensors", framework="pt") as handle:
                layer = handle.get_tensor("layer_000")
            positions = mode_metadata["positions"]
            assert layer.shape == (1, 2, positions, positions)
            assert len(mode_metadata["labels"]) == positions

        with unittest.TestCase().assertRaisesRegex(
            FileExistsError,
            "attention artifact already exists",
        ):
            capture_attention_artifact(
                assets,
                path,
                AttentionOptions(output_dir=root / "artifact", text="hello", max_tokens=8),
                AttentionRuntime(
                    frozen_model=model,
                    html_renderer=lambda _attention, _labels: "<html></html>",
                ),
            )


def test_attention_capture_rejects_long_input() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets, path = _checkpoint(root)
        model = FrozenAttentionModel(assets.input_embeddings)

        with unittest.TestCase().assertRaisesRegex(ValueError, "maximum is 2"):
            capture_attention_artifact(
                assets,
                path,
                AttentionOptions(output_dir=root, text="hello", max_tokens=2),
                AttentionRuntime(
                    frozen_model=model,
                    html_renderer=lambda _attention, _labels: "<html></html>",
                ),
            )


def test_optional_bertviz_renderer_returns_html() -> None:
    if importlib.util.find_spec("bertviz") is None:
        raise unittest.SkipTest("bertviz is not installed")
    attention = torch.full((1, 2, 2, 2), 0.5)

    html = bertviz_html((attention,), ("first", "second"))

    assert html.startswith("<script")
    assert "first" in html


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_attention_artifact_contains_native_and_segmented_views,
            test_attention_capture_rejects_long_input,
            test_optional_bertviz_renderer_returns_html,
        )
    )
