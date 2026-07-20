from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import continuous_tokenizer.training as training_module
from continuous_tokenizer.checkpoint import load_checkpoint
from continuous_tokenizer.model_assets import ModelAssets
from continuous_tokenizer.training import TrainingOptions, train_experiment
from continuous_tokenizer.vocabulary import ByteVocabulary


class SyntheticTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def synthetic_assets(tmp_path: Path) -> ModelAssets:
    token_bytes: tuple[bytes | None, ...] = tuple(bytes([value]) for value in range(256))
    vocabulary = ByteVocabulary(
        token_bytes=token_bytes,
        ordinary_ids=tuple(range(256)),
        control_ids=(),
        byte_token_ids=tuple(range(256)),
        max_token_bytes=1,
    )
    tokenizer: Any = SyntheticTokenizer()
    return ModelAssets(
        model_id="synthetic/model",
        revision="synthetic-revision",
        tokenizer=tokenizer,
        config={"tie_word_embeddings": False},
        embedding_tensor_name="model.embed_tokens.weight",
        embedding_shard=tmp_path / "synthetic.safetensors",
        input_embeddings=torch.randn((256, 8)),
        vocabulary=vocabulary,
    )


def test_training_pipeline_writes_reloadable_checkpoint(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(training_module, "load_wikitext_documents", lambda split: [b"abc"])
    options = TrainingOptions(
        output_dir=tmp_path / "checkpoints",
        profile="small",
        batch_size=256,
        stage1_epochs=1,
        stage2_epochs=0,
        stage2_samples=0,
        validation_bytes=3,
        patience=1,
    )

    results = train_experiment(synthetic_assets(tmp_path), options, device=torch.device("cpu"))
    checkpoint = Path(results[0].checkpoint)
    codec, metadata = load_checkpoint(checkpoint)

    assert checkpoint.is_file()
    assert (options.output_dir / "run-manifest.json").is_file()
    assert metadata["model_id"] == "synthetic/model"
    assert codec.config.embedding_dim == 8
    assert results[0].round_trip
