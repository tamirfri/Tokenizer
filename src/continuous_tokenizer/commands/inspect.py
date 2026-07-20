from __future__ import annotations

import argparse
from typing import Any

from continuous_tokenizer.backbone.assets import resolve_tokenizer_assets
from continuous_tokenizer.backbone.config import text_config, tie_word_embeddings
from continuous_tokenizer.backbone.vocabulary import inspect_tokenizer


def inspect_model(args: argparse.Namespace) -> dict[str, Any]:
    assets = resolve_tokenizer_assets(args.model, args.revision)
    model_config = text_config(assets.config)
    embedding_rows = int(model_config["vocab_size"])
    vocabulary = inspect_tokenizer(assets.tokenizer, embedding_rows=embedding_rows)
    return {
        "model_id": assets.model_id,
        "revision": assets.revision,
        "hidden_size": int(model_config["hidden_size"]),
        "tie_word_embeddings": tie_word_embeddings(assets.config),
        **vocabulary.to_summary(),
    }
