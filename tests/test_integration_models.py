from __future__ import annotations

import os

import pytest

from continuous_tokenizer.model_assets import resolve_tokenizer_assets
from continuous_tokenizer.vocabulary import inspect_tokenizer

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", "openai/gpt-oss-20b"])
def test_real_tokenizer_has_all_atomic_bytes(model_id: str) -> None:
    assets = resolve_tokenizer_assets(model_id)
    vocabulary = inspect_tokenizer(assets.tokenizer, embedding_rows=assets.config["vocab_size"])

    assert len(vocabulary.byte_token_ids) == 256
