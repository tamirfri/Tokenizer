from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import torch

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.contracts.profiles import Profile

TEST_PROFILE: Final = Profile("small", 8, 1, 1, 1, 4, 2, 16)


@contextmanager
def limited_torch_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


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
        config={"tie_word_embeddings": False, "removable_input_table": False},
        embedding_tensor_name="model.embed_tokens.weight",
        embedding_shard=tmp_path / "synthetic.safetensors",
        input_embeddings=torch.randn((256, 8)),
        vocabulary=vocabulary,
    )


def pair_assets(tmp_path: Path, embedding: torch.Tensor | None = None) -> ModelAssets:
    assets = synthetic_assets(tmp_path)
    pair_embedding = torch.randn((1, 8)) if embedding is None else embedding
    return replace(
        assets,
        input_embeddings=torch.cat((assets.input_embeddings, pair_embedding)),
        vocabulary=ByteVocabulary(
            token_bytes=(*assets.vocabulary.token_bytes, b"ab"),
            ordinary_ids=(256,),
            control_ids=tuple(range(256)),
            byte_token_ids=assets.vocabulary.byte_token_ids,
            max_token_bytes=2,
        ),
    )


def subset_assets(tmp_path: Path, rows: int = 2048) -> ModelAssets:
    if rows < 1000:
        raise ValueError("subset fixture requires at least 1,000 rows")
    assets = synthetic_assets(tmp_path)
    pair_rows = min(744, rows - 256)
    pair_payloads = tuple(index.to_bytes(2, "big") for index in range(pair_rows))
    triple_payloads = tuple(b"\xff" + index.to_bytes(2, "big") for index in range(rows - 256 - pair_rows))
    token_bytes = (
        *assets.vocabulary.token_bytes,
        *pair_payloads,
        *triple_payloads,
    )
    return replace(
        assets,
        input_embeddings=torch.randn((rows, 8)),
        vocabulary=ByteVocabulary(
            token_bytes=token_bytes,
            ordinary_ids=tuple(range(rows)),
            control_ids=(),
            byte_token_ids=tuple(range(256)),
            max_token_bytes=3,
        ),
    )
