from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, final

import torch
from torch import Tensor, nn

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.data.corpus import synthetic_documents

SYNTHETIC_MODEL_ID: Final = "continuous-tokenizer/synthetic-model"
SYNTHETIC_MODEL_REVISION: Final = "synthetic"


@final
class SyntheticByteTokenizer:
    eos_token_id = None

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise ValueError("the synthetic tokenizer has no special tokens")
        return list(text.encode("utf-8"))


@final
class _SyntheticBackbone(nn.Module):
    def __init__(self, embeddings: Tensor) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding.from_pretrained(embeddings, freeze=True)

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        **_: Any,
    ) -> Any:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one synthetic backbone input")
        hidden = self.embed_tokens(input_ids) if input_ids is not None else inputs_embeds
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=() if use_cache else past_key_values,
        )


@final
class SyntheticCausalLM(nn.Module):
    def __init__(self, embeddings: Tensor) -> None:
        super().__init__()
        self.base_model = _SyntheticBackbone(embeddings)
        self.lm_head = nn.Linear(embeddings.shape[1], embeddings.shape[0], bias=False)
        nn.init.zeros_(self.lm_head.weight)
        self.config = SimpleNamespace(
            hidden_size=embeddings.shape[1],
            intermediate_size=2 * embeddings.shape[1],
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            layer_types=("full_attention",),
            vocab_size=embeddings.shape[0],
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.embed_tokens

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        logits_to_keep: Tensor | int | None = None,
        **kwargs: Any,
    ) -> Any:
        output = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logit_hidden = output.last_hidden_state
        if isinstance(logits_to_keep, Tensor):
            logit_hidden = logit_hidden.index_select(-2, logits_to_keep.to(device=logit_hidden.device))
        elif logits_to_keep is not None:
            logit_hidden = logit_hidden[:, -logits_to_keep:]
        return SimpleNamespace(
            logits=self.lm_head(logit_hidden),
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
        )


def synthetic_model_assets() -> ModelAssets:
    generator = torch.Generator().manual_seed(11)
    byte_embeddings = torch.randn((256, 16), generator=generator)
    documents = [document for split in ("train", "validation", "test") for document in synthetic_documents(split)]
    pairs = sorted({document[index : index + 2] for document in documents for index in range(len(document) - 1)})
    pair_embeddings = torch.stack([(byte_embeddings[pair[0]] + 2 * byte_embeddings[pair[1]]) / 3 for pair in pairs])
    embeddings = torch.cat((byte_embeddings, pair_embeddings))
    token_bytes = tuple(bytes([value]) for value in range(256)) + tuple(pairs)
    ordinary_ids = tuple(range(len(token_bytes)))
    vocabulary = ByteVocabulary(
        token_bytes=token_bytes,
        ordinary_ids=ordinary_ids,
        control_ids=(),
        byte_token_ids=tuple(range(256)),
        max_token_bytes=2,
    )
    return ModelAssets(
        model_id=SYNTHETIC_MODEL_ID,
        revision=SYNTHETIC_MODEL_REVISION,
        tokenizer=SyntheticByteTokenizer(),
        config={
            "hidden_size": 16,
            "vocab_size": len(token_bytes),
            "tie_word_embeddings": True,
            "removable_input_table": False,
        },
        embedding_tensor_name="synthetic.embed_tokens.weight",
        embedding_shard=Path("synthetic-embedding-table.safetensors"),
        input_embeddings=embeddings,
        vocabulary=vocabulary,
    )
