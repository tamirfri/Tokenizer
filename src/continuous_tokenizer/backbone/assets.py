from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, final

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer

from continuous_tokenizer.backbone.config import model_loader
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary, inspect_tokenizer

PREFERRED_EMBEDDING_NAMES: Final = (
    "backbone.embeddings.weight",
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "model.tok_embeddings.weight",
    "embed_tokens.weight",
)


@dataclass(slots=True)
class TokenizerAssets:
    model_id: str
    revision: str
    tokenizer: Any
    config: dict[str, Any]


@final
@dataclass(slots=True)
class ModelAssets(TokenizerAssets):
    embedding_tensor_name: str
    embedding_shard: Path
    input_embeddings: torch.Tensor
    vocabulary: ByteVocabulary


def _resolve_model(model_id: str, revision: str | None) -> tuple[str, frozenset[str]]:
    info = HfApi().model_info(model_id, revision=revision)
    resolved_revision = info.sha
    if resolved_revision is None:
        raise ValueError(f"Hugging Face did not resolve a revision for {model_id}")
    filenames = frozenset(sibling.rfilename for sibling in info.siblings or [])
    return resolved_revision, filenames


def _load_tokenizer_assets(model_id: str, revision: str) -> TokenizerAssets:
    config_path = hf_hub_download(model_id, "config.json", revision=revision)
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"{model_id} does not expose a fast tokenizer")
    return TokenizerAssets(model_id, revision, tokenizer, config)


def resolve_tokenizer_assets(model_id: str, revision: str | None = None) -> TokenizerAssets:
    resolved_revision, _ = _resolve_model(model_id, revision)
    return _load_tokenizer_assets(model_id, resolved_revision)


def _find_embedding_shard(model_id: str, revision: str, filenames: frozenset[str]) -> tuple[str, Path]:
    index_name = "model.safetensors.index.json"
    if index_name in filenames:
        index_path = hf_hub_download(model_id, index_name, revision=revision)
        with Path(index_path).open(encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        tensor_name = next((name for name in PREFERRED_EMBEDDING_NAMES if name in weight_map), None)
        if tensor_name is None:
            candidates = [name for name in weight_map if name.endswith("embed_tokens.weight")]
            if len(candidates) != 1:
                raise ValueError(f"could not identify the input embedding tensor in {model_id}")
            tensor_name = candidates[0]
        shard = hf_hub_download(model_id, weight_map[tensor_name], revision=revision)
        return tensor_name, Path(shard)

    single_name = "model.safetensors"
    if single_name not in filenames:
        raise ValueError(f"{model_id} has no Safetensors checkpoint")
    shard = Path(hf_hub_download(model_id, single_name, revision=revision))
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    tensor_name = next((name for name in PREFERRED_EMBEDDING_NAMES if name in keys), None)
    if tensor_name is None:
        raise ValueError(f"could not identify the input embedding tensor in {model_id}")
    return tensor_name, shard


def load_model_assets(model_id: str, revision: str | None = None) -> ModelAssets:
    from continuous_tokenizer.backbone.synthetic import (
        SYNTHETIC_MODEL_ID,
        SYNTHETIC_MODEL_REVISION,
        synthetic_model_assets,
    )

    if model_id == SYNTHETIC_MODEL_ID:
        if revision not in {None, SYNTHETIC_MODEL_REVISION}:
            raise ValueError(f"unsupported synthetic model revision: {revision}")
        return synthetic_model_assets()
    resolved_revision, filenames = _resolve_model(model_id, revision)
    tokenizer_assets = _load_tokenizer_assets(model_id, resolved_revision)
    tensor_name, shard = _find_embedding_shard(model_id, resolved_revision, filenames)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        embeddings = handle.get_tensor(tensor_name)
    vocabulary = inspect_tokenizer(tokenizer_assets.tokenizer, embedding_rows=embeddings.shape[0])
    return ModelAssets(
        model_id=tokenizer_assets.model_id,
        revision=tokenizer_assets.revision,
        tokenizer=tokenizer_assets.tokenizer,
        config=tokenizer_assets.config,
        embedding_tensor_name=tensor_name,
        embedding_shard=shard,
        input_embeddings=embeddings,
        vocabulary=vocabulary,
    )


def load_frozen_causal_lm(
    assets: ModelAssets,
    device: torch.device,
    *,
    output_attentions: bool = False,
) -> nn.Module:
    from continuous_tokenizer.backbone.synthetic import SYNTHETIC_MODEL_ID, SyntheticCausalLM

    if assets.model_id == SYNTHETIC_MODEL_ID:
        model = SyntheticCausalLM(assets.input_embeddings).to(device=device)
        model.eval()
        model.requires_grad_(False)
        return model
    attention_options: dict[str, Any] = {}
    if output_attentions:
        # Attention weights require the explicit backend documented by Transformers.
        # https://huggingface.co/docs/transformers/main/attention_interface
        attention_options = {"attn_implementation": "eager", "output_attentions": True}
    model = model_loader(assets.config).from_pretrained(
        assets.model_id,
        revision=assets.revision,
        dtype=assets.input_embeddings.dtype,
        **attention_options,
    )
    model.to(device=device)
    model.eval()
    model.requires_grad_(False)
    return model
