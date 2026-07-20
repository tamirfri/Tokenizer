from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final

import torch
from torch import Tensor, nn

from continuous_tokenizer.runtime.device import module_device, module_dtype


def base_text_model(causal_lm: nn.Module) -> nn.Module:
    """Return the raw text backbone so forwarding cannot execute the vocabulary head."""
    candidate = getattr(causal_lm, "base_model", None)
    if not isinstance(candidate, nn.Module) or candidate is causal_lm:
        raise ValueError("causal language model does not expose a distinct base model")
    language_model = getattr(candidate, "language_model", None)
    if isinstance(language_model, nn.Module):
        nested = getattr(language_model, "base_model", language_model)
        if isinstance(nested, nn.Module):
            candidate = nested
    return candidate


@final
@dataclass(frozen=True, slots=True)
class BackboneOutput:
    last_hidden_state: Tensor
    past_key_values: Any


@final
class FrozenBackbone:
    def __init__(self, causal_lm: nn.Module) -> None:
        self.source_model = causal_lm.requires_grad_(False).eval()
        self.model = base_text_model(self.source_model).eval()

    @property
    def device(self) -> torch.device:
        return module_device(self.model)

    @property
    def dtype(self) -> torch.dtype:
        return module_dtype(self.model)

    @property
    def input_embeddings(self) -> nn.Module:
        getter = getattr(self.source_model, "get_input_embeddings", None)
        if not callable(getter):
            raise ValueError("causal language model has no input embedding accessor")
        embeddings = getter()
        if not isinstance(embeddings, nn.Module):
            raise ValueError("causal language model has no input embedding module")
        return embeddings

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> BackboneOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, Tensor):
            raise ValueError("base model did not return last_hidden_state")
        return BackboneOutput(hidden, getattr(outputs, "past_key_values", None))
