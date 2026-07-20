"""Greedy generation through native and input-only tokenizer paths."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, final

import torch
from torch import Tensor, nn

from continuous_tokenizer.input.adapter import (
    InputEmbeddingAdapter,
    InputEncoding,
    InputMode,
    SegmentationAlignment,
)
from continuous_tokenizer.runtime.device import module_device, module_dtype


@final
@dataclass(frozen=True, slots=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    positions_added: int


@final
class InputOnlyCausalLM:
    def __init__(
        self,
        model: nn.Module,
        adapter: InputEmbeddingAdapter,
        *,
        segmentation_alignment: SegmentationAlignment = "arbitrary",
    ) -> None:
        self.model = model.eval()
        self.adapter = adapter.eval()
        self.segmentation_alignment = segmentation_alignment
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def model_device(self) -> torch.device:
        return module_device(self.model)

    @property
    def model_dtype(self) -> torch.dtype:
        return module_dtype(self.model)

    def _model_embeddings(self, encoding: InputEncoding) -> Tensor:
        return encoding.embeddings.to(device=self.model_device, dtype=self.model_dtype).unsqueeze(0)

    def forward_token_ids(self, token_ids: Sequence[int], *, mode: InputMode) -> tuple[Any, InputEncoding]:
        encoding = self.adapter.encode_token_ids(
            token_ids,
            mode=mode,
            cache=self.adapter.codec.encoding_cache,
            alignment=self.segmentation_alignment,
        )
        outputs = self.model(
            inputs_embeds=self._model_embeddings(encoding),
            position_ids=encoding.position_ids.to(self.model_device).unsqueeze(0),
            use_cache=False,
        )
        return outputs, encoding

    @torch.inference_mode()
    def generate(
        self,
        token_ids: Sequence[int],
        *,
        mode: InputMode,
        eos_token_ids: Iterable[int],
        max_new_tokens: int,
    ) -> GenerationResult:
        encoding = self.adapter.encode_token_ids(
            token_ids,
            mode=mode,
            cache=self.adapter.codec.encoding_cache,
            alignment=self.segmentation_alignment,
        )
        outputs = self.model(
            inputs_embeds=self._model_embeddings(encoding),
            position_ids=encoding.position_ids.to(self.model_device).unsqueeze(0),
            use_cache=True,
            logits_to_keep=1,
        )
        generated: list[int] = []
        positions_added = 0
        stop_ids = frozenset(eos_token_ids)
        past = outputs.past_key_values
        logits = outputs.logits[:, -1]

        for _ in range(max_new_tokens):
            token_id = int(logits.argmax(dim=-1).item())
            generated.append(token_id)
            if token_id in stop_ids:
                break
            next_encoding = self.adapter.encode_token_ids(
                (token_id,),
                mode="segmented",
                cache=self.adapter.codec.encoding_cache,
                alignment=self.segmentation_alignment,
                position_offset=len(token_ids) + len(generated) - 1,
            )
            positions_added += len(next_encoding.positions)
            outputs = self.model(
                inputs_embeds=self._model_embeddings(next_encoding),
                position_ids=next_encoding.position_ids.to(self.model_device).unsqueeze(0),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1]
        return GenerationResult(tuple(generated), positions_added)


@torch.inference_mode()
def native_greedy_generate(
    model: nn.Module,
    token_ids: Sequence[int],
    *,
    eos_token_ids: Iterable[int],
    max_new_tokens: int,
) -> GenerationResult:
    values = torch.tensor([list(token_ids)], dtype=torch.long, device=module_device(model))
    outputs = model(input_ids=values, use_cache=True, logits_to_keep=1)
    generated: list[int] = []
    stop_ids = frozenset(eos_token_ids)
    past = outputs.past_key_values
    logits = outputs.logits[:, -1]
    for _ in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        generated.append(token_id)
        if token_id in stop_ids:
            break
        next_id = torch.tensor([[token_id]], dtype=torch.long, device=values.device)
        outputs = model(input_ids=next_id, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = outputs.past_key_values
        logits = outputs.logits[:, -1]
    return GenerationResult(tuple(generated), len(generated))
