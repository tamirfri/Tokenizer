from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import final

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from continuous_tokenizer.codec.batches import decode_span_rows
from continuous_tokenizer.codec.compilation import compile_fullgraph
from continuous_tokenizer.codec.constants import LOCAL_OUTPUT_VALUES
from continuous_tokenizer.codec.layers import ByteSpanDecoder
from continuous_tokenizer.runtime.compiler import configure_compiler_cache


@final
@dataclass(frozen=True, slots=True)
class OutputByteCodecConfig:
    embedding_dim: int
    local_dim: int
    max_span: int
    feedforward_dim: int
    decoder_layers: int
    control_count: int

    def __post_init__(self) -> None:
        if (
            min(
                self.embedding_dim,
                self.local_dim,
                self.max_span,
                self.feedforward_dim,
                self.decoder_layers,
            )
            < 1
        ):
            raise ValueError("codec dimensions, span, and layer counts must be positive")
        if self.max_span > 255:
            raise ValueError("max_span must fit in one byte")
        if self.control_count < 0:
            raise ValueError("control_count must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@final
class OutputByteCodec(nn.Module):
    """Decode a frozen-backbone hidden state into bytes or a structural control event."""

    def __init__(self, config: OutputByteCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_projection = nn.Linear(config.embedding_dim, config.local_dim)
        self.decoder_queries = nn.Parameter(torch.empty(1, config.max_span + 1, config.local_dim))
        self.decoder = ByteSpanDecoder(
            config.local_dim,
            config.feedforward_dim,
            config.decoder_layers,
        )
        self.byte_head = nn.Linear(config.local_dim, LOCAL_OUTPUT_VALUES)
        self.control_selector = nn.Linear(config.embedding_dim, config.control_count + 1)
        self._compiled_decode: Callable[[Tensor, int | None], tuple[Tensor, Tensor]] | None = None
        nn.init.normal_(self.decoder_queries, std=0.02)

    @property
    def max_span(self) -> int:
        return self.config.max_span

    @property
    def device(self) -> torch.device:
        return self.hidden_projection.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.hidden_projection.weight.dtype

    def compile_neural_paths(self, *, backend: str = "inductor") -> None:
        configure_compiler_cache(backend)
        self._compiled_decode = compile_fullgraph(self._decode_tensor, backend=backend)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> OutputByteCodec:
        result = super()._apply(fn, recurse=recurse)
        self._compiled_decode = None
        return result

    def decode_logits(
        self,
        hidden_state: Tensor,
        output_positions: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self._compiled_decode is not None:
            return self._compiled_decode(hidden_state, output_positions)
        return self._decode_tensor(hidden_state, output_positions)

    def _decode_tensor(
        self,
        hidden_state: Tensor,
        output_positions: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if hidden_state.ndim != 2 or hidden_state.shape[1] != self.config.embedding_dim:
            raise ValueError("hidden_state must have shape [batch, embedding_dim]")
        positions = self.max_span + 1 if output_positions is None else output_positions
        if not 2 <= positions <= self.max_span + 1:
            raise ValueError(f"decoder positions must be between 2 and {self.max_span + 1}")
        condition = self.hidden_projection(hidden_state)
        queries = self.decoder_queries[:, :positions].expand(hidden_state.shape[0], -1, -1)
        return self.byte_head(self.decoder(queries, condition)), self.control_selector(hidden_state)

    @torch.no_grad()
    def decode_span(self, hidden_state: Tensor) -> list[bytes | None]:
        byte_logits, _ = self.decode_logits(hidden_state)
        generated = byte_logits.argmax(dim=-1)
        return decode_span_rows(generated, max_span=self.max_span)

    def optimizer_parameter_groups(self) -> tuple[tuple[Parameter, ...], tuple[Parameter, ...]]:
        hidden_modules = (self.hidden_projection, self.decoder)
        muon_ids = {id(parameter) for module in hidden_modules for parameter in module.parameters() if parameter.ndim == 2}
        trainable = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        muon = tuple(parameter for parameter in trainable if id(parameter) in muon_ids)
        adamw = tuple(parameter for parameter in trainable if id(parameter) not in muon_ids)
        return muon, adamw

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        return self.decode_logits(hidden_state)


def decode_output_batch(
    codec: OutputByteCodec,
    hidden_state: Tensor,
    *,
    maximum_rows: int | None = None,
) -> tuple[Tensor, Tensor]:
    rows = hidden_state.shape[0]
    if rows < 1 or (maximum_rows is not None and rows > maximum_rows):
        raise ValueError("output decode batch exceeds the configured batch size")
    padded_rows = 1 << (rows - 1).bit_length() if codec.device.type == "mps" else rows
    if padded_rows != rows:
        hidden_state = F.pad(hidden_state, (0, 0, 0, padded_rows - rows))
    byte_logits, control_logits = codec(hidden_state)
    return byte_logits[:rows], control_logits[:rows]
