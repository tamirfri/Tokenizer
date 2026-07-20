from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from continuous_tokenizer.cache import EncodingCache


@dataclass(frozen=True, slots=True)
class CodecConfig:
    embedding_dim: int
    local_dim: int
    max_span: int
    heads: int
    feedforward_dim: int
    encoder_layers: int
    decoder_layers: int

    def __post_init__(self) -> None:
        if not 1 <= self.max_span <= 255:
            raise ValueError("max_span must fit in one byte")
        if self.local_dim % self.heads:
            raise ValueError("local_dim must be divisible by heads")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ContinuousByteCodec(nn.Module):
    def __init__(
        self,
        config: CodecConfig,
        byte_embeddings: Tensor,
        control_ids: Tensor | None = None,
        control_embeddings: Tensor | None = None,
    ) -> None:
        super().__init__()
        if byte_embeddings.shape != (256, config.embedding_dim):
            raise ValueError("byte_embeddings must have shape [256, embedding_dim]")

        empty_ids = torch.empty(0, dtype=torch.long)
        empty_controls = torch.empty((0, config.embedding_dim), dtype=byte_embeddings.dtype)
        self.config = config
        self.encoding_cache = EncodingCache()
        self.byte_embeddings: Tensor
        self.control_ids: Tensor
        self.control_embeddings: Tensor
        self.register_buffer("byte_embeddings", byte_embeddings.detach().clone())
        self.register_buffer(
            "control_ids", empty_ids if control_ids is None else control_ids.detach().clone()
        )
        self.register_buffer(
            "control_embeddings",
            empty_controls if control_embeddings is None else control_embeddings.detach().clone(),
        )

        self.input_projection = nn.Linear(config.embedding_dim, config.local_dim)
        self.output_projection = nn.Linear(config.local_dim, config.embedding_dim)
        self.memory_projection = nn.Linear(config.embedding_dim, config.local_dim)
        self.cls = nn.Parameter(torch.empty(1, 1, config.local_dim))
        self.encoder_positions = nn.Parameter(torch.empty(1, config.max_span + 1, config.local_dim))
        self.decoder_queries = nn.Parameter(torch.empty(1, config.max_span + 1, config.local_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.local_dim,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.local_dim),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.local_dim,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.local_dim),
        )
        self.byte_head = nn.Linear(config.local_dim, 256)
        self.reset_parameters()

    @property
    def max_span(self) -> int:
        return self.config.max_span

    @property
    def device(self) -> torch.device:
        return self.byte_embeddings.device

    @property
    def dtype(self) -> torch.dtype:
        return self.byte_embeddings.dtype

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.encoder_positions, std=0.02)
        nn.init.normal_(self.decoder_queries, std=0.02)

    def clear_encoding_cache(self) -> None:
        self.encoding_cache.clear()

    def train(self, mode: bool = True) -> ContinuousByteCodec:
        if mode:
            self.clear_encoding_cache()
        return super().train(mode)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> ContinuousByteCodec:
        result = super()._apply(fn, recurse=recurse)
        self.clear_encoding_cache()
        return result

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.clear_encoding_cache()
        return result

    def encode(self, byte_values: Tensor, valid_mask: Tensor) -> Tensor:
        if byte_values.shape != valid_mask.shape:
            raise ValueError("byte_values and valid_mask must have the same shape")
        if byte_values.shape[1] != self.max_span:
            raise ValueError(f"encoder inputs must be padded to max_span={self.max_span}")

        lengths = valid_mask.sum(dim=1)
        direct = self.byte_embeddings[byte_values[:, 0]]
        if bool((lengths == 1).all()):
            return direct

        batch_size = byte_values.shape[0]
        embedded = nn.functional.embedding(byte_values, self.byte_embeddings)
        projected = self.input_projection(embedded)
        cls = self.cls.expand(batch_size, -1, -1)
        source = torch.cat((cls, projected), dim=1) + self.encoder_positions
        cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device)
        full_valid = torch.cat((cls_valid, valid_mask), dim=1)
        encoded = self.encoder(source, src_key_padding_mask=~full_valid)
        latent = self.output_projection(encoded[:, 0])

        single = lengths == 1
        return torch.where(single[:, None], direct, latent)

    def decode_logits(self, latent: Tensor) -> Tensor:
        memory = self.memory_projection(latent).unsqueeze(1)
        queries = self.decoder_queries.expand(latent.shape[0], -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.byte_head(decoded)

    @torch.no_grad()
    def decode_greedy(self, latent: Tensor) -> list[bytes | None]:
        generated = self.decode_logits(latent).argmax(dim=-1)
        results: list[bytes | None] = []
        for row in generated:
            length = int(row[0].item())
            if not 1 <= length <= self.max_span:
                results.append(None)
                continue
            results.append(bytes(row[1 : length + 1].tolist()))
        return results

    def forward(self, byte_values: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(byte_values, valid_mask)
        return latent, self.decode_logits(latent)
