from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final, final

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.profiler import record_function

from continuous_tokenizer.codec.batches import decode_span_rows
from continuous_tokenizer.codec.compilation import (
    TOKENIZER_RECOMPILE_LIMIT,
    compile_fullgraph,
    planned_input_graph_signatures,
)
from continuous_tokenizer.codec.constants import BYTE_VALUES, CODEC_EOS, LOCAL_OUTPUT_VALUES
from continuous_tokenizer.codec.encoding_cache import EncodingCache
from continuous_tokenizer.codec.layers import KEY_VALUE_HEADS, ByteSpanDecoder, ByteSpanEncoder
from continuous_tokenizer.runtime.compiler import configure_compiler_cache
from continuous_tokenizer.runtime.tensors import tensor_fingerprint


def _tensor_version(value: Tensor | None) -> int:
    if value is None:
        return -1
    try:
        return value._version  # noqa: SLF001
    except RuntimeError as error:
        if "Inference tensors do not track version counter" not in str(error):
            raise
        return -1


@final
@dataclass(frozen=True, slots=True)
class InputByteCodecConfig:
    embedding_dim: int
    local_dim: int
    projection_dim: int
    max_span: int
    query_heads: int
    feedforward_dim: int
    encoder_layers: int
    decoder_layers: int

    def __post_init__(self) -> None:
        if (
            min(
                self.embedding_dim,
                self.local_dim,
                self.projection_dim,
                self.query_heads,
                self.feedforward_dim,
                self.encoder_layers,
                self.decoder_layers,
            )
            < 1
        ):
            raise ValueError("codec dimensions and layer counts must be positive")
        if not 1 <= self.max_span <= 255:
            raise ValueError("max_span must fit in one byte")
        if self.query_heads <= KEY_VALUE_HEADS or self.query_heads % KEY_VALUE_HEADS:
            raise ValueError("query_heads must be greater than and divisible by 2")
        if self.local_dim % self.query_heads:
            raise ValueError("local_dim must be divisible by query_heads")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class InputByteCodec(nn.Module):
    def __init__(
        self,
        config: InputByteCodecConfig,
        byte_embeddings: Tensor,
    ) -> None:
        super().__init__()
        if byte_embeddings.shape != (256, config.embedding_dim):
            raise ValueError("byte_embeddings must have shape [256, embedding_dim]")

        self.config = config
        self.encoding_cache: Final = EncodingCache()
        self._projected_byte_table: Tensor | None = None
        self._projected_byte_table_versions: tuple[int, int, int] | None = None
        self._compiled_encode: Callable[[Tensor, Tensor], Tensor] | None = None
        self._compiled_decode: Callable[[Tensor, int | None], Tensor] | None = None
        self._compiled_matches: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
        self._compiled_reconstruction: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None
        self._compiled_validation: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None
        self._planned_graph_signatures: dict[str, tuple[str, ...]] = {}
        self._encountered_graph_signatures: dict[str, set[str]] = {}
        self.byte_embeddings: Tensor
        self.register_buffer("byte_embeddings", byte_embeddings.detach().clone())

        self.input_projection = nn.Linear(config.embedding_dim, config.local_dim)
        self.decoder_projection = nn.Linear(config.embedding_dim, config.local_dim)
        self.cls = nn.Parameter(torch.empty(1, 1, config.local_dim))
        self.encoder_positions = nn.Parameter(torch.empty(1, config.max_span + 1, config.local_dim))
        self.decoder_queries = nn.Parameter(torch.empty(1, config.max_span + 1, config.local_dim))

        residual_projection = nn.Linear(config.projection_dim, config.embedding_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(config.local_dim, config.projection_dim, bias=False),
            nn.GELU(),
            residual_projection,
        )
        self.encoder = ByteSpanEncoder(
            config.local_dim,
            config.query_heads,
            config.feedforward_dim,
            config.encoder_layers,
        )
        self.decoder = ByteSpanDecoder(
            config.local_dim,
            config.feedforward_dim,
            config.decoder_layers,
        )
        self.byte_head = nn.Linear(config.local_dim, LOCAL_OUTPUT_VALUES)
        self._initialize_parameters(residual_projection)

    @property
    def max_span(self) -> int:
        return self.config.max_span

    @property
    def device(self) -> torch.device:
        return self.byte_embeddings.device

    @property
    def dtype(self) -> torch.dtype:
        return self.input_projection.weight.dtype

    @property
    def neural_paths_compiled(self) -> bool:
        return all(
            operation is not None
            for operation in (
                self._compiled_encode,
                self._compiled_decode,
                self._compiled_matches,
                self._compiled_reconstruction,
                self._compiled_validation,
            )
        )

    @property
    def projected_byte_table_cached(self) -> bool:
        return self._projected_byte_table is not None

    def _initialize_parameters(self, residual_projection: nn.Linear) -> None:
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.encoder_positions, std=0.02)
        nn.init.normal_(self.decoder_queries, std=0.02)
        nn.init.zeros_(residual_projection.weight)
        nn.init.zeros_(residual_projection.bias)

    def compile_neural_paths(
        self,
        *,
        backend: str = "inductor",
        static_rows: tuple[int, ...] = (64,),
    ) -> None:
        """Compile tensor-only operations and complete tokenizer workloads as full graphs."""
        self.clear_runtime_caches()
        self._planned_graph_signatures = planned_input_graph_signatures(
            self.max_span,
            tuple(sorted(set(static_rows))),
        )
        self._encountered_graph_signatures = {operation: set() for operation in self._planned_graph_signatures}
        configure_compiler_cache(backend)
        self._compiled_encode = compile_fullgraph(self._encode_tensor, backend=backend)
        self._compiled_decode = compile_fullgraph(self._decode_tensor, backend=backend)
        self._compiled_matches = compile_fullgraph(
            self._reconstruction_matches_tensor,
            backend=backend,
        )
        self._compiled_reconstruction = compile_fullgraph(
            self._reconstruction_tensor,
            backend=backend,
        )
        self._compiled_validation = compile_fullgraph(
            self._validation_tensor,
            backend=backend,
        )

    def graph_signature_telemetry(self) -> dict[str, Any]:
        return {
            "limit": TOKENIZER_RECOMPILE_LIMIT,
            "planned": {operation: len(signatures) for operation, signatures in self._planned_graph_signatures.items()},
            "encountered": {operation: sorted(signatures) for operation, signatures in self._encountered_graph_signatures.items()},
        }

    def _record_graph_signature(self, operation: str, rows: int, width: int) -> None:
        signatures = self._encountered_graph_signatures.setdefault(operation, set())
        signature = f"{rows}x{width}"
        if signature in signatures:
            return
        if len(signatures) >= TOKENIZER_RECOMPILE_LIMIT:
            raise RuntimeError(f"{operation} would exceed the {TOKENIZER_RECOMPILE_LIMIT}-signature compiled tokenizer policy with {signature}")
        signatures.add(signature)

    def train(self, mode: bool = True) -> InputByteCodec:
        if mode:
            self.clear_runtime_caches()
        return super().train(mode)

    def _clear_projected_byte_table(self) -> None:
        self._projected_byte_table = None
        self._projected_byte_table_versions = None

    def clear_runtime_caches(self) -> None:
        self._clear_projected_byte_table()
        self.encoding_cache.clear()

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> InputByteCodec:
        result = super()._apply(fn, recurse=recurse)
        self._compiled_encode = None
        self._compiled_decode = None
        self._compiled_matches = None
        self._compiled_reconstruction = None
        self._compiled_validation = None
        self.clear_runtime_caches()
        return result

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> tuple[list[str], list[str]]:
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.clear_runtime_caches()
        return result

    def encode(self, byte_values: Tensor, valid_mask: Tensor) -> Tensor:
        if byte_values.shape[1] == 1:
            return self.byte_embeddings[byte_values[:, 0]].to(self.dtype)
        if self._compiled_encode is not None:
            self._record_graph_signature(
                "encode",
                byte_values.shape[0],
                byte_values.shape[1],
            )
            return self._compiled_encode(byte_values, valid_mask)
        return self._encode_tensor(byte_values, valid_mask)

    def _encode_tensor(self, byte_values: Tensor, valid_mask: Tensor) -> Tensor:
        if byte_values.shape != valid_mask.shape:
            raise ValueError("byte_values and valid_mask must have the same shape")
        input_span = byte_values.shape[1]
        if not 1 <= input_span <= self.max_span:
            raise ValueError(f"encoder input width must be between 1 and {self.max_span}")

        lengths = valid_mask.sum(dim=1)
        batch_size = byte_values.shape[0]
        embedding_table = self.byte_embeddings.to(self.dtype)
        direct = embedding_table[byte_values[:, 0]]
        position_weights = torch.arange(
            1,
            input_span + 1,
            device=byte_values.device,
            dtype=self.dtype,
        )
        position_weights = position_weights.unsqueeze(0) * valid_mask.to(self.dtype)
        flat_values = byte_values.flatten()
        offsets = torch.arange(
            0,
            flat_values.numel(),
            input_span,
            device=byte_values.device,
        )
        baseline = F.embedding_bag(
            flat_values,
            embedding_table,
            offsets,
            mode="sum",
            per_sample_weights=position_weights.flatten(),
        )
        baseline /= position_weights.sum(dim=1, keepdim=True).clamp_min(1)
        with record_function("tokenizer.encoder_input_projection"):
            projected_table = self._project_byte_table(embedding_table)
            projected = F.embedding(byte_values, projected_table)
            cls = self.cls.expand(batch_size, -1, -1)
            source = torch.cat((cls, projected), dim=1) + self.encoder_positions[:, : input_span + 1]
            cls_valid = torch.ones(
                (batch_size, 1),
                dtype=torch.bool,
                device=valid_mask.device,
            )
            full_valid = torch.cat((cls_valid, valid_mask), dim=1)
        with record_function("tokenizer.encoder_layers"):
            encoded = self.encoder(source, full_valid)
        with record_function("tokenizer.encoder_output_projection"):
            latent = baseline + self.output_projection(encoded[:, 0])

        single = lengths == 1
        return torch.where(single[:, None], direct, latent)

    def _project_byte_table(self, embedding_table: Tensor) -> Tensor:
        inference_cache_allowed = not self.training and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
        if not inference_cache_allowed:
            return self.input_projection(embedding_table)
        versions = (
            _tensor_version(self.byte_embeddings),
            _tensor_version(self.input_projection.weight),
            _tensor_version(self.input_projection.bias),
        )
        if self._projected_byte_table is None or self._projected_byte_table_versions != versions:
            self._projected_byte_table = self.input_projection(embedding_table)
            self._projected_byte_table_versions = versions
        return self._projected_byte_table

    def atomic_latent(self, value: int) -> Tensor:
        if not 0 <= value < BYTE_VALUES:
            raise ValueError("atomic byte value must be between 0 and 255")
        return self.byte_embeddings[value].to(self.dtype)

    def training_parameter_groups(
        self,
    ) -> tuple[tuple[Parameter, ...], tuple[Parameter, ...]]:
        encoder = (
            self.cls,
            self.encoder_positions,
            *self.input_projection.parameters(),
            *self.encoder.parameters(),
            *self.output_projection.parameters(),
        )
        decoder = (
            self.decoder_queries,
            *self.decoder_projection.parameters(),
            *self.decoder.parameters(),
            *self.byte_head.parameters(),
        )
        encoder_ids = {id(parameter) for parameter in encoder}
        decoder_ids = {id(parameter) for parameter in decoder}
        all_ids = {id(parameter) for parameter in self.parameters()}
        if encoder_ids & decoder_ids or encoder_ids | decoder_ids != all_ids:
            raise RuntimeError("codec parameters do not form disjoint encoder and decoder groups")
        return encoder, decoder

    def encoder_fingerprint(self) -> str:
        encoder_parameters, _ = self.training_parameter_groups()
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        tensors = [("byte_embeddings", self.byte_embeddings)]
        tensors.extend((name, parameter) for name, parameter in self.named_parameters() if id(parameter) in encoder_ids)
        return tensor_fingerprint(tensors)

    @torch.no_grad()
    def load_decoder_state(self, source: InputByteCodec) -> None:
        if self.config != source.config:
            raise ValueError("decoder state source has a different codec configuration")
        _, target_parameters = self.training_parameter_groups()
        _, source_parameters = source.training_parameter_groups()
        for target, value in zip(target_parameters, source_parameters, strict=True):
            target.copy_(value)

    def set_trainable_components(
        self,
        *,
        encoder: bool,
        decoder: bool,
    ) -> tuple[Parameter, ...]:
        self._clear_projected_byte_table()
        encoder_parameters, decoder_parameters = self.training_parameter_groups()
        for parameter in encoder_parameters:
            parameter.requires_grad_(encoder)
        for parameter in decoder_parameters:
            parameter.requires_grad_(decoder)
        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)

    def optimizer_parameter_groups(
        self,
        trainable_parameters: tuple[Parameter, ...],
    ) -> tuple[tuple[Parameter, ...], tuple[Parameter, ...]]:
        hidden_parameters = (
            *self.input_projection.parameters(),
            *self.decoder_projection.parameters(),
            *self.output_projection.parameters(),
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        )
        hidden_matrix_ids = {id(parameter) for parameter in hidden_parameters if parameter.ndim == 2}
        muon_parameters = tuple(parameter for parameter in trainable_parameters if id(parameter) in hidden_matrix_ids)
        muon_ids = {id(parameter) for parameter in muon_parameters}
        adamw_parameters = tuple(parameter for parameter in trainable_parameters if id(parameter) not in muon_ids)
        return muon_parameters, adamw_parameters

    def decode_logits(self, latent: Tensor, output_positions: int | None = None) -> Tensor:
        if self._compiled_decode is not None:
            positions = self.max_span + 1 if output_positions is None else output_positions
            self._record_graph_signature("decode", latent.shape[0], positions)
            return self._compiled_decode(latent, output_positions)
        return self._decode_tensor(latent, output_positions)

    def _decode_tensor(self, latent: Tensor, output_positions: int | None = None) -> Tensor:
        positions = self.max_span + 1 if output_positions is None else output_positions
        if not 2 <= positions <= self.max_span + 1:
            raise ValueError(f"decoder positions must be between 2 and {self.max_span + 1}")
        queries = self.decoder_queries[:, :positions].expand(latent.shape[0], -1, -1)
        condition = self.decoder_projection(latent)
        decoded = self.decoder(queries, condition)
        return self.byte_head(decoded)

    def reconstruction_matches(
        self,
        latent: Tensor,
        byte_values: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if self._compiled_matches is not None:
            self._record_graph_signature(
                "matches",
                byte_values.shape[0],
                byte_values.shape[1],
            )
            return self._compiled_matches(latent, byte_values, valid_mask)
        return self._reconstruction_matches_tensor(latent, byte_values, valid_mask)

    def reconstruction_logits(
        self,
        byte_values: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self._compiled_reconstruction is not None:
            self._record_graph_signature(
                "reconstruction",
                byte_values.shape[0],
                byte_values.shape[1],
            )
            return self._compiled_reconstruction(byte_values, valid_mask)
        latent = self.encode(byte_values, valid_mask)
        return latent, self.decode_logits(latent, byte_values.shape[1] + 1)

    def _reconstruction_tensor(
        self,
        byte_values: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        latent = self._encode_tensor(byte_values, valid_mask)
        return latent, self._decode_tensor(latent, byte_values.shape[1] + 1)

    def encode_and_reconstruction_matches(
        self,
        byte_values: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self._compiled_validation is not None:
            self._record_graph_signature(
                "validation",
                byte_values.shape[0],
                byte_values.shape[1],
            )
            return self._compiled_validation(byte_values, valid_mask)
        latent = self.encode(byte_values, valid_mask)
        return latent, self.reconstruction_matches(latent, byte_values, valid_mask)

    def _validation_tensor(
        self,
        byte_values: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        latent = self._encode_tensor(byte_values, valid_mask)
        matches = self._reconstruction_matches_tensor(latent, byte_values, valid_mask)
        return latent, matches

    def _reconstruction_matches_tensor(
        self,
        latent: Tensor,
        byte_values: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        generated = self._decode_tensor(latent, byte_values.shape[1] + 1).argmax(dim=-1)
        payload_matches = ((generated[:, :-1] == byte_values) | ~valid_mask).all(dim=1)
        eos_positions = valid_mask.sum(dim=1, dtype=torch.long)
        eos_matches = generated.gather(1, eos_positions[:, None]).squeeze(1) == CODEC_EOS
        return payload_matches & eos_matches

    @torch.no_grad()
    def decode_greedy(
        self,
        latent: Tensor,
        *,
        maximum_length: int | None = None,
    ) -> list[bytes | None]:
        logits = self.decode_logits(latent) if maximum_length is None else self.decode_logits(latent, maximum_length + 1)
        generated = logits.argmax(dim=-1)
        return decode_span_rows(generated, max_span=self.max_span)

    def forward(self, byte_values: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        return self.reconstruction_logits(byte_values, valid_mask)
