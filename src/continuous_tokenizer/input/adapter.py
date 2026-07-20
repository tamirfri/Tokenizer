from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, final

import torch
from torch import Tensor, nn

from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.checkpoints import cache_namespace, load_checkpoint
from continuous_tokenizer.codec.encoding_cache import EncodingCache
from continuous_tokenizer.codec.input import InputByteCodec
from continuous_tokenizer.input.evidence import verify_checkpoint_source
from continuous_tokenizer.input.segmentation import (
    DYNAMIC_SEGMENTATION_MAX_BYTES,
    EncodedSpan,
    encode_spans,
    greedy_segment,
    validate_spans,
)


@final
@dataclass(frozen=True, slots=True)
class ByteRun:
    data: bytes

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("byte runs must not be empty")


@final
@dataclass(frozen=True, slots=True)
class ControlToken:
    token_id: int

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("control token IDs must be non-negative")


type InputPiece = ByteRun | ControlToken
type InputPosition = EncodedSpan | ControlToken
type InputMode = Literal["compatibility", "segmented"]
type SegmentationAlignment = Literal["aligned", "arbitrary"]


@final
@dataclass(frozen=True, slots=True)
class InputEncoding:
    embeddings: Tensor
    positions: tuple[InputPosition, ...]
    position_ids: Tensor

    def __post_init__(self) -> None:
        if self.embeddings.shape[0] != len(self.positions):
            raise ValueError("embeddings and positions must have equal lengths")
        if self.position_ids.shape != (len(self.positions),):
            raise ValueError("position IDs must have one value per position")

    @classmethod
    def from_positions(
        cls,
        embeddings: Sequence[Tensor],
        positions: Sequence[InputPosition],
        position_ids: Tensor,
    ) -> InputEncoding:
        if len(embeddings) != len(positions):
            raise ValueError("embeddings and positions must have equal lengths")
        if not embeddings:
            raise ValueError("input positions must not be empty")
        stacked = torch.stack(tuple(embeddings))
        owned_positions = tuple(
            EncodedSpan(position.data, stacked[index], position.atomic) if isinstance(position, EncodedSpan) else position
            for index, position in enumerate(positions)
        )
        return cls(stacked, owned_positions, position_ids)

    @property
    def atomic_spans(self) -> int:
        return sum(position.atomic for position in self.positions if isinstance(position, EncodedSpan))


@final
class InputEmbeddingAdapter(nn.Module):
    def __init__(
        self,
        codec: InputByteCodec,
        vocabulary: ByteVocabulary,
        control_ids: Tensor,
        control_embeddings: Tensor,
        *,
        namespace: str,
    ) -> None:
        super().__init__()
        if control_ids.ndim != 1:
            raise ValueError("control IDs must be one-dimensional")
        if control_embeddings.shape != (control_ids.numel(), codec.config.embedding_dim):
            raise ValueError("control embeddings must match control IDs and embedding dimension")
        if tuple(sorted(control_ids.tolist())) != tuple(control_ids.tolist()):
            raise ValueError("control IDs must be sorted")
        if tuple(control_ids.tolist()) != vocabulary.control_ids:
            raise ValueError("checkpoint controls do not match the tokenizer vocabulary")

        self.codec = codec
        self.vocabulary = vocabulary
        self.namespace = namespace
        self.control_ids: Tensor
        self.control_embeddings: Tensor
        self.register_buffer("control_ids", control_ids.detach().clone())
        self.register_buffer("control_embeddings", control_embeddings.detach().clone())
        self._control_rows = {token_id: row for row, token_id in enumerate(control_ids.detach().cpu().tolist())}

    @classmethod
    def from_checkpoint(
        cls,
        assets: ModelAssets,
        checkpoint: Path,
        *,
        device: torch.device,
    ) -> LoadedInputAdapter:
        loaded = load_checkpoint(checkpoint, device=device)
        if loaded.metadata.get("model_revision") != assets.revision:
            raise ValueError("checkpoint and source model revisions do not match")
        verify_checkpoint_source(
            assets,
            loaded.metadata,
            loaded.codec.byte_embeddings,
            loaded.controls.ids,
            loaded.controls.embeddings,
        )
        fingerprint = sha256_file(checkpoint)
        adapter = cls(
            loaded.codec,
            assets.vocabulary,
            loaded.controls.ids,
            loaded.controls.embeddings,
            namespace=cache_namespace(assets.revision, fingerprint),
        )
        return LoadedInputAdapter(adapter, loaded.metadata, fingerprint)

    @property
    def device(self) -> torch.device:
        return self.codec.device

    def pieces_from_token_ids(self, token_ids: Sequence[int]) -> tuple[InputPiece, ...]:
        pieces: list[InputPiece] = []
        pending = bytearray()
        for token_id in token_ids:
            value = self.vocabulary.payload_for(token_id)
            if value is not None:
                pending.extend(value)
                continue
            if pending:
                pieces.append(ByteRun(bytes(pending)))
                pending.clear()
            pieces.append(ControlToken(token_id))
        if pending:
            pieces.append(ByteRun(bytes(pending)))
        return tuple(pieces)

    def _control_embedding(self, token_id: int) -> Tensor:
        try:
            row = self._control_rows[token_id]
        except KeyError as error:
            raise ValueError(f"token ID {token_id} is not a known control token") from error
        return self.control_embeddings[row]

    def encode_compatibility(
        self,
        token_ids: Sequence[int],
        *,
        cache: EncodingCache | None = None,
        position_offset: int = 0,
    ) -> InputEncoding:
        ordinary: list[tuple[int, bytes]] = []
        values: list[Tensor | None] = []
        positions: list[InputPosition | None] = []
        for token_id in token_ids:
            value = self.vocabulary.payload_for(token_id)
            if value is None:
                values.append(self._control_embedding(token_id))
                positions.append(ControlToken(token_id))
            else:
                ordinary.append((len(values), value))
                values.append(None)
                positions.append(None)

        if ordinary:
            encoded = encode_spans(
                self.codec,
                [value for _, value in ordinary],
                cache=cache,
                namespace=self.namespace,
            )
            for latent, (index, value) in zip(encoded, ordinary, strict=True):
                values[index] = latent
                positions[index] = EncodedSpan(value, latent, atomic=len(value) == 1)
        if not values or any(value is None for value in values):
            raise ValueError("token IDs must produce at least one input position")
        return InputEncoding.from_positions(
            [value for value in values if value is not None],
            [position for position in positions if position is not None],
            torch.arange(
                position_offset,
                position_offset + len(positions),
                dtype=torch.long,
                device=self.device,
            ),
        )

    def encode_pieces(
        self,
        pieces: Sequence[InputPiece],
        *,
        cache: EncodingCache | None = None,
        position_offset: int = 0,
    ) -> InputEncoding:
        embeddings: list[Tensor] = []
        positions: list[InputPosition] = []
        for piece in pieces:
            if isinstance(piece, ControlToken):
                embeddings.append(self._control_embedding(piece.token_id))
                positions.append(piece)
                continue
            spans = greedy_segment(
                self.codec,
                piece.data,
                cache=cache,
                namespace=self.namespace,
            )
            embeddings.extend(span.latent for span in spans)
            positions.extend(spans)
        if not embeddings:
            raise ValueError("input pieces must produce at least one input position")
        return InputEncoding.from_positions(
            embeddings,
            positions,
            torch.arange(
                position_offset,
                position_offset + len(positions),
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _encode_aligned(
        self,
        token_ids: Sequence[int],
        *,
        cache: EncodingCache | None,
        position_offset: int,
    ) -> InputEncoding:
        embeddings: list[Tensor] = []
        positions: list[InputPosition] = []
        position_ids: list[int] = []
        index = 0
        candidate_limit = min(DYNAMIC_SEGMENTATION_MAX_BYTES, self.codec.max_span)
        while index < len(token_ids):
            token_id = token_ids[index]
            first = self.vocabulary.payload_for(token_id)
            if first is None:
                embeddings.append(self._control_embedding(token_id))
                positions.append(ControlToken(token_id))
                position_ids.append(position_offset + index)
                index += 1
                continue

            candidates: list[tuple[int, bytes]] = []
            combined = bytearray(first)
            end = index + 1
            while end < len(token_ids):
                value = self.vocabulary.payload_for(token_ids[end])
                if value is None or len(combined) + len(value) > candidate_limit:
                    break
                combined.extend(value)
                candidates.append((end, bytes(combined)))
                end += 1

            selected_end = index
            selected_data = first
            selected_latent: Tensor | None = None
            if candidates:
                latents, matches = validate_spans(
                    self.codec,
                    [data for _, data in candidates],
                    static_size=candidate_limit,
                    cache=cache,
                    namespace=self.namespace,
                )
                for (candidate_end, data), latent, matches_source in zip(candidates, latents, matches.tolist(), strict=True):
                    if matches_source:
                        selected_end = candidate_end
                        selected_data = data
                        selected_latent = latent
            if selected_latent is None:
                selected_latent = encode_spans(
                    self.codec,
                    [first],
                    cache=cache,
                    namespace=self.namespace,
                )[0]
            embeddings.append(selected_latent)
            positions.append(EncodedSpan(selected_data, selected_latent, atomic=len(selected_data) == 1))
            position_ids.append(position_offset + selected_end)
            index = selected_end + 1

        if not embeddings:
            raise ValueError("token IDs must produce at least one input position")
        return InputEncoding.from_positions(
            embeddings,
            positions,
            torch.tensor(position_ids, dtype=torch.long, device=self.device),
        )

    def _encode_arbitrary(
        self,
        token_ids: Sequence[int],
        *,
        cache: EncodingCache | None,
        position_offset: int,
    ) -> InputEncoding:
        embeddings: list[Tensor] = []
        positions: list[InputPosition] = []
        position_ids: list[int] = []
        run = bytearray()
        byte_ends: list[int] = []
        native_positions: list[int] = []

        def flush() -> None:
            if not run:
                return
            spans = greedy_segment(
                self.codec,
                bytes(run),
                cache=cache,
                namespace=self.namespace,
            )
            end = 0
            for span in spans:
                end += len(span.data)
                native_index = bisect_left(byte_ends, end)
                embeddings.append(span.latent)
                positions.append(span)
                position_ids.append(native_positions[native_index])
            run.clear()
            byte_ends.clear()
            native_positions.clear()

        for index, token_id in enumerate(token_ids):
            value = self.vocabulary.payload_for(token_id)
            if value is None:
                flush()
                embeddings.append(self._control_embedding(token_id))
                positions.append(ControlToken(token_id))
                position_ids.append(position_offset + index)
                continue
            run.extend(value)
            byte_ends.append(len(run))
            native_positions.append(position_offset + index)
        flush()
        if not embeddings:
            raise ValueError("token IDs must produce at least one input position")
        return InputEncoding.from_positions(
            embeddings,
            positions,
            torch.tensor(position_ids, dtype=torch.long, device=self.device),
        )

    def encode_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        mode: InputMode,
        cache: EncodingCache | None = None,
        alignment: SegmentationAlignment = "arbitrary",
        position_offset: int = 0,
    ) -> InputEncoding:
        if mode == "compatibility":
            return self.encode_compatibility(
                token_ids,
                cache=cache,
                position_offset=position_offset,
            )
        if alignment == "aligned":
            return self._encode_aligned(
                token_ids,
                cache=cache,
                position_offset=position_offset,
            )
        return self._encode_arbitrary(
            token_ids,
            cache=cache,
            position_offset=position_offset,
        )

    def encode_bytes(
        self,
        data: bytes,
        *,
        cache: EncodingCache | None = None,
    ) -> InputEncoding:
        return self.encode_pieces((ByteRun(data),), cache=cache)


@final
@dataclass(frozen=True, slots=True)
class LoadedInputAdapter:
    adapter: InputEmbeddingAdapter
    metadata: dict[str, Any]
    fingerprint: str
