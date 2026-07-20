from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol, final

import torch
from torch import Tensor

from continuous_tokenizer.codec.constants import CODEC_EOS
from continuous_tokenizer.input.segmentation import EncodedSpan

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from continuous_tokenizer.backbone.assets import ModelAssets


class IndependentDecoder(Protocol):
    def decode_logits(
        self,
        latent: Tensor,
        output_positions: int | None = None,
    ) -> Tensor: ...


def _add_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _tensor_bytes(tensor: Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_sha256(name: str, tensor: Tensor) -> str:
    digest = hashlib.sha256()
    _add_bytes(digest, name.encode())
    _add_bytes(digest, str(tensor.dtype).encode())
    for dimension in tensor.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    _add_bytes(digest, _tensor_bytes(tensor))
    return digest.hexdigest()


def _mapping_sha256(rows: Iterable[tuple[int, bytes]]) -> str:
    digest = hashlib.sha256()
    for token_id, payload in rows:
        digest.update(token_id.to_bytes(8, "big"))
        _add_bytes(digest, payload)
    return digest.hexdigest()


def input_source_identity(assets: ModelAssets) -> dict[str, Any]:
    vocabulary = assets.vocabulary
    embeddings = assets.input_embeddings
    byte_ids = torch.tensor(vocabulary.byte_token_ids, dtype=torch.long)
    control_ids = torch.tensor(vocabulary.control_ids, dtype=torch.long)
    canonical_rows = tuple((token_id, vocabulary.bytes_for(token_id)) for token_id in vocabulary.compatibility_ids)
    return {
        "embedding_tensor": {
            "name": assets.embedding_tensor_name,
            "dtype": str(embeddings.dtype),
            "shape": list(embeddings.shape),
            "sha256": _tensor_sha256(assets.embedding_tensor_name, embeddings),
        },
        "canonical_vocabulary_rows": len(canonical_rows),
        "canonical_vocabulary_sha256": _mapping_sha256(canonical_rows),
        "atomic_byte_token_ids": list(vocabulary.byte_token_ids),
        "atomic_byte_rows_sha256": _tensor_sha256(
            "atomic_byte_rows",
            embeddings.index_select(0, byte_ids),
        ),
        "structural_control_ids": list(vocabulary.control_ids),
        "structural_control_rows_sha256": _tensor_sha256(
            "structural_control_rows",
            embeddings.index_select(0, control_ids),
        ),
    }


def _require_exact_rows(stored: Tensor, expected: Tensor, message: str) -> None:
    stored = stored.detach().cpu()
    if stored.dtype != expected.dtype or not torch.equal(stored, expected):
        raise ValueError(message)


def verify_checkpoint_source(
    assets: ModelAssets,
    metadata: dict[str, Any],
    byte_embeddings: Tensor,
    control_ids: Tensor,
    control_embeddings: Tensor,
) -> None:
    if metadata.get("source_identity") != input_source_identity(assets):
        raise ValueError("checkpoint source identity does not match the loaded model assets")
    vocabulary = assets.vocabulary
    expected_byte_rows = assets.input_embeddings[torch.tensor(vocabulary.byte_token_ids, dtype=torch.long)]
    _require_exact_rows(
        byte_embeddings,
        expected_byte_rows,
        "checkpoint atomic byte rows do not match the source embedding tensor",
    )
    expected_control_ids = torch.tensor(vocabulary.control_ids, dtype=torch.long)
    if not torch.equal(control_ids.detach().cpu(), expected_control_ids):
        raise ValueError("checkpoint structural control IDs do not match the source vocabulary")
    expected_control_rows = assets.input_embeddings.index_select(0, expected_control_ids)
    _require_exact_rows(
        control_embeddings,
        expected_control_rows,
        "checkpoint structural control rows do not match the source embedding tensor",
    )


def alias_group_evidence(assets: ModelAssets) -> list[dict[str, Any]]:
    vocabulary = assets.vocabulary
    canonical_by_payload = {vocabulary.bytes_for(token_id): token_id for token_id in vocabulary.compatibility_ids}
    aliases_by_payload: dict[bytes, list[int]] = {}
    for token_id in vocabulary.alias_ids:
        aliases_by_payload.setdefault(vocabulary.bytes_for(token_id), []).append(token_id)

    groups = []
    for payload, alias_ids in sorted(
        aliases_by_payload.items(),
        key=lambda item: (item[0], item[1]),
    ):
        canonical_id = canonical_by_payload[payload]
        canonical = assets.input_embeddings[canonical_id]
        rows = []
        for alias_id in sorted(alias_ids):
            alias = assets.input_embeddings[alias_id]
            difference = alias.float() - canonical.float()
            rows.append(
                {
                    "alias_id": alias_id,
                    "source_row_equal": torch.equal(alias, canonical),
                    "source_row_l2_distance": math.sqrt(
                        float(difference.square().sum().item()),
                    ),
                    "source_row_maximum_absolute_distance": float(
                        difference.abs().max().item(),
                    ),
                }
            )
        equal = all(bool(row["source_row_equal"]) for row in rows)
        groups.append(
            {
                "payload_hex": payload.hex(),
                "canonical_id": canonical_id,
                "alias_ids": sorted(alias_ids),
                "source_rows": rows,
                "applicability": ("applicable_equal_source_rows" if equal else "inapplicable_noncanonical_source_row_disagreement"),
            }
        )
    return groups


@final
@dataclass(frozen=True, slots=True)
class SpanEvidence:
    start: int
    end: int
    payload_hex: str
    atomic: bool
    source_dtype_latent_hex: str
    independent_decoded_hex: str
    eos_exact: bool | None
    empirical_match: bool


@final
@dataclass(frozen=True, slots=True)
class SegmentationEvidence:
    semantic_sha256: str
    source_bookkeeping_round_trip: bool
    empirical_round_trip: bool
    rows: tuple[SpanEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_sha256": self.semantic_sha256,
            "source_bookkeeping_round_trip": self.source_bookkeeping_round_trip,
            "empirical_round_trip": self.empirical_round_trip,
            "rows": [asdict(row) for row in self.rows],
        }


def _decoded_rows(
    codec: IndependentDecoder,
    spans: Sequence[EncodedSpan],
) -> dict[int, tuple[bytes, bool]]:
    selected = [(index, span) for index, span in enumerate(spans) if not span.atomic]
    if not selected:
        return {}
    latents = torch.stack([span.latent for _, span in selected])
    maximum_length = max(len(span.data) for _, span in selected)
    generated = codec.decode_logits(latents, maximum_length + 1).argmax(dim=-1)
    decoded: dict[int, tuple[bytes, bool]] = {}
    for (index, span), row in zip(selected, generated, strict=True):
        values = row.detach().cpu().tolist()
        try:
            eos_position = values.index(CODEC_EOS)
        except ValueError:
            eos_position = len(values)
        payload = bytes(value for value in values[:eos_position] if 0 <= value < 256)
        decoded[index] = (payload, eos_position == len(span.data))
    return decoded


@torch.inference_mode()
def segmentation_evidence(
    codec: IndependentDecoder,
    spans: Sequence[EncodedSpan],
    source: bytes,
    *,
    source_dtype: torch.dtype,
) -> SegmentationEvidence:
    decoded = _decoded_rows(codec, spans)
    rows = []
    position = 0
    empirical = bytearray()
    digest = hashlib.sha256()
    for index, span in enumerate(spans):
        start = position
        position += len(span.data)
        if span.atomic:
            reconstructed = span.data
            eos_exact = None
        else:
            reconstructed, eos_exact = decoded[index]
        empirical.extend(reconstructed)
        latent_bytes = _tensor_bytes(span.latent.to(dtype=source_dtype))
        match = reconstructed == span.data and eos_exact is not False
        row = SpanEvidence(
            start=start,
            end=position,
            payload_hex=span.data.hex(),
            atomic=span.atomic,
            source_dtype_latent_hex=latent_bytes.hex(),
            independent_decoded_hex=reconstructed.hex(),
            eos_exact=eos_exact,
            empirical_match=match,
        )
        rows.append(row)
        digest.update(start.to_bytes(8, "big"))
        digest.update(position.to_bytes(8, "big"))
        _add_bytes(digest, span.data)
        digest.update(bytes([span.atomic]))
        _add_bytes(digest, latent_bytes)
        _add_bytes(digest, reconstructed)
        digest.update(bytes([2 if eos_exact is None else eos_exact]))
    return SegmentationEvidence(
        semantic_sha256=digest.hexdigest(),
        source_bookkeeping_round_trip=b"".join(span.data for span in spans) == source,
        empirical_round_trip=bytes(empirical) == source and all(row.empirical_match for row in rows),
        rows=tuple(rows),
    )
