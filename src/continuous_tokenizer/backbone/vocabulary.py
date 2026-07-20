from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Final, final


class VocabularyError(ValueError):
    """Raised when a tokenizer cannot provide an unambiguous byte vocabulary."""


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte alphabet used by GPT-style ByteLevel BPE."""
    # Source: https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/gpt2/tokenization_gpt2.py
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("\u00a1"), ord("\u00ac") + 1))
    visible += list(range(ord("\u00ae"), ord("\u00ff") + 1))
    encoded = visible.copy()
    extra = 0
    for value in range(256):
        if value not in visible:
            visible.append(value)
            encoded.append(256 + extra)
            extra += 1
    return dict(zip(visible, (chr(value) for value in encoded), strict=True))


_UNICODE_TO_BYTE: Final = {character: value for value, character in bytes_to_unicode().items()}
_BYTE_FALLBACK_TOKEN: Final = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def byte_level_token_to_bytes(token: str) -> bytes:
    try:
        return bytes(_UNICODE_TO_BYTE[character] for character in token)
    except KeyError as error:
        character = error.args[0]
        raise VocabularyError(f"token contains a non-ByteLevel character: {character!r}") from error


def _ordinary_token_bytes(tokenizer: Any, token_id: int, token: str) -> bytes:
    fallback = _BYTE_FALLBACK_TOKEN.fullmatch(token)
    if fallback is not None:
        return bytes([int(fallback.group(1), 16)])
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if backend is not None and hasattr(backend, "decode") and decoder is not None and type(decoder).__name__ != "ByteLevel":
        return str(backend.decode([token_id], skip_special_tokens=False)).encode("utf-8")
    try:
        return byte_level_token_to_bytes(token)
    except VocabularyError:
        if backend is None or not hasattr(backend, "decode"):
            raise
        return str(backend.decode([token_id], skip_special_tokens=False)).encode("utf-8")


@final
@dataclass(frozen=True, slots=True)
class ByteVocabulary:
    token_bytes: tuple[bytes | None, ...]
    ordinary_ids: tuple[int, ...]
    control_ids: tuple[int, ...]
    byte_token_ids: tuple[int, ...]
    max_token_bytes: int
    compatibility_ids: tuple[int, ...] = ()
    out_of_table_control_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.compatibility_ids:
            object.__setattr__(self, "compatibility_ids", self.ordinary_ids)

    @property
    def alias_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.ordinary_ids) - set(self.compatibility_ids)))

    @property
    def unavailable_ids(self) -> tuple[int, ...]:
        assigned = set(self.ordinary_ids) | set(self.control_ids)
        return tuple(token_id for token_id in range(len(self.token_bytes)) if token_id not in assigned)

    def payload_for(self, token_id: int) -> bytes | None:
        if not 0 <= token_id < len(self.token_bytes):
            raise VocabularyError(f"token ID {token_id} is outside the embedding table")
        value = self.token_bytes[token_id]
        if value is None and token_id not in self.control_ids:
            raise VocabularyError(f"token ID {token_id} is an unavailable embedding row")
        return value

    def bytes_for(self, token_id: int) -> bytes:
        value = self.payload_for(token_id)
        if value is None:
            raise VocabularyError(f"token ID {token_id} is a control token")
        return value

    def to_summary(self) -> dict[str, int]:
        payload_counts = Counter(self.token_bytes[token_id] for token_id in self.ordinary_ids)
        return {
            "vocabulary_size": len(self.token_bytes),
            "ordinary_tokens": len(self.ordinary_ids),
            "compatibility_tokens": len(self.compatibility_ids),
            "duplicate_aliases": len(self.alias_ids),
            "control_tokens": len(self.control_ids),
            "unavailable_rows": len(self.unavailable_ids),
            "out_of_table_controls": len(self.out_of_table_control_ids),
            "atomic_bytes": len(self.byte_token_ids),
            "max_token_bytes": self.max_token_bytes,
            "ambiguous_byte_sequences": sum(count > 1 for count in payload_counts.values()),
        }


@final
@dataclass(frozen=True, slots=True)
class _VocabularyRows:
    total: int
    inverse: dict[int, str]
    special_ids: set[int]
    out_of_table_control_ids: tuple[int, ...]


def _token_for_id(tokenizer: Any, token_id: int, inverse_vocab: dict[int, str]) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    if model is not None and hasattr(model, "id_to_token"):
        token = model.id_to_token(token_id)
        if token is not None:
            return str(token)
    try:
        return inverse_vocab[token_id]
    except KeyError as error:
        raise VocabularyError(f"tokenizer has no token string for ID {token_id}") from error


def _vocabulary_rows(tokenizer: Any, embedding_rows: int | None) -> _VocabularyRows:
    vocab = tokenizer.get_vocab()
    inverse_vocab = {token_id: token for token, token_id in vocab.items()}
    total_rows = embedding_rows if embedding_rows is not None else max(inverse_vocab) + 1
    added_ids = set(tokenizer.get_added_vocab().values())
    special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    unembedded_ids = {token_id for token_id in set(inverse_vocab) | added_ids if token_id < 0 or token_id >= total_rows}
    if unembedded_ids - special_ids:
        raise VocabularyError("tokenizer IDs exceed the embedding table")
    return _VocabularyRows(
        total=total_rows,
        inverse={token_id: token for token_id, token in inverse_vocab.items() if 0 <= token_id < total_rows},
        special_ids=special_ids,
        out_of_table_control_ids=tuple(sorted(special_ids - set(range(total_rows)))),
    )


def _ordinary_rows(
    tokenizer: Any,
    rows: _VocabularyRows,
) -> tuple[list[bytes | None], list[int], dict[bytes, list[int]]]:
    token_bytes: list[bytes | None] = [None] * rows.total
    ordinary_ids: list[int] = []
    by_bytes: dict[bytes, list[int]] = defaultdict(list)
    for token_id in sorted(rows.inverse):
        if token_id in rows.special_ids:
            continue
        token = _token_for_id(tokenizer, token_id, rows.inverse)
        value = _ordinary_token_bytes(tokenizer, token_id, token)
        if not value:
            raise VocabularyError(f"ordinary token ID {token_id} has an empty byte sequence")
        if len(value) > 255:
            raise VocabularyError(f"token ID {token_id} is {len(value)} bytes; maximum is 255")
        token_bytes[token_id] = value
        ordinary_ids.append(token_id)
        by_bytes[value].append(token_id)
    return token_bytes, ordinary_ids, by_bytes


def _validate_decoder_round_trip(
    tokenizer: Any,
    ordinary_ids: list[int],
    token_bytes: list[bytes | None],
) -> None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "decode"):
        return
    mismatches: list[tuple[int, bytes, str]] = []
    for token_id in ordinary_ids:
        expected = token_bytes[token_id]
        if expected is None:
            continue
        try:
            expected_text = expected.decode("utf-8")
        except UnicodeDecodeError:
            continue
        actual_text = str(backend.decode([token_id], skip_special_tokens=False))
        if actual_text != expected_text:
            mismatches.append((token_id, expected, actual_text))
            if len(mismatches) == 3:
                break
    if mismatches:
        raise VocabularyError(f"tokenizer decoder round-trip mismatch: {mismatches!r}")


def _canonical_rows(
    tokenizer: Any,
    inverse_vocab: dict[int, str],
    by_bytes: dict[bytes, list[int]],
) -> dict[bytes, int]:
    canonical_by_bytes: dict[bytes, int] = {}
    for value, ids in by_bytes.items():
        fallback_ids = [token_id for token_id in ids if _BYTE_FALLBACK_TOKEN.fullmatch(_token_for_id(tokenizer, token_id, inverse_vocab))]
        canonical_by_bytes[value] = fallback_ids[0] if len(fallback_ids) == 1 else min(ids)
    return canonical_by_bytes


def _atomic_byte_ids(canonical_by_bytes: dict[bytes, int]) -> tuple[int, ...]:
    byte_ids = []
    for value in range(256):
        payload = bytes([value])
        if payload not in canonical_by_bytes:
            raise VocabularyError(f"byte {value} has no atomic representation")
        byte_ids.append(canonical_by_bytes[payload])
    return tuple(byte_ids)


def inspect_tokenizer(tokenizer: Any, *, embedding_rows: int | None = None) -> ByteVocabulary:
    rows = _vocabulary_rows(tokenizer, embedding_rows)
    token_bytes, ordinary_ids, by_bytes = _ordinary_rows(tokenizer, rows)
    _validate_decoder_round_trip(tokenizer, ordinary_ids, token_bytes)
    canonical_by_bytes = _canonical_rows(tokenizer, rows.inverse, by_bytes)
    max_token_bytes = max((len(value) for value in by_bytes), default=1)
    compatibility_ids = tuple(sorted(canonical_by_bytes.values()))
    control_ids = tuple(sorted(rows.special_ids & set(range(rows.total))))
    return ByteVocabulary(
        token_bytes=tuple(token_bytes),
        ordinary_ids=tuple(ordinary_ids),
        control_ids=control_ids,
        byte_token_ids=_atomic_byte_ids(canonical_by_bytes),
        max_token_bytes=max_token_bytes,
        compatibility_ids=compatibility_ids,
        out_of_table_control_ids=rows.out_of_table_control_ids,
    )
