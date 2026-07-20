from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


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


_UNICODE_TO_BYTE = {character: value for value, character in bytes_to_unicode().items()}


def byte_level_token_to_bytes(token: str) -> bytes:
    try:
        return bytes(_UNICODE_TO_BYTE[character] for character in token)
    except KeyError as error:
        character = error.args[0]
        raise VocabularyError(f"token contains a non-ByteLevel character: {character!r}") from error


@dataclass(frozen=True, slots=True)
class ByteVocabulary:
    token_bytes: tuple[bytes | None, ...]
    ordinary_ids: tuple[int, ...]
    control_ids: tuple[int, ...]
    byte_token_ids: tuple[int, ...]
    max_token_bytes: int

    def bytes_for(self, token_id: int) -> bytes:
        value = self.token_bytes[token_id]
        if value is None:
            raise VocabularyError(f"token ID {token_id} is a control token")
        return value

    def to_summary(self) -> dict[str, int]:
        return {
            "vocabulary_size": len(self.token_bytes),
            "ordinary_tokens": len(self.ordinary_ids),
            "control_tokens": len(self.control_ids),
            "atomic_bytes": len(self.byte_token_ids),
            "max_token_bytes": self.max_token_bytes,
        }


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


def inspect_tokenizer(tokenizer: Any, *, embedding_rows: int | None = None) -> ByteVocabulary:
    vocab = tokenizer.get_vocab()
    inverse_vocab = {token_id: token for token, token_id in vocab.items()}
    total_rows = embedding_rows if embedding_rows is not None else max(inverse_vocab) + 1
    if inverse_vocab and max(inverse_vocab) >= total_rows:
        raise VocabularyError("tokenizer IDs exceed the embedding table")

    added_ids = set(tokenizer.get_added_vocab().values())
    added_ids.update(int(token_id) for token_id in tokenizer.all_special_ids)
    token_bytes: list[bytes | None] = [None] * total_rows
    ordinary_ids: list[int] = []
    by_bytes: dict[bytes, list[int]] = defaultdict(list)

    for token_id in sorted(inverse_vocab):
        if token_id in added_ids:
            continue
        token = _token_for_id(tokenizer, token_id, inverse_vocab)
        value = byte_level_token_to_bytes(token)
        if len(value) > 255:
            raise VocabularyError(f"token ID {token_id} is {len(value)} bytes; maximum is 255")
        token_bytes[token_id] = value
        ordinary_ids.append(token_id)
        by_bytes[value].append(token_id)

    collisions = {value: ids for value, ids in by_bytes.items() if len(ids) > 1}
    if collisions:
        sample = next(iter(collisions.items()))
        raise VocabularyError(f"ordinary tokens are not injective; example collision: {sample!r}")

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "decode"):
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
            raise VocabularyError(f"ByteLevel decoder round-trip mismatch: {mismatches!r}")

    byte_ids: list[int] = []
    for value in range(256):
        ids = by_bytes.get(bytes([value]), [])
        if len(ids) != 1:
            raise VocabularyError(
                f"byte {value} has {len(ids)} atomic token IDs; expected exactly 1"
            )
        byte_ids.append(ids[0])

    max_token_bytes = max((len(value) for value in by_bytes), default=1)
    control_ids = tuple(sorted(set(range(total_rows)) - set(ordinary_ids)))
    return ByteVocabulary(
        token_bytes=tuple(token_bytes),
        ordinary_ids=tuple(ordinary_ids),
        control_ids=control_ids,
        byte_token_ids=tuple(byte_ids),
        max_token_bytes=max_token_bytes,
    )
