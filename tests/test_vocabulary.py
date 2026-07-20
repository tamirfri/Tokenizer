from __future__ import annotations

from typing import Any

import pytest

from continuous_tokenizer.vocabulary import VocabularyError, bytes_to_unicode, inspect_tokenizer


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int], added: dict[str, int] | None = None) -> None:
        self._vocab = vocab
        self._added = {} if added is None else added
        self.all_special_ids = list(self._added.values())
        self.backend_tokenizer: Any = None

    def get_vocab(self) -> dict[str, int]:
        return {**self._vocab, **self._added}

    def get_added_vocab(self) -> dict[str, int]:
        return self._added.copy()


class DuplicateBackendModel:
    def id_to_token(self, token_id: int) -> str | None:
        if token_id in (0, 256):
            return "!"
        return None


class DuplicateBackend:
    model = DuplicateBackendModel()


class MismatchingBackend:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del token_ids, skip_special_tokens
        return "wrong"


def atomic_vocab() -> dict[str, int]:
    return {character: value for value, character in bytes_to_unicode().items()}


def test_inspect_finds_every_atomic_byte_and_controls() -> None:
    tokenizer = FakeTokenizer(atomic_vocab(), {"<control>": 256})

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=257)

    assert vocabulary.byte_token_ids == tuple(range(256))
    assert vocabulary.control_ids == (256,)
    assert vocabulary.ordinary_ids == tuple(range(256))
    assert vocabulary.bytes_for(128) == b"\x80"


def test_inspect_rejects_duplicate_byte_sequences() -> None:
    vocab = atomic_vocab()
    vocab["duplicate"] = 256
    tokenizer = FakeTokenizer(vocab)
    tokenizer.backend_tokenizer = DuplicateBackend()

    with pytest.raises(VocabularyError, match="not injective"):
        inspect_tokenizer(tokenizer, embedding_rows=257)


def test_inspect_rejects_missing_atomic_byte() -> None:
    vocab = atomic_vocab()
    del vocab[bytes_to_unicode()[255]]

    with pytest.raises(VocabularyError, match="byte 255"):
        inspect_tokenizer(FakeTokenizer(vocab), embedding_rows=256)


def test_inspect_validates_backend_decoder_round_trip() -> None:
    tokenizer = FakeTokenizer(atomic_vocab())
    tokenizer.backend_tokenizer = MismatchingBackend()

    with pytest.raises(VocabularyError, match="decoder round-trip"):
        inspect_tokenizer(tokenizer, embedding_rows=256)
