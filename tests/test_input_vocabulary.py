from __future__ import annotations

import os
import unittest
from typing import Any

from continuous_tokenizer.backbone.assets import resolve_tokenizer_assets
from continuous_tokenizer.backbone.config import text_config
from continuous_tokenizer.backbone.vocabulary import (
    VocabularyError,
    bytes_to_unicode,
    inspect_tokenizer,
)


class FakeTokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        added: dict[str, int] | None = None,
        *,
        special_ids: list[int] | None = None,
    ) -> None:
        self._vocab = vocab
        self._added = {} if added is None else added
        self.all_special_ids = list(self._added.values()) if special_ids is None else special_ids
        self.backend_tokenizer: Any = None

    def get_vocab(self) -> dict[str, int]:
        return {**self._vocab, **self._added}

    def get_added_vocab(self) -> dict[str, int]:
        return self._added.copy()


class MismatchingBackend:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del token_ids, skip_special_tokens
        return "wrong"


class SentencePieceModel:
    def id_to_token(self, token_id: int) -> str | None:
        return "\u2581hello" if token_id == 256 else None


class SentencePieceBackend:
    model = SentencePieceModel()

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        token_id = token_ids[0]
        return " hello" if token_id == 256 else bytes([token_id]).decode("utf-8", errors="ignore")


class LiteralUnicodeBackend:
    decoder = object()

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        token_id = token_ids[0]
        return "Ā" if token_id == 256 else bytes([token_id]).decode("utf-8", errors="ignore")


class DuplicateByteBackend:
    decoder = object()

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        token_id = token_ids[0]
        return "\x01" if token_id == 256 else bytes([token_id]).decode("utf-8", errors="ignore")


class DuplicateSpanBackend:
    decoder = object()

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        token_id = token_ids[0]
        return "ab" if token_id in {256, 257} else bytes([token_id]).decode("utf-8", errors="ignore")


class AddedNewlineBackend:
    decoder = object()

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        token_id = token_ids[0]
        return "\n\n" if token_id == 256 else bytes([token_id]).decode("utf-8", errors="ignore")


def atomic_vocab() -> dict[str, int]:
    return {character: value for value, character in bytes_to_unicode().items()}


def test_inspect_finds_every_atomic_byte_and_controls() -> None:
    tokenizer = FakeTokenizer(atomic_vocab(), {"<control>": 256})

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=257)

    assert vocabulary.byte_token_ids == tuple(range(256))
    assert vocabulary.control_ids == (256,)
    assert vocabulary.ordinary_ids == tuple(range(256))
    assert vocabulary.compatibility_ids == tuple(range(256))
    assert vocabulary.unavailable_ids == ()
    assert vocabulary.bytes_for(128) == b"\x80"


def test_inspect_excludes_structural_tokens_without_embedding_rows() -> None:
    tokenizer = FakeTokenizer(atomic_vocab(), {"<unembedded_control>": 256})

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=256)

    assert len(vocabulary.token_bytes) == 256
    assert vocabulary.control_ids == ()
    assert vocabulary.out_of_table_control_ids == (256,)
    assert vocabulary.ordinary_ids == tuple(range(256))


def test_inspect_rejects_ordinary_tokens_without_embedding_rows() -> None:
    vocab = atomic_vocab()
    vocab["ordinary"] = 256

    with unittest.TestCase().assertRaisesRegex(VocabularyError, "tokenizer IDs exceed"):
        inspect_tokenizer(FakeTokenizer(vocab), embedding_rows=256)


def test_inspect_preserves_duplicate_rows_and_prefers_byte_fallback_atom() -> None:
    vocab = {f"<0x{value:02X}>": value for value in range(256)}
    vocab["\x01"] = 256
    tokenizer = FakeTokenizer(vocab)
    tokenizer.backend_tokenizer = DuplicateByteBackend()

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=257)

    assert vocabulary.ordinary_ids == tuple(range(257))
    assert vocabulary.compatibility_ids == tuple(range(256))
    assert vocabulary.alias_ids == (256,)
    assert vocabulary.byte_token_ids == tuple(range(256))
    assert vocabulary.bytes_for(1) == vocabulary.bytes_for(256) == b"\x01"
    assert vocabulary.to_summary()["ambiguous_byte_sequences"] == 1


def test_inspect_uses_lowest_id_for_non_atomic_aliases() -> None:
    vocab = {f"<0x{value:02X}>": value for value in range(256)}
    vocab.update({"first": 256, "second": 257})
    tokenizer = FakeTokenizer(vocab)
    tokenizer.backend_tokenizer = DuplicateSpanBackend()

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=258)

    assert 256 in vocabulary.compatibility_ids
    assert 257 not in vocabulary.compatibility_ids
    assert vocabulary.alias_ids == (257,)
    assert vocabulary.bytes_for(256) == vocabulary.bytes_for(257) == b"ab"


def test_inspect_distinguishes_gemma_added_ordinary_unavailable_and_external_rows() -> None:
    tokenizer = FakeTokenizer(
        {f"<0x{value:02X}>": value for value in range(256)},
        {
            "\n\n": 256,
            "<eos>": 257,
            "<image_soft_token>": 259,
        },
        special_ids=[257, 259],
    )
    tokenizer.backend_tokenizer = AddedNewlineBackend()

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=259)

    assert vocabulary.control_ids == (257,)
    assert vocabulary.bytes_for(256) == b"\n\n"
    assert 256 in vocabulary.compatibility_ids
    assert vocabulary.unavailable_ids == (258,)
    assert vocabulary.out_of_table_control_ids == (259,)
    assert vocabulary.to_summary()["unavailable_rows"] == 1
    assert vocabulary.to_summary()["out_of_table_controls"] == 1
    with unittest.TestCase().assertRaisesRegex(VocabularyError, "unavailable embedding row"):
        vocabulary.payload_for(258)


def test_inspect_records_qwen_unavailable_embedding_rows() -> None:
    tokenizer = FakeTokenizer(atomic_vocab(), {"<control>": 256})

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=260)

    assert vocabulary.control_ids == (256,)
    assert vocabulary.unavailable_ids == (257, 258, 259)


def test_inspect_rejects_out_of_table_added_non_special_rows() -> None:
    tokenizer = FakeTokenizer(
        atomic_vocab(),
        {"<reserved>": 256},
        special_ids=[],
    )

    with unittest.TestCase().assertRaisesRegex(VocabularyError, "tokenizer IDs exceed"):
        inspect_tokenizer(tokenizer, embedding_rows=256)


def test_inspect_rejects_missing_atomic_byte() -> None:
    vocab = atomic_vocab()
    del vocab[bytes_to_unicode()[255]]

    with unittest.TestCase().assertRaisesRegex(VocabularyError, "byte 255"):
        inspect_tokenizer(FakeTokenizer(vocab), embedding_rows=256)


def test_inspect_rejects_empty_ordinary_token() -> None:
    vocab = atomic_vocab()
    vocab[""] = 256

    with unittest.TestCase().assertRaisesRegex(VocabularyError, "empty byte sequence"):
        inspect_tokenizer(FakeTokenizer(vocab), embedding_rows=257)


def test_inspect_validates_backend_decoder_round_trip() -> None:
    tokenizer = FakeTokenizer(atomic_vocab())
    tokenizer.backend_tokenizer = MismatchingBackend()

    with unittest.TestCase().assertRaisesRegex(VocabularyError, "decoder round-trip"):
        inspect_tokenizer(tokenizer, embedding_rows=256)


def test_inspect_uses_decoder_for_sentencepiece_tokens() -> None:
    vocab = atomic_vocab()
    vocab["\u2581hello"] = 256
    tokenizer = FakeTokenizer(vocab)
    tokenizer.backend_tokenizer = SentencePieceBackend()

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=257)

    assert vocabulary.bytes_for(256) == b" hello"


def test_inspect_uses_literal_unicode_for_non_bytelevel_tokenizers() -> None:
    vocab = {f"<0x{value:02X}>": value for value in range(256)}
    vocab["Ā"] = 256
    tokenizer = FakeTokenizer(vocab)
    tokenizer.backend_tokenizer = LiteralUnicodeBackend()

    vocabulary = inspect_tokenizer(tokenizer, embedding_rows=257)

    assert vocabulary.byte_token_ids == tuple(range(256))
    assert vocabulary.bytes_for(256) == "Ā".encode()


def test_real_tokenizers_have_all_atomic_bytes() -> None:
    if os.getenv("RUN_MODEL_TESTS") != "1":
        raise unittest.SkipTest("set RUN_MODEL_TESTS=1")

    models = (
        ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
        ("google/gemma-3-270m-it", "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"),
    )
    for model_id, revision in models:
        with unittest.TestCase().subTest(model_id=model_id):
            assets = resolve_tokenizer_assets(model_id, revision)
            vocabulary = inspect_tokenizer(
                assets.tokenizer,
                embedding_rows=text_config(assets.config)["vocab_size"],
            )

            assert assets.revision == revision
            assert len(vocabulary.byte_token_ids) == 256


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_inspect_finds_every_atomic_byte_and_controls,
            test_inspect_excludes_structural_tokens_without_embedding_rows,
            test_inspect_rejects_ordinary_tokens_without_embedding_rows,
            test_inspect_preserves_duplicate_rows_and_prefers_byte_fallback_atom,
            test_inspect_uses_lowest_id_for_non_atomic_aliases,
            test_inspect_distinguishes_gemma_added_ordinary_unavailable_and_external_rows,
            test_inspect_records_qwen_unavailable_embedding_rows,
            test_inspect_rejects_out_of_table_added_non_special_rows,
            test_inspect_rejects_missing_atomic_byte,
            test_inspect_rejects_empty_ordinary_token,
            test_inspect_validates_backend_decoder_round_trip,
            test_inspect_uses_decoder_for_sentencepiece_tokens,
            test_inspect_uses_literal_unicode_for_non_bytelevel_tokenizers,
            test_real_tokenizers_have_all_atomic_bytes,
        )
    )
