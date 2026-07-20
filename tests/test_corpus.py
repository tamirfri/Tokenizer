from __future__ import annotations

from continuous_tokenizer.corpus import joined_prefix


def test_joined_prefix_never_splits_utf8_codepoint() -> None:
    value = joined_prefix(["a€b".encode()], max_bytes=3)

    assert value == b"a"
    assert value.decode("utf-8") == "a"
