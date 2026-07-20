from __future__ import annotations

import hashlib
import heapq
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, Final, final

from datasets import load_dataset

DATASET_ID: Final = "Salesforce/wikitext"
DATASET_CONFIG: Final = "wikitext-103-raw-v1"
DATASET_REVISION: Final = "b08601e04326c79dfdd32d625aee71d232d685c3"
SYNTHETIC_DATASET_ID: Final = "continuous-tokenizer/synthetic-bytes"
SYNTHETIC_DATASET_REVISION: Final = "synthetic"

type TokenWindow = tuple[tuple[int, ...], tuple[int, ...]]


@final
@dataclass(frozen=True, slots=True)
class ContentWindow:
    document_sha256: str
    start: int
    payload: bytes
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_sha256": self.document_sha256,
            "start": self.start,
            "sha256": self.sha256,
            "payload_hex": self.payload.hex(),
        }


@final
@dataclass(frozen=True, slots=True)
class TokenWindowSampling:
    count: int
    prompt_tokens: int
    continuation_tokens: int
    seed: int


def synthetic_documents(split: str) -> list[bytes]:
    prefix = {"train": "train", "validation": "validation", "test": "test"}.get(split)
    if prefix is None:
        raise ValueError(f"unsupported synthetic split: {split}")
    return [
        f"{prefix}: continuous byte spans are deterministic.".encode(),
        f"{prefix}: שלום, 世界, delta.".encode(),
        f"{prefix}: def encode(data: bytes) -> bytes: return data".encode(),
    ]


def stream_corpus_documents(
    split: str,
    *,
    dataset_id: str = DATASET_ID,
    config: str = DATASET_CONFIG,
    revision: str = DATASET_REVISION,
    max_rows: int,
) -> Iterable[bytes]:
    if max_rows < 1:
        raise ValueError("corpus row bound must be positive")
    if dataset_id == SYNTHETIC_DATASET_ID:
        if revision != SYNTHETIC_DATASET_REVISION:
            raise ValueError(f"unsupported synthetic dataset revision: {revision}")
        yield from islice(synthetic_documents(split), max_rows)
        return
    dataset = load_dataset(
        dataset_id,
        config,
        split=split,
        revision=revision,
        streaming=True,
    )
    yielded = 0
    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str) or not text:
            continue
        yield text.encode("utf-8")
        yielded += 1
        if yielded == max_rows:
            return


def load_corpus_documents(
    split: str,
    *,
    dataset_id: str = DATASET_ID,
    config: str = DATASET_CONFIG,
    revision: str = DATASET_REVISION,
    max_rows: int = 4096,
) -> list[bytes]:
    return list(
        stream_corpus_documents(
            split,
            dataset_id=dataset_id,
            config=config,
            revision=revision,
            max_rows=max_rows,
        )
    )


def sample_spans(
    documents: Sequence[bytes],
    *,
    count: int,
    seed: int,
    minimum: int = 2,
    maximum: int = 64,
) -> list[bytes]:
    eligible = [document for document in documents if len(document) >= minimum]
    if not eligible:
        raise ValueError("the corpus contains no eligible documents")
    randomizer = random.Random(seed)
    spans: list[bytes] = []
    for _ in range(count):
        document = randomizer.choice(eligible)
        length = randomizer.randint(minimum, min(maximum, len(document)))
        start = randomizer.randint(0, len(document) - length)
        spans.append(document[start : start + length])
    return spans


def sample_token_windows(
    token_ids: Sequence[int],
    *,
    count: int,
    prompt_tokens: int,
    continuation_tokens: int,
) -> tuple[TokenWindow, ...]:
    window = prompt_tokens + continuation_tokens
    if len(token_ids) < window:
        raise ValueError("the corpus is shorter than one token window")
    available = len(token_ids) - window + 1
    selected = min(count, available)
    stride = max(1, available // selected)
    result: list[TokenWindow] = []
    for index in range(selected):
        start = min(index * stride, available - 1)
        result.append(
            (
                tuple(token_ids[start : start + prompt_tokens]),
                tuple(token_ids[start + prompt_tokens : start + window]),
            )
        )
    return tuple(result)


def _utf8_prefix(data: bytes, maximum: int) -> bytes:
    value = data[:maximum]
    while value:
        try:
            value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            value = value[: error.start]
        else:
            return value
    return b""


def _utf8_windows(document: bytes, window_bytes: int) -> Iterable[tuple[int, bytes]]:
    encoded = bytearray()
    start = 0
    for character in document.decode("utf-8", errors="strict"):
        value = character.encode("utf-8")
        if encoded and len(encoded) + len(value) > window_bytes:
            payload = bytes(encoded)
            yield start, payload
            start += len(payload)
            encoded.clear()
        encoded.extend(value)
    if encoded:
        yield start, bytes(encoded)


def sample_content_windows(
    documents: Sequence[bytes],
    *,
    maximum_bytes: int,
    window_bytes: int = 256,
    seed: int = 17,
) -> tuple[ContentWindow, ...]:
    if maximum_bytes < 1 or window_bytes < 1:
        raise ValueError("content window byte bounds must be positive")
    candidates: dict[tuple[str, int, bytes], tuple[bytes, ContentWindow]] = {}
    seed_bytes = seed.to_bytes(8, "big", signed=True)
    for document in documents:
        document_digest = hashlib.sha256(document).digest()
        document_sha256 = document_digest.hex()
        for start, payload in _utf8_windows(document, window_bytes):
            rank = hashlib.sha256(seed_bytes + document_digest + start.to_bytes(8, "big") + payload).digest()
            window = ContentWindow(
                document_sha256=document_sha256,
                start=start,
                payload=payload,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            candidates[(document_sha256, start, payload)] = (rank, window)
    if not candidates:
        raise ValueError("the corpus contains no non-empty document windows")

    selected = []
    remaining = maximum_bytes
    for _, window in sorted(candidates.values(), key=lambda item: (item[0], item[1].sha256)):
        payload = _utf8_prefix(window.payload, remaining)
        if not payload:
            continue
        selected.append(
            ContentWindow(
                document_sha256=window.document_sha256,
                start=window.start,
                payload=payload,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        remaining -= len(payload)
        if remaining == 0:
            break
    return tuple(selected)


def sample_document_token_windows(
    documents: Sequence[bytes],
    encode: Callable[[str], Sequence[int]],
    sampling: TokenWindowSampling,
) -> tuple[TokenWindow, ...]:
    count = sampling.count
    prompt_tokens = sampling.prompt_tokens
    continuation_tokens = sampling.continuation_tokens
    window_size = prompt_tokens + continuation_tokens
    if count < 1 or min(prompt_tokens, continuation_tokens) < 1:
        raise ValueError("token window counts and lengths must be positive")
    candidates: list[tuple[bytes, TokenWindow]] = []
    for document in sorted(set(documents)):
        token_ids = tuple(encode(document.decode("utf-8", errors="strict")))
        for start in range(len(token_ids) - window_size + 1):
            payload = document + start.to_bytes(8, "big")
            rank = hashlib.sha256(
                sampling.seed.to_bytes(8, "big", signed=True) + payload,
            ).digest()
            candidates.append(
                (
                    rank,
                    (
                        token_ids[start : start + prompt_tokens],
                        token_ids[start + prompt_tokens : start + window_size],
                    ),
                )
            )
    if len(candidates) < count:
        raise ValueError("the corpus has fewer eligible token windows than requested")
    return tuple(window for _, window in heapq.nsmallest(count, candidates, key=lambda item: item[0]))


def joined_prefix(documents: Iterable[bytes], *, max_bytes: int) -> bytes:
    result = bytearray()
    for document in documents:
        if result:
            result.extend(b"\n")
        remaining = max_bytes - len(result)
        if remaining <= 0:
            break
        result.extend(document[:remaining])
    value = bytes(result[:max_bytes])
    try:
        value.decode("utf-8")
    except UnicodeDecodeError as error:
        value = value[: error.start]
    return value
