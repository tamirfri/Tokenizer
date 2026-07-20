from __future__ import annotations

import random
from collections.abc import Iterable

from datasets import load_dataset

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"


def load_wikitext_documents(split: str) -> list[bytes]:
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    return [text.encode("utf-8") for text in dataset["text"] if text]


def sample_spans(
    documents: list[bytes],
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
