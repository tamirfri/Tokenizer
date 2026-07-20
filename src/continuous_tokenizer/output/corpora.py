from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True, slots=True)
class SelectedOutputDocuments:
    documents: tuple[bytes, ...]
    sha256: str
    document_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.documents or len(self.documents) != len(self.document_sha256):
            raise ValueError("selected output documents must be non-empty and fully hashed")


def select_output_documents(
    documents: Sequence[bytes],
    *,
    count: int,
    seed: int,
    excluded_sha256: frozenset[str] = frozenset(),
) -> SelectedOutputDocuments:
    if count < 1 or seed < 0:
        raise ValueError("output document count must be positive and seed non-negative")
    candidates = []
    for index, document in enumerate(documents):
        document_digest = hashlib.sha256(document).digest()
        document_sha256 = document_digest.hex()
        if document_sha256 in excluded_sha256:
            continue
        rank = hashlib.sha256(
            seed.to_bytes(8, "big") + index.to_bytes(8, "big") + document_digest,
        ).digest()
        candidates.append((rank, document_digest, document))
    selected = sorted(candidates)[:count]
    if len(selected) != count:
        raise ValueError(f"output corpus requires {count} distinct eligible documents")
    digest = hashlib.sha256()
    for _, document_digest, document in selected:
        digest.update(document_digest)
        digest.update(len(document).to_bytes(8, "big"))
        digest.update(document)
    return SelectedOutputDocuments(
        documents=tuple(document for _, _, document in selected),
        sha256=digest.hexdigest(),
        document_sha256=tuple(document_digest.hex() for _, document_digest, _ in selected),
    )
