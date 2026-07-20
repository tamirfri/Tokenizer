from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from continuous_tokenizer.contracts.parsing import (
    is_lowercase_sha256,
    mapping_fingerprint,
)

PROSPECTIVE_INPUT_SUBSET_FILENAME: Final = "prospective-vocabulary-subset.json"
PROSPECTIVE_INPUT_SUBSET_KIND: Final = "prospective_input_vocabulary_subset"
PROSPECTIVE_INPUT_SUBSET_ALGORITHM: Final = "content_hashed_length_stratified_non_atomic_compatibility_rows"
_FIELDS: Final = {
    "schema_version",
    "artifact_kind",
    "requested_rows",
    "selected_rows",
    "subset_seed",
    "algorithm",
    "subset_sha256",
    "batch_size",
    "maximum_span",
    "width_buckets",
    "vocabulary_batches_per_epoch",
    "rows",
}


def _integer_at_least(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _signed_64_bit_integer(value: object) -> bool:
    return _integer_at_least(value, -(1 << 63)) and cast(int, value) < 1 << 63


def _subset_rows(
    rows: Sequence[object],
    maximum_span: int,
) -> tuple[list[dict[str, object]], dict[int, int], list[int], list[str]]:
    canonical: list[dict[str, object]] = []
    bucket_counts: dict[int, int] = {}
    token_ids: list[int] = []
    errors: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"token_id", "bytes"}:
            errors.append("prospective vocabulary subset row is not canonical")
            continue
        token_id = row.get("token_id")
        encoded = row.get("bytes")
        try:
            payload = bytes.fromhex(encoded) if isinstance(encoded, str) else b""
        except ValueError:
            payload = b""
        if not _integer_at_least(token_id, 0) or not isinstance(encoded, str) or not 1 < len(payload) <= maximum_span:
            errors.append("prospective vocabulary subset row is invalid")
            continue
        parsed_token_id = cast(int, token_id)
        token_ids.append(parsed_token_id)
        canonical.append({"token_id": parsed_token_id, "bytes": encoded})
        width = min(1 << (len(payload) - 1).bit_length(), maximum_span)
        bucket_counts[width] = bucket_counts.get(width, 0) + 1
    return canonical, bucket_counts, token_ids, errors


def prospective_vocabulary_subset_errors(  # noqa: C901 - Canonical artifact checks stay together.
    value: object,
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["prospective vocabulary subset artifact must be an object"]
    values = cast(Mapping[str, Any], value)
    if set(values) != _FIELDS:
        return ["prospective vocabulary subset artifact fields are not canonical"]
    errors: list[str] = []
    requested_rows = values["requested_rows"]
    selected_rows = values["selected_rows"]
    batch_size = values["batch_size"]
    maximum_span = values["maximum_span"]
    batch_count = values["vocabulary_batches_per_epoch"]
    if values["schema_version"] != 1 or values["artifact_kind"] != PROSPECTIVE_INPUT_SUBSET_KIND:
        errors.append("prospective vocabulary subset artifact schema is unsupported")
    if not _integer_at_least(requested_rows, 1) or selected_rows != requested_rows:
        errors.append("prospective vocabulary subset row count is invalid")
    if (
        not _signed_64_bit_integer(values["subset_seed"])
        or values["algorithm"] != PROSPECTIVE_INPUT_SUBSET_ALGORITHM
        or not is_lowercase_sha256(values["subset_sha256"])
    ):
        errors.append("prospective vocabulary subset identity is invalid")
    batch_contract_valid = _integer_at_least(batch_size, 1) and _integer_at_least(maximum_span, 32) and _integer_at_least(batch_count, 1)
    if not batch_contract_valid:
        errors.append("prospective vocabulary subset batch contract is invalid")
    rows = values["rows"]
    if not isinstance(rows, list) or len(rows) != selected_rows:
        errors.append("prospective vocabulary subset rows are incomplete")
        return errors
    canonical, bucket_counts, token_ids, row_errors = _subset_rows(
        rows,
        cast(int, maximum_span) if _integer_at_least(maximum_span, 32) else 32,
    )
    errors.extend(row_errors)
    if token_ids != sorted(set(token_ids)):
        errors.append("prospective vocabulary subset token IDs are not sorted and unique")
    if mapping_fingerprint(canonical) != values["subset_sha256"]:
        errors.append("prospective vocabulary subset content hash mismatch")
    safe_batch_size = cast(int, batch_size) if _integer_at_least(batch_size, 1) else 1
    expected_buckets = [
        {
            "width": width,
            "rows": count,
            "batches": (count + safe_batch_size - 1) // safe_batch_size,
        }
        for width, count in sorted(bucket_counts.items())
    ]
    if values["width_buckets"] != expected_buckets:
        errors.append("prospective vocabulary subset width buckets are invalid")
    if batch_count != sum(bucket["batches"] for bucket in expected_buckets):
        errors.append("prospective vocabulary subset batch count is invalid")
    return errors
