from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import final

import torch
from torch import Tensor

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.codec.batches import build_byte_batch, span_bucket_width
from continuous_tokenizer.codec.compilation import DYNAMIC_SEGMENTATION_MAX_BYTES
from continuous_tokenizer.runtime.tensors import tensor_bytes


@final
@dataclass(frozen=True, slots=True)
class VocabularyBucket:
    token_ids: Tensor
    byte_values: Tensor
    valid_mask: Tensor
    source_targets: Tensor


@final
@dataclass(frozen=True, slots=True)
class VocabularyBatch:
    bucket: int
    rows: Tensor
    logical_rows: int


def build_vocabulary_groups(
    assets: ModelAssets,
    token_ids: Sequence[int] | None = None,
) -> tuple[VocabularyBucket, ...]:
    vocabulary = assets.vocabulary
    max_span = max(
        vocabulary.max_token_bytes,
        DYNAMIC_SEGMENTATION_MAX_BYTES,
    )
    ids = vocabulary.compatibility_ids if token_ids is None else token_ids
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("vocabulary training IDs must be non-empty and unique")
    if not set(ids).issubset(vocabulary.compatibility_ids):
        raise ValueError("vocabulary training IDs must contain only compatibility rows")

    by_width: dict[int, list[int]] = {}
    for token_id in ids:
        length = len(vocabulary.bytes_for(token_id))
        if length == 1:
            continue
        width = span_bucket_width(length, max_span=max_span)
        by_width.setdefault(width, []).append(token_id)

    buckets = []
    for bucket_ids in by_width.values():
        token_ids_tensor = torch.tensor(bucket_ids, dtype=torch.long)
        batch = build_byte_batch(
            [vocabulary.bytes_for(token_id) for token_id in bucket_ids],
            max_span=max_span,
            device=torch.device("cpu"),
        )
        buckets.append(
            VocabularyBucket(
                token_ids=token_ids_tensor,
                byte_values=batch.byte_values,
                valid_mask=batch.valid_mask,
                source_targets=assets.input_embeddings.index_select(0, token_ids_tensor),
            )
        )
    return tuple(buckets)


def build_vocabulary_batches(
    groups: tuple[VocabularyBucket, ...],
    batch_size: int,
    generator: torch.Generator,
) -> list[VocabularyBatch]:
    if batch_size < 1:
        raise ValueError("vocabulary batch size must be positive")
    batches: list[VocabularyBatch] = []
    for bucket, values in enumerate(groups):
        rows = torch.randperm(len(values.token_ids), generator=generator)
        for selected in rows.split(batch_size):
            logical_rows = len(selected)
            padding = batch_size - logical_rows
            padded = torch.cat((selected, selected[-1:].expand(padding))) if padding else selected
            batches.append(VocabularyBatch(bucket, padded, logical_rows))
    order = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[index] for index in order]


def stage_vocabulary_groups(
    groups: tuple[VocabularyBucket, ...],
    device: torch.device,
) -> tuple[VocabularyBucket, ...]:
    return tuple(
        replace(
            group,
            byte_values=group.byte_values.to(device),
            valid_mask=group.valid_mask.to(device),
            source_targets=group.source_targets.to(device),
        )
        for group in groups
    )


def vocabulary_bucket_offsets(
    groups: tuple[VocabularyBucket, ...],
) -> tuple[int, ...]:
    offsets = []
    total = 0
    for group in groups:
        offsets.append(total)
        total += len(group.token_ids)
    return tuple(offsets)


def vocabulary_bucket_tensor_bytes(groups: tuple[VocabularyBucket, ...]) -> int:
    return sum(
        tensor_bytes(value)
        for group in groups
        for value in (
            group.token_ids,
            group.byte_values,
            group.valid_mask,
            group.source_targets,
        )
    )
