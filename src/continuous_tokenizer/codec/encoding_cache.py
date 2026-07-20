"""Bounded encoding-only cache for deterministic codec evaluation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, final

import torch
from torch import Tensor

from continuous_tokenizer.runtime.tensors import tensor_bytes


@final
@dataclass(frozen=True, slots=True)
class CacheKey:
    namespace: str
    dtype: str
    span: bytes
    rows: int
    width: int


@final
@dataclass(frozen=True, slots=True)
class CacheStats:
    entries: int
    tensor_bytes: int
    capacity_bytes: int
    hits: int
    misses: int
    stores: int
    evictions: int
    coalesced: int

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def since(self, previous: CacheStats) -> CacheStats:
        if self.capacity_bytes != previous.capacity_bytes:
            raise ValueError("cache capacity changed between snapshots")
        hits = self.hits - previous.hits
        misses = self.misses - previous.misses
        stores = self.stores - previous.stores
        evictions = self.evictions - previous.evictions
        coalesced = self.coalesced - previous.coalesced
        if min(hits, misses, stores, evictions, coalesced) < 0:
            raise ValueError("cache counters were reset between snapshots")
        return CacheStats(
            entries=self.entries,
            tensor_bytes=self.tensor_bytes,
            capacity_bytes=self.capacity_bytes,
            hits=hits,
            misses=misses,
            stores=stores,
            evictions=evictions,
            coalesced=coalesced,
        )


@final
class EncodingCache:
    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes: Final = max_bytes
        self._items: OrderedDict[CacheKey, Tensor] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self._coalesced = 0

    def _get_cpu(self, key: CacheKey) -> Tensor | None:
        value = self._items.get(key)
        if value is None:
            self._misses += 1
            return None
        self._items.move_to_end(key)
        self._hits += 1
        return value

    def get(self, key: CacheKey, *, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        value = self._get_cpu(key)
        return None if value is None else value.to(device=device, dtype=dtype, copy=True)

    def get_many_cpu(self, keys: Sequence[CacheKey]) -> tuple[Tensor | None, ...]:
        values: dict[CacheKey, Tensor | None] = {}
        result: list[Tensor | None] = []
        for key in keys:
            if key in values:
                self._coalesced += 1
            else:
                values[key] = self._get_cpu(key)
            result.append(values[key])
        return tuple(result)

    def put(self, key: CacheKey, value: Tensor) -> None:
        self._put_cpu(key, value.detach().to(device="cpu", copy=True))

    def put_many(self, keys: Sequence[CacheKey], values: Tensor) -> None:
        if len(keys) != values.shape[0]:
            raise ValueError("cache keys and values must have equal lengths")
        if not keys or self.max_bytes == 0:
            return
        stored = values.detach().to(device="cpu", copy=True)
        for key, value in zip(keys, stored, strict=True):
            self._put_cpu(key, value.clone())

    def _put_cpu(self, key: CacheKey, stored: Tensor) -> None:
        size = tensor_bytes(stored)
        if size > self.max_bytes or self.max_bytes == 0:
            return
        previous = self._items.pop(key, None)
        if previous is not None:
            self._bytes -= tensor_bytes(previous)
        self._items[key] = stored
        self._bytes += size
        self._stores += 1
        while self._bytes > self.max_bytes:
            _, removed = self._items.popitem(last=False)
            self._bytes -= tensor_bytes(removed)
            self._evictions += 1

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self._coalesced = 0

    def info(self) -> CacheStats:
        return CacheStats(
            entries=len(self._items),
            tensor_bytes=self._bytes,
            capacity_bytes=self.max_bytes,
            hits=self._hits,
            misses=self._misses,
            stores=self._stores,
            evictions=self._evictions,
            coalesced=self._coalesced,
        )
