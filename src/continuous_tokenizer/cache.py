from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CacheKey:
    namespace: str
    dtype: str
    span: bytes


@dataclass(frozen=True, slots=True)
class CacheInfo:
    entries: int
    bytes: int
    max_bytes: int
    hits: int
    misses: int


class EncodingCache:
    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self._items: OrderedDict[CacheKey, Tensor] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey, *, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        value = self._items.get(key)
        if value is None:
            self._misses += 1
            return None
        self._items.move_to_end(key)
        self._hits += 1
        return value.to(device=device, dtype=dtype).clone()

    def put(self, key: CacheKey, value: Tensor) -> None:
        stored = value.detach().to(device="cpu").clone()
        size = stored.numel() * stored.element_size()
        if size > self.max_bytes or self.max_bytes == 0:
            return
        previous = self._items.pop(key, None)
        if previous is not None:
            self._bytes -= previous.numel() * previous.element_size()
        self._items[key] = stored
        self._bytes += size
        while self._bytes > self.max_bytes:
            _, removed = self._items.popitem(last=False)
            self._bytes -= removed.numel() * removed.element_size()

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def info(self) -> CacheInfo:
        return CacheInfo(
            entries=len(self._items),
            bytes=self._bytes,
            max_bytes=self.max_bytes,
            hits=self._hits,
            misses=self._misses,
        )
