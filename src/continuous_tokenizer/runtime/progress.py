from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

LOGGER = logging.getLogger("continuous_tokenizer.progress")
DEFAULT_UPDATES = 20


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    logging.getLogger("continuous_tokenizer").setLevel(logging.INFO)


def log_event(event: str, **fields: Any) -> None:
    values = {"event": event, **fields}
    message = " ".join(f"{name}={json.dumps(value, separators=(',', ':'), sort_keys=True)}" for name, value in values.items())
    LOGGER.info("%s", message)


@dataclass(slots=True)
class ProgressTracker:
    phase: str
    total: int
    context: dict[str, Any] = field(default_factory=dict)
    updates: int = DEFAULT_UPDATES
    _started: float = field(default_factory=perf_counter, init=False)

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError("progress total must be non-negative")
        if self.updates < 1:
            raise ValueError("progress update count must be positive")

    def update(self, completed: int, **fields: Any) -> None:
        if not 1 <= completed <= self.total:
            raise ValueError("completed progress must be within the declared total")
        interval = max(1, math.ceil(self.total / self.updates))
        if completed not in {1, self.total} and completed % interval != 0:
            return
        elapsed = perf_counter() - self._started
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = self.total - completed
        eta = remaining / rate if rate > 0 else None
        log_event(
            "progress",
            phase=self.phase,
            completed=completed,
            total=self.total,
            percent=round(100.0 * completed / self.total, 1),
            elapsed_seconds=round(elapsed, 1),
            rate_per_second=rate,
            eta_seconds=None if eta is None else round(eta, 1),
            projected_total_seconds=(None if eta is None else round(elapsed + eta, 1)),
            **self.context,
            **fields,
        )
