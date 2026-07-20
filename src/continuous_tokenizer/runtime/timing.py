from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Final, final

import psutil
import torch

from continuous_tokenizer.runtime.device import synchronize_device
from continuous_tokenizer.runtime.environment import mps_memory_bytes, peak_rss_bytes

TIMING_OBSERVATION_SCHEMA_VERSION: Final = 1


@final
@dataclass(frozen=True, slots=True)
class TimingObservation:
    schema_version: int
    wall_seconds: float
    synchronization_count: int
    host_to_device_bytes: int
    device_to_host_bytes: int
    rss_before_bytes: int
    rss_after_bytes: int
    peak_rss_bytes: int
    mps_allocated_before_bytes: int
    mps_allocated_after_bytes: int
    peak_mps_allocated_bytes: int
    mps_driver_before_bytes: int
    mps_driver_after_bytes: int
    peak_mps_driver_bytes: int
    mps_peak_method: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _MemorySample:
    rss_bytes: int
    peak_rss_bytes: int
    mps_allocated_bytes: int
    mps_driver_bytes: int


def _memory_sample(device: torch.device, process: psutil.Process) -> _MemorySample:
    rss_bytes = process.memory_info().rss
    rss_peak_bytes = peak_rss_bytes()
    mps_allocated_bytes, mps_driver_bytes = mps_memory_bytes(device)
    return _MemorySample(
        rss_bytes=rss_bytes,
        peak_rss_bytes=rss_peak_bytes,
        mps_allocated_bytes=mps_allocated_bytes,
        mps_driver_bytes=mps_driver_bytes,
    )


def timed_observation[Result](
    callable_: Callable[[], Result],
    device: torch.device,
    *,
    host_to_device_bytes: int = 0,
    device_to_host_bytes: int = 0,
) -> tuple[Result, TimingObservation]:
    if host_to_device_bytes < 0 or device_to_host_bytes < 0:
        raise ValueError("transfer byte counts must be non-negative")
    process = psutil.Process()
    before = _memory_sample(device, process)
    synchronize_device(device)
    started = time.perf_counter()
    result = callable_()
    synchronize_device(device)
    wall_seconds = time.perf_counter() - started
    after = _memory_sample(device, process)
    synchronized = device.type in {"cuda", "mps"}
    return result, TimingObservation(
        schema_version=TIMING_OBSERVATION_SCHEMA_VERSION,
        wall_seconds=wall_seconds,
        synchronization_count=2 if synchronized else 0,
        host_to_device_bytes=host_to_device_bytes,
        device_to_host_bytes=device_to_host_bytes,
        rss_before_bytes=before.rss_bytes,
        rss_after_bytes=after.rss_bytes,
        peak_rss_bytes=max(before.peak_rss_bytes, after.peak_rss_bytes),
        mps_allocated_before_bytes=before.mps_allocated_bytes,
        mps_allocated_after_bytes=after.mps_allocated_bytes,
        peak_mps_allocated_bytes=max(
            before.mps_allocated_bytes,
            after.mps_allocated_bytes,
        ),
        mps_driver_before_bytes=before.mps_driver_bytes,
        mps_driver_after_bytes=after.mps_driver_bytes,
        peak_mps_driver_bytes=max(
            before.mps_driver_bytes,
            after.mps_driver_bytes,
        ),
        mps_peak_method="boundary_samples" if device.type == "mps" else "not_applicable",
    )


def timed_call[Result](
    callable_: Callable[[], Result],
    device: torch.device,
) -> tuple[Result, float]:
    result, observation = timed_observation(callable_, device)
    return result, observation.wall_seconds


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    median = statistics.median(values)
    p95 = values[0] if len(values) == 1 else statistics.quantiles(values, n=100, method="inclusive")[94]
    return {"median": median, "p95": p95}
