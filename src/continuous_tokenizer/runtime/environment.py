from __future__ import annotations

import platform
import resource
import sys
from importlib.metadata import version
from typing import Any

import psutil
import torch


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def dependency_environment(device: torch.device) -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": version("torch"),
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "huggingface_hub": version("huggingface-hub"),
        "device": str(device),
    }


def mps_memory_bytes(device: torch.device) -> tuple[int, int]:
    if device.type != "mps":
        return 0, 0
    return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()


def runtime_environment(device: torch.device) -> dict[str, Any]:
    mps_allocated_bytes, mps_driver_allocated_bytes = mps_memory_bytes(device)
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "device": str(device),
        "rss_bytes": psutil.Process().memory_info().rss,
        "peak_rss_bytes": peak_rss_bytes(),
        "mps_allocated_bytes": mps_allocated_bytes,
        "mps_driver_allocated_bytes": mps_driver_allocated_bytes,
    }
