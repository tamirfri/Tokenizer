from __future__ import annotations

import os
import sys
from pathlib import Path


def compiler_cache_directory() -> Path:
    configured = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(sys.prefix) / ".cache" / "torchinductor"


def configure_compiler_cache(backend: str) -> None:
    if backend == "inductor":
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            str(compiler_cache_directory()),
        )
