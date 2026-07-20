from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from continuous_tokenizer.codec import CodecConfig, ContinuousByteCodec

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: Path,
    codec: ContinuousByteCodec,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": codec.config.to_dict(),
        "metadata": metadata,
        "state_dict": codec.state_dict(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ContinuousByteCodec, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format in {path}")
    state_dict = payload["state_dict"]
    forbidden = [key for key in state_dict if "original" in key or "vocabulary_table" in key]
    if forbidden:
        raise ValueError(f"checkpoint contains forbidden source-table keys: {forbidden}")
    config = CodecConfig(**payload["config"])
    codec = ContinuousByteCodec(
        config,
        state_dict["byte_embeddings"],
        state_dict["control_ids"],
        state_dict["control_embeddings"],
    )
    codec.load_state_dict(state_dict)
    codec.to(device)
    codec.eval()
    return codec, dict(payload["metadata"])


def checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
