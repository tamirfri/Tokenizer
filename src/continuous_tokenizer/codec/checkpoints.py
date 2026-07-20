"""Tokenizer checkpoint persistence and deployment inventory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, final

import torch
from torch import Tensor

from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.codec.output import OutputByteCodec, OutputByteCodecConfig
from continuous_tokenizer.runtime.tensors import tensor_bytes

type CodecDirection = Literal["input_only", "output_only"]


@final
class FrozenControls(NamedTuple):
    ids: Tensor
    embeddings: Tensor


@final
@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    codec: InputByteCodec
    metadata: dict[str, Any]
    controls: FrozenControls


@final
@dataclass(frozen=True, slots=True)
class LoadedOutputCheckpoint:
    codec: OutputByteCodec
    metadata: dict[str, Any]
    control_ids: Tensor


def _checkpoint_state(codec: torch.nn.Module, storage_dtype: torch.dtype) -> dict[str, Tensor]:
    return {
        name: value.detach().to(
            device="cpu",
            dtype=storage_dtype if value.is_floating_point() else value.dtype,
        )
        for name, value in codec.state_dict().items()
    }


def _save_checkpoint_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint_payload(
    path: Path,
    *,
    direction: CodecDirection,
    expected_keys: set[str],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != expected_keys or payload["direction"] != direction:
        label = direction.removesuffix("_only")
        raise ValueError(f"checkpoint is not a current {label} codec")
    state_dict = payload["state_dict"]
    forbidden = [key for key in state_dict if "original" in key or "vocabulary_table" in key]
    if forbidden:
        raise ValueError(f"checkpoint contains forbidden source-table keys: {forbidden}")
    return payload


def save_checkpoint(
    path: Path,
    codec: InputByteCodec,
    metadata: dict[str, Any],
    *,
    control_ids: Tensor | None = None,
    control_embeddings: Tensor | None = None,
) -> None:
    if (control_ids is None) != (control_embeddings is None):
        raise ValueError("control IDs and embeddings must be provided together")
    ids = torch.empty(0, dtype=torch.long) if control_ids is None else control_ids.detach().cpu()
    embeddings = torch.empty((0, codec.config.embedding_dim), dtype=codec.dtype) if control_embeddings is None else control_embeddings.detach().cpu()
    if embeddings.shape != (ids.numel(), codec.config.embedding_dim):
        raise ValueError("control embeddings must match control IDs and embedding dimension")
    storage_dtype = codec.byte_embeddings.dtype
    payload = {
        "direction": "input_only",
        "config": codec.config.to_dict(),
        "metadata": metadata,
        "state_dict": _checkpoint_state(codec, storage_dtype),
        "control_ids": ids,
        "control_embeddings": embeddings,
    }
    _save_checkpoint_payload(path, payload)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    payload = _load_checkpoint_payload(
        path,
        direction="input_only",
        expected_keys={
            "direction",
            "config",
            "metadata",
            "state_dict",
            "control_ids",
            "control_embeddings",
        },
    )
    state_dict = payload["state_dict"]
    config = InputByteCodecConfig(**payload["config"])
    parameter_dtype = state_dict["input_projection.weight"].dtype
    codec = InputByteCodec(config, state_dict["byte_embeddings"]).to(dtype=parameter_dtype)
    codec.load_state_dict(state_dict)
    codec.to(device)
    if codec.device.type == "mps":
        codec.compile_neural_paths()
    codec.eval()
    ids = payload["control_ids"]
    embeddings = payload["control_embeddings"]
    controls = FrozenControls(ids.to(device), embeddings.to(device))
    if controls.embeddings.shape != (controls.ids.numel(), config.embedding_dim):
        raise ValueError("checkpoint control embeddings do not match control IDs")
    return LoadedCheckpoint(codec, dict(payload["metadata"]), controls)


def save_output_checkpoint(
    path: Path,
    codec: OutputByteCodec,
    metadata: dict[str, Any],
    *,
    control_ids: Tensor,
) -> None:
    if control_ids.shape != (codec.config.control_count,):
        raise ValueError("control IDs must match the output codec control count")
    payload = {
        "direction": "output_only",
        "config": codec.config.to_dict(),
        "metadata": metadata,
        "state_dict": _checkpoint_state(codec, codec.dtype),
        "control_ids": control_ids.detach().cpu(),
    }
    _save_checkpoint_payload(path, payload)


def load_output_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedOutputCheckpoint:
    payload = _load_checkpoint_payload(
        path,
        direction="output_only",
        expected_keys={
            "direction",
            "config",
            "metadata",
            "state_dict",
            "control_ids",
        },
    )
    state_dict = payload["state_dict"]
    config = OutputByteCodecConfig(**payload["config"])
    parameter_dtype = state_dict["hidden_projection.weight"].dtype
    codec = OutputByteCodec(config).to(dtype=parameter_dtype)
    codec.load_state_dict(state_dict)
    codec.to(device)
    if codec.device.type == "mps":
        codec.compile_neural_paths()
    codec.eval()
    control_ids = payload["control_ids"].to(device)
    if control_ids.shape != (config.control_count,):
        raise ValueError("checkpoint controls do not match the output codec")
    return LoadedOutputCheckpoint(codec, dict(payload["metadata"]), control_ids)


def cache_namespace(model_revision: str, codec_hash: str) -> str:
    if not model_revision or not codec_hash:
        raise ValueError("model revision and codec hash must not be empty")
    return f"{model_revision}:{codec_hash}"


def checkpoint_tensor_inventory(path: Path) -> dict[str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {
        **{f"codec.{name}": tensor for name, tensor in payload["state_dict"].items()},
        "controls.ids": payload.get("control_ids", torch.empty(0, dtype=torch.long)),
        "controls.embeddings": payload.get("control_embeddings", torch.empty(0)),
    }
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": tensor_bytes(tensor),
        }
        for name, tensor in sorted(tensors.items())
    }
