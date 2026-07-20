from __future__ import annotations

import errno
import json
import os
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

from continuous_tokenizer.artifacts.store import write_json_atomic
from continuous_tokenizer.runtime.tensors import tensor_bytes


def checkpoint_index(directory: Path) -> dict[str, Any]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))
    model_path = directory / "model.safetensors"
    if not model_path.is_file():
        raise ValueError(f"no Safetensors checkpoint found in {directory}")
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        weight_map = dict.fromkeys(handle.keys(), model_path.name)
    return {"metadata": {}, "weight_map": weight_map}


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)


def filter_checkpoint_tensor(
    source: Path,
    output: Path,
    tensor_name: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"filtered checkpoint already exists: {output}")
    index = checkpoint_index(source)
    weight_map = dict(index["weight_map"])
    tensor_shard = weight_map.pop(tensor_name, None)
    if tensor_shard is None:
        raise ValueError(f"tensor is absent from the source index: {tensor_name}")
    output.mkdir(parents=True)
    omitted_tensor_bytes = 0

    for shard_name in sorted(set(weight_map.values()) | {tensor_shard}):
        source_shard = source / shard_name
        target_shard = output / shard_name
        target_shard.parent.mkdir(parents=True, exist_ok=True)
        if shard_name != tensor_shard:
            _link_or_copy(source_shard, target_shard)
            continue
        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            omitted_tensor = handle.get_tensor(tensor_name)
            omitted_tensor_bytes = tensor_bytes(omitted_tensor)
            names = handle.keys()
            tensors = {name: handle.get_tensor(name) for name in names if name != tensor_name}
            metadata = handle.metadata()
        temporary = target_shard.with_suffix(f"{target_shard.suffix}.tmp")
        save_file(tensors, temporary, metadata=metadata)
        Path(temporary).replace(target_shard)

    shutil.copy2(source / "config.json", output / "config.json")
    total_size = sum(path.stat().st_size for path in output.glob("*.safetensors") if path.is_file())
    filtered_index = {
        "metadata": {**dict(index.get("metadata", {})), "total_size": total_size},
        "weight_map": weight_map,
    }
    write_json_atomic(output / "model.safetensors.index.json", filtered_index)
    with safe_open(output / tensor_shard, framework="pt", device="cpu") as handle:
        names = handle.keys()
        if tensor_name in names:
            raise RuntimeError("filtered shard still contains the omitted tensor")
    return {
        "directory": str(output),
        "tensor": tensor_name,
        "tensor_absent": tensor_name not in weight_map,
        "omitted_tensor_bytes": omitted_tensor_bytes,
        "serialized_tensor_count": len(weight_map),
        "serialized_bytes": total_size,
        "shards": len(set(weight_map.values())),
    }


def load_filtered_state(model: nn.Module, directory: Path) -> tuple[dict[str, Tensor], int]:
    weight_map = checkpoint_index(directory)["weight_map"]
    loaded: set[str] = set()
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(directory / shard_name, framework="pt", device="cpu") as handle:
            names = handle.keys()
            state = {name: handle.get_tensor(name) for name in names}
        result = model.load_state_dict(state, strict=False, assign=True)
        if result.unexpected_keys:
            raise ValueError(f"unexpected tensors in {shard_name}: {result.unexpected_keys}")
        loaded.update(state)

    model_state = dict(model.state_dict())
    missing = sorted(set(model_state) - loaded)
    meta = sorted(name for name, tensor in model_state.items() if tensor.is_meta)
    if missing or meta:
        raise ValueError(f"filtered model is incomplete; missing={missing[:5]}, meta={meta[:5]}")
    return model_state, len(loaded)
