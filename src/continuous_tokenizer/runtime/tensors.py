from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch
from torch import Tensor, nn


def tensor_fingerprint(tensors: Iterable[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_fingerprint(module: nn.Module) -> str:
    return tensor_fingerprint(module.state_dict().items())


def module_state_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def module_bytes(module: nn.Module) -> int:
    return sum(tensor_bytes(tensor) for tensor in module.state_dict().values())


def cache_tensor_bytes(cache: object) -> int:
    total = 0
    seen: set[tuple[str, int]] = set()

    def add(value: object) -> None:
        nonlocal total
        if isinstance(value, Tensor):
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr())
            if identity not in seen:
                seen.add(identity)
                total += storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                add(item)
        elif isinstance(value, list | tuple):
            for item in value:
                add(item)

    for layer in getattr(cache, "layers", ()):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            add(getattr(layer, name, None))
    return total
