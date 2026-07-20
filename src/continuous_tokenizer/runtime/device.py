from __future__ import annotations

import torch
from torch import nn


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def declared_device(name: str) -> torch.device:
    device = torch.device(name)
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if not available[name]:
        raise RuntimeError(f"declared device is unavailable: {name}")
    return device


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def resolve_model_device(
    requested: torch.device | None,
    frozen_model: nn.Module | None,
) -> torch.device:
    if frozen_model is None:
        return default_device() if requested is None else requested
    actual = module_device(frozen_model)
    if requested is not None and not _devices_equivalent(requested, actual):
        raise ValueError("the requested device does not match the frozen model")
    return actual


def _devices_equivalent(left: torch.device, right: torch.device) -> bool:
    return left.type == right.type and (left.index is None or right.index is None or left.index == right.index)
