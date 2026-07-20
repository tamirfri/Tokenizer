from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.artifacts.store import load_json_object, write_json_atomic


@dataclass(slots=True)
class ResumeManager:
    root: Path
    experiment_fingerprint: str
    source_commit: str
    source_state_sha256: str
    dependency_lock_sha256: str
    resuming: bool
    snapshot_interval: int = 1
    _snapshot_count: int = field(default=0, init=False, repr=False)
    _snapshot_bytes: int = field(default=0, init=False, repr=False)
    _snapshot_seconds: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.snapshot_interval < 1:
            raise ValueError("resume snapshot interval must be positive")

    def save(self, phase: str, epoch: int, state: Mapping[str, Any]) -> dict[str, int | float]:
        self._validate_phase(phase)
        completed = state.get("completed") is True
        state_path = self.root / "phase-final" / f"{phase}.pt" if completed else self.recovery_root / f"{phase}.pt"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        if completed and state_path.exists():
            raise FileExistsError(f"phase-final evidence already exists: {phase}")
        payload = {
            "phase": phase,
            "epoch": epoch,
            **self._source_contract(),
            "state": dict(state),
        }
        temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
        started = perf_counter()
        torch.save(payload, temporary)
        temporary.replace(state_path)
        elapsed = perf_counter() - started
        written_bytes = state_path.stat().st_size
        self._snapshot_count += 1
        self._snapshot_bytes += written_bytes
        self._snapshot_seconds += elapsed
        if completed:
            write_json_atomic(
                state_path.with_suffix(".json"),
                {
                    "kind": "immutable_phase_final",
                    "phase": phase,
                    "epoch": epoch,
                    **self._source_contract(),
                    "state": state_path.name,
                    "state_sha256": sha256_file(state_path),
                },
            )
            operational = self.recovery_root / f"{phase}.pt"
            operational.unlink(missing_ok=True)
        return {
            "snapshot_bytes": written_bytes,
            "snapshot_seconds": elapsed,
        }

    @property
    def recovery_root(self) -> Path:
        return self.root.parent / f".{self.root.name}.recovery"

    def should_snapshot(self, boundary: int) -> bool:
        return boundary % self.snapshot_interval == 0

    def telemetry(self) -> dict[str, int | float]:
        return {
            "snapshot_interval": self.snapshot_interval,
            "snapshots_written": self._snapshot_count,
            "snapshot_bytes_written": self._snapshot_bytes,
            "snapshot_write_seconds": self._snapshot_seconds,
        }

    def cleanup(self) -> None:
        if self.recovery_root.is_dir() and not any(self.recovery_root.iterdir()):
            self.recovery_root.rmdir()

    def latest(self, phase: str) -> dict[str, Any] | None:
        if not self.resuming:
            return None
        self._validate_phase(phase)
        final_path = self.root / "phase-final" / f"{phase}.pt"
        state_path = final_path if final_path.is_file() else self.recovery_root / f"{phase}.pt"
        if not state_path.is_file():
            return None
        if state_path == final_path:
            metadata = load_json_object(final_path.with_suffix(".json"))
            if set(metadata) != {
                "kind",
                "phase",
                "epoch",
                "experiment_fingerprint",
                "source_commit",
                "source_state_sha256",
                "dependency_lock_sha256",
                "state",
                "state_sha256",
            }:
                raise ValueError("phase-final evidence is not canonical")
            if metadata.get("state_sha256") != sha256_file(final_path):
                raise ValueError("phase-final snapshot hash does not match its evidence")
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("resume snapshot payload must be a mapping")
        expected = {
            "phase": phase,
            **self._source_contract(),
        }
        if set(payload) != {*expected, "epoch", "state"}:
            raise ValueError("resume snapshot payload is not canonical")
        if any(payload[name] != value for name, value in expected.items()):
            raise ValueError("resume snapshot does not match the experiment source contract")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError("resume snapshot state must be a mapping")
        return {
            "phase": phase,
            "epoch": int(payload["epoch"]),
            **state,
        }

    def _source_contract(self) -> dict[str, str]:
        return {
            "experiment_fingerprint": self.experiment_fingerprint,
            "source_commit": self.source_commit,
            "source_state_sha256": self.source_state_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }

    @staticmethod
    def _validate_phase(phase: str) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", phase) is None:
            raise ValueError("resume phase must contain only lowercase letters, digits, and hyphens")


def capture_torch_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(device)
    return state


def restore_torch_rng_state(device: torch.device, state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    accelerator = state.get("accelerator")
    if accelerator is None:
        return
    if device.type == "mps":
        torch.mps.set_rng_state(accelerator)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(accelerator, device)
