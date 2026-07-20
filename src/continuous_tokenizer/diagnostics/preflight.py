from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, final

import torch
from torch import nn

from continuous_tokenizer.artifacts.store import write_json_atomic
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.data.corpus import stream_corpus_documents
from continuous_tokenizer.runtime.compiler import compiler_cache_directory
from continuous_tokenizer.runtime.progress import log_event


@final
@dataclass(frozen=True, slots=True)
class PreflightCheck:
    passed: bool
    details: dict[str, Any]


def _directory_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _existing_parent(path: Path) -> Path:
    return next(candidate for candidate in (path, *path.parents) if candidate.exists())


def storage_check(path: Path, spec: ExperimentSpec) -> PreflightCheck:
    usage = shutil.disk_usage(_existing_parent(path))
    policy = spec.runtime
    cache_directory = compiler_cache_directory()
    artifact_bytes = _directory_bytes(path)
    cache_bytes = _directory_bytes(cache_directory)
    required = policy.projected_run_bytes + policy.storage_reserve_bytes
    return PreflightCheck(
        passed=usage.free >= required,
        details={
            "path": str(path),
            "filesystem_total_bytes": usage.total,
            "filesystem_used_bytes": usage.used,
            "free_bytes": usage.free,
            "existing_artifact_bytes": artifact_bytes,
            "inductor_cache_path": str(cache_directory),
            "inductor_cache_bytes": cache_bytes,
            "projected_run_bytes": policy.projected_run_bytes,
            "inductor_cache_estimate_bytes": policy.inductor_cache_estimate_bytes,
            "additional_inductor_cache_estimate_bytes": max(
                0,
                policy.inductor_cache_estimate_bytes - cache_bytes,
            ),
            "storage_reserve_bytes": policy.storage_reserve_bytes,
            "required_free_bytes": required,
            "projected_free_after_run_and_cache_bytes": (usage.free - policy.projected_run_bytes),
        },
    )


def require_storage(
    path: Path,
    spec: ExperimentSpec,
    *,
    refusal_message: str,
) -> None:
    projection = storage_check(path, spec)
    log_event(
        "storage_projection",
        experiment=spec.name,
        **projection.details,
    )
    if not projection.passed:
        raise RuntimeError(refusal_message)


def run_preflight(  # noqa: PLR0913 - Preflight inputs are explicit evidence dependencies.
    spec: ExperimentSpec,
    output_path: Path,
    *,
    device: torch.device,
    load_full_model: Callable[[], nn.Module] | None,
    identity: Mapping[str, Any],
    representative_mps_verified: bool = False,
) -> tuple[dict[str, Any], nn.Module | None]:
    if output_path.exists():
        raise FileExistsError(f"preflight artifact already exists: {output_path}")
    checks = {
        "storage": storage_check(output_path.parent, spec),
        "mps": _mps_check(device, spec),
        "inductor_cache": _cache_check(spec),
        "dataset_access": _dataset_check(spec),
        "cold_inductor_compilation": (
            PreflightCheck(
                True,
                {
                    "status": "reused_source_lock_device_bound_verification",
                    "device": str(device),
                    "work_avoided": "duplicate_cold_inductor_probe",
                },
            )
            if representative_mps_verified
            else _cold_compile_check(device)
        ),
    }
    model, full_model_check = _full_model_check(load_full_model)
    checks["full_model_access"] = full_model_check
    artifact = {
        "kind": "immutable_run_preflight",
        **identity,
        "policy": asdict(spec.runtime),
        "all_passed": all(check.passed for check in checks.values()),
        "checks": {
            name: {
                "passed": check.passed,
                **check.details,
            }
            for name, check in checks.items()
        },
    }
    write_json_atomic(output_path, artifact)
    if not artifact["all_passed"]:
        failed = ", ".join(name for name, check in checks.items() if not check.passed)
        raise RuntimeError(f"run preflight failed: {failed}")
    return artifact, model


def _mps_check(device: torch.device, spec: ExperimentSpec) -> PreflightCheck:
    if device.type != "mps":
        return PreflightCheck(True, {"status": "not_applicable", "device": str(device)})
    available = torch.backends.mps.is_available()
    recommended = torch.mps.recommended_max_memory() if available else 0
    return PreflightCheck(
        available and recommended >= spec.runtime.minimum_mps_memory_bytes,
        {
            "available": available,
            "built": torch.backends.mps.is_built(),
            "recommended_max_memory_bytes": recommended,
            "minimum_mps_memory_bytes": spec.runtime.minimum_mps_memory_bytes,
            "current_allocated_bytes": (torch.mps.current_allocated_memory() if available else 0),
            "driver_allocated_bytes": (torch.mps.driver_allocated_memory() if available else 0),
        },
    )


def _cache_check(spec: ExperimentSpec) -> PreflightCheck:
    directory = compiler_cache_directory()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".continuous-tokenizer-preflight-{os.getpid()}"
        probe.write_bytes(b"preflight")
        probe.unlink()
        usage = shutil.disk_usage(directory)
        return PreflightCheck(
            True,
            {
                "path": str(directory),
                "writable": True,
                "free_bytes": usage.free,
                "current_cache_bytes": _directory_bytes(directory),
                "estimated_cache_bytes": spec.runtime.inductor_cache_estimate_bytes,
                "storage_reserve_bytes": spec.runtime.storage_reserve_bytes,
            },
        )
    except OSError as error:
        return PreflightCheck(
            False,
            {
                "path": str(directory),
                "writable": False,
                "error": f"{type(error).__name__}: {error}",
            },
        )


def _dataset_check(spec: ExperimentSpec) -> PreflightCheck:
    try:
        document = next(
            iter(
                stream_corpus_documents(
                    "train",
                    dataset_id=spec.dataset.dataset_id,
                    config=spec.dataset.config,
                    revision=spec.dataset.revision,
                    max_rows=1,
                )
            )
        )
        return PreflightCheck(
            bool(document),
            {
                "dataset_id": spec.dataset.dataset_id,
                "revision": spec.dataset.revision,
                "sample_bytes": len(document),
            },
        )
    except Exception as error:  # noqa: BLE001 - Preflight records every access failure.
        return PreflightCheck(
            False,
            {
                "dataset_id": spec.dataset.dataset_id,
                "revision": spec.dataset.revision,
                "error": f"{type(error).__name__}: {error}",
            },
        )


def _cold_compile_check(device: torch.device) -> PreflightCheck:
    if device.type != "mps":
        return PreflightCheck(True, {"status": "not_applicable", "device": str(device)})
    try:

        def operation(value: torch.Tensor) -> torch.Tensor:
            return value.square().add(1)

        compiled = torch.compile(
            operation,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        result = compiled(torch.ones(8, device=device))
        torch.mps.synchronize()
        return PreflightCheck(
            bool(torch.equal(result.cpu(), torch.full((8,), 2.0))),
            {
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": False,
                "device": str(device),
            },
        )
    except Exception as error:  # noqa: BLE001 - Preflight records every compiler failure.
        return PreflightCheck(
            False,
            {
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": False,
                "error": f"{type(error).__name__}: {error}",
            },
        )


def _full_model_check(
    loader: Callable[[], nn.Module] | None,
) -> tuple[nn.Module | None, PreflightCheck]:
    if loader is None:
        return None, PreflightCheck(True, {"status": "not_applicable"})
    try:
        model = loader()
        return model, PreflightCheck(
            True,
            {
                "implementation": (f"{type(model).__module__}.{type(model).__qualname__}"),
            },
        )
    except Exception as error:  # noqa: BLE001 - Preflight records every model access failure.
        return None, PreflightCheck(
            False,
            {"error": f"{type(error).__name__}: {error}"},
        )
