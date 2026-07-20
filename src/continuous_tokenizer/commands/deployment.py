from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.artifacts.manifest import load_verified_run_manifest
from continuous_tokenizer.artifacts.store import RunDirectory, load_json_object
from continuous_tokenizer.campaigns.deployment import run_deployment
from continuous_tokenizer.contracts.deployment import DeploymentSpec
from continuous_tokenizer.contracts.output import (
    OUTPUT_FIDELITY_PROMPT_SET,
    OUTPUT_FIDELITY_PROMPT_SET_SHA256,
)


def deployment(args: argparse.Namespace) -> dict[str, Any]:
    return run_deployment(
        DeploymentSpec.load(args.spec),
        args.output_dir,
    )


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _deployment_spec_text(
    quality_run: Path,
    output_path: Path,
) -> str:
    manifest_path = quality_run / "manifest-final.json"
    manifest = load_verified_run_manifest(manifest_path)
    result = load_json_object(quality_run / "result.json")
    if (
        manifest.status != "passed"
        or result.get("mode") != manifest.mode
        or result.get("evidence_scope") != "final"
        or result.get("operational_status") != "completed"
    ):
        raise ValueError(
            "deployment specification requires a sealed completed final quality run",
        )
    checkpoint_relative = manifest.artifacts.get("checkpoint")
    checkpoint_hash = manifest.artifact_hashes.get("checkpoint")
    if checkpoint_relative is None or checkpoint_hash is None:
        raise ValueError("sealed quality run has no checkpoint artifact")
    checkpoint = quality_run / checkpoint_relative
    if sha256_path(checkpoint) != checkpoint_hash:
        raise ValueError("sealed quality-run checkpoint hash mismatch")
    device = manifest.environment.get("device")
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("sealed quality run has no supported deployment device")
    relative_run = os.path.relpath(quality_run.resolve(), output_path.parent.resolve())
    relative_checkpoint = os.path.relpath(checkpoint.resolve(), output_path.parent.resolve())
    values = {
        "name": f"{quality_run.name}-deployment",
        "mode": manifest.mode,
        "device": device,
        "quality_run": relative_run,
        "quality_manifest_sha256": sha256_path(manifest_path),
        "checkpoint": relative_checkpoint,
        "checkpoint_sha256": checkpoint_hash,
    }
    lines = [f"{name} = {_toml_string(str(value))}" for name, value in values.items()]
    lines.extend(
        (
            f"prompt_set = {_toml_string(OUTPUT_FIDELITY_PROMPT_SET)}",
            f"prompt_set_sha256 = {_toml_string(OUTPUT_FIDELITY_PROMPT_SET_SHA256)}",
            "repetitions = 3",
            "max_steps = 16",
            "max_bytes = 1024",
        ),
    )
    return "\n".join(lines) + "\n"


def deployment_spec(args: argparse.Namespace) -> dict[str, Any]:
    output = RunDirectory(args.output_dir)
    generated: list[str] = []
    names: set[str] = set()
    for quality_run in args.quality_runs:
        path = quality_run.resolve()
        name = f"{path.name}.toml"
        if name in names:
            raise ValueError(f"deployment specification filename collision: {name}")
        names.add(name)
        destination = output.path(name)
        output.write_text(name, _deployment_spec_text(path, destination))
        DeploymentSpec.load(destination)
        generated.append(str(destination))
    return {
        "status": "completed",
        "specifications": generated,
    }
