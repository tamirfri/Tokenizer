from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.artifacts.store import load_json_object
from continuous_tokenizer.contracts.manifest import RunManifest


def load_artifact(path: Path) -> Mapping[str, Any]:
    return load_json_object(path)


def load_verified_run_manifest(path: Path) -> RunManifest:
    manifest = RunManifest.load(path)
    artifacts = {str(name): path.parent / str(relative) for name, relative in manifest.artifacts.items()}
    missing = sorted(name for name, artifact_path in artifacts.items() if not artifact_path.exists())
    if missing:
        raise ValueError(f"manifest references missing artifacts in {path}: {missing}")
    for name, artifact_path in artifacts.items():
        if manifest.artifact_hashes[name] != sha256_path(artifact_path):
            raise ValueError(f"artifact hash mismatch for {name} in {path}")
    return manifest
