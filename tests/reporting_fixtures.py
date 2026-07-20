from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256


@dataclass(frozen=True, slots=True)
class ManifestOptions:
    model: str = "continuous-tokenizer/synthetic-model"
    revision: str = "revision"
    claims_passed: bool | None = None
    profile: str = "large"


_DEFAULT_MANIFEST_OPTIONS = ManifestOptions()


def write_manifest(
    directory: Path,
    *,
    experiment: str,
    status: str,
    options: ManifestOptions = _DEFAULT_MANIFEST_OPTIONS,
) -> None:
    directory.mkdir(parents=True)
    model = options.model
    revision = options.revision
    current_profile = options.profile
    manifest = {
        "experiment_name": experiment,
        "mode": "input_only",
        "codec_direction": "input_only",
        "experiment_fingerprint": "a" * 64,
        "replication_fingerprint": "b" * 64,
        "model_id": model,
        "model_revision": revision,
        "dataset_id": "synthetic/data",
        "dataset_revision": "revision",
        "embedding_tensor": "embed.weight",
        "source_dtype": "torch.float32",
        "seed": 17,
        "stages": ["vocabulary"],
        "source_commit": "commit",
        "source_dirty": False,
        "source_state_sha256": "a" * 64,
        "dependency_lock_sha256": "b" * 64,
        "installed_package": {
            "name": "continuous-byte-tokenizer",
            "version": "0.1.0",
            "content_sha256": "c" * 64,
        },
        "claim_vocabulary_sha256": CLAIM_VOCABULARY_SHA256,
        "source_assets": {
            "model_config": {
                "locator": "synthetic://model/config",
                "sha256": "f" * 64,
            }
        },
        "inputs": {},
        "codec_attention": {"query_heads": 4, "key_value_heads": 2, "enable_gqa": True},
        "environment": {"device": "cpu"},
        "trainable_parameters": [],
        "frozen_backbone_fingerprint": None,
        "native_head_used": True,
        "feedback_policy": "native_output_tokens",
        "artifacts": {"experiment": "experiment.json"},
        "artifact_hashes": {},
        "status": status,
        "verification": {"provided": False},
    }
    (directory / "experiment.json").write_text(
        json.dumps(
            {
                "name": experiment,
                "mode": "input_only",
                "seed": 17,
                "model": {
                    "model_id": model,
                    "revision": revision,
                    "evaluation": "full",
                },
                "dataset": {
                    "dataset_id": "synthetic/data",
                    "config": "default",
                    "revision": "revision",
                },
                "training": {"profile": current_profile},
            }
        ),
        encoding="utf-8",
    )
    if status == "passed":
        synthetic = model == "continuous-tokenizer/synthetic-model"
        if synthetic:
            scope = "synthetic"
            verdict = "not_evaluated"
        elif current_profile == "small":
            scope = "diagnostic"
            verdict = "not_applicable_diagnostic"
        else:
            scope = "final"
            verdict = "unsupported"
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "mode": "input_only",
                    "evidence_scope": scope,
                    "operational_status": "completed",
                    "scientific_verdict": verdict,
                    "experiment": {
                        "name": experiment,
                        "model": {"model_id": model},
                        "training": {"profile": current_profile},
                    },
                }
            ),
            encoding="utf-8",
        )
    if options.claims_passed is not None:
        (directory / "tokenizer-metrics.json").write_text(
            json.dumps(
                {
                    "kind": "tokenizer_metrics",
                    "acceptance": {"overall": options.claims_passed},
                }
            ),
            encoding="utf-8",
        )
    refresh_manifest(directory, manifest=manifest)


def refresh_manifest(
    directory: Path,
    *,
    manifest: Mapping[str, object] | None = None,
) -> None:
    path = directory / "manifest-final.json"
    if manifest is None:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    artifacts = {
        "experiment": "experiment.json",
        **({"verification": "verification"} if (directory / "verification").is_dir() else {}),
        **{
            name: filename
            for name, filename in (
                ("result", "result.json"),
                ("tokenizer", "tokenizer-metrics.json"),
                ("failure", "failure.json"),
                ("report", "artifact-report.md"),
            )
            if (directory / filename).is_file()
        },
    }
    current = {
        **manifest,
        "artifacts": artifacts,
        "artifact_hashes": {name: sha256_path(directory / filename) for name, filename in artifacts.items()},
    }
    path.write_text(json.dumps(current), encoding="utf-8")
