from __future__ import annotations

import json
import math
import statistics
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist
from typing import Any

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    verify_artifact,
    write_evidence_manifest,
)
from continuous_tokenizer.artifacts.hashing import cached_sha256_path, sha256_path
from continuous_tokenizer.artifacts.manifest import load_verified_run_manifest
from continuous_tokenizer.artifacts.store import RunDirectory
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256
from continuous_tokenizer.contracts.manifest import RunManifest
from continuous_tokenizer.contracts.profiles import (
    CAMPAIGN_PROFILE_NAME,
)


def test_run_directory_writes_atomically_and_refuses_overwrite() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "run"
        run = RunDirectory(root)

        path = run.write_json("nested/result.json", {"passed": True})

        assert path.read_text(encoding="utf-8").endswith("\n")
        assert not list(root.rglob("*.tmp"))
        with unittest.TestCase().assertRaisesRegex(FileExistsError, "already exists"):
            RunDirectory(root)


def test_artifact_path_hashes_files_and_directories_deterministically() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first"
        second = root / "second"
        (first / "nested").mkdir(parents=True)
        second.mkdir()
        (first / "nested/b.bin").write_bytes(b"second")
        (first / "a.bin").write_bytes(b"first")
        (second / "a.bin").write_bytes(b"first")
        (second / "nested").mkdir()
        (second / "nested/b.bin").write_bytes(b"second")

        assert sha256_path(first / "a.bin") == sha256_path(second / "a.bin")
        assert sha256_path(first) == sha256_path(second)
        (second / "nested/b.bin").write_bytes(b"changed")
        assert sha256_path(first) != sha256_path(second)


def test_cached_artifact_hash_invalidates_same_size_replacement() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.json"
        path.write_bytes(b'{"value":1}')
        first = cached_sha256_path(path)

        path.write_bytes(b'{"value":2}')

        assert cached_sha256_path(path) != first


def write_completed_run(
    root: Path,
    *,
    seed: int,
    density: float | None,
    profile: str = CAMPAIGN_PROFILE_NAME,
) -> Path:
    root.mkdir()
    completed = density is not None
    manifest = RunManifest(
        experiment_name="replication",
        mode="input_only",
        codec_direction="input_only",
        experiment_fingerprint=f"{seed:064x}",
        replication_fingerprint="b" * 64,
        model_id="synthetic/model",
        model_revision="model-revision",
        dataset_id="synthetic/data",
        dataset_revision="data-revision",
        embedding_tensor="embedding.weight",
        source_dtype="torch.float32",
        seed=seed,
        stages=("vocabulary", "reconstruction"),
        source_commit="commit",
        source_dirty=False,
        source_state_sha256="c" * 64,
        dependency_lock_sha256="d" * 64,
        installed_package={
            "name": "continuous-byte-tokenizer",
            "version": "0.1.0",
            "content_sha256": "e" * 64,
        },
        claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
        source_assets={
            "model_config": {
                "locator": "synthetic://model/config",
                "sha256": "f" * 64,
            }
        },
        inputs={},
        codec_attention={"query_heads": 4, "key_value_heads": 2, "enable_gqa": True},
        environment={"python": "3.14"},
        trainable_parameters=("encoder.weight",),
        frozen_backbone_fingerprint=None,
        native_head_used=True,
        feedback_policy="native_output_tokens",
        artifacts=(
            {
                "tokenizer_metrics": "tokenizer-metrics.json",
                "result": "result.json",
                "experiment": "experiment.json",
            }
            if completed
            else {"result": "result.json", "experiment": "experiment.json"}
        ),
        artifact_hashes={},
        status="passed" if completed else "failed",
    )
    (root / "experiment.json").write_text(
        json.dumps(
            {
                "name": "replication",
                "mode": "input_only",
                "seed": seed,
                "model": {
                    "model_id": "synthetic/model",
                    "revision": "model-revision",
                    "evaluation": "full",
                },
                "dataset": {
                    "dataset_id": "synthetic/data",
                    "config": "default",
                    "revision": "data-revision",
                },
                "training": {"profile": profile},
            }
        ),
        encoding="utf-8",
    )
    if not completed:
        result = {
            "mode": "input_only",
            "evidence_scope": "final",
            "operational_status": "completed",
            "scientific_verdict": "unsupported",
            "training": {
                "candidate_reference_state_ratio": 0.4,
                "compatibility_embedding_metrics": {"normalized_rmse": 0.2},
            },
        }
        (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
        manifest = replace(
            manifest,
            artifact_hashes={
                "experiment": sha256_path(root / "experiment.json"),
                "result": sha256_path(root / "result.json"),
            },
        )
        (root / "manifest-final.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
        return root
    metrics = {
        "acceptance": {
            "overall": True,
            "embedding_fit": True,
            "density": True,
            "compactness": True,
        },
        "density": {"native_tokens_per_continuous_token": density},
        "compactness": {"candidate_reference_state_ratio": 0.25},
        "embedding_fit": {"normalized_rmse": 0.001},
    }
    (root / "tokenizer-metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    result = {
        "mode": "input_only",
        "evidence_scope": "final",
        "operational_status": "completed",
        "scientific_verdict": "supported",
        "experiment": {"seed": seed, "training": {"profile": profile}},
        "tokenizer": metrics,
    }
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    manifest = replace(
        manifest,
        artifact_hashes={
            "tokenizer_metrics": sha256_path(root / "tokenizer-metrics.json"),
            "result": sha256_path(root / "result.json"),
            "experiment": sha256_path(root / "experiment.json"),
        },
    )
    (root / "manifest-final.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return root


def test_manifest_loading_rejects_noncanonical_fields_and_verifies_hashes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = write_completed_run(Path(directory) / "run", seed=17, density=1.2)
        manifest_path = root / "manifest-final.json"

        manifest = load_verified_run_manifest(manifest_path)
        assert manifest.artifact_hashes == {
            "tokenizer_metrics": sha256_path(root / "tokenizer-metrics.json"),
            "result": sha256_path(root / "result.json"),
            "experiment": sha256_path(root / "experiment.json"),
        }

        values = json.loads(manifest_path.read_text(encoding="utf-8"))
        values["obsolete"] = True
        manifest_path.write_text(json.dumps(values), encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(ValueError, "unknown manifest fields"):
            load_verified_run_manifest(manifest_path)

        values.pop("obsolete")
        manifest_path.write_text(json.dumps(values), encoding="utf-8")
        (root / "tokenizer-metrics.json").write_text("{}", encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(ValueError, "artifact hash mismatch"):
            load_verified_run_manifest(manifest_path)


def test_artifact_verification_rejects_timing_arithmetic_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = write_completed_run(
            Path(directory) / "run",
            seed=17,
            density=1.2,
        )
        result_path = root / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["llm"] = {
            "performance": {
                "schema_version": 1,
                "measurement": {
                    "expected_raw_pairs": 1,
                    "recorded_raw_pairs": 1,
                },
                "raw_pairs": [],
                "native": {},
                "compatibility": {},
                "segmented": {},
            }
        }
        result_path.write_text(json.dumps(result), encoding="utf-8")
        manifest_path = root / "manifest-final.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_hashes"]["result"] = sha256_path(result_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        verification = verify_artifact(root)

    assert not verification["valid"]
    assert "prefill raw pair count differs from measurement metadata" in verification["errors"]


def _performance_ablation() -> dict[str, Any]:
    semantic = "f" * 64
    baseline = [2.0 + index / 100 for index in range(20)]
    optimized = [value * 0.75 for value in baseline]
    log_ratios = [
        math.log(optimized_value / baseline_value)
        for baseline_value, optimized_value in zip(
            baseline,
            optimized,
            strict=True,
        )
    ]
    mean = statistics.fmean(log_ratios)
    half_width = NormalDist().inv_cdf(0.975) * statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
    condition = {
        "warmups": 5,
        "repetitions": 20,
        "no_concurrent_accelerator_work": True,
        "no_concurrent_processes": True,
        "semantic_sha256": semantic,
        "raw_pairs": [
            {
                "repetition": index,
                "baseline_seconds": baseline_value,
                "optimized_seconds": optimized_value,
                "baseline_semantic_sha256": semantic,
                "optimized_semantic_sha256": semantic,
            }
            for index, (baseline_value, optimized_value) in enumerate(
                zip(baseline, optimized, strict=True),
            )
        ],
        "summary": {
            "baseline_median_seconds": statistics.median(baseline),
            "optimized_median_seconds": statistics.median(optimized),
            "median_ratio": statistics.median(optimized) / statistics.median(baseline),
            "geometric_mean_ratio": math.exp(mean),
            "confidence_method": "paired_log_ratio_normal",
            "confidence_level": 0.95,
            "confidence_95_low": math.exp(mean - half_width),
            "confidence_95_high": math.exp(mean + half_width),
        },
    }
    identity = {
        "source_state_sha256": "a" * 64,
        "dependency_lock_sha256": "b" * 64,
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "revision",
        "data_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "workload_sha256": "e" * 64,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "performance_ablation",
        "mode": "input_only",
        "evidence_scope": "operational_secondary",
        "operational_status": "completed",
        "final_evidence": False,
        "evidence_role": "operational_and_secondary_only",
        "baseline": identity,
        "optimized": {**identity, "source_state_sha256": "0" * 64},
        "optimization_ids": ["adaptive-frontier"],
        "semantic_sha256": semantic,
        "conditions": {
            name: json.loads(json.dumps(condition))
            for name in (
                "cold_compile",
                "warm_compile",
                "cache_disabled",
                "cache_cold",
                "cache_warm",
            )
        },
    }


def test_performance_ablation_recomputes_summaries_and_rejects_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "ablation"
        root.mkdir()
        artifact_path = root / "performance-ablation.json"
        artifact = _performance_ablation()
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        write_evidence_manifest(
            root,
            EvidenceManifest(
                artifact_kind="performance_ablation",
                mode="input_only",
                status="completed",
                identity=EvidenceIdentity(
                    source_commit="commit",
                    source_dirty=False,
                    source_state_sha256="0" * 64,
                    dependency_lock_sha256="b" * 64,
                    installed_package={
                        "name": "continuous-byte-tokenizer",
                        "version": "0.1.0",
                        "content_sha256": "c" * 64,
                    },
                    claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
                    source_assets={},
                    verification={"provided": False},
                    model_id="Qwen/Qwen3.5-0.8B",
                    model_revision="revision",
                ),
                parents={},
                inputs={},
                artifacts={"performance_ablation": artifact_path},
            ),
        )
        assert verify_artifact(root)["valid"]

        artifact["conditions"]["cache_warm"]["summary"]["median_ratio"] = 0.01
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        manifest_path = root / "evidence-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["performance_ablation"]["sha256"] = sha256_path(
            artifact_path,
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        verification = verify_artifact(root)

    assert not verification["valid"]
    assert any("summary differs from raw pairs" in error for error in verification["errors"])


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_run_directory_writes_atomically_and_refuses_overwrite,
            test_artifact_path_hashes_files_and_directories_deterministically,
            test_cached_artifact_hash_invalidates_same_size_replacement,
            test_manifest_loading_rejects_noncanonical_fields_and_verifies_hashes,
            test_artifact_verification_rejects_timing_arithmetic_tampering,
            test_performance_ablation_recomputes_summaries_and_rejects_tampering,
        )
    )
