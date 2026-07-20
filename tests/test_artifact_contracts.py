from __future__ import annotations

import ast
import unittest
from dataclasses import fields
from pathlib import Path

from continuous_tokenizer.contracts.manifest import RunManifest

_CAMPAIGN_ARTIFACT_FILENAMES = {
    "ablations.json",
    "artifact-report.md",
    "distillation.json",
    "evidence-manifest.json",
    "experiment.json",
    "failure.json",
    "llm-metrics.json",
    "llm-report.md",
    "manifest-final.json",
    "manifest-start.json",
    "output-metrics.json",
    "output-trajectory-cache-resume.json",
    "output-trajectory-cache.json",
    "performance-metrics.json",
    "project-report.md",
    "project.json",
    "replication-report.md",
    "replication.json",
    "result.json",
    "samples.jsonl",
    "search-report.md",
    "search-spec.json",
    "search.json",
    "selected-experiment.toml",
    "tokenizer-metrics.json",
    "tokenizer-report.md",
    "training-result.json",
    "verification.json",
    "vocabulary-sample.json",
}


class ArtifactContractTests(unittest.TestCase):
    def test_campaign_manifest_has_one_current_schema(self) -> None:
        repository = Path(__file__).parents[1]
        self.assertFalse((repository / "src/continuous_tokenizer/contracts/versions.py").exists())
        self.assertEqual(
            tuple(field.name for field in fields(RunManifest)),
            (
                "experiment_name",
                "mode",
                "codec_direction",
                "experiment_fingerprint",
                "replication_fingerprint",
                "model_id",
                "model_revision",
                "dataset_id",
                "dataset_revision",
                "embedding_tensor",
                "source_dtype",
                "seed",
                "stages",
                "source_commit",
                "source_dirty",
                "source_state_sha256",
                "dependency_lock_sha256",
                "installed_package",
                "claim_vocabulary_sha256",
                "source_assets",
                "inputs",
                "codec_attention",
                "environment",
                "trainable_parameters",
                "frozen_backbone_fingerprint",
                "native_head_used",
                "feedback_policy",
                "artifacts",
                "artifact_hashes",
                "status",
                "verification",
            ),
        )

    def test_campaign_artifact_filenames_remain_present_in_publishers_and_consumers(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / "continuous_tokenizer"
        literals: set[str] = set()
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            literals.update(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str))

        self.assertEqual(_CAMPAIGN_ARTIFACT_FILENAMES - literals, set())
