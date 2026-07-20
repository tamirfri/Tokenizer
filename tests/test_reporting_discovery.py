from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from reporting_fixtures import ManifestOptions, write_manifest

import continuous_tokenizer.reporting.discovery as discovery_module
from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    write_evidence_manifest,
)
from continuous_tokenizer.commands.evidence import aggregate
from continuous_tokenizer.contracts.claims import (
    CLAIM_VOCABULARY_SHA256,
)
from continuous_tokenizer.reporting.discovery import (
    artifact_profile,
    discover_artifact_runs,
    discover_deployment_artifacts,
    discover_replication_artifacts,
    discover_report_artifacts,
    discover_search_artifacts,
    discover_state_budget_artifacts,
    discover_study_artifacts,
)
from tests.test_replication import write_completed_run, write_output_run
from tests.test_state_budget import _result


def _seal(
    directory: Path,
    *,
    kind: Literal[
        "deployment",
        "project",
        "replication",
        "search",
        "state_budget",
        "study",
    ],
    mode: str,
    status: str,
    artifact: Path,
) -> None:
    write_evidence_manifest(
        directory,
        EvidenceManifest(
            artifact_kind=kind,
            mode=mode,
            status=status,
            identity=EvidenceIdentity(
                source_commit="commit",
                source_dirty=False,
                source_state_sha256="a" * 64,
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
            artifacts={kind: artifact},
        ),
    )


def test_report_discovery_builds_one_directory_inventory() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "nested").mkdir()
        (root / "nested/ignored.txt").write_text("ignored", encoding="utf-8")
        with patch.object(
            discovery_module,
            "directory_files",
            wraps=discovery_module.directory_files,
        ) as inventory:
            assert discover_report_artifacts(root) == ()

        assert inventory.call_count == 1


def test_artifact_discovery_prefers_supported_claims_and_labels_verdicts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_completed_run(
            root / "real",
            seed=17,
            density=1.25,
        )
        write_manifest(root / "failed", experiment="failed-run", status="failed")
        write_manifest(
            root / "synthetic",
            experiment="synthetic-run",
            status="passed",
            options=ManifestOptions(claims_passed=True),
        )
        stale = root / "stale"
        stale.mkdir()
        (stale / "manifest-final.json").write_text("{}", encoding="utf-8")
        diagnostic = root / "diagnostic"
        diagnostic.mkdir()
        (diagnostic / "result.json").write_text(
            json.dumps(
                {
                    "mode": "input_only",
                    "evidence_scope": "diagnostic",
                    "operational_status": "completed",
                    "scientific_verdict": "not_applicable_diagnostic",
                    "experiment": {"name": "diagnostic-run"},
                    "tokenizer": {
                        "model": {"id": "Qwen/Qwen3.5-0.8B"},
                        "acceptance": {"overall": False},
                    },
                }
            ),
            encoding="utf-8",
        )
        search = root / "search"
        search.mkdir()
        (search / "search.json").write_text(
            json.dumps(
                {
                    "mode": "input_only",
                    "evidence_scope": "search",
                    "operational_status": "completed",
                    "scientific_verdict": "not_applicable_search",
                    "name": "alignment-search",
                    "model_id": "Qwen/Qwen3.5-0.8B",
                }
            ),
            encoding="utf-8",
        )
        _seal(
            search,
            kind="search",
            mode="input_only",
            status="completed",
            artifact=search / "search.json",
        )

        runs = discover_artifact_runs(root)

        assert [run.directory.name for run in runs] == [
            "real",
            "synthetic",
            "failed",
        ]
        assert runs[0].label == ("INPUT RUN | REAL MODEL | COMPLETED | SUPPORTED | replication | real")
        assert runs[1].label == ("INPUT RUN | SYNTHETIC | COMPLETED | NOT_EVALUATED | synthetic-run | synthetic")
        (search_artifact,) = discover_search_artifacts(root)
        assert search_artifact.label == ("INPUT SEARCH | COMPLETED | NOT_APPLICABLE_SEARCH | alignment-search | search")
        assert {artifact.kind for artifact in discover_report_artifacts(root)} == {"run", "search"}


def test_discovery_excludes_artifacts_from_an_old_claim_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        search = root / "old-search"
        search.mkdir()
        artifact = search / "search.json"
        artifact.write_text(
            json.dumps(
                {
                    "mode": "input_only",
                    "operational_status": "completed",
                    "scientific_verdict": "not_applicable_search",
                    "name": "old-search",
                    "model_id": "Qwen/Qwen3.5-0.8B",
                }
            ),
            encoding="utf-8",
        )
        _seal(
            search,
            kind="search",
            mode="input_only",
            status="completed",
            artifact=artifact,
        )
        manifest_path = search / "evidence-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["claim_vocabulary_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        assert discover_search_artifacts(root) == ()
        assert discover_report_artifacts(root) == ()


def test_replication_discovery_is_typed_mode_aware_and_not_counted_as_a_run() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_output_run(root / f"output-{seed}", seed) for seed in (17, 23, 41))
        replication = root / "nested" / "replication"
        aggregate(
            Namespace(
                runs=runs,
                output_dir=replication,
            ),
        )
        (replication / "result.json").write_text(
            json.dumps(
                {
                    "mode": "output_only",
                    "experiment": {"name": "embedded-result"},
                    "tokenizer": {"model": {"id": "Qwen/Qwen3.5-0.8B"}},
                }
            ),
            encoding="utf-8",
        )
        invalid = root / "invalid"
        invalid.mkdir()
        (invalid / "replication.json").write_text(
            json.dumps(
                {
                    "mode": "input_only",
                    "operational_status": "completed",
                    "scientific_verdict": "supported",
                    "model": {"id": "model"},
                }
            ),
            encoding="utf-8",
        )

        (artifact,) = discover_replication_artifacts(root)

        assert artifact.mode == "output_only"
        assert artifact.operational_status == "completed"
        assert artifact.scientific_verdict == "supported"
        assert artifact.model == "Qwen/Qwen3.5-0.8B"
        assert artifact.label == "OUTPUT REPLICATION | COMPLETED | SUPPORTED | replication"
        assert len(discover_artifact_runs(root)) == 3


def test_real_small_profile_is_labeled_diagnostic_without_a_claim_verdict() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_manifest(
            root / "small",
            experiment="small-pilot",
            status="passed",
            options=ManifestOptions(
                model="Qwen/Qwen3.5-0.8B",
                claims_passed=True,
                profile="small",
            ),
        )

        (run,) = discover_artifact_runs(root)

        assert run.status == "diagnostic"
        assert run.claims_passed is None
        assert run.label == ("INPUT RUN | REAL MODEL | COMPLETED | NOT_APPLICABLE_DIAGNOSTIC | small-pilot | small")


def test_artifact_profile_requires_canonical_experiment_shape() -> None:
    assert artifact_profile({"experiment": {"training": {"profile": "large"}}}) is None
    assert artifact_profile({"training": {"profile": "small"}}) == "small"


def test_discovery_quarantines_unsealed_or_semantically_invalid_evidence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        study = root / "study"
        study.mkdir()
        study_result = {
            "artifact_kind": "output_oracle_study",
            "mode": "output_only",
            "operational_status": "completed",
            "evidence_scope": "selection",
            "scientific_verdict": "not_applicable_selection",
            "model": {"id": "Qwen/Qwen3.5-0.8B"},
        }
        (study / "result.json").write_text(json.dumps(study_result))
        _seal(
            study,
            kind="study",
            mode="output_only",
            status="completed",
            artifact=study / "result.json",
        )
        deployment = root / "deployment"
        deployment.mkdir()
        deployment_result = {
            "kind": "deployment_evidence",
            "mode": "input_only",
            "operational_status": "completed",
            "applicability": {
                "applicable": False,
                "reason": "tied table",
            },
            "raw_repetitions": [],
            "physical_reference_tensor_absent": None,
            "output_equivalent": None,
            "hidden_equivalent": None,
            "deployment_compactness_claimable": None,
        }
        (deployment / "deployment.json").write_text(json.dumps(deployment_result))
        _seal(
            deployment,
            kind="deployment",
            mode="input_only",
            status="completed",
            artifact=deployment / "deployment.json",
        )
        invalid_replication = root / "invalid-replication"
        invalid_replication.mkdir()
        (invalid_replication / "replication.json").write_text(
            json.dumps(
                {
                    "mode": "input_only",
                    "evidence_scope": "replication",
                    "operational_status": "completed",
                    "replication_complete": True,
                    "runs": [],
                    "failed_runs": [],
                    "claims": [],
                    "metrics": {},
                }
            )
        )
        _seal(
            invalid_replication,
            kind="replication",
            mode="input_only",
            status="completed",
            artifact=invalid_replication / "replication.json",
        )
        unsealed = root / "unsealed"
        unsealed.mkdir()
        (unsealed / "search.json").write_text(
            json.dumps(
                {
                    "mode": "input_only",
                    "operational_status": "completed",
                    "evidence_scope": "search",
                    "scientific_verdict": "not_applicable_search",
                    "model_id": "model",
                    "name": "unsealed",
                }
            )
        )

        assert len(discover_study_artifacts(root)) == 1
        assert len(discover_deployment_artifacts(root)) == 1
        assert discover_replication_artifacts(invalid_replication) == ()
        assert discover_search_artifacts(unsealed) == ()

        (study / "result.json").write_text('{"tampered": true}')
        assert discover_study_artifacts(root) == ()


def test_state_budget_discovery_is_separate_and_requires_semantic_verification() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        budget = root / "budget"
        budget.mkdir()
        path = budget / "joint-state-budget.json"
        path.write_text(json.dumps(_result().to_dict()), encoding="utf-8")
        _seal(
            budget,
            kind="state_budget",
            mode="cross_directional",
            status="completed",
            artifact=path,
        )

        with patch(
            "continuous_tokenizer.reporting.discovery.verify_artifact",
            return_value={"valid": True, "errors": []},
        ):
            (artifact,) = discover_state_budget_artifacts(root)

        assert artifact.kind == "state_budget"
        assert artifact.mode == "cross_directional"
        assert artifact.model == ("Qwen/Qwen3.5-0.8B + google/gemma-3-270m-it")
        assert artifact.label.startswith("FUTURE PREREQUISITE | JOINT STATE BUDGET")

        with patch(
            "continuous_tokenizer.reporting.discovery.verify_artifact",
            return_value={"valid": False, "errors": ["invalid arithmetic"]},
        ):
            assert discover_state_budget_artifacts(root) == ()


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_report_discovery_builds_one_directory_inventory,
            test_artifact_discovery_prefers_supported_claims_and_labels_verdicts,
            test_discovery_excludes_artifacts_from_an_old_claim_contract,
            test_replication_discovery_is_typed_mode_aware_and_not_counted_as_a_run,
            test_real_small_profile_is_labeled_diagnostic_without_a_claim_verdict,
            test_artifact_profile_requires_canonical_experiment_shape,
            test_discovery_quarantines_unsealed_or_semantically_invalid_evidence,
            test_state_budget_discovery_is_separate_and_requires_semantic_verification,
        )
    )
