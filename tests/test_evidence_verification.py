from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    verify_artifact,
    write_evidence_manifest,
)
from continuous_tokenizer.contracts.claims import (
    CLAIM_VOCABULARY_SHA256,
)
from continuous_tokenizer.reporting.replication import aggregate_runs
from tests.test_replication import write_completed_run


def _write_rehashed_replication(
    artifact: Path,
    evidence_path: Path,
    evidence: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    evidence["artifacts"]["replication"]["sha256"] = hashlib.sha256(
        artifact.read_bytes(),
    ).hexdigest()
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")


def _verify_rehashed_replication(
    output: Path,
    artifact: Path,
    evidence_path: Path,
    evidence: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    _write_rehashed_replication(
        artifact,
        evidence_path,
        evidence,
        payload,
    )
    return verify_artifact(output)


def _assert_false_exactness_is_rejected(
    output: Path,
    artifact: Path,
    evidence_path: Path,
    evidence: dict[str, Any],
    replication: dict[str, Any],
) -> None:
    fabricated = deepcopy(replication)
    fabricated["seed_evidence"][0]["metrics"]["density_exact"] = False
    assert (
        fabricated["seed_evidence"][0]["metrics"]["native_tokens_per_continuous_token"]
        > fabricated["seed_evidence"][0]["thresholds"]["minimum_native_tokens_per_continuous_token"]
    )
    result = _verify_rehashed_replication(
        output,
        artifact,
        evidence_path,
        evidence,
        fabricated,
    )
    assert not result["valid"]
    assert any("claims differ from raw directional metrics" in error for error in result["errors"])


def test_recursive_evidence_verification_checks_hashes_seeds_and_intervals() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "inputs"
        source.mkdir()
        contract = source / "run-manifests.json"
        contract.write_text('{"runs":[17,23,41]}', encoding="utf-8")
        output = root / "replication"
        output.mkdir()
        runs = tuple(
            write_completed_run(
                root / f"run-{seed}",
                seed=seed,
                density=density,
            )
            for seed, density in ((17, 1.2), (23, 1.4), (41, 1.3))
        )
        replication = aggregate_runs(runs)
        artifact = output / "replication.json"
        artifact.write_text(json.dumps(replication), encoding="utf-8")
        write_evidence_manifest(
            output,
            EvidenceManifest(
                artifact_kind="replication",
                mode="input_only",
                status="completed",
                identity=EvidenceIdentity(
                    source_commit="commit",
                    source_dirty=True,
                    source_state_sha256="a" * 64,
                    dependency_lock_sha256="b" * 64,
                    installed_package={
                        "name": "continuous-byte-tokenizer",
                        "version": "0.1.0",
                        "content_sha256": "c" * 64,
                    },
                    claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
                    source_assets=replication["source_assets"],
                    verification={"all_provided": True, "all_passed": True},
                    model_id="Qwen/Qwen3.5-0.8B",
                    model_revision="2fc06364715b967f1860aea9cf38778875588b17",
                ),
                parents={},
                inputs={"run_contracts": contract},
                artifacts={"replication": artifact},
            ),
        )

        verified = verify_artifact(output)

        assert verified["valid"], verified
        assert not verified["errors"]
        assert str(artifact.resolve()) in verified["checked"]

        evidence_path = output / "evidence-manifest.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        fabricated = deepcopy(replication)
        fabricated["scientific_verdict"] = "unsupported"
        headline_result = _verify_rehashed_replication(
            output,
            artifact,
            evidence_path,
            evidence,
            fabricated,
        )

        assert not headline_result["valid"]
        assert any("scientific verdict differs from canonical headline claims" in error for error in headline_result["errors"])

        fabricated = deepcopy(replication)
        fabricated_claim = next(claim for claim in fabricated["claims"] if claim["claim_id"] == "input.held_out_position_compression")
        fabricated_claim["role"] = "secondary"
        role_result = _verify_rehashed_replication(
            output,
            artifact,
            evidence_path,
            evidence,
            fabricated,
        )

        assert not role_result["valid"]
        assert any("differs from the vocabulary" in error for error in role_result["errors"])

        _assert_false_exactness_is_rejected(
            output,
            artifact,
            evidence_path,
            evidence,
            replication,
        )

        fabricated = deepcopy(replication)
        fabricated_claim = next(claim for claim in fabricated["claims"] if claim["claim_id"] == "input.held_out_position_compression")
        fabricated_claim["verdict"] = "unsupported"
        fabricated_result = _verify_rehashed_replication(
            output,
            artifact,
            evidence_path,
            evidence,
            fabricated,
        )

        assert not fabricated_result["valid"]
        assert any("claims differ from raw directional metrics" in error for error in fabricated_result["errors"])

        missing = deepcopy(replication)
        missing["claims"] = [claim for claim in missing["claims"] if claim["claim_id"] != "input.held_out_position_compression"]
        missing_result = _verify_rehashed_replication(
            output,
            artifact,
            evidence_path,
            evidence,
            missing,
        )

        assert not missing_result["valid"]
        assert any("required claims" in error for error in missing_result["errors"])

        artifact.write_text('{"tampered":true}', encoding="utf-8")
        tampered = verify_artifact(output)
        assert not tampered["valid"]
        assert any("hash mismatch" in error for error in tampered["errors"])


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite((unittest.FunctionTestCase(test_recursive_evidence_verification_checks_hashes_seeds_and_intervals),))
