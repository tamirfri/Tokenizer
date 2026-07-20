from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    load_evidence_manifest,
    write_evidence_manifest,
)
from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.contracts.claims import (
    CLAIM_ROLES,
    CLAIM_VOCABULARY,
    CLAIM_VOCABULARY_SHA256,
    claim_record,
)
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import InputGateSpec
from continuous_tokenizer.contracts.output import OutputGateSpec


def test_directional_claim_vocabulary_is_strict_and_content_addressed() -> None:
    identifiers = [claim.claim_id for claim in CLAIM_VOCABULARY]

    assert len(identifiers) == len(set(identifiers))
    assert len(CLAIM_VOCABULARY_SHA256) == 64
    assert {claim.mode for claim in CLAIM_VOCABULARY} == {
        "input_only",
        "output_only",
    }
    assert {claim.category for claim in CLAIM_VOCABULARY} == {
        "quality",
        "efficiency",
        "analytical",
        "deployment",
        "applicability",
    }
    assert {claim.basis for claim in CLAIM_VOCABULARY} == {
        "measured",
        "analytical",
        "applicability",
    }
    assert {claim.role for claim in CLAIM_VOCABULARY} == set(CLAIM_ROLES)
    assert [claim.claim_id for claim in CLAIM_VOCABULARY if claim.mode == "input_only" and claim.role == "primary"] == ["input.held_out_position_compression"]
    assert [claim.claim_id for claim in CLAIM_VOCABULARY if claim.mode == "output_only" and claim.role == "primary"] == ["output.semi_autoregressive_density"]
    assert all((claim.category == "applicability") == (claim.role == "applicability") for claim in CLAIM_VOCABULARY)
    assert all(
        "reporting.replication._" not in claim.producer_symbol and "reporting.project._attach" not in claim.producer_symbol for claim in CLAIM_VOCABULARY
    )
    assert (
        claim_record(
            "input_only",
            "input.held_out_position_compression",
            "supported",
        ).to_dict()["role"]
        == "primary"
    )
    serialized_claims = repr(CLAIM_VOCABULARY)
    assert "input.held_out_position_compression" in identifiers
    assert "input.fixed_subset_alignment_feasibility" in identifiers
    assert all(term not in serialized_claims for term in ("lossless_dense", "/dense", " dense "))
    with unittest.TestCase().assertRaisesRegex(ValueError, "not registered"):
        claim_record(
            "output_only",
            "input.held_out_position_compression",
            "supported",
        )


def test_every_directional_experiment_registers_current_claim_gates() -> None:
    repository = Path(__file__).parents[1]
    paths = sorted(path for path in (repository / "experiments").rglob("*.toml") if "[gates]" in path.read_text(encoding="utf-8"))
    specifications = [ExperimentSpec.load(path) for path in paths]

    assert len(specifications) == 28
    for specification in specifications:
        if isinstance(specification.gates, InputGateSpec):
            assert specification.gates.minimum_native_tokens_per_continuous_token == 1.10
            assert specification.gates.maximum_candidate_reference_state_ratio == 0.50
            assert specification.gates.maximum_segmented_mean_kl == 0.10
            assert specification.gates.minimum_segmented_top1_agreement == 0.90
        else:
            assert isinstance(specification.gates, OutputGateSpec)
            assert specification.gates.minimum_native_tokens_per_attempted_macro_step == 1.10
            assert specification.gates.minimum_rollout_event_agreement == 0.50
            assert specification.gates.maximum_invalid_events == 0
            assert specification.gates.minimum_stop_precision == 1.0
            assert specification.gates.minimum_stop_recall == 1.0


def test_evidence_manifest_uses_portable_locators_and_refuses_missing_inputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "evidence"
        output.mkdir()
        parent = root / "parent.json"
        parent.write_text("{}", encoding="utf-8")
        artifact = output / "result.json"
        artifact.write_text("{}", encoding="utf-8")
        identity = EvidenceIdentity(
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
            model_id="model",
            model_revision="revision",
        )
        write_evidence_manifest(
            output,
            EvidenceManifest(
                artifact_kind="study",
                mode="input_only",
                status="completed",
                identity=identity,
                parents={"selection": parent},
                inputs={},
                artifacts={"result": artifact},
            ),
        )

        manifest = load_evidence_manifest(output / "evidence-manifest.json")

        assert manifest["parents"]["selection"]["locator"] == "../parent.json"
        assert manifest["parents"]["selection"]["sha256"] == sha256_path(parent)
        assert manifest["artifacts"]["result"]["locator"] == "result.json"
        assert manifest["installed_package"] == identity.installed_package
        assert manifest["verification"] == identity.verification
        with unittest.TestCase().assertRaises(FileNotFoundError):
            write_evidence_manifest(
                output,
                EvidenceManifest(
                    artifact_kind="study",
                    mode="input_only",
                    status="completed",
                    identity=identity,
                    parents={},
                    inputs={"missing": root / "missing.json"},
                    artifacts={"result": artifact},
                ),
            )
        parent.write_text('{"tampered":true}', encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(ValueError, "parents artifact"):
            load_evidence_manifest(output / "evidence-manifest.json")


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_directional_claim_vocabulary_is_strict_and_content_addressed,
            test_every_directional_experiment_registers_current_claim_gates,
            test_evidence_manifest_uses_portable_locators_and_refuses_missing_inputs,
        )
    )
