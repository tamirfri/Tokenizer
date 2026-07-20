from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.artifacts.software_validation import (
    load_software_validation_inputs,
)
from continuous_tokenizer.contracts.claim_derivation import (
    FINAL_VERIFICATION_CHECKS,
)
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256
from continuous_tokenizer.contracts.manifest import RunManifest
from continuous_tokenizer.contracts.statements import (
    STATEMENT_REGISTRY,
    SoftwareValidationInputs,
    SourceBinding,
    SyntheticCampaignEvidence,
    SyntheticMode,
    VerificationEvidence,
    statement_trace,
)


def _source(character: str = "a") -> SourceBinding:
    return SourceBinding(character * 64, "b" * 64)


def _inputs(
    *,
    output_source: SourceBinding | None = None,
    output_passed: bool = True,
) -> SoftwareValidationInputs:
    source = _source()
    return SoftwareValidationInputs(
        verification=VerificationEvidence(source, True, "verification.json"),
        synthetic_campaigns=(
            SyntheticCampaignEvidence(
                "input_only",
                source,
                "completed",
                True,
                "input/result.json",
            ),
            SyntheticCampaignEvidence(
                "output_only",
                source if output_source is None else output_source,
                "completed",
                output_passed,
                "output/result.json",
            ),
        ),
    )


def _verification(root: Path) -> Path:
    directory = root / "verification"
    logs = directory / "logs"
    logs.mkdir(parents=True)
    checks = {}
    for name in sorted(FINAL_VERIFICATION_CHECKS):
        log = f"{name}\n".encode()
        path = logs / f"{name}.log"
        path.write_bytes(log)
        checks[name] = {
            "command": ["test", name],
            "passed": True,
            "return_code": 0,
            "seconds": 0.1,
            "log": f"logs/{name}.log",
            "log_sha256": hashlib.sha256(log).hexdigest(),
        }
    path = directory / "verification.json"
    path.write_text(
        json.dumps(
            {
                "kind": "complete_verification",
                "source_commit": "commit",
                "source_dirty": True,
                "source_state_sha256": "c" * 64,
                "dependency_lock_sha256": "d" * 64,
                "all_passed": True,
                "checks": checks,
            }
        )
    )
    return path


def _synthetic(root: Path, mode: SyntheticMode) -> Path:
    directory = root / mode
    directory.mkdir()
    experiment = directory / "experiment.json"
    experiment.write_text("{}")
    result = directory / "result.json"
    if mode == "input_only":
        value = {
            "mode": mode,
            "evidence_scope": "synthetic",
            "operational_status": "completed",
            "tokenizer": {"acceptance": {"overall": True}},
            "vocabulary": {"atomic_bytes": 256},
            "gates": {"tokenizer": True},
        }
        native_head_used = True
        stages = ("vocabulary", "reconstruction")
    else:
        value = {
            "mode": mode,
            "evidence_scope": "synthetic",
            "operational_status": "completed",
            "output": {
                "native_head_invocations": 0,
            },
            "gates": {
                "direct_feedback": True,
                "invalid_events": True,
                "valid_non_empty_termination": True,
            },
        }
        native_head_used = False
        stages = ("output_codec",)
    result.write_text(json.dumps(value))
    manifest = RunManifest(
        experiment_name=f"synthetic-{mode}",
        mode=mode,
        codec_direction=mode,
        experiment_fingerprint="a" * 64,
        replication_fingerprint="b" * 64,
        model_id="continuous-tokenizer/synthetic-model",
        model_revision="synthetic",
        dataset_id="continuous-tokenizer/synthetic-bytes",
        dataset_revision="synthetic",
        embedding_tensor=None,
        source_dtype=None,
        seed=17,
        stages=stages,
        source_commit="commit",
        source_dirty=True,
        source_state_sha256="c" * 64,
        dependency_lock_sha256="d" * 64,
        installed_package={
            "name": "continuous-byte-tokenizer",
            "version": "0.1.0",
            "content_sha256": "e" * 64,
        },
        claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
        source_assets={},
        inputs={},
        codec_attention=({"query_heads": 4, "key_value_heads": 2, "enable_gqa": True} if mode == "input_only" else {"type": "none"}),
        environment={"python": "3.14"},
        trainable_parameters=(),
        frozen_backbone_fingerprint="f" * 64,
        native_head_used=native_head_used,
        feedback_policy=("native_output_tokens" if mode == "input_only" else "longest_native_byte_match"),
        artifacts={
            "experiment": "experiment.json",
            "result": "result.json",
        },
        artifact_hashes={
            "experiment": sha256_path(experiment),
            "result": sha256_path(result),
        },
        status="passed",
        verification={"provided": False},
    )
    (directory / "manifest-final.json").write_text(json.dumps(manifest.to_dict()))
    return directory


def test_registry_reserves_proved_for_three_protocol_statements() -> None:
    traces = statement_trace()
    proved = tuple(trace.definition.statement_id for trace in traces if trace.status == "proved")

    assert proved == (
        "protocol.accepted_span_exactness",
        "protocol.atomic_fallback",
        "protocol.exhaustive_local_maximality",
    )
    assert all(trace.status == "not_validated" for trace in traces if trace.definition.kind == "software")


def test_registry_supplies_complete_backward_trace_metadata() -> None:
    assert len({definition.statement_id for definition in STATEMENT_REGISTRY}) == len(STATEMENT_REGISTRY)
    for definition in STATEMENT_REGISTRY:
        assert definition.paper_label
        assert definition.summary
        assert definition.implementation_symbols
        assert definition.validating_test_ids
        assert definition.evidence_requirement
        assert definition.scope in {"input_only", "output_only", "shared"}
        assert all(symbol.startswith("continuous_tokenizer.") for symbol in definition.implementation_symbols)
        assert all(test_id.startswith("test_") for test_id in definition.validating_test_ids)


def test_software_validation_requires_matching_source_bound_evidence() -> None:
    validated = {trace.definition.statement_id: trace.status for trace in statement_trace(_inputs())}
    assert all(status == "validated" for statement_id, status in validated.items() if statement_id.startswith("software."))

    mismatched = {trace.definition.statement_id: trace.status for trace in statement_trace(_inputs(output_source=_source("c")))}
    assert mismatched["software.cache_semantics"] == "validated"
    assert mismatched["software.input_path"] == "validated"
    assert mismatched["software.output_path"] == "not_validated"
    assert mismatched["software.backbone_immutability"] == "not_validated"


def test_software_validation_rejects_failed_or_incomplete_inputs() -> None:
    failed_output = {trace.definition.statement_id: trace.status for trace in statement_trace(_inputs(output_passed=False))}
    assert failed_output["software.output_path"] == "not_validated"
    assert failed_output["software.backbone_immutability"] == "not_validated"

    source = _source()
    input_only = SoftwareValidationInputs(
        verification=VerificationEvidence(source, True, "verification.json"),
        synthetic_campaigns=(
            SyntheticCampaignEvidence(
                "input_only",
                source,
                "completed",
                True,
                "input/result.json",
            ),
        ),
    )
    partial = {trace.definition.statement_id: trace.status for trace in statement_trace(input_only)}
    assert partial["software.cache_semantics"] == "validated"
    assert partial["software.output_path"] == "not_validated"
    assert partial["software.backbone_immutability"] == "not_validated"


def test_shared_loader_derives_validation_only_from_semantic_artifacts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        verification = _verification(root)
        input_synthetic = _synthetic(root, "input_only")
        output_synthetic = _synthetic(root, "output_only")

        inputs = load_software_validation_inputs(
            verification,
            input_synthetic,
            output_synthetic,
        )

        assert all(trace.status == "validated" for trace in statement_trace(inputs) if trace.definition.kind == "software")
        result_path = output_synthetic / "result.json"
        result = json.loads(result_path.read_text())
        result["gates"]["direct_feedback"] = False
        result_path.write_text(json.dumps(result))
        manifest_path = output_synthetic / "manifest-final.json"
        manifest = RunManifest.load(manifest_path)
        manifest = replace(
            manifest,
            artifact_hashes={
                **manifest.artifact_hashes,
                "result": sha256_path(result_path),
            },
        )
        manifest_path.write_text(json.dumps(manifest.to_dict()))
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "registered software checks",
        ):
            load_software_validation_inputs(
                verification,
                input_synthetic,
                output_synthetic,
            )


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_registry_reserves_proved_for_three_protocol_statements,
            test_registry_supplies_complete_backward_trace_metadata,
            test_software_validation_requires_matching_source_bound_evidence,
            test_software_validation_rejects_failed_or_incomplete_inputs,
            test_shared_loader_derives_validation_only_from_semantic_artifacts,
        )
    )
