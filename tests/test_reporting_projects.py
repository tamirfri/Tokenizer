from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    load_evidence_manifest,
    verify_artifact,
    write_evidence_manifest,
)
from continuous_tokenizer.commands.evidence import aggregate
from continuous_tokenizer.contracts.claim_derivation import GEMMA_MODEL, QWEN_MODEL
from continuous_tokenizer.contracts.input_study import (
    INPUT_ALIGNMENT_FEASIBILITY_STAGES,
    INPUT_ALIGNMENT_TRAINING_SEEDS,
    failed_alignment_gates,
)
from continuous_tokenizer.reporting.discovery import discover_project_artifacts
from continuous_tokenizer.reporting.project import assemble_project_artifact
from continuous_tokenizer.reporting.replication_markdown import replication_report
from tests.test_replication import (
    rewrite_hashed_artifact,
    write_completed_run,
    write_output_run,
)
from tests.test_statement_contract import _synthetic, _verification


def _replication(
    root: Path,
    *,
    model: tuple[str, str],
    mode: str = "input_only",
    embedding_alignment_passes: bool = True,
    behavior_passes: bool = True,
) -> Path:
    runs = tuple(
        (
            write_output_run(root / f"{model[0].split('/')[-1]}-{seed}", seed, model)
            if mode == "output_only"
            else write_completed_run(
                root / f"{model[0].split('/')[-1]}-{seed}",
                seed=seed,
                density=1.25,
                model=model,
            )
        )
        for seed in (17, 23, 41)
    )
    if mode == "input_only":
        for run in runs:
            if not embedding_alignment_passes:
                tokenizer = json.loads(
                    (run / "tokenizer-metrics.json").read_text(),
                )
                tokenizer["embedding_fit"]["normalized_rmse"] = 0.5
                rewrite_hashed_artifact(
                    run,
                    "tokenizer-metrics.json",
                    tokenizer,
                )
            if not behavior_passes:
                llm = json.loads((run / "llm-metrics.json").read_text())
                llm["teacher_forced"]["segmented"]["mean_kl"] = 0.2
                rewrite_hashed_artifact(run, "llm-metrics.json", llm)
    output = root / f"{model[0].split('/')[-1]}-{mode}-replication"
    aggregate(Namespace(runs=runs, output_dir=output))
    return output


def _claim(project: dict[str, object], claim_id: str) -> str:
    claims = project["claims"]
    assert isinstance(claims, list)
    records = (cast(Mapping[str, object], record) for record in claims if isinstance(record, Mapping))
    return next(str(record["verdict"]) for record in records if record.get("claim_id") == claim_id)


def _alignment_study(
    root: Path,
    *,
    model: tuple[str, str],
    replication: Path,
    passing: bool,
    label: str,
) -> Path:
    directory = root / f"{model[0].split('/')[-1]}-{label}-alignment-study"
    directory.mkdir()
    gates = {
        "maximum_normalized_rmse": 0.01,
        "minimum_cosine_p01": 0.999,
        "minimum_cosine_p50": 0.9999,
    }
    stages = []
    for size in INPUT_ALIGNMENT_FEASIBILITY_STAGES:
        subset_sha256 = hashlib.sha256(
            f"{model[0]}:{size}:17".encode(),
        ).hexdigest()
        seed_results = []
        for seed in INPUT_ALIGNMENT_TRAINING_SEEDS:
            seed_passed = passing or size != INPUT_ALIGNMENT_FEASIBILITY_STAGES[-1]
            metrics = {
                "normalized_rmse": 0.005 if seed_passed else 0.02,
                "cosine_similarity_p01": 0.9995,
                "cosine_similarity_p50": 0.99995,
            }
            failed_gates = failed_alignment_gates(metrics, gates)
            seed_results.append(
                {
                    "vocabulary_subset_size": size,
                    "training_seed": seed,
                    "status": "passed" if seed_passed else "failed_gate",
                    "reason": "test",
                    "failed_gates": failed_gates,
                    "subset_sha256": subset_sha256,
                    "alignment": {"embedding_metrics": metrics},
                    "artifact_hashes": {"alignment_result": "a" * 64},
                },
            )
        failed_seeds = [
            {
                "training_seed": seed["training_seed"],
                "failed_gates": seed["failed_gates"],
            }
            for seed in seed_results
            if seed["status"] == "failed_gate"
        ]
        stages.append(
            {
                "vocabulary_subset_size": size,
                "status": "failed_gate" if failed_seeds else "passed",
                "reason": "test",
                "failed_gates": failed_seeds,
                "vocabulary_subset": {
                    "requested_rows": size,
                    "token_ids": [256],
                    "sha256": subset_sha256,
                    "algorithm": "test",
                },
                "subset_sha256": subset_sha256,
                "training_seeds": list(INPUT_ALIGNMENT_TRAINING_SEEDS),
                "seed_results": seed_results,
            },
        )
    result = {
        "artifact_kind": "input_alignment_feasibility_study",
        "mode": "input_only",
        "operational_status": "completed",
        "evidence_scope": "selection",
        "model_id": model[0],
        "model_revision": model[1],
        "model": {"id": model[0], "revision": model[1]},
        "study": {
            "prospective": True,
            "training_seeds": list(INPUT_ALIGNMENT_TRAINING_SEEDS),
            "subset_seed": 17,
        },
        "training_seeds": list(INPUT_ALIGNMENT_TRAINING_SEEDS),
        "subset_seed": 17,
        "stages": stages,
        "feasibility_passed": passing,
        "acceptance_gates": gates,
        "prospective": True,
        "final_evidence": False,
        "full_model_evaluation_performed": False,
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    replication_manifest = load_evidence_manifest(
        replication / "evidence-manifest.json",
    )
    source = cast(Mapping[str, object], replication_manifest["source"])
    write_evidence_manifest(
        directory,
        EvidenceManifest(
            artifact_kind="study",
            mode="input_only",
            status="completed",
            identity=EvidenceIdentity(
                source_commit=str(source["commit"]),
                source_dirty=bool(source["dirty"]),
                source_state_sha256=str(source["state_sha256"]),
                dependency_lock_sha256=str(
                    replication_manifest["dependency_lock_sha256"],
                ),
                installed_package=cast(
                    Mapping[str, str],
                    replication_manifest["installed_package"],
                ),
                claim_vocabulary_sha256=str(
                    replication_manifest["claim_vocabulary_sha256"],
                ),
                source_assets={},
                verification={"provided": False},
                model_id=model[0],
                model_revision=model[1],
            ),
            parents={},
            inputs={},
            artifacts={"result": result_path},
        ),
    )
    return directory


def test_replication_report_shows_primary_model_seed_intervals() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        replication = _replication(root, model=GEMMA_MODEL)
        summary = json.loads(
            (replication / "replication.json").read_text(),
        )

    report = replication_report(summary)

    assert f"- Model: `{GEMMA_MODEL[0]}`" in report
    assert "| 17 | completed | supported |" in report
    assert "95% confidence interval" in report
    assert "primary-model seeds 17, 23, and 41" in report


def test_input_project_requires_equal_primary_three_seed_replications() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        qwen = _replication(root, model=QWEN_MODEL)
        gemma = _replication(root, model=GEMMA_MODEL)
        output = root / "project"

        project = assemble_project_artifact((qwen, gemma), output)

        assert project["mode"] == "input_only"
        assert project["scientific_verdict"] == "supported"
        assert project["cross_model_verdict"] == "supported"
        assert [model["model"]["id"] for model in project["models"]] == [
            QWEN_MODEL[0],
            GEMMA_MODEL[0],
        ]
        assert _claim(project, "input.cross_model_confirmation") == "supported"
        assert _claim(project, "input.fixed_subset_alignment_feasibility") == ("incomplete")
        assert _claim(project, "input.full_vocabulary_embedding_compatibility") == ("supported")
        assert _claim(project, "input.tokenizer_latency_improvement") == ("incomplete")
        assert _claim(project, "input.prompt_cache_reduction") == "incomplete"
        assert _claim(project, "input.end_to_end_latency_improvement") == ("incomplete")
        assert _claim(project, "input.prefill_compute_reduction") == "incomplete"
        statement_statuses = {trace["statement_id"]: trace["verdict"] for trace in project["statement_traces"]}
        assert statement_statuses["protocol.accepted_span_exactness"] == "proved"
        assert statement_statuses["software.input_path"] == "not_validated"
        claim_trace = next(trace for trace in project["claim_traces"] if trace["claim_id"] == "input.full_vocabulary_embedding_compatibility")
        assert [parent["model"]["id"] for parent in claim_trace["parent_model_evidence"]] == [QWEN_MODEL[0], GEMMA_MODEL[0]]
        assert all(parent["seeds"] == [17, 23, 41] for parent in claim_trace["parent_model_evidence"])
        report = (output / "project-report.md").read_text()
        assert report.count("Equal primary final replication") == 2
        assert "Cross-tokenizer" not in report
        assert "## Protocol Proofs" in report
        assert "## Software Validation" in report
        assert "## Per-model Empirical Support" in report
        assert "## Cross-model Confirmation" in report
        assert "Deployment performance is secondary and final-only." in report
        assert "Runtime speed does not support the future joint tensor-state" in report
        assert verify_artifact(output)["valid"]
        discovered = discover_project_artifacts(root)
        assert discovered[0].directory == output
        assert discovered[0].model == f"{QWEN_MODEL[0]} + {GEMMA_MODEL[0]}"

        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "exactly two primary-model",
        ):
            assemble_project_artifact((qwen,), root / "missing-model")
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "duplicate primary models",
        ):
            assemble_project_artifact((qwen, qwen), root / "duplicate-model")


def test_output_project_combines_independent_primary_replications() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        qwen = _replication(root, model=QWEN_MODEL, mode="output_only")
        gemma = _replication(root, model=GEMMA_MODEL, mode="output_only")

        project = assemble_project_artifact(
            (qwen, gemma),
            root / "output-project",
        )

        assert project["mode"] == "output_only"
        assert project["cross_model_verdict"] == "supported"
        assert _claim(project, "output.cross_model_confirmation") == "supported"
        assert _claim(project, "output.semi_autoregressive_density") == "supported"
        assert verify_artifact(root / "output-project")["valid"]


def test_input_project_headline_ignores_failed_alignment() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        project = assemble_project_artifact(
            (
                _replication(
                    root,
                    model=QWEN_MODEL,
                    embedding_alignment_passes=False,
                ),
                _replication(
                    root,
                    model=GEMMA_MODEL,
                    embedding_alignment_passes=False,
                ),
            ),
            root / "project",
        )

        assert project["category_verdicts"]["quality"] == "unsupported"
        assert (
            _claim(
                project,
                "input.full_vocabulary_embedding_compatibility",
            )
            == "unsupported"
        )
        assert _claim(project, "input.held_out_position_compression") == ("supported")
        assert project["scientific_verdict"] == "supported"
        assert project["cross_model_verdict"] == "supported"
        report = (root / "project" / "project-report.md").read_text()
        assert ("Full-vocabulary embedding alignment is independent: **UNSUPPORTED**") in report
        assert "usable exact held-out input position compression: **SUPPORTED**" in report
        assert verify_artifact(root / "project")["valid"]


def test_input_project_behavior_blocks_headline() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        project = assemble_project_artifact(
            (
                _replication(
                    root,
                    model=QWEN_MODEL,
                    behavior_passes=False,
                ),
                _replication(root, model=GEMMA_MODEL),
            ),
            root / "project",
        )

        assert _claim(project, "input.held_out_position_compression") == ("supported")
        assert (
            _claim(
                project,
                "input.registered_behavioral_similarity_tolerances",
            )
            == "unsupported"
        )
        assert project["scientific_verdict"] == "unsupported"
        assert project["cross_model_verdict"] == "unsupported"
        report = (root / "project" / "project-report.md").read_text()
        assert "usable exact held-out input position compression: **UNSUPPORTED**" in report
        assert "registered behavioral similarity **UNSUPPORTED**" in report
        assert "Exact held-out bytes" in report
        assert verify_artifact(root / "project")["valid"]


def test_input_project_alignment_claim_requires_every_model_stage_and_seed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        qwen = _replication(root, model=QWEN_MODEL)
        gemma = _replication(root, model=GEMMA_MODEL)
        qwen_study = _alignment_study(
            root,
            model=QWEN_MODEL,
            replication=qwen,
            passing=True,
            label="passing",
        )
        gemma_study = _alignment_study(
            root,
            model=GEMMA_MODEL,
            replication=gemma,
            passing=True,
            label="passing",
        )

        supported = assemble_project_artifact(
            (qwen, gemma),
            root / "supported-project",
            alignment_studies=(qwen_study, gemma_study),
        )
        assert _claim(supported, "input.fixed_subset_alignment_feasibility") == "supported"
        assert verify_artifact(root / "supported-project")["valid"]

        failed_gemma_study = _alignment_study(
            root,
            model=GEMMA_MODEL,
            replication=gemma,
            passing=False,
            label="failed",
        )
        unsupported = assemble_project_artifact(
            (qwen, gemma),
            root / "unsupported-project",
            alignment_studies=(qwen_study, failed_gemma_study),
        )
        assert _claim(unsupported, "input.fixed_subset_alignment_feasibility") == "unsupported"
        assert verify_artifact(root / "unsupported-project")["valid"]


def test_project_trace_mutation_fails_semantic_verification_after_rehash() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "project"
        assemble_project_artifact(
            (
                _replication(root, model=QWEN_MODEL),
                _replication(root, model=GEMMA_MODEL),
            ),
            output,
        )
        project_path = output / "project.json"
        project = json.loads(project_path.read_text())
        project["statement_traces"][0]["verdict"] = "not_validated"
        project_path.write_text(json.dumps(project))
        manifest_path = output / "evidence-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["project"]["sha256"] = hashlib.sha256(project_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

        verified = verify_artifact(output)

        assert not verified["valid"]
        assert any("statement traces differ from sealed validation inputs" in error for error in verified["errors"])


def test_project_seals_and_rederives_complete_software_validation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "project"
        project = assemble_project_artifact(
            (
                _replication(root, model=QWEN_MODEL),
                _replication(root, model=GEMMA_MODEL),
            ),
            output,
            software_validation_paths=(
                _verification(root),
                _synthetic(root, "input_only"),
                _synthetic(root, "output_only"),
            ),
        )

        assert all(trace["verdict"] == "validated" for trace in project["statement_traces"] if trace["kind"] == "software")
        assert verify_artifact(output)["valid"]
        manifest = load_evidence_manifest(output / "evidence-manifest.json")
        assert set(manifest["inputs"]) == {
            "software_verification",
            "software_input_synthetic",
            "software_output_synthetic",
        }


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_replication_report_shows_primary_model_seed_intervals,
            test_input_project_requires_equal_primary_three_seed_replications,
            test_output_project_combines_independent_primary_replications,
            test_input_project_headline_ignores_failed_alignment,
            test_input_project_behavior_blocks_headline,
            test_input_project_alignment_claim_requires_every_model_stage_and_seed,
            test_project_trace_mutation_fails_semantic_verification_after_rehash,
            test_project_seals_and_rederives_complete_software_validation,
        )
    )
