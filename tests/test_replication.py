from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.contracts.claim_derivation import (
    FINAL_VERIFICATION_CHECKS,
    QWEN_MODEL,
    WIKITEXT_DATASET,
)
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256
from continuous_tokenizer.contracts.manifest import RunManifest
from continuous_tokenizer.contracts.profiles import (
    CAMPAIGN_PROFILE_NAME,
    DIAGNOSTIC_PROFILE_NAME,
)
from continuous_tokenizer.reporting.replication import aggregate_runs


def stored_claim(summary: dict[str, object], claim_id: str) -> str:
    claims = summary["claims"]
    assert isinstance(claims, list)
    records = (cast(Mapping[str, object], claim) for claim in claims if isinstance(claim, Mapping))
    return next(str(claim["verdict"]) for claim in records if claim.get("claim_id") == claim_id)


def experiment_contract(
    seed: int,
    mode: str,
    profile: str,
    model: tuple[str, str] = QWEN_MODEL,
) -> dict[str, object]:
    stages = ["output_codec"] if mode == "output_only" else ["vocabulary", "reconstruction"]
    return {
        "name": f"output-{seed}" if mode == "output_only" else "replication",
        "mode": mode,
        "device": "cpu",
        "model": {
            "model_id": model[0],
            "revision": model[1],
            "evaluation": "full",
        },
        "dataset": {
            "dataset_id": WIKITEXT_DATASET[0],
            "config": "default",
            "revision": WIKITEXT_DATASET[1],
        },
        "stages": stages,
        "seed": seed,
        "training": {"profile": profile},
        "evaluation": {},
        "gates": (
            {
                "minimum_direct_feedback_equality": 1.0,
                "maximum_invalid_events": 0,
                "minimum_valid_non_empty_termination": 1.0,
                "minimum_control_prompt_coverage": 0.25,
                "minimum_control_precision": 1.0,
                "minimum_control_recall": 1.0,
                "minimum_stop_precision": 1.0,
                "minimum_stop_recall": 1.0,
                "minimum_native_tokens_per_attempted_macro_step": 1.1,
                "minimum_rollout_event_agreement": 0.5,
                "maximum_candidate_reference_state_ratio": 0.5,
            }
            if mode == "output_only"
            else {
                "maximum_normalized_rmse": 0.01,
                "minimum_cosine_p01": 0.999,
                "minimum_cosine_p50": 0.9999,
                "minimum_native_tokens_per_continuous_token": 1.1,
                "maximum_candidate_reference_state_ratio": 0.5,
                "maximum_segmented_mean_kl": 0.1,
                "maximum_segmented_nll_delta": 0.1,
                "minimum_segmented_top1_agreement": 0.9,
                "minimum_segmented_generation_byte_similarity": 0.5,
            }
        ),
        "runtime": {},
        "evidence_scope": "final",
        "prospective_selection": None,
        "search_selections": [],
        "study_selections": [],
        "efficiency_pilot": None,
        "efficiency_pilot_sha256": None,
    }


def verification_inventory() -> dict[str, object]:
    return {
        "provided": True,
        "all_passed": True,
        "checks": {name: {"passed": True} for name in FINAL_VERIFICATION_CHECKS},
    }


def performance_measurement(
    prompt_count: int = 4,
    repetitions: int = 20,
) -> dict[str, object]:
    modes = ("native", "compatibility", "segmented")
    prompt_hashes = tuple(f"{index + 1:064x}" for index in range(prompt_count))
    pairs = []
    for pair_execution_order in range(prompt_count * repetitions):
        repetition, prompt_execution_order = divmod(
            pair_execution_order,
            prompt_count,
        )
        prompt_index = (repetition + prompt_execution_order) % prompt_count
        offset = pair_execution_order % len(modes)
        path_order = modes[offset:] + modes[:offset]
        pairs.append(
            {
                "prompt_index": prompt_index,
                "prompt_sha256": prompt_hashes[prompt_index],
                "repetition": repetition,
                "prompt_execution_order": prompt_execution_order,
                "pair_execution_order": pair_execution_order,
                "path_order": list(path_order),
                "paths": {
                    mode: {
                        "execution_order": execution_order,
                        "input_preparation_seconds": 0.1,
                        "model_prefill_seconds": (0.9 if mode == "native" else 0.8),
                        "time_to_first_logit_seconds": (1.0 if mode == "native" else 0.9),
                    }
                    for execution_order, mode in enumerate(path_order)
                },
            }
        )
    return {
        "measurement": {
            "prompt_count": prompt_count,
            "prompt_set_sha256": "f" * 64,
            "prompt_sha256": list(prompt_hashes),
            "prompt_content": "native_token_id_sequence",
            "hash_algorithm": "sha256",
            "prompt_hash_serialization": "compact_json_integer_array",
            "prompt_set_hash_serialization": "compact_json_array_of_integer_arrays",
            "warmups": 5,
            "repetitions": repetitions,
            "expected_raw_pairs": prompt_count * repetitions,
            "recorded_raw_pairs": prompt_count * repetitions,
            "warmup_scope": "each_registered_prompt_and_path",
            "prompt_order": "cyclic_rotation_by_repetition",
            "path_order": "cyclic_rotation_by_pair_execution_order",
            "pairing": "same_prompt_and_repetition",
        },
        "raw_pairs": pairs,
    }


def write_completed_run(
    root: Path,
    *,
    seed: int,
    density: float | None,
    profile: str = CAMPAIGN_PROFILE_NAME,
    model: tuple[str, str] = QWEN_MODEL,
) -> Path:
    root.mkdir()
    (root / "verification").mkdir()
    (root / "verification/verification.json").write_text(json.dumps(verification_inventory()))
    completed = density is not None
    manifest = RunManifest(
        experiment_name="replication",
        mode="input_only",
        codec_direction="input_only",
        experiment_fingerprint=f"{seed:064x}",
        replication_fingerprint="b" * 64,
        model_id=model[0],
        model_revision=model[1],
        dataset_id=WIKITEXT_DATASET[0],
        dataset_revision=WIKITEXT_DATASET[1],
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
                "locator": f"hf://{model[0]}@{model[1]}/config.json",
                "sha256": "f" * 64,
            },
            "input_embedding_tensor": {
                "locator": f"hf://{model[0]}@{model[1]}#embed.weight",
                "sha256": "a" * 64,
            },
            "tokenizer_vocabulary": {
                "locator": f"hf://{model[0]}@{model[1]}/tokenizer",
                "sha256": "b" * 64,
            },
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
                "llm_metrics": "llm-metrics.json",
                "result": "result.json",
                "experiment": "experiment.json",
                "verification": "verification",
            }
            if completed
            else {
                "result": "result.json",
                "experiment": "experiment.json",
                "verification": "verification",
            }
        ),
        artifact_hashes={},
        status="passed" if completed else "failed",
        verification=verification_inventory(),
    )
    (root / "experiment.json").write_text(
        json.dumps(experiment_contract(seed, "input_only", profile, model)),
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
                "verification": sha256_path(root / "verification"),
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
        "density": {
            "native_tokens_per_continuous_token": density,
            "round_trip": True,
        },
        "compactness": {"candidate_reference_state_ratio": 0.25},
        "embedding_fit": {
            "normalized_rmse": 0.001,
            "cosine_similarity_p01": 1.0,
            "cosine_similarity_p50": 1.0,
            "reconstruction_fraction": 1.0,
        },
        "raw_byte_fixtures": {"binary": {"round_trip": True}},
        "density_strata": {
            "vocabulary": {"windows": [{"empirical_round_trip": True}]},
            "wikitext": {"windows": [{"empirical_round_trip": True}]},
        },
    }
    (root / "tokenizer-metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    llm_metrics = {
        "options": {
            "warmups": 5,
            "repetitions": 20,
            "performance_prompts": 4,
        },
        "teacher_forced": {
            "segmented": {
                "mean_kl": 0.01,
                "teacher_nll": 1.0,
                "student_nll": 1.01,
                "top1_agreement": 1.0,
            }
        },
        "generation": {
            "samples": 1,
            "segmented_mean_byte_similarity": 1.0,
        },
        "performance": {
            **performance_measurement(),
            "native": {
                "materialized_cache_bytes": 100,
                "total_analytical_flops": 100,
                "time_to_first_logit_median_seconds": 1.0,
            },
            "segmented": {
                "materialized_cache_bytes": 75,
                "total_analytical_flops": 75,
                "time_to_first_logit_median_seconds": 0.9,
            },
        },
    }
    (root / "llm-metrics.json").write_text(
        json.dumps(llm_metrics),
        encoding="utf-8",
    )
    result = {
        "mode": "input_only",
        "evidence_scope": "final",
        "operational_status": "completed",
        "scientific_verdict": "supported",
        "experiment": experiment_contract(seed, "input_only", profile, model),
        "tokenizer": metrics,
        "gates": {
            "maximum_segmented_mean_kl": True,
            "maximum_segmented_nll_delta": True,
            "minimum_segmented_top1_agreement": True,
            "minimum_segmented_generation_byte_similarity": True,
        },
    }
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    manifest = replace(
        manifest,
        artifact_hashes={
            "tokenizer_metrics": sha256_path(root / "tokenizer-metrics.json"),
            "llm_metrics": sha256_path(root / "llm-metrics.json"),
            "result": sha256_path(root / "result.json"),
            "experiment": sha256_path(root / "experiment.json"),
            "verification": sha256_path(root / "verification"),
        },
    )
    (root / "manifest-final.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return root


def rewrite_hashed_artifact(
    root: Path,
    filename: str,
    artifact: dict[str, object],
) -> None:
    artifact_path = root / filename
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    manifest_path = root / "manifest-final.json"
    manifest = json.loads(manifest_path.read_text())
    artifact_name = {
        "tokenizer-metrics.json": "tokenizer_metrics",
        "llm-metrics.json": "llm_metrics",
        "result.json": "result",
    }[filename]
    manifest["artifact_hashes"][artifact_name] = sha256_path(artifact_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_replication_aggregation_requires_distinct_compatible_runs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = write_completed_run(root / "first", seed=17, density=1.2)
        second = write_completed_run(root / "second", seed=23, density=1.4)
        third = write_completed_run(root / "third", seed=41, density=1.3)

        summary = aggregate_runs((first, second, third))

        assert summary["mode"] == "input_only"
        assert summary["evidence_scope"] == "replication"
        assert summary["operational_status"] == "completed"
        assert summary["scientific_verdict"] == "supported"
        assert [run["seed"] for run in summary["runs"]] == [17, 23, 41]
        assert all(run["operational_status"] == "completed" for run in summary["runs"])
        assert abs(summary["metrics"]["native_tokens_per_continuous_token"]["mean"] - 1.3) < 1e-12
        assert summary["metrics"]["native_tokens_per_continuous_token"]["confidence_95_high"] is not None
        assert summary["metrics"]["native_tokens_per_continuous_token"]["confidence_method"] == "student_t"
        assert summary["metrics"]["native_tokens_per_continuous_token"]["degrees_of_freedom"] == 2
        assert summary["metrics"]["native_tokens_per_continuous_token"]["raw_values"] == [
            {"seed": 17, "value": 1.2},
            {"seed": 23, "value": 1.4},
            {"seed": 41, "value": 1.3},
        ]
        assert summary["metrics"]["native_tokens_per_continuous_token"]["confidence_eligible"]
        assert summary["category_verdicts"]["quality"] == "supported"
        assertions = unittest.TestCase()
        with assertions.assertRaisesRegex(ValueError, "distinct seeds"):
            aggregate_runs((first, first, third))
        with assertions.assertRaisesRegex(ValueError, "exactly three"):
            aggregate_runs((first, second))


def test_input_replication_headline_ignores_failed_alignment() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(
            write_completed_run(
                root / f"failed-alignment-{seed}",
                seed=seed,
                density=1.2,
            )
            for seed in (17, 23, 41)
        )
        for run in runs:
            metrics = json.loads((run / "tokenizer-metrics.json").read_text())
            metrics["embedding_fit"]["normalized_rmse"] = 0.5
            rewrite_hashed_artifact(
                run,
                "tokenizer-metrics.json",
                metrics,
            )

        summary = aggregate_runs(runs)

    assert (
        stored_claim(
            summary,
            "input.full_vocabulary_embedding_compatibility",
        )
        == "unsupported"
    )
    assert stored_claim(summary, "input.held_out_position_compression") == ("supported")
    assert summary["category_verdicts"]["quality"] == "unsupported"
    assert summary["scientific_verdict"] == "supported"


def test_input_replication_behavior_blocks_headline() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(
            write_completed_run(
                root / f"failed-behavior-{seed}",
                seed=seed,
                density=1.2,
            )
            for seed in (17, 23, 41)
        )
        for run in runs:
            metrics = json.loads((run / "llm-metrics.json").read_text())
            metrics["teacher_forced"]["segmented"]["mean_kl"] = 0.2
            rewrite_hashed_artifact(run, "llm-metrics.json", metrics)

        summary = aggregate_runs(runs)

    assert stored_claim(summary, "input.held_out_position_compression") == ("supported")
    assert (
        stored_claim(
            summary,
            "input.registered_behavioral_similarity_tolerances",
        )
        == "unsupported"
    )
    assert summary["scientific_verdict"] == "unsupported"


def test_replication_aggregation_keeps_failed_hypotheses() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_completed_run(root / f"failed-{seed}", seed=seed, density=None) for seed in (17, 23, 41))

        summary = aggregate_runs(runs)

        assert not summary["replication_complete"]
        assert summary["operational_status"] == "incomplete"
        assert summary["scientific_verdict"] == "incomplete"
        assert len(summary["failed_runs"]) == 3
        assert summary["metrics"]["native_tokens_per_continuous_token"]["count"] == 0
        assert summary["metrics"]["native_tokens_per_continuous_token"]["mean"] is None
        assert summary["metrics"]["native_tokens_per_continuous_token"]["minimum"] is None
        assert summary["metrics"]["native_tokens_per_continuous_token"]["maximum"] is None
        assert summary["metrics"]["native_tokens_per_continuous_token"]["confidence_95_low"] is None
        assert summary["metrics"]["candidate_reference_state_ratio"]["count"] == 0
        assert summary["metrics"]["candidate_reference_state_ratio"]["mean"] is None


def _rewrite_input_model_role(
    root: Path,
    model_id: str,
    revision: str,
) -> None:
    experiment_path = root / "experiment.json"
    experiment = json.loads(experiment_path.read_text())
    experiment["model"]["model_id"] = model_id
    experiment["model"]["revision"] = revision
    experiment_path.write_text(json.dumps(experiment))
    manifest_path = root / "manifest-final.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_id"] = model_id
    manifest["model_revision"] = revision
    manifest["artifact_hashes"]["experiment"] = sha256_path(experiment_path)
    manifest_path.write_text(json.dumps(manifest))


def test_complete_qwen_replication_is_confidence_interval_eligible() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(
            write_completed_run(
                root / f"qwen-{seed}",
                seed=seed,
                density=density,
            )
            for seed, density in ((17, 1.2), (23, 1.4), (41, 1.3))
        )
        for run in runs:
            _rewrite_input_model_role(
                run,
                "Qwen/Qwen3.5-0.8B",
                "2fc06364715b967f1860aea9cf38778875588b17",
            )

        summary = aggregate_runs(runs)

    density = summary["metrics"]["native_tokens_per_continuous_token"]
    assert density["confidence_eligible"]
    assert density["confidence_method"] == "student_t"
    assert density["degrees_of_freedom"] == 2
    assert density["confidence_95_low"] is not None
    assert density["confidence_95_high"] is not None


def test_replication_aggregation_rejects_diagnostic_profile() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(
            write_completed_run(
                root / f"diagnostic-{seed}",
                seed=seed,
                density=1.2,
                profile=DIAGNOSTIC_PROFILE_NAME,
            )
            for seed in (17, 23, 41)
        )

        with unittest.TestCase().assertRaisesRegex(ValueError, "campaign profile"):
            aggregate_runs(runs)


def test_replication_aggregation_rejects_mixed_modes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_completed_run(root / f"mixed-{seed}", seed=seed, density=1.2) for seed in (17, 23, 41))
        manifest_path = runs[-1] / "manifest-final.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["mode"] = "output_only"
        manifest["codec_direction"] = "output_only"
        manifest_path.write_text(json.dumps(manifest))

        with unittest.TestCase().assertRaisesRegex(ValueError, "mixed tokenizer modes"):
            aggregate_runs(runs)


def write_output_run(
    root: Path,
    seed: int,
    model: tuple[str, str] = QWEN_MODEL,
) -> Path:
    root.mkdir()
    (root / "verification").mkdir()
    (root / "verification/verification.json").write_text(json.dumps(verification_inventory()))
    manifest = RunManifest(
        experiment_name=f"output-{seed}",
        mode="output_only",
        codec_direction="output_only",
        experiment_fingerprint=f"{seed:064x}",
        replication_fingerprint="e" * 64,
        model_id=model[0],
        model_revision=model[1],
        dataset_id=WIKITEXT_DATASET[0],
        dataset_revision=WIKITEXT_DATASET[1],
        embedding_tensor="embedding.weight",
        source_dtype="torch.float32",
        seed=seed,
        stages=("output_codec",),
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
                "locator": f"hf://{model[0]}@{model[1]}/config.json",
                "sha256": "f" * 64,
            },
            "input_embedding_tensor": {
                "locator": f"hf://{model[0]}@{model[1]}#embed.weight",
                "sha256": "a" * 64,
            },
            "tokenizer_vocabulary": {
                "locator": f"hf://{model[0]}@{model[1]}/tokenizer",
                "sha256": "b" * 64,
            },
        },
        inputs={},
        codec_attention={"type": "none"},
        environment={"python": "3.14"},
        trainable_parameters=("decoder.weight",),
        frozen_backbone_fingerprint="frozen",
        native_head_used=False,
        feedback_policy="longest_native_byte_match",
        artifacts={
            "experiment": "experiment.json",
            "result": "result.json",
            "verification": "verification",
        },
        artifact_hashes={},
        status="passed",
        verification=verification_inventory(),
    )
    experiment = experiment_contract(
        seed,
        "output_only",
        CAMPAIGN_PROFILE_NAME,
        model,
    )
    (root / "experiment.json").write_text(json.dumps(experiment))
    result = {
        "mode": "output_only",
        "evidence_scope": "final",
        "operational_status": "completed",
        "experiment": {
            **experiment,
        },
        "output": {
            "direct_feedback_equality": 1.0,
            "direct_feedback_byte_equality": 1.0,
            "direct_feedback_token_equality": 1.0,
            "byte_accuracy": 1.0,
            "valid_non_empty_termination": 1.0,
            "rollout_event_agreement": 1.0,
            "rollout_byte_agreement": 1.0,
            "rollout_token_agreement": 1.0,
            "bytes_per_macro_step": 2.0,
            "native_tokens_represented": 4,
            "native_tokens_per_attempted_macro_step": 2.0,
            "candidate_reference_state_ratio": 0.25,
            "invalid_events": 0,
            "control_evidence": {
                "coverage": 1.0,
                "precision": 1.0,
                "recall": 1.0,
            },
            "stop_control": {"precision": 1.0, "recall": 1.0},
            "native_head_invocations": 0,
            "deployment": None,
        },
        "gates": {"exact": True},
        "scientific_verdict": "supported",
    }
    (root / "result.json").write_text(json.dumps(result))
    manifest = replace(
        manifest,
        artifact_hashes={
            "experiment": sha256_path(root / "experiment.json"),
            "result": sha256_path(root / "result.json"),
            "verification": sha256_path(root / "verification"),
        },
    )
    (root / "manifest-final.json").write_text(json.dumps(manifest.to_dict()))
    return root


def rewrite_output_result(root: Path, result: dict[str, object]) -> None:
    result_path = root / "result.json"
    result_path.write_text(json.dumps(result))
    manifest_path = root / "manifest-final.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"]["result"] = sha256_path(result_path)
    manifest_path.write_text(json.dumps(manifest))


def test_output_replication_aggregation_is_mode_specific() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_output_run(root / f"output-{seed}", seed) for seed in (17, 23, 41))

        summary = aggregate_runs(runs)

        assert summary["mode"] == "output_only"
        assert summary["evidence_scope"] == "replication"
        assert summary["operational_status"] == "completed"
        assert summary["scientific_verdict"] == "supported"
        assert [run["seed"] for run in summary["runs"]] == [17, 23, 41]
        assert summary["metrics"]["bytes_per_macro_step"]["mean"] == 2.0
        assert stored_claim(summary, "output.direct_feedback_exactness") == ("supported")
        assert stored_claim(summary, "output.control_exactness") == "supported"
        assert stored_claim(summary, "output.native_head_bypass") == "supported"


def test_output_replication_keeps_failed_runs_and_available_metrics() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_output_run(root / f"output-{seed}", seed) for seed in (17, 23, 41))
        manifest_path = runs[1] / "manifest-final.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest))

        summary = aggregate_runs(runs)

        assert not summary["replication_complete"]
        assert summary["scientific_verdict"] == "incomplete"
        assert [run["seed"] for run in summary["failed_runs"]] == [23]
        assert summary["metrics"]["bytes_per_macro_step"]["count"] == 2
        assert summary["metrics"]["bytes_per_macro_step"]["raw_values"] == [
            {"seed": 17, "value": 2.0},
            {"seed": 41, "value": 2.0},
        ]
        assert not summary["metrics"]["bytes_per_macro_step"]["confidence_eligible"]
        assert summary["metrics"]["bytes_per_macro_step"]["confidence_95_low"] is None
        assert stored_claim(summary, "output.native_head_bypass") == "incomplete"


def test_output_replication_preserves_structural_infeasibility() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_output_run(root / f"output-{seed}", seed) for seed in (17, 23, 41))
        for run in runs:
            result = json.loads((run / "result.json").read_text())
            result["output"]["structurally_unrepresentable"] = True
            rewrite_output_result(run, result)

        summary = aggregate_runs(runs)

        assert summary["structurally_unrepresentable"]
        assert stored_claim(summary, "output.semi_autoregressive_density") == "unsupported"


def test_output_replication_rejects_non_full_quality_runs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_output_run(root / f"output-{seed}", seed) for seed in (17, 23, 41))
        result_path = runs[0] / "result.json"
        result = json.loads(result_path.read_text())
        result["experiment"]["model"]["evaluation"] = "partial"
        rewrite_output_result(runs[0], result)

        with unittest.TestCase().assertRaisesRegex(ValueError, "full quality-evaluation"):
            aggregate_runs(runs)


def test_output_replication_requires_registered_final_seeds() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_output_run(root / f"output-{seed}", seed) for seed in (1, 2, 3))

        summary = aggregate_runs(runs)

        assert not summary["replication_complete"]
        assert summary["scientific_verdict"] == "incomplete"
        assert summary["category_verdicts"]["quality"] == "incomplete"


def test_output_replication_rejects_search_evidence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = tuple(write_output_run(root / f"output-{seed}", seed) for seed in (17, 23, 41))
        result_path = runs[0] / "result.json"
        result = json.loads(result_path.read_text())
        result["evidence_scope"] = "search"
        rewrite_output_result(runs[0], result)

        with unittest.TestCase().assertRaisesRegex(ValueError, "final scope"):
            aggregate_runs(runs)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_replication_aggregation_requires_distinct_compatible_runs,
            test_input_replication_headline_ignores_failed_alignment,
            test_input_replication_behavior_blocks_headline,
            test_replication_aggregation_keeps_failed_hypotheses,
            test_complete_qwen_replication_is_confidence_interval_eligible,
            test_replication_aggregation_rejects_diagnostic_profile,
            test_replication_aggregation_rejects_mixed_modes,
            test_output_replication_aggregation_is_mode_specific,
            test_output_replication_keeps_failed_runs_and_available_metrics,
            test_output_replication_preserves_structural_infeasibility,
            test_output_replication_rejects_non_full_quality_runs,
            test_output_replication_requires_registered_final_seeds,
            test_output_replication_rejects_search_evidence,
        )
    )
