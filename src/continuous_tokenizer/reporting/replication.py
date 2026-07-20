from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final, Literal, Self, cast, final

from continuous_tokenizer.artifacts.evidence import student_t_95_interval
from continuous_tokenizer.artifacts.manifest import (
    load_artifact,
    load_verified_run_manifest,
)
from continuous_tokenizer.artifacts.store import load_json_object
from continuous_tokenizer.contracts.claim_derivation import (
    FINAL_VERIFICATION_CHECKS,
    GEMMA_MODEL,
    QWEN_MODEL,
    WIKITEXT_DATASET,
    derive_input_claim_verdicts,
    derive_input_headline_verdict,
    derive_output_claim_verdicts,
)
from continuous_tokenizer.contracts.claims import (
    ClaimVerdict,
    claim_category_verdicts,
    claim_records,
)
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.manifest import RunManifest
from continuous_tokenizer.contracts.profiles import CAMPAIGN_PROFILE_NAME
from continuous_tokenizer.reporting.discovery import artifact_directory_profile

_CAMPAIGN_SEEDS: Final = frozenset({17, 23, 41})
_PRIMARY_MODELS: Final = frozenset({QWEN_MODEL, GEMMA_MODEL})
_OUTPUT_REPLICATION_METRICS: Final = (
    "exact_event_agreement",
    "byte_accuracy",
    "valid_non_empty_termination",
    "direct_feedback_byte_equality",
    "direct_feedback_token_equality",
    "direct_feedback_equality",
    "rollout_event_agreement",
    "rollout_byte_agreement",
    "rollout_token_agreement",
    "exact_full_sequence_rate",
    "first_divergence_position",
    "matched_prefix_density",
    "bytes_per_macro_step",
    "native_tokens_represented",
    "native_tokens_per_attempted_macro_step",
    "attempted_macro_steps",
    "invalid_events",
    "invalid_fraction",
    "truncated_events",
    "candidate_reference_state_ratio",
    "native_head_invocations",
)


def aggregate_numeric(
    values: Sequence[tuple[int, float]],
    *,
    confidence_eligible: bool = False,
) -> dict[str, Any]:
    raw_values = [{"seed": seed, "value": value} for seed, value in sorted(values)]
    numeric = [value for _, value in values]
    distinct_seeds = {seed for seed, _ in values}
    confidence_eligible = confidence_eligible and distinct_seeds == _CAMPAIGN_SEEDS and len(numeric) == len(_CAMPAIGN_SEEDS)
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "maximum": None,
            "raw_values": raw_values,
            "confidence_eligible": confidence_eligible,
            "confidence_level": None,
            "confidence_method": None,
            "degrees_of_freedom": None,
            "confidence_95_low": None,
            "confidence_95_high": None,
        }
    mean = sum(numeric) / len(numeric)
    interval = student_t_95_interval(numeric) if confidence_eligible else None
    return {
        "count": len(numeric),
        "mean": mean,
        "minimum": min(numeric),
        "maximum": max(numeric),
        "raw_values": raw_values,
        "confidence_eligible": confidence_eligible,
        "confidence_level": None if interval is None else 0.95,
        "confidence_method": None if interval is None else "student_t",
        "degrees_of_freedom": None if interval is None else 2,
        "confidence_95_low": None if interval is None else interval[0],
        "confidence_95_high": None if interval is None else interval[1],
    }


@final
@dataclass(frozen=True, slots=True)
class CompletedRun:
    directory: str
    profile: str | None
    seed: int
    status: Literal["passed", "failed"]
    evidence_scope: str
    operational_status: str
    scientific_verdict: str
    tokenizer_passed: bool
    embedding_fit_passed: bool
    density_passed: bool
    compactness_passed: bool
    behavior_gates_passed: bool | None
    native_tokens_per_continuous_token: float | None
    candidate_reference_state_ratio: float | None
    normalized_rmse: float
    prompt_cache_ratio: float | None
    prefill_flops_ratio: float | None
    time_to_first_logit_ratio: float | None
    segmented_mean_kl: float | None
    claim_metrics: dict[str, object]
    claim_thresholds: dict[str, object]

    @classmethod
    def load(cls, directory: Path) -> Self:
        manifest = load_verified_run_manifest(directory / "manifest-final.json")
        if manifest.status != "passed":
            raise ValueError(f"run did not complete operationally: {directory}")
        result = load_artifact(directory / "result.json")
        tokenizer_path = directory / "tokenizer-metrics.json"
        tokenizer = load_artifact(tokenizer_path) if tokenizer_path.is_file() else None
        llm_path = directory / "llm-metrics.json"
        llm = load_artifact(llm_path) if llm_path.is_file() else None
        segmented_mean_kl = None if llm is None else float(llm["teacher_forced"]["segmented"]["mean_kl"])
        result_gates = result.get("gates")
        behavior_gate_names = {
            "maximum_segmented_mean_kl",
            "maximum_segmented_nll_delta",
            "minimum_segmented_top1_agreement",
            "minimum_segmented_generation_byte_similarity",
        }
        behavior_gates_passed = (
            None
            if not isinstance(result_gates, Mapping) or not behavior_gate_names.issubset(result_gates)
            else all(bool(result_gates[name]) for name in behavior_gate_names)
        )
        if llm is None:
            prompt_cache_ratio = None
            prefill_flops_ratio = None
            time_to_first_logit_ratio = None
            latency: dict[str, object] | None = None
        else:
            native = llm["performance"]["native"]
            segmented_performance = llm["performance"]["segmented"]
            prompt_cache_ratio = _ratio(
                segmented_performance["materialized_cache_bytes"],
                native["materialized_cache_bytes"],
            )
            prefill_flops_ratio = _ratio(
                segmented_performance["total_analytical_flops"],
                native["total_analytical_flops"],
            )
            time_to_first_logit_ratio = _ratio(
                segmented_performance["time_to_first_logit_median_seconds"],
                native["time_to_first_logit_median_seconds"],
            )
            latency = _latency_evidence(llm)
        if tokenizer is None:
            training = result.get("training")
            if not isinstance(training, Mapping):
                raise ValueError(f"completed run has no training evidence: {directory}")
            embedding_fit = training["compatibility_embedding_metrics"]
            tokenizer_passed = False
            embedding_fit_passed = False
            density_passed = False
            compactness_passed = False
            native_tokens_per_continuous_token = None
            candidate_reference_state_ratio = float(
                training["candidate_reference_state_ratio"],
            )
        else:
            embedding_fit = tokenizer["embedding_fit"]
            tokenizer_passed = bool(tokenizer["acceptance"]["overall"])
            embedding_fit_passed = bool(tokenizer["acceptance"]["embedding_fit"])
            density_passed = bool(tokenizer["acceptance"]["density"])
            compactness_passed = bool(tokenizer["acceptance"]["compactness"])
            native_tokens_per_continuous_token = float(tokenizer["density"]["native_tokens_per_continuous_token"])
            candidate_reference_state_ratio = float(
                tokenizer["compactness"]["candidate_reference_state_ratio"],
            )
        embedding_fit = cast(Mapping[str, Any], embedding_fit)
        claim_metrics, claim_thresholds = _input_claim_evidence(
            tokenizer,
            llm,
            result,
            {
                "evidence_scope": result.get("evidence_scope"),
                "performance_context": (None if llm is None else llm.get("performance_claim_context")),
                "tokenizer_performance": tokenizer,
                "native_tokens_per_continuous_token": native_tokens_per_continuous_token,
                "candidate_reference_state_ratio": candidate_reference_state_ratio,
                "prompt_cache_ratio": prompt_cache_ratio,
                "native_prompt_cache_bytes": (None if llm is None else native["materialized_cache_bytes"]),
                "segmented_prompt_cache_bytes": (None if llm is None else segmented_performance["materialized_cache_bytes"]),
                "prefill_flops_ratio": prefill_flops_ratio,
                "native_prefill_flops": (None if llm is None else native["total_analytical_flops"]),
                "segmented_prefill_flops": (None if llm is None else segmented_performance["total_analytical_flops"]),
                "latency": latency,
            },
        )
        return cls(
            directory=str(directory),
            profile=artifact_directory_profile(directory),
            seed=manifest.seed,
            status=manifest.status,
            evidence_scope=str(result["evidence_scope"]),
            operational_status=str(result["operational_status"]),
            scientific_verdict=str(result["scientific_verdict"]),
            tokenizer_passed=tokenizer_passed,
            embedding_fit_passed=embedding_fit_passed,
            density_passed=density_passed,
            compactness_passed=compactness_passed,
            behavior_gates_passed=behavior_gates_passed,
            native_tokens_per_continuous_token=native_tokens_per_continuous_token,
            candidate_reference_state_ratio=candidate_reference_state_ratio,
            normalized_rmse=float(embedding_fit["normalized_rmse"]),
            prompt_cache_ratio=prompt_cache_ratio,
            prefill_flops_ratio=prefill_flops_ratio,
            time_to_first_logit_ratio=time_to_first_logit_ratio,
            segmented_mean_kl=segmented_mean_kl,
            claim_metrics=claim_metrics,
            claim_thresholds=claim_thresholds,
        )


def _latency_evidence(
    llm: Mapping[str, Any],
) -> dict[str, object] | None:
    options = cast(Mapping[str, Any], llm.get("options", {}))
    performance = llm.get("performance")
    if not isinstance(performance, Mapping):
        return None
    measurement = performance.get("measurement")
    raw_pairs = performance.get("raw_pairs")
    if not isinstance(measurement, Mapping) or not isinstance(raw_pairs, list):
        return None
    return {
        "registered_prompt_count": options.get("performance_prompts"),
        "registered_warmups": options.get("warmups"),
        "registered_repetitions": options.get("repetitions"),
        **dict(measurement),
        "raw_pairs": raw_pairs,
    }


def _input_claim_evidence(
    tokenizer: Mapping[str, Any] | None,
    llm: Mapping[str, Any] | None,
    result: Mapping[str, Any],
    measurements: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    embedding = (
        cast(Mapping[str, Any], tokenizer.get("embedding_fit", {}))
        if tokenizer is not None
        else cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], result["training"])["compatibility_embedding_metrics"],
        )
    )
    density = cast(Mapping[str, Any], tokenizer.get("density", {})) if tokenizer is not None else {}
    strata = cast(Mapping[str, Any], tokenizer.get("density_strata", {})) if tokenizer is not None else {}
    fixtures = cast(Mapping[str, Any], tokenizer.get("raw_byte_fixtures", {})) if tokenizer is not None else {}
    segmented = cast(Mapping[str, Any], llm["teacher_forced"]["segmented"]) if llm is not None else {}
    generation = cast(Mapping[str, Any], llm.get("generation", {})) if llm is not None else {}
    exact_density = (
        tokenizer is not None
        and density.get("round_trip") is True
        and all(isinstance(value, Mapping) and value.get("round_trip") is True for value in fixtures.values())
        and all(
            isinstance(section, Mapping)
            and all(
                isinstance(window, Mapping) and window.get("empirical_round_trip") is True
                for window in cast(
                    Sequence[Any],
                    section.get("windows", ()),
                )
            )
            for name, section in strata.items()
            if name in {"vocabulary", "wikitext"}
        )
    )
    experiment = cast(Mapping[str, Any], result.get("experiment", {}))
    return (
        {
            "normalized_rmse": embedding.get("normalized_rmse"),
            "cosine_similarity_p01": embedding.get("cosine_similarity_p01"),
            "cosine_similarity_p50": embedding.get("cosine_similarity_p50"),
            "reconstruction_fraction": embedding.get("reconstruction_fraction"),
            "density_exact": exact_density,
            **measurements,
            "segmented_mean_kl": segmented.get("mean_kl"),
            "segmented_nll_delta": (
                float(segmented["student_nll"]) - float(segmented["teacher_nll"]) if "student_nll" in segmented and "teacher_nll" in segmented else None
            ),
            "segmented_top1_agreement": segmented.get("top1_agreement"),
            "segmented_generation_byte_similarity": (generation.get("segmented_mean_byte_similarity") if generation.get("samples") not in {None, 0} else None),
        },
        dict(cast(Mapping[str, Any], experiment.get("gates", {}))),
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _failed_run(directory: Path, manifest: RunManifest) -> dict[str, Any]:
    failure_path = directory / "failure.json"
    failure = dict(load_artifact(failure_path)) if failure_path.is_file() else None
    result_path = directory / "result.json"
    result = dict(load_artifact(result_path)) if result_path.is_file() else None
    return {
        "directory": str(directory),
        "seed": manifest.seed,
        "operational_status": "failed",
        "scientific_verdict": "not_evaluated",
        "failure": failure,
        "result": result,
    }


def _replication_metadata(
    mode: str,
    *,
    complete: bool,
    scientific_verdict: ClaimVerdict,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "evidence_scope": "replication",
        "operational_status": "completed" if complete else "incomplete",
        "scientific_verdict": scientific_verdict,
        "replication_complete": complete,
    }


def _aggregate_output_runs(
    directories: Sequence[Path],
    manifests: Sequence[RunManifest],
) -> dict[str, Any]:
    completed_pairs = tuple((directory, manifest) for directory, manifest in zip(directories, manifests, strict=True) if manifest.status == "passed")
    results = tuple(load_artifact(directory / "result.json") for directory, _ in completed_pairs)
    failed_runs = [_failed_run(directory, manifest) for directory, manifest in zip(directories, manifests, strict=True) if manifest.status == "failed"]
    if any(result["evidence_scope"] != "final" for result in results):
        raise ValueError("output replication refuses non-final search evidence")
    if any(result["experiment"].get("model", {}).get("evaluation", "full") != "full" for result in results):
        raise ValueError("output replication requires full quality-evaluation runs")
    profiles = {result["experiment"]["training"]["profile"] for result in results}
    if results and profiles != {CAMPAIGN_PROFILE_NAME}:
        raise ValueError(f"replication evidence requires the {CAMPAIGN_PROFILE_NAME!r} campaign profile")
    seed_values = [int(result["experiment"]["seed"]) for result in results]
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("replication runs must use distinct seeds")
    seeds = set(seed_values)
    first = manifests[0]
    complete = not failed_runs and seeds == _CAMPAIGN_SEEDS and len(results) == len(_CAMPAIGN_SEEDS)
    structurally_unrepresentable = bool(results) and all(
        cast(Mapping[str, Any], result["output"]).get("structurally_unrepresentable") is True for result in results
    )
    outputs = [cast(Mapping[str, Any], result["output"]) for result in results]
    thresholds = [cast(Mapping[str, Any], cast(Mapping[str, Any], result["experiment"])["gates"]) for result in results]
    claim_verdicts = derive_output_claim_verdicts(
        outputs,
        thresholds,
        complete=complete,
        structurally_unrepresentable=structurally_unrepresentable,
    )
    claims = claim_records("output_only", claim_verdicts)
    category_verdicts = claim_category_verdicts(claims)
    scientific_verdict = category_verdicts["quality"]
    if structurally_unrepresentable and complete:
        scientific_verdict = "unsupported"
    metadata = _replication_metadata(
        "output_only",
        complete=complete,
        scientific_verdict=scientific_verdict,
    )
    run_rows = [
        {
            "directory": str(directory),
            "seed": manifest.seed,
            "mode": "output_only",
            "evidence_scope": str(result["evidence_scope"]),
            "operational_status": str(result["operational_status"]),
            "scientific_verdict": str(result["scientific_verdict"]),
            "metrics": dict(cast(Mapping[str, Any], result["output"])),
            "thresholds": dict(
                cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], result["experiment"])["gates"],
                )
            ),
        }
        for result, (directory, manifest) in zip(
            results,
            completed_pairs,
            strict=True,
        )
    ]
    confidence_eligible = complete and (first.model_id, first.model_revision) in _PRIMARY_MODELS
    return {
        **metadata,
        "model": {"id": first.model_id, "revision": first.model_revision},
        "dataset": {"id": first.dataset_id, "revision": first.dataset_revision},
        "profile": CAMPAIGN_PROFILE_NAME,
        "requested_runs": len(directories),
        "runs": run_rows,
        "seed_evidence": [
            {
                "seed": row["seed"],
                "operational_status": row["operational_status"],
                "scientific_verdict": row["scientific_verdict"],
                "metrics": row["metrics"],
                "thresholds": row["thresholds"],
                "claims": claim_records(
                    "output_only",
                    derive_output_claim_verdicts(
                        (cast(Mapping[str, object], row["metrics"]),),
                        (cast(Mapping[str, object], row["thresholds"]),),
                        complete=True,
                        structurally_unrepresentable=(cast(Mapping[str, Any], row["metrics"]).get("structurally_unrepresentable") is True),
                    ),
                ),
            }
            for row in run_rows
        ]
        + [
            {
                "seed": row["seed"],
                "operational_status": row["operational_status"],
                "scientific_verdict": row["scientific_verdict"],
                "metrics": {},
                "thresholds": {},
                "claims": claim_records(
                    "output_only",
                    dict.fromkeys(claim_verdicts, "incomplete"),
                ),
                "failure": row["failure"],
            }
            for row in failed_runs
        ],
        "failed_runs": failed_runs,
        "structurally_unrepresentable": structurally_unrepresentable,
        "metrics": {
            name: aggregate_numeric(
                [
                    (manifest.seed, float(result["output"][name]))
                    for result, (_, manifest) in zip(
                        results,
                        completed_pairs,
                        strict=True,
                    )
                    if result["output"].get(name) is not None
                ],
                confidence_eligible=(confidence_eligible and name != "candidate_reference_state_ratio"),
            )
            for name in _OUTPUT_REPLICATION_METRICS
        },
        "claims": claims,
        "category_verdicts": category_verdicts,
    }


def aggregate_runs(directories: Sequence[Path]) -> dict[str, Any]:
    if len(directories) != 3:
        raise ValueError("replication requires exactly three supplied runs")
    manifests = tuple(load_verified_run_manifest(directory / "manifest-final.json") for directory in directories)
    if any(manifest.status not in {"passed", "failed"} for manifest in manifests):
        raise ValueError("replication requires final completed or failed run manifests")
    modes = {manifest.mode for manifest in manifests}
    if len(modes) != 1:
        raise ValueError("replication aggregation refuses mixed tokenizer modes")
    contract = _replication_contract(manifests[0])
    if any(_replication_contract(item) != contract for item in manifests[1:]):
        raise ValueError("replication runs must share model, data, code, stages, and dependencies")
    if len({manifest.seed for manifest in manifests}) != 3:
        raise ValueError("replication runs must use exactly three distinct seeds")
    for directory, manifest in zip(directories, manifests, strict=True):
        _validate_run_identity(directory, manifest)
    summary = _aggregate_output_runs(directories, manifests) if manifests[0].mode == "output_only" else _aggregate_input_runs(directories, manifests)
    first = manifests[0]
    summary.update(
        {
            "source_commit": first.source_commit,
            "source_dirty": first.source_dirty,
            "source_state_sha256": first.source_state_sha256,
            "dependency_lock_sha256": first.dependency_lock_sha256,
            "installed_package": dict(first.installed_package),
            "claim_vocabulary_sha256": first.claim_vocabulary_sha256,
            "source_assets": {name: dict(value) for name, value in first.source_assets.items()},
            "verification": {
                "all_provided": all(manifest.verification.get("provided") is True for manifest in manifests),
                "all_passed": all(manifest.verification.get("all_passed") is True for manifest in manifests),
                "runs": [
                    {
                        "seed": manifest.seed,
                        **dict(manifest.verification),
                    }
                    for manifest in manifests
                ],
            },
        }
    )
    return summary


def _aggregate_input_runs(
    directories: Sequence[Path],
    manifests: Sequence[RunManifest],
) -> dict[str, Any]:
    completed_directories = tuple(directory for directory, manifest in zip(directories, manifests, strict=True) if manifest.status == "passed")
    runs = tuple(CompletedRun.load(directory) for directory in completed_directories)
    failed_runs = [_failed_run(directory, manifest) for directory, manifest in zip(directories, manifests, strict=True) if manifest.status == "failed"]
    if runs and any(run.profile != CAMPAIGN_PROFILE_NAME for run in runs):
        raise ValueError(f"replication evidence requires the {CAMPAIGN_PROFILE_NAME!r} campaign profile")
    if any(run.evidence_scope != "final" for run in runs):
        raise ValueError("input replication refuses non-final evidence")
    seeds = [run.seed for run in runs]
    complete = not failed_runs and set(seeds) == _CAMPAIGN_SEEDS and len(runs) == len(_CAMPAIGN_SEEDS)
    claim_verdicts = derive_input_claim_verdicts(
        [run.claim_metrics for run in runs],
        [run.claim_thresholds for run in runs],
        complete=complete,
    )
    claims = claim_records("input_only", claim_verdicts)
    category_verdicts = claim_category_verdicts(claims)
    scientific_verdict = derive_input_headline_verdict(claim_verdicts)
    metadata = _replication_metadata(
        "input_only",
        complete=complete,
        scientific_verdict=scientific_verdict,
    )
    confidence_eligible = complete and (manifests[0].model_id, manifests[0].model_revision) in _PRIMARY_MODELS
    return {
        **metadata,
        "model": {
            "id": manifests[0].model_id,
            "revision": manifests[0].model_revision,
        },
        "dataset": {
            "id": manifests[0].dataset_id,
            "revision": manifests[0].dataset_revision,
        },
        "profile": CAMPAIGN_PROFILE_NAME,
        "requested_runs": len(directories),
        "runs": [_input_run_row(run) for run in runs],
        "seed_evidence": [_input_seed_evidence(run) for run in runs]
        + [
            {
                "seed": row["seed"],
                "operational_status": row["operational_status"],
                "scientific_verdict": row["scientific_verdict"],
                "metrics": {},
                "thresholds": {},
                "claims": claim_records(
                    "input_only",
                    dict.fromkeys(claim_verdicts, "incomplete"),
                ),
                "failure": row["failure"],
            }
            for row in failed_runs
        ],
        "failed_runs": failed_runs,
        "claims": claims,
        "category_verdicts": category_verdicts,
        "metrics": {
            "native_tokens_per_continuous_token": aggregate_numeric(
                [(run.seed, run.native_tokens_per_continuous_token) for run in runs if run.native_tokens_per_continuous_token is not None],
                confidence_eligible=confidence_eligible,
            ),
            "candidate_reference_state_ratio": aggregate_numeric(
                [(run.seed, run.candidate_reference_state_ratio) for run in runs if run.candidate_reference_state_ratio is not None],
            ),
            "normalized_rmse": aggregate_numeric(
                [(run.seed, run.normalized_rmse) for run in runs],
                confidence_eligible=confidence_eligible,
            ),
            "segmented_mean_kl": aggregate_numeric(
                [(run.seed, run.segmented_mean_kl) for run in runs if run.segmented_mean_kl is not None],
                confidence_eligible=confidence_eligible,
            ),
            "prompt_cache_ratio": aggregate_numeric(
                [(run.seed, run.prompt_cache_ratio) for run in runs if run.prompt_cache_ratio is not None],
                confidence_eligible=confidence_eligible,
            ),
            "prefill_flops_ratio": aggregate_numeric(
                [(run.seed, run.prefill_flops_ratio) for run in runs if run.prefill_flops_ratio is not None],
                confidence_eligible=confidence_eligible,
            ),
            "time_to_first_logit_ratio": aggregate_numeric(
                [(run.seed, run.time_to_first_logit_ratio) for run in runs if run.time_to_first_logit_ratio is not None],
                confidence_eligible=confidence_eligible,
            ),
        },
    }


def _input_run_row(run: CompletedRun) -> dict[str, Any]:
    return {
        "directory": run.directory,
        "profile": run.profile,
        "seed": run.seed,
        "mode": "input_only",
        "status": run.status,
        "evidence_scope": run.evidence_scope,
        "operational_status": run.operational_status,
        "scientific_verdict": run.scientific_verdict,
        "metrics": {
            **run.claim_metrics,
            "time_to_first_logit_ratio": run.time_to_first_logit_ratio,
        },
        "thresholds": run.claim_thresholds,
    }


def _input_seed_evidence(run: CompletedRun) -> dict[str, object]:
    claim_verdicts = derive_input_claim_verdicts(
        (run.claim_metrics,),
        (run.claim_thresholds,),
        complete=True,
    )
    return {
        "seed": run.seed,
        "operational_status": run.operational_status,
        "scientific_verdict": derive_input_headline_verdict(claim_verdicts),
        "metrics": {
            **run.claim_metrics,
            "time_to_first_logit_ratio": run.time_to_first_logit_ratio,
        },
        "thresholds": run.claim_thresholds,
        "claims": claim_records("input_only", claim_verdicts),
    }


def _validate_run_identity(directory: Path, manifest: RunManifest) -> None:
    experiment_path = directory / "experiment.json"
    if not experiment_path.is_file():
        raise ValueError(f"run has no immutable experiment contract: {directory}")
    experiment = load_json_object(experiment_path)
    expected_fields = {field.name for field in fields(ExperimentSpec)}
    if set(experiment) != expected_fields:
        raise ValueError(f"run experiment contract is not canonical: {directory}")
    expected = {
        "name": manifest.experiment_name,
        "mode": manifest.mode,
        "seed": manifest.seed,
    }
    if any(experiment[name] != value for name, value in expected.items()):
        raise ValueError(f"run experiment identity disagrees with its manifest: {directory}")
    model = experiment.get("model")
    dataset = experiment.get("dataset")
    if not isinstance(model, Mapping) or set(model) != {"model_id", "revision", "evaluation"}:
        raise ValueError(f"run model contract is not canonical: {directory}")
    if not isinstance(dataset, Mapping) or set(dataset) != {"dataset_id", "config", "revision"}:
        raise ValueError(f"run dataset contract is not canonical: {directory}")
    if (
        model["model_id"] != manifest.model_id
        or model["revision"] != manifest.model_revision
        or dataset["dataset_id"] != manifest.dataset_id
        or dataset["revision"] != manifest.dataset_revision
    ):
        raise ValueError(f"run source identity disagrees with its manifest: {directory}")
    _validate_final_run_evidence(directory, manifest, experiment)


def _validate_final_run_evidence(
    directory: Path,
    manifest: RunManifest,
    experiment: Mapping[str, Any],
) -> None:
    if (manifest.model_id, manifest.model_revision) not in _PRIMARY_MODELS:
        raise ValueError("replication evidence must use a pinned primary model")
    if (manifest.dataset_id, manifest.dataset_revision) != WIKITEXT_DATASET:
        raise ValueError("replication evidence must use the pinned WikiText dataset")
    result_path = directory / "result.json"
    if manifest.status == "passed":
        result = load_json_object(result_path)
        if result.get("evidence_scope") != "final":
            raise ValueError("replication evidence must have final scope")
    training = cast(Mapping[str, Any], experiment.get("training", {}))
    if training.get("profile") != CAMPAIGN_PROFILE_NAME:
        raise ValueError(f"replication evidence requires the {CAMPAIGN_PROFILE_NAME!r} campaign profile")
    required_assets = {
        "model_config",
        "input_embedding_tensor",
        "tokenizer_vocabulary",
    }
    if not required_assets.issubset(manifest.source_assets):
        raise ValueError("replication evidence lacks the pinned source-asset inventory")
    expected_prefix = f"hf://{manifest.model_id}@{manifest.model_revision}"
    if any(not str(manifest.source_assets[name].get("locator", "")).startswith(expected_prefix) for name in required_assets):
        raise ValueError("replication source assets do not match the pinned model")
    checks = manifest.verification.get("checks")
    if (
        manifest.verification.get("provided") is not True
        or manifest.verification.get("all_passed") is not True
        or not isinstance(checks, Mapping)
        or set(checks) != FINAL_VERIFICATION_CHECKS
        or any(not isinstance(checks[name], Mapping) or checks[name].get("passed") is not True for name in FINAL_VERIFICATION_CHECKS)
    ):
        raise ValueError("replication evidence lacks the exact passing final verification inventory")


def _replication_contract(manifest: RunManifest) -> tuple[Any, ...]:
    return (
        manifest.mode,
        manifest.codec_direction,
        manifest.model_id,
        manifest.model_revision,
        manifest.dataset_id,
        manifest.dataset_revision,
        manifest.stages,
        manifest.source_commit,
        manifest.source_dirty,
        manifest.source_state_sha256,
        manifest.dependency_lock_sha256,
        json.dumps(
            manifest.installed_package,
            sort_keys=True,
            separators=(",", ":"),
        ),
        manifest.claim_vocabulary_sha256,
        json.dumps(
            manifest.source_assets,
            sort_keys=True,
            separators=(",", ":"),
        ),
        manifest.replication_fingerprint,
        manifest.codec_attention,
        json.dumps(manifest.verification, sort_keys=True, separators=(",", ":")),
    )


def encoding_cache_speedup(runs: Sequence[Mapping[str, Any]]) -> float | None:
    by_mode = {run["mode"]: run for run in runs}
    disabled = by_mode.get("disabled")
    warm = by_mode.get("warm")
    if disabled is None or warm is None or not warm.get("seconds", 0):
        return None
    return float(disabled["seconds"]) / float(warm["seconds"])
