from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast, final

from torch import nn

from continuous_tokenizer.artifacts.evidence import (
    EVIDENCE_MANIFEST_FILENAME,
    load_evidence_manifest,
)
from continuous_tokenizer.artifacts.hashing import (
    installed_distribution_identity,
    sha256_file,
    sha256_path,
)
from continuous_tokenizer.artifacts.source import find_project_root, source_state
from continuous_tokenizer.artifacts.store import (
    RunDirectory,
    json_compatible_object,
    load_json_object,
)
from continuous_tokenizer.backbone.assets import (
    ModelAssets,
    load_frozen_causal_lm,
    load_model_assets,
)
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import (
    InputEvaluationSpec,
    InputGateSpec,
    InputTrainingSpec,
)
from continuous_tokenizer.contracts.input_study import (
    INPUT_SELECTION_CANDIDATES,
    InputAlignmentFeasibilityStudySpec,
    InputCompressionFeasibilityStudySpec,
    InputSelectionStudySpec,
    failed_alignment_gates,
)
from continuous_tokenizer.contracts.parsing import mapping_fingerprint
from continuous_tokenizer.data.corpus import (
    joined_prefix,
    load_corpus_documents,
    sample_content_windows,
)
from continuous_tokenizer.input.adapter import InputEmbeddingAdapter, SegmentationAlignment
from continuous_tokenizer.input.benchmark.tokenizer import (
    TokenizerMetricRequest,
    tokenizer_metrics,
)
from continuous_tokenizer.input.evaluation import (
    EvaluationRuntime,
    evaluate_input_replacement,
    evaluate_input_selection,
    evaluation_options_from_spec,
    teacher_forced_policy_from_spec,
)
from continuous_tokenizer.input.studies import (
    CandidateLengthRequest,
    candidate_length_report,
    input_behavior_gates,
    registered_vocabulary_subset,
    select_input_candidate,
)
from continuous_tokenizer.input.training.distillation import (
    DistillationOptions,
    DistillationRequest,
    distill_checkpoint,
)
from continuous_tokenizer.input.training.run import (
    TrainingResult,
    train_experiment,
    training_options_from_spec,
)
from continuous_tokenizer.input.training.vocabulary import fit_vocabulary_alignment
from continuous_tokenizer.reporting.discovery import load_training_progress
from continuous_tokenizer.reporting.input_study_markdown import (
    input_alignment_feasibility_report,
    input_compression_feasibility_report,
    input_study_report,
)
from continuous_tokenizer.runtime.device import declared_device
from continuous_tokenizer.runtime.environment import (
    dependency_environment,
    runtime_environment,
)
from continuous_tokenizer.runtime.resume import ResumeManager

_DISTILLATION_CANDIDATES: tuple[tuple[str, SegmentationAlignment], ...] = (
    ("token_aligned_distillation", "aligned"),
    ("arbitrary_boundary_distillation", "arbitrary"),
)


@final
@dataclass(frozen=True, slots=True)
class _StudyContext:
    run: RunDirectory
    assets: ModelAssets
    experiment: ExperimentSpec
    validation_data: bytes
    study: InputSelectionStudySpec
    source_commit: str = ""
    source_state_sha256: str = ""
    dependency_lock_sha256: str = ""
    resuming: bool = False


@final
@dataclass(frozen=True, slots=True)
class _AlignmentStageContext:
    requested_rows: int
    subset: Mapping[str, object]
    token_ids: tuple[int, ...]
    training_seeds: tuple[int, ...]
    gates: InputGateSpec


@final
@dataclass(frozen=True, slots=True)
class _AlignmentRuntime:
    run: RunDirectory
    assets: ModelAssets
    experiment: ExperimentSpec
    device: Any
    resume: bool


def _input_batch_size(experiment: ExperimentSpec) -> int:
    training = experiment.training
    if not isinstance(training, InputTrainingSpec):
        raise ValueError("input study requires input training settings")
    return min(training.batch_size, experiment.runtime.cache_chunk_rows)


def _load_study_assets(experiment: ExperimentSpec) -> ModelAssets:
    assets = load_model_assets(
        experiment.model.model_id,
        experiment.model.revision,
    )
    if assets.revision != experiment.model.revision:
        raise ValueError("resolved model revision differs from the registered study")
    return assets


def _distillation_options(
    experiment: ExperimentSpec,
    alignment: SegmentationAlignment,
) -> DistillationOptions:
    training = experiment.training
    if not isinstance(training, InputTrainingSpec):
        raise ValueError("input selection requires input training settings")
    return DistillationOptions(
        epochs=training.distillation_epochs,
        windows=training.distillation_windows,
        prompt_tokens=training.distillation_prompt_tokens,
        continuation_tokens=training.distillation_continuation_tokens,
        vocabulary_replay=training.batch_size,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        seed=experiment.seed,
        alignment=alignment,
    )


def _validation_data(
    experiment: ExperimentSpec,
    maximum_bytes: int,
) -> bytes:
    return joined_prefix(
        load_corpus_documents(
            "validation",
            dataset_id=experiment.dataset.dataset_id,
            config=experiment.dataset.config,
            revision=experiment.dataset.revision,
            max_rows=experiment.runtime.corpus_max_rows,
        ),
        max_bytes=maximum_bytes,
    )


def _candidate_metrics(
    context: _StudyContext,
    model: nn.Module,
    checkpoint: Path,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = context.run
    assets = context.assets
    experiment = context.experiment
    evaluation = experiment.evaluation
    training = experiment.training
    if not isinstance(evaluation, InputEvaluationSpec) or not isinstance(
        training,
        InputTrainingSpec,
    ):
        raise ValueError("input selection requires input-only settings")
    candidate_dir = run.path(f"candidates/{name}")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = candidate_dir / "tokenizer-metrics.json"
    validation_path = candidate_dir / "validation-metrics.json"
    lengths_path = candidate_dir / "candidate-lengths.json"
    if context.resuming and all(path.is_file() for path in (tokenizer_path, validation_path, lengths_path)):
        tokenizer = dict(load_json_object(tokenizer_path))
        validation = dict(load_json_object(validation_path))
        lengths = dict(load_json_object(lengths_path))
        return {
            "name": name,
            "checkpoint": str(checkpoint.relative_to(run.root)),
            "tokenizer": tokenizer,
            "validation": validation,
            "candidate_lengths": lengths,
        }, lengths
    loaded = InputEmbeddingAdapter.from_checkpoint(
        assets,
        checkpoint,
        device=declared_device(experiment.device),
    )
    validation_windows = sample_content_windows(
        [context.validation_data],
        maximum_bytes=len(context.validation_data),
    )
    tokenizer = tokenizer_metrics(
        assets,
        checkpoint,
        loaded,
        TokenizerMetricRequest(
            test_windows=validation_windows,
            batch_size=_input_batch_size(experiment),
            retrieval_rows=evaluation.retrieval_queries,
            repetitions=1,
            dataset_id=experiment.dataset.dataset_id,
            dataset_config=experiment.dataset.config,
            dataset_revision=experiment.dataset.revision,
            dataset_split="validation",
        ),
    )
    validation = evaluate_input_selection(
        assets,
        checkpoint,
        replace(
            evaluation_options_from_spec(
                experiment,
                candidate_dir,
                segmentation_alignment="arbitrary",
                dataset_split="validation",
            ),
            generation_samples=0,
            warmups=0,
            repetitions=1,
        ),
        EvaluationRuntime(
            frozen_model=model,
            resume_manager=ResumeManager(
                candidate_dir,
                mapping_fingerprint(
                    {
                        "study": context.study.fingerprint(),
                        "candidate": name,
                    },
                ),
                context.source_commit,
                context.source_state_sha256,
                context.dependency_lock_sha256,
                context.resuming,
            ),
            resume_phase=f"input-study-{name}",
            teacher_forced_policy=teacher_forced_policy_from_spec(experiment),
            calibration_cache_directory=(find_project_root(context.run.root) / ".cache" / "input-evaluation-calibration"),
            dependency_lock_sha256=context.dependency_lock_sha256,
        ),
    )
    lengths = candidate_length_report(
        loaded.adapter.codec,
        CandidateLengthRequest(
            assets=assets,
            validation_data=context.validation_data,
            candidate_lengths=context.study.candidate_lengths,
            binary_samples_per_length=context.study.binary_samples_per_length,
            seed=experiment.seed,
            batch_size=_input_batch_size(experiment),
        ),
    )
    run.write_json(f"candidates/{name}/tokenizer-metrics.json", tokenizer)
    run.write_json(f"candidates/{name}/validation-metrics.json", validation)
    run.write_json(f"candidates/{name}/candidate-lengths.json", lengths)
    return {
        "name": name,
        "checkpoint": str(checkpoint.relative_to(run.root)),
        "tokenizer": tokenizer,
        "validation": validation,
        "candidate_lengths": lengths,
    }, lengths


def _run_candidate_selection(
    context: _StudyContext,
    model: nn.Module,
    reconstruction_checkpoint: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    run = context.run
    assets = context.assets
    experiment = context.experiment
    documents = load_corpus_documents(
        "train",
        dataset_id=experiment.dataset.dataset_id,
        config=experiment.dataset.config,
        revision=experiment.dataset.revision,
        max_rows=experiment.runtime.corpus_max_rows,
    )
    checkpoints = {
        "reconstruction_only": reconstruction_checkpoint,
    }
    distillation: dict[str, Any] = {}
    for name, alignment in _DISTILLATION_CANDIDATES:
        checkpoint = run.path(f"candidates/{name}/checkpoint.pt")
        distillation_path = run.path(f"candidates/{name}/distillation.json")
        if context.resuming and checkpoint.is_file() and distillation_path.is_file():
            result_values = dict(load_json_object(distillation_path))
        else:
            result = distill_checkpoint(
                DistillationRequest(
                    assets=assets,
                    checkpoint=reconstruction_checkpoint,
                    output=checkpoint,
                    documents=documents,
                    options=_distillation_options(experiment, alignment),
                    frozen_model=model,
                    resume_manager=ResumeManager(
                        run.path(f"candidates/{name}"),
                        mapping_fingerprint(
                            {
                                "study": context.study.fingerprint(),
                                "candidate": name,
                                "phase": "distillation",
                            },
                        ),
                        context.source_commit,
                        context.source_state_sha256,
                        context.dependency_lock_sha256,
                        context.resuming,
                    ),
                ),
            )
            result_values = result.to_dict()
        checkpoints[name] = checkpoint
        distillation[name] = result_values
        run.write_json(f"candidates/{name}/distillation.json", result_values)

    candidates = []
    length_reports = []
    for name in INPUT_SELECTION_CANDIDATES:
        candidate, lengths = _candidate_metrics(
            context,
            model,
            checkpoints[name],
            name,
        )
        if name in distillation:
            candidate["distillation"] = distillation[name]
        candidates.append(candidate)
        length_reports.append({"label": f"candidate:{name}", **lengths})
    selection = select_input_candidate(candidates)
    selected_name = str(selection["selected_candidate"])
    selection["strategy"] = selected_name
    selection["selected_checkpoint"] = str(
        checkpoints[selected_name].relative_to(run.root),
    )
    run.write_json("selection.json", selection)
    run.write_json("candidates.json", candidates)
    return candidates, selection, length_reports


def _study_manifest(
    context: _StudyContext,
    artifacts: dict[str, str],
    registered: Mapping[str, Any],
) -> dict[str, Any]:
    run = context.run
    study = context.study
    experiment = context.experiment
    assets = context.assets
    return {
        "artifact_kind": "input_selection_study",
        "mode": "input_only",
        "status": "completed",
        "operational_status": "completed",
        "evidence_scope": "selection",
        "scientific_verdict": "not_applicable_selection",
        "study_fingerprint": study.fingerprint(),
        "experiment_fingerprint": experiment.fingerprint(),
        "model_id": assets.model_id,
        "model_revision": assets.revision,
        "source_commit": registered["source_commit"],
        "source_dirty": registered["source_dirty"],
        "source_state_sha256": registered["source_state_sha256"],
        "dependency_lock_sha256": registered["dependency_lock_sha256"],
        "installed_package": registered["installed_package"],
        "verification": {"provided": False},
        "artifacts": artifacts,
        "artifact_hashes": {name: sha256_path(run.root / relative) for name, relative in artifacts.items() if (run.root / relative).exists()},
    }


def _registered_study(
    study_path: Path,
    study: (InputSelectionStudySpec | InputAlignmentFeasibilityStudySpec | InputCompressionFeasibilityStudySpec),
    experiment: ExperimentSpec,
) -> dict[str, Any]:
    project_root = find_project_root(study_path)
    source_commit, source_dirty, source_state_sha256 = source_state(project_root)
    return json_compatible_object(
        {
            "study": study.to_dict(),
            "study_fingerprint": study.fingerprint(),
            "experiment": experiment.to_dict(),
            "experiment_fingerprint": experiment.fingerprint(),
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "source_state_sha256": source_state_sha256,
            "dependency_lock_sha256": sha256_file(project_root / "uv.lock"),
            "installed_package": installed_distribution_identity(
                "continuous-byte-tokenizer",
            ),
        },
    )


def _prepare_study_directory(
    output_dir: Path,
    registered: dict[str, Any],
    *,
    resume: bool,
) -> tuple[RunDirectory, dict[str, Any] | None]:
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"study directory already exists: {output_dir}")
        if dict(load_json_object(output_dir / "study-contract.json")) != registered:
            raise ValueError("existing study directory has a different sealed identity")
        result_path = output_dir / "result.json"
        manifest_path = output_dir / "manifest-final.json"
        evidence_path = output_dir / EVIDENCE_MANIFEST_FILENAME
        if result_path.is_file() and manifest_path.is_file():
            if evidence_path.is_file():
                load_evidence_manifest(evidence_path)
            return RunDirectory(output_dir, resume=True), dict(
                load_json_object(result_path),
            )
        return RunDirectory(output_dir, resume=True), None
    if resume:
        raise FileNotFoundError("cannot resume a missing study directory")
    run = RunDirectory(output_dir)
    run.write_json("study-contract.json", registered)
    return run, None


def _completed_trial(
    path: Path,
    subset: Mapping[str, object],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    trial = dict(load_json_object(path))
    if trial.get("vocabulary_subset") != subset:
        raise ValueError("completed study trial has a different vocabulary subset")
    hashes = trial.get("artifact_hashes")
    training = trial.get("training")
    if not isinstance(hashes, dict) or not isinstance(training, dict):
        raise ValueError("completed study trial has incomplete artifact identities")
    checkpoint = Path(str(training["checkpoint"]))
    lengths = path.parent / "candidate-lengths.json"
    expected = {
        "checkpoint": sha256_file(checkpoint),
        "candidate_lengths": sha256_file(lengths),
    }
    if hashes != expected:
        raise ValueError("completed study trial artifact hash mismatch")
    return trial


def _alignment_seed_decision(
    context: _AlignmentStageContext,
    training_seed: int,
    alignment: Mapping[str, object],
    alignment_result_path: Path,
) -> dict[str, Any]:
    metrics = cast(Mapping[str, object], alignment["embedding_metrics"])
    failed_gates = failed_alignment_gates(metrics, asdict(context.gates))
    return {
        "vocabulary_subset_size": context.requested_rows,
        "training_seed": training_seed,
        "status": "failed_gate" if failed_gates else "passed",
        "reason": ("training seed failed the registered alignment gates" if failed_gates else "training seed passed all registered alignment gates"),
        "failed_gates": failed_gates,
        "subset_sha256": context.subset["sha256"],
        "alignment": dict(alignment),
        "artifact_hashes": {
            "alignment_result": sha256_file(alignment_result_path),
        },
    }


def _alignment_stage_decision(
    context: _AlignmentStageContext,
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_gates = [
        {
            "training_seed": result["training_seed"],
            "failed_gates": result["failed_gates"],
        }
        for result in seed_results
        if result["status"] == "failed_gate"
    ]
    return {
        "vocabulary_subset_size": context.requested_rows,
        "status": "failed_gate" if failed_gates else "passed",
        "reason": (
            "one or more training seeds failed the registered alignment gates" if failed_gates else "all training seeds passed the registered alignment gates"
        ),
        "failed_gates": failed_gates,
        "vocabulary_subset": dict(context.subset),
        "subset_sha256": context.subset["sha256"],
        "training_seeds": list(context.training_seeds),
        "seed_results": seed_results,
    }


def _futility_stage_decision(
    requested_rows: int,
    subset: Mapping[str, object],
    training_seeds: tuple[int, ...],
    failed_stage: Mapping[str, Any],
) -> dict[str, Any]:
    prerequisite = int(failed_stage["vocabulary_subset_size"])
    reason = f"not run because prerequisite stage {prerequisite} failed the registered alignment gates"
    return {
        "vocabulary_subset_size": requested_rows,
        "status": "not_run_futility",
        "reason": reason,
        "failed_gates": failed_stage["failed_gates"],
        "failed_prerequisite_stage": prerequisite,
        "vocabulary_subset": dict(subset),
        "subset_sha256": subset["sha256"],
        "training_seeds": list(training_seeds),
        "seed_results": [
            {
                "training_seed": training_seed,
                "status": "not_run_futility",
                "reason": reason,
            }
            for training_seed in training_seeds
        ],
    }


def _load_or_reconcile_alignment_seed(
    run: RunDirectory,
    context: _AlignmentStageContext,
    training_seed: int,
    *,
    resume: bool,
) -> dict[str, Any] | None:
    seed_dir = run.path(
        f"stages/{context.requested_rows}/seeds/{training_seed}",
    )
    status_path = seed_dir / "status.json"
    alignment_path = seed_dir / "alignment-result.json"
    if not resume or not alignment_path.is_file():
        if status_path.is_file():
            raise ValueError("alignment seed status exists without its alignment result")
        return None
    alignment = dict(load_json_object(alignment_path))
    expected = _alignment_seed_decision(
        context,
        training_seed,
        alignment,
        alignment_path,
    )
    if status_path.is_file() and dict(load_json_object(status_path)) != expected:
        raise ValueError("sealed alignment seed status does not match its immutable result")
    if not status_path.is_file():
        run.write_json(
            f"stages/{context.requested_rows}/seeds/{training_seed}/status.json",
            expected,
        )
    return expected


def _load_or_reconcile_alignment_stage(
    run: RunDirectory,
    context: _AlignmentStageContext,
    *,
    resume: bool,
) -> tuple[list[dict[str, Any] | None], dict[str, Any] | None]:
    seed_results = [
        _load_or_reconcile_alignment_seed(
            run,
            context,
            training_seed,
            resume=resume,
        )
        for training_seed in context.training_seeds
    ]
    status_path = run.path(f"stages/{context.requested_rows}/status.json")
    completed_results = [result for result in seed_results if result is not None]
    if len(completed_results) != len(context.training_seeds):
        if status_path.is_file():
            raise ValueError("alignment stage status exists before every training seed completed")
        return seed_results, None
    expected = _alignment_stage_decision(
        context,
        completed_results,
    )
    if status_path.is_file() and dict(load_json_object(status_path)) != expected:
        raise ValueError("sealed alignment stage status does not match its immutable seed results")
    if not status_path.is_file():
        run.write_json(
            f"stages/{context.requested_rows}/status.json",
            expected,
        )
    return seed_results, expected


def _write_futility_stages(
    run: RunDirectory,
    assets: ModelAssets,
    study: InputAlignmentFeasibilityStudySpec,
    failed_stage: Mapping[str, Any],
    remaining_sizes: tuple[int, ...],
) -> list[dict[str, Any]]:
    skipped = []
    for requested_rows in remaining_sizes:
        subset = registered_vocabulary_subset(
            assets,
            requested_rows,
            study.subset_seed,
        ).to_dict()
        expected = _futility_stage_decision(
            requested_rows,
            subset,
            study.training_seeds,
            failed_stage,
        )
        path = run.path(f"stages/{requested_rows}/status.json")
        seeds_path = path.parent / "seeds"
        if seeds_path.exists():
            raise ValueError("futility-skipped stage contains alignment work")
        if path.is_file():
            if dict(load_json_object(path)) != expected:
                raise ValueError("sealed futility stage status is inconsistent")
        else:
            run.write_json(f"stages/{requested_rows}/status.json", expected)
        skipped.append(expected)
    return skipped


def _run_alignment_stage(
    runtime: _AlignmentRuntime,
    context: _AlignmentStageContext,
) -> dict[str, Any]:
    seed_results, decision = _load_or_reconcile_alignment_stage(
        runtime.run,
        context,
        resume=runtime.resume,
    )
    if decision is not None:
        return decision
    for seed_index, training_seed in enumerate(context.training_seeds):
        if seed_results[seed_index] is not None:
            continue
        seed_dir = runtime.run.path(
            f"stages/{context.requested_rows}/seeds/{training_seed}",
        )
        options = replace(
            training_options_from_spec(runtime.experiment, seed_dir),
            stages=("vocabulary",),
            reconstruction_epochs=0,
            reconstruction_samples=0,
            seed=training_seed,
            vocabulary_token_ids=context.token_ids,
        )
        alignment = asdict(
            fit_vocabulary_alignment(
                runtime.assets,
                options,
                device=runtime.device,
                token_ids=context.token_ids,
            ),
        )
        alignment_path = seed_dir / "alignment-result.json"
        if not alignment_path.is_file():
            raise RuntimeError("alignment training did not publish its seed result")
        seed_result = _alignment_seed_decision(
            context,
            training_seed,
            alignment,
            alignment_path,
        )
        runtime.run.write_json(
            f"stages/{context.requested_rows}/seeds/{training_seed}/status.json",
            seed_result,
        )
        seed_results[seed_index] = seed_result
    completed_seed_results = [result for result in seed_results if result is not None]
    if len(completed_seed_results) != len(context.training_seeds):
        raise AssertionError("alignment stage did not complete every training seed")
    decision = _alignment_stage_decision(context, completed_seed_results)
    runtime.run.write_json(
        f"stages/{context.requested_rows}/status.json",
        decision,
    )
    return decision


def _feasibility_manifest(
    run: RunDirectory,
    study: InputAlignmentFeasibilityStudySpec | InputCompressionFeasibilityStudySpec,
    experiment: ExperimentSpec,
    registered: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    artifact_kind = "input_alignment_feasibility_study" if isinstance(study, InputAlignmentFeasibilityStudySpec) else "input_compression_feasibility_study"
    return {
        "artifact_kind": artifact_kind,
        "mode": "input_only",
        "status": "completed",
        "operational_status": "completed",
        "evidence_scope": "selection",
        "scientific_verdict": "not_applicable_selection",
        "study_fingerprint": study.fingerprint(),
        "experiment_fingerprint": experiment.fingerprint(),
        "model_id": experiment.model.model_id,
        "model_revision": experiment.model.revision,
        "source_commit": registered["source_commit"],
        "source_dirty": registered["source_dirty"],
        "source_state_sha256": registered["source_state_sha256"],
        "dependency_lock_sha256": registered["dependency_lock_sha256"],
        "installed_package": registered["installed_package"],
        "verification": {"provided": False},
        "artifacts": dict(artifacts),
        "artifact_hashes": {name: sha256_path(run.root / relative) for name, relative in artifacts.items()},
    }


def run_input_alignment_feasibility_study(
    study_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    study = InputAlignmentFeasibilityStudySpec.load(study_path)
    experiment = study.load_experiment(study_path)
    training = experiment.training
    gates = experiment.gates
    if not isinstance(training, InputTrainingSpec) or not isinstance(gates, InputGateSpec):
        raise ValueError("input alignment feasibility requires input-only settings")
    registered = _registered_study(study_path, study, experiment)
    run, completed = _prepare_study_directory(
        output_dir,
        registered,
        resume=resume,
    )
    if completed is not None:
        return completed
    assets = _load_study_assets(experiment)
    device = declared_device(experiment.device)
    runtime = _AlignmentRuntime(
        run,
        assets,
        experiment,
        device,
        resume,
    )

    stages: list[dict[str, Any]] = []
    for index, requested_rows in enumerate(study.vocabulary_subset_sizes):
        subset = registered_vocabulary_subset(
            assets,
            requested_rows,
            study.subset_seed,
        )
        context = _AlignmentStageContext(
            requested_rows,
            subset.to_dict(),
            subset.token_ids,
            study.training_seeds,
            gates,
        )
        decision = _run_alignment_stage(runtime, context)
        stages.append(decision)
        if decision["status"] == "failed_gate":
            stages.extend(
                _write_futility_stages(
                    run,
                    assets,
                    study,
                    decision,
                    study.vocabulary_subset_sizes[index + 1 :],
                ),
            )
            break

    feasibility_passed = all(stage["status"] == "passed" for stage in stages)
    result = json_compatible_object(
        {
            "artifact_kind": "input_alignment_feasibility_study",
            "mode": "input_only",
            "operational_status": "completed",
            "evidence_scope": "selection",
            "scientific_verdict": "not_applicable_selection",
            "study": study.to_dict(),
            "study_fingerprint": study.fingerprint(),
            "experiment_fingerprint": experiment.fingerprint(),
            "model_id": assets.model_id,
            "model_revision": assets.revision,
            "model": {
                "id": assets.model_id,
                "revision": assets.revision,
                "embedding_tensor": assets.embedding_tensor_name,
                "source_dtype": str(assets.input_embeddings.dtype),
            },
            "training_seeds": study.training_seeds,
            "subset_seed": study.subset_seed,
            "stages": stages,
            "feasibility_passed": feasibility_passed,
            "acceptance_gates": {
                "maximum_normalized_rmse": gates.maximum_normalized_rmse,
                "minimum_cosine_p01": gates.minimum_cosine_p01,
                "minimum_cosine_p50": gates.minimum_cosine_p50,
            },
            "environment": dependency_environment(device),
            "training_performed": True,
            "decoder_training_performed": False,
            "reconstruction_training_performed": False,
            "distillation_performed": False,
            "full_model_evaluation_performed": False,
            "prospective": True,
            "final_evidence": False,
        },
    )
    run.write_json("result.json", result)
    run.write_text(
        "study-report.md",
        input_alignment_feasibility_report(result),
    )
    artifacts = {
        "contract": "study-contract.json",
        "stages": "stages",
        "result": "result.json",
        "report": "study-report.md",
    }
    run.write_json(
        "manifest-final.json",
        _feasibility_manifest(
            run,
            study,
            experiment,
            registered,
            artifacts,
        ),
    )
    return result


_COMPRESSION_STAGES = (
    "mechanism_exactness",
    "held_out_density",
    "candidate_behavior",
    "final_freeze_eligibility",
)


@final
@dataclass(frozen=True, slots=True)
class _CompressionRuntime:
    run: RunDirectory
    assets: ModelAssets
    experiment: ExperimentSpec
    study: InputCompressionFeasibilityStudySpec
    registered: Mapping[str, Any]
    device: Any
    resume: bool


def _compression_resume_manager(
    runtime: _CompressionRuntime,
    root: Path,
    identity: Mapping[str, object],
) -> ResumeManager:
    return ResumeManager(
        root,
        mapping_fingerprint(
            {
                "study": runtime.study.fingerprint(),
                **identity,
            },
        ),
        str(runtime.registered["source_commit"]),
        str(runtime.registered["source_state_sha256"]),
        str(runtime.registered["dependency_lock_sha256"]),
        runtime.resume and root.exists(),
        runtime.experiment.runtime.snapshot_interval,
    )


def _failed_gate(
    measurement: tuple[str, str, object, object, str],
    *,
    passed: bool,
) -> list[dict[str, object]]:
    if passed:
        return []
    gate, metric, measured, threshold, comparison = measurement
    return [
        {
            "gate": gate,
            "metric": metric,
            "measured": measured,
            "threshold": threshold,
            "comparison": comparison,
        },
    ]


def _completed_compression_seed(
    run: RunDirectory,
    stage: str,
    training_seed: int,
    *,
    resume: bool,
) -> dict[str, Any] | None:
    path = run.path(f"stages/{stage}/seeds/{training_seed}/result.json")
    if not resume or not path.is_file():
        return None
    result = dict(load_json_object(path))
    artifacts = result.get("artifacts")
    hashes = result.get("artifact_hashes")
    if not isinstance(artifacts, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("completed compression seed lacks artifact identities")
    expected = {str(name): sha256_path(run.root / str(relative)) for name, relative in artifacts.items()}
    if dict(hashes) != expected:
        raise ValueError("completed compression seed artifact hash mismatch")
    return result


def _write_compression_seed(
    run: RunDirectory,
    stage: str,
    training_seed: int,
    result: dict[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    compatible = json_compatible_object(
        {
            **result,
            "artifacts": dict(artifacts),
            "artifact_hashes": {name: sha256_path(run.root / relative) for name, relative in artifacts.items()},
        },
    )
    run.write_json(
        f"stages/{stage}/seeds/{training_seed}/result.json",
        compatible,
    )
    return compatible


def _compression_stage_decision(
    stage: str,
    training_seeds: tuple[int, ...],
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if tuple(result.get("training_seed") for result in seed_results) != training_seeds:
        raise ValueError("compression stage does not contain every registered seed")
    failed = [
        {
            "training_seed": result["training_seed"],
            "failed_gates": result["failed_gates"],
        }
        for result in seed_results
        if result["status"] == "failed_gate"
    ]
    decision: dict[str, Any] = {
        "stage": stage,
        "status": "failed_gate" if failed else "passed",
        "reason": ("one or more training seeds failed registered stage gates" if failed else "all training seeds passed registered stage gates"),
        "failed_gates": failed,
        "training_seeds": list(training_seeds),
        "seed_results": seed_results,
    }
    if stage == "candidate_behavior" and not failed:
        eligible = set(INPUT_SELECTION_CANDIDATES)
        for result in seed_results:
            eligible.intersection_update(result["eligible_candidates"])
        if not eligible:
            decision["status"] = "failed_gate"
            decision["reason"] = "no behavior-eligible candidate was shared by every training seed"
            decision["failed_gates"] = [
                {
                    "gate": "common_behavior_candidate",
                    "metric": "eligible_candidate_intersection",
                    "measured": [],
                    "threshold": "at least one common candidate",
                    "comparison": "contains",
                },
            ]
        else:
            decision["eligible_candidates"] = [name for name in INPUT_SELECTION_CANDIDATES if name in eligible]
            decision["selected_candidate"] = min(
                eligible,
                key=lambda name: (
                    sum(next(candidate["score"] for candidate in result["candidates"] if candidate["name"] == name)[0] for result in seed_results),
                    INPUT_SELECTION_CANDIDATES.index(name),
                ),
            )
    return decision


def _write_compression_stage(
    run: RunDirectory,
    decision: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    path = run.path(f"stages/{decision['stage']}/status.json")
    compatible = json_compatible_object(decision)
    if path.is_file():
        if not resume or dict(load_json_object(path)) != compatible:
            raise ValueError("sealed compression stage status is inconsistent")
    else:
        run.write_json(f"stages/{decision['stage']}/status.json", compatible)
    return compatible


def _compression_futility_stage(
    run: RunDirectory,
    stage: str,
    training_seeds: tuple[int, ...],
    failed_stage: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    work = run.path(f"stages/{stage}/seeds")
    if work.exists():
        raise ValueError("futility-skipped compression stage contains seed work")
    reason = f"not run because aggregate prerequisite {failed_stage['stage']} failed"
    return _write_compression_stage(
        run,
        {
            "stage": stage,
            "status": "not_run_futility",
            "reason": reason,
            "failed_prerequisite_stage": failed_stage["stage"],
            "failed_gates": failed_stage["failed_gates"],
            "training_seeds": list(training_seeds),
            "seed_results": [
                {
                    "training_seed": seed,
                    "status": "not_run_futility",
                    "reason": reason,
                    "failed_gates": failed_stage["failed_gates"],
                    "raw_metrics": {},
                }
                for seed in training_seeds
            ],
        },
        resume=resume,
    )


def _run_compression_mechanism_stage(
    runtime: _CompressionRuntime,
) -> dict[str, Any]:
    run = runtime.run
    assets = runtime.assets
    experiment = runtime.experiment
    study = runtime.study
    gates = cast(InputGateSpec, experiment.gates)
    subset = registered_vocabulary_subset(
        assets,
        study.vocabulary_subset_size,
        study.subset_seed,
    )
    seed_results = []
    for training_seed in study.training_seeds:
        completed = _completed_compression_seed(
            run,
            "mechanism_exactness",
            training_seed,
            resume=runtime.resume,
        )
        if completed is not None:
            seed_results.append(completed)
            continue
        seed_root = run.path(
            f"stages/mechanism_exactness/seeds/{training_seed}",
        )
        options = replace(
            training_options_from_spec(experiment, seed_root / "checkpoints"),
            stages=("vocabulary", "reconstruction"),
            seed=training_seed,
            vocabulary_token_ids=subset.token_ids,
        )
        training_result = train_experiment(
            assets,
            options,
            device=runtime.device,
            resume_manager=_compression_resume_manager(
                runtime,
                options.output_dir,
                {
                    "stage": "mechanism_exactness",
                    "training_seed": training_seed,
                    "subset": subset.to_dict(),
                },
            ),
        )
        checkpoint = Path(training_result.checkpoint)
        loaded = InputEmbeddingAdapter.from_checkpoint(
            assets,
            checkpoint,
            device=runtime.device,
        )
        lengths = candidate_length_report(
            loaded.adapter.codec,
            CandidateLengthRequest(
                assets=assets,
                validation_data=b"",
                candidate_lengths=study.candidate_lengths,
                binary_samples_per_length=study.binary_samples_per_length,
                seed=training_seed,
                batch_size=_input_batch_size(experiment),
                vocabulary_token_ids=subset.token_ids,
            ),
        )
        run.write_json(
            f"stages/mechanism_exactness/seeds/{training_seed}/candidate-lengths.json",
            lengths,
        )
        accepted = sum(
            int(row["accepted_spans"])
            for source in ("vocabulary", "arbitrary_binary")
            for row in cast(Mapping[str, Mapping[str, Any]], lengths[source]["metrics"]).values()
        )
        failed_gates = [
            *_failed_gate(
                (
                    "exact_byte_round_trip",
                    "round_trip",
                    training_result.round_trip,
                    True,
                    "equal",
                ),
                passed=training_result.round_trip,
            ),
            *_failed_gate(
                (
                    "minimum_multibyte_accepted_spans",
                    "accepted_multibyte_spans",
                    accepted,
                    1,
                    "greater_than_or_equal",
                ),
                passed=accepted >= 1,
            ),
        ]
        artifacts = {
            "checkpoint": str(checkpoint.relative_to(run.root)),
            "training_result": str(
                (options.output_dir / "training-result.json").relative_to(run.root),
            ),
            "candidate_lengths": (f"stages/mechanism_exactness/seeds/{training_seed}/candidate-lengths.json"),
        }
        seed_results.append(
            _write_compression_seed(
                run,
                "mechanism_exactness",
                training_seed,
                {
                    "training_seed": training_seed,
                    "status": "failed_gate" if failed_gates else "passed",
                    "failed_gates": failed_gates,
                    "subset_sha256": subset.sha256,
                    "raw_metrics": {
                        "training": asdict(training_result),
                        "candidate_lengths": lengths,
                        "accepted_multibyte_spans": accepted,
                        "alignment_failed_gates": failed_alignment_gates(
                            training_result.embedding_metrics,
                            asdict(gates),
                        ),
                        "alignment_is_continuation_gate": False,
                    },
                },
                artifacts,
            ),
        )
    return _write_compression_stage(
        run,
        _compression_stage_decision(
            "mechanism_exactness",
            study.training_seeds,
            seed_results,
        ),
        resume=runtime.resume,
    )


def _run_compression_density_stage(
    runtime: _CompressionRuntime,
    mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    run = runtime.run
    assets = runtime.assets
    experiment = runtime.experiment
    study = runtime.study
    evaluation = cast(InputEvaluationSpec, experiment.evaluation)
    gates = cast(InputGateSpec, experiment.gates)
    documents = load_corpus_documents(
        "validation",
        dataset_id=experiment.dataset.dataset_id,
        config=experiment.dataset.config,
        revision=experiment.dataset.revision,
        max_rows=experiment.runtime.corpus_max_rows,
    )
    windows = sample_content_windows(
        documents,
        maximum_bytes=study.validation_bytes,
        seed=study.subset_seed,
    )
    corpus = {
        "dataset_id": experiment.dataset.dataset_id,
        "dataset_config": experiment.dataset.config,
        "dataset_revision": experiment.dataset.revision,
        "split": "validation",
        "bytes": sum(len(window.payload) for window in windows),
        "content_sha256": mapping_fingerprint(
            {"windows": [window.to_dict() for window in windows]},
        ),
        "windows": [window.to_dict() for window in windows],
        "untouched_by_training": True,
    }
    run.write_json("validation-corpus.json", corpus)
    by_seed = {int(result["training_seed"]): result for result in cast(Sequence[Mapping[str, Any]], mechanism["seed_results"])}
    seed_results = []
    for training_seed in study.training_seeds:
        completed = _completed_compression_seed(
            run,
            "held_out_density",
            training_seed,
            resume=runtime.resume,
        )
        if completed is not None:
            seed_results.append(completed)
            continue
        checkpoint = run.root / str(by_seed[training_seed]["artifacts"]["checkpoint"])
        loaded = InputEmbeddingAdapter.from_checkpoint(
            assets,
            checkpoint,
            device=runtime.device,
        )
        metrics = tokenizer_metrics(
            assets,
            checkpoint,
            loaded,
            TokenizerMetricRequest(
                test_windows=windows,
                batch_size=_input_batch_size(experiment),
                retrieval_rows=evaluation.retrieval_queries,
                repetitions=evaluation.tokenizer_repetitions,
                dataset_id=experiment.dataset.dataset_id,
                dataset_config=experiment.dataset.config,
                dataset_revision=experiment.dataset.revision,
                dataset_split="validation",
            ),
        )
        metrics_path = f"stages/held_out_density/seeds/{training_seed}/tokenizer-metrics.json"
        run.write_json(metrics_path, metrics)
        density = cast(Mapping[str, Any], metrics["density"])
        wikitext = cast(
            Sequence[Mapping[str, Any]],
            metrics["density_strata"]["wikitext"]["windows"],
        )
        exact = bool(density["round_trip"]) and all(row["empirical_round_trip"] is True for row in wikitext)
        ratio = float(density["native_tokens_per_continuous_token"])
        failed_gates = [
            *_failed_gate(
                (
                    "exact_held_out_round_trip",
                    "held_out_round_trip",
                    exact,
                    True,
                    "equal",
                ),
                passed=exact,
            ),
            *_failed_gate(
                (
                    "minimum_native_tokens_per_continuous_token",
                    "native_tokens_per_continuous_token",
                    ratio,
                    gates.minimum_native_tokens_per_continuous_token,
                    "greater_than_or_equal",
                ),
                passed=(ratio >= gates.minimum_native_tokens_per_continuous_token),
            ),
        ]
        seed_results.append(
            _write_compression_seed(
                run,
                "held_out_density",
                training_seed,
                {
                    "training_seed": training_seed,
                    "status": "failed_gate" if failed_gates else "passed",
                    "failed_gates": failed_gates,
                    "validation_content_sha256": corpus["content_sha256"],
                    "raw_metrics": metrics,
                    "alignment_is_continuation_gate": False,
                },
                {
                    "checkpoint": str(checkpoint.relative_to(run.root)),
                    "tokenizer_metrics": metrics_path,
                    "validation_corpus": "validation-corpus.json",
                },
            ),
        )
    return _write_compression_stage(
        run,
        _compression_stage_decision(
            "held_out_density",
            study.training_seeds,
            seed_results,
        ),
        resume=runtime.resume,
    )


def _compression_behavior_gates(
    metrics: Mapping[str, Any],
    gates: InputGateSpec,
) -> tuple[list[dict[str, object]], list[float]]:
    segmented = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], metrics["teacher_forced"])["segmented"],
    )
    generation = cast(Mapping[str, Any], metrics["generation"])
    nll_delta = float(segmented["student_nll"]) - float(segmented["teacher_nll"])
    generation_similarity = float(
        generation["segmented_mean_byte_similarity"],
    )
    values = (
        (
            "maximum_segmented_mean_kl",
            "segmented_mean_kl",
            float(segmented["mean_kl"]),
            gates.maximum_segmented_mean_kl,
            "less_than_or_equal",
        ),
        (
            "maximum_segmented_nll_delta",
            "segmented_nll_delta",
            nll_delta,
            gates.maximum_segmented_nll_delta,
            "less_than_or_equal",
        ),
        (
            "minimum_segmented_top1_agreement",
            "segmented_top1_agreement",
            float(segmented["top1_agreement"]),
            gates.minimum_segmented_top1_agreement,
            "greater_than_or_equal",
        ),
        (
            "minimum_segmented_generation_byte_similarity",
            "segmented_generation_byte_similarity",
            generation_similarity,
            gates.minimum_segmented_generation_byte_similarity,
            "greater_than_or_equal",
        ),
    )
    passed = input_behavior_gates(metrics, gates)
    failed: list[dict[str, object]] = [
        {
            "gate": gate,
            "metric": metric,
            "measured": measured,
            "threshold": threshold,
            "comparison": comparison,
        }
        for gate, metric, measured, threshold, comparison in values
        if not passed[gate]
    ]
    return failed, [
        float(segmented["mean_kl"]),
        nll_delta,
        -float(segmented["top1_agreement"]),
        -generation_similarity,
    ]


def _run_compression_behavior_stage(
    runtime: _CompressionRuntime,
    mechanism: Mapping[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    run = runtime.run
    assets = runtime.assets
    experiment = runtime.experiment
    study = runtime.study
    gates = cast(InputGateSpec, experiment.gates)
    documents = load_corpus_documents(
        "train",
        dataset_id=experiment.dataset.dataset_id,
        config=experiment.dataset.config,
        revision=experiment.dataset.revision,
        max_rows=experiment.runtime.corpus_max_rows,
    )
    by_seed = {int(result["training_seed"]): result for result in cast(Sequence[Mapping[str, Any]], mechanism["seed_results"])}
    seed_results = []
    for training_seed in study.training_seeds:
        completed = _completed_compression_seed(
            run,
            "candidate_behavior",
            training_seed,
            resume=runtime.resume,
        )
        if completed is not None:
            seed_results.append(completed)
            continue
        reconstruction = run.root / str(by_seed[training_seed]["artifacts"]["checkpoint"])
        candidates = []
        artifacts: dict[str, str] = {}
        for name, alignment in (
            ("reconstruction_only", None),
            *_DISTILLATION_CANDIDATES,
        ):
            candidate_root = run.path(
                f"stages/candidate_behavior/seeds/{training_seed}/candidates/{name}",
            )
            checkpoint = reconstruction
            if alignment is not None:
                checkpoint = candidate_root / "checkpoint.pt"
                distillation_path = candidate_root / "distillation.json"
                if not (runtime.resume and checkpoint.is_file() and distillation_path.is_file()):
                    distillation = distill_checkpoint(
                        DistillationRequest(
                            assets=assets,
                            checkpoint=reconstruction,
                            output=checkpoint,
                            documents=documents,
                            options=replace(
                                _distillation_options(experiment, alignment),
                                seed=training_seed,
                            ),
                            frozen_model=model,
                            resume_manager=_compression_resume_manager(
                                runtime,
                                candidate_root,
                                {
                                    "stage": "candidate_behavior",
                                    "training_seed": training_seed,
                                    "candidate": name,
                                },
                            ),
                        ),
                    )
                    run.write_json(
                        str(distillation_path.relative_to(run.root)),
                        distillation.to_dict(),
                    )
                artifacts[f"{name}_distillation"] = str(
                    distillation_path.relative_to(run.root),
                )
            evaluation_root = candidate_root / "evaluation"
            metrics_path = evaluation_root / "llm-metrics.json"
            if runtime.resume and metrics_path.is_file():
                metrics = dict(load_json_object(metrics_path))
            else:
                metrics = evaluate_input_replacement(
                    assets,
                    checkpoint,
                    evaluation_options_from_spec(
                        experiment,
                        evaluation_root,
                        segmentation_alignment="arbitrary",
                        seed=training_seed,
                        dataset_split="validation",
                    ),
                    EvaluationRuntime(
                        device=runtime.device,
                        frozen_model=model,
                        resume_manager=_compression_resume_manager(
                            runtime,
                            evaluation_root,
                            {
                                "stage": "candidate_behavior",
                                "training_seed": training_seed,
                                "candidate": name,
                                "phase": "evaluation",
                            },
                        ),
                        resume_phase=f"compression-{training_seed}-{name}",
                        teacher_forced_policy=teacher_forced_policy_from_spec(experiment),
                        calibration_cache_directory=find_project_root(runtime.run.root) / ".cache" / "input-evaluation-calibration",
                        dependency_lock_sha256=str(runtime.registered["dependency_lock_sha256"]),
                    ),
                )
            failed_gates, score = _compression_behavior_gates(metrics, gates)
            candidates.append(
                {
                    "name": name,
                    "status": "failed_gate" if failed_gates else "passed",
                    "failed_gates": failed_gates,
                    "score": score,
                    "raw_metrics": metrics,
                },
            )
            artifacts[f"{name}_checkpoint"] = str(
                checkpoint.relative_to(run.root),
            )
            artifacts[f"{name}_evaluation"] = str(
                metrics_path.relative_to(run.root),
            )
        eligible = [candidate["name"] for candidate in candidates if candidate["status"] == "passed"]
        failed_gates = (
            []
            if eligible
            else [
                {
                    "candidate": candidate["name"],
                    "failed_gates": candidate["failed_gates"],
                }
                for candidate in candidates
            ]
        )
        seed_results.append(
            _write_compression_seed(
                run,
                "candidate_behavior",
                training_seed,
                {
                    "training_seed": training_seed,
                    "status": "failed_gate" if failed_gates else "passed",
                    "failed_gates": failed_gates,
                    "eligible_candidates": eligible,
                    "candidates": candidates,
                    "raw_metrics": {
                        "candidate_count": len(candidates),
                        "eligible_candidates": eligible,
                    },
                    "alignment_is_continuation_gate": False,
                },
                artifacts,
            ),
        )
    return _write_compression_stage(
        run,
        _compression_stage_decision(
            "candidate_behavior",
            study.training_seeds,
            seed_results,
        ),
        resume=runtime.resume,
    )


def _compression_freeze_eligibility_stage(
    run: RunDirectory,
    study: InputCompressionFeasibilityStudySpec,
    behavior: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    selected = str(behavior["selected_candidate"])
    return _write_compression_stage(
        run,
        {
            "stage": "final_freeze_eligibility",
            "status": "passed",
            "reason": ("all prerequisites passed; eligibility recorded without freezing an experiment"),
            "failed_gates": [],
            "training_seeds": list(study.training_seeds),
            "selected_candidate": selected,
            "seed_results": [
                {
                    "training_seed": seed,
                    "status": "passed",
                    "failed_gates": [],
                    "raw_metrics": {
                        "eligible": True,
                        "selected_candidate": selected,
                        "final_experiment_created": False,
                        "freeze_performed": False,
                        "final_claim_created": False,
                    },
                }
                for seed in study.training_seeds
            ],
            "eligibility_recorded": True,
            "final_experiment_created": False,
            "freeze_performed": False,
            "final_claim_created": False,
        },
        resume=resume,
    )


def _finish_compression_futility(
    runtime: _CompressionRuntime,
    completed_stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failed_stage = completed_stages[-1]
    remaining_stages = (
        _compression_futility_stage(
            runtime.run,
            stage,
            runtime.study.training_seeds,
            failed_stage,
            resume=runtime.resume,
        )
        for stage in _COMPRESSION_STAGES[len(completed_stages) :]
    )
    return [*completed_stages, *remaining_stages]


def _run_compression_ladder(
    runtime: _CompressionRuntime,
) -> list[dict[str, Any]]:
    mechanism = _run_compression_mechanism_stage(runtime)
    stages = [mechanism]
    if mechanism["status"] == "failed_gate":
        return _finish_compression_futility(runtime, stages)

    density = _run_compression_density_stage(runtime, mechanism)
    stages.append(density)
    if density["status"] == "failed_gate":
        return _finish_compression_futility(runtime, stages)

    behavior = _run_compression_behavior_stage(
        runtime,
        mechanism,
        load_frozen_causal_lm(runtime.assets, runtime.device),
    )
    stages.append(behavior)
    if behavior["status"] == "failed_gate":
        return _finish_compression_futility(runtime, stages)

    stages.append(
        _compression_freeze_eligibility_stage(
            runtime.run,
            runtime.study,
            behavior,
            resume=runtime.resume,
        ),
    )
    return stages


def run_input_compression_feasibility_study(
    study_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    study = InputCompressionFeasibilityStudySpec.load(study_path)
    experiment = study.load_experiment(study_path)
    registered = _registered_study(study_path, study, experiment)
    run, completed = _prepare_study_directory(
        output_dir,
        registered,
        resume=resume,
    )
    if completed is not None:
        return completed
    assets = _load_study_assets(experiment)
    device = declared_device(experiment.device)
    runtime = _CompressionRuntime(
        run,
        assets,
        experiment,
        study,
        registered,
        device,
        resume,
    )
    stages = _run_compression_ladder(runtime)
    feasibility_passed = all(stage["status"] == "passed" for stage in stages)
    behavior_ran = any(stage["stage"] == "candidate_behavior" and stage["status"] != "not_run_futility" for stage in stages)
    result = json_compatible_object(
        {
            "artifact_kind": "input_compression_feasibility_study",
            "mode": "input_only",
            "operational_status": "completed",
            "evidence_scope": "selection",
            "scientific_verdict": "not_applicable_selection",
            "study": study.to_dict(),
            "study_fingerprint": study.fingerprint(),
            "experiment_fingerprint": experiment.fingerprint(),
            "model_id": assets.model_id,
            "model_revision": assets.revision,
            "model": {
                "id": assets.model_id,
                "revision": assets.revision,
                "embedding_tensor": assets.embedding_tensor_name,
                "source_dtype": str(assets.input_embeddings.dtype),
            },
            "training_seeds": study.training_seeds,
            "subset_seed": study.subset_seed,
            "stages": stages,
            "feasibility_passed": feasibility_passed,
            "acceptance_gates": asdict(experiment.gates),
            "environment": dependency_environment(device),
            "training_performed": True,
            "decoder_training_performed": True,
            "reconstruction_training_performed": True,
            "distillation_performed": behavior_ran,
            "full_model_evaluation_performed": behavior_ran,
            "prospective": True,
            "freeze_eligibility_recorded": feasibility_passed,
            "final_experiment_created": False,
            "freeze_performed": False,
            "final_claim_created": False,
            "final_evidence": False,
        },
    )
    run.write_json("result.json", result)
    run.write_text(
        "study-report.md",
        input_compression_feasibility_report(result),
    )
    artifacts = {
        "contract": "study-contract.json",
        "stages": "stages",
        "result": "result.json",
        "report": "study-report.md",
    }
    if run.path("validation-corpus.json").is_file():
        artifacts["validation_corpus"] = "validation-corpus.json"
    run.write_json(
        "manifest-final.json",
        _feasibility_manifest(
            run,
            study,
            experiment,
            registered,
            artifacts,
        ),
    )
    return result


def run_input_selection_study(
    study_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    study = InputSelectionStudySpec.load(study_path)
    experiment = study.load_experiment(study_path)
    training = experiment.training
    if not isinstance(training, InputTrainingSpec):
        raise ValueError("input selection requires input training settings")
    registered = _registered_study(study_path, study, experiment)
    run, completed = _prepare_study_directory(
        output_dir,
        registered,
        resume=resume,
    )
    if completed is not None:
        return completed
    assets = _load_study_assets(experiment)
    device = declared_device(experiment.device)
    validation_data = _validation_data(experiment, study.validation_bytes)
    context = _StudyContext(
        run,
        assets,
        experiment,
        validation_data,
        study,
        str(registered["source_commit"]),
        str(registered["source_state_sha256"]),
        str(registered["dependency_lock_sha256"]),
        resume,
    )
    run.write_json(
        "validation-corpus.json",
        {
            "dataset_id": experiment.dataset.dataset_id,
            "dataset_config": experiment.dataset.config,
            "dataset_revision": experiment.dataset.revision,
            "split": "validation",
            "bytes": len(validation_data),
            "sha256": hashlib.sha256(validation_data).hexdigest(),
            "untouched_by_training": True,
        },
    )

    trials: list[dict[str, Any]] = []
    length_reports: list[dict[str, Any]] = []
    for requested_rows in study.vocabulary_subset_sizes:
        subset = registered_vocabulary_subset(
            assets,
            requested_rows,
            experiment.seed,
        )
        label = "complete" if requested_rows == 0 else str(requested_rows)
        trial_dir = run.path(f"trials/{label}")
        subset_values = subset.to_dict()
        completed_trial = (
            _completed_trial(
                trial_dir / "trial.json",
                subset_values,
            )
            if resume
            else None
        )
        if completed_trial is not None:
            trials.append(completed_trial)
            length_reports.append(
                {
                    "label": f"trial:{label}",
                    **completed_trial["candidate_lengths"],
                },
            )
            continue
        trial_resuming = resume and trial_dir.is_dir()
        options = replace(
            training_options_from_spec(experiment, trial_dir / "checkpoints"),
            vocabulary_token_ids=subset.token_ids,
        )
        training_result: TrainingResult = train_experiment(
            assets,
            options,
            device=device,
            resume_manager=ResumeManager(
                trial_dir / "checkpoints",
                mapping_fingerprint(
                    {
                        "study": study.fingerprint(),
                        "label": label,
                        "subset": subset_values,
                    },
                ),
                context.source_commit,
                context.source_state_sha256,
                context.dependency_lock_sha256,
                trial_resuming,
                experiment.runtime.snapshot_interval,
            ),
        )
        loaded = InputEmbeddingAdapter.from_checkpoint(
            assets,
            Path(training_result.checkpoint),
            device=device,
        )
        lengths = candidate_length_report(
            loaded.adapter.codec,
            CandidateLengthRequest(
                assets=assets,
                validation_data=validation_data,
                candidate_lengths=study.candidate_lengths,
                binary_samples_per_length=study.binary_samples_per_length,
                seed=experiment.seed,
                batch_size=_input_batch_size(experiment),
                vocabulary_token_ids=subset.token_ids,
            ),
        )
        run.write_json(f"vocabulary-subsets/{label}.json", subset_values)
        run.write_json(f"trials/{label}/candidate-lengths.json", lengths)
        trial = {
            "label": label,
            "vocabulary_subset": subset_values,
            "training": asdict(training_result),
            "candidate_lengths": lengths,
            "training_progress": load_training_progress(trial_dir),
            "telemetry_directory": str(
                (trial_dir / "checkpoints/progress").relative_to(run.root),
            ),
            "evidence_scope": "selection",
            "final_evidence": False,
            "artifact_hashes": {
                "checkpoint": sha256_file(Path(training_result.checkpoint)),
                "candidate_lengths": sha256_file(
                    trial_dir / "candidate-lengths.json",
                ),
            },
        }
        run.write_json(f"trials/{label}/trial.json", trial)
        trials.append(trial)
        length_reports.append({"label": f"trial:{label}", **lengths})

    candidates: list[dict[str, Any]] = []
    selection: dict[str, Any] | None = None
    if study.run_selection:
        model = load_frozen_causal_lm(assets, device)
        candidates, selection, candidate_reports = _run_candidate_selection(
            context,
            model,
            Path(trials[0]["training"]["checkpoint"]),
        )
        length_reports.extend(candidate_reports)

    result = json_compatible_object(
        {
            "artifact_kind": "input_selection_study",
            "mode": "input_only",
            "operational_status": "completed",
            "evidence_scope": "selection",
            "scientific_verdict": "not_applicable_selection",
            "verification": {"provided": False},
            "study": study.to_dict(),
            "study_fingerprint": study.fingerprint(),
            "experiment_fingerprint": experiment.fingerprint(),
            "model_id": assets.model_id,
            "model_revision": assets.revision,
            "model": {
                "id": assets.model_id,
                "revision": assets.revision,
                "embedding_tensor": assets.embedding_tensor_name,
                "source_dtype": str(assets.input_embeddings.dtype),
            },
            "dataset": asdict(experiment.dataset),
            "seed": experiment.seed,
            "trials": trials,
            "candidates": candidates,
            "selection": selection,
            "candidate_length_reports": length_reports,
            "acceptance_gates": asdict(experiment.gates),
            "environment": {
                **dependency_environment(device),
                "final": runtime_environment(device),
            },
            "final_evidence": False,
        },
    )
    run.write_json("result.json", result)
    run.write_text("study-report.md", input_study_report(result))
    artifacts = {
        "contract": "study-contract.json",
        "validation_corpus": "validation-corpus.json",
        "vocabulary_subsets": "vocabulary-subsets",
        "trials": "trials",
        "result": "result.json",
        "report": "study-report.md",
    }
    if selection is not None:
        artifacts.update(
            candidates="candidates",
            selection="selection.json",
        )
    run.write_json(
        "manifest-final.json",
        _study_manifest(
            context,
            artifacts,
            registered,
        ),
    )
    return result
