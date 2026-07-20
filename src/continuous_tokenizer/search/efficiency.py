from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast, final

import optuna
import tomli_w
import torch
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FrozenTrial, TrialState

from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.artifacts.source import find_project_root, source_state
from continuous_tokenizer.artifacts.store import (
    json_compatible_object,
    load_json_object,
    write_json_atomic,
    write_text_atomic,
)
from continuous_tokenizer.backbone.assets import ModelAssets, load_model_assets
from continuous_tokenizer.contracts.experiment import (
    ExperimentSpec,
    SearchSelectionSpec,
)
from continuous_tokenizer.contracts.input import InputGateSpec, InputTrainingSpec
from continuous_tokenizer.contracts.search import EfficiencyPilotSpec
from continuous_tokenizer.input.studies import registered_vocabulary_subset
from continuous_tokenizer.input.training.run import TrainingOptions, training_options_from_spec
from continuous_tokenizer.input.training.vocabulary import fit_vocabulary_alignment
from continuous_tokenizer.runtime.device import declared_device
from continuous_tokenizer.runtime.progress import log_event
from continuous_tokenizer.search.trials import (
    _finished_trials,
    _reconcile_running_trials,
    _remaining_trials,
)


@final
@dataclass(frozen=True, slots=True)
class _CandidateParameters:
    learning_rate: float
    batch_size: int
    projection_multiplier: int
    muon_ns_steps: int


@final
@dataclass(frozen=True, slots=True)
class _EfficiencySearch:
    output_dir: Path
    spec: EfficiencyPilotSpec
    experiment: ExperimentSpec
    final_experiment: ExperimentSpec
    gates: InputGateSpec
    contract: dict[str, Any]


@final
@dataclass(frozen=True, slots=True)
class _SummaryRequest:
    output_dir: Path
    contract: dict[str, Any]
    study: optuna.Study
    baseline: dict[str, Any]
    gates: InputGateSpec
    spec: EfficiencyPilotSpec
    final_experiment: ExperimentSpec


def _vocabulary_sample(
    assets: ModelAssets,
    spec: EfficiencyPilotSpec,
) -> tuple[tuple[int, ...], str]:
    token_ids = registered_vocabulary_subset(
        assets,
        spec.vocabulary_rows,
        spec.sampler_seed,
    ).token_ids
    digest = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
    return token_ids, digest


def _candidate_options(
    base: TrainingOptions,
    spec: EfficiencyPilotSpec,
    output_dir: Path,
    parameters: _CandidateParameters,
) -> TrainingOptions:
    return replace(
        base,
        output_dir=output_dir,
        stages=("vocabulary",),
        profile=replace(
            base.profile,
            projection_multiplier=parameters.projection_multiplier,
        ),
        batch_size=parameters.batch_size,
        learning_rate=parameters.learning_rate,
        muon_ns_steps=parameters.muon_ns_steps,
        vocabulary_epochs=spec.vocabulary_epochs,
        reconstruction_epochs=0,
        reconstruction_samples=0,
        patience=spec.patience,
        evaluation_interval=spec.evaluation_interval,
    )


def _run_alignment(
    assets: ModelAssets,
    options: TrainingOptions,
    token_ids: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    started = perf_counter()
    result = fit_vocabulary_alignment(
        assets,
        options,
        device=device,
        token_ids=token_ids,
    )
    seconds = perf_counter() - started
    steps_per_epoch = (len(token_ids) + options.batch_size - 1) // options.batch_size
    return {
        **asdict(result),
        "duration_seconds": seconds,
        "steps_per_epoch": steps_per_epoch,
        "rows_per_second": len(token_ids) * options.vocabulary_epochs / max(seconds, 1e-12),
        "registered_parameters": {
            "learning_rate": options.learning_rate,
            "batch_size": options.batch_size,
            "projection_multiplier": options.profile.projection_multiplier,
            "muon_ns_steps": options.muon_ns_steps,
        },
    }


def _feasible(result: dict[str, Any], gates: InputGateSpec) -> bool:
    metrics = cast(dict[str, float], result["embedding_metrics"])
    return (
        metrics["normalized_rmse"] <= gates.maximum_normalized_rmse
        and metrics["cosine_similarity_p01"] >= gates.minimum_cosine_p01
        and metrics["cosine_similarity_p50"] >= gates.minimum_cosine_p50
        and float(result["candidate_reference_state_ratio"]) <= gates.maximum_candidate_reference_state_ratio
    )


def _serialize_trial(
    trial: FrozenTrial,
    baseline_seconds: float,
    minimum_improvement: float,
    gates: InputGateSpec,
) -> dict[str, Any]:
    result = trial.user_attrs.get("result")
    improvement = None if not isinstance(result, dict) else 1.0 - float(result["duration_seconds"]) / baseline_seconds
    return {
        "number": trial.number,
        "state": trial.state.name.lower(),
        "parameters": trial.params,
        "metrics": result,
        "feasible": isinstance(result, dict) and _feasible(result, gates),
        "runtime_improvement": improvement,
        "material_runtime_improvement": (improvement is not None and improvement >= minimum_improvement),
        "failure": trial.user_attrs.get("failure"),
        "final_evidence": False,
    }


def _selected_trial(
    trials: list[FrozenTrial],
    baseline_seconds: float,
    minimum_improvement: float,
    gates: InputGateSpec,
) -> FrozenTrial | None:
    completed: list[FrozenTrial] = []
    feasible: list[FrozenTrial] = []
    for trial in trials:
        result = trial.user_attrs.get("result")
        if trial.state is not TrialState.COMPLETE or not isinstance(result, dict):
            continue
        completed.append(trial)
        improvement = 1.0 - float(result["duration_seconds"]) / baseline_seconds
        if improvement >= minimum_improvement and _feasible(result, gates):
            feasible.append(trial)
    candidates = feasible or completed
    if not candidates:
        return None

    def score(trial: FrozenTrial) -> tuple[float, float, int]:
        result = cast(dict[str, Any], trial.user_attrs["result"])
        metrics = cast(dict[str, Any], result["embedding_metrics"])
        return (
            float(result["duration_seconds"]),
            float(metrics["normalized_rmse"]),
            trial.number,
        )

    return min(candidates, key=score)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Input Tokenizer Efficiency Pilot",
        "",
        f"- Status: `{summary['status']}`",
        "- Evidence scope: `search`",
        "- Final model evidence: `false`",
        f"- Baseline seconds: `{summary['baseline']['duration_seconds']:.6f}`",
        f"- Minimum runtime improvement: `{summary['minimum_runtime_improvement']:.2%}`",
        f"- Selected trial: `{summary['selected_trial']}`",
        "",
        (
            "Every failed and infeasible attempt remains in `search.json`. Selection requires the "
            "unchanged alignment and compactness gates plus the registered runtime improvement."
        ),
    ]
    return "\n".join(lines) + "\n"


def _write_summary(request: _SummaryRequest) -> dict[str, Any]:
    output_dir = request.output_dir
    contract = request.contract
    baseline = request.baseline
    gates = request.gates
    spec = request.spec
    final_experiment = request.final_experiment
    trials = request.study.get_trials(deepcopy=False)
    selected = _selected_trial(
        trials,
        float(baseline["duration_seconds"]),
        spec.minimum_runtime_improvement,
        gates,
    )
    serialized = [
        _serialize_trial(
            trial,
            float(baseline["duration_seconds"]),
            spec.minimum_runtime_improvement,
            gates,
        )
        for trial in trials
    ]
    finished = _finished_trials(trials)
    selected_result = None if selected is None else selected.user_attrs.get("result")
    runtime_improvement = (
        None if not isinstance(selected_result, dict) else 1.0 - float(selected_result["duration_seconds"]) / float(baseline["duration_seconds"])
    )
    selection_feasible = (
        isinstance(selected_result, dict)
        and _feasible(selected_result, gates)
        and runtime_improvement is not None
        and runtime_improvement >= spec.minimum_runtime_improvement
    )
    selected_parameters = (
        None
        if selected is None
        else {
            **dict(selected.params),
            "weight_decay": final_experiment.training.weight_decay,
        }
    )
    summary = {
        **contract,
        "status": "completed" if finished >= spec.trials else "running",
        "operational_status": "completed" if finished >= spec.trials else "running",
        "baseline": baseline,
        "minimum_runtime_improvement": spec.minimum_runtime_improvement,
        "finished_trials": finished,
        "completed_trials": sum(trial.state is TrialState.COMPLETE for trial in trials),
        "failed_trials": sum(trial.state in {TrialState.FAIL, TrialState.PRUNED} for trial in trials),
        "trials": serialized,
        "selected_trial": None if selected is None else selected.number,
        "selected_parameters": selected_parameters,
        "selected_metrics": selected_result,
        "selection_feasible": (None if selected is None else selection_feasible),
        "selected_alignment_passed": (None if not isinstance(selected_result, dict) else _feasible(selected_result, gates)),
        "selected_compactness_passed": (
            None
            if not isinstance(selected_result, dict)
            else float(selected_result["candidate_reference_state_ratio"]) <= gates.maximum_candidate_reference_state_ratio
        ),
        "selected_efficiency_passed": (None if selected is None else selection_feasible),
    }
    summary["artifacts"] = {
        **cast(dict[str, Any], summary["artifacts"]),
        "selected_experiment": (None if selected is None else "selected-experiment.toml"),
    }
    write_json_atomic(output_dir / "search.json", summary)
    write_text_atomic(output_dir / "search-report.md", _report(summary))
    if selected is not None and selected_parameters is not None:
        artifact_sha256 = sha256_file(output_dir / "search.json")
        selection = SearchSelectionSpec(
            search_kind="efficiency",
            artifact="search.json",
            artifact_sha256=artifact_sha256,
            selected_trial=selected.number,
            search_fingerprint=spec.fingerprint(),
            model_id=final_experiment.model.model_id,
            model_revision=final_experiment.model.revision,
            profile=final_experiment.training.profile,
            selected_parameters=selected_parameters,
            feasible=selection_feasible,
        )
        experiment = replace(
            final_experiment,
            name=f"{final_experiment.name}-efficiency-selected",
            evidence_scope="final",
            search_selections=(selection,),
            efficiency_pilot="search.json",
            efficiency_pilot_sha256=artifact_sha256,
        ).to_toml_dict()
        training = cast(dict[str, Any], experiment["training"])
        training.update(selected_parameters)
        write_text_atomic(
            output_dir / "selected-experiment.toml",
            tomli_w.dumps(experiment),
        )
    return summary


def _validate_efficiency_experiments(
    experiment: ExperimentSpec,
    final_experiment: ExperimentSpec,
) -> InputGateSpec:
    if not isinstance(experiment.training, InputTrainingSpec) or not isinstance(experiment.gates, InputGateSpec):
        raise ValueError("efficiency pilot requires an input-only experiment")
    if (
        final_experiment.mode != "input_only"
        or final_experiment.model.model_id != experiment.model.model_id
        or final_experiment.model.revision != experiment.model.revision
        or final_experiment.training.profile != experiment.training.profile
    ):
        raise ValueError("efficiency final experiment does not match its pilot model and profile")
    return experiment.gates


def _register_efficiency_contract(
    output_dir: Path,
    registered: dict[str, Any],
    *,
    resume: bool,
) -> None:
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"efficiency search directory already exists: {output_dir}")
        if dict(load_json_object(output_dir / "search-spec.json")) != registered:
            raise ValueError("existing efficiency search has a different contract")
        return
    if resume:
        raise FileNotFoundError("cannot resume a missing efficiency search")
    output_dir.mkdir(parents=True)
    write_json_atomic(output_dir / "search-spec.json", registered)


def _register_vocabulary_sample(
    output_dir: Path,
    assets: ModelAssets,
    spec: EfficiencyPilotSpec,
) -> tuple[tuple[int, ...], str]:
    token_ids, vocabulary_sha256 = _vocabulary_sample(assets, spec)
    sample = {"token_ids": list(token_ids), "sha256": vocabulary_sha256}
    sample_path = output_dir / "vocabulary-sample.json"
    if sample_path.is_file():
        if dict(load_json_object(sample_path)) != sample:
            raise ValueError("existing efficiency search has a different vocabulary sample")
    else:
        write_json_atomic(sample_path, sample)
    return token_ids, vocabulary_sha256


def _prepare_efficiency_search(
    spec_path: Path,
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[_EfficiencySearch, ModelAssets, tuple[int, ...]]:
    spec = EfficiencyPilotSpec.load(spec_path)
    experiment = ExperimentSpec.load((spec_path.parent / spec.experiment).resolve())
    final_experiment = ExperimentSpec.load((spec_path.parent / spec.final_experiment).resolve())
    gates = _validate_efficiency_experiments(experiment, final_experiment)
    project_root = find_project_root(spec_path)
    source_commit, source_dirty, source_state_sha256 = source_state(project_root)
    dependency_lock_sha256 = sha256_file(project_root / "uv.lock")
    registered = json_compatible_object(
        {
            "search": spec.to_dict(),
            "experiment": experiment.to_dict(),
            "final_experiment": final_experiment.to_dict(),
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "source_state_sha256": source_state_sha256,
            "dependency_lock_sha256": dependency_lock_sha256,
        }
    )
    _register_efficiency_contract(output_dir, registered, resume=resume)
    assets = load_model_assets(experiment.model.model_id, experiment.model.revision)
    token_ids, vocabulary_sha256 = _register_vocabulary_sample(output_dir, assets, spec)
    contract = {
        "mode": "input_only",
        "evidence_scope": "search",
        "scientific_verdict": "not_applicable_search",
        "name": spec.name,
        "search_fingerprint": spec.fingerprint(),
        "experiment_fingerprint": experiment.fingerprint(),
        "final_experiment_fingerprint": final_experiment.fingerprint(),
        "model_id": final_experiment.model.model_id,
        "model_revision": final_experiment.model.revision,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "source_state_sha256": source_state_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "verification": {"provided": False},
        "profile": final_experiment.training.profile,
        "requested_trials": spec.trials,
        "vocabulary_rows": spec.vocabulary_rows,
        "vocabulary_sha256": vocabulary_sha256,
        "search": spec.to_dict(),
        "experiment": experiment.to_dict(),
        "final_experiment": final_experiment.to_dict(),
        "artifacts": {
            "contract": "search-spec.json",
            "vocabulary_sample": "vocabulary-sample.json",
            "baseline": "baseline.json",
            "study": "optuna-journal.log",
            "report": "search-report.md",
            "selected_experiment": None,
        },
    }
    return (
        _EfficiencySearch(
            output_dir=output_dir,
            spec=spec,
            experiment=experiment,
            final_experiment=final_experiment,
            gates=gates,
            contract=contract,
        ),
        assets,
        token_ids,
    )


def _write_prepared(search: _EfficiencySearch) -> dict[str, Any]:
    prepared = {
        **search.contract,
        "status": "prepared",
        "operational_status": "running",
        "baseline": None,
        "finished_trials": 0,
        "trials": [],
        "selected_trial": None,
        "selection_feasible": None,
    }
    write_json_atomic(search.output_dir / "search.json", prepared)
    write_text_atomic(
        search.output_dir / "search-report.md",
        _report(
            {
                **prepared,
                "baseline": {"duration_seconds": 0.0},
                "minimum_runtime_improvement": search.spec.minimum_runtime_improvement,
            }
        ),
    )
    return prepared


def _baseline(
    search: _EfficiencySearch,
    assets: ModelAssets,
    token_ids: tuple[int, ...],
    device: torch.device,
    base: TrainingOptions,
) -> dict[str, Any]:
    baseline_path = search.output_dir / "baseline.json"
    if baseline_path.is_file():
        return dict(load_json_object(baseline_path))
    parameters = _CandidateParameters(
        learning_rate=base.learning_rate,
        batch_size=base.batch_size,
        projection_multiplier=base.profile.projection_multiplier,
        muon_ns_steps=base.muon_ns_steps,
    )
    result = _run_alignment(
        assets,
        _candidate_options(
            base,
            search.spec,
            search.output_dir / "baseline",
            parameters,
        ),
        token_ids,
        device,
    )
    write_json_atomic(baseline_path, result)
    return result


def _enqueue_coverage_trials(study: optuna.Study, spec: EfficiencyPilotSpec) -> None:
    coverage = max(
        len(spec.space.batch_sizes),
        len(spec.space.projection_multipliers),
        len(spec.space.muon_ns_steps),
    )
    ratio = spec.space.learning_rate_max / spec.space.learning_rate_min
    for index in range(coverage):
        fraction = index / max(coverage - 1, 1)
        study.enqueue_trial(
            {
                "learning_rate": spec.space.learning_rate_min * math.pow(ratio, fraction),
                "batch_size": spec.space.batch_sizes[index % len(spec.space.batch_sizes)],
                "projection_multiplier": spec.space.projection_multipliers[index % len(spec.space.projection_multipliers)],
                "muon_ns_steps": spec.space.muon_ns_steps[index % len(spec.space.muon_ns_steps)],
            }
        )


def _create_efficiency_study(
    search: _EfficiencySearch,
    *,
    resume: bool,
) -> optuna.Study:
    study = optuna.create_study(
        study_name=search.spec.name,
        storage=JournalStorage(JournalFileBackend(str(search.output_dir / "optuna-journal.log"))),
        sampler=TPESampler(seed=search.spec.sampler_seed),
        directions=("minimize", "minimize", "maximize", "maximize"),
        load_if_exists=resume,
    )
    if resume:
        _reconcile_running_trials(study)
    if not study.trials:
        _enqueue_coverage_trials(study, search.spec)
    return study


def _suggest_parameters(
    trial: optuna.Trial,
    spec: EfficiencyPilotSpec,
) -> _CandidateParameters:
    return _CandidateParameters(
        learning_rate=trial.suggest_float(
            "learning_rate",
            spec.space.learning_rate_min,
            spec.space.learning_rate_max,
            log=True,
        ),
        batch_size=trial.suggest_categorical(
            "batch_size",
            spec.space.batch_sizes,
        ),
        projection_multiplier=trial.suggest_categorical(
            "projection_multiplier",
            spec.space.projection_multipliers,
        ),
        muon_ns_steps=trial.suggest_categorical(
            "muon_ns_steps",
            spec.space.muon_ns_steps,
        ),
    )


def _efficiency_objective(
    search: _EfficiencySearch,
    assets: ModelAssets,
    token_ids: tuple[int, ...],
    device: torch.device,
    base: TrainingOptions,
) -> Callable[[optuna.Trial], tuple[float, float, float, float]]:
    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        options = _candidate_options(
            base,
            search.spec,
            search.output_dir / "trials" / f"{trial.number:04d}",
            _suggest_parameters(trial, search.spec),
        )
        try:
            result = _run_alignment(assets, options, token_ids, device)
        except Exception as error:
            trial.set_user_attr(
                "failure",
                {"type": type(error).__name__, "message": str(error)},
            )
            raise
        trial.set_user_attr("result", result)
        metrics = cast(dict[str, float], result["embedding_metrics"])
        return (
            float(result["duration_seconds"]),
            metrics["normalized_rmse"],
            metrics["cosine_similarity_p01"],
            metrics["cosine_similarity_p50"],
        )

    return objective


def _run_efficiency_trials(
    search: _EfficiencySearch,
    assets: ModelAssets,
    token_ids: tuple[int, ...],
    *,
    resume: bool,
) -> dict[str, Any]:
    device = declared_device(search.experiment.device)
    base = training_options_from_spec(search.experiment, search.output_dir / "unused")
    baseline = _baseline(search, assets, token_ids, device, base)
    study = _create_efficiency_study(search, resume=resume)

    def write_summary(current: optuna.Study) -> dict[str, Any]:
        return _write_summary(
            _SummaryRequest(
                output_dir=search.output_dir,
                contract=search.contract,
                study=current,
                baseline=baseline,
                gates=search.gates,
                spec=search.spec,
                final_experiment=search.final_experiment,
            )
        )

    def progress(current: optuna.Study, _completed: FrozenTrial) -> None:
        write_summary(current)

    remaining = _remaining_trials(study.trials, search.spec.trials)
    log_event(
        "efficiency_search_started",
        search=search.spec.name,
        remaining_trials=remaining,
    )
    study.optimize(
        _efficiency_objective(search, assets, token_ids, device, base),
        n_trials=remaining,
        n_jobs=1,
        callbacks=(progress,),
        catch=(Exception,),
    )
    return write_summary(study)


def run_efficiency_search(
    spec_path: Path,
    output_dir: Path,
    *,
    resume: bool,
    prepare_only: bool,
) -> dict[str, Any]:
    search, assets, token_ids = _prepare_efficiency_search(
        spec_path,
        output_dir,
        resume=resume,
    )
    if prepare_only:
        return _write_prepared(search)
    result = _run_efficiency_trials(
        search,
        assets,
        token_ids,
        resume=resume,
    )
    log_event(
        "efficiency_search_completed",
        search=search.spec.name,
        selected_trial=result["selected_trial"],
    )
    return result
