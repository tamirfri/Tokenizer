from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, cast, final

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
from continuous_tokenizer.contracts.search import SearchSpec
from continuous_tokenizer.input.studies import registered_vocabulary_subset
from continuous_tokenizer.input.training.run import TrainingOptions, training_options_from_spec
from continuous_tokenizer.input.training.vocabulary import fit_vocabulary_alignment
from continuous_tokenizer.reporting.search_markdown import search_report
from continuous_tokenizer.runtime.device import declared_device
from continuous_tokenizer.runtime.progress import log_event
from continuous_tokenizer.search.trials import (
    _finished_trials,
    _reconcile_running_trials,
    _remaining_trials,
)

METRIC_NAMES: Final = ("normalized_rmse", "cosine_p01", "cosine_p50")


@final
@dataclass(frozen=True, slots=True)
class _SearchArtifactRequest:
    output_dir: Path
    spec: SearchSpec
    experiment: ExperimentSpec
    final_experiment: ExperimentSpec
    study: optuna.Study
    vocabulary_sha256: str
    source_identity: tuple[str, bool, str]
    dependency_lock_sha256: str


@final
@dataclass(frozen=True, slots=True)
class _AlignmentSearch:
    output_dir: Path
    spec: SearchSpec
    experiment: ExperimentSpec
    final_experiment: ExperimentSpec
    gates: InputGateSpec
    source_identity: tuple[str, bool, str]
    dependency_lock_sha256: str


def select_trial(trials: list[FrozenTrial]) -> FrozenTrial | None:
    completed = [trial for trial in trials if trial.state is TrialState.COMPLETE and trial.values is not None]
    if not completed:
        return None

    def score(trial: FrozenTrial) -> tuple[float, float, float, int]:
        values = trial.values
        if values is None:
            raise AssertionError("completed search trial has no values")
        return values[0], -values[1], -values[2], trial.number

    return min(completed, key=score)


def _serialize_trial(trial: FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name.lower(),
        "parameters": trial.params,
        "metrics": (None if trial.values is None else dict(zip(METRIC_NAMES, trial.values, strict=True))),
        "duration_seconds": (None if trial.duration is None else trial.duration.total_seconds()),
        "alignment": trial.user_attrs.get("alignment"),
        "failure": trial.user_attrs.get("failure"),
        "final_evidence": False,
    }


def _selected_experiment(
    experiment: ExperimentSpec,
    selected: FrozenTrial,
    selection: SearchSelectionSpec,
) -> dict[str, Any]:
    values = experiment.to_toml_dict()
    values["name"] = f"{experiment.name}-search-selected"
    values["evidence_scope"] = "final"
    values["search_selections"] = [asdict(selection)]
    values["runtime"] = asdict(experiment.runtime)
    training = cast(dict[str, Any], values["training"])
    training.update(selected.params)
    return values


def _validate_experiments(
    spec: SearchSpec,
    experiment: ExperimentSpec,
    final_experiment: ExperimentSpec,
) -> InputGateSpec:
    if not isinstance(experiment.training, InputTrainingSpec) or not isinstance(experiment.gates, InputGateSpec):
        raise ValueError("vocabulary search requires an input-only experiment")
    if (
        final_experiment.mode != "input_only"
        or final_experiment.model.model_id != experiment.model.model_id
        or final_experiment.model.revision != experiment.model.revision
        or final_experiment.training.profile != experiment.training.profile
    ):
        raise ValueError("alignment search final experiment does not match its pilot model and profile")
    baseline = experiment.training
    if not spec.space.learning_rate_min <= baseline.learning_rate <= spec.space.learning_rate_max:
        raise ValueError("baseline learning rate is outside the registered search space")
    if baseline.weight_decay not in spec.space.weight_decays:
        raise ValueError("baseline weight decay is outside the registered search space")
    if baseline.batch_size not in spec.space.batch_sizes:
        raise ValueError("baseline batch size is outside the registered search space")
    return experiment.gates


def _write_search_artifacts(request: _SearchArtifactRequest) -> dict[str, Any]:
    output_dir = request.output_dir
    spec = request.spec
    experiment = request.experiment
    final_experiment = request.final_experiment
    if not isinstance(experiment.training, InputTrainingSpec) or not isinstance(experiment.gates, InputGateSpec):
        raise ValueError("vocabulary search requires an input-only experiment")
    training = experiment.training
    gates = experiment.gates
    trials = request.study.get_trials(deepcopy=False)
    selected = select_trial(trials)
    selected_values = None if selected is None else selected.values
    alignment_passed = (
        None
        if selected_values is None
        else selected_values[0] <= gates.maximum_normalized_rmse
        and selected_values[1] >= gates.minimum_cosine_p01
        and selected_values[2] >= gates.minimum_cosine_p50
    )
    alignment = None if selected is None else selected.user_attrs.get("alignment")
    compactness_passed = (
        None if not isinstance(alignment, dict) else float(alignment["candidate_reference_state_ratio"]) <= gates.maximum_candidate_reference_state_ratio
    )
    completed_trials = sum(trial.state is TrialState.COMPLETE for trial in trials)
    failed_trials = sum(trial.state in {TrialState.FAIL, TrialState.PRUNED} for trial in trials)
    finished_count = _finished_trials(trials)
    selection_feasible = bool(alignment_passed and compactness_passed) if selected is not None else None
    source_commit, source_dirty, source_state_sha256 = request.source_identity
    summary = {
        "mode": "input_only",
        "evidence_scope": "search",
        "operational_status": ("completed" if finished_count >= spec.trials else "running"),
        "scientific_verdict": "not_applicable_search",
        "status": "complete" if finished_count >= spec.trials else "running",
        "name": spec.name,
        "search_fingerprint": spec.fingerprint(),
        "experiment_fingerprint": experiment.fingerprint(),
        "final_experiment_fingerprint": final_experiment.fingerprint(),
        "model_id": final_experiment.model.model_id,
        "model_revision": final_experiment.model.revision,
        "profile": final_experiment.training.profile,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "source_state_sha256": source_state_sha256,
        "dependency_lock_sha256": request.dependency_lock_sha256,
        "verification": {"provided": False},
        "vocabulary_rows": spec.vocabulary_rows,
        "vocabulary_sha256": request.vocabulary_sha256,
        "sampler": "TPESampler",
        "sampler_seed": spec.sampler_seed,
        "sampler_startup_trials": min(3, max(1, spec.trials // 2)),
        "baseline_parameters": {
            "learning_rate": training.learning_rate,
            "weight_decay": training.weight_decay,
            "batch_size": training.batch_size,
        },
        "requested_trials": spec.trials,
        "finished_trials": finished_count,
        "completed_trials": completed_trials,
        "failed_trials": failed_trials,
        "selected_trial": None if selected is None else selected.number,
        "selected_parameters": None if selected is None else dict(selected.params),
        "selection_feasible": selection_feasible,
        "selected_alignment_passed": alignment_passed,
        "selected_compactness_passed": compactness_passed,
        "trials": [_serialize_trial(trial) for trial in trials],
        "artifacts": {
            "contract": "search-spec.json",
            "vocabulary_sample": "vocabulary-sample.json",
            "report": "search-report.md",
            "selected_experiment": None if selected is None else "selected-experiment.toml",
        },
    }
    write_json_atomic(output_dir / "search.json", summary)
    write_text_atomic(output_dir / "search-report.md", search_report(summary))
    if selected is not None:
        selection = SearchSelectionSpec(
            search_kind="alignment",
            artifact="search.json",
            artifact_sha256=sha256_file(output_dir / "search.json"),
            selected_trial=selected.number,
            search_fingerprint=spec.fingerprint(),
            model_id=final_experiment.model.model_id,
            model_revision=final_experiment.model.revision,
            profile=final_experiment.training.profile,
            selected_parameters=dict(selected.params),
            feasible=bool(selection_feasible),
        )
        selected_experiment = _selected_experiment(final_experiment, selected, selection)
        write_text_atomic(
            output_dir / "selected-experiment.toml",
            tomli_w.dumps(selected_experiment),
        )
    return summary


def _load_search(
    spec_path: Path,
    output_dir: Path,
    *,
    resume: bool,
) -> _AlignmentSearch:
    spec = SearchSpec.load(spec_path)
    experiment = ExperimentSpec.load((spec_path.parent / spec.experiment).resolve())
    final_experiment = ExperimentSpec.load((spec_path.parent / spec.final_experiment).resolve())
    gates = _validate_experiments(spec, experiment, final_experiment)
    if output_dir.exists() and not resume:
        raise FileExistsError(f"search directory already exists: {output_dir}")
    project_root = find_project_root(spec_path)
    source_identity = source_state(project_root)
    dependency_lock_sha256 = sha256_file(project_root / "uv.lock")
    registered = json_compatible_object(
        {
            "search": spec.to_dict(),
            "experiment": experiment.to_dict(),
            "final_experiment": final_experiment.to_dict(),
            "source_commit": source_identity[0],
            "source_dirty": source_identity[1],
            "source_state_sha256": source_identity[2],
            "dependency_lock_sha256": dependency_lock_sha256,
        }
    )
    if output_dir.exists():
        if dict(load_json_object(output_dir / "search-spec.json")) != registered:
            raise ValueError("existing search directory has a different specification")
    else:
        output_dir.mkdir(parents=True)
        write_json_atomic(output_dir / "search-spec.json", registered)
    return _AlignmentSearch(
        output_dir=output_dir,
        spec=spec,
        experiment=experiment,
        final_experiment=final_experiment,
        gates=gates,
        source_identity=source_identity,
        dependency_lock_sha256=dependency_lock_sha256,
    )


def _prepare_vocabulary(
    search: _AlignmentSearch,
) -> tuple[ModelAssets, torch.device, tuple[int, ...], str, TrainingOptions]:
    assets = load_model_assets(
        search.experiment.model.model_id,
        search.experiment.model.revision,
    )
    device = declared_device(search.experiment.device)
    vocabulary_ids = registered_vocabulary_subset(
        assets,
        search.spec.vocabulary_rows,
        search.spec.sampler_seed,
    ).token_ids
    digest = hashlib.sha256(json.dumps(vocabulary_ids, separators=(",", ":")).encode()).hexdigest()
    sample = {"token_ids": list(vocabulary_ids), "sha256": digest}
    sample_path = search.output_dir / "vocabulary-sample.json"
    if sample_path.is_file():
        if dict(load_json_object(sample_path)) != sample:
            raise ValueError("existing search directory has a different vocabulary sample")
    else:
        write_json_atomic(sample_path, sample)
    base = training_options_from_spec(search.experiment, search.output_dir / "unused")
    return assets, device, vocabulary_ids, digest, base


def _create_study(search: _AlignmentSearch, *, resume: bool) -> optuna.Study:
    storage = JournalStorage(JournalFileBackend(str(search.output_dir / "optuna-journal.log")))
    startup_trials = min(3, max(1, search.spec.trials // 2))
    study = optuna.create_study(
        study_name=search.spec.name,
        storage=storage,
        sampler=TPESampler(
            seed=search.spec.sampler_seed,
            n_startup_trials=startup_trials,
        ),
        directions=("minimize", "maximize", "maximize"),
        load_if_exists=resume,
    )
    if resume:
        _reconcile_running_trials(study)
    if not study.trials:
        training = search.experiment.training
        study.enqueue_trial(
            {
                "learning_rate": training.learning_rate,
                "weight_decay": training.weight_decay,
                "batch_size": training.batch_size,
            },
            user_attrs={"baseline": True},
        )
    return study


def _artifact_writer(
    search: _AlignmentSearch,
    vocabulary_sha256: str,
) -> Callable[[optuna.Study], dict[str, Any]]:
    def write(current: optuna.Study) -> dict[str, Any]:
        return _write_search_artifacts(
            _SearchArtifactRequest(
                output_dir=search.output_dir,
                spec=search.spec,
                experiment=search.experiment,
                final_experiment=search.final_experiment,
                study=current,
                vocabulary_sha256=vocabulary_sha256,
                source_identity=search.source_identity,
                dependency_lock_sha256=search.dependency_lock_sha256,
            )
        )

    return write


def _objective(
    search: _AlignmentSearch,
    assets: ModelAssets,
    device: torch.device,
    vocabulary_ids: tuple[int, ...],
    base_options: TrainingOptions,
) -> Callable[[optuna.Trial], tuple[float, float, float]]:
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        learning_rate = trial.suggest_float(
            "learning_rate",
            search.spec.space.learning_rate_min,
            search.spec.space.learning_rate_max,
            log=True,
        )
        weight_decay = trial.suggest_categorical(
            "weight_decay",
            search.spec.space.weight_decays,
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            search.spec.space.batch_sizes,
        )
        log_event(
            "search_trial_started",
            search=search.spec.name,
            trial=trial.number,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )
        options = replace(
            base_options,
            output_dir=search.output_dir / "trials" / f"{trial.number:04d}",
            stages=("vocabulary",),
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            vocabulary_epochs=search.spec.vocabulary_epochs,
            reconstruction_epochs=0,
            reconstruction_samples=0,
            patience=search.spec.patience,
            evaluation_interval=search.spec.evaluation_interval,
        )
        try:
            result = fit_vocabulary_alignment(
                assets,
                options,
                device=device,
                token_ids=vocabulary_ids,
            )
        except Exception as error:
            trial.set_user_attr(
                "failure",
                {"type": type(error).__name__, "message": str(error)},
            )
            raise
        trial.set_user_attr("alignment", asdict(result))
        if result.candidate_reference_state_ratio > search.gates.maximum_candidate_reference_state_ratio:
            log_event(
                "search_trial_pruned",
                search=search.spec.name,
                trial=trial.number,
                reason="candidate_state_budget",
                candidate_reference_state_ratio=result.candidate_reference_state_ratio,
            )
            raise optuna.TrialPruned(
                "candidate state exceeds the registered compactness budget",
            )
        metrics = result.embedding_metrics
        log_event(
            "search_trial_completed",
            search=search.spec.name,
            trial=trial.number,
            normalized_rmse=metrics["normalized_rmse"],
            cosine_p01=metrics["cosine_similarity_p01"],
            cosine_p50=metrics["cosine_similarity_p50"],
            candidate_reference_state_ratio=result.candidate_reference_state_ratio,
        )
        return (
            float(metrics["normalized_rmse"]),
            float(metrics["cosine_similarity_p01"]),
            float(metrics["cosine_similarity_p50"]),
        )

    return objective


def _optimize(
    search: _AlignmentSearch,
    study: optuna.Study,
    objective: Callable[[optuna.Trial], tuple[float, float, float]],
    write_artifacts: Callable[[optuna.Study], dict[str, Any]],
) -> dict[str, Any]:
    def write_progress(current: optuna.Study, completed: FrozenTrial) -> None:
        write_artifacts(current)
        log_event(
            "search_progress",
            search=search.spec.name,
            completed_trial=completed.number,
            trial_state=completed.state.name.lower(),
            finished_trials=_finished_trials(current.trials),
            requested_trials=search.spec.trials,
        )

    try:
        study.optimize(
            objective,
            n_trials=_remaining_trials(study.trials, search.spec.trials),
            n_jobs=1,
            callbacks=(write_progress,),
            catch=(Exception,),
        )
    except BaseException:
        write_artifacts(study)
        raise
    return write_artifacts(study)


def run_vocabulary_search(
    spec_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    prepare_only: bool = False,
) -> dict[str, Any]:
    search = _load_search(spec_path, output_dir, resume=resume)
    spec = search.spec
    log_event(
        "search_started",
        search=spec.name,
        mode="input_only",
        requested_trials=spec.trials,
        resume=resume,
        prepare_only=prepare_only,
    )
    assets, device, vocabulary_ids, vocabulary_sha256, base_options = _prepare_vocabulary(search)
    study = _create_study(search, resume=resume)
    write_artifacts = _artifact_writer(search, vocabulary_sha256)
    prepared = write_artifacts(study)
    if prepare_only:
        log_event("search_prepared", search=spec.name, requested_trials=spec.trials)
        return prepared
    result = _optimize(
        search,
        study,
        _objective(search, assets, device, vocabulary_ids, base_options),
        write_artifacts,
    )
    log_event(
        "search_completed",
        search=spec.name,
        finished_trials=result["finished_trials"],
        completed_trials=result["completed_trials"],
        failed_trials=result["failed_trials"],
        selected_trial=result["selected_trial"],
    )
    return result
