from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Never

import optuna
import tomli_w
from optuna.trial import FrozenTrial, TrialState

from continuous_tokenizer.artifacts.evidence import (
    EVIDENCE_MANIFEST_FILENAME,
    load_evidence_manifest,
)
from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.artifacts.manifest import load_artifact
from continuous_tokenizer.artifacts.source import find_project_root, source_state
from continuous_tokenizer.artifacts.store import (
    json_compatible_object,
    load_json_object,
    write_json_atomic,
    write_text_atomic,
)
from continuous_tokenizer.campaigns.output import (
    OutputExperimentRunner,
    OutputOracleCeilingError,
    OutputPilotCorpus,
    OutputRunnerOptions,
)
from continuous_tokenizer.contracts.experiment import (
    ExperimentSpec,
    SearchSelectionSpec,
)
from continuous_tokenizer.contracts.output import OutputEvaluationSpec, OutputTrainingSpec
from continuous_tokenizer.contracts.search import OutputSearchSpec
from continuous_tokenizer.data.corpus import load_corpus_documents
from continuous_tokenizer.output.corpora import select_output_documents
from continuous_tokenizer.reporting.search_markdown import search_report
from continuous_tokenizer.runtime.progress import log_event
from continuous_tokenizer.search.trials import (
    _finished_trials,
    _reconcile_running_trials,
    _remaining_trials,
)


@dataclass(frozen=True, slots=True)
class _OutputSearchContext:
    output_dir: Path
    search: OutputSearchSpec
    baseline: ExperimentSpec
    final_experiment: ExperimentSpec
    pilot_corpus: OutputPilotCorpus
    pilot_metadata: dict[str, Any]
    project_root: Path
    oracle_selection: dict[str, Any] | None
    source_commit: str
    source_dirty: bool
    source_state_sha256: str
    dependency_lock_sha256: str


def _pilot_corpus(
    search: OutputSearchSpec,
    baseline: ExperimentSpec,
) -> tuple[OutputPilotCorpus, dict[str, Any]]:
    evaluation = baseline.evaluation
    if not isinstance(evaluation, OutputEvaluationSpec):
        raise ValueError("output search requires output evaluation corpus contracts")
    documents = load_corpus_documents(
        "train",
        dataset_id=baseline.dataset.dataset_id,
        config=baseline.dataset.config,
        revision=baseline.dataset.revision,
    )
    required = search.pilot_documents * 3
    if len(documents) < required:
        raise ValueError(f"output search requires at least {required} pilot documents")
    training = select_output_documents(
        documents,
        count=search.pilot_documents,
        seed=evaluation.training_corpus.seed,
    )
    checkpoint_selection = select_output_documents(
        documents,
        count=search.pilot_documents,
        seed=evaluation.checkpoint_selection_corpus.seed,
        excluded_sha256=frozenset(training.document_sha256),
    )
    excluded = frozenset(training.document_sha256 + checkpoint_selection.document_sha256)
    oracle_validation = select_output_documents(
        documents,
        count=search.pilot_documents,
        seed=evaluation.oracle_validation_corpus.seed,
        excluded_sha256=excluded,
    )
    digest = hashlib.sha256()
    for corpus in (training, checkpoint_selection, oracle_validation):
        digest.update(bytes.fromhex(corpus.sha256))
    return (
        OutputPilotCorpus(
            training_documents=training.documents,
            checkpoint_selection_documents=checkpoint_selection.documents,
            oracle_validation_documents=oracle_validation.documents,
        ),
        {
            "training_documents": len(training.documents),
            "training_seed": evaluation.training_corpus.seed,
            "training_sha256": training.sha256,
            "checkpoint_selection_documents": len(checkpoint_selection.documents),
            "checkpoint_selection_seed": evaluation.checkpoint_selection_corpus.seed,
            "checkpoint_selection_sha256": checkpoint_selection.sha256,
            "oracle_validation_documents": len(oracle_validation.documents),
            "oracle_validation_seed": evaluation.oracle_validation_corpus.seed,
            "oracle_validation_sha256": oracle_validation.sha256,
            "sha256": digest.hexdigest(),
        },
    )


def _serialize_trial(trial: FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "parameters": dict(trial.params),
        "metrics": trial.user_attrs.get("metrics"),
        "gates": trial.user_attrs.get("gates"),
        "failure": trial.user_attrs.get("failure") or trial.user_attrs.get("prune_reason"),
        "final_evidence": False,
    }


def _all_gates_feasible(trial: FrozenTrial) -> bool:
    gates = trial.user_attrs.get("gates")
    return isinstance(gates, dict) and bool(gates) and all(value is True for value in gates.values())


def _select_output_trial(
    completed: list[FrozenTrial],
    *,
    fallback_rule: str,
) -> FrozenTrial | None:
    if not completed:
        return None
    feasible = [trial for trial in completed if _all_gates_feasible(trial)]
    candidates = feasible or completed
    if not feasible and fallback_rule != "best_exact_full_sequence_rate":
        raise ValueError("output search fallback rule is not registered")
    return min(
        candidates,
        key=lambda trial: (
            float("inf") if trial.value is None else trial.value,
            trial.number,
        ),
    )


def _oracle_selection(path: Path, baseline: ExperimentSpec) -> dict[str, Any]:
    directory = path if path.is_dir() else path.parent
    load_evidence_manifest(directory / EVIDENCE_MANIFEST_FILENAME)
    result_path = directory / "result.json"
    artifact = dict(load_artifact(result_path))
    selection = artifact.get("selection")
    if (
        artifact.get("artifact_kind") != "output_oracle_study"
        or artifact.get("operational_status") != "completed"
        or artifact.get("model_id") != baseline.model.model_id
        or artifact.get("model_revision") != baseline.model.revision
        or not isinstance(selection, dict)
        or not isinstance(selection.get("max_span"), int)
    ):
        raise ValueError("sealed output oracle study has no registered selection for this search")
    return {
        "artifact": str(result_path.resolve()),
        "artifact_sha256": sha256_file(result_path),
        "study_fingerprint": artifact["study_fingerprint"],
        "max_span": selection["max_span"],
        "selection_feasible": bool(selection.get("selection_feasible")),
    }


def _apply_oracle_selection(
    artifact_path: Path | None,
    baseline: ExperimentSpec,
    final_experiment: ExperimentSpec,
) -> tuple[ExperimentSpec, ExperimentSpec, dict[str, Any] | None]:
    if artifact_path is None:
        return baseline, final_experiment, None
    selection = _oracle_selection(artifact_path, baseline)
    selected_span = int(selection["max_span"])
    return (
        replace(
            baseline,
            training=replace(baseline.training, max_span=selected_span),
        ),
        replace(
            final_experiment,
            training=replace(final_experiment.training, max_span=selected_span),
        ),
        selection,
    )


def _study_summary(
    contract: dict[str, Any],
    study: optuna.Study,
    *,
    status: str,
    operational_status: str,
    failure: dict[str, str] | None = None,
) -> dict[str, Any]:
    trials = study.get_trials(deepcopy=False)
    completed = [trial for trial in trials if trial.state is TrialState.COMPLETE]
    failed = [trial for trial in trials if trial.state in {TrialState.FAIL, TrialState.PRUNED}]
    fallback_rule = str(contract["search"]["fallback_rule"])
    selected = _select_output_trial(completed, fallback_rule=fallback_rule)
    selected_gates = None if selected is None else selected.user_attrs.get("gates")
    selection_feasible = None if selected is None else _all_gates_feasible(selected)
    selection_policy = None
    if selected is not None:
        selection_policy = "all_gates_feasible" if selection_feasible else fallback_rule
    return {
        **contract,
        "status": status,
        "operational_status": operational_status,
        "finished_trials": _finished_trials(trials),
        "completed_trials": len(completed),
        "failed_trials": len(failed),
        "trials": [_serialize_trial(trial) for trial in trials],
        "selected_trial": None if selected is None else selected.number,
        "selected_parameters": None if selected is None else dict(selected.params),
        "selected_metrics": None if selected is None else selected.user_attrs.get("metrics"),
        "selected_gates": selected_gates,
        "selection_feasible": selection_feasible,
        "selection_policy": selection_policy,
        "selected_output_passed": selection_feasible,
        "failure": failure,
    }


def _write_search_artifacts(output_dir: Path, summary: dict[str, Any]) -> None:
    write_json_atomic(output_dir / "search.json", summary)
    write_text_atomic(output_dir / "search-report.md", search_report(summary))


def _record_prune(
    trial: optuna.Trial,
    reason: str,
    *,
    search_name: str,
) -> None:
    trial.set_user_attr("prune_reason", reason)
    log_event(
        "search_trial_pruned",
        search=search_name,
        trial=trial.number,
        reason=reason,
    )


def _prune_infeasible_oracle(
    trial: optuna.Trial,
    error: OutputOracleCeilingError,
    *,
    search_name: str,
) -> Never:
    reason = str(error)
    trial.set_user_attr(
        "metrics",
        {
            "native_head_oracle_ceilings": error.ceilings,
            "oracle_feasible": False,
        },
    )
    trial.set_user_attr("gates", {"oracle_feasible": False})
    _record_prune(trial, reason, search_name=search_name)
    raise optuna.TrialPruned(reason) from error


def _load_search_context(
    spec_path: Path,
    output_dir: Path,
    oracle_study_artifact: Path | None,
) -> _OutputSearchContext:
    search = OutputSearchSpec.load(spec_path)
    baseline = ExperimentSpec.load(spec_path.parent / search.experiment)
    final_experiment = ExperimentSpec.load(spec_path.parent / search.final_experiment)
    if baseline.mode != "output_only" or not isinstance(
        baseline.training,
        OutputTrainingSpec,
    ):
        raise ValueError("output search must reference an output-only experiment")
    if (
        final_experiment.mode != "output_only"
        or final_experiment.model.model_id != baseline.model.model_id
        or final_experiment.model.revision != baseline.model.revision
        or final_experiment.training.profile != baseline.training.profile
    ):
        raise ValueError(
            "output search final experiment does not match its pilot model and profile",
        )
    baseline, final_experiment, oracle_selection = _apply_oracle_selection(
        oracle_study_artifact,
        baseline,
        final_experiment,
    )
    pilot_corpus, pilot_metadata = _pilot_corpus(search, baseline)
    project_root = find_project_root(spec_path)
    source_commit, source_dirty, source_state_sha256 = source_state(project_root)
    return _OutputSearchContext(
        output_dir=output_dir,
        search=search,
        baseline=baseline,
        final_experiment=final_experiment,
        pilot_corpus=pilot_corpus,
        pilot_metadata=pilot_metadata,
        project_root=project_root,
        oracle_selection=oracle_selection,
        source_commit=source_commit,
        source_dirty=source_dirty,
        source_state_sha256=source_state_sha256,
        dependency_lock_sha256=sha256_file(project_root / "uv.lock"),
    )


def _registered_search(context: _OutputSearchContext) -> dict[str, Any]:
    return json_compatible_object(
        {
            "search": context.search.to_dict(),
            "experiment": context.baseline.to_dict(),
            "final_experiment": context.final_experiment.to_dict(),
            "pilot_corpus": context.pilot_metadata,
            "oracle_study": context.oracle_selection,
            "source_commit": context.source_commit,
            "source_dirty": context.source_dirty,
            "source_state_sha256": context.source_state_sha256,
            "dependency_lock_sha256": context.dependency_lock_sha256,
        },
    )


def _search_contract(
    context: _OutputSearchContext,
    *,
    prepare_only: bool,
) -> dict[str, Any]:
    search = context.search
    baseline = context.baseline
    final_experiment = context.final_experiment
    return json_compatible_object(
        {
            "mode": "output_only",
            "evidence_scope": "search",
            "operational_status": "running",
            "scientific_verdict": "not_applicable_search",
            "name": search.name,
            "search_fingerprint": search.fingerprint(),
            "experiment_fingerprint": baseline.fingerprint(),
            "final_experiment_fingerprint": final_experiment.fingerprint(),
            "model_id": final_experiment.model.model_id,
            "model_revision": final_experiment.model.revision,
            "profile": final_experiment.training.profile,
            "source_commit": context.source_commit,
            "source_dirty": context.source_dirty,
            "source_state_sha256": context.source_state_sha256,
            "dependency_lock_sha256": context.dependency_lock_sha256,
            "verification": {"provided": False},
            "requested_trials": search.trials,
            "finished_trials": 0,
            "failed_trials": 0,
            "status": "prepared" if prepare_only else "running",
            "search": search.to_dict(),
            "experiment": baseline.to_dict(),
            "pilot_corpus": context.pilot_metadata,
            "oracle_study": context.oracle_selection,
            "trials": [],
            "selected_trial": None,
            "selected_parameters": None,
            "selected_metrics": None,
            "selected_gates": None,
            "selection_feasible": None,
            "selected_output_passed": None,
            "failure": None,
            "artifacts": {
                "contract": "search-spec.json",
                "study": None,
                "report": "search-report.md",
                "selected_experiment": None,
                "trajectory_cache": None,
            },
        },
    )


def _prepare_search_directory(
    context: _OutputSearchContext,
    registered: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any] | None:
    output_dir = context.output_dir
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"search directory already exists: {output_dir}")
        if dict(load_json_object(output_dir / "search-spec.json")) != registered:
            raise ValueError(
                "existing output search directory has a different specification",
            )
        return dict(load_artifact(output_dir / "search.json"))
    if resume:
        raise FileNotFoundError("cannot resume a missing output search directory")
    output_dir.mkdir(parents=True)
    write_json_atomic(output_dir / "search-spec.json", registered)
    return None


def _create_output_study(
    context: _OutputSearchContext,
    contract: dict[str, Any],
    *,
    resume: bool,
) -> optuna.Study:
    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=context.search.sampler_seed),
            storage=f"sqlite:///{context.output_dir / 'optuna.db'}",
            study_name=context.search.name,
            load_if_exists=resume,
        )
        if resume:
            _reconcile_running_trials(study)
    except Exception as error:
        failed_contract = {
            **contract,
            "status": "failed",
            "operational_status": "failed",
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        _write_search_artifacts(context.output_dir, failed_contract)
        raise
    return study


def _optimize_output_study(
    context: _OutputSearchContext,
    contract: dict[str, Any],
    study: optuna.Study,
    *,
    running_status: str,
) -> None:
    search = context.search

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float(
            "learning_rate",
            search.learning_rate_min,
            search.learning_rate_max,
            log=True,
        )
        weight_decay = trial.suggest_categorical(
            "weight_decay",
            search.weight_decays,
        )
        batch_size = trial.suggest_categorical("batch_size", search.batch_sizes)
        log_event(
            "search_trial_started",
            search=search.name,
            trial=trial.number,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )
        candidate = replace(
            context.baseline,
            name=f"{context.baseline.name}-search-trial-{trial.number}",
            training=replace(
                context.baseline.training,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
            ),
        )
        try:
            result = OutputExperimentRunner(
                candidate,
                context.output_dir / f"trial-{trial.number:04d}",
                context.project_root,
                OutputRunnerOptions(
                    pilot_corpus=context.pilot_corpus,
                    trajectory_cache_directory=context.output_dir / "trajectory-cache",
                ),
            ).run()
        except OutputOracleCeilingError as error:
            _prune_infeasible_oracle(trial, error, search_name=search.name)
        except Exception as error:
            trial.set_user_attr(
                "failure",
                {"type": type(error).__name__, "message": str(error)},
            )
            log_event(
                "search_trial_failed",
                search=search.name,
                trial=trial.number,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise
        exact = float(result["output"]["exact_full_sequence_rate"])
        trial.set_user_attr("metrics", result["output"])
        trial.set_user_attr("gates", result["gates"])
        if not result["output"]["oracle_feasible"]:
            reason = "native-head oracle ceiling cannot satisfy the registered fidelity and density budget"
            _record_prune(trial, reason, search_name=search.name)
            raise optuna.TrialPruned(reason)
        log_event(
            "search_trial_completed",
            search=search.name,
            trial=trial.number,
            direct_feedback_equality=exact,
        )
        return 1.0 - exact

    def write_progress(current: optuna.Study, completed: FrozenTrial) -> None:
        _write_search_artifacts(
            context.output_dir,
            _study_summary(
                contract,
                current,
                status=running_status,
                operational_status="running",
            ),
        )
        log_event(
            "search_progress",
            search=search.name,
            completed_trial=completed.number,
            trial_state=completed.state.name.lower(),
            finished_trials=_finished_trials(current.trials),
            requested_trials=search.trials,
        )

    try:
        study.optimize(
            objective,
            n_trials=_remaining_trials(study.trials, search.trials),
            n_jobs=1,
            callbacks=(write_progress,),
            catch=(Exception,),
        )
    except Exception as error:
        failed_summary = _study_summary(
            contract,
            study,
            status="failed",
            operational_status="failed",
            failure={"type": type(error).__name__, "message": str(error)},
        )
        _write_search_artifacts(context.output_dir, failed_summary)
        raise


def _complete_output_search(
    context: _OutputSearchContext,
    contract: dict[str, Any],
    study: optuna.Study,
) -> dict[str, Any]:
    search = context.search
    completed = [trial for trial in study.trials if trial.state is TrialState.COMPLETE]
    if not completed:
        no_candidate = _study_summary(
            contract,
            study,
            status="completed_no_candidate",
            operational_status="completed",
        )
        _write_search_artifacts(context.output_dir, no_candidate)
        log_event(
            "search_completed",
            search=search.name,
            finished_trials=no_candidate["finished_trials"],
            completed_trials=no_candidate["completed_trials"],
            failed_trials=no_candidate["failed_trials"],
            selected_trial=None,
        )
        return no_candidate
    selected = _select_output_trial(
        completed,
        fallback_rule=search.fallback_rule,
    )
    if selected is None:
        raise RuntimeError("completed output search lost its selected trial")
    final_experiment = context.final_experiment
    selected_training = replace(
        final_experiment.training,
        learning_rate=float(selected.params["learning_rate"]),
        weight_decay=float(selected.params["weight_decay"]),
        batch_size=int(selected.params["batch_size"]),
    )
    completed_summary = _study_summary(
        contract,
        study,
        status="completed",
        operational_status="completed",
    )
    completed_summary["artifacts"] = {
        **completed_summary["artifacts"],
        "selected_experiment": "selected-experiment.toml",
    }
    _write_search_artifacts(context.output_dir, completed_summary)
    selection = SearchSelectionSpec(
        search_kind="output",
        artifact="search.json",
        artifact_sha256=sha256_file(context.output_dir / "search.json"),
        selected_trial=selected.number,
        search_fingerprint=search.fingerprint(),
        model_id=final_experiment.model.model_id,
        model_revision=final_experiment.model.revision,
        profile=final_experiment.training.profile,
        selected_parameters=dict(selected.params),
        feasible=bool(completed_summary["selection_feasible"]),
    )
    selected_experiment = replace(
        final_experiment,
        name=f"{final_experiment.name}-search-selected",
        evidence_scope="final",
        search_selections=(selection,),
        training=selected_training,
    ).to_toml_dict()
    write_text_atomic(
        context.output_dir / "selected-experiment.toml",
        tomli_w.dumps(selected_experiment),
    )
    log_event(
        "search_completed",
        search=search.name,
        finished_trials=completed_summary["finished_trials"],
        completed_trials=completed_summary["completed_trials"],
        failed_trials=completed_summary["failed_trials"],
        selected_trial=completed_summary["selected_trial"],
    )
    return completed_summary


def _complete_infeasible_oracle_search(
    context: _OutputSearchContext,
    contract: dict[str, Any],
) -> dict[str, Any]:
    training = context.final_experiment.training
    parameters = {
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "batch_size": training.batch_size,
    }
    reason = "the sealed native-head oracle proves the registered output targets structurally unrepresentable; no training was performed"
    summary = {
        **contract,
        "status": "completed_unsupported",
        "operational_status": "completed",
        "finished_trials": 1,
        "completed_trials": 1,
        "failed_trials": 0,
        "trials": [
            {
                "number": 0,
                "state": "COMPLETE",
                "parameters": parameters,
                "metrics": {
                    "oracle_feasible": False,
                    "training_performed": False,
                },
                "gates": {"oracle_feasible": False},
                "failure": reason,
                "final_evidence": False,
            },
        ],
        "selected_trial": 0,
        "selected_parameters": parameters,
        "selected_metrics": {
            "oracle_feasible": False,
            "training_performed": False,
        },
        "selected_gates": {"oracle_feasible": False},
        "selection_feasible": False,
        "selection_policy": "infeasible_oracle",
        "selected_output_passed": False,
        "failure": None,
        "artifacts": {
            **contract["artifacts"],
            "selected_experiment": "selected-experiment.toml",
        },
    }
    _write_search_artifacts(context.output_dir, summary)
    selection = SearchSelectionSpec(
        search_kind="output",
        artifact="search.json",
        artifact_sha256=sha256_file(context.output_dir / "search.json"),
        selected_trial=0,
        search_fingerprint=context.search.fingerprint(),
        model_id=context.final_experiment.model.model_id,
        model_revision=context.final_experiment.model.revision,
        profile=context.final_experiment.training.profile,
        selected_parameters=parameters,
        feasible=False,
    )
    selected_experiment = replace(
        context.final_experiment,
        name=f"{context.final_experiment.name}-oracle-unsupported",
        evidence_scope="final",
        search_selections=(selection,),
        training=replace(context.final_experiment.training, **parameters),
    ).to_toml_dict()
    write_text_atomic(
        context.output_dir / "selected-experiment.toml",
        tomli_w.dumps(selected_experiment),
    )
    log_event(
        "search_completed",
        search=context.search.name,
        finished_trials=1,
        completed_trials=1,
        failed_trials=0,
        selected_trial=0,
        training_performed=False,
    )
    return summary


def run_output_search(
    spec_path: Path,
    output_dir: Path,
    *,
    resume: bool,
    prepare_only: bool,
    oracle_study_artifact: Path | None = None,
) -> dict[str, Any]:
    context = _load_search_context(spec_path, output_dir, oracle_study_artifact)
    search = context.search
    log_event(
        "search_started",
        search=search.name,
        mode="output_only",
        requested_trials=search.trials,
        resume=resume,
        prepare_only=prepare_only,
    )
    registered = _registered_search(context)
    existing = _prepare_search_directory(context, registered, resume=resume)
    contract = _search_contract(context, prepare_only=prepare_only)
    if existing is None:
        _write_search_artifacts(output_dir, contract)
    if prepare_only:
        log_event("search_prepared", search=search.name, requested_trials=search.trials)
        if existing is not None:
            return existing
        return contract

    if context.oracle_selection is not None and context.oracle_selection["selection_feasible"] is False:
        return _complete_infeasible_oracle_search(context, contract)

    study = _create_output_study(context, contract, resume=resume)
    contract["artifacts"] = {
        **contract["artifacts"],
        "study": "optuna.db",
        "trajectory_cache": "trajectory-cache",
    }
    running_status = "resumed" if resume else "running"
    _write_search_artifacts(
        output_dir,
        _study_summary(
            contract,
            study,
            status=running_status,
            operational_status="running",
        ),
    )
    _optimize_output_study(
        context,
        contract,
        study,
        running_status=running_status,
    )
    return _complete_output_search(context, contract, study)
