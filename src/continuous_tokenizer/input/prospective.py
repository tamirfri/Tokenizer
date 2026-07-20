from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast, final

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    write_evidence_manifest,
)
from continuous_tokenizer.artifacts.hashing import (
    installed_distribution_identity,
    sha256_file,
)
from continuous_tokenizer.artifacts.manifest import load_verified_run_manifest
from continuous_tokenizer.artifacts.source import find_project_root
from continuous_tokenizer.artifacts.store import (
    load_json_object,
    write_json_atomic,
    write_text_atomic,
)
from continuous_tokenizer.campaigns.dispatch import create_experiment_runner
from continuous_tokenizer.campaigns.lifecycle import (
    ProspectiveBudgetExhaustedError,
)
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import InputEvaluationSpec, InputTrainingSpec
from continuous_tokenizer.contracts.output import OutputEvaluationSpec, OutputTrainingSpec
from continuous_tokenizer.contracts.prospective import (
    PROSPECTIVE_NON_FINAL_FLAGS,
    PROSPECTIVE_REPORT_FILENAME,
    PROSPECTIVE_RESULT_FILENAME,
    PROSPECTIVE_RESUME_FILENAME,
    CandidateOutcome,
    ProspectiveSpec,
    ProspectiveTier,
    futility_stages,
    prospective_result_errors,
    prospective_stage_records,
    select_smallest_candidate,
)
from continuous_tokenizer.contracts.prospective_subset import (
    PROSPECTIVE_INPUT_SUBSET_ALGORITHM,
)
from continuous_tokenizer.input.studies import RegisteredVocabularySubsetRequest
from continuous_tokenizer.reporting.prospective_markdown import prospective_markdown

type CampaignExecutor = Callable[[ExperimentSpec, Path, bool], Mapping[str, Any]]
type ProspectiveSettings = InputTrainingSpec | InputEvaluationSpec | OutputTrainingSpec | OutputEvaluationSpec


@final
@dataclass(frozen=True, slots=True)
class ProspectiveExecutionPolicy:
    tier: ProspectiveTier
    expected_wall_seconds: int
    maximum_wall_seconds: int
    stop_boundary: str
    futility_enabled: bool
    started: float
    deadline: float

    @classmethod
    def from_spec(
        cls,
        spec: ProspectiveSpec,
        *,
        started: float | None = None,
    ) -> ProspectiveExecutionPolicy:
        monotonic_start = perf_counter() if started is None else started
        return cls(
            tier=spec.tier,
            expected_wall_seconds=spec.wall_clock.expected_seconds,
            maximum_wall_seconds=spec.wall_clock.maximum_seconds,
            stop_boundary=spec.wall_clock.stop_boundary,
            futility_enabled=spec.tier != "final_evidence",
            started=monotonic_start,
            deadline=monotonic_start + spec.wall_clock.maximum_seconds,
        )

    def __post_init__(self) -> None:
        if (
            self.expected_wall_seconds <= 0
            or self.maximum_wall_seconds < self.expected_wall_seconds
            or self.stop_boundary != "epoch_or_stage"
            or self.deadline != self.started + self.maximum_wall_seconds
        ):
            raise ValueError("prospective execution policy is invalid")

    @property
    def wall_clock_enabled(self) -> bool:
        return self.tier != "final_evidence"

    def enforce_boundary(self, boundary: str) -> None:
        if not self.wall_clock_enabled:
            return
        now = perf_counter()
        if now >= self.deadline:
            raise ProspectiveBudgetExhaustedError(
                boundary,
                now - self.started,
            )

    def execution_fingerprint(self, experiment_fingerprint: str) -> str:
        if self.tier == "final_evidence":
            return experiment_fingerprint
        from continuous_tokenizer.contracts.parsing import mapping_fingerprint

        return mapping_fingerprint(
            {
                "experiment_fingerprint": experiment_fingerprint,
                "tier": self.tier,
                "expected_wall_seconds": self.expected_wall_seconds,
                "maximum_wall_seconds": self.maximum_wall_seconds,
                "stop_boundary": self.stop_boundary,
                "futility_enabled": self.futility_enabled,
            },
        )


def _replace_known(
    value: ProspectiveSettings,
    updates: Mapping[str, Any],
) -> ProspectiveSettings:
    fields = value.__dataclass_fields__
    return replace(value, **{name: item for name, item in updates.items() if name in fields})


def _experiment_work_units(experiment: ExperimentSpec) -> dict[str, int]:
    training = experiment.training
    evaluation = experiment.evaluation
    if isinstance(training, InputTrainingSpec) and isinstance(
        evaluation,
        InputEvaluationSpec,
    ):
        return {
            "behavior_samples": evaluation.samples,
            "distillation_windows": training.distillation_windows,
            "generation_samples": evaluation.generation_samples,
            "reconstruction_samples": training.reconstruction_samples,
            "validation_bytes": training.validation_bytes,
            "vocabulary_epochs": training.vocabulary_epochs,
            "vocabulary_rows": experiment.runtime.corpus_max_rows,
        }
    if isinstance(training, OutputTrainingSpec) and isinstance(
        evaluation,
        OutputEvaluationSpec,
    ):
        return {
            "behavior_samples": evaluation.samples,
            "distillation_windows": 0,
            "generation_samples": 0,
            "reconstruction_samples": 0,
            "validation_bytes": evaluation.max_output_bytes,
            "vocabulary_epochs": training.epochs,
            "vocabulary_rows": experiment.runtime.corpus_max_rows,
        }
    raise TypeError("prospective experiment settings do not match their mode")


def _bounded_experiment(spec: ProspectiveSpec) -> ExperimentSpec:
    experiment = spec.load_experiment()
    if spec.tier in {"mechanism_smoke", "final_evidence"}:
        if spec.tier == "mechanism_smoke" and _experiment_work_units(experiment) != spec.wall_clock.work_units:
            raise ValueError(
                "prospective work units differ from the mechanism experiment",
            )
        return experiment
    design = spec.design
    if isinstance(experiment.training, InputTrainingSpec):
        updates = {
            "vocabulary_epochs": design["vocabulary_epochs"],
            "reconstruction_epochs": design.get("reconstruction_epochs", 1),
            "reconstruction_samples": design.get("reconstruction_samples", 512),
            "validation_bytes": design.get("validation_bytes", 1024),
            "patience": design.get("patience", experiment.training.patience),
            "distillation_windows": design.get(
                "distillation_windows",
                experiment.training.distillation_windows,
            ),
        }
        training = cast(InputTrainingSpec, _replace_known(experiment.training, updates))
        evaluation = cast(
            InputEvaluationSpec,
            _replace_known(
                experiment.evaluation,
                {
                    "samples": design.get("behavior_samples", 4),
                    "generation_samples": design.get("generation_samples", 1),
                },
            ),
        )
    else:
        output_evaluation = cast(OutputEvaluationSpec, experiment.evaluation)
        training = cast(
            OutputTrainingSpec,
            _replace_known(
                experiment.training,
                {"epochs": design.get("vocabulary_epochs", 3)},
            ),
        )
        evaluation = cast(
            OutputEvaluationSpec,
            _replace_known(
                experiment.evaluation,
                {
                    "samples": design.get("behavior_samples", 2),
                    "max_output_bytes": design.get(
                        "validation_bytes",
                        output_evaluation.max_output_bytes,
                    ),
                },
            ),
        )
    runtime = replace(
        experiment.runtime,
        corpus_max_rows=int(design.get("vocabulary_rows", experiment.runtime.corpus_max_rows)),
    )
    bounded = replace(
        experiment,
        training=training,
        evaluation=evaluation,
        runtime=runtime,
        evidence_scope="candidate",
    )
    if _experiment_work_units(bounded) != spec.wall_clock.work_units:
        raise ValueError(
            "prospective work units differ from the bounded child experiment",
        )
    return bounded


def _gate_passed(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, Mapping):
        return None
    passed = value.get("passed")
    return passed if isinstance(passed, bool) else None


def _gate_map(result: Mapping[str, Any]) -> dict[str, bool | None]:
    direct = result.get("gates")
    tokenizer = result.get("tokenizer")
    nested = tokenizer.get("gates") if isinstance(tokenizer, Mapping) else None
    return {
        **({str(name): _gate_passed(value) for name, value in nested.items()} if isinstance(nested, Mapping) else {}),
        **({str(name): _gate_passed(value) for name, value in direct.items()} if isinstance(direct, Mapping) else {}),
    }


def _all_named(
    gates: Mapping[str, bool | None],
    fragments: Sequence[str],
) -> bool | None:
    selected = [passed for name, passed in gates.items() if any(fragment in name for fragment in fragments)]
    if any(passed is False for passed in selected):
        return False
    if not selected or any(passed is None for passed in selected):
        return None
    return True


def _outcome(name: str, kind: str, result: Mapping[str, Any], exhausted: bool) -> CandidateOutcome:
    gates = _gate_map(result)
    operational = result.get("operational_status") == "completed"
    execution = result.get("prospective_execution")
    child_exhausted = isinstance(execution, Mapping) and execution.get("budget_exhausted") is True
    return CandidateOutcome(
        name=name,
        kind=cast(Any, kind),
        operational_passed=operational,
        invariant_passed=operational and result.get("invariant_failure") is not True,
        exactness_passed=_all_named(
            gates,
            (
                "exact",
                "round_trip",
                "compatibility",
                "alignment",
                "direct_feedback",
                "invalid",
                "termination",
            ),
        ),
        density_passed=_all_named(gates, ("density", "tokens_per_", "macro_step")),
        behavior_passed=_all_named(
            gates,
            ("behavior", "kl", "nll", "top1", "generation", "rollout"),
        ),
        compactness_passed=_all_named(
            gates,
            ("compact", "state_ratio"),
        ),
        alignment_passed=_all_named(gates, ("alignment", "rmse", "cosine")),
        budget_exhausted=exhausted or child_exhausted,
    )


def _stages(outcome: CandidateOutcome) -> list[dict[str, str]]:
    return prospective_stage_records(futility_stages(outcome))


def _default_executor(
    verification: Path | None,
    spec: ProspectiveSpec,
    *,
    started: float | None = None,
) -> CampaignExecutor:
    subset_request = _prospective_input_subset_request(spec)
    execution_policy = ProspectiveExecutionPolicy.from_spec(
        spec,
        started=started,
    )

    def execute(experiment: ExperimentSpec, output_dir: Path, resume: bool) -> Mapping[str, Any]:
        return create_experiment_runner(
            experiment,
            output_dir,
            find_project_root(spec.path),
            verification,
            resume=resume,
            prospective_input_subset=subset_request,
            prospective_execution_policy=execution_policy,
        ).run()

    return execute


def _prospective_input_subset_request(
    spec: ProspectiveSpec,
) -> RegisteredVocabularySubsetRequest | None:
    if spec.mode != "input_only" or spec.tier not in {
        "feasibility_screen",
        "candidate_selection",
    }:
        return None
    return RegisteredVocabularySubsetRequest(
        requested_rows=int(spec.design["vocabulary_rows"]),
        subset_seed=int(spec.design["subset_seed"]),
        subset_sha256=str(spec.design["subset_sha256"]),
        algorithm=PROSPECTIVE_INPUT_SUBSET_ALGORITHM,
        work_units=tuple(sorted(spec.wall_clock.work_units.items())),
    )


def _resume_state(
    spec: ProspectiveSpec,
    output_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    path = output_dir / PROSPECTIVE_RESUME_FILENAME
    if not resume:
        return {"schema_version": 1, "spec_fingerprint": spec.fingerprint(), "completed": []}
    state = dict(load_json_object(path))
    if set(state) != {"schema_version", "spec_fingerprint", "completed"}:
        raise ValueError("prospective resume state is not canonical")
    if state["schema_version"] != 1 or state["spec_fingerprint"] != spec.fingerprint():
        raise ValueError("prospective resume state differs from its registered wrapper")
    if not isinstance(state["completed"], list):
        raise ValueError("prospective resume completed entries must be an array")
    return state


def _save_resume(output_dir: Path, state: Mapping[str, Any]) -> None:
    write_json_atomic(output_dir / PROSPECTIVE_RESUME_FILENAME, state)


def _selection_result(
    spec: ProspectiveSpec,
    output_dir: Path,
    *,
    resume: bool,
    execute: CampaignExecutor,
    started: float,
) -> tuple[list[dict[str, Any]], list[CandidateOutcome], dict[str, Any]]:
    state = _resume_state(spec, output_dir, resume=resume)
    completed = cast(list[str], state["completed"])
    rows: list[dict[str, Any]] = []
    outcomes: list[CandidateOutcome] = []
    for candidate in spec.candidates():
        trial_dir = output_dir / "trials" / candidate.name
        result_path = trial_dir / "result.json"
        if candidate.name in completed:
            child = load_json_object(result_path)
        else:
            elapsed = perf_counter() - started
            if spec.wall_clock.exhausted_at_boundary(elapsed, at_boundary=True):
                break
            base = _bounded_experiment(spec)
            training = _replace_known(base.training, candidate.parameters)
            child = execute(replace(base, training=training), trial_dir, False)
            completed.append(candidate.name)
            _save_resume(output_dir, {**state, "completed": completed})
        exhausted = spec.wall_clock.exhausted_at_boundary(
            perf_counter() - started,
            at_boundary=True,
        )
        outcome = _outcome(candidate.name, candidate.kind, child, exhausted)
        outcomes.append(outcome)
        rows.append(
            {
                "name": candidate.name,
                "kind": candidate.kind,
                "parameters": dict(candidate.parameters),
                "outcome": asdict(outcome),
                "stages": _stages(outcome),
                "child_result": str(result_path.relative_to(output_dir)),
            },
        )
        if outcome.budget_exhausted:
            break
        if not outcome.operational_passed or not outcome.invariant_passed:
            break
    selected, feasible = select_smallest_candidate(spec.candidates(), outcomes)
    selection = {
        "selection_feasible": feasible,
        "selected_candidate": selected.name if selected is not None else None,
        "selected_strategy": (selected.parameters.get("strategy") if selected is not None else None),
        "selected_configuration": (dict(selected.parameters) if selected is not None else None),
        "alignment": [asdict(outcome) for outcome in outcomes if outcome.kind == "alignment"],
        "validation_only": True,
        "final_test_loaded": False,
    }
    return rows, outcomes, selection


def _seal(
    spec: ProspectiveSpec,
    output_dir: Path,
    result_path: Path,
    report_path: Path,
    child_directories: Sequence[Path],
) -> None:
    project_root = find_project_root(spec.path)
    if child_directories:
        child = load_verified_run_manifest(
            child_directories[0] / "manifest-final.json",
        )
        source_commit = child.source_commit
        source_dirty = child.source_dirty
        source_state_sha256 = child.source_state_sha256
        dependency_lock_sha256 = child.dependency_lock_sha256
        source_assets = child.source_assets
        verification = child.verification
        model_id = child.model_id
        model_revision = child.model_revision
    else:
        from continuous_tokenizer.artifacts.source import source_state

        source_commit, source_dirty, source_state_sha256 = source_state(
            project_root,
        )
        dependency_lock_sha256 = sha256_file(project_root / "uv.lock")
        source_assets = {}
        verification = {
            "provided": False,
            "all_passed": False,
            "checks": {},
        }
        model = spec.load_final_reference().model
        model_id = model.model_id
        model_revision = model.revision
    write_evidence_manifest(
        output_dir,
        EvidenceManifest(
            artifact_kind=cast(Any, spec.artifact_kind),
            mode=spec.mode,
            status=str(load_json_object(result_path)["operational_status"]),
            identity=EvidenceIdentity(
                source_commit=source_commit,
                source_dirty=source_dirty,
                source_state_sha256=source_state_sha256,
                dependency_lock_sha256=dependency_lock_sha256,
                installed_package=installed_distribution_identity(
                    "continuous-byte-tokenizer",
                ),
                claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
                source_assets=source_assets,
                verification=verification,
                model_id=model_id,
                model_revision=model_revision,
            ),
            parents={f"trial_{index}": directory / "manifest-final.json" for index, directory in enumerate(child_directories)},
            inputs={"spec": spec.path},
            artifacts={
                "result": result_path,
                "report": report_path,
            },
        ),
    )


def run_prospective(
    spec_path: Path,
    output_dir: Path,
    verification: Path | None,
    *,
    resume: bool = False,
    execute: CampaignExecutor | None = None,
) -> dict[str, Any]:
    spec = ProspectiveSpec.load(spec_path)
    if spec.tier == "final_evidence":
        return dict(
            (execute or _default_executor(verification, spec))(
                spec.load_experiment(),
                output_dir,
                resume,
            ),
        )
    if output_dir.exists() and not resume:
        raise FileExistsError(f"prospective output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=resume)
    started = perf_counter()
    executor = execute or _default_executor(
        verification,
        spec,
        started=started,
    )
    child_directories: list[Path] = []
    if spec.tier == "candidate_selection":
        rows, outcomes, selection = _selection_result(
            spec,
            output_dir,
            resume=resume,
            execute=executor,
            started=started,
        )
        exhausted = any(outcome.budget_exhausted for outcome in outcomes)
        operational_failed = any(not outcome.operational_passed or not outcome.invariant_passed for outcome in outcomes)
        completed_names = {str(row["name"]) for row in rows}
        stages = [
            {
                "name": candidate.name,
                "status": (
                    "stopped_budget"
                    if candidate.name in completed_names and any(outcome.name == candidate.name and outcome.budget_exhausted for outcome in outcomes)
                    else "completed"
                    if candidate.name in completed_names
                    else "not_run_budget"
                    if exhausted
                    else "not_run_operational_failure"
                    if operational_failed
                    else "not_run_futility"
                ),
            }
            for candidate in spec.candidates()
        ]
        verdict = "not_evaluated" if operational_failed else ("supported" if selection["selection_feasible"] and not exhausted else "unsupported")
        child_directories.extend(output_dir / "trials" / str(row["name"]) for row in rows)
    else:
        child_dir = output_dir / "execution"
        child = executor(_bounded_experiment(spec), child_dir, resume)
        child_directories.append(child_dir)
        exhausted = spec.wall_clock.exhausted_at_boundary(
            perf_counter() - started,
            at_boundary=True,
        )
        outcome = _outcome(spec.name, "efficiency", child, exhausted)
        exhausted = outcome.budget_exhausted
        stages = _stages(outcome)
        rows = [{"name": spec.name, "outcome": asdict(outcome)}]
        selection = {
            "selection_feasible": False,
            "selected_candidate": None,
            "selected_strategy": None,
            "selected_configuration": None,
            "alignment": [],
            "validation_only": True,
            "final_test_loaded": False,
        }
        operational_failed = not outcome.operational_passed or not outcome.invariant_passed
        verdict = (
            "not_evaluated"
            if operational_failed
            else ("supported" if all(stage["status"] == "passed" for stage in stages) and not exhausted else "unsupported")
        )
    elapsed = perf_counter() - started
    result = {
        "schema_version": 1,
        "artifact_kind": spec.artifact_kind,
        "tier": spec.tier,
        "name": spec.name,
        "mode": spec.mode,
        "operational_status": "failed" if operational_failed else "completed",
        "scientific_verdict": verdict,
        "spec_fingerprint": spec.fingerprint(),
        "budget_exhausted": exhausted,
        "wall_clock": {
            "elapsed_seconds": elapsed,
            "expected_seconds": spec.wall_clock.expected_seconds,
            "maximum_seconds": spec.wall_clock.maximum_seconds,
            "stop_boundary": spec.wall_clock.stop_boundary,
            "calibration": {
                "locator": spec.wall_clock.calibration.locator,
                "sha256": spec.wall_clock.calibration.sha256,
            },
            "work_units": dict(spec.wall_clock.work_units),
        },
        "stages": stages,
        "selection": {**selection, "trials": rows},
        **PROSPECTIVE_NON_FINAL_FLAGS,
    }
    if errors := prospective_result_errors(result):
        raise ValueError("; ".join(errors))
    result_path = output_dir / PROSPECTIVE_RESULT_FILENAME
    report_path = output_dir / PROSPECTIVE_REPORT_FILENAME
    write_json_atomic(result_path, result)
    write_text_atomic(report_path, prospective_markdown(result))
    _seal(spec, output_dir, result_path, report_path, child_directories)
    return result
