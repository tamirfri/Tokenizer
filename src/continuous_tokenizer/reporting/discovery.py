from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, cast, final

from continuous_tokenizer.artifacts.evidence import (
    EVIDENCE_MANIFEST_FILENAME,
    load_evidence_manifest,
    verify_artifact,
)
from continuous_tokenizer.artifacts.hashing import (
    FileStatIdentity,
    directory_files,
    file_stat_identity,
)
from continuous_tokenizer.artifacts.manifest import (
    load_artifact,
    load_verified_run_manifest,
)
from continuous_tokenizer.artifacts.store import load_json_object
from continuous_tokenizer.contracts.manifest import RunManifest
from continuous_tokenizer.contracts.profiles import DIAGNOSTIC_PROFILE_NAME
from continuous_tokenizer.contracts.prospective import prospective_result_errors
from continuous_tokenizer.contracts.state_budget import StateBudgetResult
from continuous_tokenizer.reporting.shared import display_name

_TRAINING_PHASE_ORDER: Final = {
    "alignment": 0,
    "reconstruction": 1,
    "dynamic_reconstruction": 2,
}
ArtifactMode = Literal["input_only", "output_only"]
ArtifactKind = Literal[
    "run",
    "replication",
    "project",
    "performance_ablation",
    "search",
    "state_budget",
    "study",
    "deployment",
    "prospective",
]
_ARTIFACT_DISPLAY_ORDER: Final[dict[ArtifactKind, int]] = {
    "project": 0,
    "replication": 1,
    "run": 2,
    "prospective": 3,
    "study": 4,
    "search": 5,
    "deployment": 6,
    "performance_ablation": 7,
    "state_budget": 8,
}


def _artifact_mode(value: object) -> ArtifactMode:
    if value not in {"input_only", "output_only"}:
        raise ValueError("artifact mode must be input_only or output_only")
    return cast(ArtifactMode, value)


def _required_string(artifact: Mapping[str, Any], name: str) -> str:
    value = artifact.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact {name} must be a non-empty string")
    return value


def _model_id(artifact: Mapping[str, Any]) -> str:
    model = artifact.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("artifact model must be an object")
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("artifact model.id must be a non-empty string")
    return model_id


def _primary_model_label(model_ids: Sequence[str]) -> str:
    if len(model_ids) != 2:
        raise ValueError("artifact must contain two primary models")
    return " + ".join(model_ids)


def _mode_label(mode: ArtifactMode) -> str:
    return "INPUT" if mode == "input_only" else "OUTPUT"


def _validate_standalone_manifest(
    directory: Path,
    expected_kind: Literal[
        "deployment",
        "performance_ablation",
        "project",
        "replication",
        "search",
        "state_budget",
        "study",
    ],
) -> None:
    path = directory / EVIDENCE_MANIFEST_FILENAME
    if not path.is_file():
        raise ValueError(f"unsealed {expected_kind} artifact")
    verification = verify_artifact(directory)
    if verification["valid"] is not True:
        raise ValueError(f"invalid {expected_kind} evidence")
    manifest = load_evidence_manifest(path)
    if manifest["artifact_kind"] != expected_kind:
        raise ValueError(f"expected {expected_kind} evidence")


@final
@dataclass(frozen=True, slots=True)
class ReplicationArtifact:
    directory: Path
    mode: ArtifactMode
    operational_status: str
    scientific_verdict: str
    model: str
    kind: Literal["replication"] = "replication"

    @property
    def label(self) -> str:
        return f"{_mode_label(self.mode)} REPLICATION | {self.operational_status.upper()} | {self.scientific_verdict.upper()} | {self.directory.name}"


@final
@dataclass(frozen=True, slots=True)
class ProjectArtifact:
    directory: Path
    mode: ArtifactMode
    operational_status: str
    scientific_verdict: str
    model: str
    kind: Literal["project"] = "project"

    @property
    def label(self) -> str:
        return f"{_mode_label(self.mode)} PROJECT | {self.operational_status.upper()} | {self.scientific_verdict.upper()} | {self.directory.name}"


@final
@dataclass(frozen=True, slots=True)
class PerformanceAblationArtifact:
    directory: Path
    mode: ArtifactMode
    operational_status: str
    scientific_verdict: str
    model: str
    kind: Literal["performance_ablation"] = "performance_ablation"

    @property
    def label(self) -> str:
        return f"{_mode_label(self.mode)} PERFORMANCE ABLATION | OPERATIONAL/SECONDARY | {self.operational_status.upper()} | {self.directory.name}"


def discover_performance_ablation_artifacts(
    root: Path,
    *,
    _inventory: ArtifactInventory | None = None,
) -> tuple[PerformanceAblationArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts = []
    for path in _paths(root, "performance-ablation.json", _inventory):
        try:
            _validate_standalone_manifest(path.parent, "performance_ablation")
            ablation = load_artifact(path)
            baseline = ablation.get("baseline")
            if not isinstance(baseline, Mapping):
                continue
            artifacts.append(
                PerformanceAblationArtifact(
                    directory=path.parent,
                    mode=_artifact_mode(
                        load_evidence_manifest(
                            path.parent / EVIDENCE_MANIFEST_FILENAME,
                        )["mode"],
                    ),
                    operational_status=_required_string(
                        ablation,
                        "operational_status",
                    ),
                    scientific_verdict="not_final_evidence",
                    model=str(baseline["model_id"]),
                ),
            )
        except KeyError, TypeError, ValueError:
            continue
    return tuple(
        sorted(
            artifacts,
            key=lambda item: item.directory.stat().st_mtime,
            reverse=True,
        ),
    )


def discover_replication_artifacts(
    root: Path,
    *,
    _inventory: ArtifactInventory | None = None,
) -> tuple[ReplicationArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts = []
    for path in _paths(root, "replication.json", _inventory):
        try:
            _validate_standalone_manifest(path.parent, "replication")
            replication = load_artifact(path)
            if replication.get("evidence_scope") != "replication":
                continue
            artifacts.append(
                ReplicationArtifact(
                    directory=path.parent,
                    mode=_artifact_mode(replication.get("mode")),
                    operational_status=_required_string(replication, "operational_status"),
                    scientific_verdict=_required_string(replication, "scientific_verdict"),
                    model=_model_id(replication),
                )
            )
        except KeyError, TypeError, ValueError:
            continue
    return tuple(sorted(artifacts, key=lambda item: item.directory.stat().st_mtime, reverse=True))


def discover_project_artifacts(
    root: Path,
    *,
    _inventory: ArtifactInventory | None = None,
) -> tuple[ProjectArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts = []
    for path in _paths(root, "project.json", _inventory):
        try:
            _validate_standalone_manifest(path.parent, "project")
            project = load_artifact(path)
            models = project.get("models")
            if not isinstance(models, list) or len(models) != 2:
                continue
            model_names = tuple(_model_id(value) for value in models if isinstance(value, Mapping))
            artifacts.append(
                ProjectArtifact(
                    directory=path.parent,
                    mode=_artifact_mode(project.get("mode")),
                    operational_status=_required_string(project, "operational_status"),
                    scientific_verdict=_required_string(project, "scientific_verdict"),
                    model=_primary_model_label(model_names),
                )
            )
        except KeyError, TypeError, ValueError:
            continue
    return tuple(sorted(artifacts, key=lambda item: item.directory.stat().st_mtime, reverse=True))


@final
@dataclass(frozen=True, slots=True)
class ArtifactRun:
    directory: Path
    mode: ArtifactMode
    evidence_scope: str
    operational_status: str
    scientific_verdict: str
    experiment: str
    model: str
    status: Literal["passed", "failed", "diagnostic"]
    synthetic: bool
    claims_passed: bool | None
    kind: Literal["run"] = "run"

    @property
    def label(self) -> str:
        evidence = "SYNTHETIC" if self.synthetic else "REAL MODEL"
        return (
            f"{_mode_label(self.mode)} RUN | {evidence} | "
            f"{self.operational_status.upper()} | {self.scientific_verdict.upper()} | "
            f"{self.experiment} | {self.directory.name}"
        )


@final
@dataclass(frozen=True, slots=True)
class SearchArtifact:
    directory: Path
    mode: ArtifactMode
    operational_status: str
    scientific_verdict: str
    model: str
    name: str
    kind: Literal["search"] = "search"

    @property
    def label(self) -> str:
        return f"{_mode_label(self.mode)} SEARCH | {self.operational_status.upper()} | {self.scientific_verdict.upper()} | {self.name} | {self.directory.name}"


@final
@dataclass(frozen=True, slots=True)
class StudyArtifact:
    directory: Path
    mode: ArtifactMode
    operational_status: str
    scientific_verdict: str
    model: str
    name: str
    kind: Literal["study"] = "study"

    @property
    def label(self) -> str:
        return f"{_mode_label(self.mode)} STUDY | {self.operational_status.upper()} | {self.name} | {self.directory.name}"


@final
@dataclass(frozen=True, slots=True)
class DeploymentArtifact:
    directory: Path
    mode: ArtifactMode
    operational_status: str
    scientific_verdict: str
    model: str
    kind: Literal["deployment"] = "deployment"

    @property
    def label(self) -> str:
        return f"{_mode_label(self.mode)} DEPLOYMENT | {self.operational_status.upper()} | {self.scientific_verdict.upper()} | {self.directory.name}"


@final
@dataclass(frozen=True, slots=True)
class StateBudgetArtifact:
    directory: Path
    operational_status: str
    scientific_verdict: str
    model: str
    mode: Literal["cross_directional"] = "cross_directional"
    kind: Literal["state_budget"] = "state_budget"

    @property
    def label(self) -> str:
        return f"FUTURE PREREQUISITE | JOINT STATE BUDGET | {self.operational_status.upper()} | {self.scientific_verdict.upper()} | {self.directory.name}"


@final
@dataclass(frozen=True, slots=True)
class ProspectiveArtifact:
    directory: Path
    mode: ArtifactMode
    operational_status: str
    scientific_verdict: str
    model: str
    name: str
    tier: str
    kind: Literal["prospective"] = "prospective"

    @property
    def label(self) -> str:
        tier = display_name(self.tier).upper()
        return f"{_mode_label(self.mode)} PROSPECTIVE {tier} | {self.operational_status.upper()} | {self.name}"


ReportArtifact = (
    ArtifactRun
    | ReplicationArtifact
    | ProjectArtifact
    | PerformanceAblationArtifact
    | SearchArtifact
    | StudyArtifact
    | ProspectiveArtifact
    | DeploymentArtifact
    | StateBudgetArtifact
)
StandaloneArtifact = (
    ReplicationArtifact | ProjectArtifact | PerformanceAblationArtifact | SearchArtifact | StudyArtifact | DeploymentArtifact | StateBudgetArtifact
)
type ArtifactInventory = Mapping[str, tuple[Path, ...]]
type ArtifactFileIdentity = tuple[str, FileStatIdentity]


def _artifact_identity(root: Path) -> tuple[ArtifactFileIdentity, ...]:
    if not root.is_dir():
        return ()
    files = directory_files(root)
    return tuple((str(path), file_stat_identity(path)) for path in files)


def _artifact_inventory(
    root: Path,
) -> tuple[
    dict[str, tuple[Path, ...]],
    tuple[ArtifactFileIdentity, ...],
]:
    identity = _artifact_identity(root)
    return _inventory_from_identity(identity), identity


def _inventory_from_identity(
    identity: tuple[ArtifactFileIdentity, ...],
) -> dict[str, tuple[Path, ...]]:
    by_name: dict[str, list[Path]] = {}
    for path_string, _ in identity:
        path = Path(path_string)
        by_name.setdefault(path.name, []).append(path)
    return {name: tuple(paths) for name, paths in by_name.items()}


def _paths(
    root: Path,
    filename: str,
    inventory: ArtifactInventory | None,
) -> tuple[Path, ...]:
    if inventory is not None:
        return inventory.get(filename, ())
    return _artifact_inventory(root)[0].get(filename, ())


def _artifact_claims_passed(directory: Path) -> bool | None:
    try:
        tokenizer_path = directory / "tokenizer-metrics.json"
        if tokenizer_path.is_file():
            tokenizer = load_artifact(tokenizer_path)
            return bool(tokenizer["acceptance"]["overall"])
        result_path = directory / "result.json"
        if result_path.is_file():
            result = load_artifact(result_path)
            if result["mode"] == "output_only":
                return result["scientific_verdict"] == "supported"
            tokenizer = result.get("tokenizer")
            if isinstance(tokenizer, Mapping):
                return bool(tokenizer["acceptance"]["overall"])
    except KeyError, TypeError, ValueError:
        return None
    return None


def _optional_artifact(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_artifact(path)
    except TypeError, ValueError:
        return None


def discover_search_artifacts(
    root: Path,
    *,
    _inventory: ArtifactInventory | None = None,
) -> tuple[SearchArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts = []
    for path in _paths(root, "search.json", _inventory):
        try:
            _validate_standalone_manifest(path.parent, "search")
            search = load_artifact(path)
            artifacts.append(
                SearchArtifact(
                    directory=path.parent,
                    mode=_artifact_mode(search.get("mode")),
                    operational_status=_required_string(search, "operational_status"),
                    scientific_verdict=_required_string(search, "scientific_verdict"),
                    model=_required_string(search, "model_id"),
                    name=_required_string(search, "name"),
                )
            )
        except KeyError, TypeError, ValueError:
            continue
    return tuple(sorted(artifacts, key=lambda item: item.directory.stat().st_mtime, reverse=True))


def discover_study_artifacts(
    root: Path,
    *,
    _inventory: ArtifactInventory | None = None,
) -> tuple[StudyArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts = []
    for path in _paths(root, "result.json", _inventory):
        try:
            _validate_standalone_manifest(path.parent, "study")
            study = load_artifact(path)
            study_spec = study.get("study")
            name = str(study_spec.get("name")) if isinstance(study_spec, Mapping) and study_spec.get("name") else _required_string(study, "artifact_kind")
            artifacts.append(
                StudyArtifact(
                    directory=path.parent,
                    mode=_artifact_mode(study.get("mode")),
                    operational_status=_required_string(
                        study,
                        "operational_status",
                    ),
                    scientific_verdict=_required_string(
                        study,
                        "scientific_verdict",
                    ),
                    model=_model_id(study),
                    name=name,
                )
            )
        except KeyError, TypeError, ValueError:
            continue
    return tuple(
        sorted(
            artifacts,
            key=lambda item: item.directory.stat().st_mtime,
            reverse=True,
        )
    )


def _deployment_verdict(deployment: Mapping[str, Any]) -> str:
    applicability = deployment.get("applicability")
    if isinstance(applicability, Mapping) and applicability.get("applicable") is False:
        return "inapplicable"
    claimable = deployment.get("deployment_compactness_claimable")
    if claimable is None:
        return "incomplete"
    return "supported" if claimable is True else "unsupported"


def discover_deployment_artifacts(
    root: Path,
    *,
    _inventory: ArtifactInventory | None = None,
) -> tuple[DeploymentArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts = []
    for path in _paths(root, "deployment.json", _inventory):
        try:
            _validate_standalone_manifest(path.parent, "deployment")
            deployment = load_artifact(path)
            manifest = load_evidence_manifest(path.parent / EVIDENCE_MANIFEST_FILENAME)
            artifacts.append(
                DeploymentArtifact(
                    directory=path.parent,
                    mode=_artifact_mode(deployment.get("mode")),
                    operational_status=_required_string(
                        deployment,
                        "operational_status",
                    ),
                    scientific_verdict=_deployment_verdict(deployment),
                    model=str(manifest["model"]["id"]),
                )
            )
        except KeyError, TypeError, ValueError:
            continue
    return tuple(
        sorted(
            artifacts,
            key=lambda item: item.directory.stat().st_mtime,
            reverse=True,
        )
    )


def discover_state_budget_artifacts(
    root: Path,
    *,
    _inventory: ArtifactInventory | None = None,
) -> tuple[StateBudgetArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts = []
    for path in _paths(root, "joint-state-budget.json", _inventory):
        try:
            _validate_standalone_manifest(path.parent, "state_budget")
            result = StateBudgetResult.from_mapping(load_artifact(path))
            models = tuple(dict.fromkeys(row.model_id for row in result.per_seed))
            artifacts.append(
                StateBudgetArtifact(
                    directory=path.parent,
                    operational_status=result.operational_status,
                    scientific_verdict=result.verdict,
                    model=_primary_model_label(models),
                )
            )
        except KeyError, TypeError, ValueError:
            continue
    return tuple(
        sorted(
            artifacts,
            key=lambda item: item.directory.stat().st_mtime,
            reverse=True,
        )
    )


def discover_prospective_artifacts(
    root: Path,
    *,
    _inventory: ArtifactInventory | None = None,
) -> tuple[ProspectiveArtifact, ...]:
    if not root.is_dir():
        return ()
    artifacts = []
    for path in _paths(root, "prospective.json", _inventory):
        try:
            verification = verify_artifact(path.parent)
            if verification["valid"] is not True:
                continue
            result = load_artifact(path)
            if prospective_result_errors(result):
                continue
            manifest = load_evidence_manifest(
                path.parent / EVIDENCE_MANIFEST_FILENAME,
            )
            if manifest["artifact_kind"] != result["artifact_kind"]:
                continue
            artifacts.append(
                ProspectiveArtifact(
                    directory=path.parent,
                    mode=_artifact_mode(result.get("mode")),
                    operational_status=_required_string(
                        result,
                        "operational_status",
                    ),
                    scientific_verdict=_required_string(
                        result,
                        "scientific_verdict",
                    ),
                    model=str(manifest["model"]["id"]),
                    name=_required_string(result, "name"),
                    tier=_required_string(result, "tier"),
                ),
            )
        except KeyError, TypeError, ValueError:
            continue
    return tuple(
        sorted(
            artifacts,
            key=lambda item: item.directory.stat().st_mtime,
            reverse=True,
        ),
    )


def artifact_profile(artifact: Mapping[str, Any]) -> str | None:
    training = artifact.get("training")
    if isinstance(training, Mapping) and isinstance(training.get("profile"), str):
        return str(training["profile"])
    return None


def artifact_directory_profile(directory: Path) -> str | None:
    experiment_path = directory / "experiment.json"
    if not experiment_path.is_file():
        return None
    return artifact_profile(load_json_object(experiment_path))


def _discover_standalone_artifacts(
    root: Path,
    inventory: ArtifactInventory,
) -> tuple[StandaloneArtifact, ...]:
    return (
        *discover_project_artifacts(root, _inventory=inventory),
        *discover_replication_artifacts(root, _inventory=inventory),
        *discover_performance_ablation_artifacts(root, _inventory=inventory),
        *discover_search_artifacts(root, _inventory=inventory),
        *discover_study_artifacts(root, _inventory=inventory),
        *discover_deployment_artifacts(root, _inventory=inventory),
        *discover_state_budget_artifacts(root, _inventory=inventory),
    )


def _verified_run_manifest(path: Path) -> RunManifest:
    verification = verify_artifact(path.parent)
    if verification["valid"] is not True:
        raise ValueError("invalid run evidence")
    return load_verified_run_manifest(path)


def _discover_artifact_runs(
    root: Path,
    inventory: ArtifactInventory | None = None,
) -> tuple[ArtifactRun, ...]:
    if not root.is_dir():
        return ()
    runs: list[ArtifactRun] = []
    for path in _paths(root, "manifest-final.json", inventory):
        try:
            manifest = _verified_run_manifest(path)
            profile = artifact_directory_profile(path.parent)
        except KeyError, TypeError, ValueError:
            continue
        if manifest.status == "running":
            continue
        result = _optional_artifact(path.parent / "result.json")
        if result is not None:
            try:
                if _artifact_mode(result.get("mode")) == manifest.mode:
                    evidence_scope = _required_string(result, "evidence_scope")
                    operational_status = _required_string(result, "operational_status")
                    scientific_verdict = _required_string(result, "scientific_verdict")
                else:
                    result = None
            except TypeError, ValueError:
                result = None
        if result is None and manifest.status == "passed":
            continue
        if result is None:
            operational_status = "failed"
            evidence_scope = "diagnostic"
            scientific_verdict = "not_evaluated"
        synthetic = evidence_scope == "synthetic"
        diagnostic = profile == DIAGNOSTIC_PROFILE_NAME and not synthetic
        runs.append(
            ArtifactRun(
                directory=path.parent,
                mode=_artifact_mode(manifest.mode),
                evidence_scope=evidence_scope,
                operational_status=operational_status,
                scientific_verdict=scientific_verdict,
                experiment=manifest.experiment_name,
                model=manifest.model_id,
                status="diagnostic" if diagnostic else manifest.status,
                synthetic=synthetic,
                claims_passed=None if diagnostic else _artifact_claims_passed(path.parent),
            )
        )
    return tuple(
        sorted(
            runs,
            key=lambda run: (
                run.claims_passed is True,
                not run.synthetic,
                run.status == "passed",
                run.directory.stat().st_mtime,
            ),
            reverse=True,
        )
    )


def discover_artifact_runs(root: Path) -> tuple[ArtifactRun, ...]:
    return _discover_artifact_runs(root)


def _discover_report_artifacts(
    root: Path,
    inventory: ArtifactInventory,
) -> tuple[ReportArtifact, ...]:
    standalone = _discover_standalone_artifacts(root, inventory)
    artifacts: tuple[ReportArtifact, ...] = (
        *standalone,
        *discover_prospective_artifacts(root, _inventory=inventory),
        *_discover_artifact_runs(root, inventory),
    )
    return tuple(
        sorted(
            artifacts,
            key=lambda artifact: (
                _ARTIFACT_DISPLAY_ORDER[artifact.kind],
                -artifact.directory.stat().st_mtime,
            ),
        )
    )


@lru_cache(maxsize=32)
def _cached_report_artifacts(
    root: str,
    identity: tuple[ArtifactFileIdentity, ...],
) -> tuple[ReportArtifact, ...]:
    return _discover_report_artifacts(
        Path(root),
        _inventory_from_identity(identity),
    )


def discover_report_artifacts(root: Path) -> tuple[ReportArtifact, ...]:
    absolute = root.absolute()
    return _cached_report_artifacts(str(absolute), _artifact_identity(absolute))


def artifact_index(artifacts: Sequence[Any], requested: str | None) -> int:
    if requested:
        for index, artifact in enumerate(artifacts):
            directory = artifact.directory
            if directory.name == requested or str(directory) == requested:
                return index
    return 0


@lru_cache(maxsize=256)
def _sealed_artifact_paths(
    directory: str,
    manifest_path: str,
    _identity: FileStatIdentity,
) -> frozenset[Path]:
    root = Path(directory)
    path = Path(manifest_path)
    if path.name == EVIDENCE_MANIFEST_FILENAME:
        manifest = load_evidence_manifest(path)
        return frozenset(
            (root / str(entry["locator"])).resolve()
            for value in cast(Mapping[str, Any], manifest["artifacts"]).values()
            if isinstance(value, Mapping)
            for entry in (cast(Mapping[str, Any], value),)
        )
    run = RunManifest.load(path)
    return frozenset((root / relative).resolve() for relative in run.artifacts.values())


def sealed_artifact_paths(directory: Path) -> frozenset[Path]:
    evidence_path = directory / EVIDENCE_MANIFEST_FILENAME
    if evidence_path.is_file():
        return _sealed_artifact_paths(
            str(directory.resolve()),
            str(evidence_path.resolve()),
            file_stat_identity(evidence_path),
        )
    manifest_path = directory / "manifest-final.json"
    return _sealed_artifact_paths(
        str(directory.resolve()),
        str(manifest_path.resolve()),
        file_stat_identity(manifest_path),
    )


def acceptance_rows(result: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    tokenizer = result.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        return ()
    gates = tokenizer.get("gates")
    if not isinstance(gates, Mapping):
        return ()
    return tuple(
        {
            "gate": display_name(name),
            "measured": gate["measured"],
            "operator": gate["operator"],
            "threshold": gate["threshold"],
            "passed": gate["passed"],
        }
        for name, gate in gates.items()
        if isinstance(gate, Mapping)
    )


def load_training_progress(directory: Path) -> tuple[dict[str, Any], ...]:
    progress_directory = directory / "checkpoints" / "progress"
    if not progress_directory.is_dir():
        return ()
    progress = []
    for path in progress_directory.glob("*.json"):
        item = dict(load_json_object(path))
        if "epoch" in item and "phase" in item:
            progress.append(item)
    return tuple(
        sorted(
            progress,
            key=lambda item: (
                _TRAINING_PHASE_ORDER.get(str(item["phase"]), len(_TRAINING_PHASE_ORDER)),
                int(item["epoch"]),
            ),
        )
    )


def training_progress_rows(
    progress: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for item in progress:
        metrics = item.get("embedding_metrics", {})
        rows.append(
            {
                "phase": str(item["phase"]),
                "epoch": int(item["epoch"]),
                "training_loss": item.get("training_loss"),
                "normalized_rmse": metrics.get("normalized_rmse"),
                "cosine_p01": metrics.get("cosine_similarity_p01"),
                "cosine_p50": metrics.get("cosine_similarity_p50"),
                "source_dtype_equal": metrics.get("exact_fraction"),
                "reconstruction": metrics.get("reconstruction_fraction"),
                "native_tokens_per_continuous_token": item.get("native_tokens_per_continuous_token"),
                "selected": item.get("selected"),
            }
        )
    return tuple(rows)
