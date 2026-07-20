from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.contracts.claims import CURRENT_DESIGN_NOTICE
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.output import (
    OutputEvaluationSpec,
    OutputGateSpec,
)
from continuous_tokenizer.contracts.output_study import (
    OUTPUT_ORACLE_SELECTION_RULE,
    OutputOracleStudySpec,
)
from continuous_tokenizer.data.corpus import load_corpus_documents
from continuous_tokenizer.output.corpora import select_output_documents
from continuous_tokenizer.output.generation import output_stop_control_ids
from continuous_tokenizer.output.trajectory_cache import (
    OutputTrajectoryOptions,
    build_prepared_output_corpus,
    native_head_oracle_ceilings,
    oracle_ceiling_passes_gates,
)
from continuous_tokenizer.runtime.device import declared_device
from continuous_tokenizer.runtime.environment import dependency_environment


def _metric(
    metrics: Mapping[str, float | int | bool | None],
    name: str,
) -> float:
    value = metrics[name]
    return float(value) if isinstance(value, int | float) else float("-inf")


def _prompt_sequences(
    assets: ModelAssets,
    experiment: ExperimentSpec,
    evaluation: OutputEvaluationSpec,
) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any]]:
    corpus = evaluation.oracle_validation_corpus
    documents = load_corpus_documents(
        corpus.split,
        dataset_id=experiment.dataset.dataset_id,
        config=experiment.dataset.config,
        revision=experiment.dataset.revision,
        max_rows=experiment.runtime.corpus_max_rows,
    )
    selected = select_output_documents(
        documents,
        count=evaluation.samples,
        seed=corpus.seed,
    )
    sequences = tuple(tuple(assets.tokenizer.encode(document.decode("utf-8"), add_special_tokens=False)) for document in selected.documents)
    return sequences, {
        "split": corpus.split,
        "seed": corpus.seed,
        "documents": len(selected.documents),
        "sha256": selected.sha256,
    }


def _selection(
    ceilings: Mapping[str, Mapping[str, float | int | bool | None]],
    gates: OutputGateSpec,
) -> dict[str, Any]:
    feasible = [int(span) for span, metrics in ceilings.items() if oracle_ceiling_passes_gates(metrics, gates)]
    if feasible:
        selected = max(feasible)
        policy = "largest_feasible_span"
    else:
        selected = max(
            (int(span) for span in ceilings),
            key=lambda span: (
                _metric(
                    ceilings[str(span)],
                    "exact_native_sequence_rate_ceiling",
                ),
                _metric(
                    ceilings[str(span)],
                    "native_tokens_per_attempted_macro_step_ceiling",
                ),
                span,
            ),
        )
        policy = "best_registered_oracle_ceiling"
    return {
        "selection_rule": OUTPUT_ORACLE_SELECTION_RULE,
        "selected_max_span": selected,
        "max_span": selected,
        "selection_feasible": bool(feasible),
        "selection_policy": policy,
        "feasible_spans": sorted(feasible),
    }


def _registered_study(
    study_path: Path,
    study: OutputOracleStudySpec,
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


def run_output_oracle_study(
    study_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    study = OutputOracleStudySpec.load(study_path)
    experiment = study.load_experiment(study_path)
    evaluation = experiment.evaluation
    gates = experiment.gates
    if not isinstance(evaluation, OutputEvaluationSpec) or not isinstance(gates, OutputGateSpec):
        raise ValueError("output oracle study requires output-only settings")
    registered = _registered_study(study_path, study, experiment)
    run, completed = _prepare_study_directory(
        output_dir,
        registered,
        resume=resume,
    )
    if completed is not None:
        return completed
    assets = load_model_assets(experiment.model.model_id, experiment.model.revision)
    if assets.revision != experiment.model.revision:
        raise ValueError("resolved model revision differs from the registered study")
    device = declared_device(experiment.device)
    model = load_frozen_causal_lm(assets, device)
    stop_control_ids = output_stop_control_ids(assets.tokenizer, assets.vocabulary)
    prompt_sequences, corpus_metadata = _prompt_sequences(
        assets,
        experiment,
        evaluation,
    )
    corpus = build_prepared_output_corpus(
        FrozenBackbone(model),
        assets.vocabulary,
        prompt_sequences,
        OutputTrajectoryOptions(
            max_span=max(assets.vocabulary.max_token_bytes, *study.span_limits),
            stop_control_ids=stop_control_ids,
            max_native_tokens=evaluation.max_macro_steps,
            max_bytes=evaluation.max_output_bytes,
        ),
    )
    ceilings = native_head_oracle_ceilings(
        corpus,
        assets.vocabulary,
        span_limits=study.span_limits,
    )
    selection = _selection(ceilings, gates)
    study_values = study.to_dict()
    study_fingerprint = study.fingerprint()
    experiment_fingerprint = experiment.fingerprint()
    result = json_compatible_object(
        {
            "artifact_kind": "output_oracle_study",
            "mode": "output_only",
            "operational_status": "completed",
            "evidence_scope": "selection",
            "scientific_verdict": "not_applicable_selection",
            "study": study_values,
            "study_fingerprint": study_fingerprint,
            "experiment_fingerprint": experiment_fingerprint,
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
            "native_head_oracle_ceilings": ceilings,
            "oracle_validation_corpus": corpus_metadata,
            "selection": selection,
            "acceptance_gates": asdict(gates),
            "environment": dependency_environment(device),
            "training_performed": False,
            "final_evidence": False,
        },
    )
    run.write_json("result.json", result)
    run.write_text(
        "study-report.md",
        "\n".join(
            (
                "# Output Native-Head Oracle Study",
                "",
                "- Evidence scope: `selection`",
                "- Training performed: `false`",
                f"- Current evidence boundary: {CURRENT_DESIGN_NOTICE}",
                "- Span ladder: `1, 2, 4, 8`; this reduced non-final study has lower research power.",
                f"- Selected maximum span: `{selection['selected_max_span']}`",
                "",
            )
        ),
    )
    artifacts = {
        "contract": "study-contract.json",
        "result": "result.json",
        "report": "study-report.md",
    }
    run.write_json(
        "manifest-final.json",
        {
            "artifact_kind": "output_oracle_study",
            "mode": "output_only",
            "status": "completed",
            "operational_status": "completed",
            "evidence_scope": "selection",
            "scientific_verdict": "not_applicable_selection",
            "study_fingerprint": study_fingerprint,
            "experiment_fingerprint": experiment_fingerprint,
            "model_id": assets.model_id,
            "model_revision": assets.revision,
            "source_commit": registered["source_commit"],
            "source_dirty": registered["source_dirty"],
            "source_state_sha256": registered["source_state_sha256"],
            "dependency_lock_sha256": registered["dependency_lock_sha256"],
            "installed_package": registered["installed_package"],
            "verification": {"provided": False},
            "artifacts": artifacts,
            "artifact_hashes": {name: sha256_path(run.root / relative) for name, relative in artifacts.items()},
        },
    )
    return result
