from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _search_value(value: Any, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _selected_search_trial(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    selected = summary.get("selected_trial")
    return next(
        (trial for trial in summary.get("trials", ()) if isinstance(trial, Mapping) and trial.get("number") == selected),
        None,
    )


def search_report(summary: Mapping[str, Any]) -> str:
    output_only = summary["mode"] == "output_only"
    title = f"Output Tokenizer Search: {summary['name']}" if output_only else f"Vocabulary Alignment Search: {summary['name']}"
    selected = summary.get("selected_trial")
    selected_trial = _selected_search_trial(summary)
    search_contract = summary.get("search", {})
    sampler_seed = summary.get(
        "sampler_seed",
        search_contract.get("sampler_seed") if isinstance(search_contract, Mapping) else None,
    )
    lines = [
        f"# {title}",
        "",
        "**SEARCH EVIDENCE ONLY — NO TRIAL OR SELECTION IS FINAL MODEL EVIDENCE.**",
        "",
        "## Search Status",
        "",
        f"- Status: `{summary['status']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Evidence scope: `{summary['evidence_scope']}`",
        f"- Operational status: `{summary['operational_status']}`",
        f"- Scientific verdict: `{summary['scientific_verdict']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Model revision: `{summary['model_revision']}`",
        f"- Finished trials: `{summary['finished_trials']}` / `{summary['requested_trials']}`",
        f"- Successful trials: `{summary.get('completed_trials', summary['finished_trials'])}`",
        f"- Failed or pruned attempts: `{summary['failed_trials']}`",
        f"- Selected trial: `{selected if selected is not None else 'none'}`",
        "",
        "## Immutable Contract",
        "",
        f"- Requested trials: `{summary['requested_trials']}`",
        f"- Sampler seed: `{sampler_seed}`",
    ]
    if output_only:
        pilot = summary["pilot_corpus"]
        lines.extend(
            [
                f"- Pilot training documents: `{pilot['training_documents']}`",
                f"- Pilot checkpoint-selection documents: `{pilot['checkpoint_selection_documents']}`",
                f"- Pilot oracle-validation documents: `{pilot['oracle_validation_documents']}`",
                f"- Pilot corpus SHA-256: `{pilot['sha256']}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Search fingerprint: `{summary['search_fingerprint']}`",
                f"- Experiment fingerprint: `{summary['experiment_fingerprint']}`",
                f"- Search vocabulary rows: `{summary['vocabulary_rows']}`",
                f"- Search vocabulary SHA-256: `{summary['vocabulary_sha256']}`",
                f"- TPE startup trials: `{summary['sampler_startup_trials']}`",
                f"- Baseline parameters: `{summary['baseline_parameters']}`",
            ]
        )
    lines.extend(["", "## Selected Candidate", ""])
    if output_only:
        lines.extend(
            [
                f"- Parameters: `{summary.get('selected_parameters')}`",
                f"- Metrics: `{summary.get('selected_metrics')}`",
                f"- Gates: `{summary.get('selected_gates')}`",
                f"- All selected gates passed: `{summary.get('selected_output_passed')}`",
            ]
        )
    else:
        alignment = summary["selected_alignment_passed"]
        compactness = summary["selected_compactness_passed"]
        selected_parameters = None if selected_trial is None else selected_trial["parameters"]
        selected_metrics = None if selected_trial is None else selected_trial["metrics"]
        lines.extend(
            [
                f"- Selected alignment gate: `{'pending' if alignment is None else alignment}`",
                f"- Selected compactness gate: `{'pending' if compactness is None else compactness}`",
                f"- Parameters: `{selected_parameters}`",
                f"- Metrics: `{selected_metrics}`",
            ]
        )
    lines.extend(["", "## All Trials", ""])
    if output_only:
        lines.extend(
            [
                (
                    "| Trial | State | LR | Weight decay | Batch | Direct feedback | Event rollout | "
                    "Native tokens/attempt | Candidate/reference state ratio | Gates | Failure/prune reason |"
                ),
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for trial in summary["trials"]:
            parameters = trial["parameters"]
            metrics = trial["metrics"] or {}
            lines.append(
                f"| {trial['number']} | {trial['state']} | "
                f"{_search_value(parameters.get('learning_rate'), precision=3)} | "
                f"{_search_value(parameters.get('weight_decay'), precision=3)} | "
                f"{_search_value(parameters.get('batch_size'))} | "
                f"{_search_value(metrics.get('direct_feedback_equality'))} | "
                f"{_search_value(metrics.get('rollout_event_agreement'))} | "
                f"{_search_value(metrics.get('native_tokens_per_attempted_macro_step'))} | "
                f"{_search_value(metrics.get('candidate_reference_state_ratio'))} | "
                f"`{trial.get('gates')}` | `{trial.get('failure') or 'none'}` |"
            )
    else:
        lines.extend(
            [
                "| Trial | State | LR | Weight decay | Batch | NRMSE | Cosine p01 | Cosine p50 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for trial in summary["trials"]:
            parameters = trial["parameters"]
            metrics = trial["metrics"] or {}
            lines.append(
                f"| {trial['number']} | {trial['state']} | "
                f"{_search_value(parameters.get('learning_rate'), precision=3)} | "
                f"{_search_value(parameters.get('weight_decay'), precision=3)} | "
                f"{_search_value(parameters.get('batch_size'))} | "
                f"{_search_value(metrics.get('normalized_rmse'))} | "
                f"{_search_value(metrics.get('cosine_p01'))} | "
                f"{_search_value(metrics.get('cosine_p50'))} |"
            )
    failure = summary.get("failure")
    if failure is not None:
        lines.extend(["", "## Search Failure", "", f"- `{failure}`"])
    return "\n".join(lines) + "\n"
