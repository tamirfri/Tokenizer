from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from continuous_tokenizer.reporting.shared import current_design_notice_line


def _current_boundary_lines() -> list[str]:
    return [
        current_design_notice_line(),
        "- Reduced study denominators lower research power; study outcomes remain non-final.",
    ]


def _length_rows(
    source: str,
    metrics: Mapping[str, Any],
) -> list[str]:
    rows = []
    for length, value in metrics.items():
        item = cast(Mapping[str, Any], value)
        bytes_per_position = float(item["bytes_per_position_with_atomic_fallback"])
        rows.append(
            f"| {source} | {length} | {item['candidates']} | {float(item['reconstruction_fraction']):.4f} | {bytes_per_position:.4f} |",
        )
    return rows


def input_study_report(result: Mapping[str, Any]) -> str:
    study = cast(Mapping[str, Any], result["study"])
    trials = cast(Sequence[Mapping[str, Any]], result["trials"])
    lines = [
        "# Input Selection Study",
        "",
        f"- Study: `{study['name']}`",
        f"- Kind: `{study['kind']}`",
        f"- Evidence scope: `{result['evidence_scope']}`",
        f"- Model: `{result['model']['id']}`",
        f"- Seed: `{result['seed']}`",
        "- Final-evidence status: **NOT APPLICABLE — selection study**",
        *_current_boundary_lines(),
        "",
        "## Registered scaling trials",
        "",
    ]
    for trial in trials:
        subset = cast(Mapping[str, Any], trial["vocabulary_subset"])
        training = cast(Mapping[str, Any], trial["training"])
        lines.extend(
            [
                f"### {subset['requested_rows'] or 'complete'} vocabulary rows",
                "",
                f"- Content SHA-256: `{subset['sha256']}`",
                f"- Checkpoint: `{training['checkpoint']}`",
                f"- Source-dtype reconstruction: `{float(training['embedding_metrics']['reconstruction_fraction']):.4f}`",
                f"- Per-epoch losses, gradient norms, wall time, and peak memory: `{trial['telemetry_directory']}`",
                "",
            ],
        )
    selection = result.get("selection")
    if isinstance(selection, Mapping):
        lines.extend(
            [
                "## Candidate selection",
                "",
                f"- Selected candidate: `{selection['selected_candidate']}`",
                f"- Selection feasible: `{selection['selection_feasible']}`",
                f"- Rule: `{selection['selection_rule']}`",
                "- Metrics split: `validation` (untouched by training)",
                "",
            ],
        )
    lines.extend(
        [
            "## Exact candidate-length measurements",
            "",
            "| Source | Length | Candidates | Reconstruction | Bytes/fallback position |",
            "|---|---:|---:|---:|---:|",
        ],
    )
    reports = cast(Sequence[Mapping[str, Any]], result["candidate_length_reports"])
    for report in reports:
        label = str(report["label"])
        for source in ("vocabulary", "wikitext_validation", "arbitrary_binary"):
            source_report = cast(Mapping[str, Any], report[source])
            lines.extend(
                _length_rows(
                    f"{label}/{source}",
                    cast(Mapping[str, Any], source_report["metrics"]),
                ),
            )
    lines.extend(
        [
            "",
            "All existing embedding, reconstruction, density, and compactness gates remain unchanged.",
            "Study outcomes select a registered candidate but are not final model evidence.",
        ],
    )
    return "\n".join(lines) + "\n"


def input_alignment_feasibility_report(result: Mapping[str, Any]) -> str:
    study = cast(Mapping[str, Any], result["study"])
    stages = cast(Sequence[Mapping[str, Any]], result["stages"])
    lines = [
        "# Prospective Input Alignment Feasibility Study",
        "",
        f"- Study: `{study['name']}`",
        f"- Model: `{result['model']['id']}`",
        f"- Training seeds: `{', '.join(str(seed) for seed in result['training_seeds'])}`",
        f"- Fixed subset seed: `{result['subset_seed']}`",
        "- Evidence scope: `selection`",
        "- Work performed: encoder vocabulary alignment only",
        "- Final-model verdict: **NONE — prospective feasibility evidence only**",
        *_current_boundary_lines(),
        "",
        "## Staged continuation decisions",
        "",
        "| Vocabulary rows | Subset SHA-256 | Status | Reason | Failed seeds |",
        "|---:|---|---|---|---|",
    ]
    for stage in stages:
        failed = cast(Sequence[Mapping[str, Any]], stage["failed_gates"])
        failed_seeds = ", ".join(str(row["training_seed"]) for row in failed) or "none"
        lines.append(
            f"| {stage['vocabulary_subset_size']} | `{stage['subset_sha256']}` | {stage['status']} | {stage['reason']} | {failed_seeds} |",
        )
        lines.extend(
            [
                "",
                f"### {stage['vocabulary_subset_size']} rows by seed",
                "",
                "| Training seed | Status | Normalized RMSE | Cosine p01 | Cosine p50 | Failed gates |",
                "|---:|---|---:|---:|---:|---|",
            ],
        )
        seed_results = cast(
            Sequence[Mapping[str, Any]],
            stage["seed_results"],
        )
        for seed_result in seed_results:
            alignment = seed_result.get("alignment")
            metrics = cast(Mapping[str, Any], alignment).get("embedding_metrics") if isinstance(alignment, Mapping) else None
            metric_values = cast(Mapping[str, Any], metrics) if isinstance(metrics, Mapping) else {}
            failed_seed_gates = seed_result.get("failed_gates")
            gate_names = (
                ", ".join(str(gate["gate"]) for gate in cast(Sequence[Mapping[str, Any]], failed_seed_gates))
                if isinstance(failed_seed_gates, Sequence)
                else "not run"
            )
            lines.append(
                f"| {seed_result['training_seed']} | {seed_result['status']} | "
                f"{metric_values.get('normalized_rmse', '—')} | "
                f"{metric_values.get('cosine_similarity_p01', '—')} | "
                f"{metric_values.get('cosine_similarity_p50', '—')} | "
                f"{gate_names or 'none'} |",
            )
    lines.extend(
        [
            "",
            "Stages marked `not_run_futility` were prospectively skipped after a prerequisite "
            "failed unchanged alignment gates. They are not execution failures and provide no "
            "empirical final-model claim.",
            "",
            f"Feasibility continuation outcome: `{result['feasibility_passed']}`.",
        ],
    )
    return "\n".join(lines) + "\n"


def input_compression_feasibility_report(result: Mapping[str, Any]) -> str:
    study = cast(Mapping[str, Any], result["study"])
    stages = cast(Sequence[Mapping[str, Any]], result["stages"])
    lines = [
        "# Prospective Input Compression Feasibility Study",
        "",
        f"- Study: `{study['name']}`",
        f"- Model: `{result['model']['id']}`",
        f"- Training seeds: `{', '.join(str(seed) for seed in result['training_seeds'])}`",
        "- Evidence scope: `selection`",
        "- Alignment: measured and reported, never a continuation gate",
        "- Final-model verdict: **NONE — prospective feasibility evidence only**",
        "- Final experiment created: `False`",
        "- Freeze performed: `False`",
        "- Final claim created: `False`",
        *_current_boundary_lines(),
        "",
        "## Prospective stage ladder",
        "",
        "| Stage | Status | Reason | Failed seeds/gates |",
        "|---|---|---|---|",
    ]
    for stage in stages:
        failures = cast(Sequence[Mapping[str, Any]], stage["failed_gates"])
        failed = ", ".join((str(row["training_seed"]) if "training_seed" in row else str(row["gate"])) for row in failures)
        lines.append(
            f"| {stage['stage']} | {stage['status']} | {stage['reason']} | {failed or 'none'} |",
        )
        lines.extend(
            [
                "",
                f"### {stage['stage']} by seed",
                "",
                "| Training seed | Status | Failed gates | Raw metrics persisted |",
                "|---:|---|---|---|",
            ],
        )
        for seed in cast(Sequence[Mapping[str, Any]], stage["seed_results"]):
            seed_failures = seed.get("failed_gates")
            gate_names = (
                ", ".join(str(gate.get("gate", gate.get("training_seed", "aggregate"))) for gate in cast(Sequence[Mapping[str, Any]], seed_failures))
                if isinstance(seed_failures, Sequence)
                else "not run"
            )
            lines.append(
                f"| {seed['training_seed']} | {seed['status']} | {gate_names or 'none'} | `{bool(seed.get('raw_metrics'))}` |",
            )
    lines.extend(
        [
            "",
            "Stages marked `not_run_futility` were not executed after the first failed all-seed aggregate prerequisite.",
            "",
            f"Compression feasibility outcome: `{result['feasibility_passed']}`.",
            "A passing outcome records eligibility only; it does not freeze an experiment or create final evidence.",
        ],
    )
    return "\n".join(lines) + "\n"
