from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from continuous_tokenizer.reporting.shared import (
    canonical_claim_lines,
    category_verdict_lines,
    claim_verdicts,
    current_design_lines,
    display_name,
    input_headline_line,
    input_headline_operands,
    status_lines,
)


def _aggregate_metric_line(name: str, metric: Mapping[str, Any]) -> str:
    raw = metric.get("raw_values", "not recorded")
    if metric["count"] == 0:
        return f"| {name} | 0 | `{raw}` | n/a | n/a | n/a | not estimated |"
    low = metric.get("confidence_95_low")
    high = metric.get("confidence_95_high")
    interval = "not estimated" if low is None or high is None else f"[{float(low):.6f}, {float(high):.6f}]"
    return f"| {name} | {metric['count']} | `{raw}` | {metric['mean']:.6f} | {metric['minimum']:.6f} | {metric['maximum']:.6f} | {interval} |"


def _input_headline_lines(summary: Mapping[str, Any]) -> list[str]:
    verdicts = claim_verdicts(summary["claims"])
    rows = ["| " + " | ".join(str(value) for value in input_headline_operands(run).values()) + " |" for run in summary["runs"]]
    return [
        "## Headline Verdict",
        "",
        input_headline_line(),
        (f"- Usable exact held-out input position compression: **{str(summary['scientific_verdict']).upper()}**"),
        (
            "- Composition: exact held-out position compression "
            f"**{verdicts['input.held_out_position_compression'].upper()}** "
            "plus registered behavioral similarity "
            f"**{verdicts['input.registered_behavioral_similarity_tolerances'].upper()}**"
        ),
        (f"- Full-vocabulary embedding alignment is independent: **{verdicts['input.full_vocabulary_embedding_compatibility'].upper()}**"),
        "",
        "### Stored headline operands by seed",
        "",
        (
            "| Seed | Exact held-out bytes | Native/continuous positions | Minimum | "
            "KL | Maximum KL | NLL delta | Maximum NLL delta | Top-1 | Minimum "
            "top-1 | Generation-byte similarity | Minimum generation-byte similarity |"
        ),
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        ("The report presents stored operands and verdicts. It does not recompute acceptance."),
        "",
    ]


def replication_report(summary: Mapping[str, Any]) -> str:
    direction = "Output-Only" if summary.get("mode") == "output_only" else "Input-Only"
    lines = [
        f"# {direction} Replication Summary",
        "",
        *(
            _input_headline_lines(summary)
            if summary.get("mode") == "input_only"
            else (
                "## Headline Verdict",
                "",
                f"- Scientific verdict: **{str(summary['scientific_verdict']).upper()}**",
                "",
            )
        ),
        *current_design_lines(),
        "",
        "## Replication Status",
        "",
        *status_lines(summary),
        f"- Model: `{summary['model']['id']}`",
        f"- Revision: `{summary['model']['revision']}`",
        f"- Requested runs: `{summary['requested_runs']}`",
        f"- Completed runs: `{len(summary['runs'])}`",
        f"- Failed runs: `{len(summary['failed_runs'])}`",
        f"- Replication complete: `{summary['replication_complete']}`",
        "",
        "## Category Verdicts",
        "",
        *category_verdict_lines(summary["category_verdicts"]),
        "",
        "## Per-Seed Evidence",
        "",
        "| Seed | Operational status | Scientific verdict |",
        "|---:|---|---|",
    ]
    lines.extend(f"| {row['seed']} | {row['operational_status']} | {row['scientific_verdict']} |" for row in summary["seed_evidence"])
    lines.extend(
        [
            "",
            "## Claims",
            "",
            *canonical_claim_lines(summary["claims"]),
            "",
            "## Aggregate Metrics",
            "",
            "Primary density and behavior remain the scientific headline. Deployment performance "
            "is secondary and final-only. Research throughput is operational evidence. Runtime "
            "speed does not support the future joint tensor-state prerequisite.",
            "",
            "| Metric | Count | Raw per-seed values | Mean | Minimum | Maximum | 95% confidence interval |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_aggregate_metric_line(display_name(name), metric) for name, metric in summary["metrics"].items())
    lines.extend(
        [
            "",
            "Student-t intervals are present only for eligible seed-dependent metrics from completed final primary-model seeds 17, 23, and 41.",
        ]
    )
    if summary["failed_runs"]:
        lines.extend(
            [
                "",
                "## Failed Runs",
                "",
                *[f"- Seed `{run['seed']}`: `{run['failure']}`" for run in summary["failed_runs"]],
            ]
        )
    return "\n".join(lines) + "\n"
