from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from continuous_tokenizer.contracts.claims import DEFAULT_PERFORMANCE_LIMITATION


def performance_ablation_report(artifact: Mapping[str, Any]) -> str:
    lines = [
        "# Performance Ablation",
        "",
        "> Operational and secondary evidence only. This artifact cannot promote a final "
        "deployment claim, the primary density-and-behavior headline, or the future joint "
        "tensor-state prerequisite.",
        f"> {DEFAULT_PERFORMANCE_LIMITATION}",
        "",
        f"- Operational status: `{artifact['operational_status']}`",
        f"- Exact semantic digest: `{artifact['semantic_sha256']}`",
        "- Optimization IDs: " + ", ".join(f"`{value}`" for value in artifact["optimization_ids"]),
        "",
        "## Paired Conditions",
        "",
        "| Condition | Warmups | Pairs | Baseline median | Optimized median | Ratio | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, condition in artifact["conditions"].items():
        summary = condition["summary"]
        lines.append(
            f"| {name} | {condition['warmups']} | {condition['repetitions']} | "
            f"{summary['baseline_median_seconds']:.6f} | "
            f"{summary['optimized_median_seconds']:.6f} | "
            f"{summary['median_ratio']:.6f} | "
            f"[{summary['confidence_95_low']:.6f}, {summary['confidence_95_high']:.6f}] |",
        )
    lines.extend(
        (
            "",
            "Every summary is recomputed from sealed raw pairs during semantic verification. Regressions remain visible as ratios at or above one.",
        ),
    )
    return "\n".join(lines) + "\n"
