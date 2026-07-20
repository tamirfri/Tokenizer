from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from continuous_tokenizer.reporting.shared import (
    current_design_notice_line,
    display_name,
)


def prospective_stop_markdown(
    experiment_name: str,
    stop_reason: str,
    boundary: str,
    statuses: Mapping[str, str],
) -> str:
    lines = [
        f"# {experiment_name}",
        "",
        "- Operational status: **completed**",
        "- Scientific verdict: **unsupported**",
        f"- Prospective stop reason: **{stop_reason}**",
        f"- Stop boundary: `{boundary}`",
        "- Final claims: **none**",
        "",
        "## Prospective stages",
        "",
        *(f"- `{name}`: **{status}**" for name, status in statuses.items()),
    ]
    return "\n".join(lines) + "\n"


def prospective_markdown(result: Mapping[str, Any]) -> str:
    tier = display_name(result["tier"])
    lines = [
        f"# {result['name']}",
        "",
        f"- Tier: **{tier}**",
        f"- Mode: **{str(result['mode']).replace('_', ' ')}**",
        f"- Operational status: **{result['operational_status']}**",
        f"- Scientific verdict: **{result['scientific_verdict']}**",
        f"- Budget exhausted: **{result['budget_exhausted']}**",
        current_design_notice_line(),
    ]
    if result["tier"] != "final_evidence":
        lines.extend(
            [
                "- Final claims: **NOT ALLOWED**",
                "- Project evidence eligibility: **NO**",
                "- Replication eligibility: **NO**",
            ],
        )
    wall_clock = result.get("wall_clock")
    if isinstance(wall_clock, Mapping):
        lines.extend(
            [
                "",
                "## Wall-clock contract",
                "",
                f"- Elapsed seconds: {wall_clock.get('elapsed_seconds')}",
                f"- Expected seconds: {wall_clock.get('expected_seconds')}",
                f"- Maximum seconds: {wall_clock.get('maximum_seconds')}",
                "- Stop boundary: epoch or stage only",
            ],
        )
    stages = result.get("stages")
    if isinstance(stages, Sequence) and not isinstance(stages, str | bytes):
        lines.extend(["", "## Registered stages", ""])
        lines.extend(f"- `{stage.get('name')}`: **{stage.get('status')}**" for stage in stages if isinstance(stage, Mapping))
    selection = result.get("selection")
    if isinstance(selection, Mapping):
        lines.extend(
            [
                "",
                "## Selection",
                "",
                f"- Feasible: **{selection.get('selection_feasible')}**",
                f"- Selected candidate: `{selection.get('selected_candidate')}`",
                "- Alignment result is reported independently from efficiency selection.",
            ],
        )
    return "\n".join(lines) + "\n"
