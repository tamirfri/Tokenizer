from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from continuous_tokenizer.contracts.claims import (
    CLAIM_ROLES,
    CURRENT_DESIGN_NOTICE,
    DEFAULT_PERFORMANCE_LIMITATION,
    INPUT_HEADLINE,
)

_INPUT_HEADLINE_OPERANDS: Final = (
    ("exact held-out bytes", "density_exact", None),
    (
        "native/continuous positions",
        "native_tokens_per_continuous_token",
        None,
    ),
    (
        "minimum positions ratio",
        None,
        "minimum_native_tokens_per_continuous_token",
    ),
    ("segmented KL", "segmented_mean_kl", None),
    ("maximum KL", None, "maximum_segmented_mean_kl"),
    ("NLL delta", "segmented_nll_delta", None),
    ("maximum NLL delta", None, "maximum_segmented_nll_delta"),
    ("top-1 agreement", "segmented_top1_agreement", None),
    ("minimum top-1", None, "minimum_segmented_top1_agreement"),
    (
        "generation-byte similarity",
        "segmented_generation_byte_similarity",
        None,
    ),
    (
        "minimum generation-byte similarity",
        None,
        "minimum_segmented_generation_byte_similarity",
    ),
)


def display_name(value: object) -> str:
    return str(value).replace("_", " ").title()


def optional_mapping(value: object) -> Mapping[str, Any] | None:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else None


def optional_metric(value: Any, *, percentage: bool = False) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{value:.2%}" if percentage else f"{value:.6f}"


def status_lines(result: Mapping[str, Any]) -> list[str]:
    return [
        f"- Mode: `{result['mode']}`",
        f"- Evidence scope: `{result['evidence_scope']}`",
        f"- Operational status: `{result['operational_status']}`",
        f"- Scientific verdict: `{result['scientific_verdict']}`",
    ]


def current_design_lines() -> list[str]:
    return [
        "## Current Evidence Boundary",
        "",
        CURRENT_DESIGN_NOTICE,
        "",
        DEFAULT_PERFORMANCE_LIMITATION,
        "",
        "Faster or smaller execution never substitutes for exact held-out position compression or registered behavioral similarity.",
    ]


def current_design_notice_line() -> str:
    return f"- Current evidence boundary: {CURRENT_DESIGN_NOTICE}"


def input_headline_line() -> str:
    return f"- Definition: **{INPUT_HEADLINE}**"


def claim_label(value: bool | str | None) -> str:
    if isinstance(value, str):
        return display_name(value).upper()
    if value is None:
        return "NOT MEASURED"
    return "SUPPORTED" if value else "UNSUPPORTED"


def claim_role_groups(
    claims: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    grouped = {role: [] for role in CLAIM_ROLES}
    for claim in claims:
        role = claim.get("role")
        if isinstance(role, str) and role in grouped:
            grouped[role].append(claim)
    return tuple((role, tuple(records)) for role in CLAIM_ROLES if (records := grouped[role]))


def claim_verdicts(
    claims: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {str(claim["claim_id"]): str(claim["verdict"]) for claim in claims}


def input_headline_operands(
    run: Mapping[str, Any],
) -> dict[str, object]:
    metrics = cast(Mapping[str, object], run["metrics"])
    thresholds = cast(Mapping[str, object], run["thresholds"])
    operands: dict[str, object] = {"seed": run["seed"]}
    for label, metric_name, threshold_name in _INPUT_HEADLINE_OPERANDS:
        if metric_name is not None:
            operands[label] = metrics.get(metric_name)
        else:
            operands[label] = thresholds.get(threshold_name)
    return operands


def parent_evidence_summary(trace: Mapping[str, Any]) -> str:
    return "; ".join(
        (
            f"`{parent['model']['id']}` seeds "
            f"`{', '.join(str(seed) for seed in parent['seeds'])}` -> "
            f"`{parent['parent_pointer']}` -> "
            f"**{str(parent['verdict']).upper()}**"
        )
        for parent in trace["parent_model_evidence"]
    )


def canonical_claim_lines(claims: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for role, records in claim_role_groups(claims):
        lines.extend((f"### {display_name(role)} claims", ""))
        for claim in records:
            pointers = ", ".join(f"`{pointer}`" for pointer in claim["evidence_pointers"])
            lines.extend(
                (
                    f"- `{claim['claim_id']}` — **{claim['label']}**",
                    f"  - Role / category / basis: `{claim['role']}` / `{claim['category']}` / `{claim['basis']}`",
                    f"  - Applicability: {claim['applicability']}",
                    f"  - Preregistered gate or policy: {claim['gate_or_policy']}",
                    f"  - Producer symbol: `{claim['producer_symbol']}`",
                    f"  - Evidence pointer(s): {pointers}",
                    f"  - Denominator / sample context: {claim['denominator_context']}",
                    f"  - Verdict: **{str(claim['verdict']).upper()}**",
                    f"  - Reason: {claim['reason']}",
                )
            )
    return lines


def category_verdict_lines(verdicts: Mapping[Any, object]) -> list[str]:
    return [f"- {display_name(name)}: **{str(verdict).upper()}**" for name, verdict in verdicts.items()]
