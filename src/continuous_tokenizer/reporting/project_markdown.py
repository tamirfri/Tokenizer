from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from continuous_tokenizer.reporting.shared import (
    canonical_claim_lines,
    category_verdict_lines,
    claim_role_groups,
    claim_verdicts,
    current_design_lines,
    display_name,
    input_headline_line,
    input_headline_operands,
    parent_evidence_summary,
    status_lines,
)


def _project_metric_line(name: str, metric: Mapping[str, Any], *, suffix: str = "") -> str:
    if metric["count"] == 0:
        return f"- {name}: `not measured`; raw per-seed values `{metric.get('raw_values', [])}`"
    low = metric.get("confidence_95_low")
    high = metric.get("confidence_95_high")
    interval = "`not estimated`" if low is None or high is None else f"`[{float(low):.6f}, {float(high):.6f}]`"
    return (
        f"- {name}: mean `{metric['mean']:.6f}{suffix}`, "
        f"range `[{metric['minimum']:.6f}, {metric['maximum']:.6f}]`, "
        f"95% confidence interval {interval}; raw per-seed values "
        f"`{metric.get('raw_values', 'not recorded')}`"
    )


def _statement_trace_lines(trace: Mapping[str, Any]) -> list[str]:
    pointers = trace["canonical_artifact_pointers"] or ["not required — protocol construction"]
    return [
        f"### `{trace['statement_id']}` — {trace['paper_label']}",
        "",
        f"- Statement / contract ID: `{trace['statement_id']}`",
        f"- Class / scope: `{trace['kind']}` / `{trace['scope']}`",
        f"- Statement: {trace['statement']}",
        "- Implementation symbol(s): " + ", ".join(f"`{value}`" for value in trace["implementation_symbols"]),
        "- Validating test ID(s): " + ", ".join(f"`{value}`" for value in trace["validating_test_ids"]),
        "- Canonical artifact / JSON pointer(s): " + ", ".join(f"`{value}`" for value in pointers),
        f"- Model / seed denominator: {trace['model_seed_denominator']}",
        f"- Verdict: **{str(trace['verdict']).replace('_', ' ').upper()}**",
        f"- Reason: {trace['reason']}",
        "",
    ]


def _claim_trace_lines(trace: Mapping[str, Any]) -> list[str]:
    return [
        f"### `{trace['claim_id']}` — {trace['paper_label']}",
        "",
        f"- Paper label / claim ID: {trace['paper_label']} / `{trace['claim_id']}`",
        f"- Evidence class: `{trace['evidence_class']}`",
        f"- Producer: `{trace['producer_symbol']}`",
        "- Canonical artifact / JSON pointer(s): " + ", ".join(f"`{value}`" for value in trace["canonical_artifact_pointers"]),
        "- Parent model pointer(s): " + parent_evidence_summary(trace),
        f"- Model / seed denominator: {trace['model_seed_denominator']}",
        f"- Verdict: **{str(trace['verdict']).upper()}**",
        f"- Reason: {trace['reason']}",
        "",
    ]


def _input_headline_lines(project: Mapping[str, Any]) -> list[str]:
    verdicts = claim_verdicts(project["claims"])
    rows = [
        "| "
        + " | ".join(
            str(value)
            for value in (
                model["model"]["id"],
                *input_headline_operands(run).values(),
            )
        )
        + " |"
        for model in project["models"]
        for run in model["replication"]["runs"]
    ]
    return [
        "## Headline Verdict",
        "",
        input_headline_line(),
        (f"- Cross-model usable exact held-out input position compression: **{str(project['scientific_verdict']).upper()}**"),
        (
            "- Composition: exact held-out position compression "
            f"**{verdicts['input.held_out_position_compression'].upper()}** "
            "plus registered behavioral similarity "
            f"**{verdicts['input.registered_behavioral_similarity_tolerances'].upper()}**"
        ),
        (f"- Full-vocabulary embedding alignment is independent: **{verdicts['input.full_vocabulary_embedding_compatibility'].upper()}**"),
        (f"- Prospective fixed-subset alignment feasibility is independent and non-final: **{verdicts['input.fixed_subset_alignment_feasibility'].upper()}**"),
        "",
        "### Stored headline operands by model and seed",
        "",
        (
            "| Model | Seed | Exact held-out bytes | Native/continuous positions | "
            "Minimum | KL | Maximum KL | NLL delta | Maximum NLL delta | Top-1 | "
            "Minimum top-1 | Generation-byte similarity | Minimum generation-byte similarity |"
        ),
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        ("The report presents stored operands and verdicts. It does not recompute acceptance."),
        "",
    ]


def _trace_sections(
    statement_traces: list[Mapping[str, Any]],
    claim_traces: list[Mapping[str, Any]],
    claims: list[Mapping[str, Any]],
) -> list[str]:
    lines = ["", "## Protocol Proofs", ""]
    for trace in statement_traces:
        if trace["kind"] == "protocol":
            lines.extend(_statement_trace_lines(trace))
    lines.extend(["## Software Validation", ""])
    for trace in statement_traces:
        if trace["kind"] == "software":
            lines.extend(_statement_trace_lines(trace))
    lines.extend(
        [
            "## Per-model Empirical Support",
            "",
            "Prospective alignment studies, search selections, and diagnostics are non-final evidence and cannot supply final model verdicts.",
            "",
        ]
    )
    per_model_by_id = {str(trace["claim_id"]): trace for trace in claim_traces if not str(trace["claim_id"]).endswith("cross_model_confirmation")}
    for role, records in claim_role_groups(claims):
        role_traces = [per_model_by_id[str(record["claim_id"])] for record in records if str(record["claim_id"]) in per_model_by_id]
        if role_traces:
            lines.extend([f"### {display_name(role)} claims", ""])
            for trace in role_traces:
                lines.extend(_claim_trace_lines(trace))
    lines.extend(["## Cross-model Confirmation", ""])
    for trace in claim_traces:
        if str(trace["claim_id"]).endswith("cross_model_confirmation"):
            lines.extend(_claim_trace_lines(trace))
    lines.extend(["## Canonical Project Claim Mapping", ""])
    lines.extend(canonical_claim_lines(claims))
    return lines


def project_report(project: Mapping[str, Any]) -> str:
    mode = str(project["mode"])
    claims = project["claims"]
    models = project["models"]
    output_only = mode == "output_only"
    title = "Continuous Byte Tokenizer Output Project Evidence" if output_only else "Continuous Byte Tokenizer Input Project Evidence"
    lines = [
        f"# {title}",
        "",
        *(
            _input_headline_lines(project)
            if not output_only
            else (
                "## Headline Verdict",
                "",
                f"- Cross-model scientific verdict: **{str(project['scientific_verdict']).upper()}**",
                "",
            )
        ),
        *current_design_lines(),
        "",
        "## Project Status",
        "",
        *status_lines(project),
        f"- Cross-model verdict: `{project['cross_model_verdict']}`",
        f"- Independent primary-model replications: `{len(models)}`",
        "",
        "Operational completion and scientific support are separate. This report publishes stored artifact verdicts and does not recompute acceptance.",
        "Primary density and behavior appear first. Deployment performance is secondary and "
        "final-only. Research throughput is operational. Runtime speed does not support the "
        "future joint tensor-state prerequisite.",
        "",
        "## Model Roles",
        "",
        "| Evidence role | Model | Runs |",
        "|---|---|---:|",
        *[f"| Equal primary final replication | `{model['model']['id']}` | {len(model['replication']['runs'])} |" for model in models],
        "",
        "## Independent Category Verdicts",
        "",
        *category_verdict_lines(project["category_verdicts"]),
    ]
    if output_only:
        lines.append("- Native vocabulary head used during generation: **NO**")
    lines.extend(
        _trace_sections(
            project["statement_traces"],
            project["claim_traces"],
            claims,
        )
    )
    for model in models:
        quality = model["replication"]
        lines.extend(["", f"## Aggregate Evidence — {model['model']['id']}", ""])
        if output_only:
            lines.extend(_project_metric_line(display_name(name), metric) for name, metric in quality["metrics"].items())
        else:
            metrics = quality["metrics"]
            lines.extend(
                (
                    _project_metric_line(
                        "Native tokens/continuous token",
                        metrics["native_tokens_per_continuous_token"],
                        suffix="x",
                    ),
                    _project_metric_line(
                        "Candidate/reference state ratio",
                        metrics["candidate_reference_state_ratio"],
                    ),
                    _project_metric_line("Segmented KL", metrics["segmented_mean_kl"]),
                    _project_metric_line("Prompt-cache ratio", metrics["prompt_cache_ratio"]),
                    _project_metric_line("Prefill-FLOP ratio", metrics["prefill_flops_ratio"]),
                    _project_metric_line(
                        "Time-to-first-logit ratio",
                        metrics["time_to_first_logit_ratio"],
                    ),
                )
            )
    findings = [f"`{claim['claim_id']}`: {claim['verdict']} — {claim['reason']}" for claim in claims if claim["verdict"] != "supported"]
    lines.extend(
        [
            "",
            "## Failed And Unsupported Evidence",
            "",
            *(f"- {finding}" for finding in findings),
        ]
    )
    if not findings:
        lines.append("- None")
    failed_runs = [(model["model"]["id"], run) for model in models for run in model["replication"].get("failed_runs", ())]
    lines.extend(
        (f"- {model_id} failed seed `{run['seed']}`: `{run.get('failure')}`" for model_id, run in failed_runs),
    )
    if not failed_runs:
        lines.append("- Failed replication runs: none")
    return "\n".join(lines) + "\n"
