from __future__ import annotations

from continuous_tokenizer.contracts.state_budget import StateBudgetResult
from continuous_tokenizer.reporting.shared import current_design_notice_line


def state_budget_report(result: StateBudgetResult) -> str:
    lines = [
        "# Joint vocabulary tensor-state budget",
        "",
        "## Future prerequisite verdict",
        "",
        "- Evidence role: `future prerequisite`",
        current_design_notice_line(),
        (f"- Registered conclusion: `{result.conclusion}` — **{result.verdict.upper()}**"),
        (f"- Worst-case candidate/reference ratio: `{result.worst_case_ratio:.6f}`"),
        (f"- Preregistered maximum ratio: `{result.config.maximum_ratio:.6f}`"),
        "",
        (
            "This is a cross-directional ordinary-vocabulary tensor-state arithmetic "
            "prerequisite. It is not evidence of a combined tokenizer, physical "
            "removal, resident-memory reduction, peak-memory reduction, or runtime "
            "behavior."
        ),
        "",
        "## Mandatory non-claims",
        "",
        *[f"- `{name}`: `false`" for name in result.non_claims.to_dict()],
        "",
        "## Per-seed tensor arithmetic",
        "",
        (
            "| Model | Seed | Reference layout | Input codec | Output codec | "
            "Atomic byte rows | Shared control IDs | Shared control rows | "
            "Candidate total | Reference input | Reference output | "
            "Reference deduplicated total | Ratio |"
        ),
        ("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"),
    ]
    for row in result.per_seed:
        arithmetic = row.arithmetic
        layout = "tied once" if row.tie_word_embeddings else "untied input + output"
        lines.append(
            f"| `{row.model_id}` | {row.seed} | {layout} | "
            f"{arithmetic.input_codec_bytes:,} | "
            f"{arithmetic.output_codec_bytes:,} | "
            f"{arithmetic.atomic_byte_rows_bytes:,} | "
            f"{arithmetic.shared_control_id_bytes:,} | "
            f"{arithmetic.shared_control_row_bytes:,} | "
            f"{arithmetic.candidate_tensor_state_bytes:,} | "
            f"{arithmetic.reference_input_table_bytes:,} | "
            f"{arithmetic.reference_output_head_bytes:,} | "
            f"{arithmetic.reference_tensor_state_bytes:,} | "
            f"{row.ratio:.6f} |",
        )
    lines.extend(
        [
            "",
            (
                "Shared control IDs and copied control rows are counted once. "
                "A tied native vocabulary tensor is counted once; untied input "
                "and output vocabulary tensors are counted separately."
            ),
            "",
        ],
    )
    return "\n".join(lines)
