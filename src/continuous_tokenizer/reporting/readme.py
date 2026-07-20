from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from continuous_tokenizer.artifacts.evidence import (
    EVIDENCE_MANIFEST_FILENAME,
    load_evidence_manifest,
    verify_artifact,
)
from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.artifacts.manifest import load_artifact
from continuous_tokenizer.artifacts.store import write_text_atomic
from continuous_tokenizer.contracts.claims import claim_records, directional_claims
from continuous_tokenizer.contracts.experiment import TokenizerMode
from continuous_tokenizer.contracts.statements import (
    SoftwareValidationInputs,
    SourceBinding,
    StatementTrace,
    statement_trace,
)
from continuous_tokenizer.reporting.shared import (
    claim_role_groups,
    display_name,
)

README_LEDGER_BEGIN: Final = "<!-- BEGIN GENERATED EVIDENCE LEDGER -->"
README_LEDGER_END: Final = "<!-- END GENERATED EVIDENCE LEDGER -->"


@dataclass(frozen=True, slots=True)
class _ProjectEvidence:
    project_path: Path
    manifest_sha256: str
    project: Mapping[str, Any]


def _load_project(path: Path, mode: TokenizerMode) -> _ProjectEvidence:
    directory = path if path.is_dir() else path.parent
    verification = verify_artifact(directory)
    if verification["valid"] is not True:
        raise ValueError(f"{mode} project artifact verification failed: " + "; ".join(cast(Sequence[str], verification["errors"])))
    manifest_path = directory / EVIDENCE_MANIFEST_FILENAME
    manifest = load_evidence_manifest(manifest_path)
    if manifest["artifact_kind"] != "project" or manifest["mode"] != mode or manifest["status"] != "completed":
        raise ValueError(f"{mode} README evidence must be a completed sealed project artifact")
    project_path = directory / "project.json"
    project = load_artifact(project_path)
    if project.get("mode") != mode:
        raise ValueError(f"{mode} README project mode does not match its argument")
    return _ProjectEvidence(
        project_path=project_path,
        manifest_sha256=sha256_path(manifest_path),
        project=project,
    )


def _claim_records(
    mode: TokenizerMode,
    evidence: _ProjectEvidence | None,
) -> tuple[Mapping[str, Any], ...]:
    if evidence is None:
        return tuple(
            claim_records(
                mode,
                {definition.claim_id: "incomplete" for definition in directional_claims(mode)},
                reason="no final semantically verified sealed project artifact was supplied",
            )
        )
    values = evidence.project.get("claims")
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise ValueError(f"{mode} project does not contain canonical claim records")
    return tuple(cast(Mapping[str, Any], value) for value in values if isinstance(value, Mapping))


def _display_path(path: Path, readme_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=readme_path.parent.resolve())).as_posix()


def _claim_lines(
    record: Mapping[str, Any],
    *,
    evidence: _ProjectEvidence | None,
    claim_index: int,
    readme_path: Path,
) -> list[str]:
    if evidence is None:
        artifact_pointer = "not available (no sealed project supplied)"
        manifest_sha256 = "not available (no sealed project supplied)"
    else:
        artifact_pointer = f"{_display_path(evidence.project_path, readme_path)}#/claims/{claim_index}"
        manifest_sha256 = evidence.manifest_sha256
    pointers = ", ".join(f"`{pointer}`" for pointer in record["evidence_pointers"])
    return [
        f"- `{record['claim_id']}` — **{record['label']}** — `{record['role']}` — Verdict: **{str(record['verdict']).upper()}**",
        f"  - Decisive metric or policy: {record['gate_or_policy']}",
        f"  - Evidence: `{artifact_pointer}`; raw records {pointers}; Evidence manifest SHA-256 `{manifest_sha256}`",
        f"  - Scope: {record['denominator_context']}. {record['reason']}",
    ]


def _statement_lines(trace: StatementTrace) -> list[str]:
    definition = trace.definition
    symbols = ", ".join(f"`{symbol}`" for symbol in definition.implementation_symbols)
    tests = ", ".join(f"`{test_id}`" for test_id in definition.validating_test_ids)
    if trace.evidence_pointers:
        evidence = ", ".join(f"`{pointer}`" for pointer in trace.evidence_pointers)
    elif definition.kind == "protocol":
        evidence = "not required (construction proof)"
    else:
        evidence = "not supplied"
    return [
        f"- `{definition.statement_id}` — **{definition.paper_label}** — `{definition.kind}` — Status: **{trace.status.replace('_', ' ').upper()}**",
        f"  - Statement: {definition.summary}",
        f"  - Evidence: {evidence}; implementation {symbols}; tests {tests}",
        f"  - Reason: {trace.reason}",
    ]


def render_readme_ledger(
    readme_path: Path,
    *,
    input_project: Path | None = None,
    output_project: Path | None = None,
    software_validation: SoftwareValidationInputs | None = None,
) -> str:
    evidence_by_mode = {
        "input_only": None if input_project is None else _load_project(input_project, "input_only"),
        "output_only": None if output_project is None else _load_project(output_project, "output_only"),
    }
    if software_validation is not None:
        for evidence in evidence_by_mode.values():
            if evidence is not None and software_validation.verification.source != SourceBinding(
                str(evidence.project["source_state_sha256"]),
                str(evidence.project["dependency_lock_sha256"]),
            ):
                raise ValueError("software validation bundle source identity differs from README project evidence")
    lines = [
        README_LEDGER_BEGIN,
        "## Generated evidence ledger",
        "",
        "> [!IMPORTANT]",
        "> This section is generated by `tokenizer readme`. Do not edit it by hand.",
        "> Only semantically verified, sealed project artifacts can supply empirical verdicts.",
        "> Missing final evidence is `INCOMPLETE`; it is never interpreted as zero or unsupported.",
        "",
        "### Protocol proofs",
        "",
        "`PROVED` is reserved for construction consequences. `VALIDATED` requires source-bound "
        "verification and both registered synthetic campaigns. Empirical claims remain separate.",
        "",
    ]
    traces = statement_trace(software_validation)
    for trace in (trace for trace in traces if trace.definition.kind == "protocol"):
        lines.extend(_statement_lines(trace))
    lines.extend(("### Software validation", ""))
    for trace in (trace for trace in traces if trace.definition.kind == "software"):
        lines.extend(_statement_lines(trace))
    lines.extend(("### Empirical claim ledger", ""))
    for mode, title in (
        ("input_only", "Input-only claims"),
        ("output_only", "Output-only claims"),
    ):
        evidence = evidence_by_mode[mode]
        records = _claim_records(mode, evidence)
        lines.extend((f"### {title}", ""))
        if evidence is None:
            lines.extend(
                (
                    "**No final semantically verified sealed project evidence was supplied.**",
                    "",
                )
            )
        claim_indexes = {str(record["claim_id"]): index for index, record in enumerate(records)}
        for role, role_records in claim_role_groups(records):
            lines.extend((f"#### {display_name(role)} claims", ""))
            for record in role_records:
                lines.extend(
                    _claim_lines(
                        record,
                        evidence=evidence,
                        claim_index=claim_indexes[str(record["claim_id"])],
                        readme_path=readme_path,
                    )
                )
    lines.append(README_LEDGER_END)
    return "\n".join(lines) + "\n"


def update_readme(
    readme_path: Path,
    *,
    input_project: Path | None = None,
    output_project: Path | None = None,
    software_validation: SoftwareValidationInputs | None = None,
    check: bool = False,
) -> bool:
    current = readme_path.read_text(encoding="utf-8")
    if current.count(README_LEDGER_BEGIN) != 1 or current.count(README_LEDGER_END) != 1:
        raise ValueError("README must contain exactly one generated evidence ledger marker region")
    before, remainder = current.split(README_LEDGER_BEGIN, 1)
    _, after = remainder.split(README_LEDGER_END, 1)
    ledger = render_readme_ledger(
        readme_path,
        input_project=input_project,
        output_project=output_project,
        software_validation=software_validation,
    )
    rendered = before + ledger + after.lstrip("\n")
    changed = rendered != current
    if check and changed:
        raise ValueError("README generated evidence ledger is out of date")
    if changed and not check:
        write_text_atomic(readme_path, rendered)
    return changed
