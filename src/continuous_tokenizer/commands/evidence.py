from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    write_evidence_manifest,
)
from continuous_tokenizer.artifacts.evidence import (
    verify_artifact as verify_artifact_directory,
)
from continuous_tokenizer.artifacts.software_validation import (
    load_software_validation_inputs,
)
from continuous_tokenizer.artifacts.store import RunDirectory
from continuous_tokenizer.campaigns.state_budget import run_state_budget
from continuous_tokenizer.reporting.project import assemble_project_artifact
from continuous_tokenizer.reporting.readme import update_readme
from continuous_tokenizer.reporting.replication import aggregate_runs
from continuous_tokenizer.reporting.replication_markdown import replication_report


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = RunDirectory(args.output_dir)
    summary = aggregate_runs(args.runs)
    output.write_json("replication.json", summary)
    output.write_text("replication-report.md", replication_report(summary))
    write_evidence_manifest(
        args.output_dir,
        EvidenceManifest(
            artifact_kind="replication",
            mode=str(summary["mode"]),
            status=str(summary["operational_status"]),
            identity=EvidenceIdentity(
                source_commit=str(summary["source_commit"]),
                source_dirty=bool(summary["source_dirty"]),
                source_state_sha256=str(summary["source_state_sha256"]),
                dependency_lock_sha256=str(summary["dependency_lock_sha256"]),
                installed_package=dict(summary["installed_package"]),
                claim_vocabulary_sha256=str(summary["claim_vocabulary_sha256"]),
                source_assets=dict(summary["source_assets"]),
                verification=dict(summary["verification"]),
                model_id=str(summary["model"]["id"]),
                model_revision=str(summary["model"]["revision"]),
            ),
            parents={f"run_seed_{Path(run).name}": Path(run) / "manifest-final.json" for run in args.runs},
            inputs={f"experiment_seed_{Path(run).name}": Path(run) / "experiment.json" for run in args.runs},
            artifacts={
                "replication": args.output_dir / "replication.json",
                "report": args.output_dir / "replication-report.md",
            },
        ),
    )
    return summary


def _software_validation_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path] | None:
    verification = args.verification
    input_synthetic = args.input_synthetic
    output_synthetic = args.output_synthetic
    if verification is None and input_synthetic is None and output_synthetic is None:
        return None
    if verification is None or input_synthetic is None or output_synthetic is None:
        raise ValueError("software validation requires --verification, --input-synthetic, and --output-synthetic together")
    return verification, input_synthetic, output_synthetic


def project_report(args: argparse.Namespace) -> dict[str, Any]:
    return assemble_project_artifact(
        args.primary_replications,
        args.output_dir,
        alignment_studies=args.alignment_studies,
        deployments=args.deployments,
        software_validation_paths=_software_validation_paths(args),
    )


def state_budget(args: argparse.Namespace) -> dict[str, object]:
    return run_state_budget(
        args.input_project,
        args.output_project,
        args.output_dir,
    )


def verify_artifact(args: argparse.Namespace) -> dict[str, Any]:
    result = verify_artifact_directory(args.artifact)
    if not result["valid"]:
        raise ValueError("artifact verification failed: " + "; ".join(result["errors"]))
    return result


def readme(args: argparse.Namespace) -> dict[str, Any]:
    path = Path("README.md")
    validation_paths = _software_validation_paths(args)
    software_validation = None if validation_paths is None else load_software_validation_inputs(*validation_paths)
    changed = update_readme(
        path,
        input_project=args.input_project,
        output_project=args.output_project,
        software_validation=software_validation,
        check=args.check,
    )
    return {
        "readme": str(path),
        "checked": bool(args.check),
        "changed": changed,
    }
