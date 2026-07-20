from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from continuous_tokenizer.artifacts.evidence import seal_generated_evidence
from continuous_tokenizer.artifacts.source import find_project_root
from continuous_tokenizer.campaigns.dispatch import create_experiment_runner
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.prospective import PROSPECTIVE_ARTIFACT_KINDS
from continuous_tokenizer.diagnostics.preflight import require_storage


def _require_storage(
    spec_path: Path,
    output_dir: Path,
    experiment_path: str,
) -> None:
    experiment = ExperimentSpec.load(
        (spec_path.parent / experiment_path).resolve(),
    )
    require_storage(
        output_dir.parent,
        experiment,
        refusal_message="insufficient storage for the next execution while preserving the registered reserve",
    )


def _require_search_storage(
    args: argparse.Namespace,
    search_values: dict[str, Any],
) -> None:
    if not args.prepare_only:
        _require_storage(
            args.spec,
            args.output_dir,
            str(search_values["experiment"]),
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    with args.spec.open("rb") as handle:
        values = tomllib.load(handle)
    if values.get("artifact_kind") in PROSPECTIVE_ARTIFACT_KINDS.values():
        from continuous_tokenizer.input.prospective import run_prospective

        return run_prospective(
            args.spec,
            args.output_dir,
            args.verification,
            resume=args.resume,
        )
    return create_experiment_runner(
        ExperimentSpec.load(args.spec),
        args.output_dir,
        find_project_root(args.spec),
        args.verification,
        resume=args.resume,
    ).run()


def search(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return _search(args)
    finally:
        seal_generated_evidence(
            args.spec,
            args.output_dir,
            artifact_kind="search",
        )


def _search(args: argparse.Namespace) -> dict[str, Any]:
    with args.spec.open("rb") as handle:
        search_values = tomllib.load(handle)
    _require_search_storage(args, search_values)
    if search_values.get("study") == "efficiency":
        try:
            from continuous_tokenizer.search.efficiency import run_efficiency_search
        except ModuleNotFoundError as error:
            if error.name in {"optuna", "tomli_w"}:
                raise RuntimeError("install the search dependency group with `uv sync --group search`") from error
            raise
        return run_efficiency_search(
            args.spec,
            args.output_dir,
            resume=args.resume,
            prepare_only=args.prepare_only,
        )
    if search_values.get("mode") == "output_only":
        if not args.prepare_only and args.oracle_study is None:
            raise ValueError("output search requires --oracle-study before training")
        try:
            from continuous_tokenizer.search.output import run_output_search
        except ModuleNotFoundError as error:
            if error.name == "optuna":
                raise RuntimeError("install the search dependency group with `uv sync --group search`") from error
            raise
        return run_output_search(
            args.spec,
            args.output_dir,
            resume=args.resume,
            prepare_only=args.prepare_only,
            oracle_study_artifact=args.oracle_study,
        )
    try:
        from continuous_tokenizer.search.alignment import run_vocabulary_search
    except ModuleNotFoundError as error:
        if error.name in {"optuna", "tomli_w"}:
            raise RuntimeError("install the search dependency group with `uv sync --group search`") from error
        raise
    return run_vocabulary_search(
        args.spec,
        args.output_dir,
        resume=args.resume,
        prepare_only=args.prepare_only,
    )


def study(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return _study(args)
    finally:
        seal_generated_evidence(
            args.spec,
            args.output_dir,
            artifact_kind="study",
        )


def _study(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"study directory already exists: {args.output_dir}",
        )
    with args.spec.open("rb") as handle:
        values = tomllib.load(handle)
    _require_storage(
        args.spec,
        args.output_dir,
        str(values["experiment"]),
    )
    study_kind = values.get("study")
    if study_kind in {
        "input_selection",
        "input_alignment_feasibility",
        "input_compression_feasibility",
    }:
        from continuous_tokenizer.input.study import (
            run_input_alignment_feasibility_study,
            run_input_compression_feasibility_study,
            run_input_selection_study,
        )

        input_studies = {
            "input_selection": run_input_selection_study,
            "input_alignment_feasibility": run_input_alignment_feasibility_study,
            "input_compression_feasibility": run_input_compression_feasibility_study,
        }
        return input_studies[study_kind](
            args.spec,
            args.output_dir,
            resume=args.resume,
        )
    if study_kind == "output_oracle":
        from continuous_tokenizer.output.study import run_output_oracle_study

        return run_output_oracle_study(
            args.spec,
            args.output_dir,
            resume=args.resume,
        )
    raise ValueError("unsupported registered study")
