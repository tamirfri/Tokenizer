from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from continuous_tokenizer.commands.campaign import run, search, study
from continuous_tokenizer.commands.deployment import deployment, deployment_spec
from continuous_tokenizer.commands.evidence import (
    aggregate,
    project_report,
    readme,
    state_budget,
    verify_artifact,
)
from continuous_tokenizer.commands.freeze import freeze
from continuous_tokenizer.commands.input import attention, benchmark, evaluate, segment, train
from continuous_tokenizer.commands.inspect import inspect_model
from continuous_tokenizer.commands.output import generate
from continuous_tokenizer.commands.verify import verify
from continuous_tokenizer.contracts.profiles import (
    CAMPAIGN_PROFILE_NAME,
    TRAINING_PROFILE_NAMES,
)
from continuous_tokenizer.runtime.progress import configure_logging


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Hugging Face model ID")
    parser.add_argument("--revision", help="Optional model revision")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=TRAINING_PROFILE_NAMES,
        default=CAMPAIGN_PROFILE_NAME,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--vocabulary-epochs", type=int, default=10)
    parser.add_argument("--reconstruction-epochs", type=int, default=1)
    parser.add_argument("--reconstruction-samples", type=int, default=2_048)
    parser.add_argument("--reconstruction-vocabulary-fraction", type=float, default=0.75)
    parser.add_argument("--validation-bytes", type=int, default=2_048)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--evaluation-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--generation-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--performance-prompts", type=int, default=2)


def _add_segment_arguments(parser: argparse.ArgumentParser) -> None:
    _add_model_argument(parser)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("text", nargs="?")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--hex", dest="hex_bytes")
    inputs.add_argument("--file", type=Path)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    _add_model_argument(parser)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-test-bytes", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retrieval-rows", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=2)


def _add_attention_arguments(parser: argparse.ArgumentParser) -> None:
    _add_model_argument(parser)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("text")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--alignment", choices=("aligned", "arbitrary"), default="arbitrary")


def _add_train_command_arguments(parser: argparse.ArgumentParser) -> None:
    _add_model_argument(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    _add_training_arguments(parser)


def _add_evaluate_command_arguments(parser: argparse.ArgumentParser) -> None:
    _add_model_argument(parser)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    _add_evaluation_arguments(parser)


def _add_software_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--input-synthetic", type=Path)
    parser.add_argument("--output-synthetic", type=Path)


type _ArgumentAdder = Callable[[argparse.ArgumentParser], None]

_INPUT_COMMANDS: tuple[tuple[str, str, _ArgumentAdder], ...] = (
    ("inspect", "validate a model byte vocabulary", _add_model_argument),
    ("train", "train the input tokenizer against a frozen embedding table", _add_train_command_arguments),
    ("segment", "segment text or raw bytes with a trained tokenizer", _add_segment_arguments),
    ("benchmark", "benchmark a trained tokenizer", _add_benchmark_arguments),
    ("evaluate", "compare frozen-model input paths", _add_evaluate_command_arguments),
    ("attention", "capture native and segmented attention diagnostics", _add_attention_arguments),
)


def _add_directional_parsers(subparsers: Any) -> None:
    for name, help_text, add_arguments in _INPUT_COMMANDS:
        add_arguments(subparsers.add_parser(name, help=help_text))

    input_parser = subparsers.add_parser("input", help="input-only tokenizer diagnostics")
    input_commands = input_parser.add_subparsers(dest="direction_command", required=True)
    for name, _help_text, add_arguments in _INPUT_COMMANDS:
        add_arguments(input_commands.add_parser(name))

    output_parser = subparsers.add_parser("output", help="output-only tokenizer diagnostics")
    output_commands = output_parser.add_subparsers(dest="direction_command", required=True)
    output_generate = output_commands.add_parser("generate", help="generate byte spans without the native vocabulary head")
    _add_model_argument(output_generate)
    output_generate.add_argument("checkpoint", type=Path)
    output_generate.add_argument("prompt")
    output_generate.add_argument("--max-macro-steps", type=int, default=16)
    output_generate.add_argument("--max-bytes", type=int, default=512)


def _add_orchestration_parsers(subparsers: Any) -> None:
    run_parser = subparsers.add_parser("run", help="execute a pinned experiment specification")
    run_parser.add_argument("spec", type=Path)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--verification", type=Path)
    run_parser.add_argument("--resume", action="store_true")

    deployment_parser = subparsers.add_parser(
        "deployment",
        help="measure sealed quality checkpoint deployment state",
    )
    deployment_parser.add_argument("spec", type=Path)
    deployment_parser.add_argument("--output-dir", type=Path, required=True)

    deployment_spec_parser = subparsers.add_parser(
        "deployment-spec",
        help="materialize deployment specifications from sealed quality runs",
    )
    deployment_spec_parser.add_argument("quality_runs", nargs="+", type=Path)
    deployment_spec_parser.add_argument("--output-dir", type=Path, required=True)

    search_parser = subparsers.add_parser("search", help="search registered tokenizer hyperparameters")
    search_parser.add_argument("spec", type=Path)
    search_parser.add_argument("--output-dir", type=Path, required=True)
    search_parser.add_argument("--resume", action="store_true")
    search_parser.add_argument("--prepare-only", action="store_true")
    search_parser.add_argument(
        "--oracle-study",
        type=Path,
        help="Sealed output-oracle study directory or result",
    )

    study_parser = subparsers.add_parser(
        "study",
        help="execute a registered non-final research study",
    )
    study_parser.add_argument("spec", type=Path)
    study_parser.add_argument("--output-dir", type=Path, required=True)
    study_parser.add_argument("--resume", action="store_true")

    freeze_parser = subparsers.add_parser(
        "freeze",
        help="materialize final specifications from current prospective selections",
    )
    freeze_parser.add_argument("artifacts", nargs="+", type=Path)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate", help="summarize immutable replication runs")
    aggregate_parser.add_argument("runs", nargs="+", type=Path)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)

    report_parser = subparsers.add_parser(
        "project-report",
        help="combine equal primary cross-model replications",
    )
    report_parser.add_argument("primary_replications", nargs=2, type=Path)
    report_parser.add_argument("--alignment-studies", nargs=2, type=Path, default=())
    report_parser.add_argument("--deployments", nargs=2, type=Path, default=())
    _add_software_validation_arguments(report_parser)
    report_parser.add_argument("--output-dir", type=Path, required=True)

    state_budget_parser = subparsers.add_parser(
        "state-budget",
        help="assemble a sealed cross-directional tensor-state budget",
    )
    state_budget_parser.add_argument("input_project", type=Path)
    state_budget_parser.add_argument("output_project", type=Path)
    state_budget_parser.add_argument("--output-dir", type=Path, required=True)

    verify_parser = subparsers.add_parser(
        "verify",
        help="record fast checks by default or the complete final-evidence inventory",
    )
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser.add_argument(
        "--complete",
        action="store_true",
        help="run every required final-evidence verification check",
    )
    verify_parser.add_argument("--slow", action="store_true", help="include slow compilation and synthetic checks")
    verify_parser.add_argument("--streamlit", action="store_true", help="include the Streamlit dashboard suite")
    verify_parser.add_argument("--model-tokenizers", action="store_true", help="include network-gated primary-model tokenizer checks")
    artifact_verify_parser = subparsers.add_parser(
        "verify-artifact",
        help="validate immutable evidence artifacts",
    )
    artifact_verify_parser.add_argument("artifact", type=Path)

    readme_parser = subparsers.add_parser(
        "readme",
        help="render the generated README evidence ledger",
    )
    readme_parser.add_argument(
        "--input-project",
        type=Path,
        help="Sealed input-only project artifact",
    )
    readme_parser.add_argument(
        "--output-project",
        type=Path,
        help="Sealed output-only project artifact",
    )
    _add_software_validation_arguments(readme_parser)
    readme_parser.add_argument("--check", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokenizer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_directional_parsers(subparsers)
    _add_orchestration_parsers(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers: dict[str, Any] = {
        "inspect": inspect_model,
        "train": train,
        "segment": segment,
        "benchmark": benchmark,
        "evaluate": evaluate,
        "attention": attention,
        "generate": generate,
        "run": run,
        "deployment": deployment,
        "deployment-spec": deployment_spec,
        "search": search,
        "study": study,
        "freeze": freeze,
        "aggregate": aggregate,
        "project-report": project_report,
        "state-budget": state_budget,
        "verify": verify,
        "verify-artifact": verify_artifact,
        "readme": readme,
    }
    command = args.direction_command if args.command in {"input", "output"} else args.command
    result = handlers[command](args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
