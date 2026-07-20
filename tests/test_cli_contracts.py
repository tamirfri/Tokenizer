from __future__ import annotations

import argparse
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, cast

from continuous_tokenizer.cli import build_parser

_MODEL = (
    ("model", (), None, True, None, None, None, "Hugging Face model ID", "_StoreAction"),
    (
        "revision",
        ("--revision",),
        None,
        False,
        None,
        None,
        None,
        "Optional model revision",
        "_StoreAction",
    ),
)


_CHECKPOINT = ("checkpoint", (), None, True, None, "Path", None, None, "_StoreAction")


_REQUIRED_OUTPUT_DIR = (
    "output_dir",
    ("--output-dir",),
    None,
    True,
    None,
    "Path",
    None,
    None,
    "_StoreAction",
)


_RESULTS_OUTPUT_DIR = (
    "output_dir",
    ("--output-dir",),
    None,
    False,
    "Path:results",
    "Path",
    None,
    None,
    "_StoreAction",
)


_SOFTWARE_VALIDATION = (
    (
        "verification",
        ("--verification",),
        None,
        False,
        None,
        "Path",
        None,
        None,
        "_StoreAction",
    ),
    (
        "input_synthetic",
        ("--input-synthetic",),
        None,
        False,
        None,
        "Path",
        None,
        None,
        "_StoreAction",
    ),
    (
        "output_synthetic",
        ("--output-synthetic",),
        None,
        False,
        None,
        "Path",
        None,
        None,
        "_StoreAction",
    ),
)


_TRAINING = (
    (
        "profile",
        ("--profile",),
        None,
        False,
        "large",
        None,
        ("small", "large"),
        None,
        "_StoreAction",
    ),
    ("batch_size", ("--batch-size",), None, False, 32, "int", None, None, "_StoreAction"),
    (
        "learning_rate",
        ("--learning-rate",),
        None,
        False,
        3e-4,
        "float",
        None,
        None,
        "_StoreAction",
    ),
    (
        "weight_decay",
        ("--weight-decay",),
        None,
        False,
        0.0,
        "float",
        None,
        None,
        "_StoreAction",
    ),
    (
        "vocabulary_epochs",
        ("--vocabulary-epochs",),
        None,
        False,
        10,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    (
        "reconstruction_epochs",
        ("--reconstruction-epochs",),
        None,
        False,
        1,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    (
        "reconstruction_samples",
        ("--reconstruction-samples",),
        None,
        False,
        2_048,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    (
        "reconstruction_vocabulary_fraction",
        ("--reconstruction-vocabulary-fraction",),
        None,
        False,
        0.75,
        "float",
        None,
        None,
        "_StoreAction",
    ),
    (
        "validation_bytes",
        ("--validation-bytes",),
        None,
        False,
        2_048,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    ("patience", ("--patience",), None, False, 2, "int", None, None, "_StoreAction"),
    (
        "evaluation_interval",
        ("--evaluation-interval",),
        None,
        False,
        2,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    ("seed", ("--seed",), None, False, 17, "int", None, None, "_StoreAction"),
)


_TRAIN = (*_MODEL, _REQUIRED_OUTPUT_DIR, *_TRAINING)


_SEGMENT = (
    *_MODEL,
    _CHECKPOINT,
    ("text", (), "?", False, None, None, None, None, "_StoreAction"),
    ("hex_bytes", ("--hex",), None, False, None, None, None, None, "_StoreAction"),
    ("file", ("--file",), None, False, None, "Path", None, None, "_StoreAction"),
    (
        "cache",
        ("--cache", "--no-cache"),
        0,
        False,
        True,
        None,
        None,
        None,
        "BooleanOptionalAction",
    ),
)


_BENCHMARK = (
    *_MODEL,
    _CHECKPOINT,
    _RESULTS_OUTPUT_DIR,
    (
        "max_test_bytes",
        ("--max-test-bytes",),
        None,
        False,
        2_048,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    ("batch_size", ("--batch-size",), None, False, 32, "int", None, None, "_StoreAction"),
    (
        "retrieval_rows",
        ("--retrieval-rows",),
        None,
        False,
        128,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    ("repetitions", ("--repetitions",), None, False, 2, "int", None, None, "_StoreAction"),
)


_EVALUATE = (
    *_MODEL,
    _CHECKPOINT,
    _RESULTS_OUTPUT_DIR,
    ("samples", ("--samples",), None, False, 16, "int", None, None, "_StoreAction"),
    (
        "prompt_tokens",
        ("--prompt-tokens",),
        None,
        False,
        64,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    (
        "continuation_tokens",
        ("--continuation-tokens",),
        None,
        False,
        16,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    (
        "generation_samples",
        ("--generation-samples",),
        None,
        False,
        2,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    (
        "max_new_tokens",
        ("--max-new-tokens",),
        None,
        False,
        16,
        "int",
        None,
        None,
        "_StoreAction",
    ),
    (
        "performance_prompts",
        ("--performance-prompts",),
        None,
        False,
        2,
        "int",
        None,
        None,
        "_StoreAction",
    ),
)


_ATTENTION = (
    *_MODEL,
    _CHECKPOINT,
    ("text", (), None, True, None, None, None, None, "_StoreAction"),
    _RESULTS_OUTPUT_DIR,
    ("max_tokens", ("--max-tokens",), None, False, 32, "int", None, None, "_StoreAction"),
    (
        "alignment",
        ("--alignment",),
        None,
        False,
        "arbitrary",
        None,
        ("aligned", "arbitrary"),
        None,
        "_StoreAction",
    ),
)


_COMMAND_ARGUMENTS = {
    ("inspect",): _MODEL,
    ("train",): _TRAIN,
    ("segment",): _SEGMENT,
    ("benchmark",): _BENCHMARK,
    ("evaluate",): _EVALUATE,
    ("attention",): _ATTENTION,
    ("input", "inspect"): _MODEL,
    ("input", "train"): _TRAIN,
    ("input", "segment"): _SEGMENT,
    ("input", "benchmark"): _BENCHMARK,
    ("input", "evaluate"): _EVALUATE,
    ("input", "attention"): _ATTENTION,
    ("output", "generate"): (
        *_MODEL,
        _CHECKPOINT,
        ("prompt", (), None, True, None, None, None, None, "_StoreAction"),
        (
            "max_macro_steps",
            ("--max-macro-steps",),
            None,
            False,
            16,
            "int",
            None,
            None,
            "_StoreAction",
        ),
        ("max_bytes", ("--max-bytes",), None, False, 512, "int", None, None, "_StoreAction"),
    ),
    ("run",): (
        ("spec", (), None, True, None, "Path", None, None, "_StoreAction"),
        _REQUIRED_OUTPUT_DIR,
        (
            "verification",
            ("--verification",),
            None,
            False,
            None,
            "Path",
            None,
            None,
            "_StoreAction",
        ),
        ("resume", ("--resume",), 0, False, False, None, None, None, "_StoreTrueAction"),
    ),
    ("deployment",): (
        ("spec", (), None, True, None, "Path", None, None, "_StoreAction"),
        _REQUIRED_OUTPUT_DIR,
    ),
    ("deployment-spec",): (
        (
            "quality_runs",
            (),
            "+",
            True,
            None,
            "Path",
            None,
            None,
            "_StoreAction",
        ),
        _REQUIRED_OUTPUT_DIR,
    ),
    ("search",): (
        ("spec", (), None, True, None, "Path", None, None, "_StoreAction"),
        _REQUIRED_OUTPUT_DIR,
        ("resume", ("--resume",), 0, False, False, None, None, None, "_StoreTrueAction"),
        (
            "prepare_only",
            ("--prepare-only",),
            0,
            False,
            False,
            None,
            None,
            None,
            "_StoreTrueAction",
        ),
        (
            "oracle_study",
            ("--oracle-study",),
            None,
            False,
            None,
            "Path",
            None,
            "Sealed output-oracle study directory or result",
            "_StoreAction",
        ),
    ),
    ("study",): (
        ("spec", (), None, True, None, "Path", None, None, "_StoreAction"),
        _REQUIRED_OUTPUT_DIR,
        (
            "resume",
            ("--resume",),
            0,
            False,
            False,
            None,
            None,
            None,
            "_StoreTrueAction",
        ),
    ),
    ("freeze",): (
        ("artifacts", (), "+", True, None, "Path", None, None, "_StoreAction"),
        _REQUIRED_OUTPUT_DIR,
    ),
    ("aggregate",): (
        ("runs", (), "+", True, None, "Path", None, None, "_StoreAction"),
        _REQUIRED_OUTPUT_DIR,
    ),
    ("project-report",): (
        (
            "primary_replications",
            (),
            2,
            True,
            None,
            "Path",
            None,
            None,
            "_StoreAction",
        ),
        (
            "alignment_studies",
            ("--alignment-studies",),
            2,
            False,
            (),
            "Path",
            None,
            None,
            "_StoreAction",
        ),
        (
            "deployments",
            ("--deployments",),
            2,
            False,
            (),
            "Path",
            None,
            None,
            "_StoreAction",
        ),
        *_SOFTWARE_VALIDATION,
        _REQUIRED_OUTPUT_DIR,
    ),
    ("state-budget",): (
        (
            "input_project",
            (),
            None,
            True,
            None,
            "Path",
            None,
            None,
            "_StoreAction",
        ),
        (
            "output_project",
            (),
            None,
            True,
            None,
            "Path",
            None,
            None,
            "_StoreAction",
        ),
        _REQUIRED_OUTPUT_DIR,
    ),
    ("verify",): (
        _REQUIRED_OUTPUT_DIR,
        (
            "complete",
            ("--complete",),
            0,
            False,
            False,
            None,
            None,
            "run every required final-evidence verification check",
            "_StoreTrueAction",
        ),
        (
            "slow",
            ("--slow",),
            0,
            False,
            False,
            None,
            None,
            "include slow compilation and synthetic checks",
            "_StoreTrueAction",
        ),
        (
            "streamlit",
            ("--streamlit",),
            0,
            False,
            False,
            None,
            None,
            "include the Streamlit dashboard suite",
            "_StoreTrueAction",
        ),
        (
            "model_tokenizers",
            ("--model-tokenizers",),
            0,
            False,
            False,
            None,
            None,
            "include network-gated primary-model tokenizer checks",
            "_StoreTrueAction",
        ),
    ),
    ("verify-artifact",): (
        (
            "artifact",
            (),
            None,
            True,
            None,
            "Path",
            None,
            None,
            "_StoreAction",
        ),
    ),
    ("readme",): (
        (
            "input_project",
            ("--input-project",),
            None,
            False,
            None,
            "Path",
            None,
            "Sealed input-only project artifact",
            "_StoreAction",
        ),
        (
            "output_project",
            ("--output-project",),
            None,
            False,
            None,
            "Path",
            None,
            "Sealed output-only project artifact",
            "_StoreAction",
        ),
        *_SOFTWARE_VALIDATION,
        (
            "check",
            ("--check",),
            0,
            False,
            False,
            None,
            None,
            None,
            "_StoreTrueAction",
        ),
    ),
}


_COMMAND_HELP = {
    "inspect": "validate a model byte vocabulary",
    "train": "train the input tokenizer against a frozen embedding table",
    "segment": "segment text or raw bytes with a trained tokenizer",
    "benchmark": "benchmark a trained tokenizer",
    "evaluate": "compare frozen-model input paths",
    "attention": "capture native and segmented attention diagnostics",
    "input": "input-only tokenizer diagnostics",
    "output": "output-only tokenizer diagnostics",
    "run": "execute a pinned experiment specification",
    "deployment": "measure sealed quality checkpoint deployment state",
    "deployment-spec": "materialize deployment specifications from sealed quality runs",
    "search": "search registered tokenizer hyperparameters",
    "study": "execute a registered non-final research study",
    "freeze": "materialize final specifications from current prospective selections",
    "aggregate": "summarize immutable replication runs",
    "project-report": "combine equal primary cross-model replications",
    "state-budget": "assemble a sealed cross-directional tensor-state budget",
    "verify": "record fast checks by default or the complete final-evidence inventory",
    "verify-artifact": "validate immutable evidence artifacts",
    "readme": "render the generated README evidence ledger",
}


_MINIMUM_ARGUMENTS = {
    ("inspect",): ("model",),
    ("train",): ("model", "--output-dir", "out"),
    ("segment",): ("model", "checkpoint.pt", "text"),
    ("benchmark",): ("model", "checkpoint.pt"),
    ("evaluate",): ("model", "checkpoint.pt"),
    ("attention",): ("model", "checkpoint.pt", "text"),
    ("input", "inspect"): ("model",),
    ("input", "train"): ("model", "--output-dir", "out"),
    ("input", "segment"): ("model", "checkpoint.pt", "text"),
    ("input", "benchmark"): ("model", "checkpoint.pt"),
    ("input", "evaluate"): ("model", "checkpoint.pt"),
    ("input", "attention"): ("model", "checkpoint.pt", "text"),
    ("output", "generate"): ("model", "checkpoint.pt", "prompt"),
    ("run",): ("experiment.toml", "--output-dir", "out"),
    ("deployment",): ("deployment.toml", "--output-dir", "out"),
    ("deployment-spec",): ("quality-run", "--output-dir", "out"),
    ("search",): ("search.toml", "--output-dir", "out"),
    ("study",): ("study.toml", "--output-dir", "out"),
    ("freeze",): ("artifact", "--output-dir", "out"),
    ("aggregate",): ("run-17", "--output-dir", "out"),
    ("project-report",): (
        "qwen-replication",
        "gemma-replication",
        "--output-dir",
        "out",
    ),
    ("state-budget",): (
        "input-project",
        "output-project",
        "--output-dir",
        "out",
    ),
    ("verify",): ("--output-dir", "out"),
    ("verify-artifact",): ("artifact",),
    ("readme",): (),
}


def _normalized_default(value: object) -> object:
    return f"Path:{value}" if isinstance(value, Path) else value


def _parser_actions(parser: argparse.ArgumentParser) -> list[Any]:
    return cast(list[Any], parser._actions)


def _action_contract(parser: argparse.ArgumentParser) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            action.dest,
            tuple(action.option_strings),
            action.nargs,
            action.required,
            _normalized_default(action.default),
            None if action.type is None else getattr(action.type, "__name__", str(action.type)),
            None if action.choices is None else tuple(action.choices),
            action.help,
            type(action).__name__,
        )
        for action in _parser_actions(parser)
        if action.dest != "help" and type(action).__name__ != "_SubParsersAction"
    )


def _command_parsers(
    parser: argparse.ArgumentParser,
    path: tuple[str, ...] = (),
) -> dict[tuple[str, ...], argparse.ArgumentParser]:
    result: dict[tuple[str, ...], argparse.ArgumentParser] = {}
    for action in _parser_actions(parser):
        if type(action).__name__ != "_SubParsersAction":
            continue
        for name, child in action.choices.items():
            command_path = (*path, name)
            result[command_path] = child
            result.update(_command_parsers(child, command_path))
    return result


class CliContractTests(unittest.TestCase):
    def test_public_command_arguments_defaults_and_types_are_frozen(self) -> None:
        parser = build_parser()
        parsers = _command_parsers(parser)

        self.assertEqual(set(parsers), set(_COMMAND_ARGUMENTS) | {("input",), ("output",)})
        for path, expected in _COMMAND_ARGUMENTS.items():
            with self.subTest(command=" ".join(path)):
                self.assertEqual(_action_contract(parsers[path]), expected)
                parsed = parser.parse_args([*path, *_MINIMUM_ARGUMENTS[path]])
                self.assertEqual(parsed.command, path[0])

    def test_top_level_command_spellings_and_help_summaries_are_frozen(self) -> None:
        parser = build_parser()
        subparsers = next(action for action in _parser_actions(parser) if type(action).__name__ == "_SubParsersAction")

        self.assertEqual(tuple(subparsers.choices), tuple(_COMMAND_HELP))
        self.assertEqual(
            {action.dest: action.help for action in subparsers._choices_actions},
            _COMMAND_HELP,
        )

    def test_every_public_command_supports_help_without_running_it(self) -> None:
        parser = build_parser()
        for path in ((), *_command_parsers(parser)):
            stdout = StringIO()
            stderr = StringIO()
            with (
                self.subTest(command=" ".join(path) or "tokenizer"),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                parser.parse_args([*path, "--help"])
            self.assertEqual(raised.exception.code, 0)
            help_text = stdout.getvalue()
            self.assertTrue(help_text.startswith(f"usage: tokenizer {' '.join(path)}".rstrip()))
            self.assertIn("-h, --help", help_text)
            self.assertEqual(stderr.getvalue(), "")

    def test_top_level_input_commands_remain_aliases_for_nested_commands(self) -> None:
        parser = build_parser()
        for command in ("inspect", "train", "segment", "benchmark", "evaluate", "attention"):
            arguments = _MINIMUM_ARGUMENTS[command,]
            direct = vars(parser.parse_args([command, *arguments]))
            nested = vars(parser.parse_args(["input", command, *arguments]))
            direct.pop("command")
            nested.pop("command")
            nested.pop("direction_command")
            self.assertEqual(direct, nested)
