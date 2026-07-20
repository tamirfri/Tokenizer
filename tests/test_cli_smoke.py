from __future__ import annotations

import unittest
from pathlib import Path

from continuous_tokenizer.cli import build_parser
from continuous_tokenizer.commands.evidence import _software_validation_paths
from continuous_tokenizer.contracts.profiles import CAMPAIGN_PROFILE_NAME


def test_cli_exposes_every_planned_command() -> None:
    parser = build_parser()

    for command in (
        "inspect",
        "train",
        "segment",
        "benchmark",
        "evaluate",
        "attention",
        "run",
        "search",
        "aggregate",
        "project-report",
        "verify",
    ):
        args = [command, "model"]
        if command == "train":
            args += ["--output-dir", "out"]
        elif command == "segment":
            args += ["checkpoint.pt", "text"]
        elif command in {"benchmark", "evaluate"}:
            args += ["checkpoint.pt"]
        elif command == "attention":
            args += ["checkpoint.pt", "short text"]
        elif command in {"run", "search", "aggregate", "project-report"}:
            if command == "project-report":
                args.append("second-model-replication")
            args += ["--output-dir", "out"]
        elif command == "verify":
            args = [command, "--output-dir", "out"]
        assert parser.parse_args(args).command == command
    assert parser.parse_args(["train", "model", "--output-dir", "out"]).profile == CAMPAIGN_PROFILE_NAME


def test_paper_surfaces_require_complete_validation_bundle() -> None:
    parser = build_parser()
    readme = parser.parse_args(
        [
            "readme",
            "--verification",
            "verification/verification.json",
            "--input-synthetic",
            "input-synthetic",
            "--output-synthetic",
            "output-synthetic",
        ]
    )
    assert _software_validation_paths(readme) == (
        Path("verification/verification.json"),
        Path("input-synthetic"),
        Path("output-synthetic"),
    )
    partial = parser.parse_args(["readme", "--verification", "verification.json"])
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "requires --verification, --input-synthetic, and --output-synthetic",
    ):
        _software_validation_paths(partial)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_cli_exposes_every_planned_command,
            test_paper_surfaces_require_complete_validation_bundle,
        )
    )
