from __future__ import annotations

from continuous_tokenizer.cli import build_parser


def test_cli_exposes_every_planned_command() -> None:
    parser = build_parser()

    for command in ("inspect", "train", "segment", "benchmark", "run-all"):
        args = [command, "model"]
        if command == "train":
            args += ["--output-dir", "out"]
        elif command == "segment":
            args += ["checkpoint.pt", "text"]
        elif command == "benchmark":
            args += ["checkpoint.pt"]
        elif command == "run-all":
            args += ["--output-dir", "out"]
        assert parser.parse_args(args).command == command
