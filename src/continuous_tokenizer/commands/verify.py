from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from continuous_tokenizer.artifacts.source import find_project_root
from continuous_tokenizer.diagnostics.verification import run_verification


def verify(args: argparse.Namespace) -> dict[str, Any]:
    result = run_verification(
        find_project_root(Path.cwd()),
        args.output_dir,
        slow=args.slow,
        streamlit=args.streamlit,
        model_tokenizers=args.model_tokenizers,
        complete=args.complete,
    )
    if not result["all_passed"]:
        raise RuntimeError(f"preflight verification failed; inspect {args.output_dir}")
    return result
