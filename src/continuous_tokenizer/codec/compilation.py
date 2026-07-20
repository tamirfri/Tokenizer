from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, cast

import torch

TOKENIZER_RECOMPILE_LIMIT: Final = 64
SEGMENTATION_STATIC_ROWS: Final = (1, 2, 4, 8)
DYNAMIC_SEGMENTATION_MAX_BYTES: Final = 32


def bounded_widths(max_span: int) -> tuple[int, ...]:
    widths = []
    width = 1
    while width < max_span:
        widths.append(width)
        width *= 2
    widths.append(max_span)
    return tuple(dict.fromkeys(widths))


def planned_input_graph_signatures(
    max_span: int,
    static_rows: tuple[int, ...],
) -> dict[str, tuple[str, ...]]:
    if not static_rows or any(rows < 1 for rows in static_rows):
        raise ValueError("compiled static row counts must be positive")
    widths = bounded_widths(max_span)
    signatures: dict[str, set[str]] = {
        operation: {f"{rows}x{width}" for rows in static_rows for width in widths} for operation in ("encode", "reconstruction", "validation")
    }
    signatures["decode"] = {f"{rows}x{width + 1}" for rows in static_rows for width in widths}
    signatures["matches"] = set(signatures["validation"])
    for width in bounded_widths(min(max_span, DYNAMIC_SEGMENTATION_MAX_BYTES)):
        if width == 1:
            continue
        for frontier in SEGMENTATION_STATIC_ROWS:
            rows = frontier * max(1, width // 2)
            signature = f"{rows}x{width}"
            signatures["encode"].add(signature)
            signatures["matches"].add(signature)
            signatures["validation"].add(signature)
    planned = {operation: tuple(sorted(values)) for operation, values in signatures.items()}
    oversized = {operation: len(values) for operation, values in planned.items() if len(values) > TOKENIZER_RECOMPILE_LIMIT}
    if oversized:
        raise ValueError(f"planned tokenizer graph signatures exceed the {TOKENIZER_RECOMPILE_LIMIT}-signature policy: {oversized}")
    return planned


def compile_fullgraph[**P, R](
    function: Callable[P, R],
    *,
    backend: str,
) -> Callable[P, R]:
    compiled = torch.compile(
        function,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )

    def invoke(*args: P.args, **kwargs: P.kwargs) -> R:
        compiler_config = cast(Any, torch.compiler.config)
        with compiler_config.patch(recompile_limit=TOKENIZER_RECOMPILE_LIMIT):
            return compiled(*args, **kwargs)

    return invoke
