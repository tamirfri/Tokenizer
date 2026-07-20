from __future__ import annotations

import argparse
from typing import Any

from continuous_tokenizer.backbone.assets import load_frozen_causal_lm, load_model_assets
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.codec.checkpoints import load_output_checkpoint
from continuous_tokenizer.output.generation import (
    OutputOnlyGenerator,
    output_stop_control_ids,
    output_stop_control_metadata,
)
from continuous_tokenizer.runtime.device import default_device


def generate(args: argparse.Namespace) -> dict[str, Any]:
    device = default_device()
    assets = load_model_assets(args.model, args.revision)
    loaded = load_output_checkpoint(args.checkpoint, device=device)
    if loaded.metadata.get("model_revision") != assets.revision:
        raise ValueError("checkpoint and source model revisions do not match")
    model = load_frozen_causal_lm(assets, device)
    generator = OutputOnlyGenerator(
        FrozenBackbone(model),
        loaded.codec,
        assets.vocabulary,
        loaded.control_ids,
    )
    prompt_ids = tuple(assets.tokenizer.encode(args.prompt, add_special_tokens=False))
    stop_control_ids = output_stop_control_ids(assets.tokenizer, assets.vocabulary)
    result = generator.generate(
        prompt_ids,
        stop_control_ids=stop_control_ids,
        max_macro_steps=args.max_macro_steps,
        max_bytes=args.max_bytes,
    )
    return {
        "mode": "output_only",
        "data_hex": result.data.hex(),
        "text": result.data.decode("utf-8", errors="replace"),
        "macro_steps": result.macro_steps,
        "native_tokens_represented": result.native_tokens_represented,
        "invalid_events": result.invalid_events,
        "stop_control": output_stop_control_metadata(
            assets.tokenizer,
            assets.vocabulary,
        ),
    }
