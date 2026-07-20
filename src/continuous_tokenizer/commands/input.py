from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from continuous_tokenizer.artifacts.hashing import sha256_file
from continuous_tokenizer.backbone.assets import load_model_assets
from continuous_tokenizer.codec.checkpoints import (
    cache_namespace,
    load_checkpoint,
)
from continuous_tokenizer.contracts.profiles import profile_named
from continuous_tokenizer.diagnostics.attention import AttentionOptions, capture_attention_artifact
from continuous_tokenizer.input.benchmark.run import BenchmarkOptions, benchmark_experiment
from continuous_tokenizer.input.evaluation import EvaluationOptions, evaluate_input_replacement
from continuous_tokenizer.input.segmentation import greedy_segment, reconstruct
from continuous_tokenizer.input.training.run import TrainingOptions, train_experiment
from continuous_tokenizer.runtime.device import default_device


def _training_options(args: argparse.Namespace, output_dir: Path) -> TrainingOptions:
    return TrainingOptions(
        output_dir=output_dir,
        profile=profile_named(args.profile),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        vocabulary_epochs=args.vocabulary_epochs,
        reconstruction_epochs=args.reconstruction_epochs,
        reconstruction_samples=args.reconstruction_samples,
        reconstruction_vocabulary_fraction=args.reconstruction_vocabulary_fraction,
        validation_bytes=args.validation_bytes,
        patience=args.patience,
        evaluation_interval=args.evaluation_interval,
        seed=args.seed,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_model_assets(args.model, args.revision)
    return asdict(train_experiment(assets, _training_options(args, args.output_dir)))


@torch.inference_mode()
def segment(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_checkpoint(args.checkpoint, device=default_device())
    codec = loaded.codec
    metadata = loaded.metadata
    if metadata.get("model_id") != args.model:
        raise ValueError("checkpoint model ID does not match the requested model")
    provided = sum(value is not None for value in (args.text, args.hex_bytes, args.file))
    if provided != 1:
        raise ValueError("segment requires exactly one of TEXT, --hex, or --file")
    if args.hex_bytes is not None:
        data = bytes.fromhex(args.hex_bytes)
        input_format = "hex"
    elif args.file is not None:
        data = args.file.read_bytes()
        input_format = "file"
    else:
        data = args.text.encode("utf-8")
        input_format = "utf8"
    cache = codec.encoding_cache if args.cache else None
    segments = greedy_segment(
        codec,
        data,
        cache=cache,
        namespace=cache_namespace(str(metadata["model_revision"]), sha256_file(args.checkpoint)),
    )
    return {
        "model_id": args.model,
        "checkpoint_model_id": metadata.get("model_id"),
        "input_bytes": len(data),
        "input_format": input_format,
        "spans": [item.data.hex() for item in segments],
        "span_lengths": [len(item.data) for item in segments],
        "round_trip": reconstruct(segments) == data,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_model_assets(args.model, args.revision)
    return benchmark_experiment(
        assets,
        args.checkpoint,
        args.output_dir,
        BenchmarkOptions(
            max_test_bytes=args.max_test_bytes,
            batch_size=args.batch_size,
            retrieval_rows=args.retrieval_rows,
            repetitions=args.repetitions,
        ),
    )


def _evaluation_options(args: argparse.Namespace, output_dir: Path) -> EvaluationOptions:
    return EvaluationOptions(
        output_dir=output_dir,
        samples=args.samples,
        prompt_tokens=args.prompt_tokens,
        continuation_tokens=args.continuation_tokens,
        generation_samples=args.generation_samples,
        max_new_tokens=args.max_new_tokens,
        performance_prompts=args.performance_prompts,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_model_assets(args.model, args.revision)
    return evaluate_input_replacement(
        assets,
        args.checkpoint,
        _evaluation_options(args, args.output_dir),
    )


def attention(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_model_assets(args.model, args.revision)
    return capture_attention_artifact(
        assets,
        args.checkpoint,
        AttentionOptions(
            output_dir=args.output_dir,
            text=args.text,
            max_tokens=args.max_tokens,
            segmentation_alignment=args.alignment,
        ),
    )
