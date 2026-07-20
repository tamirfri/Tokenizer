from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from continuous_tokenizer.benchmark import benchmark_experiment
from continuous_tokenizer.checkpoint import checkpoint_fingerprint, load_checkpoint
from continuous_tokenizer.model_assets import load_model_assets, resolve_tokenizer_assets
from continuous_tokenizer.segmenter import greedy_segment, reconstruct
from continuous_tokenizer.training import TrainingOptions, train_experiment
from continuous_tokenizer.vocabulary import inspect_tokenizer


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Hugging Face model ID")
    parser.add_argument("--revision", help="Optional model revision")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=("auto", "small", "medium", "large"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage2-samples", type=int, default=250_000)
    parser.add_argument("--validation-bytes", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokenizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate a model byte vocabulary")
    _add_model_argument(inspect_parser)

    train_parser = subparsers.add_parser("train", help="train a codec against an embedding table")
    _add_model_argument(train_parser)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    _add_training_arguments(train_parser)

    segment_parser = subparsers.add_parser("segment", help="segment UTF-8 text with a codec")
    _add_model_argument(segment_parser)
    segment_parser.add_argument("checkpoint", type=Path)
    segment_parser.add_argument("text")
    segment_parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)

    benchmark_parser = subparsers.add_parser("benchmark", help="benchmark a trained codec")
    _add_model_argument(benchmark_parser)
    benchmark_parser.add_argument("checkpoint", type=Path)
    benchmark_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    benchmark_parser.add_argument("--max-test-bytes", type=int, default=16_384)
    benchmark_parser.add_argument("--batch-size", type=int, default=32)

    run_all_parser = subparsers.add_parser("run-all", help="inspect, train, and benchmark")
    _add_model_argument(run_all_parser)
    run_all_parser.add_argument("--output-dir", type=Path, required=True)
    _add_training_arguments(run_all_parser)
    return parser


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    assets = resolve_tokenizer_assets(args.model, args.revision)
    embedding_rows = int(assets.config["vocab_size"])
    vocabulary = inspect_tokenizer(assets.tokenizer, embedding_rows=embedding_rows)
    return {
        "model_id": assets.model_id,
        "revision": assets.revision,
        "hidden_size": int(assets.config["hidden_size"]),
        "tie_word_embeddings": bool(assets.config.get("tie_word_embeddings", False)),
        **vocabulary.to_summary(),
    }


def _training_options(args: argparse.Namespace, output_dir: Path) -> TrainingOptions:
    return TrainingOptions(
        output_dir=output_dir,
        profile=args.profile,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage2_samples=args.stage2_samples,
        validation_bytes=args.validation_bytes,
        patience=args.patience,
        seed=args.seed,
    )


def _train(args: argparse.Namespace) -> list[dict[str, Any]]:
    assets = load_model_assets(args.model, args.revision)
    results = train_experiment(assets, _training_options(args, args.output_dir))
    return [asdict(result) for result in results]


@torch.inference_mode()
def _segment(args: argparse.Namespace) -> dict[str, Any]:
    codec, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("model_id") != args.model:
        raise ValueError("checkpoint model ID does not match the requested model")
    data = args.text.encode("utf-8")
    cache = codec.encoding_cache if args.cache else None
    segments = greedy_segment(
        codec,
        data,
        cache=cache,
        namespace=checkpoint_fingerprint(args.checkpoint),
    )
    return {
        "model_id": args.model,
        "checkpoint_model_id": metadata.get("model_id"),
        "input_bytes": len(data),
        "spans": [segment.data.hex() for segment in segments],
        "span_lengths": [len(segment.data) for segment in segments],
        "round_trip": reconstruct(segments) == data,
    }


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_model_assets(args.model, args.revision)
    return benchmark_experiment(
        assets,
        args.checkpoint,
        args.output_dir,
        max_test_bytes=args.max_test_bytes,
        batch_size=args.batch_size,
    )


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_model_assets(args.model, args.revision)
    checkpoints_dir = args.output_dir / "checkpoints"
    training_results = train_experiment(
        assets,
        _training_options(args, checkpoints_dir),
    )
    selected = next((result for result in training_results if result.passed), training_results[-1])
    metrics = benchmark_experiment(
        assets,
        Path(selected.checkpoint),
        args.output_dir,
        batch_size=args.batch_size,
    )
    return {
        "inspection": assets.vocabulary.to_summary(),
        "training": [asdict(result) for result in training_results],
        "benchmark": metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "inspect": _inspect,
        "train": _train,
        "segment": _segment,
        "benchmark": _benchmark,
        "run-all": _run_all,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
