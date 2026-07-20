from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from continuous_tokenizer.batching import build_span_batch, byte_reconstruction_loss
from continuous_tokenizer.checkpoint import save_checkpoint
from continuous_tokenizer.codec import CodecConfig, ContinuousByteCodec
from continuous_tokenizer.config import PROFILES, Profile, profile_named
from continuous_tokenizer.corpus import (
    DATASET_CONFIG,
    DATASET_ID,
    joined_prefix,
    load_wikitext_documents,
    sample_spans,
)
from continuous_tokenizer.metrics import (
    evaluate_embeddings,
    module_bytes,
    tensor_bytes,
)
from continuous_tokenizer.model_assets import ModelAssets
from continuous_tokenizer.runtime import default_device
from continuous_tokenizer.segmenter import greedy_segment, reconstruct


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    output_dir: Path
    profile: str = "auto"
    batch_size: int = 32
    learning_rate: float = 3e-4
    stage1_epochs: int = 100
    stage2_epochs: int = 10
    stage2_samples: int = 250_000
    validation_bytes: int = 4096
    patience: int = 5
    seed: int = 17


@dataclass(frozen=True, slots=True)
class ProfileResult:
    profile: str
    checkpoint: str
    embedding_metrics: dict[str, int | float]
    codec_bytes: int
    source_table_bytes: int
    memory_ratio: float
    density_ratio: float
    round_trip: bool
    passed: bool


def build_codec(assets: ModelAssets, profile: Profile, device: torch.device) -> ContinuousByteCodec:
    vocabulary = assets.vocabulary
    byte_ids = torch.tensor(vocabulary.byte_token_ids, dtype=torch.long)
    control_ids = torch.tensor(vocabulary.control_ids, dtype=torch.long)
    byte_embeddings = assets.input_embeddings[byte_ids].float()
    control_embeddings = assets.input_embeddings[control_ids].float()
    max_span = max(vocabulary.max_token_bytes, 64)
    config = CodecConfig(
        embedding_dim=assets.input_embeddings.shape[1],
        local_dim=profile.local_dim,
        max_span=max_span,
        heads=profile.heads,
        feedforward_dim=profile.feedforward_dim,
        encoder_layers=profile.encoder_layers,
        decoder_layers=profile.decoder_layers,
    )
    return ContinuousByteCodec(config, byte_embeddings, control_ids, control_embeddings).to(device)


def _normalized_embedding_loss(latent: Tensor, target: Tensor) -> Tensor:
    numerator = F.mse_loss(latent.float(), target.float())
    denominator = target.float().square().mean().clamp_min(1e-12)
    return numerator / denominator


def _train_vocabulary_epoch(
    codec: ContinuousByteCodec,
    assets: ModelAssets,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    codec.train()
    ids = torch.tensor(assets.vocabulary.ordinary_ids, dtype=torch.long)
    order = ids[torch.randperm(len(ids), generator=generator)]
    total = 0.0
    batches = 0
    for batch_ids_tensor in order.split(batch_size):
        batch_ids = batch_ids_tensor.tolist()
        spans = [assets.vocabulary.bytes_for(token_id) for token_id in batch_ids]
        batch = build_span_batch(spans, max_span=codec.max_span, device=device)
        latent, logits = codec(batch.byte_values, batch.valid_mask)
        target = assets.input_embeddings[batch_ids].to(device=device, dtype=latent.dtype)
        loss = _normalized_embedding_loss(latent, target)
        loss = loss + byte_reconstruction_loss(logits, batch.framed_targets, batch.target_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().item())
        batches += 1
    return total / max(batches, 1)


def _train_mixed_epoch(
    codec: ContinuousByteCodec,
    assets: ModelAssets,
    corpus_spans: list[bytes],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    randomizer: random.Random,
) -> float:
    codec.train()
    ordinary_ids = assets.vocabulary.ordinary_ids
    total = 0.0
    batches = 0
    randomizer.shuffle(corpus_spans)
    half = max(1, batch_size // 2)
    for start in range(0, len(corpus_spans), half):
        dynamic = corpus_spans[start : start + half]
        vocab_ids = randomizer.choices(ordinary_ids, k=len(dynamic))
        vocabulary_spans = [assets.vocabulary.bytes_for(token_id) for token_id in vocab_ids]
        spans = vocabulary_spans + dynamic
        batch = build_span_batch(spans, max_span=codec.max_span, device=device)
        latent, logits = codec(batch.byte_values, batch.valid_mask)
        target = assets.input_embeddings[vocab_ids].to(device=device, dtype=latent.dtype)
        loss = _normalized_embedding_loss(latent[: len(vocab_ids)], target)
        loss = loss + byte_reconstruction_loss(logits, batch.framed_targets, batch.target_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().item())
        batches += 1
    return total / max(batches, 1)


@torch.inference_mode()
def _density_metrics(
    codec: ContinuousByteCodec,
    tokenizer: Any,
    data: bytes,
    *,
    namespace: str,
) -> tuple[float, bool]:
    codec.eval()
    text = data.decode("utf-8", errors="strict")
    original_ids = tokenizer.encode(text, add_special_tokens=False)
    segments = greedy_segment(codec, data, namespace=namespace)
    ratio = len(original_ids) / max(len(segments), 1)
    return ratio, reconstruct(segments) == data


def _experiment_metadata(assets: ModelAssets) -> dict[str, Any]:
    return {
        "model_id": assets.model_id,
        "model_revision": assets.revision,
        "embedding_tensor_name": assets.embedding_tensor_name,
        "source_dtype": str(assets.input_embeddings.dtype),
        "source_shape": list(assets.input_embeddings.shape),
        "tie_word_embeddings": bool(assets.config.get("tie_word_embeddings", False)),
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG},
    }


def _serialized_options(options: TrainingOptions) -> dict[str, Any]:
    return {**asdict(options), "output_dir": str(options.output_dir)}


def _metadata(assets: ModelAssets, profile: Profile, options: TrainingOptions) -> dict[str, Any]:
    return {
        **_experiment_metadata(assets),
        "profile": asdict(profile),
        "training": {
            **_serialized_options(options),
            "python": sys.version,
            "torch": str(torch.__version__),
        },
    }


def _run_manifest(assets: ModelAssets, options: TrainingOptions) -> dict[str, Any]:
    return {
        **_experiment_metadata(assets),
        "embedding_shard": assets.embedding_shard.name,
        "source_table_bytes": tensor_bytes(assets.input_embeddings),
        "options": _serialized_options(options),
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "transformers": version("transformers"),
            "huggingface_hub": version("huggingface-hub"),
            "datasets": version("datasets"),
        },
    }


def _profile_sequence(name: str) -> tuple[Profile, ...]:
    if name == "auto":
        return PROFILES
    return (profile_named(name),)


def train_experiment(
    assets: ModelAssets,
    options: TrainingOptions,
    *,
    device: torch.device | None = None,
) -> list[ProfileResult]:
    selected_device = default_device() if device is None else device
    torch.manual_seed(options.seed)
    generator = torch.Generator().manual_seed(options.seed)
    randomizer = random.Random(options.seed)
    train_documents = load_wikitext_documents("train") if options.stage2_epochs else []
    validation_documents = load_wikitext_documents("validation")
    corpus_spans = (
        sample_spans(
            train_documents,
            count=options.stage2_samples,
            seed=options.seed,
            maximum=64,
        )
        if options.stage2_epochs
        else []
    )
    validation_data = joined_prefix(validation_documents, max_bytes=options.validation_bytes)
    source_bytes = tensor_bytes(assets.input_embeddings)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    with (options.output_dir / "run-manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_run_manifest(assets, options), handle, indent=2, sort_keys=True)
    results: list[ProfileResult] = []

    for profile in _profile_sequence(options.profile):
        codec = build_codec(assets, profile, selected_device)
        optimizer = torch.optim.AdamW(codec.parameters(), lr=options.learning_rate)
        best_exact = -1
        stale_epochs = 0

        for _epoch in range(options.stage1_epochs):
            _train_vocabulary_epoch(
                codec,
                assets,
                optimizer,
                batch_size=options.batch_size,
                device=selected_device,
                generator=generator,
            )
            metrics = evaluate_embeddings(
                codec,
                assets.vocabulary,
                assets.input_embeddings,
                batch_size=options.batch_size,
                device=selected_device,
            )
            if metrics.exact_rows > best_exact:
                best_exact = metrics.exact_rows
                stale_epochs = 0
            else:
                stale_epochs += 1
            if metrics.exact_fraction == 1.0 or stale_epochs >= options.patience:
                break

        for _epoch in range(options.stage2_epochs):
            _train_mixed_epoch(
                codec,
                assets,
                corpus_spans,
                optimizer,
                batch_size=options.batch_size,
                device=selected_device,
                randomizer=randomizer,
            )

        metrics = evaluate_embeddings(
            codec,
            assets.vocabulary,
            assets.input_embeddings,
            batch_size=options.batch_size,
            device=selected_device,
        )
        density_ratio, round_trip = _density_metrics(
            codec, assets.tokenizer, validation_data, namespace=assets.revision
        )
        checkpoint_path = options.output_dir / f"{profile.name}.pt"
        save_checkpoint(checkpoint_path, codec, _metadata(assets, profile, options))
        codec_size = module_bytes(codec)
        memory_ratio = codec_size / source_bytes
        passed = (
            metrics.exact_fraction == 1.0
            and metrics.reconstruction_fraction == 1.0
            and memory_ratio <= 0.5
            and round_trip
            and density_ratio >= 1.1
        )
        result = ProfileResult(
            profile=profile.name,
            checkpoint=str(checkpoint_path),
            embedding_metrics=metrics.to_dict(),
            codec_bytes=codec_size,
            source_table_bytes=source_bytes,
            memory_ratio=memory_ratio,
            density_ratio=density_ratio,
            round_trip=round_trip,
            passed=passed,
        )
        results.append(result)
        with (options.output_dir / "training-results.json").open("w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in results], handle, indent=2, sort_keys=True)
        if passed:
            break
    return results
