from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch

from continuous_tokenizer.cache import EncodingCache
from continuous_tokenizer.checkpoint import checkpoint_fingerprint, load_checkpoint
from continuous_tokenizer.codec import ContinuousByteCodec
from continuous_tokenizer.corpus import joined_prefix, load_wikitext_documents
from continuous_tokenizer.metrics import evaluate_embeddings, module_bytes, tensor_bytes
from continuous_tokenizer.model_assets import ModelAssets
from continuous_tokenizer.runtime import default_device
from continuous_tokenizer.segmenter import greedy_segment, reconstruct


@dataclass(frozen=True, slots=True)
class SegmentationRun:
    mode: str
    spans: int
    seconds: float
    bytes_per_span: float
    round_trip: bool
    cache_entries: int
    cache_bytes: int


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _run_segmentation(
    codec: ContinuousByteCodec,
    data: bytes,
    *,
    mode: str,
    cache: EncodingCache | None,
    namespace: str,
) -> SegmentationRun:
    _synchronize(codec.device)
    started = time.perf_counter()
    segments = greedy_segment(codec, data, cache=cache, namespace=namespace)
    _synchronize(codec.device)
    seconds = time.perf_counter() - started
    info = cache.info() if cache else None
    return SegmentationRun(
        mode=mode,
        spans=len(segments),
        seconds=seconds,
        bytes_per_span=len(data) / max(len(segments), 1),
        round_trip=reconstruct(segments) == data,
        cache_entries=0 if info is None else info.entries,
        cache_bytes=0 if info is None else info.bytes,
    )


def _markdown(metrics: dict[str, Any]) -> str:
    fit = metrics["embedding_fit"]
    memory = metrics["memory"]
    density = metrics["density"]
    lines = [
        "# Continuous Byte Tokenizer Benchmark",
        "",
        f"- Model: `{metrics['model']['id']}`",
        f"- Revision: `{metrics['model']['revision']}`",
        f"- Exact embedding rows: `{fit['exact_rows']}/{fit['rows']}`",
        f"- Exact vocabulary reconstruction: `{fit['reconstruction_fraction']:.2%}`",
        f"- Source embedding table: `{memory['source_table_bytes']:,}` bytes",
        f"- Codec state: `{memory['codec_bytes']:,}` bytes",
        f"- Memory ratio: `{memory['ratio']:.4f}`",
        f"- Original bytes/token: `{density['original_bytes_per_token']:.4f}`",
        f"- Continuous bytes/span: `{density['continuous_bytes_per_span']:.4f}`",
        f"- Density ratio: `{density['ratio']:.4f}`",
        "",
        "## Acceptance",
        "",
        f"- Embedding fit: **{'PASS' if metrics['acceptance']['embedding_fit'] else 'FAIL'}**",
        f"- Memory reduction: **{'PASS' if metrics['acceptance']['memory'] else 'FAIL'}**",
        f"- Input density: **{'PASS' if metrics['acceptance']['density'] else 'FAIL'}**",
        f"- Overall: **{'PASS' if metrics['acceptance']['overall'] else 'FAIL'}**",
        "",
        "## Cache comparison",
        "",
        "| Mode | Seconds | Spans | Bytes/span | Cache bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in metrics["segmentation_runs"]:
        lines.append(
            f"| {run['mode']} | {run['seconds']:.6f} | {run['spans']} | "
            f"{run['bytes_per_span']:.4f} | {run['cache_bytes']} |"
        )
    lines += [
        "",
        "The benchmark is tokenizer-only and makes no language-model quality claim.",
    ]
    return "\n".join(lines) + "\n"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def benchmark_experiment(
    assets: ModelAssets,
    checkpoint: Path,
    output_dir: Path,
    *,
    max_test_bytes: int = 16_384,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict[str, Any]:
    selected_device = default_device() if device is None else device
    codec, checkpoint_metadata = load_checkpoint(checkpoint, device=selected_device)
    fingerprint = checkpoint_fingerprint(checkpoint)
    if checkpoint_metadata.get("model_revision") != assets.revision:
        raise ValueError("checkpoint and source model revisions do not match")

    test_data = joined_prefix(load_wikitext_documents("test"), max_bytes=max_test_bytes)
    text = test_data.decode("utf-8", errors="strict")
    original_ids = assets.tokenizer.encode(text, add_special_tokens=False)
    embedding_metrics = evaluate_embeddings(
        codec,
        assets.vocabulary,
        assets.input_embeddings,
        batch_size=batch_size,
        device=selected_device,
    )

    uncached = _run_segmentation(
        codec, test_data, mode="disabled", cache=None, namespace=fingerprint
    )
    cache = codec.encoding_cache
    cache.clear()
    cold = _run_segmentation(codec, test_data, mode="cold", cache=cache, namespace=fingerprint)
    warm = _run_segmentation(codec, test_data, mode="warm", cache=cache, namespace=fingerprint)
    if not (uncached.spans == cold.spans == warm.spans):
        raise RuntimeError("encoding cache changed segmentation")

    source_bytes = tensor_bytes(assets.input_embeddings)
    codec_bytes = module_bytes(codec)
    density_ratio = len(original_ids) / max(uncached.spans, 1)
    process = psutil.Process()
    metrics: dict[str, Any] = {
        "model": {
            "id": assets.model_id,
            "revision": assets.revision,
            "embedding_tensor": assets.embedding_tensor_name,
            "source_dtype": str(assets.input_embeddings.dtype),
            "tie_word_embeddings": bool(assets.config.get("tie_word_embeddings", False)),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": fingerprint,
            "serialized_bytes": checkpoint.stat().st_size,
        },
        "embedding_fit": embedding_metrics.to_dict(),
        "memory": {
            "source_table_bytes": source_bytes,
            "codec_bytes": codec_bytes,
            "ratio": codec_bytes / source_bytes,
            "rss_bytes": process.memory_info().rss,
            "peak_rss_bytes": _peak_rss_bytes(),
            "mps_allocated_bytes": (
                torch.mps.current_allocated_memory() if selected_device.type == "mps" else 0
            ),
        },
        "density": {
            "test_bytes": len(test_data),
            "original_tokens": len(original_ids),
            "continuous_spans": uncached.spans,
            "original_bytes_per_token": len(test_data) / max(len(original_ids), 1),
            "continuous_bytes_per_span": uncached.bytes_per_span,
            "ratio": density_ratio,
            "round_trip": uncached.round_trip,
        },
        "segmentation_runs": [asdict(run) for run in (uncached, cold, warm)],
    }
    acceptance = {
        "embedding_fit": (
            embedding_metrics.exact_fraction == 1.0
            and embedding_metrics.reconstruction_fraction == 1.0
        ),
        "memory": codec_bytes <= source_bytes / 2,
        "density": uncached.round_trip and density_ratio >= 1.1,
    }
    acceptance["overall"] = all(acceptance.values())
    metrics["acceptance"] = acceptance

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    (output_dir / "report.md").write_text(_markdown(metrics), encoding="utf-8")
    return metrics
