from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import torch

from continuous_tokenizer.artifacts.store import write_json_atomic, write_text_atomic
from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.data.corpus import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    load_corpus_documents,
    sample_content_windows,
)
from continuous_tokenizer.input.adapter import InputEmbeddingAdapter
from continuous_tokenizer.input.benchmark.tokenizer import (
    TokenizerMetricRequest,
    tokenizer_metrics,
    tokenizer_report,
)
from continuous_tokenizer.runtime.device import default_device


@final
@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    max_test_bytes: int = 16_384
    batch_size: int = 32
    retrieval_rows: int | None = 4096
    repetitions: int = 5
    dataset_id: str = DATASET_ID
    dataset_config: str = DATASET_CONFIG
    dataset_revision: str = DATASET_REVISION
    device: torch.device | None = None


def benchmark_experiment(
    assets: ModelAssets,
    checkpoint: Path,
    output_dir: Path,
    options: BenchmarkOptions | None = None,
) -> dict[str, Any]:
    options = BenchmarkOptions() if options is None else options
    if options.repetitions < 1:
        raise ValueError("benchmark repetitions must be positive")
    selected_device = default_device() if options.device is None else options.device
    loaded = InputEmbeddingAdapter.from_checkpoint(
        assets,
        checkpoint,
        device=selected_device,
    )
    documents = load_corpus_documents(
        "test",
        dataset_id=options.dataset_id,
        config=options.dataset_config,
        revision=options.dataset_revision,
        max_rows=options.max_test_bytes,
    )
    test_windows = sample_content_windows(
        documents,
        maximum_bytes=options.max_test_bytes,
    )
    metrics = tokenizer_metrics(
        assets,
        checkpoint,
        loaded,
        TokenizerMetricRequest(
            test_windows=test_windows,
            batch_size=options.batch_size,
            retrieval_rows=options.retrieval_rows,
            repetitions=options.repetitions,
            dataset_id=options.dataset_id,
            dataset_config=options.dataset_config,
            dataset_revision=options.dataset_revision,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "tokenizer-metrics.json", metrics)
    write_text_atomic(output_dir / "tokenizer-report.md", tokenizer_report(metrics))
    return metrics
