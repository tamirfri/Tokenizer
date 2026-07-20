from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from input_training_fixtures import (
    TEST_PROFILE,
    limited_torch_threads,
    synthetic_assets,
)

import continuous_tokenizer.input.benchmark.prefill as prefill_benchmark_module
import continuous_tokenizer.input.benchmark.run as benchmark_module
import continuous_tokenizer.input.benchmark.tokenizer as tokenizer_benchmark_module
import continuous_tokenizer.input.training.run as training_module
from continuous_tokenizer.backbone.config import input_table_is_removable, text_config
from continuous_tokenizer.codec.compute import (
    input_encode_batch_flops,
    input_encode_row_flops,
    input_table_projection_flops,
)
from continuous_tokenizer.codec.input import InputByteCodecConfig
from continuous_tokenizer.contracts.performance import (
    tokenizer_performance_errors,
)
from continuous_tokenizer.data.corpus import joined_prefix
from continuous_tokenizer.input.benchmark.run import BenchmarkOptions, benchmark_experiment
from continuous_tokenizer.input.training.run import TrainingOptions, train_experiment
from continuous_tokenizer.runtime.tensors import cache_tensor_bytes
from continuous_tokenizer.runtime.timing import timing_summary


def test_input_table_memory_gate_applies_only_to_removable_tables() -> None:
    assert not input_table_is_removable({"tie_word_embeddings": True})
    assert not input_table_is_removable({"tie_word_embeddings": False, "removable_input_table": False})
    assert input_table_is_removable({"tie_word_embeddings": False})
    text = SimpleNamespace(hidden_size=8)
    assert text_config(SimpleNamespace(text_config=text)) is text


def test_hybrid_model_efficiency_uses_text_config_and_materialized_cache() -> None:
    text_config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=("linear_attention", "full_attention"),
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
    )
    outer_config = SimpleNamespace(text_config=text_config)
    shared = torch.zeros((2, 3))
    cache = SimpleNamespace(
        layers=(
            SimpleNamespace(
                conv_states={0: shared},
                recurrent_states={0: shared},
            ),
            SimpleNamespace(
                keys=torch.zeros((1, 1, 5, 4)),
                values=torch.zeros((1, 1, 5, 4)),
            ),
        )
    )

    assert prefill_benchmark_module._estimated_prefill_flops(outer_config, 5) == (prefill_benchmark_module._estimated_prefill_flops(text_config, 5))
    full_attention = SimpleNamespace(**{**vars(text_config), "layer_types": ("full_attention", "full_attention")})
    assert prefill_benchmark_module._estimated_prefill_flops(text_config, 5) != prefill_benchmark_module._estimated_prefill_flops(full_attention, 5)
    assert cache_tensor_bytes(cache) == (20 + 20 + 6) * 4


def test_prefill_flops_support_gemma_sliding_attention() -> None:
    positions = 5
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=("sliding_attention", "sliding_attention"),
        sliding_window=2,
    )
    full_attention = SimpleNamespace(**{**vars(config), "layer_types": ("full_attention", "full_attention")})

    full_flops = prefill_benchmark_module._estimated_prefill_flops(full_attention, positions)
    sliding_flops = prefill_benchmark_module._estimated_prefill_flops(config, positions)

    query_dim = config.num_attention_heads * config.head_dim
    assert full_flops - sliding_flops == (2 * 4 * positions * (positions - config.sliding_window) * query_dim)


def test_codec_flops_use_observed_candidate_lengths() -> None:
    adapter = SimpleNamespace(
        codec=SimpleNamespace(config=object(), max_span=64),
        namespace="test",
        pieces_from_token_ids=lambda _prompt: (prefill_benchmark_module.ByteRun(b"abc"),),
    )
    segmented = SimpleNamespace(
        stats=SimpleNamespace(candidate_lengths={1: 3, 2: 2, 3: 1}),
        spans=(
            SimpleNamespace(data=b"a"),
            SimpleNamespace(data=b"b"),
            SimpleNamespace(data=b"c"),
        ),
    )

    with (
        patch.object(prefill_benchmark_module, "segment_bytes", return_value=segmented),
        patch.object(
            prefill_benchmark_module,
            "input_encode_batch_flops",
            side_effect=lambda _config, lengths, *, neural_invocations: sum(length * count * 10 for length, count in lengths.items()) + neural_invocations,
        ),
        patch.object(
            prefill_benchmark_module,
            "input_validation_flops",
            side_effect=lambda _config, length: length * 100,
        ),
    ):
        compute = prefill_benchmark_module._codec_compute(
            cast(Any, adapter),
            (1,),
            "segmented",
        )

    assert compute.candidate_lengths == {1: 3, 2: 2, 3: 1}
    assert compute.logical_candidates == 3
    assert compute.neural_candidate_rows > compute.logical_candidates
    assert compute.encode_flops > 0
    assert compute.validation_flops > 0


def test_input_table_projection_flops_are_counted_per_batch() -> None:
    config = InputByteCodecConfig(
        embedding_dim=8,
        local_dim=8,
        projection_dim=16,
        max_span=8,
        query_heads=4,
        feedforward_dim=16,
        encoder_layers=1,
        decoder_layers=1,
    )
    lengths = {2: 3, 4: 2}

    measured = input_encode_batch_flops(
        config,
        lengths,
        neural_invocations=2,
    )

    expected = 2 * input_table_projection_flops(config) + sum(count * input_encode_row_flops(config, length) for length, count in lengths.items())
    repeated_projection = sum(count * (input_table_projection_flops(config) + input_encode_row_flops(config, length)) for length, count in lengths.items())
    assert measured == expected
    assert measured < repeated_projection


def test_compiled_warmup_covers_every_bounded_signature() -> None:
    class RecordingCodec:
        neural_paths_compiled = True
        max_span = 8
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[int, ...]]] = []

        def encode(self, byte_values, _valid_mask):
            self.calls.append(("encode", tuple(byte_values.shape)))
            return torch.zeros((byte_values.shape[0], 8))

        def reconstruction_logits(self, byte_values, _valid_mask):
            self.calls.append(("reconstruction", tuple(byte_values.shape)))
            return torch.zeros((byte_values.shape[0], byte_values.shape[1] + 1, 257))

        def encode_and_reconstruction_matches(self, byte_values, _valid_mask):
            self.calls.append(("validation", tuple(byte_values.shape)))
            return (
                torch.zeros((byte_values.shape[0], 8)),
                torch.ones(byte_values.shape[0], dtype=torch.bool),
            )

    codec = RecordingCodec()

    compilation = tokenizer_benchmark_module._warm_compiled_tokenizer(
        cast(Any, codec),
        b"abcdefgh",
    )

    expected_shapes = {(8, 2), (16, 4), (32, 8)}
    assert compilation["warm_compile"]["status"] == "complete"
    assert compilation["cold_compile"]["status"] == "unavailable"
    assert not compilation["cold_compile"]["process_isolated"]
    for operation in ("encode", "reconstruction", "validation"):
        assert {shape for name, shape in codec.calls if name == operation} == expected_shapes


def test_benchmark_writes_complete_tokenizer_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            limited_torch_threads(),
            patch.object(training_module, "load_corpus_documents", return_value=[b"abc"]),
            patch.object(
                benchmark_module,
                "load_corpus_documents",
                return_value=[b"a", b"bc"],
            ),
            patch.object(
                tokenizer_benchmark_module,
                "_raw_byte_fixtures",
                return_value={"binary": b"\x00\xff"},
            ),
            patch.object(
                tokenizer_benchmark_module,
                "segmentation_evidence",
                wraps=tokenizer_benchmark_module.segmentation_evidence,
            ) as segmentation_evidence,
        ):
            assets = synthetic_assets(root)
            assets.config = {"tie_word_embeddings": False}
            options = TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=256,
                vocabulary_epochs=1,
                reconstruction_epochs=0,
                reconstruction_samples=0,
                validation_bytes=3,
                patience=1,
            )
            result = train_experiment(assets, options, device=torch.device("cpu"))
            output_dir = root / "artifact"

            metrics = benchmark_experiment(
                assets,
                Path(result.checkpoint),
                output_dir,
                BenchmarkOptions(
                    max_test_bytes=3,
                    batch_size=256,
                    retrieval_rows=16,
                    device=torch.device("cpu"),
                ),
            )

        assert metrics["embedding_fit"]["retrieval_queries"] == 16
        assert metrics["embedding_fit"]["retrieval_candidates"] == 256
        assert metrics["codec"] == {
            "query_heads": 4,
            "key_value_heads": 2,
            "enable_gqa": True,
        }
        assert not metrics["compilation"]["enabled"]
        assert metrics["compilation"]["eager"]["status"] == "complete"
        assert metrics["compilation"]["cold_compile"]["status"] == "not_applicable"
        assert metrics["benchmark_contract"]["content_window_boundaries_preserved"]
        assert metrics["raw_byte_fixtures"]["binary"]["round_trip"]
        assert metrics["gates"]["raw_binary_fixtures"] == {
            "measured": 1,
            "operator": "==",
            "threshold": 1,
            "passed": True,
        }
        assert metrics["gates"]["maximum_normalized_rmse"]["threshold"] == 0.01
        assert set(metrics["density"]) == {
            "test_bytes",
            "native_tokens",
            "continuous_tokens",
            "bytes_per_native_token",
            "bytes_per_continuous_token",
            "native_tokens_per_continuous_token",
            "round_trip",
            "alignment",
        }
        assert not {"original_tokens", "continuous_spans", "ratio"} & metrics["density"].keys()
        runs = {run["mode"]: run for run in metrics["segmentation_runs"]}
        assert len({run["semantic_sha256"] for run in runs.values()}) == 1
        assert all(run["round_trip"] for run in runs.values())
        assert all(run["source_bookkeeping_round_trip"] for run in runs.values())
        assert all(run["repetitions"] == 5 for run in runs.values())
        assert all(len(run["raw_observations"]) == 5 for run in runs.values())
        assert all(run["content_windows"] == 2 for run in runs.values())
        assert all(run["logical_candidates"] == run["candidates"] for run in runs.values())
        assert all(run["neural_candidate_rows"] >= run["logical_candidates"] for run in runs.values())
        assert [observation["execution_order"] for observation in runs["disabled"]["raw_observations"]] == [0, 2, 1, 0, 2]
        assert all(run["p95_seconds"] >= run["seconds"] for run in runs.values())
        assert runs["cold"]["cache_misses"] > 0
        assert runs["warm"]["cache_misses"] == 0
        assert runs["warm"]["cache_hit_rate"] == 1.0
        assert runs["warm"]["cache_tensor_bytes"] > 0
        assert runs["warm"]["process_rss_bytes"] > 0
        assert metrics["compactness"]["warm_cache_tensor_bytes"] == runs["warm"]["cache_tensor_bytes"]
        assert segmentation_evidence.call_count == (2 + 1 + 2 + min(256, len(assets.vocabulary.compatibility_ids)))
        assert not tokenizer_performance_errors(metrics)
        assert (output_dir / "tokenizer-metrics.json").is_file()
        assert (output_dir / "tokenizer-report.md").is_file()


def test_joined_prefix_never_splits_utf8_codepoint() -> None:
    value = joined_prefix(["a€b".encode()], max_bytes=3)

    assert value == b"a"
    assert value.decode("utf-8") == "a"


def test_single_timing_sample_has_identical_median_and_p95() -> None:
    sample = 0.8838819999946281

    assert timing_summary([sample]) == {"median": sample, "p95": sample}


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_input_table_memory_gate_applies_only_to_removable_tables,
            test_hybrid_model_efficiency_uses_text_config_and_materialized_cache,
            test_prefill_flops_support_gemma_sliding_attention,
            test_codec_flops_use_observed_candidate_lengths,
            test_input_table_projection_flops_are_counted_per_batch,
            test_compiled_warmup_covers_every_bounded_signature,
            test_benchmark_writes_complete_tokenizer_artifact,
            test_joined_prefix_never_splits_utf8_codepoint,
            test_single_timing_sample_has_identical_median_and_p95,
        )
    )
