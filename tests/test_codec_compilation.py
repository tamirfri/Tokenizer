from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from codec_fixtures import make_codec

from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.batches import (
    build_span_batch,
    byte_reconstruction_loss,
)
from continuous_tokenizer.codec.compilation import planned_input_graph_signatures
from continuous_tokenizer.input.alignment import (
    EmbeddingEvaluationRequest,
    evaluate_embeddings,
)
from continuous_tokenizer.input.segmentation import reconstruct, segment_bytes
from continuous_tokenizer.runtime.compiler import compiler_cache_directory, configure_compiler_cache


def test_compiler_cache_honors_external_directory() -> None:
    with patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": "/tmp/custom-inductor-cache"}):
        assert compiler_cache_directory() == Path("/tmp/custom-inductor-cache")

    with patch.dict(os.environ):
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        assert compiler_cache_directory() == Path(sys.prefix) / ".cache" / "torchinductor"
        configure_compiler_cache("eager")
        assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
        configure_compiler_cache("inductor")
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(Path(sys.prefix) / ".cache" / "torchinductor")


def test_compiled_calls_scope_the_bounded_recompile_limit() -> None:
    default_limit = torch.compiler.config.recompile_limit
    codec = make_codec(max_span=16).eval()
    codec.compile_neural_paths(backend="eager")

    for width in range(2, 13):
        values = torch.zeros((2, width), dtype=torch.long)
        valid = torch.ones_like(values, dtype=torch.bool)
        with torch.inference_mode():
            codec.encode(values, valid)

    assert torch.compiler.config.recompile_limit == default_limit
    assert not codec.projected_byte_table_cached


def test_compiled_shape_plan_stays_bounded_and_rejects_oversized_plans() -> None:
    planned = planned_input_graph_signatures(64, (64,))
    qwen_planned = planned_input_graph_signatures(128, (64,))

    assert len(planned["validation"]) == 25
    assert len(planned["encode"]) == 25
    assert len(planned["matches"]) == 25
    assert len(planned["reconstruction"]) == 7
    assert len(planned["decode"]) == 7
    assert len(qwen_planned["validation"]) == 26
    with unittest.TestCase().assertRaisesRegex(ValueError, "exceed"):
        make_codec(max_span=64).compile_neural_paths(
            backend="eager",
            static_rows=tuple(range(1, 65)),
        )


@unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS") == "1", "set RUN_SLOW_TESTS=1")
def test_neural_paths_are_full_graph_compilable() -> None:
    codec = make_codec().eval()
    batch = build_span_batch([b"ab", b"abc"], max_span=codec.max_span, device=codec.device)
    codec.compile_neural_paths(backend="eager")
    assert codec.neural_paths_compiled

    reconstructed_latent, reconstructed_logits = codec.reconstruction_logits(
        batch.byte_values,
        batch.valid_mask,
    )
    validated_latent, validated_matches = codec.encode_and_reconstruction_matches(
        batch.byte_values,
        batch.valid_mask,
    )

    assert reconstructed_latent.shape == (2, 16)
    assert reconstructed_logits.shape == (2, 5, 257)
    assert validated_latent.shape == reconstructed_latent.shape
    assert validated_matches.shape == (2,)

    codec.train()
    codec.zero_grad(set_to_none=True)
    _, training_logits = codec.reconstruction_logits(batch.byte_values, batch.valid_mask)
    byte_reconstruction_loss(
        training_logits,
        batch.framed_targets,
        batch.target_mask,
    ).backward()
    assert codec.input_projection.weight.grad is not None
    assert codec.decoder_projection.weight.grad is not None


@unittest.skipUnless(
    os.environ.get("RUN_SLOW_TESTS") == "1" and torch.backends.mps.is_available(),
    "set RUN_SLOW_TESTS=1 on an MPS host",
)
def test_mps_gqa_forward_and_backward_compile_as_full_graphs() -> None:
    device = torch.device("mps")
    codec = make_codec().to(device).train()
    batch = build_span_batch([b"ab", b"abc"], max_span=codec.max_span, device=device)
    codec.compile_neural_paths()

    _, logits = codec.reconstruction_logits(batch.byte_values, batch.valid_mask)
    loss = byte_reconstruction_loss(logits, batch.framed_targets, batch.target_mask)
    loss.backward()
    torch.mps.synchronize()

    assert torch.isfinite(loss)
    parameters = dict(codec.encoder.named_parameters())
    assert parameters["layers.0.attention.key.weight"].grad is not None
    assert parameters["layers.0.attention.value.weight"].grad is not None

    deployment_codec = make_codec(max_span=64).to(device=device, dtype=torch.bfloat16).eval()
    deployment_batch = build_span_batch(
        [bytes(range(length)) for length in range(2, 65)],
        max_span=deployment_codec.max_span,
        device=device,
    )
    deployment_codec.compile_neural_paths()
    _, matches = deployment_codec.encode_and_reconstruction_matches(
        deployment_batch.byte_values,
        deployment_batch.valid_mask,
    )
    torch.mps.synchronize()

    assert matches.shape == (63,)
    segmented = segment_bytes(deployment_codec, bytes(range(64)))
    assert reconstruct(segmented.spans) == bytes(range(64))


@unittest.skipUnless(
    os.environ.get("RUN_SLOW_TESTS") == "1" and torch.backends.mps.is_available(),
    "set RUN_SLOW_TESTS=1 on an MPS host",
)
def test_mps_qwen_sized_vocabulary_evaluation_uses_static_rows() -> None:
    rows = 2048
    pair_rows = 744
    payloads = (
        *(bytes([value]) for value in range(256)),
        *(index.to_bytes(2, "big") for index in range(pair_rows)),
        *(b"\xff" + index.to_bytes(2, "big") for index in range(rows - 256 - pair_rows)),
    )
    vocabulary = ByteVocabulary(
        token_bytes=payloads,
        ordinary_ids=tuple(range(rows)),
        control_ids=(),
        byte_token_ids=tuple(range(256)),
        max_token_bytes=3,
    )
    device = torch.device("mps")
    codec = make_codec(max_span=64).to(device=device, dtype=torch.bfloat16).eval()
    codec.compile_neural_paths(static_rows=(64,))
    source_embeddings = torch.randn(
        (rows, codec.config.embedding_dim),
        dtype=torch.bfloat16,
    )

    metrics = evaluate_embeddings(
        codec,
        vocabulary,
        source_embeddings,
        EmbeddingEvaluationRequest(
            batch_size=64,
            device=device,
        ),
    )
    torch.mps.synchronize()
    encountered = codec.graph_signature_telemetry()["encountered"]["validation"]

    assert metrics.rows == rows
    assert encountered
    assert {signature.split("x", maxsplit=1)[0] for signature in encountered} == {"64"}


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_compiler_cache_honors_external_directory,
            test_compiled_calls_scope_the_bounded_recompile_limit,
            test_compiled_shape_plan_stays_bounded_and_rejects_oversized_plans,
            test_neural_paths_are_full_graph_compilable,
            test_mps_gqa_forward_and_backward_compile_as_full_graphs,
            test_mps_qwen_sized_vocabulary_evaluation_uses_static_rows,
        )
    )
