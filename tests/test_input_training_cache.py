from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from input_training_fixtures import TEST_PROFILE, pair_assets

import continuous_tokenizer.input.training.runtime as selection_module
from continuous_tokenizer.codec.batches import (
    build_span_batch,
    byte_reconstruction_loss,
)
from continuous_tokenizer.input.training.cache import build_frozen_span_cache
from continuous_tokenizer.input.training.run import TrainingOptions


def test_frozen_span_cache_preserves_latents_losses_and_decoder_gradients() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = selection_module.TrainingRuntime(
            pair_assets(root),
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=1,
            ),
            torch.device("cpu"),
        )
        uncached = runtime.build_codec(TEST_PROFILE)
        cached = copy.deepcopy(uncached)
        uncached.set_trainable_components(encoder=False, decoder=True)
        cached.set_trainable_components(encoder=False, decoder=True)
        span = b"ab"
        batch = build_span_batch([span], max_span=uncached.max_span, device=torch.device("cpu"))
        with torch.no_grad():
            uncached_latent = uncached.encode(batch.byte_values, batch.valid_mask)
        replay = build_frozen_span_cache(
            cached,
            (span,),
            batch_size=1,
            device=torch.device("cpu"),
        )
        cached_latent, targets, target_mask, positions = replay.select(torch.tensor([0]))

        uncached_loss = byte_reconstruction_loss(
            uncached.decode_logits(uncached_latent, batch.framed_targets.shape[1]),
            batch.framed_targets,
            batch.target_mask,
        )
        cached_loss = byte_reconstruction_loss(
            cached.decode_logits(cached_latent, positions),
            targets,
            target_mask,
        )
        uncached_loss.backward()
        cached_loss.backward()

        assert torch.equal(cached_latent, uncached_latent)
        assert torch.equal(cached_loss, uncached_loss)
        for (left_name, left), (right_name, right) in zip(
            uncached.named_parameters(),
            cached.named_parameters(),
            strict=True,
        ):
            assert left_name == right_name
            if left.requires_grad:
                assert left.grad is not None and right.grad is not None
                assert torch.equal(left.grad, right.grad)

        with torch.no_grad():
            cached.cls.add_(1)
        with unittest.TestCase().assertRaisesRegex(ValueError, "encoder state"):
            replay.validate(cached)


def test_cached_source_dtype_evaluation_reuses_static_encoder_metrics() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        assets.input_embeddings = assets.input_embeddings.to(torch.bfloat16)
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=16,
            ),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(TEST_PROFILE)
        evaluator = runtime.deployment_evaluator(codec)
        encoded_rows: list[int] = []
        matched_rows: list[int] = []

        def encode(byte_values, valid_mask):
            encoded_rows.append(byte_values.shape[0])
            return evaluator._encode_tensor(byte_values, valid_mask)

        def matches(latent, byte_values, valid_mask):
            matched_rows.append(byte_values.shape[0])
            return evaluator._reconstruction_matches_tensor(
                latent,
                byte_values,
                valid_mask,
            )

        evaluator._compiled_encode = encode
        evaluator._compiled_matches = matches
        cache = runtime.cached_deployment_evaluation(codec, evaluator)

        with patch.object(evaluator, "encode", wraps=evaluator.encode) as encode:
            cached_metrics = runtime.evaluate_cached_deployment(codec, evaluator, cache)
        encode.assert_not_called()
        uncached_metrics = runtime.evaluate_deployment(codec, evaluator)

        assert cached_metrics == uncached_metrics
        assert encoded_rows and set(encoded_rows) == {16}
        assert matched_rows and set(matched_rows) == {16}
        assert cache.static_rows == 16


def test_frozen_span_cache_pads_every_compiled_encoder_batch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = selection_module.TrainingRuntime(
            pair_assets(root),
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=4,
            ),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(TEST_PROFILE)
        spans = (b"ab", b"cd", b"ef", b"gh", b"ijk")

        with patch.object(codec, "encode", wraps=codec.encode) as encode:
            cache = build_frozen_span_cache(
                codec,
                spans,
                batch_size=4,
                device=torch.device("cpu"),
            )

        assert cache.latents.shape[0] == len(spans)
        assert cache.lengths.tolist() == [len(span) for span in spans]
        assert encode.call_count == 2
        assert all(call.args[0].shape[0] == 4 for call in encode.call_args_list)


def test_frozen_span_cache_deduplicates_latents_without_changing_weighting() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = selection_module.TrainingRuntime(
            pair_assets(root),
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=3,
            ),
            torch.device("cpu"),
        )
        direct = runtime.build_codec(TEST_PROFILE)
        cached = copy.deepcopy(direct)
        direct.set_trainable_components(encoder=False, decoder=True)
        cached.set_trainable_components(encoder=False, decoder=True)
        spans = [b"ab", b"ab", b"c"]
        batch = build_span_batch(
            spans,
            max_span=direct.max_span,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            direct_latents = direct.encode(batch.byte_values, batch.valid_mask)
        replay = build_frozen_span_cache(
            cached,
            tuple(spans),
            batch_size=3,
            device=torch.device("cpu"),
        )
        latent, targets, target_mask, positions = replay.select(torch.tensor([0, 1, 2]))
        direct_loss = byte_reconstruction_loss(
            direct.decode_logits(direct_latents, positions),
            batch.framed_targets,
            batch.target_mask,
        )
        cached_loss = byte_reconstruction_loss(
            cached.decode_logits(latent, positions),
            targets,
            target_mask,
        )
        direct_loss.backward()
        cached_loss.backward()

        assert replay.latents.shape[0] == 2
        assert replay.occurrence_to_unique.tolist() == [0, 0, 1]
        assert torch.equal(latent, direct_latents)
        assert torch.equal(targets, batch.framed_targets)
        assert torch.equal(target_mask, batch.target_mask)
        assert torch.equal(cached_loss, direct_loss)
        for (left_name, left), (right_name, right) in zip(
            direct.named_parameters(),
            cached.named_parameters(),
            strict=True,
        ):
            assert left_name == right_name
            if left.requires_grad:
                assert left.grad is not None and right.grad is not None
                assert torch.equal(left.grad, right.grad)


def test_training_runtime_reuses_only_identity_matching_caches() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        assets.input_embeddings = assets.input_embeddings.to(torch.bfloat16)
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=2,
            ),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(TEST_PROFILE)
        evaluator = runtime.deployment_evaluator(codec)
        assert runtime.deployment_evaluator(codec) is evaluator
        first = runtime.cached_deployment_evaluation(codec, evaluator)
        assert runtime.cached_deployment_evaluation(codec, evaluator) is first
        spans = (b"ab", b"ab")
        first_spans, reused = runtime.frozen_span_cache(
            codec,
            spans,
            batch_size=2,
        )
        second_spans, reused_again = runtime.frozen_span_cache(
            codec,
            spans,
            batch_size=2,
        )
        reordered, reordered_reused = runtime.frozen_span_cache(
            codec,
            tuple(reversed((*spans, b"c"))),
            batch_size=2,
        )

        assert not reused
        assert reused_again
        assert first_spans is second_spans
        assert not reordered_reused
        assert reordered is not first_spans
        telemetry = runtime.cache_telemetry()
        assert telemetry["deployment_evaluator_builds"] == 1
        assert telemetry["deployment_evaluator_reuses"] == 1
        assert telemetry["source_dtype_cache_builds"] == 1
        assert telemetry["source_dtype_cache_reuses"] == 1
        assert telemetry["frozen_span_cache_builds"] == 2
        assert telemetry["frozen_span_cache_reuses"] == 1
        assert telemetry["accelerator_length_synchronizations"] == 0


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_frozen_span_cache_preserves_latents_losses_and_decoder_gradients,
            test_cached_source_dtype_evaluation_reuses_static_encoder_metrics,
            test_frozen_span_cache_pads_every_compiled_encoder_batch,
            test_frozen_span_cache_deduplicates_latents_without_changing_weighting,
            test_training_runtime_reuses_only_identity_matching_caches,
        )
    )
