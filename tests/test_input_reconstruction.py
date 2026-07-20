from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from input_training_fixtures import TEST_PROFILE, pair_assets, synthetic_assets

import continuous_tokenizer.input.training.reconstruction as reconstruction_module
import continuous_tokenizer.input.training.runtime as selection_module
from continuous_tokenizer.contracts.profiles import Profile
from continuous_tokenizer.input.alignment import EmbeddingEvaluationRequest, evaluate_embeddings
from continuous_tokenizer.input.training.run import TrainingOptions
from continuous_tokenizer.runtime.resume import ResumeManager


def test_reconstruction_training_freezes_encoder_and_preserves_source_table() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=2,
                vocabulary_epochs=1,
                reconstruction_epochs=1,
                reconstruction_samples=1,
            ),
            torch.device("cpu"),
        )
        fitter = reconstruction_module.ReconstructionFitter(runtime)
        codec = runtime.build_codec(Profile("test", 8, 1, 1, 1, 4, 2, 16))
        parameters = codec.set_trainable_components(encoder=False, decoder=True)
        optimizers = runtime.optimizers(codec, parameters)
        source_before = assets.input_embeddings.clone()
        encoder_before = {name: parameter.detach().clone() for name, parameter in codec.named_parameters() if not parameter.requires_grad}
        decoder_before = codec.byte_head.weight.detach().clone()

        fitter.train_mixed_epoch(
            reconstruction_module.MixedEpochRequest(
                codec=codec,
                corpus_spans=[b"bc"],
                optimizers=optimizers,
                randomizer=random.Random(17),
            )
        )

        assert torch.equal(assets.input_embeddings, source_before)
        assert not torch.equal(codec.byte_head.weight, decoder_before)
        for name, parameter in codec.named_parameters():
            if name in encoder_before:
                assert torch.equal(parameter, encoder_before[name])


def test_dynamic_reconstruction_keeps_progress_before_alignment_passes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = selection_module.TrainingRuntime(
            pair_assets(root),
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                reconstruction_epochs=1,
                reconstruction_samples=1,
            ),
            torch.device("cpu"),
        )
        fitter = reconstruction_module.ReconstructionFitter(runtime)
        codec = runtime.build_codec(TEST_PROFILE)
        metrics = runtime.evaluate(codec)
        assert not runtime.options.embedding_targets.accepts(metrics)
        before = codec.cls.detach().clone()

        def train_epoch(
            _fitter: reconstruction_module.ReconstructionFitter,
            request: reconstruction_module.MixedEpochRequest,
        ) -> float:
            with torch.no_grad():
                request.codec.cls.add_(1)
            return 1.0

        with (
            patch.object(
                reconstruction_module.ReconstructionFitter,
                "train_mixed_epoch",
                autospec=True,
                side_effect=train_epoch,
            ),
            patch.object(
                selection_module.TrainingRuntime,
                "cached_deployment_evaluation",
                autospec=True,
                return_value=SimpleNamespace(tensor_bytes=0),
            ),
            patch.object(
                selection_module.TrainingRuntime,
                "evaluate_cached_deployment",
                autospec=True,
                side_effect=(metrics, metrics),
            ),
            patch.object(
                selection_module.TrainingRuntime,
                "density_metrics",
                autospec=True,
                side_effect=((0.5, True), (0.75, True)),
            ),
        ):
            fitter.fit(codec, TEST_PROFILE, [b"ab"], b"ab", random.Random(17))

        progress = json.loads((root / "checkpoints/progress/small-dynamic-reconstruction-001.json").read_text())
        assert not torch.equal(codec.cls, before)
        assert progress["selected"] is True


def test_dynamic_reconstruction_preserves_previously_passing_alignment() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = selection_module.TrainingRuntime(
            synthetic_assets(root),
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                reconstruction_epochs=1,
                reconstruction_samples=1,
            ),
            torch.device("cpu"),
        )
        fitter = reconstruction_module.ReconstructionFitter(runtime)
        codec = runtime.build_codec(TEST_PROFILE)
        passing = runtime.evaluate(codec)
        failing = replace(
            passing,
            normalized_rmse=1.0,
            cosine_similarity_p01=0.0,
            cosine_similarity_p50=0.0,
        )
        assert runtime.options.embedding_targets.accepts(passing)
        before = codec.cls.detach().clone()

        def train_epoch(
            _fitter: reconstruction_module.ReconstructionFitter,
            request: reconstruction_module.MixedEpochRequest,
        ) -> float:
            with torch.no_grad():
                request.codec.cls.add_(1)
            return 1.0

        with (
            patch.object(
                reconstruction_module.ReconstructionFitter,
                "train_mixed_epoch",
                autospec=True,
                side_effect=train_epoch,
            ),
            patch.object(
                selection_module.TrainingRuntime,
                "cached_deployment_evaluation",
                autospec=True,
                return_value=SimpleNamespace(tensor_bytes=0),
            ),
            patch.object(
                selection_module.TrainingRuntime,
                "evaluate_cached_deployment",
                autospec=True,
                side_effect=(passing, failing),
            ),
            patch.object(
                selection_module.TrainingRuntime,
                "density_metrics",
                autospec=True,
                side_effect=((1.0, True), (2.0, True)),
            ),
        ):
            fitter.fit(codec, TEST_PROFILE, [b"ab"], b"ab", random.Random(17))

        assert torch.equal(codec.cls, before)


def test_dynamic_reconstruction_resume_matches_uninterrupted_training() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        options = TrainingOptions(
            output_dir=root / "checkpoints",
            profile=TEST_PROFILE,
            batch_size=2,
            reconstruction_epochs=2,
            reconstruction_samples=1,
        )
        torch.manual_seed(17)
        initial_runtime = selection_module.TrainingRuntime(assets, options, torch.device("cpu"))
        initial = initial_runtime.build_codec(TEST_PROFILE)
        interrupted_runtime = selection_module.TrainingRuntime(
            assets,
            options,
            torch.device("cpu"),
            ResumeManager(root, "experiment", "commit", "source", "lock", False),
        )
        interrupted = copy.deepcopy(initial)
        original_save = ResumeManager.save

        def save_then_interrupt(manager, phase, epoch, state):
            original_save(manager, phase, epoch, state)
            raise KeyboardInterrupt

        with (
            patch.object(ResumeManager, "save", autospec=True, side_effect=save_then_interrupt),
            unittest.TestCase().assertRaises(KeyboardInterrupt),
        ):
            reconstruction_module.ReconstructionFitter(interrupted_runtime).fit(
                interrupted,
                TEST_PROFILE,
                [b"bc"],
                b"ab",
                random.Random(17),
            )

        resumed = copy.deepcopy(initial)
        resumed_runtime = selection_module.TrainingRuntime(
            assets,
            options,
            torch.device("cpu"),
            ResumeManager(root, "experiment", "commit", "source", "lock", True),
        )
        reconstruction_module.ReconstructionFitter(resumed_runtime).fit(
            resumed,
            TEST_PROFILE,
            [b"bc"],
            b"ab",
            random.Random(17),
        )
        uninterrupted = copy.deepcopy(initial)
        reconstruction_module.ReconstructionFitter(initial_runtime).fit(
            uninterrupted,
            TEST_PROFILE,
            [b"bc"],
            b"ab",
            random.Random(17),
        )

        for name, value in uninterrupted.state_dict().items():
            assert torch.equal(resumed.state_dict()[name], value), name


def test_complete_checkpoint_selection_prioritizes_exact_reconstruction() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = synthetic_assets(root)
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(output_dir=root / "checkpoints", profile=TEST_PROFILE),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(Profile("test", 8, 1, 1, 1, 4, 2, 16))
        metrics = evaluate_embeddings(
            codec,
            assets.vocabulary,
            assets.input_embeddings,
            EmbeddingEvaluationRequest(
                batch_size=256,
                device=torch.device("cpu"),
            ),
        )
        exact = replace(
            metrics,
            reconstruction_rows=metrics.rows,
            reconstruction_fraction=1.0,
            normalized_rmse=10.0,
        )
        approximate = replace(
            metrics,
            reconstruction_rows=metrics.rows - 1,
            reconstruction_fraction=(metrics.rows - 1) / metrics.rows,
            normalized_rmse=0.0,
        )

        assert runtime.compatibility_score(exact) > runtime.compatibility_score(approximate)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_reconstruction_training_freezes_encoder_and_preserves_source_table,
            test_dynamic_reconstruction_keeps_progress_before_alignment_passes,
            test_dynamic_reconstruction_preserves_previously_passing_alignment,
            test_dynamic_reconstruction_resume_matches_uninterrupted_training,
            test_complete_checkpoint_selection_prioritizes_exact_reconstruction,
        )
    )
