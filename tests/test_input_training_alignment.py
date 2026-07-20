from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from input_training_fixtures import (
    TEST_PROFILE,
    pair_assets,
    subset_assets,
    synthetic_assets,
)

import continuous_tokenizer.input.training.runtime as selection_module
import continuous_tokenizer.input.training.vocabulary as vocabulary_module
from continuous_tokenizer.contracts.profiles import Profile
from continuous_tokenizer.input.alignment import (
    EmbeddingEvaluationRequest,
    embedding_alignment_loss,
    evaluate_embeddings,
)
from continuous_tokenizer.input.training.run import TrainingOptions
from continuous_tokenizer.input.training.vocabulary_batches import (
    build_vocabulary_batches,
    build_vocabulary_groups,
)
from continuous_tokenizer.runtime.resume import ResumeManager


def test_embedding_alignment_loss_weights_rows_independently() -> None:
    target = torch.tensor(((1.0, 0.0), (100.0, 0.0)))
    equally_wrong = torch.tensor(((0.0, 1.0), (0.0, 100.0)))
    first_row_wrong = torch.tensor(((0.0, 1.0), (100.0, 0.0)))

    assert torch.isclose(embedding_alignment_loss(target, target), torch.tensor(0.0))
    assert torch.isclose(
        embedding_alignment_loss(first_row_wrong, target) * 2,
        embedding_alignment_loss(equally_wrong, target),
    )


def test_checkpoint_selection_evaluates_source_dtype_state() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = synthetic_assets(root)
        assets.input_embeddings = assets.input_embeddings.to(torch.bfloat16)
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=64,
            ),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(TEST_PROFILE)
        evaluator = runtime.deployment_evaluator(codec)
        assert codec.dtype == torch.float32
        assert evaluator.dtype == torch.bfloat16
        assert all(not parameter.requires_grad for parameter in evaluator.parameters())
        validation_rows: list[int] = []

        def validate(byte_values, valid_mask):
            validation_rows.append(byte_values.shape[0])
            return evaluator._validation_tensor(byte_values, valid_mask)

        evaluator._compiled_validation = validate

        with torch.no_grad():
            codec.cls.fill_(1.234567)
        runtime.evaluate_deployment(codec, evaluator)
        with torch.no_grad():
            codec.cls.fill_(2.345678)
        runtime.evaluate_deployment(codec, runtime.deployment_evaluator(codec))

        assert torch.equal(evaluator.cls, codec.cls.to(torch.bfloat16))
        assert len(validation_rows) == 8
        assert set(validation_rows) == {64}


def test_input_alignment_epoch_resume_matches_uninterrupted_training() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        options = TrainingOptions(
            output_dir=root / "checkpoints",
            profile=TEST_PROFILE,
            batch_size=1,
            vocabulary_epochs=2,
            patience=10,
            evaluation_interval=1,
        )
        torch.manual_seed(17)
        initial_runtime = selection_module.TrainingRuntime(
            assets,
            options,
            torch.device("cpu"),
        )
        initial = initial_runtime.build_codec(TEST_PROFILE)
        interrupted_manager = ResumeManager(
            root,
            "experiment",
            "commit",
            "source",
            "lock",
            False,
        )
        interrupted_runtime = selection_module.TrainingRuntime(
            assets,
            options,
            torch.device("cpu"),
            interrupted_manager,
        )
        interrupted = copy.deepcopy(initial)
        original_save = ResumeManager.save

        def save_then_interrupt(manager, phase, epoch, state):
            original_save(manager, phase, epoch, state)
            raise KeyboardInterrupt

        with (
            patch.object(
                ResumeManager,
                "save",
                autospec=True,
                side_effect=save_then_interrupt,
            ),
            unittest.TestCase().assertRaises(KeyboardInterrupt),
        ):
            vocabulary_module.VocabularyFitter(interrupted_runtime).fit_encoder(
                interrupted,
                TEST_PROFILE,
                torch.Generator().manual_seed(17),
                token_ids=(256,),
            )

        resumed = copy.deepcopy(initial)
        resumed_runtime = selection_module.TrainingRuntime(
            assets,
            options,
            torch.device("cpu"),
            ResumeManager(
                root,
                "experiment",
                "commit",
                "source",
                "lock",
                True,
            ),
        )
        vocabulary_module.VocabularyFitter(resumed_runtime).fit_encoder(
            resumed,
            TEST_PROFILE,
            torch.Generator().manual_seed(17),
            token_ids=(256,),
        )
        uninterrupted = copy.deepcopy(initial)
        vocabulary_module.VocabularyFitter(initial_runtime).fit_encoder(
            uninterrupted,
            TEST_PROFILE,
            torch.Generator().manual_seed(17),
            token_ids=(256,),
        )

        for name, value in uninterrupted.state_dict().items():
            assert torch.equal(resumed.state_dict()[name], value), name


def test_input_vocabulary_decoder_resume_matches_uninterrupted_training() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        options = TrainingOptions(
            output_dir=root / "checkpoints",
            profile=TEST_PROFILE,
            batch_size=1,
            vocabulary_epochs=2,
            patience=10,
            evaluation_interval=2,
        )
        torch.manual_seed(17)
        initial_runtime = selection_module.TrainingRuntime(assets, options, torch.device("cpu"))
        initial = initial_runtime.build_codec(TEST_PROFILE)
        groups = vocabulary_module.build_vocabulary_groups(assets, (256,))
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
            vocabulary_module.VocabularyFitter(interrupted_runtime)._fit_decoder(
                interrupted,
                TEST_PROFILE,
                torch.Generator().manual_seed(17),
                groups,
                (256,),
            )

        resumed = copy.deepcopy(initial)
        resumed_runtime = selection_module.TrainingRuntime(
            assets,
            options,
            torch.device("cpu"),
            ResumeManager(root, "experiment", "commit", "source", "lock", True),
        )
        vocabulary_module.VocabularyFitter(resumed_runtime)._fit_decoder(
            resumed,
            TEST_PROFILE,
            torch.Generator().manual_seed(17),
            groups,
            (256,),
        )
        uninterrupted = copy.deepcopy(initial)
        vocabulary_module.VocabularyFitter(initial_runtime)._fit_decoder(
            uninterrupted,
            TEST_PROFILE,
            torch.Generator().manual_seed(17),
            groups,
            (256,),
        )

        for name, value in uninterrupted.state_dict().items():
            assert torch.equal(resumed.state_dict()[name], value), name


def test_input_alignment_honors_recovery_snapshot_interval_and_final_state() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        options = TrainingOptions(
            output_dir=root / "checkpoints",
            profile=TEST_PROFILE,
            batch_size=1,
            vocabulary_epochs=3,
            patience=10,
            evaluation_interval=1,
        )
        manager = ResumeManager(
            root,
            "experiment",
            "commit",
            "source",
            "lock",
            False,
            snapshot_interval=2,
        )
        runtime = selection_module.TrainingRuntime(
            assets,
            options,
            torch.device("cpu"),
            manager,
        )
        codec = runtime.build_codec(TEST_PROFILE)

        vocabulary_module.VocabularyFitter(runtime).fit_encoder(
            codec,
            TEST_PROFILE,
            torch.Generator().manual_seed(17),
            token_ids=(256,),
        )

        telemetry = manager.telemetry()
        assert telemetry["snapshot_interval"] == 2
        assert telemetry["snapshots_written"] == 2
        assert telemetry["snapshot_bytes_written"] > 0
        state = manager.latest("input-alignment")
        assert state is None
        resumed = ResumeManager(
            root,
            "experiment",
            "commit",
            "source",
            "lock",
            True,
            snapshot_interval=2,
        ).latest("input-alignment")
        assert resumed is not None
        assert resumed["completed"] is True
        assert resumed["epoch"] == 3
        assert resumed["best_codec"]


def test_alignment_evaluation_does_not_run_decoder() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = synthetic_assets(root)
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(output_dir=root / "checkpoints", profile=TEST_PROFILE),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(Profile("test", 8, 1, 1, 1, 4, 2, 16))

        with patch.object(codec, "reconstruction_matches") as reconstruction_matches:
            metrics = evaluate_embeddings(
                codec,
                assets.vocabulary,
                assets.input_embeddings,
                EmbeddingEvaluationRequest(
                    batch_size=256,
                    device=torch.device("cpu"),
                    reconstruction=False,
                ),
            )

        reconstruction_matches.assert_not_called()
        assert metrics.reconstruction_rows == 0


def test_alignment_evaluation_can_use_a_registered_vocabulary_subset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = pair_assets(root)
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(output_dir=root / "checkpoints", profile=TEST_PROFILE),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(Profile("test", 8, 1, 1, 1, 4, 2, 16))
        selected = assets.vocabulary.ordinary_ids

        metrics = evaluate_embeddings(
            codec,
            assets.vocabulary,
            assets.input_embeddings,
            EmbeddingEvaluationRequest(
                batch_size=256,
                device=torch.device("cpu"),
                reconstruction=False,
                token_ids=selected,
            ),
        )

        assert metrics.rows == len(selected)
        with patch.object(
            selection_module.TrainingRuntime,
            "evaluate",
            autospec=True,
            return_value=metrics,
        ) as evaluation:
            assert (
                runtime.evaluate_deployment(
                    codec,
                    codec,
                    reconstruction=False,
                    token_ids=selected,
                )
                is metrics
            )
        evaluation.assert_called_once_with(
            runtime,
            codec,
            reconstruction=False,
            token_ids=selected,
        )
        with unittest.TestCase().assertRaisesRegex(ValueError, "unique compatibility"):
            evaluate_embeddings(
                codec,
                assets.vocabulary,
                assets.input_embeddings,
                EmbeddingEvaluationRequest(
                    batch_size=256,
                    device=torch.device("cpu"),
                    token_ids=(*selected, selected[0]),
                ),
            )


def test_qwen_sized_evaluation_and_training_batches_use_static_rows() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = subset_assets(root)
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=64,
                cache_chunk_rows=64,
            ),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(TEST_PROFILE).eval()
        validation_rows: list[int] = []

        def validate(byte_values, valid_mask):
            validation_rows.append(byte_values.shape[0])
            return codec._validation_tensor(byte_values, valid_mask)

        codec._compiled_validation = validate
        metrics = runtime.evaluate(codec)
        groups = build_vocabulary_groups(assets)
        batches = build_vocabulary_batches(
            groups,
            64,
            torch.Generator().manual_seed(17),
        )

        assert metrics.rows == 2048
        assert validation_rows
        assert set(validation_rows) == {64}
        assert {len(group.token_ids) % 64 for group in groups} == {24, 40}
        assert all(len(batch.rows) == 64 for batch in batches)
        assert sum(batch.logical_rows for batch in batches) == 2048 - 256


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_embedding_alignment_loss_weights_rows_independently,
            test_checkpoint_selection_evaluates_source_dtype_state,
            test_input_alignment_epoch_resume_matches_uninterrupted_training,
            test_input_vocabulary_decoder_resume_matches_uninterrupted_training,
            test_input_alignment_honors_recovery_snapshot_interval_and_final_state,
            test_alignment_evaluation_does_not_run_decoder,
            test_alignment_evaluation_can_use_a_registered_vocabulary_subset,
            test_qwen_sized_evaluation_and_training_batches_use_static_rows,
        )
    )
