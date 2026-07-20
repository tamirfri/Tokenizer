from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest import mock

import torch
from torch import Tensor, nn

import continuous_tokenizer.output.evaluation as output_evaluation_module
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.synthetic import SyntheticCausalLM, synthetic_model_assets
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.output import (
    OutputByteCodec,
    OutputByteCodecConfig,
)
from continuous_tokenizer.output.evaluation import (
    OutputEvaluationOptions,
    _mps_staging_policy,
    evaluate_output_codec,
)
from continuous_tokenizer.output.training import (
    OutputCodecTrainer,
    OutputTrainerContext,
    OutputTrainingOptions,
)
from continuous_tokenizer.output.trajectory_cache import (
    OutputCacheIdentity,
    OutputCorpusPreparation,
    OutputTrajectoryOptions,
    _cache_descriptor,
    _cache_key,
    _OutputCacheEnvironment,
    build_prepared_output_corpus,
    native_head_oracle_ceilings,
    prepare_output_corpus,
    prepared_output_corpus_digest,
)
from continuous_tokenizer.runtime.resume import ResumeManager


def _small_output_config() -> OutputByteCodecConfig:
    return OutputByteCodecConfig(
        embedding_dim=16,
        local_dim=8,
        max_span=2,
        feedforward_dim=16,
        decoder_layers=1,
        control_count=0,
    )


class _FixedNativeHead(nn.Module):
    def __init__(self, vocabulary_size: int, token_id: int) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.token_id = token_id

    def forward(self, hidden: Tensor) -> Tensor:
        logits = torch.full(
            (*hidden.shape[:-1], self.vocabulary_size),
            -1.0,
            device=hidden.device,
        )
        logits[..., self.token_id] = 1.0
        return logits


class OutputModeTests(unittest.TestCase):
    def test_output_cache_key_invalidates_for_every_bound_identity(self) -> None:
        identity = OutputCacheIdentity(
            source_commit="commit",
            source_dirty=True,
            source_state_sha256="source",
            dependency_lock_sha256="lock",
            model_revision="model",
            model_config_sha256="config",
            frozen_backbone_fingerprint="backbone",
            tokenizer_revision="tokenizer",
        )

        def key(
            selected: OutputCacheIdentity = identity,
            *,
            transformers_version: str = "5.14.1",
            model_implementation: str = "module.Model",
        ) -> str:
            return _cache_key(
                _cache_descriptor(
                    OutputCorpusPreparation(
                        identity=selected,
                        split="training",
                        trajectory=OutputTrajectoryOptions(
                            max_span=8,
                            stop_control_ids=frozenset({256}),
                            max_native_tokens=8,
                            max_bytes=128,
                        ),
                    ),
                    _OutputCacheEnvironment(
                        transformers_version=transformers_version,
                        model_implementation=model_implementation,
                        dtype=torch.bfloat16,
                    ),
                    corpus_sha256="corpus",
                )
            )

        baseline = key()
        replacements = {
            "source_commit": "other-commit",
            "source_dirty": False,
            "source_state_sha256": "other-source",
            "dependency_lock_sha256": "other-lock",
            "model_revision": "other-model",
            "model_config_sha256": "other-config",
            "frozen_backbone_fingerprint": "other-backbone",
            "tokenizer_revision": "other-tokenizer",
        }
        for field, value in replacements.items():
            with self.subTest(field=field):
                self.assertNotEqual(key(replace(identity, **{field: value})), baseline)
        self.assertNotEqual(key(transformers_version="5.15.0"), baseline)
        self.assertNotEqual(key(model_implementation="other.Model"), baseline)

    def test_native_head_oracle_ceiling_reports_registered_span_limits(self) -> None:
        token_bytes = (*tuple(bytes([value]) for value in range(256)), b"abcd")
        vocabulary = ByteVocabulary(
            token_bytes,
            tuple(range(257)),
            (),
            tuple(range(256)),
            4,
        )
        model = SyntheticCausalLM(torch.randn(257, 8))
        cast(Any, model).lm_head = _FixedNativeHead(257, 256)
        corpus = build_prepared_output_corpus(
            FrozenBackbone(model),
            vocabulary,
            ((0,),),
            OutputTrajectoryOptions(
                max_span=4,
                max_native_tokens=1,
                max_bytes=4,
            ),
        )

        ceilings = native_head_oracle_ceilings(corpus, vocabulary)

        self.assertEqual(tuple(ceilings), ("1", "2", "4", "8"))
        self.assertFalse(ceilings["1"]["feasible"])
        self.assertIsNone(ceilings["1"]["exact_native_sequence_rate_ceiling"])
        self.assertTrue(ceilings["4"]["feasible"])
        self.assertEqual(ceilings["4"]["exact_native_sequence_rate_ceiling"], 1.0)

    def test_prepared_corpus_serializes_ragged_native_token_targets(self) -> None:
        vocabulary = ByteVocabulary(
            tuple(bytes([value]) for value in range(256)),
            tuple(range(256)),
            (),
            tuple(range(256)),
            1,
        )
        model = SyntheticCausalLM(torch.randn(256, 8))
        cast(Any, model).lm_head = _FixedNativeHead(256, ord("a"))
        corpus = build_prepared_output_corpus(
            FrozenBackbone(model),
            vocabulary,
            ((0,),),
            OutputTrajectoryOptions(
                max_span=2,
                max_native_tokens=3,
                max_bytes=3,
            ),
        )

        self.assertEqual(corpus.target_native_token_offsets, (0, 2, 3))
        self.assertEqual(corpus.target_native_tokens(0), (ord("a"), ord("a")))
        self.assertEqual(corpus.target_native_tokens(1), (ord("a"),))

    def test_prepared_output_corpus_cache_is_exact_and_reuses_backbone_work(self) -> None:
        assets = synthetic_model_assets()
        backbone = FrozenBackbone(SyntheticCausalLM(assets.input_embeddings))
        sequences = (tuple(range(16)), tuple(range(16, 32)))
        identity = OutputCacheIdentity(
            source_commit="commit",
            source_dirty=True,
            source_state_sha256="source",
            dependency_lock_sha256="lock",
            model_revision=assets.revision,
            model_config_sha256="config",
            frozen_backbone_fingerprint="backbone",
            tokenizer_revision=assets.revision,
        )

        def preparation(selected: OutputCacheIdentity) -> OutputCorpusPreparation:
            return OutputCorpusPreparation(
                identity=selected,
                split="training",
                trajectory=OutputTrajectoryOptions(max_span=2),
                cache_directory=Path(directory),
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(backbone, "forward", wraps=backbone.forward) as forward,
        ):
            fresh, fresh_info = prepare_output_corpus(
                backbone,
                assets.vocabulary,
                sequences,
                preparation(identity),
            )
            fresh_calls = forward.call_count
            cached, cached_info = prepare_output_corpus(
                backbone,
                assets.vocabulary,
                sequences,
                preparation(identity),
            )
            changed_source, changed_source_info = prepare_output_corpus(
                backbone,
                assets.vocabulary,
                sequences,
                preparation(
                    replace(
                        identity,
                        source_commit="other-commit",
                    ),
                ),
            )
            _, changed_lock_info = prepare_output_corpus(
                backbone,
                assets.vocabulary,
                sequences,
                preparation(
                    replace(identity, dependency_lock_sha256="other-lock"),
                ),
            )
            _, changed_backbone_info = prepare_output_corpus(
                backbone,
                assets.vocabulary,
                sequences,
                preparation(
                    replace(
                        identity,
                        frozen_backbone_fingerprint="other-backbone",
                    ),
                ),
            )

        self.assertFalse(fresh_info.hit)
        self.assertTrue(cached_info.hit)
        self.assertFalse(changed_source_info.hit)
        self.assertFalse(changed_lock_info.hit)
        self.assertFalse(changed_backbone_info.hit)
        self.assertEqual(forward.call_count, fresh_calls * 4)
        self.assertEqual(fresh.hidden.device.type, "cpu")
        self.assertEqual(fresh.sequence_offsets, cached.sequence_offsets)
        self.assertEqual(fresh.sequence_offsets, changed_source.sequence_offsets)
        self.assertEqual(fresh_info.trajectory_sha256, cached_info.trajectory_sha256)
        self.assertEqual(
            prepared_output_corpus_digest(fresh),
            prepared_output_corpus_digest(cached),
        )
        for name, tensor in fresh.tensors().items():
            self.assertTrue(torch.equal(tensor, cached.tensors()[name]), name)

    def test_output_epoch_resume_matches_uninterrupted_training(self) -> None:
        assets = synthetic_model_assets()
        model = SyntheticCausalLM(assets.input_embeddings)
        backbone = FrozenBackbone(model)
        torch.manual_seed(17)
        initial = OutputByteCodec(_small_output_config())
        training = build_prepared_output_corpus(
            backbone,
            assets.vocabulary,
            (tuple(range(16)),),
            OutputTrajectoryOptions(max_span=2),
        )
        selection = build_prepared_output_corpus(
            backbone,
            assets.vocabulary,
            (tuple(range(16, 32)),),
            OutputTrajectoryOptions(max_span=2),
        )
        options = OutputTrainingOptions(
            epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=17,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interrupted_manager = ResumeManager(
                root,
                "experiment",
                "commit",
                "source",
                "lock",
                False,
            )
            interrupted = copy.deepcopy(initial)
            original_save = ResumeManager.save

            def save_then_interrupt(manager, phase, epoch, state):
                original_save(manager, phase, epoch, state)
                raise KeyboardInterrupt

            with (
                mock.patch.object(
                    ResumeManager,
                    "save",
                    autospec=True,
                    side_effect=save_then_interrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                OutputCodecTrainer(
                    interrupted,
                    options,
                    OutputTrainerContext(
                        backbone=backbone,
                        vocabulary=assets.vocabulary,
                        deployment_dtype=torch.float32,
                        resume_manager=interrupted_manager,
                    ),
                ).run(training, selection)

            resumed = copy.deepcopy(initial)
            resumed_result = OutputCodecTrainer(
                resumed,
                options,
                OutputTrainerContext(
                    backbone=backbone,
                    vocabulary=assets.vocabulary,
                    deployment_dtype=torch.float32,
                    resume_manager=ResumeManager(
                        root,
                        "experiment",
                        "commit",
                        "source",
                        "lock",
                        True,
                    ),
                ),
            ).run(training, selection)
            uninterrupted = copy.deepcopy(initial)
            uninterrupted_result = OutputCodecTrainer(
                uninterrupted,
                options,
                OutputTrainerContext(
                    backbone=backbone,
                    vocabulary=assets.vocabulary,
                    deployment_dtype=torch.float32,
                ),
            ).run(training, selection)

        self.assertEqual(resumed_result, uninterrupted_result)
        for name, value in uninterrupted.state_dict().items():
            self.assertTrue(torch.equal(resumed.state_dict()[name], value), name)

    def test_output_training_honors_snapshot_interval_and_final_state(self) -> None:
        assets = synthetic_model_assets()
        backbone = FrozenBackbone(SyntheticCausalLM(assets.input_embeddings))
        training = build_prepared_output_corpus(
            backbone,
            assets.vocabulary,
            (tuple(range(16)),),
            OutputTrajectoryOptions(max_span=2),
        )
        selection = build_prepared_output_corpus(
            backbone,
            assets.vocabulary,
            (tuple(range(16, 32)),),
            OutputTrajectoryOptions(max_span=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ResumeManager(
                root,
                "experiment",
                "commit",
                "source",
                "lock",
                False,
                snapshot_interval=2,
            )
            trainer = OutputCodecTrainer(
                OutputByteCodec(_small_output_config()),
                OutputTrainingOptions(
                    epochs=3,
                    batch_size=8,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    seed=17,
                ),
                OutputTrainerContext(
                    backbone=backbone,
                    vocabulary=assets.vocabulary,
                    deployment_dtype=torch.float32,
                    resume_manager=manager,
                ),
            )
            with mock.patch.object(trainer, "_evaluate", return_value=0.0):
                trainer.run(training, selection)

            self.assertEqual(manager.telemetry()["snapshots_written"], 2)
            completed = ResumeManager(
                root,
                "experiment",
                "commit",
                "source",
                "lock",
                True,
                snapshot_interval=2,
            ).latest("output-codec")
            self.assertIsNotNone(completed)
            assert completed is not None
            self.assertTrue(completed["completed"])
            self.assertEqual(completed["epoch"], 3)
            self.assertIsNotNone(completed["best_codec"])

    def test_output_evaluation_batches_across_sequences_and_partial_tail(self) -> None:
        assets = synthetic_model_assets()
        backbone = FrozenBackbone(SyntheticCausalLM(assets.input_embeddings))
        corpus = build_prepared_output_corpus(
            backbone,
            assets.vocabulary,
            (tuple(range(8)), tuple(range(8, 16))),
            OutputTrajectoryOptions(max_span=2),
        )
        codec = OutputByteCodec(_small_output_config())
        batch_size = max(corpus.examples - 1, 2)
        rows: list[int] = []
        original = output_evaluation_module.decode_output_batch

        def record_rows(
            selected_codec: OutputByteCodec,
            hidden: Tensor,
            *,
            maximum_rows: int,
        ) -> tuple[Tensor, Tensor]:
            rows.append(hidden.shape[0])
            return original(
                selected_codec,
                hidden,
                maximum_rows=maximum_rows,
            )

        with mock.patch.object(
            output_evaluation_module,
            "decode_output_batch",
            side_effect=record_rows,
        ):
            metrics = evaluate_output_codec(
                codec,
                corpus,
                assets.vocabulary,
                OutputEvaluationOptions(batch_size=batch_size),
            )

        self.assertEqual(rows, [batch_size, 1])
        self.assertEqual(metrics.evaluation_telemetry["batches"], 2)
        self.assertLess(
            cast(float, metrics.evaluation_telemetry["batch_fill_ratio"]),
            1.0,
        )
        self.assertEqual(
            metrics.evaluation_telemetry["sequence_order"],
            (0, 1),
        )

    def test_mps_staging_policy_records_deterministic_fallback(self) -> None:
        self.assertEqual(
            _mps_staging_policy(torch.device("mps"), 128, None),
            (False, "guard_disabled"),
        )
        self.assertEqual(
            _mps_staging_policy(torch.device("mps"), 128, 127),
            (False, "memory_guard_exceeded"),
        )
        self.assertEqual(
            _mps_staging_policy(torch.device("mps"), 128, 128),
            (True, None),
        )
        self.assertEqual(
            _mps_staging_policy(torch.device("cpu"), 128, 128),
            (False, "not_mps"),
        )

    def test_output_evaluation_resume_is_metric_equivalent(self) -> None:
        assets = synthetic_model_assets()
        backbone = FrozenBackbone(SyntheticCausalLM(assets.input_embeddings))
        corpus = build_prepared_output_corpus(
            backbone,
            assets.vocabulary,
            (tuple(range(8)), tuple(range(8, 16))),
            OutputTrajectoryOptions(max_span=2),
        )
        torch.manual_seed(17)
        codec = OutputByteCodec(_small_output_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_save = ResumeManager.save

            def save_then_interrupt(manager, phase, epoch, state):
                original_save(manager, phase, epoch, state)
                raise KeyboardInterrupt

            with (
                mock.patch.object(
                    ResumeManager,
                    "save",
                    autospec=True,
                    side_effect=save_then_interrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                evaluate_output_codec(
                    codec,
                    corpus,
                    assets.vocabulary,
                    OutputEvaluationOptions(
                        batch_size=4,
                        resume_manager=ResumeManager(
                            root,
                            "experiment",
                            "commit",
                            "source",
                            "lock",
                            False,
                        ),
                    ),
                )

            resumed = evaluate_output_codec(
                codec,
                corpus,
                assets.vocabulary,
                OutputEvaluationOptions(
                    batch_size=4,
                    resume_manager=ResumeManager(
                        root,
                        "experiment",
                        "commit",
                        "source",
                        "lock",
                        True,
                    ),
                ),
            )
            uninterrupted = evaluate_output_codec(
                codec,
                corpus,
                assets.vocabulary,
                OutputEvaluationOptions(batch_size=4),
            )

        self.assertEqual(resumed, uninterrupted)
