from __future__ import annotations

import unittest
from typing import cast
from unittest import mock

import torch
from torch import Tensor, nn

from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.synthetic import SyntheticCausalLM, synthetic_model_assets
from continuous_tokenizer.campaigns.output import _seeded_output_codec
from continuous_tokenizer.codec.output import (
    OutputByteCodec,
    OutputByteCodecConfig,
    decode_output_batch,
)
from continuous_tokenizer.output.training import (
    OutputCodecTrainer,
    OutputTrainerContext,
    OutputTrainingOptions,
)
from continuous_tokenizer.output.trajectory_cache import (
    OutputTrajectoryOptions,
    build_prepared_output_corpus,
)
from continuous_tokenizer.runtime.tensors import parameter_fingerprint


def _small_output_config() -> OutputByteCodecConfig:
    return OutputByteCodecConfig(
        embedding_dim=16,
        local_dim=8,
        max_span=2,
        feedforward_dim=16,
        decoder_layers=1,
        control_count=0,
    )


class OutputModeTests(unittest.TestCase):
    def test_output_codec_construction_is_seeded_deterministically(self) -> None:
        config = _small_output_config()
        torch.manual_seed(999)
        torch.rand(5)
        first = _seeded_output_codec(
            config,
            seed=23,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        torch.manual_seed(41)
        torch.rand(7)
        second = _seeded_output_codec(
            config,
            seed=23,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertTrue(all(torch.equal(first.state_dict()[name], value) for name, value in second.state_dict().items()))

    def test_output_training_preserves_frozen_backbone(self) -> None:
        assets = synthetic_model_assets()
        model = SyntheticCausalLM(assets.input_embeddings)
        backbone = FrozenBackbone(model)
        codec = OutputByteCodec(_small_output_config())
        before = parameter_fingerprint(model)
        trainer = OutputCodecTrainer(
            codec,
            OutputTrainingOptions(
                epochs=1,
                batch_size=8,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=17,
            ),
            OutputTrainerContext(
                backbone=backbone,
                vocabulary=assets.vocabulary,
                deployment_dtype=torch.float32,
            ),
        )
        training_sequences = (tuple(range(16)),)
        selection_sequences = (tuple(range(16, 32)),)
        training_corpus = build_prepared_output_corpus(
            backbone,
            assets.vocabulary,
            training_sequences,
            OutputTrajectoryOptions(max_span=codec.max_span),
        )
        selection_corpus = build_prepared_output_corpus(
            backbone,
            assets.vocabulary,
            selection_sequences,
            OutputTrajectoryOptions(max_span=codec.max_span),
        )
        with (
            mock.patch.object(trainer, "_evaluate", return_value=1.0) as evaluate,
            mock.patch.object(backbone, "forward", wraps=backbone.forward) as forward,
        ):
            result = trainer.run(training_corpus, selection_corpus)

        self.assertIs(evaluate.call_args.args[0], selection_corpus)
        forward.assert_not_called()
        self.assertEqual(parameter_fingerprint(model), before)
        self.assertEqual(result.backbone_fingerprint, before)
        self.assertTrue(result.trainable_parameters)

    def test_output_decode_batches_use_power_of_two_mps_shapes(self) -> None:
        class RecordingCodec(nn.Module):
            def __init__(self, device_type: str) -> None:
                super().__init__()
                self._device = torch.device(device_type)
                self.rows: list[int] = []

            @property
            def device(self) -> torch.device:
                return self._device

            def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
                self.rows.append(hidden.shape[0])
                return (
                    torch.zeros((hidden.shape[0], 2, 257)),
                    torch.zeros((hidden.shape[0], 1)),
                )

        mps_codec = RecordingCodec("mps")
        cpu_codec = RecordingCodec("cpu")

        byte_logits, control_logits = decode_output_batch(
            cast(OutputByteCodec, mps_codec),
            torch.zeros((5, 16)),
            maximum_rows=8,
        )
        decode_output_batch(
            cast(OutputByteCodec, cpu_codec),
            torch.zeros((5, 16)),
            maximum_rows=8,
        )
        unbounded_byte_logits, unbounded_control_logits = decode_output_batch(
            cast(OutputByteCodec, mps_codec),
            torch.zeros((9, 16)),
        )

        self.assertEqual(mps_codec.rows, [8, 16])
        self.assertEqual(cpu_codec.rows, [5])
        self.assertEqual(byte_logits.shape, (5, 2, 257))
        self.assertEqual(control_logits.shape, (5, 1))
        self.assertEqual(unbounded_byte_logits.shape, (9, 2, 257))
        self.assertEqual(unbounded_control_logits.shape, (9, 1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            decode_output_batch(
                cast(OutputByteCodec, mps_codec),
                torch.zeros((9, 16)),
                maximum_rows=8,
            )

    def test_output_state_loading_preserves_compiled_decode(self) -> None:
        codec = OutputByteCodec(_small_output_config())
        compiled_decode = codec._decode_tensor
        codec._compiled_decode = compiled_decode

        codec.load_state_dict(codec.state_dict())

        self.assertIs(codec._compiled_decode, compiled_decode)
