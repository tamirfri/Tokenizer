from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
from input_model_fixtures import make_adapter
from torch import nn

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.codec.checkpoints import load_checkpoint, save_checkpoint
from continuous_tokenizer.input.adapter import (
    InputEmbeddingAdapter,
)
from continuous_tokenizer.input.evidence import input_source_identity
from continuous_tokenizer.input.training.distillation import (
    DistillationOptions,
    DistillationRequest,
    FrozenBackboneDistiller,
    distill_checkpoint,
)
from continuous_tokenizer.runtime.resume import ResumeManager
from continuous_tokenizer.runtime.tensors import parameter_fingerprint


class TinyModel(nn.Module):
    def __init__(self, adapter: InputEmbeddingAdapter) -> None:
        super().__init__()
        self.embedding = nn.Embedding(258, 8)
        self.lm_head = nn.Linear(8, 258, bias=False)
        with torch.no_grad():
            self.embedding.weight[:256].copy_(adapter.codec.byte_embeddings)
            self.embedding.weight[256].copy_(adapter.control_embeddings[0])

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(logits=self.lm_head(hidden), past_key_values=None)


def test_distillation_trains_only_codec_parameters() -> None:
    adapter = make_adapter()
    model = TinyModel(adapter)
    assets = ModelAssets(
        model_id="synthetic/model",
        revision="synthetic-revision",
        tokenizer=SimpleNamespace(),
        config={},
        embedding_tensor_name="embedding.weight",
        embedding_shard=Path("synthetic.safetensors"),
        vocabulary=adapter.vocabulary,
        input_embeddings=torch.cat((adapter.codec.byte_embeddings, adapter.control_embeddings, torch.randn(1, 8))),
    )
    before_model = parameter_fingerprint(model)
    before_codec = {name: parameter.detach().clone() for name, parameter in adapter.codec.named_parameters()}
    encoder_parameters, decoder_parameters = adapter.codec.training_parameter_groups()
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    distiller = FrozenBackboneDistiller(
        model,
        adapter,
        assets,
        DistillationOptions(
            epochs=1,
            windows=1,
            prompt_tokens=2,
            continuation_tokens=1,
            vocabulary_replay=2,
        ),
    )

    result = distiller.run((((257, 65), (66,)),))

    assert result.steps == 1
    assert result.trainable_parameters
    assert result.options["optimizer"] == {
        "hidden_matrix_parameters": "Muon",
        "output_and_non_matrix_parameters": "AdamW",
        "muon_adjust_lr_fn": "match_rms_adamw",
        "muon_ns_steps": 5,
    }
    assert result.options["optimization_dtype"] == "torch.float32"
    assert result.options["trainable_component"] == "encoder"
    assert result.options["frozen_component"] == "decoder"
    assert parameter_fingerprint(model) == before_model
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert any(not torch.equal(before_codec[name], parameter) for name, parameter in adapter.codec.named_parameters() if id(parameter) in encoder_ids)
    assert all(torch.equal(before_codec[name], parameter) for name, parameter in adapter.codec.named_parameters() if id(parameter) in decoder_ids)


def test_distillation_optimizes_in_fp32_and_preserves_deployment_dtype() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        adapter = make_adapter()
        adapter.codec.to(dtype=torch.bfloat16)
        model = TinyModel(adapter)
        assets = ModelAssets(
            model_id="synthetic/model",
            revision="synthetic-revision",
            tokenizer=SimpleNamespace(
                encode=lambda _text, *, add_special_tokens: [257, 65, 66],  # noqa: ARG005 - Matches the tokenizer protocol.
            ),
            config={},
            embedding_tensor_name="embedding.weight",
            embedding_shard=Path("synthetic.safetensors"),
            vocabulary=adapter.vocabulary,
            input_embeddings=torch.cat(
                (
                    adapter.codec.byte_embeddings,
                    adapter.control_embeddings.to(torch.bfloat16),
                    torch.randn(1, 8, dtype=torch.bfloat16),
                )
            ),
        )
        source = root / "source.pt"
        output = root / "distilled.pt"
        save_checkpoint(
            source,
            adapter.codec,
            {
                "model_revision": assets.revision,
                "source_identity": input_source_identity(assets),
            },
            control_ids=adapter.control_ids,
            control_embeddings=adapter.control_embeddings.to(torch.bfloat16),
        )

        result = distill_checkpoint(
            DistillationRequest(
                assets=assets,
                checkpoint=source,
                output=output,
                documents=[b"abc"],
                options=DistillationOptions(
                    epochs=1,
                    windows=1,
                    prompt_tokens=2,
                    continuation_tokens=1,
                    vocabulary_replay=2,
                ),
                device=torch.device("cpu"),
                frozen_model=model,
            )
        )

        loaded = load_checkpoint(output)
        assert result.options["optimization_dtype"] == "torch.float32"
        assert loaded.codec.dtype == torch.bfloat16
        assert loaded.metadata["distillation"]["deployment_dtype"] == "torch.bfloat16"


def test_distillation_window_resume_matches_uninterrupted_training() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        torch.manual_seed(17)
        initial_adapter = make_adapter()
        initial_model = TinyModel(initial_adapter)
        assets = ModelAssets(
            model_id="synthetic/model",
            revision="synthetic-revision",
            tokenizer=SimpleNamespace(),
            config={},
            embedding_tensor_name="embedding.weight",
            embedding_shard=Path("synthetic.safetensors"),
            vocabulary=initial_adapter.vocabulary,
            input_embeddings=torch.cat(
                (
                    initial_adapter.codec.byte_embeddings,
                    initial_adapter.control_embeddings,
                    torch.randn(1, 8),
                )
            ),
        )
        options = DistillationOptions(
            epochs=2,
            windows=2,
            prompt_tokens=2,
            continuation_tokens=1,
            vocabulary_replay=2,
        )
        windows = (((257, 65), (66,)), ((257, 66), (65,)))
        interrupted_adapter = copy.deepcopy(initial_adapter)
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
            unittest.TestCase().assertRaises(KeyboardInterrupt),
        ):
            FrozenBackboneDistiller(
                copy.deepcopy(initial_model),
                interrupted_adapter,
                assets,
                options,
            ).run(
                windows,
                resume_manager=ResumeManager(
                    root,
                    "experiment",
                    "commit",
                    "source",
                    "lock",
                    False,
                    snapshot_interval=2,
                ),
            )

        resumed_adapter = copy.deepcopy(initial_adapter)
        resumed_result = FrozenBackboneDistiller(
            copy.deepcopy(initial_model),
            resumed_adapter,
            assets,
            options,
        ).run(
            windows,
            resume_manager=ResumeManager(
                root,
                "experiment",
                "commit",
                "source",
                "lock",
                True,
                snapshot_interval=2,
            ),
        )
        uninterrupted_adapter = copy.deepcopy(initial_adapter)
        uninterrupted_result = FrozenBackboneDistiller(
            copy.deepcopy(initial_model),
            uninterrupted_adapter,
            assets,
            options,
        ).run(windows)

        assert resumed_result.steps == uninterrupted_result.steps
        assert resumed_result.mean_loss == uninterrupted_result.mean_loss
        assert resumed_result.mean_kl == uninterrupted_result.mean_kl
        assert resumed_result.mean_embedding_loss == uninterrupted_result.mean_embedding_loss
        assert resumed_result.mean_reconstruction_loss == uninterrupted_result.mean_reconstruction_loss
        for name, value in uninterrupted_adapter.codec.state_dict().items():
            assert torch.equal(resumed_adapter.codec.state_dict()[name], value), name


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_distillation_trains_only_codec_parameters,
            test_distillation_optimizes_in_fp32_and_preserves_deployment_dtype,
            test_distillation_window_resume_matches_uninterrupted_training,
        )
    )
