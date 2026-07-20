from __future__ import annotations

import gc
import os
import unittest
from pathlib import Path

import torch

from continuous_tokenizer.backbone.assets import load_frozen_causal_lm, load_model_assets
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.codec.batches import build_span_batch, byte_reconstruction_loss
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.codec.output import OutputByteCodec, OutputByteCodecConfig
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.profiles import profile_named


def _model_matrix() -> tuple[tuple[str, str], ...]:
    repository = Path(__file__).parents[1]
    pairs = []
    for model in ("qwen35-0.8b", "gemma3-270m"):
        directions = [ExperimentSpec.load(repository / f"experiments/diagnostics/{model}-{direction}.toml") for direction in ("input", "output")]
        assert directions[0].model == directions[1].model
        pairs.append((directions[0].model.model_id, directions[0].model.revision))
    return tuple(pairs)


@unittest.skipUnless(os.environ.get("RUN_REAL_MODEL_TESTS") == "1", "set RUN_REAL_MODEL_TESTS=1")
def test_pinned_models_run_bounded_input_output_and_large_mps_diagnostics() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("bounded real-model diagnostics require an available MPS device")
    device = torch.device("mps")
    profile = profile_named("large")
    for model_id, revision in _model_matrix():
        with unittest.TestCase().subTest(model_id=model_id):
            assets = load_model_assets(model_id, revision)
            model = load_frozen_causal_lm(assets, device)
            backbone = FrozenBackbone(model)
            prompt_ids = tuple(assets.tokenizer.encode("bounded tokenizer diagnostic", add_special_tokens=True)[:8])
            if not prompt_ids:
                raise AssertionError(f"{model_id} produced an empty diagnostic prompt")
            token_ids = torch.tensor((prompt_ids,), dtype=torch.long, device=device)
            with torch.inference_mode():
                prompt_embeddings = backbone.input_embeddings(token_ids)
                forwarded = backbone.forward(inputs_embeds=prompt_embeddings, use_cache=False)
            hidden = forwarded.last_hidden_state[:, -1].float()
            assert forwarded.last_hidden_state.shape[:2] == token_ids.shape
            assert all(not parameter.requires_grad for parameter in model.parameters())

            byte_rows = assets.input_embeddings[torch.tensor(assets.vocabulary.byte_token_ids)].float()
            input_codec = InputByteCodec(
                InputByteCodecConfig(
                    embedding_dim=byte_rows.shape[1],
                    local_dim=profile.local_dim,
                    projection_dim=profile.projection_dim(byte_rows.shape[1]),
                    max_span=8,
                    query_heads=profile.query_heads,
                    feedforward_dim=profile.feedforward_dim,
                    encoder_layers=profile.encoder_layers,
                    decoder_layers=profile.decoder_layers,
                ),
                byte_rows,
            ).to(device)
            input_codec.train()
            input_codec.compile_neural_paths()
            batch = build_span_batch([b"bounded"], max_span=input_codec.max_span, device=device)
            _, input_logits = input_codec.reconstruction_logits(batch.byte_values, batch.valid_mask)
            input_loss = byte_reconstruction_loss(input_logits, batch.framed_targets, batch.target_mask)
            input_loss.backward()
            assert torch.isfinite(input_loss)
            assert input_codec.input_projection.weight.grad is not None

            output_codec = OutputByteCodec(
                OutputByteCodecConfig(
                    embedding_dim=hidden.shape[1],
                    local_dim=profile.local_dim,
                    max_span=8,
                    feedforward_dim=profile.feedforward_dim,
                    decoder_layers=profile.decoder_layers,
                    control_count=len(assets.vocabulary.control_ids),
                )
            ).to(device)
            output_codec.train()
            output_codec.compile_neural_paths()
            byte_logits, control_logits = output_codec.decode_logits(hidden)
            output_loss = byte_logits.square().mean() + control_logits.square().mean()
            output_loss.backward()
            torch.mps.synchronize()
            assert torch.isfinite(output_loss)
            assert output_codec.hidden_projection.weight.grad is not None

            del output_codec, input_codec, hidden, forwarded, prompt_embeddings, token_ids, backbone, model, assets
            gc.collect()
            torch.mps.empty_cache()


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite((unittest.FunctionTestCase(test_pinned_models_run_bounded_input_output_and_large_mps_diagnostics),))
