from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from input_training_fixtures import TEST_PROFILE, pair_assets, synthetic_assets
from torch import nn

import continuous_tokenizer.input.training.runtime as selection_module
import continuous_tokenizer.input.training.vocabulary as vocabulary_module
from continuous_tokenizer.backbone.synthetic import synthetic_model_assets
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.batches import (
    build_span_batch,
    span_bucket_width,
)
from continuous_tokenizer.contracts.profiles import Profile
from continuous_tokenizer.input.training.run import TrainingOptions
from continuous_tokenizer.input.training.vocabulary_batches import (
    build_vocabulary_batches,
    build_vocabulary_groups,
)
from continuous_tokenizer.training.optimizers import build_tokenizer_optimizers


def test_vocabulary_parameter_groups_are_disjoint_and_complete() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = selection_module.TrainingRuntime(
            synthetic_assets(root),
            TrainingOptions(output_dir=root / "checkpoints", profile=TEST_PROFILE),
            torch.device("cpu"),
        )
        codec = runtime.build_codec(Profile("test", 8, 1, 1, 1, 4, 2, 16))
        encoder, decoder = codec.training_parameter_groups()

        assert not {id(parameter) for parameter in encoder} & {id(parameter) for parameter in decoder}
        assert {id(parameter) for parameter in (*encoder, *decoder)} == {id(parameter) for parameter in codec.parameters()}

        parameters = (*encoder, *decoder)
        optimizers = runtime.optimizers(codec, parameters)
        muon = next(value for value in optimizers.values if isinstance(value, torch.optim.Muon))
        adamw = next(value for value in optimizers.values if isinstance(value, torch.optim.AdamW))
        muon_ids = {id(parameter) for group in muon.param_groups for parameter in group["params"]}
        adamw_ids = {id(parameter) for group in adamw.param_groups for parameter in group["params"]}
        expected_muon, expected_adamw = codec.optimizer_parameter_groups(parameters)
        assert muon_ids == {id(parameter) for parameter in expected_muon}
        assert adamw_ids == {id(parameter) for parameter in expected_adamw}
        assert id(codec.input_projection.weight) in muon_ids
        assert id(codec.output_projection[0].weight) in muon_ids
        assert id(codec.decoder_projection.weight) in muon_ids
        assert id(codec.output_projection[-1].weight) in muon_ids
        assert id(codec.byte_head.weight) in adamw_ids
        assert id(codec.output_projection[-1].bias) in adamw_ids
        assert id(codec.byte_head.bias) in adamw_ids


def test_tokenizer_optimizers_partition_parameters_between_muon_and_adamw() -> None:
    matrix = nn.Parameter(torch.randn(4, 4))
    vector = nn.Parameter(torch.randn(4))
    optimizers = build_tokenizer_optimizers(
        (matrix,),
        (vector,),
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    muon = next(value for value in optimizers.values if isinstance(value, torch.optim.Muon))
    adamw = next(value for value in optimizers.values if isinstance(value, torch.optim.AdamW))
    assert {id(parameter) for group in muon.param_groups for parameter in group["params"]} == {id(matrix)}
    assert {id(parameter) for group in adamw.param_groups for parameter in group["params"]} == {id(vector)}
    assert muon.defaults["adjust_lr_fn"] == "match_rms_adamw"

    before = (matrix.detach().clone(), vector.detach().clone())
    step = optimizers.optimize(matrix.square().mean() + vector.square().mean())
    assert not torch.equal(matrix, before[0])
    assert not torch.equal(vector, before[1])
    assert step["shared_preclip_gradient_norm"] > 0
    assert step["muon_preclip_gradient_norm"] > 0
    assert step["adamw_preclip_gradient_norm"] > 0
    telemetry = optimizers.epoch_telemetry(reset=True)
    assert telemetry["optimizer_steps"] == 1
    assert telemetry["peak_cpu_rss_bytes"] > 0
    assert optimizers.epoch_telemetry()["optimizer_steps"] == 0


def test_vocabulary_decoder_trains_only_from_selected_encoder_latents() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pair = b"ab"
        assets = pair_assets(root, torch.full((1, 8), 100.0))
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=1,
                vocabulary_epochs=1,
                reconstruction_epochs=0,
                reconstruction_samples=0,
            ),
            torch.device("cpu"),
        )
        fitter = vocabulary_module.VocabularyFitter(runtime)
        codec = runtime.build_codec(Profile("test", 8, 1, 1, 1, 4, 2, 16))
        decoder = codec.set_trainable_components(encoder=False, decoder=True)
        optimizers = runtime.optimizers(codec, decoder)
        encoder_before = {name: parameter.detach().clone() for name, parameter in codec.named_parameters() if not parameter.requires_grad}
        batch = build_span_batch([pair], max_span=codec.max_span, device=torch.device("cpu"))
        with torch.no_grad():
            expected_latent = codec.encode(batch.byte_values, batch.valid_mask)

        with patch.object(codec, "decode_logits", wraps=codec.decode_logits) as decode:
            fitter.train_decoder_epoch(
                vocabulary_module.DecoderEpochRequest(
                    codec=codec,
                    optimizers=optimizers,
                    generator=torch.Generator().manual_seed(17),
                    vocabulary_groups=build_vocabulary_groups(runtime.assets),
                )
            )

        actual_latent = decode.call_args.args[0]
        assert torch.equal(actual_latent, expected_latent)
        assert not torch.equal(actual_latent, assets.input_embeddings[[256]])
        for name, parameter in codec.named_parameters():
            if name in encoder_before:
                assert torch.equal(parameter, encoder_before[name])


def test_vocabulary_decoder_trains_after_failed_alignment() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = synthetic_assets(root)
        assets.input_embeddings = torch.cat((assets.input_embeddings, torch.full((1, 8), 100.0)))
        assets.vocabulary = ByteVocabulary(
            token_bytes=(*assets.vocabulary.token_bytes, b"ab"),
            ordinary_ids=tuple(range(257)),
            control_ids=(),
            byte_token_ids=assets.vocabulary.byte_token_ids,
            max_token_bytes=2,
        )
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=root / "checkpoints",
                profile=TEST_PROFILE,
                batch_size=256,
                vocabulary_epochs=1,
                reconstruction_epochs=0,
                reconstruction_samples=0,
            ),
            torch.device("cpu"),
        )
        fitter = vocabulary_module.VocabularyFitter(runtime)
        codec = runtime.build_codec(TEST_PROFILE)

        with patch.object(
            vocabulary_module.VocabularyFitter,
            "train_decoder_epoch",
            autospec=True,
            side_effect=vocabulary_module.VocabularyFitter.train_decoder_epoch,
        ) as train_decoder:
            fitter.fit(
                codec,
                TEST_PROFILE,
                torch.Generator().manual_seed(17),
            )

        train_decoder.assert_called_once()


def test_vocabulary_batches_group_compatible_widths_without_dropping_rows() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assets = synthetic_model_assets()
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=Path(directory) / "checkpoints",
                batch_size=17,
                vocabulary_epochs=1,
                reconstruction_epochs=0,
                reconstruction_samples=0,
            ),
            torch.device("cpu"),
        )
        groups = build_vocabulary_groups(runtime.assets)
        batches = build_vocabulary_batches(
            groups,
            runtime.options.batch_size,
            torch.Generator().manual_seed(17),
        )

        batch_ids = [groups[batch.bucket].token_ids.index_select(0, batch.rows)[: batch.logical_rows] for batch in batches]
        flattened = [token_id for batch in batch_ids for token_id in batch.tolist()]
        expected = [token_id for token_id in assets.vocabulary.compatibility_ids if len(assets.vocabulary.bytes_for(token_id)) > 1]
        assert sorted(flattened) == expected
        assert all(len(batch.rows) == runtime.options.batch_size for batch in batches)
        max_span = max(runtime.assets.vocabulary.max_token_bytes, 32)
        assert all(
            len(
                {
                    span_bucket_width(
                        len(runtime.assets.vocabulary.bytes_for(token_id)),
                        max_span=max_span,
                    )
                    for token_id in batch.tolist()
                }
            )
            == 1
            for batch in batch_ids
        )


def test_vocabulary_groups_exclude_duplicate_alias_rows() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assets = synthetic_model_assets()
        alias_id = len(assets.vocabulary.token_bytes)
        payload = assets.vocabulary.bytes_for(256)
        assets.input_embeddings = torch.cat((assets.input_embeddings, torch.randn((1, 16))))
        assets.vocabulary = ByteVocabulary(
            token_bytes=(*assets.vocabulary.token_bytes, payload),
            ordinary_ids=(*assets.vocabulary.ordinary_ids, alias_id),
            control_ids=(),
            byte_token_ids=assets.vocabulary.byte_token_ids,
            max_token_bytes=assets.vocabulary.max_token_bytes,
            compatibility_ids=assets.vocabulary.compatibility_ids,
        )
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(
                output_dir=Path(directory) / "checkpoints",
                batch_size=17,
                vocabulary_epochs=1,
                reconstruction_epochs=0,
                reconstruction_samples=0,
            ),
            torch.device("cpu"),
        )
        grouped = {token_id for group in build_vocabulary_groups(runtime.assets) for token_id in group.token_ids.tolist()}

        assert 256 in grouped
        assert alias_id not in grouped


def test_source_token_longer_than_dynamic_limit_uses_codec_width_bucket() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assets = synthetic_model_assets()
        long_token_id = len(assets.vocabulary.token_bytes)
        long_payload = bytes(range(40))
        assets = replace(
            assets,
            input_embeddings=torch.cat((assets.input_embeddings, torch.randn((1, assets.input_embeddings.shape[1])))),
            vocabulary=ByteVocabulary(
                token_bytes=(*assets.vocabulary.token_bytes, long_payload),
                ordinary_ids=(*assets.vocabulary.ordinary_ids, long_token_id),
                control_ids=assets.vocabulary.control_ids,
                byte_token_ids=assets.vocabulary.byte_token_ids,
                max_token_bytes=len(long_payload),
            ),
        )
        runtime = selection_module.TrainingRuntime(
            assets,
            TrainingOptions(output_dir=Path(directory) / "checkpoints"),
            torch.device("cpu"),
        )

        codec = runtime.build_codec(TEST_PROFILE)
        groups = build_vocabulary_groups(assets, (long_token_id,))

    assert codec.max_span == len(long_payload)
    assert len(groups) == 1
    assert groups[0].byte_values.shape == (1, len(long_payload))
    assert groups[0].valid_mask.all()


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_vocabulary_parameter_groups_are_disjoint_and_complete,
            test_tokenizer_optimizers_partition_parameters_between_muon_and_adamw,
            test_vocabulary_decoder_trains_only_from_selected_encoder_latents,
            test_vocabulary_decoder_trains_after_failed_alignment,
            test_vocabulary_batches_group_compatible_widths_without_dropping_rows,
            test_vocabulary_groups_exclude_duplicate_alias_rows,
            test_source_token_longer_than_dynamic_limit_uses_codec_width_bucket,
        )
    )
