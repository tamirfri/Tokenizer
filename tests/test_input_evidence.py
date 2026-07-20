from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from input_model_fixtures import make_adapter

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.campaigns.input import _behavior_gates
from continuous_tokenizer.codec.checkpoints import load_checkpoint, save_checkpoint
from continuous_tokenizer.codec.constants import CODEC_EOS
from continuous_tokenizer.contracts.input import InputGateSpec
from continuous_tokenizer.data.corpus import sample_content_windows
from continuous_tokenizer.input.adapter import InputEmbeddingAdapter
from continuous_tokenizer.input.evidence import (
    alias_group_evidence,
    input_source_identity,
    segmentation_evidence,
    verify_checkpoint_source,
)
from continuous_tokenizer.input.segmentation import EncodedSpan


class ScriptedDecoder:
    def __init__(self, generated: tuple[int, ...]) -> None:
        self.generated = generated

    def decode_logits(
        self,
        latent: torch.Tensor,
        output_positions: int | None = None,
    ) -> torch.Tensor:
        if output_positions is None:
            raise ValueError("the evidence test requires explicit decoder positions")
        logits = torch.full(
            (latent.shape[0], output_positions, CODEC_EOS + 1),
            -1_000.0,
        )
        for position, value in enumerate(self.generated[:output_positions]):
            logits[:, position, value] = 1_000.0
        return logits


def _assets() -> tuple[ModelAssets, InputEmbeddingAdapter]:
    adapter = make_adapter()
    assets = ModelAssets(
        model_id="synthetic/model",
        revision="synthetic-revision",
        tokenizer=SimpleNamespace(),
        config={},
        embedding_tensor_name="embedding.weight",
        embedding_shard=Path("synthetic.safetensors"),
        vocabulary=adapter.vocabulary,
        input_embeddings=torch.cat(
            (
                adapter.codec.byte_embeddings,
                adapter.control_embeddings,
                torch.randn(1, 8),
            )
        ),
    )
    return assets, adapter


def test_independent_decoder_bytes_and_exact_eos_are_empirical_evidence() -> None:
    span = EncodedSpan(b"ab", torch.tensor([1.0, 2.0]), atomic=False)

    wrong_bytes = segmentation_evidence(
        ScriptedDecoder((ord("a"), ord("x"), CODEC_EOS)),
        (span,),
        b"ab",
        source_dtype=torch.float32,
    )
    wrong_eos = segmentation_evidence(
        ScriptedDecoder((ord("a"), CODEC_EOS, ord("b"))),
        (span,),
        b"ab",
        source_dtype=torch.float32,
    )

    assert wrong_bytes.source_bookkeeping_round_trip
    assert not wrong_bytes.empirical_round_trip
    assert wrong_bytes.rows[0].eos_exact
    assert wrong_bytes.rows[0].independent_decoded_hex == b"ax".hex()
    assert not wrong_eos.empirical_round_trip
    assert not wrong_eos.rows[0].eos_exact


def test_adapter_rejects_embedding_and_vocabulary_source_mismatches() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assets, adapter = _assets()
        checkpoint = Path(directory) / "codec.pt"
        save_checkpoint(
            checkpoint,
            adapter.codec,
            {
                "model_revision": assets.revision,
                "source_identity": input_source_identity(assets),
            },
            control_ids=adapter.control_ids,
            control_embeddings=adapter.control_embeddings,
        )
        changed_embeddings = assets.input_embeddings.clone()
        changed_embeddings[-1, 0] += 1

        with unittest.TestCase().assertRaisesRegex(ValueError, "source identity"):
            InputEmbeddingAdapter.from_checkpoint(
                replace(assets, input_embeddings=changed_embeddings),
                checkpoint,
                device=torch.device("cpu"),
            )

        changed_payloads = list(assets.vocabulary.token_bytes)
        changed_payloads[-1] = b"ac"
        changed_vocabulary = replace(
            assets.vocabulary,
            token_bytes=tuple(changed_payloads),
        )
        with unittest.TestCase().assertRaisesRegex(ValueError, "source identity"):
            InputEmbeddingAdapter.from_checkpoint(
                replace(assets, vocabulary=changed_vocabulary),
                checkpoint,
                device=torch.device("cpu"),
            )


def test_reload_preserves_independent_decoder_and_rejects_source_row_mismatch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        assets, adapter = _assets()
        checkpoint = Path(directory) / "codec.pt"
        save_checkpoint(
            checkpoint,
            adapter.codec,
            {
                "model_revision": assets.revision,
                "source_identity": input_source_identity(assets),
            },
            control_ids=adapter.control_ids,
            control_embeddings=adapter.control_embeddings,
        )
        latent = torch.randn(1, adapter.codec.config.embedding_dim)
        expected_logits = adapter.codec.decode_logits(latent, 3).detach()
        loaded = load_checkpoint(checkpoint)
        with torch.no_grad():
            adapter.codec.byte_head.bias.add_(1)

        reloaded_logits = loaded.codec.decode_logits(latent, 3).detach()

        assert torch.equal(reloaded_logits, expected_logits)
        assert torch.equal(reloaded_logits.argmax(dim=-1), expected_logits.argmax(dim=-1))
        mismatched_rows = loaded.codec.byte_embeddings.clone()
        mismatched_rows[0, 0] += 1
        with unittest.TestCase().assertRaisesRegex(ValueError, "atomic byte rows"):
            verify_checkpoint_source(
                assets,
                loaded.metadata,
                mismatched_rows,
                loaded.controls.ids,
                loaded.controls.embeddings,
            )


def test_alias_evidence_exposes_noncanonical_source_row_disagreement() -> None:
    assets, _ = _assets()
    vocabulary = assets.vocabulary
    alias_id = len(vocabulary.token_bytes)
    assets = replace(
        assets,
        vocabulary=ByteVocabulary(
            token_bytes=(*vocabulary.token_bytes, b"ab"),
            ordinary_ids=(*vocabulary.ordinary_ids, alias_id),
            control_ids=vocabulary.control_ids,
            byte_token_ids=vocabulary.byte_token_ids,
            max_token_bytes=vocabulary.max_token_bytes,
            compatibility_ids=vocabulary.compatibility_ids,
        ),
        input_embeddings=torch.cat(
            (assets.input_embeddings, assets.input_embeddings[-1:] + 1),
        ),
    )

    groups = alias_group_evidence(assets)

    assert len(groups) == 1
    group = groups[0]
    assert group["payload_hex"] == b"ab".hex()
    assert group["canonical_id"] == 257
    assert group["alias_ids"] == [258]
    assert group["applicability"] == "inapplicable_noncanonical_source_row_disagreement"
    row = group["source_rows"][0]
    assert row["alias_id"] == 258
    assert not row["source_row_equal"]
    assert abs(row["source_row_l2_distance"] - 8**0.5) < 1e-6
    assert abs(row["source_row_maximum_absolute_distance"] - 1.0) < 1e-6


def test_content_hashed_document_windows_are_order_independent() -> None:
    documents = [
        b"alpha beta gamma",
        b"delta epsilon zeta",
        b"eta theta iota",
    ]

    forward = sample_content_windows(
        documents,
        maximum_bytes=24,
        window_bytes=8,
        seed=17,
    )
    reversed_order = sample_content_windows(
        list(reversed(documents)),
        maximum_bytes=24,
        window_bytes=8,
        seed=17,
    )

    assert forward == reversed_order
    assert sum(len(window.payload) for window in forward) == 24
    assert all(window.sha256 for window in forward)


def test_segmented_behavior_gates_enforce_all_preregistered_thresholds() -> None:
    gates = InputGateSpec()
    metrics = {
        "teacher_forced": {
            "segmented": {
                "mean_kl": gates.maximum_segmented_mean_kl,
                "teacher_nll": 1.0,
                "student_nll": 1.0 + gates.maximum_segmented_nll_delta / 2,
                "top1_agreement": gates.minimum_segmented_top1_agreement,
            }
        },
        "generation": {
            "samples": 1,
            "segmented_mean_byte_similarity": gates.minimum_segmented_generation_byte_similarity,
        },
    }

    assert all(_behavior_gates(metrics, gates).values())
    metrics["generation"]["segmented_mean_byte_similarity"] = 0.0
    assert not _behavior_gates(metrics, gates)["minimum_segmented_generation_byte_similarity"]
    metrics["generation"]["samples"] = 0
    assert not _behavior_gates(metrics, gates)["minimum_segmented_generation_byte_similarity"]


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_independent_decoder_bytes_and_exact_eos_are_empirical_evidence,
            test_adapter_rejects_embedding_and_vocabulary_source_mismatches,
            test_reload_preserves_independent_decoder_and_rejects_source_row_mismatch,
            test_alias_evidence_exposes_noncanonical_source_row_disagreement,
            test_content_hashed_document_windows_are_order_independent,
            test_segmented_behavior_gates_enforce_all_preregistered_thresholds,
        )
    )
