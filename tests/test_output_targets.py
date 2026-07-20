from __future__ import annotations

import unittest
from random import Random
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import torch
from torch import Tensor, nn

from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.synthetic import SyntheticCausalLM, synthetic_model_assets
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.campaigns.output import _rollout_agreement
from continuous_tokenizer.output.events import ByteSpanEvent, ControlEvent
from continuous_tokenizer.output.feedback import NativeByteSegmenter
from continuous_tokenizer.output.generation import (
    OUTPUT_STOP_CONTROL_POLICY,
    output_stop_control_ids,
    output_stop_control_metadata,
)
from continuous_tokenizer.output.targets import (
    NativeTrajectoryOptions,
    OutputPackingInfeasibleError,
    bounded_output_bytes,
    native_head_trajectory,
    output_events,
    pack_native_tokens,
    pack_native_trajectory,
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
    def test_output_stop_policy_uses_only_structural_in_table_eos(self) -> None:
        token_bytes = (*tuple(bytes([value]) for value in range(256)), None, None)
        vocabulary = ByteVocabulary(
            token_bytes=token_bytes,
            ordinary_ids=tuple(range(256)),
            control_ids=(256, 257),
            byte_token_ids=tuple(range(256)),
            max_token_bytes=1,
            out_of_table_control_ids=(258,),
        )
        tokenizer = SimpleNamespace(eos_token_id=257)

        self.assertEqual(output_stop_control_ids(tokenizer, vocabulary), frozenset({257}))
        self.assertEqual(
            output_stop_control_metadata(tokenizer, vocabulary),
            {
                "policy": OUTPUT_STOP_CONTROL_POLICY,
                "token_ids": [257],
            },
        )
        tokenizer.eos_token_id = 1
        self.assertEqual(output_stop_control_ids(tokenizer, vocabulary), frozenset())
        tokenizer.eos_token_id = 258
        self.assertEqual(output_stop_control_ids(tokenizer, vocabulary), frozenset())

    def test_feedback_is_longest_match_with_atomic_binary_fallback(self) -> None:
        vocabulary = synthetic_model_assets().vocabulary
        segmenter = NativeByteSegmenter(vocabulary)
        pair_id = next(token_id for token_id in vocabulary.ordinary_ids if len(vocabulary.bytes_for(token_id)) == 2)
        pair = vocabulary.bytes_for(pair_id)

        self.assertEqual(segmenter.segment(pair), (pair_id,))
        binary = b"\x00\xff"
        self.assertEqual(
            b"".join(vocabulary.bytes_for(token_id) for token_id in segmenter.segment(binary)),
            binary,
        )

    def test_feedback_trie_matches_reference_and_alias_tie_break(self) -> None:
        token_bytes = (
            *tuple(bytes([value]) for value in range(256)),
            b"ab",
            b"ab",
            b"abc",
        )
        vocabulary = ByteVocabulary(
            token_bytes,
            tuple(range(len(token_bytes))),
            (),
            tuple(range(256)),
            3,
        )
        segmenter = NativeByteSegmenter(vocabulary)

        def reference(data: bytes) -> tuple[int, ...]:
            ordered = sorted(
                ((vocabulary.bytes_for(token_id), token_id) for token_id in vocabulary.ordinary_ids),
                key=lambda item: (-len(item[0]), item[1]),
            )
            result: list[int] = []
            offset = 0
            while offset < len(data):
                payload, token_id = next(
                    (candidate for candidate in ordered if data.startswith(candidate[0], offset)),
                    (bytes([data[offset]]), vocabulary.byte_token_ids[data[offset]]),
                )
                result.append(token_id)
                offset += len(payload)
            return tuple(result)

        self.assertEqual(segmenter.segment(b"abcab\xff"), (258, 256, 255))
        random = Random(41)
        for length in range(1, 65):
            data = random.randbytes(length)
            self.assertEqual(segmenter.segment(data), reference(data))

    def test_packing_greedily_merges_complete_native_tokens(self) -> None:
        token_bytes = (*tuple(bytes([value]) for value in range(256)), b"ab", b"cd")
        vocabulary = ByteVocabulary(
            token_bytes,
            tuple(range(258)),
            (),
            tuple(range(256)),
            2,
        )
        events, targets = pack_native_tokens(
            (256, 257, ord("e")),
            vocabulary,
            max_span=4,
        )

        self.assertEqual(events, (ByteSpanEvent(b"abcd"), ByteSpanEvent(b"e")))
        self.assertEqual(targets, ((256, 257), (ord("e"),)))

    def test_rollout_agreement_penalizes_a_short_correct_prefix(self) -> None:
        events = (ByteSpanEvent(b"ab"), ByteSpanEvent(b"cd"))
        expected = bounded_output_bytes(
            events,
            stop_control_ids=frozenset(),
            max_macro_steps=2,
            max_bytes=4,
        )

        self.assertEqual(expected, b"abcd")
        self.assertEqual(_rollout_agreement(b"a", expected), 0.25)

    def test_overlong_native_token_makes_packing_infeasible(self) -> None:
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

        with self.assertRaises(OutputPackingInfeasibleError):
            output_events((256,), vocabulary, start=0, max_span=2)

    def test_native_oracle_is_invariant_to_packing_and_feeds_selected_token(self) -> None:
        token_bytes = (*tuple(bytes([value]) for value in range(256)), b"ab")
        vocabulary = ByteVocabulary(
            token_bytes,
            tuple(range(257)),
            (),
            tuple(range(256)),
            2,
        )
        model = SyntheticCausalLM(torch.randn(257, 8))
        cast(Any, model).lm_head = _FixedNativeHead(257, 256)
        backbone = FrozenBackbone(model)

        with mock.patch.object(backbone, "forward", wraps=backbone.forward) as forward:
            trajectory = native_head_trajectory(
                backbone,
                vocabulary,
                (0,),
                NativeTrajectoryOptions(
                    stop_control_ids=frozenset(),
                    max_native_tokens=2,
                    max_bytes=4,
                ),
            )
        packed_two = pack_native_trajectory(trajectory, vocabulary, max_span=2)
        packed_four = pack_native_trajectory(trajectory, vocabulary, max_span=4)

        self.assertEqual(trajectory.native_token_ids, (256, 256))
        self.assertEqual(
            tuple(forward.call_args_list[1].kwargs["input_ids"][0].tolist()),
            (256,),
        )
        self.assertEqual(packed_two.target_native_token_ids, ((256,), (256,)))
        self.assertEqual(packed_four.target_native_token_ids, ((256, 256),))

    def test_native_head_stop_control_is_appended_and_measured(self) -> None:
        token_bytes = (*tuple(bytes([value]) for value in range(256)), None)
        vocabulary = ByteVocabulary(
            token_bytes,
            tuple(range(256)),
            (256,),
            tuple(range(256)),
            1,
        )
        model = SyntheticCausalLM(torch.randn(257, 8))
        cast(Any, model).lm_head = _FixedNativeHead(257, 256)

        trajectory = native_head_trajectory(
            FrozenBackbone(model),
            vocabulary,
            (0,),
            NativeTrajectoryOptions(
                stop_control_ids=frozenset({256}),
                max_native_tokens=4,
                max_bytes=4,
            ),
        )

        packed = pack_native_trajectory(trajectory, vocabulary, max_span=1)
        self.assertEqual(packed.events, (ControlEvent(256),))
        self.assertEqual(trajectory.stop_controls, (256,))
        self.assertEqual(trajectory.termination_reason, "stop_control")
