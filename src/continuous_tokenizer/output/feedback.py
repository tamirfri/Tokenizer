from __future__ import annotations

from dataclasses import dataclass, field
from typing import final

from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.output.events import ByteSpanEvent, ControlEvent, OutputEvent


@dataclass(slots=True)
class _TrieNode:
    children: dict[int, _TrieNode] = field(default_factory=dict)
    token_id: int | None = None


@final
@dataclass(frozen=True, slots=True)
class NativeFeedback:
    token_ids: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("native feedback must contain at least one token")


@final
class NativeByteSegmenter:
    """Deterministically segment arbitrary bytes using longest native-token matches."""

    def __init__(self, vocabulary: ByteVocabulary) -> None:
        self.vocabulary = vocabulary
        self._root = _TrieNode()
        for token_id in vocabulary.ordinary_ids:
            value = vocabulary.bytes_for(token_id)
            node = self._root
            for byte in value:
                child = node.children.get(byte)
                if child is None:
                    child = _TrieNode()
                    node.children[byte] = child
                node = child
            node.token_id = token_id if node.token_id is None else min(node.token_id, token_id)

    def segment(self, data: bytes) -> tuple[int, ...]:
        if not data:
            raise ValueError("feedback bytes must not be empty")
        token_ids: list[int] = []
        offset = 0
        while offset < len(data):
            node = self._root
            cursor = offset
            selected_id: int | None = None
            selected_end = offset
            while cursor < len(data):
                child = node.children.get(data[cursor])
                if child is None:
                    break
                node = child
                cursor += 1
                if node.token_id is not None:
                    selected_id = node.token_id
                    selected_end = cursor
            if selected_id is None:
                selected_id = self.vocabulary.byte_token_ids[data[offset]]
                selected_end = offset + 1
            token_ids.append(selected_id)
            offset = selected_end
        reconstructed = b"".join(self.vocabulary.bytes_for(token_id) for token_id in token_ids)
        if reconstructed != data:
            raise RuntimeError("native feedback segmentation changed emitted bytes")
        return tuple(token_ids)

    def feedback(self, event: OutputEvent) -> NativeFeedback:
        if isinstance(event, ControlEvent):
            if event.token_id not in self.vocabulary.control_ids:
                raise ValueError("output control event is not a structural native token")
            return NativeFeedback((event.token_id,), b"")
        if not isinstance(event, ByteSpanEvent):
            raise TypeError("output event must be a control or byte-span event")
        return NativeFeedback(self.segment(event.data), event.data)
