from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from continuous_tokenizer.codec.constants import CODEC_EOS


@final
@dataclass(frozen=True, slots=True)
class ByteSpanEvent:
    data: bytes

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("output byte spans must not be empty")


@final
@dataclass(frozen=True, slots=True)
class ControlEvent:
    token_id: int

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("control token IDs must be non-negative")


type OutputEvent = ByteSpanEvent | ControlEvent


def output_event_from_prediction(
    selector: int,
    generated: Sequence[int],
    control_ids: tuple[int, ...],
    *,
    max_span: int,
) -> OutputEvent | None:
    if selector:
        return ControlEvent(control_ids[selector - 1])
    try:
        length = generated.index(CODEC_EOS)
    except ValueError:
        return None
    if not 1 <= length <= max_span:
        return None
    return ByteSpanEvent(bytes(generated[:length]))
