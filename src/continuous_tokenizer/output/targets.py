from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, final

import torch
from torch import Tensor, nn

from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.output.events import ByteSpanEvent, ControlEvent, OutputEvent
from continuous_tokenizer.runtime.tensors import cache_tensor_bytes
from continuous_tokenizer.runtime.timing import timed_call

type TrajectoryTermination = Literal["stop_control", "max_native_tokens", "max_bytes"]


@final
class OutputPackingInfeasibleError(ValueError):
    def __init__(self, token_id: int, payload_bytes: int, max_span: int) -> None:
        super().__init__(
            f"ordinary native token {token_id} has {payload_bytes} bytes, exceeding max_span {max_span}",
        )
        self.token_id = token_id
        self.payload_bytes = payload_bytes
        self.max_span = max_span


def output_events(
    token_ids: tuple[int, ...],
    vocabulary: ByteVocabulary,
    *,
    start: int,
    max_span: int,
) -> tuple[OutputEvent, ...]:
    if not 0 <= start <= len(token_ids):
        raise ValueError("output trajectory start is outside the token sequence")
    return pack_native_tokens(token_ids[start:], vocabulary, max_span=max_span)[0]


def pack_native_tokens(
    token_ids: tuple[int, ...],
    vocabulary: ByteVocabulary,
    *,
    max_span: int,
) -> tuple[tuple[OutputEvent, ...], tuple[tuple[int, ...], ...]]:
    if max_span < 1:
        raise ValueError("output trajectory span limit must be positive")
    events: list[OutputEvent] = []
    targets: list[tuple[int, ...]] = []
    payload = bytearray()
    payload_targets: list[int] = []
    for token_id in token_ids:
        value = vocabulary.payload_for(token_id)
        if value is not None and len(value) > max_span:
            raise OutputPackingInfeasibleError(token_id, len(value), max_span)
        if payload and (value is None or len(payload) + len(value) > max_span):
            events.append(ByteSpanEvent(bytes(payload)))
            targets.append(tuple(payload_targets))
            payload.clear()
            payload_targets.clear()
        if value is None:
            events.append(ControlEvent(token_id))
            targets.append((token_id,))
        else:
            payload.extend(value)
            payload_targets.append(token_id)
    if payload:
        events.append(ByteSpanEvent(bytes(payload)))
        targets.append(tuple(payload_targets))
    return tuple(events), tuple(targets)


@dataclass(frozen=True, slots=True)
class NativeHeadGeneration:
    prompt_token_ids: tuple[int, ...]
    native_token_ids: tuple[int, ...]
    stop_controls: tuple[int, ...]
    termination_reason: TrajectoryTermination
    attempted_native_tokens: int
    model_calls: int
    final_cache_bytes: int
    peak_cache_bytes: int
    runtime_telemetry: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.prompt_token_ids or not self.native_token_ids:
            raise ValueError("native-head generation requires a prompt and generated tokens")
        if self.termination_reason == "stop_control" and not self.stop_controls:
            raise ValueError("stop-control termination requires an observed stop control")


@final
@dataclass(frozen=True, slots=True)
class NativeHeadTrajectory(NativeHeadGeneration):
    hidden: Tensor

    def __post_init__(self) -> None:
        NativeHeadGeneration.__post_init__(self)
        if self.hidden.ndim != 2 or self.hidden.shape[0] != len(self.native_token_ids):
            raise ValueError("native-head trajectory hidden states must align with native tokens")


@final
@dataclass(frozen=True, slots=True)
class PackedOutputTrajectory:
    events: tuple[OutputEvent, ...]
    target_native_token_ids: tuple[tuple[int, ...], ...]
    hidden: Tensor

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("packed output trajectories require at least one event")
        if len(self.target_native_token_ids) != len(self.events):
            raise ValueError("packed native-token targets must align with events")
        if self.hidden.ndim != 2 or self.hidden.shape[0] != len(self.events):
            raise ValueError("packed hidden states must align with events")
        if any(not token_ids for token_ids in self.target_native_token_ids):
            raise ValueError("packed output events must represent native tokens")


@final
@dataclass(frozen=True, slots=True)
class NativeTrajectoryOptions:
    stop_control_ids: frozenset[int]
    max_native_tokens: int
    max_bytes: int
    collect_runtime_telemetry: bool = False

    def __post_init__(self) -> None:
        if min(self.max_native_tokens, self.max_bytes) < 1:
            raise ValueError("native-head trajectory limits must be positive")


def pack_native_trajectory(
    trajectory: NativeHeadTrajectory,
    vocabulary: ByteVocabulary,
    *,
    max_span: int,
) -> PackedOutputTrajectory:
    events, targets = pack_native_tokens(
        trajectory.native_token_ids,
        vocabulary,
        max_span=max_span,
    )
    first_indices: list[int] = []
    native_index = 0
    for target in targets:
        first_indices.append(native_index)
        native_index += len(target)
    positions = torch.tensor(first_indices, dtype=torch.long, device=trajectory.hidden.device)
    return PackedOutputTrajectory(
        events=events,
        target_native_token_ids=targets,
        hidden=trajectory.hidden.index_select(0, positions),
    )


def native_output_head(backbone: FrozenBackbone) -> nn.Module:
    getter = getattr(backbone.source_model, "get_output_embeddings", None)
    head = getter() if callable(getter) else getattr(backbone.source_model, "lm_head", None)
    if not isinstance(head, nn.Module):
        raise ValueError("causal language model has no native output head")
    return head


@torch.inference_mode()
def _native_head_run(  # noqa: C901, PLR0915
    backbone: FrozenBackbone,
    vocabulary: ByteVocabulary,
    prompt_token_ids: tuple[int, ...],
    options: NativeTrajectoryOptions,
    *,
    capture_hidden: bool,
) -> NativeHeadTrajectory | NativeHeadGeneration:
    if not prompt_token_ids:
        raise ValueError("output trajectories require a non-empty native prompt")

    def device_call[Result](callable_: Any) -> tuple[Result, float]:
        if options.collect_runtime_telemetry:
            return timed_call(callable_, backbone.device)
        return callable_(), 0.0

    preparation_seconds = 0.0
    backbone_seconds = 0.0
    output_decode_seconds = 0.0
    feedback_seconds = 0.0
    cache_accounting_seconds = 0.0
    synchronization_count = 0
    host_to_device_bytes = 0
    device_to_host_bytes = 0
    graph_signatures: Counter[str] = Counter()
    synchronizations_per_device_call = 2 if options.collect_runtime_telemetry and backbone.device.type in {"cuda", "mps"} else 0
    preparation_started = time.perf_counter()
    prompt = torch.tensor([prompt_token_ids], dtype=torch.long, device=backbone.device)
    preparation_seconds += time.perf_counter() - preparation_started
    if backbone.device.type != "cpu":
        host_to_device_bytes += prompt.numel() * prompt.element_size()
    graph_signatures[f"backbone_prefill:{len(prompt_token_ids)}"] += 1
    outputs, seconds = device_call(
        lambda: backbone.forward(input_ids=prompt, use_cache=True),
    )
    backbone_seconds += seconds
    synchronization_count += synchronizations_per_device_call
    hidden = outputs.last_hidden_state[:, -1]
    past: Any = outputs.past_key_values
    cache_started = time.perf_counter()
    final_cache_bytes = cache_tensor_bytes(past)
    cache_accounting_seconds += time.perf_counter() - cache_started
    peak_cache_bytes = final_cache_bytes
    model_calls = 1
    head = native_output_head(backbone)
    native_ids: list[int] = []
    hidden_rows: list[Tensor] = []
    stop_controls: list[int] = []
    emitted_bytes = 0
    native_position = len(prompt_token_ids)
    termination_reason: TrajectoryTermination = "max_native_tokens"
    attempted_native_tokens = 0

    for _ in range(options.max_native_tokens):
        attempted_native_tokens += 1
        token_id, seconds = device_call(
            lambda current_hidden=hidden: int(head(current_hidden).argmax(dim=-1).item()),
        )
        output_decode_seconds += seconds
        synchronization_count += synchronizations_per_device_call
        device_to_host_bytes += 8
        graph_signatures[f"native_head:1x{hidden.shape[-1]}"] += 1
        feedback_started = time.perf_counter()
        payload = vocabulary.payload_for(token_id)
        payload_bytes = 0 if payload is None else len(payload)
        feedback_seconds += time.perf_counter() - feedback_started
        if emitted_bytes + payload_bytes > options.max_bytes:
            termination_reason = "max_bytes"
            break
        native_ids.append(token_id)
        if capture_hidden:
            transfer_started = time.perf_counter()
            hidden_rows.append(hidden[0].detach().to("cpu"))
            preparation_seconds += time.perf_counter() - transfer_started
            if backbone.device.type != "cpu":
                device_to_host_bytes += hidden[0].numel() * hidden[0].element_size()
        if token_id in options.stop_control_ids:
            stop_controls.append(token_id)
            termination_reason = "stop_control"
            break
        preparation_started = time.perf_counter()
        ids = torch.tensor([[token_id]], dtype=torch.long, device=backbone.device)
        positions = torch.tensor([[native_position]], dtype=torch.long, device=backbone.device)
        preparation_seconds += time.perf_counter() - preparation_started
        if backbone.device.type != "cpu":
            host_to_device_bytes += ids.numel() * ids.element_size() + positions.numel() * positions.element_size()
        graph_signatures["backbone_feedback:1"] += 1
        outputs, seconds = device_call(
            lambda input_ids=ids, position_ids=positions, current_past=past: backbone.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=current_past,
                use_cache=True,
            ),
        )
        backbone_seconds += seconds
        synchronization_count += synchronizations_per_device_call
        past = outputs.past_key_values
        model_calls += 1
        cache_started = time.perf_counter()
        final_cache_bytes = cache_tensor_bytes(past)
        cache_accounting_seconds += time.perf_counter() - cache_started
        peak_cache_bytes = max(peak_cache_bytes, final_cache_bytes)
        hidden = outputs.last_hidden_state[:, -1]
        native_position += 1
        emitted_bytes += payload_bytes
    if not native_ids:
        raise ValueError("native-head trajectory produced no target tokens")
    common = {
        "prompt_token_ids": prompt_token_ids,
        "native_token_ids": tuple(native_ids),
        "stop_controls": tuple(stop_controls),
        "termination_reason": termination_reason,
        "attempted_native_tokens": attempted_native_tokens,
        "model_calls": model_calls,
        "final_cache_bytes": final_cache_bytes,
        "peak_cache_bytes": peak_cache_bytes,
        "runtime_telemetry": {
            "schema_version": 1,
            "preparation_seconds": preparation_seconds,
            "backbone_seconds": backbone_seconds,
            "output_decode_seconds": output_decode_seconds,
            "feedback_seconds": feedback_seconds,
            "cache_accounting_seconds": cache_accounting_seconds,
            "host_to_device_bytes": host_to_device_bytes,
            "device_to_host_bytes": device_to_host_bytes,
            "synchronization_count": synchronization_count,
            "backbone_calls": model_calls,
            "output_decode_calls": attempted_native_tokens,
            "feedback_calls": attempted_native_tokens,
            "cache_snapshots": model_calls,
            "graph_signature_counts": dict(sorted(graph_signatures.items())),
        },
    }
    if not capture_hidden:
        return NativeHeadGeneration(**common)
    return NativeHeadTrajectory(
        **common,
        hidden=torch.stack(hidden_rows),
    )


def native_head_trajectory(
    backbone: FrozenBackbone,
    vocabulary: ByteVocabulary,
    prompt_token_ids: tuple[int, ...],
    options: NativeTrajectoryOptions,
) -> NativeHeadTrajectory:
    result = _native_head_run(
        backbone,
        vocabulary,
        prompt_token_ids,
        options,
        capture_hidden=True,
    )
    if not isinstance(result, NativeHeadTrajectory):
        raise RuntimeError("native trajectory did not capture hidden states")
    return result


def native_head_generation(
    backbone: FrozenBackbone,
    vocabulary: ByteVocabulary,
    prompt_token_ids: tuple[int, ...],
    options: NativeTrajectoryOptions,
) -> NativeHeadGeneration:
    result = _native_head_run(
        backbone,
        vocabulary,
        prompt_token_ids,
        options,
        capture_hidden=False,
    )
    if not isinstance(result, NativeHeadGeneration):
        raise RuntimeError("native generation unexpectedly captured hidden states")
    return result


def bounded_output_bytes(
    events: tuple[OutputEvent, ...],
    *,
    stop_control_ids: frozenset[int],
    max_macro_steps: int,
    max_bytes: int,
) -> bytes:
    data = bytearray()
    for event in events[:max_macro_steps]:
        if isinstance(event, ControlEvent):
            if event.token_id in stop_control_ids:
                break
            continue
        if len(data) + len(event.data) > max_bytes:
            break
        data.extend(event.data)
        if len(data) >= max_bytes:
            break
    return bytes(data)
