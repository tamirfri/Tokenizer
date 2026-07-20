from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final, Literal, final

import torch
from torch import Tensor

from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.compute import (
    backbone_forward_flops,
    output_decode_flops,
)
from continuous_tokenizer.codec.output import OutputByteCodec
from continuous_tokenizer.output.events import (
    ByteSpanEvent,
    ControlEvent,
    OutputEvent,
    output_event_from_prediction,
)
from continuous_tokenizer.output.feedback import NativeByteSegmenter
from continuous_tokenizer.output.targets import native_output_head
from continuous_tokenizer.runtime.tensors import cache_tensor_bytes
from continuous_tokenizer.runtime.timing import timed_call

OUTPUT_STOP_CONTROL_POLICY: Final = "tokenizer_eos_structural_control"
type OutputTerminationReason = Literal[
    "stop_control",
    "invalid_event",
    "max_bytes_truncated",
    "max_bytes",
    "max_macro_steps",
]


def _backbone_flops(
    backbone: FrozenBackbone,
    query_positions: int,
    context_positions: int | None = None,
) -> int:
    config = getattr(backbone.source_model, "config", None)
    if config is None:
        return 0
    return backbone_forward_flops(config, query_positions, context_positions)


def output_stop_control_ids(tokenizer: Any, vocabulary: ByteVocabulary) -> frozenset[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    candidates = (eos,) if isinstance(eos, int) else tuple(eos or ())
    structural_ids = frozenset(vocabulary.control_ids)
    return frozenset(int(token_id) for token_id in candidates if token_id in structural_ids)


def output_stop_control_metadata(
    tokenizer: Any,
    vocabulary: ByteVocabulary,
) -> dict[str, str | list[int]]:
    return {
        "policy": OUTPUT_STOP_CONTROL_POLICY,
        "token_ids": sorted(output_stop_control_ids(tokenizer, vocabulary)),
    }


@final
@dataclass(frozen=True, slots=True)
class OutputGenerationResult:
    events: tuple[OutputEvent, ...]
    data: bytes
    macro_steps: int
    native_tokens_represented: int
    invalid_events: int
    termination_reason: OutputTerminationReason
    model_calls: int
    final_cache_bytes: int
    peak_cache_bytes: int
    analytical_backbone_flops: int
    analytical_codec_decode_flops: int
    native_head_invocations: int
    runtime_telemetry: dict[str, Any]


@dataclass(slots=True)
class _OutputRuntimeTelemetry:
    enabled: bool
    preparation_seconds: float = 0.0
    backbone_seconds: float = 0.0
    output_decode_seconds: float = 0.0
    feedback_seconds: float = 0.0
    cache_accounting_seconds: float = 0.0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    synchronization_count: int = 0
    backbone_calls: int = 0
    output_decode_calls: int = 0
    feedback_calls: int = 0
    cache_snapshots: int = 0
    graph_signature_counts: Counter[str] = field(default_factory=Counter)

    def device_call[Result](
        self,
        phase: Literal["backbone", "output_decode"],
        callable_: Any,
        device: torch.device,
    ) -> Result:
        if not self.enabled:
            return callable_()
        result, seconds = timed_call(callable_, device)
        if device.type in {"cuda", "mps"}:
            self.synchronization_count += 2
        if phase == "backbone":
            self.backbone_seconds += seconds
            self.backbone_calls += 1
        else:
            self.output_decode_seconds += seconds
            self.output_decode_calls += 1
        return result

    def cache_bytes(self, value: Any) -> int:
        if not self.enabled:
            return cache_tensor_bytes(value)
        started = time.perf_counter()
        result = cache_tensor_bytes(value)
        self.cache_accounting_seconds += time.perf_counter() - started
        self.cache_snapshots += 1
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "preparation_seconds": self.preparation_seconds,
            "backbone_seconds": self.backbone_seconds,
            "output_decode_seconds": self.output_decode_seconds,
            "feedback_seconds": self.feedback_seconds,
            "cache_accounting_seconds": self.cache_accounting_seconds,
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "synchronization_count": self.synchronization_count,
            "backbone_calls": self.backbone_calls,
            "output_decode_calls": self.output_decode_calls,
            "feedback_calls": self.feedback_calls,
            "cache_snapshots": self.cache_snapshots,
            "graph_signature_counts": dict(
                sorted(self.graph_signature_counts.items()),
            ),
        }


@dataclass(slots=True)
class _NativeHeadInvocationCounter:
    count: int = 0

    def __call__(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
    ) -> None:
        self.count += 1


@final
class OutputOnlyGenerator:
    def __init__(
        self,
        backbone: FrozenBackbone,
        codec: OutputByteCodec,
        vocabulary: ByteVocabulary,
        control_ids: Tensor,
    ) -> None:
        if control_ids.shape != (codec.config.control_count,):
            raise ValueError("control IDs must match output codec controls")
        if tuple(control_ids.detach().cpu().tolist()) != vocabulary.control_ids:
            raise ValueError("output codec controls do not match the native vocabulary")
        self.backbone = backbone
        self.codec = codec.eval()
        self.segmenter = NativeByteSegmenter(vocabulary)
        self.control_ids = tuple(control_ids.detach().cpu().tolist())
        self.native_head_invocations = 0
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)

    @contextmanager
    def _count_native_head_invocations(
        self,
    ) -> Iterator[_NativeHeadInvocationCounter]:
        counter = _NativeHeadInvocationCounter()
        handle = native_output_head(self.backbone).register_forward_pre_hook(counter)
        try:
            yield counter
        finally:
            handle.remove()
            self.native_head_invocations += counter.count

    def _event(self, hidden_state: Tensor) -> tuple[OutputEvent | None, int]:
        byte_logits, control_logits = self.codec.decode_logits(hidden_state.to(device=self.codec.device, dtype=self.codec.dtype))
        generated = byte_logits.argmax(dim=-1)[0]
        event_values = torch.cat((control_logits.argmax(dim=-1), generated)).to(
            device="cpu",
            copy=True,
        )
        values = event_values.tolist()
        transfer_bytes = event_values.numel() * event_values.element_size()
        event = output_event_from_prediction(
            values[0],
            values[1:],
            self.control_ids,
            max_span=self.codec.max_span,
        )
        return event, transfer_bytes

    @torch.inference_mode()
    def generate(  # noqa: C901, PLR0912, PLR0915
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        stop_control_ids: frozenset[int],
        max_macro_steps: int,
        max_bytes: int,
        collect_runtime_telemetry: bool = False,
    ) -> OutputGenerationResult:
        if not prompt_token_ids:
            raise ValueError("output-only generation requires a non-empty native prompt")
        if max_macro_steps < 1 or max_bytes < 1:
            raise ValueError("generation limits must be positive")
        telemetry = _OutputRuntimeTelemetry(collect_runtime_telemetry)
        with self._count_native_head_invocations() as head_invocations:
            preparation_started = time.perf_counter() if telemetry.enabled else 0.0
            prompt = torch.tensor([prompt_token_ids], dtype=torch.long, device=self.backbone.device)
            if telemetry.enabled:
                telemetry.preparation_seconds += time.perf_counter() - preparation_started
                if self.backbone.device.type != "cpu":
                    telemetry.host_to_device_bytes += prompt.numel() * prompt.element_size()
                telemetry.graph_signature_counts[f"backbone_prefill:{len(prompt_token_ids)}"] += 1
            outputs = telemetry.device_call(
                "backbone",
                lambda: self.backbone.forward(input_ids=prompt, use_cache=True),
                self.backbone.device,
            )
            hidden = outputs.last_hidden_state[:, -1]
            past: Any = outputs.past_key_values
            final_cache_bytes = telemetry.cache_bytes(past)
            peak_cache_bytes = final_cache_bytes
            model_calls = 1
            analytical_backbone_flops = _backbone_flops(
                self.backbone,
                len(prompt_token_ids),
            )
            events, emitted = [], bytearray()
            native_tokens_represented = 0
            invalid_events = 0
            native_position = len(prompt_token_ids)
            attempted_macro_steps = 0
            termination_reason: OutputTerminationReason = "max_macro_steps"

            for _ in range(max_macro_steps):
                attempted_macro_steps += 1
                event, transfer_bytes = telemetry.device_call(
                    "output_decode",
                    lambda current_hidden=hidden: self._event(current_hidden),
                    self.codec.device,
                )
                if telemetry.enabled:
                    telemetry.device_to_host_bytes += transfer_bytes
                    telemetry.graph_signature_counts[f"output_decode:1x{hidden.shape[-1]}"] += 1
                if event is None:
                    invalid_events += 1
                    termination_reason = "invalid_event"
                    break
                if isinstance(event, ByteSpanEvent) and len(emitted) + len(event.data) > max_bytes:
                    termination_reason = "max_bytes_truncated"
                    break
                events.append(event)
                feedback_started = time.perf_counter() if telemetry.enabled else 0.0
                feedback = self.segmenter.feedback(event)
                if telemetry.enabled:
                    telemetry.feedback_seconds += time.perf_counter() - feedback_started
                    telemetry.feedback_calls += 1
                feedback_token_ids = feedback.token_ids
                native_token_count = len(feedback_token_ids)
                native_tokens_represented += native_token_count
                if isinstance(event, ControlEvent) and event.token_id in stop_control_ids:
                    termination_reason = "stop_control"
                    break
                emitted.extend(feedback.data)
                if len(emitted) >= max_bytes:
                    termination_reason = "max_bytes"
                    break
                preparation_started = time.perf_counter() if telemetry.enabled else 0.0
                ids = torch.tensor([feedback_token_ids], dtype=torch.long, device=self.backbone.device)
                positions = torch.arange(
                    native_position,
                    native_position + native_token_count,
                    dtype=torch.long,
                    device=self.backbone.device,
                ).unsqueeze(0)
                if telemetry.enabled:
                    telemetry.preparation_seconds += time.perf_counter() - preparation_started
                    if self.backbone.device.type != "cpu":
                        telemetry.host_to_device_bytes += ids.numel() * ids.element_size() + positions.numel() * positions.element_size()
                    telemetry.graph_signature_counts[f"backbone_feedback:{native_token_count}"] += 1
                outputs = telemetry.device_call(
                    "backbone",
                    lambda input_ids=ids, position_ids=positions, current_past=past: self.backbone.forward(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        past_key_values=current_past,
                        use_cache=True,
                    ),
                    self.backbone.device,
                )
                past = outputs.past_key_values
                model_calls += 1
                analytical_backbone_flops += _backbone_flops(
                    self.backbone,
                    native_token_count,
                    native_position + native_token_count,
                )
                final_cache_bytes = telemetry.cache_bytes(past)
                peak_cache_bytes = max(peak_cache_bytes, final_cache_bytes)
                hidden = outputs.last_hidden_state[:, -1]
                native_position += native_token_count

        return OutputGenerationResult(
            events=tuple(events),
            data=bytes(emitted),
            macro_steps=attempted_macro_steps,
            native_tokens_represented=native_tokens_represented,
            invalid_events=invalid_events,
            termination_reason=termination_reason,
            model_calls=model_calls,
            final_cache_bytes=final_cache_bytes,
            peak_cache_bytes=peak_cache_bytes,
            analytical_backbone_flops=analytical_backbone_flops,
            analytical_codec_decode_flops=(attempted_macro_steps * output_decode_flops(self.codec.config) if isinstance(self.codec, OutputByteCodec) else 0),
            native_head_invocations=head_invocations.count,
            runtime_telemetry=telemetry.to_dict(),
        )
