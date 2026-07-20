from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import functional as F

from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.codec.constants import CODEC_EOS
from continuous_tokenizer.codec.output import OutputByteCodec, decode_output_batch
from continuous_tokenizer.output.events import (
    ByteSpanEvent,
    ControlEvent,
    OutputEvent,
    output_event_from_prediction,
)
from continuous_tokenizer.output.feedback import NativeByteSegmenter
from continuous_tokenizer.output.generation import OutputOnlyGenerator
from continuous_tokenizer.output.targets import (
    NativeTrajectoryOptions,
    bounded_output_bytes,
    native_head_trajectory,
    pack_native_trajectory,
)
from continuous_tokenizer.output.trajectory_cache import PreparedOutputCorpus
from continuous_tokenizer.runtime.resume import capture_torch_rng_state, restore_torch_rng_state

if TYPE_CHECKING:
    from continuous_tokenizer.runtime.resume import ResumeManager


@final
@dataclass(frozen=True, slots=True)
class OutputMetrics:
    byte_loss: float
    eos_loss: float
    control_loss: float
    exact_event_agreement: float
    byte_accuracy: float
    valid_non_empty_termination: float
    direct_feedback_byte_equality: float
    direct_feedback_token_equality: float
    invalid_events: int
    invalid_fraction: float
    control_events: int
    control_correct: int
    control_correctness: float | None
    stop_control_events: int
    examples: int
    evaluation_telemetry: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@final
@dataclass(frozen=True, slots=True)
class OutputRolloutMetrics:
    prompts: int
    rollout_event_agreement: float
    rollout_byte_agreement: float
    rollout_token_agreement: float
    exact_full_sequence_rate: float
    first_divergence_position: float
    first_divergence_survival: tuple[float, ...]
    matched_prefix_density: float
    output_bytes_per_macro_step: float
    native_tokens_per_attempted_macro_step: float
    attempted_macro_steps: int
    invalid_events: int
    invalid_fraction: float
    truncated_events: int
    termination_stop_control: int
    termination_invalid_event: int
    termination_max_bytes_truncated: int
    termination_max_bytes: int
    termination_max_macro_steps: int
    oracle_control_events: int
    predicted_control_events: int
    correct_control_events: int
    control_prompt_coverage: float
    control_precision: float | None
    control_recall: float | None
    control_false_positives: int
    control_false_negatives: int
    oracle_stop_control_events: int
    predicted_stop_control_events: int
    correct_stop_control_events: int
    stop_prompt_coverage: float
    stop_precision: float | None
    stop_recall: float | None
    stop_false_positives: int
    stop_false_negatives: int

    def to_dict(self) -> dict[str, float | int | tuple[float, ...] | None]:
        return asdict(self)


@final
@dataclass(frozen=True, slots=True)
class OutputRolloutOptions:
    stop_control_ids: frozenset[int]
    max_macro_steps: int
    max_bytes: int


@final
@dataclass(frozen=True, slots=True)
class OutputEvaluationOptions:
    batch_size: int = 8
    resume_manager: ResumeManager | None = None
    resume_phase: str = "output-evaluation"
    mps_staging_max_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("output evaluation batch size must be positive")
        if self.mps_staging_max_bytes is not None and self.mps_staging_max_bytes < 1:
            raise ValueError("MPS output staging guard must be positive")


def _mps_staging_policy(
    device: torch.device,
    tensor_bytes: int,
    maximum_bytes: int | None,
) -> tuple[bool, str | None]:
    if device.type != "mps":
        return False, "not_mps"
    if maximum_bytes is None:
        return False, "guard_disabled"
    if tensor_bytes > maximum_bytes:
        return False, "memory_guard_exceeded"
    return True, None


def _agreement(left: tuple[object, ...] | bytes, right: tuple[object, ...] | bytes) -> tuple[int, int]:
    if not left and not right:
        return 1, 1
    return (
        sum(left_value == right_value for left_value, right_value in zip(left, right, strict=False)),
        max(len(left), len(right), 1),
    )


def _matching_prefix(
    predicted: tuple[int, ...],
    oracle: tuple[int, ...],
) -> int:
    for index, (predicted_id, oracle_id) in enumerate(
        zip(predicted, oracle, strict=False),
    ):
        if predicted_id != oracle_id:
            return index
    return min(len(predicted), len(oracle))


def _indexed_controls(
    prompt_index: int,
    events: tuple[OutputEvent, ...],
) -> set[tuple[int, int, int]]:
    return {(prompt_index, index, event.token_id) for index, event in enumerate(events) if isinstance(event, ControlEvent)}


def _record_survival(
    counts: list[int],
    *,
    oracle_tokens: int,
    matched_prefix: int,
) -> None:
    counts.extend(0 for _ in range(oracle_tokens - len(counts)))
    for position in range(matched_prefix):
        counts[position] += 1


def _classification_counts(
    oracle: set[tuple[int, int, int]],
    predicted: set[tuple[int, int, int]],
) -> tuple[int, int, int]:
    return (
        len(oracle & predicted),
        len(predicted - oracle),
        len(oracle - predicted),
    )


@torch.inference_mode()
def evaluate_output_rollouts(
    backbone: FrozenBackbone,
    generator: OutputOnlyGenerator,
    vocabulary: ByteVocabulary,
    prompts: tuple[tuple[int, ...], ...],
    options: OutputRolloutOptions,
) -> OutputRolloutMetrics:
    if len(prompts) < 2:
        raise ValueError("output rollout evaluation requires multiple registered prompts")
    segmenter = NativeByteSegmenter(vocabulary)
    event_matches = event_total = 0
    byte_matches = byte_total = 0
    token_matches = token_total = 0
    output_bytes = attempted_steps = represented_tokens = invalid_events = truncated_events = 0
    exact_full_sequences = matched_prefix_tokens = 0
    divergence_positions: list[int] = []
    survival_counts: list[int] = []
    control_prompts = 0
    oracle_controls: set[tuple[int, int, int]] = set()
    predicted_controls: set[tuple[int, int, int]] = set()
    oracle_stops: set[tuple[int, int, int]] = set()
    predicted_stops: set[tuple[int, int, int]] = set()
    termination_counts = {
        "stop_control": 0,
        "invalid_event": 0,
        "max_bytes_truncated": 0,
        "max_bytes": 0,
        "max_macro_steps": 0,
    }
    for prompt_index, prompt in enumerate(prompts):
        oracle = native_head_trajectory(
            backbone,
            vocabulary,
            prompt,
            NativeTrajectoryOptions(
                stop_control_ids=options.stop_control_ids,
                max_native_tokens=options.max_macro_steps,
                max_bytes=options.max_bytes,
            ),
        )
        packed_oracle = pack_native_trajectory(
            oracle,
            vocabulary,
            max_span=generator.codec.max_span,
        )
        predicted = generator.generate(
            prompt,
            stop_control_ids=options.stop_control_ids,
            max_macro_steps=options.max_macro_steps,
            max_bytes=options.max_bytes,
        )
        matches, total = _agreement(predicted.events, packed_oracle.events)
        event_matches += matches
        event_total += total
        oracle_bytes = bounded_output_bytes(
            packed_oracle.events,
            stop_control_ids=options.stop_control_ids,
            max_macro_steps=options.max_macro_steps,
            max_bytes=options.max_bytes,
        )
        matches, total = _agreement(predicted.data, oracle_bytes)
        byte_matches += matches
        byte_total += total
        predicted_token_ids = tuple(token_id for event in predicted.events for token_id in segmenter.feedback(event).token_ids)
        matches, total = _agreement(predicted_token_ids, oracle.native_token_ids)
        token_matches += matches
        token_total += total
        exact_full_sequences += predicted_token_ids == oracle.native_token_ids
        matched_prefix = _matching_prefix(predicted_token_ids, oracle.native_token_ids)
        matched_prefix_tokens += matched_prefix
        divergence_positions.append(matched_prefix)
        _record_survival(
            survival_counts,
            oracle_tokens=len(oracle.native_token_ids),
            matched_prefix=matched_prefix,
        )

        oracle_control_rows = _indexed_controls(prompt_index, packed_oracle.events)
        predicted_control_rows = _indexed_controls(prompt_index, predicted.events)
        oracle_controls.update(oracle_control_rows)
        predicted_controls.update(predicted_control_rows)
        control_prompts += bool(oracle_control_rows)
        oracle_stops.update(row for row in oracle_control_rows if row[2] in options.stop_control_ids)
        predicted_stops.update(row for row in predicted_control_rows if row[2] in options.stop_control_ids)
        output_bytes += len(predicted.data)
        attempted_steps += predicted.macro_steps
        represented_tokens += predicted.native_tokens_represented
        invalid_events += predicted.invalid_events
        truncated_events += predicted.termination_reason == "max_bytes_truncated"
        termination_counts[predicted.termination_reason] += 1
    correct_controls, control_false_positives, control_false_negatives = _classification_counts(
        oracle_controls,
        predicted_controls,
    )
    correct_stops, stop_false_positives, stop_false_negatives = _classification_counts(
        oracle_stops,
        predicted_stops,
    )
    return OutputRolloutMetrics(
        prompts=len(prompts),
        rollout_event_agreement=event_matches / event_total,
        rollout_byte_agreement=byte_matches / byte_total,
        rollout_token_agreement=token_matches / token_total,
        exact_full_sequence_rate=exact_full_sequences / len(prompts),
        first_divergence_position=sum(divergence_positions) / len(prompts),
        first_divergence_survival=tuple(count / len(prompts) for count in survival_counts),
        matched_prefix_density=matched_prefix_tokens / max(attempted_steps, 1),
        output_bytes_per_macro_step=output_bytes / max(attempted_steps, 1),
        native_tokens_per_attempted_macro_step=represented_tokens / max(attempted_steps, 1),
        attempted_macro_steps=attempted_steps,
        invalid_events=invalid_events,
        invalid_fraction=invalid_events / max(attempted_steps, 1),
        truncated_events=truncated_events,
        termination_stop_control=termination_counts["stop_control"],
        termination_invalid_event=termination_counts["invalid_event"],
        termination_max_bytes_truncated=termination_counts["max_bytes_truncated"],
        termination_max_bytes=termination_counts["max_bytes"],
        termination_max_macro_steps=termination_counts["max_macro_steps"],
        oracle_control_events=len(oracle_controls),
        predicted_control_events=len(predicted_controls),
        correct_control_events=correct_controls,
        control_prompt_coverage=control_prompts / len(prompts),
        control_precision=(None if not predicted_controls else correct_controls / len(predicted_controls)),
        control_recall=(None if not oracle_controls else correct_controls / len(oracle_controls)),
        control_false_positives=control_false_positives,
        control_false_negatives=control_false_negatives,
        oracle_stop_control_events=len(oracle_stops),
        predicted_stop_control_events=len(predicted_stops),
        correct_stop_control_events=correct_stops,
        stop_prompt_coverage=len({row[0] for row in oracle_stops}) / len(prompts),
        stop_precision=(None if not predicted_stops else correct_stops / len(predicted_stops)),
        stop_recall=(None if not oracle_stops else correct_stops / len(oracle_stops)),
        stop_false_positives=stop_false_positives,
        stop_false_negatives=stop_false_negatives,
    )


def _exact_output_predictions(
    generated: Tensor,
    selectors: Tensor,
    event_targets: Tensor,
    byte_targets: Tensor,
    byte_mask: Tensor,
) -> Tensor:
    span_rows = event_targets == 0
    generated_matches = ((generated == byte_targets) | ~byte_mask).all(dim=1)
    return torch.where(
        span_rows,
        (selectors == 0) & generated_matches,
        selectors == event_targets,
    )


def exact_output_predictions(
    byte_logits: Tensor,
    control_logits: Tensor,
    event_targets: Tensor,
    byte_targets: Tensor,
    byte_mask: Tensor,
) -> Tensor:
    return _exact_output_predictions(
        byte_logits.argmax(dim=-1),
        control_logits.argmax(dim=-1),
        event_targets,
        byte_targets,
        byte_mask,
    )


def _predicted_events(
    generated: Tensor,
    selectors: Tensor,
    control_ids: tuple[int, ...],
    *,
    max_span: int,
) -> tuple[OutputEvent | None, ...]:
    selector_values = selectors.detach().cpu().tolist()
    generated_rows = generated.detach().cpu().tolist()
    return tuple(
        output_event_from_prediction(
            selector,
            row,
            control_ids,
            max_span=max_span,
        )
        for selector, row in zip(selector_values, generated_rows, strict=True)
    )


@final
@dataclass(frozen=True, slots=True)
class _DirectFeedbackBatch:
    predictions: tuple[OutputEvent | None, ...]
    control_rows: Tensor
    corpus: PreparedOutputCorpus
    start: int
    segmenter: NativeByteSegmenter


def _direct_feedback_counts(batch: _DirectFeedbackBatch) -> tuple[int, int, int, int]:
    direct_bytes = direct_tokens = invalid = correct_controls = 0
    for row, (prediction, is_control) in enumerate(
        zip(
            batch.predictions,
            batch.control_rows.detach().cpu().tolist(),
            strict=True,
        ),
    ):
        if prediction is None:
            invalid += 1
            continue
        feedback = batch.segmenter.feedback(prediction)
        native_target = batch.corpus.target_native_tokens(batch.start + row)
        if is_control:
            exact_control = isinstance(prediction, ControlEvent) and (prediction.token_id,) == native_target
            direct_bytes += exact_control
            correct_controls += exact_control
        else:
            event = batch.start + row
            payload_length = int(batch.corpus.payload_bytes[event].item())
            target_payload = bytes(batch.corpus.byte_targets[event, :payload_length].tolist())
            direct_bytes += isinstance(prediction, ByteSpanEvent) and feedback.data == target_payload
        direct_tokens += feedback.token_ids == native_target
    return direct_bytes, direct_tokens, invalid, correct_controls


def _output_evaluation_batches(
    corpus: PreparedOutputCorpus,
    options: OutputEvaluationOptions,
) -> tuple[tuple[int, int], ...]:
    snapshot_offsets = iter(())
    if options.resume_manager is not None:
        snapshot_offsets = iter(
            offset for sequence, offset in enumerate(corpus.sequence_offsets[1:], start=1) if options.resume_manager.should_snapshot(sequence)
        )
    next_snapshot = next(snapshot_offsets, corpus.examples)
    batches: list[tuple[int, int]] = []
    start = 0
    while start < corpus.examples:
        stop = min(start + options.batch_size, next_snapshot, corpus.examples)
        batches.append((start, stop))
        start = stop
        if start == next_snapshot:
            next_snapshot = next(snapshot_offsets, corpus.examples)
    return tuple(batches)


@torch.inference_mode()
def evaluate_output_codec(  # noqa: PLR0915 - Single-pass device accumulation.
    codec: OutputByteCodec,
    corpus: PreparedOutputCorpus,
    vocabulary: ByteVocabulary,
    options: OutputEvaluationOptions,
) -> OutputMetrics:
    control_ids = vocabulary.control_ids
    if len(control_ids) != codec.config.control_count:
        raise ValueError("output evaluation controls do not match the codec")
    segmenter = NativeByteSegmenter(vocabulary)
    device = codec.device
    source_corpus = corpus
    staged, staging_fallback = _mps_staging_policy(
        device,
        corpus.tensor_bytes,
        options.mps_staging_max_bytes,
    )
    if staged:
        corpus = corpus.staged(device=device, dtype=codec.dtype)
    totals = torch.zeros(14, dtype=torch.float32, device=device)
    start_sequence = 0
    state = None if options.resume_manager is None else options.resume_manager.latest(options.resume_phase)
    if state is not None:
        totals.copy_(state["totals"].to(device))
        start_sequence = int(state["next_sequence"])
        restore_torch_rng_state(device, state["torch_rng"])
    start_event = corpus.sequence_offsets[start_sequence]
    sequence_boundaries = {offset: sequence for sequence, offset in enumerate(corpus.sequence_offsets[1:], start=1)}
    evaluation_batches = _output_evaluation_batches(corpus, options)
    for start, stop in evaluation_batches:
        if start < start_event:
            continue
        hidden, event_targets, byte_targets, byte_mask = corpus.batch(
            start,
            stop,
            device=device,
            dtype=codec.dtype,
        )
        byte_logits, control_logits = decode_output_batch(
            codec,
            hidden,
            maximum_rows=options.batch_size,
        )
        span_rows = event_targets == 0
        control_rows = ~span_rows
        position_loss = F.cross_entropy(
            byte_logits.transpose(1, 2),
            byte_targets,
            reduction="none",
        )
        eos_indices = byte_mask.sum(dim=1).sub(1).clamp_min(0)
        eos_mask = F.one_hot(
            eos_indices,
            num_classes=byte_mask.shape[1],
        ).to(torch.bool)
        payload_mask = byte_mask & ~eos_mask
        span_weights = span_rows.to(position_loss.dtype)
        byte_loss = (position_loss * payload_mask).sum(dim=1) / payload_mask.sum(dim=1).clamp_min(1)
        eos_loss = position_loss.gather(1, eos_indices[:, None]).squeeze(1)
        control_loss = F.cross_entropy(
            control_logits,
            event_targets,
            reduction="none",
        )
        generated = byte_logits.argmax(dim=-1)
        selectors = control_logits.argmax(dim=-1)
        generated_eos = generated == CODEC_EOS
        first_eos = generated_eos.to(torch.int64).argmax(dim=1)
        valid = generated_eos.any(dim=1) & (first_eos >= 1) & (first_eos <= codec.max_span) & span_rows
        parsed = torch.cat((selectors[:, None], generated), dim=1).detach().cpu()
        predictions = _predicted_events(
            parsed[:, 1:],
            parsed[:, 0],
            control_ids,
            max_span=codec.max_span,
        )
        direct_bytes, direct_tokens, invalid, correct_controls = _direct_feedback_counts(
            _DirectFeedbackBatch(
                predictions=predictions,
                control_rows=parsed[:, 0] != 0,
                corpus=source_corpus,
                start=start,
                segmenter=segmenter,
            ),
        )
        totals += torch.stack(
            (
                (byte_loss * span_weights).sum(),
                (eos_loss * span_weights).sum(),
                (control_loss * control_rows).sum(),
                _exact_output_predictions(
                    generated,
                    selectors,
                    event_targets,
                    byte_targets,
                    byte_mask,
                ).sum(),
                ((generated == byte_targets) & payload_mask).sum(),
                payload_mask.sum(),
                valid.sum(),
                span_rows.sum(),
                control_rows.sum(),
                torch.tensor(direct_bytes, device=device),
                torch.tensor(direct_tokens, device=device),
                torch.tensor(invalid, device=device),
                torch.tensor(correct_controls, device=device),
                corpus.stop_targets[start:stop].to(device).sum(),
            )
        ).to(totals.dtype)
        boundary = sequence_boundaries.get(stop)
        if boundary is not None and options.resume_manager is not None and (options.resume_manager.should_snapshot(boundary) or boundary == corpus.sequences):
            options.resume_manager.save(
                options.resume_phase,
                boundary,
                {
                    "completed": boundary == corpus.sequences,
                    "totals": totals.detach().cpu(),
                    "next_sequence": boundary,
                    "torch_rng": capture_torch_rng_state(device),
                },
            )
    examples = corpus.examples
    if examples == 0:
        raise ValueError("output evaluation requires at least one next-event example")
    (
        byte_loss_total,
        eos_loss_total,
        control_loss_total,
        exact_events,
        correct_bytes,
        payload_positions,
        valid_spans,
        span_count,
        control_count,
        direct_bytes,
        direct_tokens,
        invalid_events,
        correct_controls,
        stop_controls,
    ) = totals.tolist()
    span_examples = max(1.0, span_count)
    control_examples = max(1.0, control_count)
    logical_batches = len(evaluation_batches)
    logical_host_to_device = 0
    logical_device_to_host = 0
    if device.type != "cpu":
        logical_host_to_device = 5 if staged else logical_batches * 5
        logical_device_to_host = logical_batches + 1
    return OutputMetrics(
        byte_loss=byte_loss_total / span_examples,
        eos_loss=eos_loss_total / span_examples,
        control_loss=control_loss_total / control_examples,
        exact_event_agreement=exact_events / examples,
        byte_accuracy=correct_bytes / max(payload_positions, 1.0),
        valid_non_empty_termination=valid_spans / span_examples,
        direct_feedback_byte_equality=direct_bytes / examples,
        direct_feedback_token_equality=direct_tokens / examples,
        invalid_events=int(invalid_events),
        invalid_fraction=invalid_events / examples,
        control_events=int(control_count),
        control_correct=int(correct_controls),
        control_correctness=None if control_count == 0 else correct_controls / control_count,
        stop_control_events=int(stop_controls),
        examples=examples,
        evaluation_telemetry={
            "batch_size": options.batch_size,
            "batches": logical_batches,
            "processed_rows": corpus.examples,
            "batch_fill_ratio": corpus.examples / max(logical_batches * options.batch_size, 1),
            "cross_sequence_batching": True,
            "sequence_order": tuple(range(corpus.sequences)),
            "resume_boundary": "complete_sequences_only",
            "mps_corpus_staged": staged,
            "mps_staging_max_bytes": options.mps_staging_max_bytes,
            "mps_staging_fallback": staging_fallback,
            "host_to_device_transfers": logical_host_to_device,
            "device_to_host_transfers": logical_device_to_host,
            "synchronization_count": (logical_device_to_host if device.type in {"cuda", "mps"} else 0),
        },
    )
