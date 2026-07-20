from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, final

from continuous_tokenizer.contracts.parsing import is_lowercase_sha256

type StatementKind = Literal["protocol", "software"]
type StatementScope = Literal["input_only", "output_only", "shared"]
type StatementStatus = Literal["proved", "validated", "not_validated"]
type SyntheticMode = Literal["input_only", "output_only"]


@final
@dataclass(frozen=True, slots=True)
class StatementDefinition:
    statement_id: str
    kind: StatementKind
    paper_label: str
    summary: str
    implementation_symbols: tuple[str, ...]
    validating_test_ids: tuple[str, ...]
    evidence_requirement: str
    scope: StatementScope
    required_synthetic_modes: tuple[SyntheticMode, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"protocol", "software"}:
            raise ValueError("statement kind is invalid")
        if self.scope not in {"input_only", "output_only", "shared"}:
            raise ValueError("statement scope is invalid")
        if not all(
            (
                self.statement_id,
                self.paper_label,
                self.summary,
                self.implementation_symbols,
                self.validating_test_ids,
                self.evidence_requirement,
            )
        ):
            raise ValueError("statement trace metadata must be complete")
        if self.kind == "protocol" and self.required_synthetic_modes:
            raise ValueError("protocol statements cannot require synthetic evidence")
        if self.kind == "software" and not self.required_synthetic_modes:
            raise ValueError("software statements must require synthetic evidence")
        if any(mode not in {"input_only", "output_only"} for mode in self.required_synthetic_modes):
            raise ValueError("required synthetic mode is invalid")


@final
@dataclass(frozen=True, slots=True)
class SourceBinding:
    source_state_sha256: str
    dependency_lock_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source_state_sha256", self.source_state_sha256),
            ("dependency_lock_sha256", self.dependency_lock_sha256),
        ):
            if not is_lowercase_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@final
@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    source: SourceBinding
    all_passed: bool
    pointer: str

    def __post_init__(self) -> None:
        if type(self.all_passed) is not bool:
            raise TypeError("verification all_passed must be a bool")
        if not self.pointer:
            raise ValueError("verification evidence pointer must not be empty")


@final
@dataclass(frozen=True, slots=True)
class SyntheticCampaignEvidence:
    mode: SyntheticMode
    source: SourceBinding
    operational_status: Literal["completed", "failed"]
    software_checks_passed: bool
    pointer: str

    def __post_init__(self) -> None:
        if self.mode not in {"input_only", "output_only"}:
            raise ValueError("synthetic campaign mode is invalid")
        if self.operational_status not in {"completed", "failed"}:
            raise ValueError("synthetic campaign operational status is invalid")
        if type(self.software_checks_passed) is not bool:
            raise TypeError("synthetic campaign software_checks_passed must be a bool")
        if not self.pointer:
            raise ValueError("synthetic campaign evidence pointer must not be empty")


@final
@dataclass(frozen=True, slots=True)
class SoftwareValidationInputs:
    verification: VerificationEvidence
    synthetic_campaigns: tuple[SyntheticCampaignEvidence, ...]

    def __post_init__(self) -> None:
        modes = tuple(campaign.mode for campaign in self.synthetic_campaigns)
        if len(modes) != len(set(modes)):
            raise ValueError("software validation inputs contain duplicate synthetic modes")


@final
@dataclass(frozen=True, slots=True)
class StatementTrace:
    definition: StatementDefinition
    status: StatementStatus
    evidence_pointers: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        definition = self.definition
        return {
            "statement_id": definition.statement_id,
            "paper_label": definition.paper_label,
            "kind": definition.kind,
            "scope": definition.scope,
            "statement": definition.summary,
            "implementation_symbols": list(definition.implementation_symbols),
            "validating_test_ids": list(definition.validating_test_ids),
            "evidence_requirement": definition.evidence_requirement,
            "canonical_artifact_pointers": list(self.evidence_pointers),
            "model_seed_denominator": ("not applicable — protocol/software statement, not empirical model evidence"),
            "verdict": self.status,
            "reason": self.reason,
        }


STATEMENT_REGISTRY: Final = (
    StatementDefinition(
        statement_id="protocol.accepted_span_exactness",
        kind="protocol",
        paper_label="Proposition 1 — accepted-span reconstruction is exact",
        summary=("Every accepted multi-byte discrete byte span decodes to its exact payload and terminates at the expected private CODEC_EOS position."),
        implementation_symbols=(
            "continuous_tokenizer.codec.input.InputByteCodec.reconstruction_matches",
            "continuous_tokenizer.input.segmentation.validate_spans",
        ),
        validating_test_ids=(
            "test_codec_core.test_private_eos_terminates_payload_and_rejects_invalid_frames",
            "test_input_evidence.test_independent_decoder_bytes_and_exact_eos_are_empirical_evidence",
        ),
        evidence_requirement="protocol construction: acceptance requires exact bytes and exact CODEC_EOS",
        scope="input_only",
    ),
    StatementDefinition(
        statement_id="protocol.atomic_fallback",
        kind="protocol",
        paper_label="Proposition 2 — arbitrary finite byte strings remain representable",
        summary=("Every byte has an exact atomic latent vector fallback, so any finite byte sequence remains representable as continuous tokens."),
        implementation_symbols=(
            "continuous_tokenizer.input.segmentation.greedy_segment",
            "continuous_tokenizer.input.segmentation.reconstruct",
        ),
        validating_test_ids=(
            "test_input_segmentation.test_atomic_fallback_round_trips_all_byte_values",
            "test_input_alignment.test_arbitrary_bytes_round_trip_through_atomic_fallback",
        ),
        evidence_requirement="protocol construction: one exact frozen atomic row exists for each byte value",
        scope="input_only",
    ),
    StatementDefinition(
        statement_id="protocol.exhaustive_local_maximality",
        kind="protocol",
        paper_label="Proposition 3 — exhaustive greedy selection is locally maximal",
        summary=("At each offset, every permitted candidate length is evaluated independently and the longest valid discrete byte span is selected."),
        implementation_symbols=(
            "continuous_tokenizer.input.segmentation.segment_bytes",
            "continuous_tokenizer.input.segmentation.greedy_segment",
        ),
        validating_test_ids=(
            "test_input_segmentation.test_longer_valid_span_survives_shorter_invalid_span",
            "test_input_segmentation.test_longest_of_all_valid_candidates_is_selected",
        ),
        evidence_requirement="protocol construction: exhaustive candidate evaluation without monotonic pruning",
        scope="input_only",
    ),
    StatementDefinition(
        statement_id="software.cache_semantics",
        kind="software",
        paper_label="Software statement 4 — cache state cannot alter semantics",
        summary=("Encoding-cache state may change latency but cannot change accepted spans, continuous-token segmentation, or latent vectors."),
        implementation_symbols=(
            "continuous_tokenizer.codec.encoding_cache.EncodingCache",
            "continuous_tokenizer.input.segmentation.segment_bytes",
        ),
        validating_test_ids=("test_input_segmentation.test_cache_does_not_change_segmentation",),
        evidence_requirement=("source-bound complete verification and a completed passing input-only synthetic campaign"),
        scope="input_only",
        required_synthetic_modes=("input_only",),
    ),
    StatementDefinition(
        statement_id="software.backbone_immutability",
        kind="software",
        paper_label="Software statement 5 — backbone immutability is auditable",
        summary=("Input distillation and output-codec training freeze the language-model backbone and reject any parameter-fingerprint change."),
        implementation_symbols=(
            "continuous_tokenizer.input.training.distillation.FrozenBackboneDistiller.run",
            "continuous_tokenizer.output.training.OutputCodecTrainer.run",
            "continuous_tokenizer.runtime.tensors.parameter_fingerprint",
        ),
        validating_test_ids=(
            "test_input_distillation.test_distillation_trains_only_codec_parameters",
            "test_output_training.OutputModeTests.test_output_training_preserves_frozen_backbone",
        ),
        evidence_requirement=("source-bound complete verification and completed passing input-only and output-only synthetic campaigns"),
        scope="shared",
        required_synthetic_modes=("input_only", "output_only"),
    ),
    StatementDefinition(
        statement_id="software.input_path",
        kind="software",
        paper_label="Evidence ladder — deterministic input-only software path",
        summary=(
            "The input-only path executes reconstruction-gated segmentation, continuous-token "
            "backbone input, and artifact publication in the deterministic offline campaign."
        ),
        implementation_symbols=("continuous_tokenizer.campaigns.input.InputExperimentRunner.run",),
        validating_test_ids=("test_input_campaign.test_synthetic_spec_runs_complete_offline_artifact",),
        evidence_requirement=("source-bound complete verification and a completed passing input-only synthetic campaign"),
        scope="input_only",
        required_synthetic_modes=("input_only",),
    ),
    StatementDefinition(
        statement_id="software.output_path",
        kind="software",
        paper_label="Evidence ladder — deterministic output-only software path",
        summary=(
            "The output-only path emits non-empty byte spans or structural controls, performs "
            "deterministic native-token feedback, bypasses the native head, and publishes artifacts."
        ),
        implementation_symbols=("continuous_tokenizer.campaigns.output.OutputExperimentRunner.run",),
        validating_test_ids=("test_output_campaign.OutputModeTests.test_synthetic_output_campaign_proves_end_to_end_path",),
        evidence_requirement=("source-bound complete verification and a completed passing output-only synthetic campaign"),
        scope="output_only",
        required_synthetic_modes=("output_only",),
    ),
)


def _not_validated(
    definition: StatementDefinition,
    evidence_pointers: tuple[str, ...],
    reason: str,
) -> StatementTrace:
    return StatementTrace(
        definition,
        "not_validated",
        evidence_pointers,
        reason,
    )


def _validation_trace(
    definition: StatementDefinition,
    inputs: SoftwareValidationInputs | None,
) -> StatementTrace:
    if inputs is None:
        return _not_validated(
            definition,
            (),
            "no source-bound verification and synthetic campaign inputs were supplied",
        )
    verification = inputs.verification
    if not verification.all_passed:
        return _not_validated(
            definition,
            (verification.pointer,),
            "complete verification did not pass",
        )
    campaigns = {campaign.mode: campaign for campaign in inputs.synthetic_campaigns}
    if any(mode not in campaigns for mode in definition.required_synthetic_modes):
        return _not_validated(
            definition,
            (verification.pointer,),
            "one or more required synthetic campaign inputs were not supplied",
        )
    required = tuple(campaigns[mode] for mode in definition.required_synthetic_modes)
    pointers = (verification.pointer, *(campaign.pointer for campaign in required))
    if any(campaign.source != verification.source for campaign in required):
        return _not_validated(
            definition,
            pointers,
            "verification and synthetic campaign source bindings do not match",
        )
    if any(campaign.operational_status != "completed" or not campaign.software_checks_passed for campaign in required):
        return _not_validated(
            definition,
            pointers,
            "one or more required synthetic campaigns did not complete with passing software checks",
        )
    return StatementTrace(
        definition,
        "validated",
        pointers,
        "source-bound complete verification and every required synthetic campaign passed",
    )


def statement_trace(
    inputs: SoftwareValidationInputs | None = None,
) -> tuple[StatementTrace, ...]:
    return tuple(
        StatementTrace(
            definition,
            "proved",
            (),
            "proved by the protocol construction stated in the evidence requirement",
        )
        if definition.kind == "protocol"
        else _validation_trace(definition, inputs)
        for definition in STATEMENT_REGISTRY
    )


def statement_trace_records(
    inputs: SoftwareValidationInputs | None = None,
) -> list[dict[str, object]]:
    return [trace.to_dict() for trace in statement_trace(inputs)]
