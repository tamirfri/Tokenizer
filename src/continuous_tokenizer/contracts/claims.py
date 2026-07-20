from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from typing import Final, Literal, cast, final

from continuous_tokenizer.contracts.experiment import TokenizerMode
from continuous_tokenizer.contracts.parsing import mapping_fingerprint

type ClaimCategory = Literal[
    "quality",
    "efficiency",
    "analytical",
    "deployment",
    "applicability",
]
type ClaimBasis = Literal["measured", "analytical", "applicability"]
type ClaimRole = Literal[
    "primary",
    "prerequisite",
    "secondary",
    "applicability",
]
type ClaimVerdict = Literal[
    "supported",
    "unsupported",
    "incomplete",
    "inapplicable",
]

CLAIM_CATEGORIES: Final[tuple[ClaimCategory, ...]] = (
    "quality",
    "efficiency",
    "analytical",
    "deployment",
    "applicability",
)
CLAIM_VERDICTS: Final = frozenset(
    {
        "supported",
        "unsupported",
        "incomplete",
        "inapplicable",
    }
)
CLAIM_ROLES: Final[tuple[ClaimRole, ...]] = (
    "primary",
    "prerequisite",
    "secondary",
    "applicability",
)
_PROJECT_LEVEL_CLAIMS: Final = frozenset(
    {
        "input.fixed_subset_alignment_feasibility",
        "input.cross_model_confirmation",
        "output.cross_model_confirmation",
    }
)
_MODEL_SEED_DENOMINATOR: Final = "Qwen/Qwen3.5-0.8B and google/gemma-3-270m-it; seeds 17, 23, and 41 for each model"
INPUT_HEADLINE: Final = "usable input compression = exact held-out position compression + registered behavioral similarity"
RESEARCH_THROUGHPUT_EVIDENCE_CLASS: Final = "operational_only"
DEPLOYMENT_PERFORMANCE_EVIDENCE_CLASS: Final = "secondary_final_only"
JOINT_STATE_BUDGET_EVIDENCE_CLASS: Final = "future_prerequisite"
CURRENT_DESIGN_NOTICE: Final = (
    "Only artifacts produced by the current reduced-work design are supported. Old artifacts are "
    "unsupported, and new runs are not comparable to prior configurations. The reduced stages and "
    "denominators lower research power; no new real-model result is implied."
)
DEFAULT_PERFORMANCE_LIMITATION: Final = (
    "Current default final runs use 1 warmup and 2 repetitions. Latency and time-to-first-logit "
    "claims require an explicit full-performance run with at least 5 warmups and 20 repetitions, "
    "so default-run latency evidence is incomplete."
)
INPUT_PERFORMANCE_CLAIM_IDS: Final = frozenset(
    {
        "input.tokenizer_latency_improvement",
        "input.end_to_end_latency_improvement",
        "input.prompt_cache_reduction",
        "input.prefill_compute_reduction",
    }
)


@final
@dataclass(frozen=True, slots=True)
class ClaimDefinition:
    claim_id: str
    mode: TokenizerMode
    category: ClaimCategory
    role: ClaimRole
    basis: ClaimBasis
    applicability: str
    gate_or_policy: str
    evidence_pointers: tuple[str, ...]
    label: str = ""
    producer_symbol: str = ""
    denominator_context: str = ""


@final
@dataclass(frozen=True, slots=True)
class ClaimRecord:
    claim_id: str
    mode: TokenizerMode
    category: ClaimCategory
    role: ClaimRole
    basis: ClaimBasis
    applicability: str
    gate_or_policy: str
    evidence_pointers: tuple[str, ...]
    label: str
    producer_symbol: str
    denominator_context: str
    verdict: ClaimVerdict
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "evidence_pointers": list(self.evidence_pointers),
        }


CLAIM_RECORD_FIELDS: Final = frozenset(field.name for field in fields(ClaimRecord))


_INPUT_CLAIMS: Final = (
    ClaimDefinition(
        "input.fixed_subset_alignment_feasibility",
        "input_only",
        "quality",
        "secondary",
        "measured",
        "prospectively registered deterministic subsets of 128, 256, and 512 reachable canonical ordinary-token rows at seed 17",
        "every staged Large-profile subset passes the registered normalized-RMSE and cosine gates under the sealed continuation rule",
        ("result.json#/stages",),
    ),
    ClaimDefinition(
        "input.full_vocabulary_embedding_compatibility",
        "input_only",
        "quality",
        "secondary",
        "measured",
        "Large-profile final evidence over every reachable canonical ordinary-token payload",
        "registered normalized-RMSE and cosine gates all pass",
        ("tokenizer-metrics.json#/embedding_fit",),
    ),
    ClaimDefinition(
        "input.held_out_position_compression",
        "input_only",
        "efficiency",
        "primary",
        "measured",
        "at most 2,048 held-out WikiText bytes evaluated with the Large profile",
        "minimum_native_tokens_per_continuous_token >= 1.10",
        ("tokenizer-metrics.json#/density/native_tokens_per_continuous_token",),
    ),
    ClaimDefinition(
        "input.registered_behavioral_similarity_tolerances",
        "input_only",
        "quality",
        "prerequisite",
        "measured",
        "Large-profile final evaluation with 16 teacher-forced samples and 2 generation samples, calibrated at batch size 8",
        "registered segmented KL, NLL delta, top-1, and generation-byte similarity tolerances all pass; this is comparative similarity, not non-inferiority",
        (
            "llm-metrics.json#/teacher_forced/segmented",
            "llm-metrics.json#/generation",
        ),
    ),
    ClaimDefinition(
        "input.tokenizer_latency_improvement",
        "input_only",
        "efficiency",
        "secondary",
        "measured",
        "an explicit Large-profile full-performance benchmark with the complete corrected "
        "cold/warm/cache condition matrix and at least 5 warmups and 20 repetitions",
        "registered warm-cache tokenizer latency is lower than disabled-cache latency after exact "
        "semantic-digest equality, raw-observation, denominator, and contamination checks pass",
        ("tokenizer-metrics.json#/segmentation_runs",),
    ),
    ClaimDefinition(
        "input.prompt_cache_reduction",
        "input_only",
        "efficiency",
        "secondary",
        "measured",
        "Large-profile final paired native and segmented prompts under the complete corrected condition matrix; current defaults use 2 prompts",
        "segmented/native materialized prompt-cache byte ratio < 1 after semantic, denominator, raw-observation, and contamination checks pass",
        ("llm-metrics.json#/performance",),
    ),
    ClaimDefinition(
        "input.end_to_end_latency_improvement",
        "input_only",
        "efficiency",
        "secondary",
        "measured",
        "an explicit Large-profile full-performance run over paired native and segmented prompts "
        "under direct preparation-plus-prefill timing with at least 5 warmups and 20 repetitions",
        "segmented/native direct time-to-first-logit ratio < 1 after the complete corrected timing "
        "matrix, semantic, raw-observation, denominator, and contamination checks pass",
        ("llm-metrics.json#/performance",),
    ),
    ClaimDefinition(
        "input.prefill_compute_reduction",
        "input_only",
        "analytical",
        "secondary",
        "analytical",
        "Large-profile final paired prompts under the declared frozen-backbone and local-codec FLOP model",
        "segmented/native total analytical-FLOP ratio < 1 after complete corrected conditions, semantic equality, denominator, and contamination checks pass",
        ("llm-metrics.json#/performance",),
    ),
    ClaimDefinition(
        "input.codec_reference_compactness",
        "input_only",
        "analytical",
        "secondary",
        "analytical",
        "codec state relative to the source input table",
        "maximum_candidate_reference_state_ratio <= 0.50",
        ("tokenizer-metrics.json#/compactness/candidate_reference_state_ratio",),
    ),
    ClaimDefinition(
        "input.physical_input_table_omission",
        "input_only",
        "deployment",
        "secondary",
        "measured",
        "models with a removable untied input table",
        "clean load proves the source input table is absent",
        ("deployment.json#/physical_reference_tensor_absent",),
    ),
    ClaimDefinition(
        "input.input_table_removability",
        "input_only",
        "applicability",
        "applicability",
        "applicability",
        "model architecture",
        "tied tables required for native output remain inapplicable",
        ("deployment.json#/applicability",),
    ),
    ClaimDefinition(
        "input.cross_model_confirmation",
        "input_only",
        "quality",
        "secondary",
        "measured",
        "the pinned Qwen 0.8B and Gemma 270M primary models",
        "independent Large-profile final replications at seeds 17, 23, and 41 "
        "support exact held-out position compression and registered behavioral "
        "similarity for both models",
        ("project.json#/models",),
    ),
)

_OUTPUT_CLAIMS: Final = (
    ClaimDefinition(
        "output.direct_feedback_exactness",
        "output_only",
        "quality",
        "prerequisite",
        "measured",
        "every evaluated output event",
        "byte and native-token direct-feedback equality are both 1.0",
        ("output-metrics.json#/fidelity",),
    ),
    ClaimDefinition(
        "output.valid_non_empty_termination",
        "output_only",
        "quality",
        "prerequisite",
        "measured",
        "every attempted byte-span event",
        "valid non-empty CODEC_EOS termination is 1.0",
        ("output-metrics.json#/valid_non_empty_termination",),
    ),
    ClaimDefinition(
        "output.no_invalid_events",
        "output_only",
        "quality",
        "prerequisite",
        "measured",
        "direct and rollout evaluation",
        "maximum_invalid_events == 0",
        ("output-metrics.json#/invalid_events",),
    ),
    ClaimDefinition(
        "output.control_exactness",
        "output_only",
        "quality",
        "prerequisite",
        "measured",
        "registered prompts containing oracle controls",
        "registered control coverage, precision, and recall gates pass",
        ("output-metrics.json#/control_evidence",),
    ),
    ClaimDefinition(
        "output.stop_exactness",
        "output_only",
        "quality",
        "prerequisite",
        "measured",
        "registered prompts containing oracle stop controls",
        "stop precision and recall are both 1.0",
        ("output-metrics.json#/stop_control",),
    ),
    ClaimDefinition(
        "output.rollout_fidelity",
        "output_only",
        "quality",
        "prerequisite",
        "measured",
        "registered rollout prompts",
        "minimum_rollout_event_agreement >= 0.50",
        ("output-metrics.json#/rollout_event_agreement",),
    ),
    ClaimDefinition(
        "output.semi_autoregressive_density",
        "output_only",
        "efficiency",
        "primary",
        "measured",
        "every attempted output macro-step, including invalid or truncated attempts",
        "minimum_native_tokens_per_attempted_macro_step >= 1.10 only after "
        "direct-feedback exactness, no-invalid-events, valid termination, and "
        "rollout fidelity are all supported",
        ("output-metrics.json#/output_density",),
    ),
    ClaimDefinition(
        "output.codec_reference_compactness",
        "output_only",
        "analytical",
        "secondary",
        "analytical",
        "codec state relative to the native output head",
        "maximum_candidate_reference_state_ratio <= 0.50",
        ("output-metrics.json#/candidate_reference_state_ratio",),
    ),
    ClaimDefinition(
        "output.native_head_bypass",
        "output_only",
        "deployment",
        "secondary",
        "measured",
        "output-tokenizer generation",
        "native vocabulary head is never invoked",
        ("output-metrics.json#/candidate",),
    ),
    ClaimDefinition(
        "output.physical_output_head_omission",
        "output_only",
        "deployment",
        "secondary",
        "measured",
        "models with a removable untied output head",
        "clean load proves the native output head is absent",
        ("deployment.json#/physical_reference_tensor_absent",),
    ),
    ClaimDefinition(
        "output.output_head_removability",
        "output_only",
        "applicability",
        "applicability",
        "applicability",
        "model architecture and native input-feedback requirements",
        "tied tables required for native feedback remain inapplicable",
        ("deployment.json#/applicability",),
    ),
    ClaimDefinition(
        "output.cross_model_confirmation",
        "output_only",
        "quality",
        "secondary",
        "measured",
        "the pinned Qwen 0.8B and Gemma 270M primary models",
        "independent Large-profile final replications at seeds 17, 23, and 41 support the registered output quality claims for both models",
        ("project.json#/models",),
    ),
)

_DENOMINATOR_CONTEXT: Final = {
    "input.fixed_subset_alignment_feasibility": (
        "128-, 256-, and 512-row deterministic vocabulary subsets at seed 17 in one sealed Large-profile study per primary model"
    ),
    "input.full_vocabulary_embedding_compatibility": "all reachable canonical ordinary-token rows in every completed Large-profile final seed",
    "input.held_out_position_compression": "at most 2,048 held-out WikiText bytes in every completed Large-profile final seed",
    "input.registered_behavioral_similarity_tolerances": (
        "16 teacher-forced samples and 2 generation samples in every completed Large-profile full-model seed, after batch-size-8 numerical calibration"
    ),
    "input.tokenizer_latency_improvement": (
        "disabled, cold, and warm encoding-cache conditions with at least 5 warmups and 20 raw "
        "repetitions in every completed final seed; the 1/2 default is incomplete"
    ),
    "input.prompt_cache_reduction": "2 paired native and segmented default prompts across completed seeds, with all semantic and denominator checks",
    "input.end_to_end_latency_improvement": "paired native and segmented prompts with at least 5 warmups and 20 raw repetitions; the 1/2 default is incomplete",
    "input.prefill_compute_reduction": "paired native and segmented prompts under the declared analytical FLOP model",
    "input.codec_reference_compactness": "one candidate/reference state inventory per completed seed",
    "input.physical_input_table_omission": "three clean-process deployment repetitions when architecturally applicable",
    "input.input_table_removability": "one model-architecture applicability determination",
    "input.cross_model_confirmation": "two independent primary-model replications, each containing final seeds 17, 23, and 41",
    "output.direct_feedback_exactness": "every direct macro-event from 16 evaluation samples per completed seed, with spans capped at 8 bytes",
    "output.valid_non_empty_termination": "every attempted byte-span event from 16 evaluation samples per completed seed, with spans capped at 8 bytes",
    "output.no_invalid_events": "all direct and rollout events from the 16-sample, 2-prompt current evaluation",
    "output.control_exactness": "all oracle and predicted structural-control events in the 16-sample, 2-prompt current evaluation",
    "output.stop_exactness": "all oracle and predicted stop-control events in the 16-sample, 2-prompt current evaluation",
    "output.rollout_fidelity": "the 2 registered rollout prompts, each bounded to 16 macro-steps and 512 output bytes, across completed seeds",
    "output.semi_autoregressive_density": (
        "every attempted macro-step in the 16-sample current evaluation, including invalid and truncated attempts, with spans capped at 8 bytes"
    ),
    "output.codec_reference_compactness": "one candidate/reference state inventory per completed seed",
    "output.native_head_bypass": "every output-tokenizer generation call across completed seeds",
    "output.physical_output_head_omission": "three clean-process deployment repetitions when architecturally applicable",
    "output.output_head_removability": "one model-architecture and native-feedback applicability determination",
    "output.cross_model_confirmation": "two independent primary-model replications, each containing final seeds 17, 23, and 41",
}


def _producer_symbol(claim_id: str, mode: TokenizerMode) -> str:
    if claim_id == "input.fixed_subset_alignment_feasibility":
        return "continuous_tokenizer.input.study.run_input_alignment_feasibility_study"
    if claim_id.endswith("cross_model_confirmation"):
        return "continuous_tokenizer.reporting.project.assemble_project_artifact"
    if "physical_" in claim_id or claim_id.endswith("_removability"):
        return "continuous_tokenizer.contracts.claim_derivation.derive_project_deployment_verdicts"
    direction = "input" if mode == "input_only" else "output"
    return f"continuous_tokenizer.contracts.claim_derivation.derive_{direction}_claim_verdicts"


def _reporting_definition(definition: ClaimDefinition) -> ClaimDefinition:
    return replace(
        definition,
        label=definition.claim_id.partition(".")[2].replace("_", " ").title(),
        producer_symbol=_producer_symbol(definition.claim_id, definition.mode),
        denominator_context=_DENOMINATOR_CONTEXT[definition.claim_id],
    )


CLAIM_VOCABULARY: Final = tuple(_reporting_definition(claim) for claim in (*_INPUT_CLAIMS, *_OUTPUT_CLAIMS))
CLAIM_VOCABULARY_SHA256: Final = mapping_fingerprint({"claims": [asdict(claim) for claim in CLAIM_VOCABULARY]})
_CLAIMS_BY_ID: Final = {claim.claim_id: claim for claim in CLAIM_VOCABULARY}


def directional_claims(mode: TokenizerMode) -> tuple[ClaimDefinition, ...]:
    return tuple(claim for claim in CLAIM_VOCABULARY if claim.mode == mode)


def combine_claim_verdicts(verdicts: Iterable[ClaimVerdict]) -> ClaimVerdict:
    values = tuple(verdicts)
    if not values or "incomplete" in values:
        return "incomplete"
    if "unsupported" in values:
        return "unsupported"
    if all(verdict == "inapplicable" for verdict in values):
        return "inapplicable"
    return "supported"


def claim_category_verdicts(
    records: Iterable[Mapping[str, object]],
) -> dict[ClaimCategory, ClaimVerdict]:
    values = tuple(records)
    return {
        category: combine_claim_verdicts(
            cast(ClaimVerdict, record["verdict"])
            for record in values
            if record.get("category") == category and record.get("claim_id") not in _PROJECT_LEVEL_CLAIMS
        )
        for category in CLAIM_CATEGORIES
    }


def claim_record(
    mode: TokenizerMode,
    claim_id: str,
    verdict: ClaimVerdict,
    *,
    evidence_pointers: tuple[str, ...] | None = None,
    reason: str | None = None,
) -> ClaimRecord:
    definition = _CLAIMS_BY_ID.get(claim_id)
    if definition is None or definition.mode != mode:
        raise ValueError(f"claim {claim_id!r} is not registered for {mode}")
    if verdict not in CLAIM_VERDICTS:
        raise ValueError(f"invalid claim verdict: {verdict}")
    if reason is None:
        reason = {
            "supported": "all required evidence is present and the preregistered gate or policy passed",
            "unsupported": "required evidence is complete and the preregistered gate or policy did not pass",
            "incomplete": "required final sealed evidence is missing or incomplete",
            "inapplicable": "the registered applicability policy excludes this claim",
        }[verdict]
    if not reason:
        raise ValueError("claim verdict reason must not be empty")
    return ClaimRecord(
        claim_id=definition.claim_id,
        mode=definition.mode,
        category=definition.category,
        role=definition.role,
        basis=definition.basis,
        applicability=definition.applicability,
        gate_or_policy=definition.gate_or_policy,
        evidence_pointers=definition.evidence_pointers if evidence_pointers is None else evidence_pointers,
        label=definition.label,
        producer_symbol=definition.producer_symbol,
        denominator_context=definition.denominator_context,
        verdict=verdict,
        reason=reason,
    )


def claim_records(
    mode: TokenizerMode,
    verdicts: Mapping[str, ClaimVerdict],
    *,
    reason: str | None = None,
) -> list[dict[str, object]]:
    definitions = directional_claims(mode)
    expected = {definition.claim_id for definition in definitions}
    if set(verdicts) != expected:
        raise ValueError(f"claim verdicts do not match the registered {mode} vocabulary")
    return [
        claim_record(
            mode,
            definition.claim_id,
            verdicts[definition.claim_id],
            reason=reason,
        ).to_dict()
        for definition in definitions
    ]


def project_claim_trace_records(
    claims: Sequence[Mapping[str, object]],
    models: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    traces = []
    for claim in claims:
        claim_id = str(claim["claim_id"])
        cross_model = claim_id.endswith("cross_model_confirmation")
        if claim_id == "input.fixed_subset_alignment_feasibility":
            evidence_class = "prospective_non_final_feasibility"
        elif claim_id in INPUT_PERFORMANCE_CLAIM_IDS:
            evidence_class = DEPLOYMENT_PERFORMANCE_EVIDENCE_CLASS
        elif cross_model:
            evidence_class = "cross_model_final_confirmation"
        else:
            evidence_class = f"final_{claim['basis']}"
        parent_models = []
        for model_index, model in enumerate(models):
            replication = cast(Mapping[str, object], model["replication"])
            parent_claims = cast(
                Sequence[Mapping[str, object]],
                replication["claims"],
            )
            claim_index = next(index for index, parent in enumerate(parent_claims) if parent["claim_id"] == claim_id)
            parent_claim = parent_claims[claim_index]
            seeds = sorted(
                cast(int, cast(Mapping[str, object], run)["seed"])
                for section in ("runs", "failed_runs")
                for run in cast(Sequence[object], replication.get(section, ()))
            )
            parent_models.append(
                {
                    "model": dict(cast(Mapping[str, object], model["model"])),
                    "seeds": seeds,
                    "parent_pointer": (
                        f"project.json#/models/{model_index}/scientific_verdict"
                        if cross_model
                        else f"project.json#/models/{model_index}/replication/claims/{claim_index}"
                    ),
                    "verdict": (model["scientific_verdict"] if cross_model else parent_claim["verdict"]),
                }
            )
        traces.append(
            {
                "claim_id": claim_id,
                "paper_label": claim["label"],
                "evidence_class": evidence_class,
                "producer_symbol": claim["producer_symbol"],
                "canonical_artifact_pointers": list(cast(Sequence[str], claim["evidence_pointers"])),
                "parent_model_evidence": parent_models,
                "model_seed_denominator": _MODEL_SEED_DENOMINATOR,
                "verdict": claim["verdict"],
                "reason": claim["reason"],
            }
        )
    return traces


def claim_verdict(records: object, claim_id: str) -> ClaimVerdict:
    if not isinstance(records, list):
        raise ValueError("claims must be a list of canonical claim records")
    matches = [record for record in records if isinstance(record, dict) and record.get("claim_id") == claim_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one canonical record for {claim_id}")
    verdict = matches[0].get("verdict")
    if verdict not in CLAIM_VERDICTS:
        raise ValueError(f"claim {claim_id} has an invalid verdict")
    return cast(ClaimVerdict, verdict)
