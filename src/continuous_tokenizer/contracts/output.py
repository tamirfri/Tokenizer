from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Self, final

from continuous_tokenizer.contracts.parsing import (
    mapping_fingerprint,
    non_negative_int,
    parse_defaults,
    positive,
    table,
)
from continuous_tokenizer.contracts.profiles import (
    CAMPAIGN_PROFILE_NAME,
    TRAINING_PROFILE_NAMES,
)

OUTPUT_FIDELITY_PROMPT_SET: Final = "frozen-greedy-fidelity"
OUTPUT_FIDELITY_PROMPTS: Final = (
    "Complete this sentence with one short clause: A reproducible experiment",
    "Answer with one word: what color is a clear daytime sky?",
)
OUTPUT_FIDELITY_PROMPT_SET_SHA256: Final = mapping_fingerprint(
    OUTPUT_FIDELITY_PROMPTS,
)
OUTPUT_ORACLE_SPAN_LIMITS: Final = (1, 2, 4, 8)


def registered_output_prompts(name: str, sha256: str) -> tuple[str, ...]:
    if name != OUTPUT_FIDELITY_PROMPT_SET:
        raise ValueError("evaluation.prompt_set is not registered")
    if sha256 != OUTPUT_FIDELITY_PROMPT_SET_SHA256:
        raise ValueError("evaluation.prompt_set_sha256 does not match the registered prompts")
    return OUTPUT_FIDELITY_PROMPTS


@final
@dataclass(frozen=True, slots=True)
class OutputTrainingSpec:
    profile: str = CAMPAIGN_PROFILE_NAME
    max_span: int = 8
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0

    @classmethod
    def parse(cls, value: object) -> Self:
        parsed = parse_defaults(cls, value, "training")
        if parsed.profile not in TRAINING_PROFILE_NAMES:
            choices = ", ".join(TRAINING_PROFILE_NAMES)
            raise ValueError(f"training.profile must be one of: {choices}")
        for name in ("max_span", "epochs", "batch_size"):
            positive(getattr(parsed, name), f"training.{name}")
        if parsed.max_span not in OUTPUT_ORACLE_SPAN_LIMITS:
            raise ValueError("training.max_span must be 1, 2, 4, or 8")
        positive(parsed.learning_rate, "training.learning_rate")
        positive(parsed.weight_decay, "training.weight_decay", allow_zero=True)
        return parsed


@final
@dataclass(frozen=True, slots=True)
class OutputCorpusSpec:
    split: str
    seed: int

    @classmethod
    def parse(cls, value: object, name: str, expected_split: str) -> Self:
        values = table(value, f"evaluation.{name}")
        if set(values) != {"split", "seed"}:
            raise ValueError(f"evaluation.{name} requires exactly split and seed")
        split = values.get("split")
        if split != expected_split:
            raise ValueError(
                f"evaluation.{name}.split must be {expected_split!r}",
            )
        return cls(
            split=expected_split,
            seed=non_negative_int(
                values.get("seed"),
                f"evaluation.{name}.seed",
            ),
        )


@final
@dataclass(frozen=True, slots=True)
class OutputEvaluationSpec:
    batch_size: int = 8
    samples: int = 16
    max_macro_steps: int = 16
    max_output_bytes: int = 512
    warmups: int = 1
    repetitions: int = 2
    prompt_set: str = OUTPUT_FIDELITY_PROMPT_SET
    prompt_set_sha256: str = OUTPUT_FIDELITY_PROMPT_SET_SHA256
    oracle_span_limits: tuple[int, ...] = OUTPUT_ORACLE_SPAN_LIMITS
    training_corpus: OutputCorpusSpec = OutputCorpusSpec("train", 1701)
    checkpoint_selection_corpus: OutputCorpusSpec = OutputCorpusSpec("train", 1702)
    oracle_validation_corpus: OutputCorpusSpec = OutputCorpusSpec("validation", 1703)
    final_test_corpus: OutputCorpusSpec = OutputCorpusSpec("registered_prompts", 1704)

    @classmethod
    def parse(cls, value: object) -> Self:
        values = dict(table(value, "evaluation"))
        corpus_fields = {
            "training_corpus": "train",
            "checkpoint_selection_corpus": "train",
            "oracle_validation_corpus": "validation",
            "final_test_corpus": "registered_prompts",
        }
        if any(name not in values for name in corpus_fields):
            raise ValueError("evaluation requires all four output corpus contracts")
        corpora = {name: OutputCorpusSpec.parse(values.pop(name), name, split) for name, split in corpus_fields.items()}
        raw_limits = values.pop("oracle_span_limits", list(OUTPUT_ORACLE_SPAN_LIMITS))
        parsed = parse_defaults(
            cls,
            {
                **values,
                "oracle_span_limits": tuple(raw_limits),
                **corpora,
            },
            "evaluation",
        )
        for name in (
            "batch_size",
            "samples",
            "max_macro_steps",
            "max_output_bytes",
            "warmups",
            "repetitions",
        ):
            positive(getattr(parsed, name), f"evaluation.{name}", allow_zero=name == "warmups")
        registered_output_prompts(parsed.prompt_set, parsed.prompt_set_sha256)
        if (
            not isinstance(raw_limits, list)
            or tuple(raw_limits) != OUTPUT_ORACLE_SPAN_LIMITS
            or any(not isinstance(item, int) or isinstance(item, bool) for item in raw_limits)
        ):
            raise ValueError("evaluation.oracle_span_limits must be [1, 2, 4, 8]")
        seeds = tuple(corpus.seed for corpus in corpora.values())
        if len(seeds) != len(set(seeds)):
            raise ValueError("output corpus seeds must be distinct")
        return parsed


@final
@dataclass(frozen=True, slots=True)
class OutputGateSpec:
    minimum_direct_feedback_equality: float = 1.0
    maximum_invalid_events: int = 0
    minimum_valid_non_empty_termination: float = 1.0
    minimum_control_prompt_coverage: float = 0.25
    minimum_control_precision: float = 1.0
    minimum_control_recall: float = 1.0
    minimum_stop_precision: float = 1.0
    minimum_stop_recall: float = 1.0
    minimum_native_tokens_per_attempted_macro_step: float = 1.10
    minimum_rollout_event_agreement: float = 0.50
    maximum_candidate_reference_state_ratio: float = 0.50

    @classmethod
    def parse(cls, value: object) -> Self:
        parsed = parse_defaults(cls, value, "gates")
        for name, item in asdict(parsed).items():
            positive(
                item,
                f"gates.{name}",
                allow_zero=name == "maximum_invalid_events",
            )
        bounded = (
            "minimum_direct_feedback_equality",
            "minimum_valid_non_empty_termination",
            "minimum_control_prompt_coverage",
            "minimum_control_precision",
            "minimum_control_recall",
            "minimum_stop_precision",
            "minimum_stop_recall",
            "minimum_rollout_event_agreement",
        )
        if any(getattr(parsed, name) > 1 for name in bounded):
            raise ValueError("output probability gates must not exceed one")
        return parsed
