from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal, Self, final

from continuous_tokenizer.contracts.parsing import parse_defaults, positive
from continuous_tokenizer.contracts.profiles import (
    CAMPAIGN_PROFILE_NAME,
    EFFICIENCY_MUON_NS_STEPS,
    EFFICIENCY_PROJECTION_MULTIPLIERS,
    TRAINING_PROFILE_NAMES,
)

INPUT_STRATEGIES: Final = (
    "candidate_selection",
    "reconstruction_only",
    "token_aligned_distillation",
    "arbitrary_boundary_distillation",
)
type InputStrategy = Literal[
    "candidate_selection",
    "reconstruction_only",
    "token_aligned_distillation",
    "arbitrary_boundary_distillation",
]


@final
@dataclass(frozen=True, slots=True)
class InputTrainingSpec:
    profile: str = CAMPAIGN_PROFILE_NAME
    strategy: InputStrategy = "candidate_selection"
    projection_multiplier: int = 0
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    muon_ns_steps: int = 5
    vocabulary_epochs: int = 10
    reconstruction_epochs: int = 1
    reconstruction_samples: int = 2_048
    reconstruction_vocabulary_fraction: float = 0.75
    validation_bytes: int = 2_048
    patience: int = 2
    evaluation_interval: int = 2
    distillation_epochs: int = 0
    distillation_windows: int = 0
    distillation_prompt_tokens: int = 16
    distillation_continuation_tokens: int = 4

    @classmethod
    def parse(cls, value: object) -> Self:
        parsed = parse_defaults(cls, value, "training")
        if parsed.profile not in TRAINING_PROFILE_NAMES:
            choices = ", ".join(TRAINING_PROFILE_NAMES)
            raise ValueError(f"training.profile must be one of: {choices}")
        if parsed.strategy not in INPUT_STRATEGIES:
            raise ValueError(f"training.strategy must be one of: {', '.join(INPUT_STRATEGIES)}")
        if parsed.projection_multiplier not in {0, *EFFICIENCY_PROJECTION_MULTIPLIERS}:
            raise ValueError("training.projection_multiplier must be 0, 2, 4, or 8")
        positive(parsed.batch_size, "training.batch_size")
        positive(parsed.learning_rate, "training.learning_rate")
        positive(parsed.weight_decay, "training.weight_decay", allow_zero=True)
        if parsed.muon_ns_steps not in EFFICIENCY_MUON_NS_STEPS:
            raise ValueError("training.muon_ns_steps must be 3 or 5")
        positive(parsed.vocabulary_epochs, "training.vocabulary_epochs", allow_zero=True)
        positive(parsed.reconstruction_epochs, "training.reconstruction_epochs", allow_zero=True)
        positive(parsed.reconstruction_samples, "training.reconstruction_samples", allow_zero=True)
        if not 0.0 < parsed.reconstruction_vocabulary_fraction < 1.0:
            raise ValueError("training.reconstruction_vocabulary_fraction must be between zero and one")
        positive(parsed.validation_bytes, "training.validation_bytes")
        positive(parsed.patience, "training.patience")
        positive(parsed.evaluation_interval, "training.evaluation_interval")
        positive(parsed.distillation_epochs, "training.distillation_epochs", allow_zero=True)
        positive(
            parsed.distillation_windows,
            "training.distillation_windows",
            allow_zero=True,
        )
        positive(parsed.distillation_prompt_tokens, "training.distillation_prompt_tokens")
        positive(
            parsed.distillation_continuation_tokens,
            "training.distillation_continuation_tokens",
        )
        return parsed


@final
@dataclass(frozen=True, slots=True)
class InputEvaluationSpec:
    batch_size: int = 8
    samples: int = 16
    prompt_tokens: int = 64
    continuation_tokens: int = 16
    generation_samples: int = 2
    max_new_tokens: int = 16
    warmups: int = 1
    repetitions: int = 2
    performance_prompts: int = 2
    tokenizer_repetitions: int = 2
    retrieval_queries: int = 128
    max_test_bytes: int = 2_048
    calibration_maximum_kl: float = 1e-4
    calibration_maximum_nll_delta: float = 1e-3
    calibration_minimum_top1_agreement: float = 1.0
    calibration_maximum_logit_error: float = 1e-2

    @classmethod
    def parse(cls, value: object) -> Self:
        parsed = parse_defaults(cls, value, "evaluation")
        for name, item in asdict(parsed).items():
            positive(
                item,
                f"evaluation.{name}",
                allow_zero=name in {"generation_samples", "warmups"},
            )
        if parsed.batch_size < 2 or parsed.samples < 2:
            raise ValueError("input evaluation requires batch_size and samples of at least two")
        if parsed.calibration_minimum_top1_agreement > 1:
            raise ValueError("evaluation calibration top-1 agreement must not exceed one")
        return parsed


@final
@dataclass(frozen=True, slots=True)
class InputGateSpec:
    maximum_normalized_rmse: float = 1e-2
    minimum_cosine_p01: float = 0.999
    minimum_cosine_p50: float = 0.9999
    minimum_native_tokens_per_continuous_token: float = 1.10
    maximum_candidate_reference_state_ratio: float = 0.50
    maximum_segmented_mean_kl: float = 0.10
    maximum_segmented_nll_delta: float = 0.10
    minimum_segmented_top1_agreement: float = 0.90
    minimum_segmented_generation_byte_similarity: float = 0.50

    @classmethod
    def parse(cls, value: object) -> Self:
        parsed = parse_defaults(cls, value, "gates")
        for name, item in asdict(parsed).items():
            positive(item, f"gates.{name}")
        if parsed.minimum_segmented_top1_agreement > 1:
            raise ValueError("segmented top-1 agreement gate must not exceed one")
        if parsed.minimum_segmented_generation_byte_similarity > 1:
            raise ValueError("segmented generation byte-similarity gate must not exceed one")
        return parsed
