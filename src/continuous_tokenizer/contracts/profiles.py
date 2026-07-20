from __future__ import annotations

from dataclasses import dataclass
from typing import Final, final

DIAGNOSTIC_PROFILE_NAME: Final = "small"
CAMPAIGN_PROFILE_NAME: Final = "large"
PROJECTION_DIMENSION_CAP: Final = 32_768
EFFICIENCY_BATCH_SIZES: Final = (256, 512, 1024)
EFFICIENCY_PROJECTION_MULTIPLIERS: Final = (2, 4, 8)
EFFICIENCY_MUON_NS_STEPS: Final = (3, 5)


@final
@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    local_dim: int
    projection_multiplier: int
    encoder_layers: int
    decoder_layers: int
    query_heads: int
    key_value_heads: int
    feedforward_dim: int

    def __post_init__(self) -> None:
        if self.key_value_heads != 2:
            raise ValueError("profiles require exactly two key/value heads")

    def projection_dim(self, embedding_dim: int) -> int:
        return min(
            PROJECTION_DIMENSION_CAP,
            max(self.local_dim, self.projection_multiplier * embedding_dim),
        )


PROFILES: Final = (
    Profile(
        name=DIAGNOSTIC_PROFILE_NAME,
        local_dim=128,
        projection_multiplier=4,
        encoder_layers=1,
        decoder_layers=1,
        query_heads=4,
        key_value_heads=2,
        feedforward_dim=256,
    ),
    Profile(
        name=CAMPAIGN_PROFILE_NAME,
        local_dim=256,
        projection_multiplier=8,
        encoder_layers=2,
        decoder_layers=1,
        query_heads=4,
        key_value_heads=2,
        feedforward_dim=512,
    ),
)
TRAINING_PROFILE_NAMES: Final = tuple(profile.name for profile in PROFILES)


def profile_named(name: str) -> Profile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    choices = ", ".join(TRAINING_PROFILE_NAMES)
    raise ValueError(f"unknown profile {name!r}; expected one of: {choices}")
