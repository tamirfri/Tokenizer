from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    local_dim: int
    encoder_layers: int
    decoder_layers: int
    heads: int
    feedforward_dim: int


PROFILES = (
    Profile("small", 128, 1, 1, 4, 512),
    Profile("medium", 256, 2, 2, 8, 1024),
    Profile("large", 512, 4, 4, 8, 2048),
)


def profile_named(name: str) -> Profile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    choices = ", ".join(profile.name for profile in PROFILES)
    raise ValueError(f"unknown profile {name!r}; expected one of: {choices}")
