from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast, final

from continuous_tokenizer.contracts.output import (
    OUTPUT_FIDELITY_PROMPT_SET,
    OUTPUT_FIDELITY_PROMPT_SET_SHA256,
    registered_output_prompts,
)
from continuous_tokenizer.contracts.parsing import (
    exact_fields,
    is_lowercase_sha256,
    non_empty_string,
)

type DeploymentMode = Literal["input_only", "output_only"]
type DeploymentDevice = Literal["cpu", "mps", "cuda"]


@final
@dataclass(frozen=True, slots=True)
class DeploymentSpec:
    name: str
    mode: DeploymentMode
    device: DeploymentDevice
    quality_run: Path
    quality_manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    prompt_set: str = OUTPUT_FIDELITY_PROMPT_SET
    prompt_set_sha256: str = OUTPUT_FIDELITY_PROMPT_SET_SHA256
    repetitions: int = 3
    max_steps: int = 16
    max_bytes: int = 1024

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        expected = {
            "name",
            "mode",
            "device",
            "quality_run",
            "quality_manifest_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "prompt_set",
            "prompt_set_sha256",
            "repetitions",
            "max_steps",
            "max_bytes",
        }
        exact_fields(values, expected, "deployment")
        mode = values.get("mode")
        if mode not in {"input_only", "output_only"}:
            raise ValueError("deployment.mode must be input_only or output_only")
        device = values.get("device")
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("deployment.device must be cpu, mps, or cuda")
        repetitions = values.get("repetitions")
        if repetitions != 3:
            raise ValueError("deployment.repetitions must be exactly 3")
        max_steps = values.get("max_steps")
        max_bytes = values.get("max_bytes")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
            or not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes < 1
        ):
            raise ValueError("deployment inference limits must be positive integers")
        quality_hash = non_empty_string(
            values,
            "quality_manifest_sha256",
            "deployment",
        )
        checkpoint_hash = non_empty_string(
            values,
            "checkpoint_sha256",
            "deployment",
        )
        if not is_lowercase_sha256(quality_hash) or not is_lowercase_sha256(
            checkpoint_hash,
        ):
            raise ValueError("deployment input hashes must be lowercase SHA-256")
        prompt_set = non_empty_string(values, "prompt_set", "deployment")
        prompt_hash = non_empty_string(
            values,
            "prompt_set_sha256",
            "deployment",
        )
        registered_output_prompts(prompt_set, prompt_hash)
        return cls(
            name=non_empty_string(values, "name", "deployment"),
            mode=cast(DeploymentMode, mode),
            device=cast(DeploymentDevice, device),
            quality_run=(path.parent / non_empty_string(values, "quality_run", "deployment")).resolve(),
            quality_manifest_sha256=quality_hash,
            checkpoint=(path.parent / non_empty_string(values, "checkpoint", "deployment")).resolve(),
            checkpoint_sha256=checkpoint_hash,
            prompt_set=prompt_set,
            prompt_set_sha256=prompt_hash,
            repetitions=repetitions,
            max_steps=max_steps,
            max_bytes=max_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["quality_run"] = str(self.quality_run)
        values["checkpoint"] = str(self.checkpoint)
        return values
