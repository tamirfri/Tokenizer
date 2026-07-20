from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, final

import psutil
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.profiler import record_function

from continuous_tokenizer.runtime.environment import mps_memory_bytes

MUON_ADJUST_LR_FN: Final = "match_rms_adamw"
DEFAULT_MUON_NS_STEPS: Final = 5
MAX_GRADIENT_NORM: Final = 1.0
_GRADIENT_NORM_METRICS: Final = (
    "shared_preclip_gradient_norm",
    "muon_preclip_gradient_norm",
    "adamw_preclip_gradient_norm",
)
_MEMORY_METRICS: Final = (
    "cpu_rss_bytes",
    "mps_allocated_bytes",
    "mps_driver_allocated_bytes",
)


@final
@dataclass(slots=True)
class TokenizerOptimizers:
    values: tuple[torch.optim.Optimizer, ...]
    trainable_parameters: tuple[Parameter, ...]
    muon_parameters: tuple[Parameter, ...]
    adamw_parameters: tuple[Parameter, ...]
    _telemetry: dict[str, float] = field(default_factory=dict)
    _process: psutil.Process = field(default_factory=psutil.Process, repr=False)

    @staticmethod
    def _gradient_norm(parameters: tuple[Parameter, ...], reference: Tensor) -> Tensor:
        norms = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
        if not norms:
            return reference.new_zeros((), dtype=torch.float32)
        return torch.stack(norms).norm()

    def optimize(self, loss: Tensor) -> dict[str, float]:
        with record_function("tokenizer.optimizer_zero_grad"):
            for optimizer in self.values:
                optimizer.zero_grad(set_to_none=True)
        with record_function("tokenizer.backward"):
            loss.backward()
        muon_norm = self._gradient_norm(self.muon_parameters, loss)
        adamw_norm = self._gradient_norm(self.adamw_parameters, loss)
        with record_function("tokenizer.gradient_clipping"):
            shared_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters,
                MAX_GRADIENT_NORM,
            )
        for optimizer in self.values:
            with record_function(f"tokenizer.{type(optimizer).__name__.lower()}_step"):
                optimizer.step()
        shared_value, muon_value, adamw_value = torch.stack((shared_norm.float(), muon_norm, adamw_norm)).detach().cpu().tolist()
        device = self.trainable_parameters[0].device
        cpu_rss_bytes = float(self._process.memory_info().rss)
        mps_allocated_bytes, mps_driver_allocated_bytes = mps_memory_bytes(device)
        observation = {
            "shared_preclip_gradient_norm": shared_value,
            "muon_preclip_gradient_norm": muon_value,
            "adamw_preclip_gradient_norm": adamw_value,
            "cpu_rss_bytes": cpu_rss_bytes,
            "mps_allocated_bytes": float(mps_allocated_bytes),
            "mps_driver_allocated_bytes": float(mps_driver_allocated_bytes),
        }
        self._record_telemetry(observation)
        return observation

    def _record_telemetry(self, observation: dict[str, float]) -> None:
        self._telemetry["steps"] = self._telemetry.get("steps", 0.0) + 1.0
        for name in _GRADIENT_NORM_METRICS:
            self._telemetry[f"{name}_sum"] = self._telemetry.get(f"{name}_sum", 0.0) + observation[name]
            self._telemetry[f"{name}_maximum"] = max(
                self._telemetry.get(f"{name}_maximum", 0.0),
                observation[name],
            )
        for name in _MEMORY_METRICS:
            self._telemetry[f"peak_{name}"] = max(
                self._telemetry.get(f"peak_{name}", 0.0),
                observation[name],
            )

    def epoch_telemetry(self, *, reset: bool = False) -> dict[str, int | float]:
        steps = int(self._telemetry.get("steps", 0.0))
        result: dict[str, int | float] = {
            "optimizer_steps": steps,
            "telemetry_scalar_transfers": steps,
            "telemetry_scalar_transfers_avoided": steps * 2,
        }
        for name in _GRADIENT_NORM_METRICS:
            result[f"mean_{name}"] = self._telemetry.get(f"{name}_sum", 0.0) / max(steps, 1)
            result[f"maximum_{name}"] = self._telemetry.get(
                f"{name}_maximum",
                0.0,
            )
        for name in _MEMORY_METRICS:
            peak_name = f"peak_{name}"
            result[peak_name] = int(self._telemetry.get(peak_name, 0.0))
        if reset:
            self._telemetry.clear()
        return result

    def state_dict(self) -> tuple[dict, ...]:
        return tuple(optimizer.state_dict() for optimizer in self.values)

    def load_state_dict(self, values: tuple[dict, ...]) -> None:
        if len(values) != len(self.values):
            raise ValueError("resume optimizer state count does not match")
        for optimizer, state in zip(self.values, values, strict=True):
            optimizer.load_state_dict(state)

    def resume_state(self) -> dict[str, Any]:
        return {
            "layout": self.layout(),
            "states": self.state_dict(),
            "telemetry": dict(self._telemetry),
        }

    def load_resume_state(self, value: dict[str, Any]) -> None:
        if value.get("layout") != self.layout():
            raise ValueError("resume optimizer layout does not match the active dual optimizers")
        states = value.get("states")
        if not isinstance(states, tuple):
            raise ValueError("resume optimizer states must be a tuple")
        self.load_state_dict(states)
        telemetry = value.get("telemetry")
        if not isinstance(telemetry, dict) or any(not isinstance(name, str) or not isinstance(metric, int | float) for name, metric in telemetry.items()):
            raise ValueError("resume optimizer telemetry must contain numeric metrics")
        self._telemetry = {name: float(metric) for name, metric in telemetry.items()}

    def layout(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                "groups": tuple(
                    tuple(
                        {
                            "shape": tuple(parameter.shape),
                            "dtype": str(parameter.dtype),
                        }
                        for parameter in group["params"]
                    )
                    for group in optimizer.param_groups
                ),
            }
            for optimizer in self.values
        )


def build_tokenizer_optimizers(
    muon_parameters: tuple[Parameter, ...],
    adamw_parameters: tuple[Parameter, ...],
    *,
    learning_rate: float,
    weight_decay: float,
    muon_ns_steps: int = DEFAULT_MUON_NS_STEPS,
) -> TokenizerOptimizers:
    trainable_parameters = (*muon_parameters, *adamw_parameters)
    if not trainable_parameters:
        raise ValueError("tokenizer optimizer requires trainable parameters")
    muon_ids = {id(parameter) for parameter in muon_parameters}
    adamw_ids = {id(parameter) for parameter in adamw_parameters}
    if muon_ids & adamw_ids:
        raise ValueError("Muon and AdamW parameter groups must be disjoint")
    if any(parameter.ndim != 2 for parameter in muon_parameters):
        raise ValueError("Muon parameters must be 2D matrices")
    if muon_ns_steps < 1:
        raise ValueError("Muon Newton-Schulz steps must be positive")
    optimizers: list[torch.optim.Optimizer] = []
    if muon_parameters:
        optimizers.append(
            torch.optim.Muon(
                muon_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=MUON_ADJUST_LR_FN,
            )
        )
    if adamw_parameters:
        optimizers.append(
            torch.optim.AdamW(
                adamw_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        )
    return TokenizerOptimizers(
        tuple(optimizers),
        trainable_parameters,
        muon_parameters,
        adamw_parameters,
    )


def optimizer_metadata(
    muon_ns_steps: int = DEFAULT_MUON_NS_STEPS,
) -> dict[str, str | int]:
    return {
        "hidden_matrix_parameters": torch.optim.Muon.__name__,
        "output_and_non_matrix_parameters": torch.optim.AdamW.__name__,
        "muon_adjust_lr_fn": MUON_ADJUST_LR_FN,
        "muon_ns_steps": muon_ns_steps,
    }
