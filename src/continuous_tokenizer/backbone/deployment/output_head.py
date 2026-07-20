from __future__ import annotations

import json
import multiprocessing
import resource
from pathlib import Path
from typing import Any, final

import psutil
import torch
from huggingface_hub import snapshot_download
from torch import Tensor, nn
from transformers import AutoConfig

from continuous_tokenizer.artifacts.store import write_json_atomic
from continuous_tokenizer.backbone.config import build_model_from_config, tie_word_embeddings
from continuous_tokenizer.backbone.deployment.shared import (
    checkpoint_index,
    filter_checkpoint_tensor,
    load_filtered_state,
)
from continuous_tokenizer.runtime.tensors import tensor_bytes


@final
class OmittedOutputHead(nn.Module):
    """Parameter-free proof that output-only inference does not require vocabulary logits."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        del hidden_states
        raise RuntimeError("the filtered output-only model has no native vocabulary head")


def output_tensor_name(source: Path, input_tensor: str) -> str:
    names = set(checkpoint_index(source)["weight_map"])
    preferred = (
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.language_model.lm_head.weight",
        "output.weight",
    )
    candidate = next((name for name in preferred if name in names and name != input_tensor), None)
    if candidate is None:
        matches = sorted(name for name in names if name != input_tensor and name.endswith(("lm_head.weight", "output.weight")))
        if len(matches) != 1:
            raise ValueError("could not identify a separate native output-head tensor")
        candidate = matches[0]
    return candidate


def load_filtered_output_backbone(directory: Path) -> tuple[nn.Module, dict[str, Any]]:
    config = AutoConfig.from_pretrained(directory)
    if tie_word_embeddings(config):
        raise ValueError("a tied input/output table cannot omit the native output head")
    with torch.device("meta"):
        model = build_model_from_config(config)
    setter = getattr(model, "set_output_embeddings", None)
    if not callable(setter):
        raise ValueError("model does not expose replaceable output embeddings")
    setter(OmittedOutputHead())
    model_state, loaded_tensor_count = load_filtered_state(model, directory)
    model.eval()
    return model, {
        "loaded_tensor_count": loaded_tensor_count,
        "loaded_bytes": sum(tensor_bytes(tensor) for tensor in model_state.values()),
        "output_module_parameter_count": sum(parameter.numel() for parameter in model.get_output_embeddings().parameters()),
        "meta_tensor_count": 0,
    }


def _measure_filtered_output_worker(directory: str, result_path: str) -> None:
    try:
        model, metrics = load_filtered_output_backbone(Path(directory))
        process = psutil.Process()
        write_json_atomic(
            Path(result_path),
            {
                **metrics,
                "rss_bytes": process.memory_info().rss,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "physical_output_head_omission_proven": True,
                "model_class": type(model).__name__,
            },
        )
    except Exception as error:
        write_json_atomic(
            Path(result_path),
            {
                "physical_output_head_omission_proven": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def prove_physical_output_head_omission(
    model_id: str,
    revision: str,
    input_tensor: str,
    output: Path,
) -> dict[str, Any]:
    source = Path(
        snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=["config.json", "model.safetensors*", "model-*.safetensors"],
        )
    )
    output_tensor = output_tensor_name(source, input_tensor)
    package = filter_checkpoint_tensor(source, output / "filtered-model", output_tensor)
    package["output_tensor"] = package.pop("tensor")
    package["output_tensor_absent"] = package.pop("tensor_absent")
    result_path = output / "physical-load.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_measure_filtered_output_worker,
        args=(str(output / "filtered-model"), str(result_path)),
    )
    process.start()
    process.join()
    if not result_path.is_file():
        raise RuntimeError(f"filtered output process exited with code {process.exitcode}")
    measurement = json.loads(result_path.read_text(encoding="utf-8"))
    if process.exitcode != 0 or not measurement["physical_output_head_omission_proven"]:
        raise RuntimeError(f"filtered output model load failed: {measurement}")
    return {
        "package": package,
        "load": measurement,
        "process_isolated": True,
    }
