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
    filter_checkpoint_tensor,
    load_filtered_state,
)
from continuous_tokenizer.runtime.tensors import tensor_bytes


@final
class OmittedInputEmbedding(nn.Module):
    """Parameter-free placeholder for models that receive only inputs_embeds."""

    def forward(self, input_ids: Tensor) -> Tensor:
        del input_ids
        raise RuntimeError("the filtered model accepts inputs_embeds only")


def filter_input_embedding_checkpoint(
    source: Path,
    output: Path,
    input_tensor: str,
) -> dict[str, Any]:
    result = filter_checkpoint_tensor(source, output, input_tensor)
    result["input_tensor"] = result.pop("tensor")
    result["input_tensor_absent"] = result.pop("tensor_absent")
    return result


def load_filtered_causal_lm(directory: Path) -> tuple[nn.Module, dict[str, Any]]:
    config = AutoConfig.from_pretrained(directory)
    if tie_word_embeddings(config):
        raise ValueError("a tied input/output table cannot be omitted in the input-only experiment")
    with torch.device("meta"):
        model = build_model_from_config(config)
    model.set_input_embeddings(OmittedInputEmbedding())
    model_state, loaded_tensor_count = load_filtered_state(model, directory)
    model.eval()
    loaded_bytes = sum(tensor_bytes(tensor) for tensor in model_state.values())
    return model, {
        "loaded_tensor_count": loaded_tensor_count,
        "loaded_bytes": loaded_bytes,
        "input_module_parameter_count": sum(parameter.numel() for parameter in model.get_input_embeddings().parameters()),
        "meta_tensor_count": 0,
    }


def _measure_filtered_worker(directory: str, result_path: str) -> None:
    try:
        model, metrics = load_filtered_causal_lm(Path(directory))
        process = psutil.Process()
        write_json_atomic(
            Path(result_path),
            {
                **metrics,
                "rss_bytes": process.memory_info().rss,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "physical_input_table_omission_proven": True,
                "model_class": type(model).__name__,
            },
        )
    except Exception as error:
        write_json_atomic(
            Path(result_path),
            {
                "physical_input_table_omission_proven": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def prove_physical_input_table_omission(
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
    package = filter_input_embedding_checkpoint(source, output / "filtered-model", input_tensor)
    result_path = output / "physical-load.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_measure_filtered_worker,
        args=(str(output / "filtered-model"), str(result_path)),
    )
    process.start()
    process.join()
    if not result_path.is_file():
        raise RuntimeError(f"filtered model process exited with code {process.exitcode}")
    measurement = json.loads(result_path.read_text(encoding="utf-8"))
    if process.exitcode != 0 or not measurement["physical_input_table_omission_proven"]:
        raise RuntimeError(f"filtered model load failed: {measurement}")
    return {"package": package, "load": measurement, "process_isolated": True}
