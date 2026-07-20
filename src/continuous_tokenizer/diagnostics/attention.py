from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, final

import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from continuous_tokenizer.artifacts.store import write_json_atomic, write_text_atomic
from continuous_tokenizer.backbone.assets import ModelAssets, load_frozen_causal_lm
from continuous_tokenizer.input.adapter import (
    InputEmbeddingAdapter,
    InputPosition,
    SegmentationAlignment,
)
from continuous_tokenizer.input.segmentation import EncodedSpan
from continuous_tokenizer.runtime.device import module_dtype, resolve_model_device
from continuous_tokenizer.runtime.tensors import parameter_fingerprint

type HtmlRenderer = Callable[[tuple[Tensor, ...], tuple[str, ...]], str]


@final
@dataclass(frozen=True, slots=True)
class AttentionOptions:
    output_dir: Path
    text: str
    max_tokens: int = 64
    segmentation_alignment: SegmentationAlignment = "arbitrary"

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("attention text must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("maximum token count must be positive")
        if self.segmentation_alignment not in {"aligned", "arbitrary"}:
            raise ValueError("segmented alignment must be aligned or arbitrary")


@final
@dataclass(frozen=True, slots=True)
class AttentionRuntime:
    device: torch.device | None = None
    frozen_model: nn.Module | None = None
    html_renderer: HtmlRenderer | None = None


def _byte_label(data: bytes) -> str:
    text = repr(data.decode("utf-8", errors="backslashreplace"))
    if len(text) > 30:
        text = f"{text[:27]}..."
    hexadecimal = data.hex(" ")
    if len(hexadecimal) > 32:
        hexadecimal = f"{hexadecimal[:29]}..."
    return f"{text} [{hexadecimal}]"


def _position_label(position: InputPosition) -> str:
    if isinstance(position, EncodedSpan):
        return _byte_label(position.data)
    return f"<control:{position.token_id}>"


def _native_labels(assets: ModelAssets, token_ids: Sequence[int]) -> tuple[str, ...]:
    labels: list[str] = []
    for token_id in token_ids:
        value = assets.vocabulary.payload_for(token_id)
        labels.append(_byte_label(value) if value is not None else f"<control:{token_id}>")
    return tuple(labels)


def _attentions(outputs: Any, positions: int) -> tuple[Tensor, ...]:
    values = getattr(outputs, "attentions", None)
    if not values or any(value is None for value in values):
        raise RuntimeError("the model did not return attention weights")
    tensors = tuple(value.detach().float().cpu().contiguous() for value in values)
    for tensor in tensors:
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("attention tensors must have shape [1, heads, query, key]")
        if tensor.shape[-2:] != (positions, positions):
            raise ValueError("attention dimensions do not match the input positions")
    return tensors


def bertviz_html(attentions: tuple[Tensor, ...], labels: tuple[str, ...]) -> str:
    try:
        model_view = import_module("bertviz").model_view
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional UI dependencies with `uv sync --group ui`") from error
    rendered = model_view(
        attention=attentions,
        tokens=labels,
        prettify_tokens=False,
        display_mode="light",
        html_action="return",
    )
    html = getattr(rendered, "data", None)
    if not isinstance(html, str):
        raise RuntimeError("BertViz did not return an HTML document")
    return html


def _layer_tensors(attentions: tuple[Tensor, ...]) -> dict[str, Tensor]:
    return {f"layer_{index:03d}": value for index, value in enumerate(attentions)}


def _mode_metadata(labels: tuple[str, ...], attentions: tuple[Tensor, ...], tensor_path: str, html_path: str) -> dict[str, Any]:
    return {
        "labels": labels,
        "positions": len(labels),
        "layers": len(attentions),
        "heads": int(attentions[0].shape[1]),
        "tensor_path": tensor_path,
        "html_path": html_path,
    }


def _attention_report(metadata: Mapping[str, Any]) -> str:
    native = metadata["modes"]["native"]
    segmented = metadata["modes"]["segmented"]
    lines = [
        "# BertViz Attention Diagnostic",
        "",
        "This optional eager-attention capture is diagnostic only. It is excluded from model quality, memory, compute, and latency claims.",
        "",
        f"- Model: `{metadata['model']['id']}`",
        f"- Revision: `{metadata['model']['revision']}`",
        f"- Checkpoint: `{metadata['checkpoint']['sha256']}`",
        f"- Input SHA-256: `{metadata['input']['sha256']}`",
        f"- Attention backend: `{metadata['attention_backend']}`",
        "",
        "## Views",
        "",
        "| Input path | Positions | Layers | Heads | Interactive view |",
        "|---|---:|---:|---:|---|",
    ]
    for name, mode in (("Native", native), ("Segmented", segmented)):
        lines.append(f"| {name} | {mode['positions']} | {mode['layers']} | {mode['heads']} | [Open BertViz]({mode['html_path']}) |")
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def capture_attention_artifact(
    assets: ModelAssets,
    checkpoint: Path,
    options: AttentionOptions,
    runtime: AttentionRuntime | None = None,
) -> dict[str, Any]:
    runtime = AttentionRuntime() if runtime is None else runtime
    html_renderer = bertviz_html if runtime.html_renderer is None else runtime.html_renderer
    artifact_dir = options.output_dir / "attention"
    if artifact_dir.exists():
        raise FileExistsError(f"attention artifact already exists: {artifact_dir}")
    selected_device = resolve_model_device(runtime.device, runtime.frozen_model)
    loaded = InputEmbeddingAdapter.from_checkpoint(
        assets,
        checkpoint,
        device=selected_device,
    )
    adapter = loaded.adapter
    fingerprint = loaded.fingerprint
    codec = adapter.codec
    model = load_frozen_causal_lm(assets, selected_device, output_attentions=True) if runtime.frozen_model is None else runtime.frozen_model
    model.eval()
    before = parameter_fingerprint(model)

    token_ids = tuple(assets.tokenizer.encode(options.text, add_special_tokens=False))
    if not token_ids:
        raise ValueError("the input text produced no model tokens")
    if len(token_ids) > options.max_tokens:
        raise ValueError(f"attention input has {len(token_ids)} tokens; maximum is {options.max_tokens}")
    device_ids = torch.tensor([token_ids], dtype=torch.long, device=selected_device)
    native_outputs = model(
        input_ids=device_ids,
        use_cache=False,
        output_attentions=True,
    )
    native_labels = _native_labels(assets, token_ids)
    native_attention = _attentions(native_outputs, len(native_labels))

    segmented = adapter.encode_token_ids(
        token_ids,
        mode="segmented",
        cache=codec.encoding_cache,
        alignment=options.segmentation_alignment,
    )
    segmented_outputs = model(
        inputs_embeds=segmented.embeddings.to(
            device=selected_device,
            dtype=module_dtype(model),
        ).unsqueeze(0),
        position_ids=segmented.position_ids.to(selected_device).unsqueeze(0),
        use_cache=False,
        output_attentions=True,
    )
    segmented_labels = tuple(_position_label(position) for position in segmented.positions)
    segmented_attention = _attentions(segmented_outputs, len(segmented_labels))
    after = parameter_fingerprint(model)
    if before != after:
        raise RuntimeError("frozen model parameters changed during attention capture")

    metadata = {
        "kind": "attention_diagnostic",
        "diagnostic_only": True,
        "performance_comparable": False,
        "attention_backend": "eager" if runtime.frozen_model is None else "provided",
        "model": {
            "id": assets.model_id,
            "revision": assets.revision,
            "parameter_fingerprint": before,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": fingerprint},
        "input": {
            "text": options.text,
            "sha256": hashlib.sha256(options.text.encode("utf-8")).hexdigest(),
            "native_tokens": len(token_ids),
        },
        "options": {**asdict(options), "output_dir": str(options.output_dir)},
        "modes": {
            "native": _mode_metadata(
                native_labels,
                native_attention,
                "native.safetensors",
                "native.html",
            ),
            "segmented": _mode_metadata(
                segmented_labels,
                segmented_attention,
                "segmented.safetensors",
                "segmented.html",
            ),
        },
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = options.output_dir / f".attention.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"temporary attention artifact already exists: {temporary}")
    temporary.mkdir()
    try:
        save_file(_layer_tensors(native_attention), temporary / "native.safetensors")
        save_file(_layer_tensors(segmented_attention), temporary / "segmented.safetensors")
        write_text_atomic(temporary / "native.html", html_renderer(native_attention, native_labels))
        write_text_atomic(temporary / "segmented.html", html_renderer(segmented_attention, segmented_labels))
        write_text_atomic(temporary / "report.md", _attention_report(metadata))
        write_json_atomic(temporary / "metadata.json", metadata)
        Path(temporary).replace(artifact_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata
