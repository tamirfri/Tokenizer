from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast, final

import psutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    write_evidence_manifest,
)
from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.artifacts.manifest import load_verified_run_manifest
from continuous_tokenizer.artifacts.store import (
    RunDirectory,
    load_json_object,
    write_json_atomic,
)
from continuous_tokenizer.backbone.config import (
    input_table_is_removable,
    model_loader,
    tie_word_embeddings,
)
from continuous_tokenizer.backbone.deployment.input_table import (
    filter_input_embedding_checkpoint,
    load_filtered_causal_lm,
)
from continuous_tokenizer.backbone.deployment.output_head import (
    load_filtered_output_backbone,
    output_tensor_name,
)
from continuous_tokenizer.backbone.deployment.shared import (
    filter_checkpoint_tensor,
)
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary, inspect_tokenizer
from continuous_tokenizer.codec.checkpoints import (
    load_checkpoint,
    load_output_checkpoint,
)
from continuous_tokenizer.contracts.deployment import DeploymentSpec
from continuous_tokenizer.contracts.manifest import RunManifest
from continuous_tokenizer.contracts.output import registered_output_prompts
from continuous_tokenizer.input.adapter import InputEmbeddingAdapter
from continuous_tokenizer.output.generation import OutputOnlyGenerator
from continuous_tokenizer.runtime.environment import peak_rss_bytes
from continuous_tokenizer.runtime.tensors import module_bytes, tensor_bytes

type DeploymentWorker = Callable[[Mapping[str, Any]], dict[str, Any]]
type _DeploymentVariant = Literal["native", "candidate"]


@final
@dataclass(frozen=True, slots=True)
class _MeasurementContext:
    spec: DeploymentSpec
    manifest: RunManifest
    package: Mapping[str, Any]
    output: RunDirectory
    prompt_ids: tuple[tuple[int, ...], ...]
    stop_ids: tuple[int, ...]


def _tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().to("cpu").contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def _memory(device: torch.device) -> dict[str, int]:
    return {
        "rss_bytes": psutil.Process().memory_info().rss,
        "peak_rss_bytes": peak_rss_bytes(),
        "mps_allocated_bytes": (torch.mps.current_allocated_memory() if device.type == "mps" else 0),
        "mps_driver_bytes": (torch.mps.driver_allocated_memory() if device.type == "mps" else 0),
    }


def _load_native_model(directory: Path, device: torch.device) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(directory)
    model = model_loader(config).from_pretrained(directory)
    model.to(device)
    model.eval()
    return model


def _vocabulary(request: Mapping[str, Any]) -> ByteVocabulary:
    tokenizer = AutoTokenizer.from_pretrained(
        str(request["model_id"]),
        revision=str(request["model_revision"]),
    )
    config = AutoConfig.from_pretrained(str(request["source_directory"]))
    return inspect_tokenizer(
        tokenizer,
        embedding_rows=int(config.vocab_size),
    )


def _input_inference(
    model: torch.nn.Module,
    request: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    loaded = load_checkpoint(Path(str(request["checkpoint"])), device=device)
    adapter = InputEmbeddingAdapter(
        loaded.codec,
        _vocabulary(request),
        loaded.controls.ids,
        loaded.controls.embeddings,
        namespace=str(request["checkpoint_sha256"]),
    )
    prompt = tuple(int(value) for value in request["prompt_token_ids"])
    encoding = adapter.encode_token_ids(
        prompt,
        mode="segmented",
        cache=None,
        alignment="arbitrary",
    )
    outputs = model(
        inputs_embeds=encoding.embeddings.unsqueeze(0),
        position_ids=encoding.position_ids.unsqueeze(0),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    logits = outputs.logits
    return {
        "output_sha256": _tensor_sha256(logits),
        "hidden_sha256": _tensor_sha256(hidden),
        "loaded_candidate_state_bytes": (module_bytes(loaded.codec) + tensor_bytes(loaded.controls.ids) + tensor_bytes(loaded.controls.embeddings)),
    }


def _output_inference(
    model: torch.nn.Module,
    request: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    loaded = load_output_checkpoint(
        Path(str(request["checkpoint"])),
        device=device,
    )
    backbone = FrozenBackbone(model)
    prompt = tuple(int(value) for value in request["prompt_token_ids"])
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
    hidden = backbone.forward(input_ids=prompt_tensor, use_cache=True)
    generator = OutputOnlyGenerator(
        backbone,
        loaded.codec,
        _vocabulary(request),
        loaded.control_ids,
    )
    generated = generator.generate(
        prompt,
        stop_control_ids=frozenset(int(value) for value in request["stop_ids"]),
        max_macro_steps=int(request["max_steps"]),
        max_bytes=int(request["max_bytes"]),
    )
    controls = [event.token_id for event in generated.events if hasattr(event, "token_id")]
    output = json.dumps(
        {
            "data": generated.data.hex(),
            "controls": controls,
            "termination": generated.termination_reason,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "hidden_sha256": _tensor_sha256(hidden.last_hidden_state),
        "loaded_candidate_state_bytes": (module_bytes(loaded.codec) + tensor_bytes(loaded.control_ids)),
    }


def _worker(request: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device(str(request["device"]))
    variant = str(request["variant"])
    directory = Path(
        str(
            request["source_directory" if variant == "native" else "filtered_directory"],
        ),
    )
    if variant == "native":
        model = _load_native_model(directory, device)
        physical_absence = False
    elif request["mode"] == "input_only":
        model, load = load_filtered_causal_lm(directory)
        model.to(device)
        physical_absence = load["input_module_parameter_count"] == 0
    else:
        model, load = load_filtered_output_backbone(directory)
        model.to(device)
        physical_absence = load["output_module_parameter_count"] == 0
    post_load = _memory(device)
    inference = _input_inference(model, request, device) if request["mode"] == "input_only" else _output_inference(model, request, device)
    steady = _memory(device)
    return {
        "variant": variant,
        "physical_reference_tensor_absent": physical_absence,
        "loaded_tensor_bytes": (module_bytes(model) + int(inference.pop("loaded_candidate_state_bytes"))),
        "post_load": post_load,
        "steady": steady,
        "peak": {key: max(post_load[key], steady[key]) for key in post_load},
        **inference,
    }


def _worker_entry(request: dict[str, Any], result_path: str) -> None:
    try:
        write_json_atomic(Path(result_path), _worker(request))
    except Exception as error:
        write_json_atomic(
            Path(result_path),
            {
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _spawn_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(str(request["result_path"]))
    process = multiprocessing.get_context("spawn").Process(
        target=_worker_entry,
        args=(dict(request), str(result_path)),
    )
    process.start()
    process.join()
    if not result_path.is_file():
        raise RuntimeError(
            f"deployment worker exited with code {process.exitcode}",
        )
    result = dict(load_json_object(result_path))
    if process.exitcode != 0:
        raise RuntimeError(f"deployment worker failed: {result}")
    return result


def _package_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _prepare_packages(
    spec: DeploymentSpec,
    model_id: str,
    revision: str,
    embedding_tensor: str,
    output: RunDirectory,
) -> dict[str, Any]:
    source = Path(
        snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=[
                "config.json",
                "model.safetensors*",
                "model-*.safetensors",
            ],
        ),
    )
    filtered = output.path("filtered-model")
    if spec.mode == "input_only":
        package = filter_input_embedding_checkpoint(
            source,
            filtered,
            embedding_tensor,
        )
        absent = bool(package["input_tensor_absent"])
    else:
        output_tensor = output_tensor_name(source, embedding_tensor)
        package = filter_checkpoint_tensor(source, filtered, output_tensor)
        package["output_tensor"] = package.pop("tensor")
        package["output_tensor_absent"] = package.pop("tensor_absent")
        absent = bool(package["output_tensor_absent"])
    checkpoint_bytes = spec.checkpoint.stat().st_size
    return {
        "source_directory": source,
        "filtered_directory": filtered,
        "reference_tensor_absent_from_package": absent,
        "native_serialized_package_bytes": _package_bytes(source) + checkpoint_bytes,
        "candidate_serialized_package_bytes": _package_bytes(filtered) + checkpoint_bytes,
    }


def _quality_inputs(spec: DeploymentSpec) -> tuple[RunManifest, dict[str, Any]]:
    manifest_path = spec.quality_run / "manifest-final.json"
    if sha256_path(manifest_path) != spec.quality_manifest_sha256:
        raise ValueError("deployment quality manifest hash mismatch")
    if sha256_path(spec.checkpoint) != spec.checkpoint_sha256:
        raise ValueError("deployment checkpoint hash mismatch")
    manifest = load_verified_run_manifest(manifest_path)
    result = dict(load_json_object(spec.quality_run / "result.json"))
    if manifest.status != "passed" or manifest.mode != spec.mode or result.get("evidence_scope") != "final" or result.get("operational_status") != "completed":
        raise ValueError("deployment requires a sealed completed final quality run")
    checkpoint = manifest.artifacts.get("checkpoint")
    if checkpoint is None or spec.checkpoint.resolve() != (spec.quality_run / checkpoint).resolve():
        raise ValueError("deployment checkpoint is not sealed by the quality run")
    return manifest, result


def _applicability(spec: DeploymentSpec, config: Any) -> tuple[bool, str | None]:
    if spec.mode == "input_only" and not input_table_is_removable(config):
        return False, "input table is tied or declared non-removable"
    if spec.mode == "output_only" and tie_word_embeddings(config):
        return False, "native output head is tied to required input feedback"
    return True, None


def _aggregate(
    raw: list[dict[str, Any]],
    variant: _DeploymentVariant,
) -> dict[str, Any]:
    rows = [row[variant] for row in raw]
    return {
        "serialized_package_bytes": int(
            median(row["serialized_package_bytes"] for row in rows),
        ),
        "loaded_tensor_bytes": int(
            median(row["loaded_tensor_bytes"] for row in rows),
        ),
        **{phase: {name: int(median(row[phase][name] for row in rows)) for name in rows[0][phase]} for phase in ("post_load", "steady", "peak")},
    }


def _measurement_context(
    spec: DeploymentSpec,
    manifest: RunManifest,
    package: Mapping[str, Any],
    output: RunDirectory,
) -> _MeasurementContext:
    tokenizer = cast(
        Any,
        AutoTokenizer.from_pretrained(
            manifest.model_id,
            revision=manifest.model_revision,
        ),
    )
    prompts = registered_output_prompts(
        spec.prompt_set,
        spec.prompt_set_sha256,
    )
    prompt_ids = tuple(tuple(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts)
    stop = getattr(tokenizer, "eos_token_id", None)
    stop_ids = (stop,) if isinstance(stop, int) else tuple(stop or ())
    return _MeasurementContext(
        spec=spec,
        manifest=manifest,
        package=package,
        output=output,
        prompt_ids=prompt_ids,
        stop_ids=stop_ids,
    )


def _worker_request(
    context: _MeasurementContext,
    repetition: int,
    variant: _DeploymentVariant,
) -> dict[str, Any]:
    spec = context.spec
    manifest = context.manifest
    return {
        "variant": variant,
        "mode": spec.mode,
        "device": spec.device,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "source_directory": str(context.package["source_directory"]),
        "filtered_directory": str(context.package["filtered_directory"]),
        "checkpoint": str(spec.checkpoint),
        "checkpoint_sha256": spec.checkpoint_sha256,
        "prompt_token_ids": context.prompt_ids[repetition % len(context.prompt_ids)],
        "stop_ids": context.stop_ids,
        "max_steps": spec.max_steps,
        "max_bytes": spec.max_bytes,
        "result_path": str(
            context.output.path(f"workers/{repetition}-{variant}.json"),
        ),
    }


def _paired_measurement(
    context: _MeasurementContext,
    repetition: int,
    worker: DeploymentWorker,
) -> dict[str, Any]:
    order: tuple[_DeploymentVariant, ...] = ("native", "candidate") if repetition % 2 == 0 else ("candidate", "native")
    pair: dict[str, Any] = {
        "repetition": repetition,
        "order": order,
    }
    for variant in order:
        measurement = worker(_worker_request(context, repetition, variant))
        measurement["serialized_package_bytes"] = int(
            context.package[f"{variant}_serialized_package_bytes"],
        )
        pair[variant] = measurement
    pair["output_equivalent"] = pair["native"]["output_sha256"] == pair["candidate"]["output_sha256"]
    pair["hidden_equivalent"] = pair["native"]["hidden_sha256"] == pair["candidate"]["hidden_sha256"]
    return pair


def _measure_repetitions(
    context: _MeasurementContext,
    worker: DeploymentWorker,
) -> list[dict[str, Any]]:
    return [_paired_measurement(context, repetition, worker) for repetition in range(context.spec.repetitions)]


def _deployment_result(
    spec: DeploymentSpec,
    *,
    applicable: bool,
    reason: str | None,
    raw: list[dict[str, Any]],
    package: Mapping[str, Any] | None,
) -> dict[str, Any]:
    equivalent = applicable and all(row["output_equivalent"] and row["hidden_equivalent"] for row in raw)
    omission = (
        applicable
        and package is not None
        and bool(package["reference_tensor_absent_from_package"])
        and all(row["candidate"]["physical_reference_tensor_absent"] for row in raw)
    )
    return {
        "kind": "deployment_evidence",
        "mode": spec.mode,
        "operational_status": "completed",
        "applicability": {
            "applicable": applicable,
            "reason": reason,
        },
        "quality_run": str(spec.quality_run),
        "checkpoint": str(spec.checkpoint),
        "prompt_set": {
            "name": spec.prompt_set,
            "sha256": spec.prompt_set_sha256,
        },
        "raw_repetitions": raw,
        "native": _aggregate(raw, "native") if applicable else None,
        "candidate": _aggregate(raw, "candidate") if applicable else None,
        "physical_reference_tensor_absent": omission if applicable else None,
        "output_equivalent": equivalent if applicable else None,
        "hidden_equivalent": equivalent if applicable else None,
        "deployment_compactness_claimable": (omission and equivalent if applicable else None),
    }


def _publish_deployment(
    spec: DeploymentSpec,
    output_dir: Path,
    output: RunDirectory,
    manifest: RunManifest,
    result: Mapping[str, Any],
) -> None:
    output.write_json("deployment.json", result)
    output.write_json("deployment-spec.json", spec.to_dict())
    write_evidence_manifest(
        output_dir,
        EvidenceManifest(
            artifact_kind="deployment",
            mode=spec.mode,
            status="completed",
            identity=EvidenceIdentity(
                source_commit=manifest.source_commit,
                source_dirty=manifest.source_dirty,
                source_state_sha256=manifest.source_state_sha256,
                dependency_lock_sha256=manifest.dependency_lock_sha256,
                installed_package=manifest.installed_package,
                claim_vocabulary_sha256=manifest.claim_vocabulary_sha256,
                source_assets=manifest.source_assets,
                verification=manifest.verification,
                model_id=manifest.model_id,
                model_revision=manifest.model_revision,
            ),
            parents={
                "quality_run": spec.quality_run / "manifest-final.json",
            },
            inputs={
                "checkpoint": spec.checkpoint,
                "spec": output_dir / "deployment-spec.json",
            },
            artifacts={"deployment": output_dir / "deployment.json"},
        ),
    )


def run_deployment(
    spec: DeploymentSpec,
    output_dir: Path,
    *,
    worker: DeploymentWorker = _spawn_worker,
    package_preparer: Callable[..., dict[str, Any]] = _prepare_packages,
) -> dict[str, Any]:
    manifest, _ = _quality_inputs(spec)
    output = RunDirectory(output_dir)
    config = AutoConfig.from_pretrained(
        manifest.model_id,
        revision=manifest.model_revision,
    )
    applicable, reason = _applicability(spec, config)
    raw: list[dict[str, Any]] = []
    package: dict[str, Any] | None = None
    if applicable:
        package = package_preparer(
            spec,
            manifest.model_id,
            manifest.model_revision,
            str(manifest.embedding_tensor),
            output,
        )
        raw = _measure_repetitions(
            _measurement_context(spec, manifest, package, output),
            worker,
        )
    result = _deployment_result(
        spec,
        applicable=applicable,
        reason=reason,
        raw=raw,
        package=package,
    )
    _publish_deployment(
        spec,
        output_dir,
        output,
        manifest,
        result,
    )
    return result
