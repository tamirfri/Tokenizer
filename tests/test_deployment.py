from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, call, patch

from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

import continuous_tokenizer.campaigns.deployment as deployment_module
from continuous_tokenizer.artifacts.hashing import sha256_path
from continuous_tokenizer.backbone.deployment.input_table import (
    filter_input_embedding_checkpoint,
    load_filtered_causal_lm,
)
from continuous_tokenizer.backbone.deployment.output_head import (
    load_filtered_output_backbone,
)
from continuous_tokenizer.campaigns.deployment import run_deployment
from continuous_tokenizer.commands.deployment import deployment_spec
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256
from continuous_tokenizer.contracts.deployment import DeploymentSpec
from continuous_tokenizer.contracts.output import (
    OUTPUT_FIDELITY_PROMPT_SET,
    OUTPUT_FIDELITY_PROMPT_SET_SHA256,
)


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        model_id="fake/model",
        model_revision="revision",
        embedding_tensor="embed.weight",
        source_commit="commit",
        source_dirty=True,
        source_state_sha256="1" * 64,
        dependency_lock_sha256="2" * 64,
        installed_package={
            "name": "continuous-byte-tokenizer",
            "version": "0.1.0",
            "content_sha256": "3" * 64,
        },
        claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
        source_assets={},
        verification={"provided": False},
        artifacts={"checkpoint": "checkpoint.pt"},
        status="passed",
        mode="input_only",
    )


def _spec(root: Path) -> DeploymentSpec:
    return DeploymentSpec(
        name="fake-deployment",
        mode="input_only",
        device="cpu",
        quality_run=root / "quality",
        quality_manifest_sha256="4" * 64,
        checkpoint=root / "quality/checkpoint.pt",
        checkpoint_sha256="5" * 64,
        prompt_set=OUTPUT_FIDELITY_PROMPT_SET,
        prompt_set_sha256=OUTPUT_FIDELITY_PROMPT_SET_SHA256,
    )


def test_deployment_spec_is_strict_and_requires_three_repetitions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "deployment.toml"
        valid = f"""
name = "fake-deployment"
mode = "input_only"
device = "cpu"
quality_run = "quality"
quality_manifest_sha256 = "{"4" * 64}"
checkpoint = "quality/checkpoint.pt"
checkpoint_sha256 = "{"5" * 64}"
prompt_set = "{OUTPUT_FIDELITY_PROMPT_SET}"
prompt_set_sha256 = "{OUTPUT_FIDELITY_PROMPT_SET_SHA256}"
repetitions = 3
max_steps = 16
max_bytes = 1024
"""
        path.write_text(valid, encoding="utf-8")

        assert DeploymentSpec.load(path).repetitions == 3
        path.write_text(
            valid.replace("repetitions = 3", "repetitions = 2"),
            encoding="utf-8",
        )
        with unittest.TestCase().assertRaisesRegex(ValueError, "exactly 3"):
            DeploymentSpec.load(path)


def test_public_command_materializes_strict_deployment_spec() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        quality = root / "quality"
        quality.mkdir()
        manifest_path = quality / "manifest-final.json"
        checkpoint = quality / "checkpoint.pt"
        manifest_path.write_text("{}\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        (quality / "result.json").write_text(
            json.dumps(
                {
                    "mode": "output_only",
                    "evidence_scope": "final",
                    "operational_status": "completed",
                },
            ),
            encoding="utf-8",
        )
        manifest_sha256 = sha256_path(manifest_path)
        checkpoint_sha256 = sha256_path(checkpoint)
        manifest = SimpleNamespace(
            status="passed",
            mode="output_only",
            environment={"device": "cpu"},
            artifacts={"checkpoint": "checkpoint.pt"},
            artifact_hashes={"checkpoint": checkpoint_sha256},
        )
        output = root / "specifications"
        with patch(
            "continuous_tokenizer.commands.deployment.load_verified_run_manifest",
            return_value=manifest,
        ):
            result = deployment_spec(
                argparse.Namespace(
                    quality_runs=[quality],
                    output_dir=output,
                ),
            )

        generated = Path(result["specifications"][0])
        parsed = DeploymentSpec.load(generated)

    assert parsed.mode == "output_only"
    assert parsed.quality_manifest_sha256 == manifest_sha256
    assert parsed.checkpoint_sha256 == checkpoint_sha256
    assert parsed.quality_run == quality.resolve()
    assert parsed.checkpoint == checkpoint.resolve()


def test_deployment_runs_three_paired_fake_workers() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        quality = root / "quality"
        quality.mkdir()
        (quality / "manifest-final.json").write_text("{}\n", encoding="utf-8")
        (quality / "checkpoint.pt").write_bytes(b"checkpoint")
        source = root / "source"
        filtered = root / "filtered"
        source.mkdir()
        filtered.mkdir()
        (source / "model.safetensors").write_bytes(b"source")
        (filtered / "model.safetensors").write_bytes(b"filtered")

        def prepare(*_args: object) -> dict[str, object]:
            return {
                "source_directory": source,
                "filtered_directory": filtered,
                "reference_tensor_absent_from_package": True,
                "native_serialized_package_bytes": 100,
                "candidate_serialized_package_bytes": 60,
            }

        def worker(request: Mapping[str, Any]) -> dict[str, object]:
            variant = request["variant"]
            memory = {
                "rss_bytes": 100 if variant == "native" else 70,
                "peak_rss_bytes": 110 if variant == "native" else 80,
                "mps_allocated_bytes": 0,
                "mps_driver_bytes": 0,
            }
            return {
                "variant": variant,
                "physical_reference_tensor_absent": variant == "candidate",
                "loaded_tensor_bytes": 90 if variant == "native" else 50,
                "post_load": memory,
                "steady": memory,
                "peak": memory,
                "output_sha256": "output",
                "hidden_sha256": "hidden",
            }

        def encode(text: str, *, add_special_tokens: bool) -> list[int]:
            del add_special_tokens
            return list(text.encode())

        fake_tokenizer = SimpleNamespace(encode=encode, eos_token_id=None)
        with (
            patch(
                "continuous_tokenizer.campaigns.deployment._quality_inputs",
                return_value=(_manifest(), {}),
            ),
            patch(
                "continuous_tokenizer.campaigns.deployment.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(
                    tie_word_embeddings=False,
                    removable_input_table=True,
                ),
            ),
            patch(
                "continuous_tokenizer.campaigns.deployment.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            result = run_deployment(
                _spec(root),
                root / "deployment",
                worker=worker,
                package_preparer=prepare,
            )

        assert len(result["raw_repetitions"]) == 3
        assert result["raw_repetitions"][0]["order"] == (
            "native",
            "candidate",
        )
        assert result["raw_repetitions"][1]["order"] == (
            "candidate",
            "native",
        )
        assert result["physical_reference_tensor_absent"]
        assert result["output_equivalent"]
        assert result["hidden_equivalent"]
        assert result["deployment_compactness_claimable"]
        assert result["native"]["serialized_package_bytes"] == 100
        assert result["candidate"]["serialized_package_bytes"] == 60
        assert (root / "deployment/evidence-manifest.json").is_file()


def test_deployment_baseline_and_candidate_use_fresh_spawn_workers() -> None:
    class ImmediateProcess:
        def __init__(self, *, target: object, args: tuple[object, ...]) -> None:
            self.target = cast(Any, target)
            self.args = args
            self.exitcode = 0

        def start(self) -> None:
            self.target(*self.args)

        def join(self) -> None:
            return

    context = SimpleNamespace(Process=ImmediateProcess)
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.object(
            deployment_module.multiprocessing,
            "get_context",
            return_value=context,
        ) as get_context,
        patch.object(
            deployment_module,
            "_worker",
            side_effect=lambda request: {"variant": request["variant"]},
        ),
    ):
        root = Path(directory)
        results = [
            deployment_module._spawn_worker(
                {
                    "variant": variant,
                    "result_path": str(root / f"{variant}.json"),
                }
            )
            for variant in ("native", "candidate")
        ]

    assert results == [{"variant": "native"}, {"variant": "candidate"}]
    assert get_context.call_args_list == [call("spawn")] * 2


def test_tied_deployment_is_inapplicable_without_workers() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        quality = root / "quality"
        quality.mkdir()
        (quality / "manifest-final.json").write_text("{}\n", encoding="utf-8")
        (quality / "checkpoint.pt").write_bytes(b"checkpoint")
        worker = Mock()
        prepare = Mock()
        with (
            patch(
                "continuous_tokenizer.campaigns.deployment._quality_inputs",
                return_value=(_manifest(), {}),
            ),
            patch(
                "continuous_tokenizer.campaigns.deployment.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(tie_word_embeddings=True),
            ),
        ):
            result = run_deployment(
                _spec(root),
                root / "deployment",
                worker=worker,
                package_preparer=prepare,
            )

        worker.assert_not_called()
        prepare.assert_not_called()
        assert not result["applicability"]["applicable"]
        assert result["physical_reference_tensor_absent"] is None
        assert result["deployment_compactness_claimable"] is None


def test_filtered_checkpoint_omits_input_table_and_loads_without_embedding_parameters() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "filtered"
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=256,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=1,
                tie_word_embeddings=False,
            )
        )
        model.save_pretrained(source, safe_serialization=True, max_shard_size="5KB")
        tensor_name = "transformer.wte.weight"

        package = filter_input_embedding_checkpoint(source, output, tensor_name)
        filtered, measurement = load_filtered_causal_lm(output)

        index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert tensor_name not in index["weight_map"]
        assert package["input_tensor_absent"]
        assert package["omitted_tensor_bytes"] == 256 * 8 * 4
        assert measurement["input_module_parameter_count"] == 0
        assert measurement["meta_tensor_count"] == 0
        filtered_embeddings = cast(Any, filtered).get_input_embeddings()
        source_embeddings = cast(Any, model).get_input_embeddings()
        assert isinstance(filtered_embeddings, nn.Module)
        assert isinstance(source_embeddings, nn.Embedding)
        assert sum(parameter.numel() for parameter in filtered_embeddings.parameters()) == 0
        assert source_embeddings.weight.numel() == 256 * 8


def test_filtered_checkpoint_rejects_tied_embeddings() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "filtered"
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=256,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=1,
                tie_word_embeddings=True,
            )
        )
        model.save_pretrained(source, safe_serialization=True, max_shard_size="5KB")
        filter_input_embedding_checkpoint(source, output, "transformer.wte.weight")

        with unittest.TestCase().assertRaisesRegex(ValueError, "tied"):
            load_filtered_causal_lm(output)


def test_filtered_checkpoint_omits_output_head_but_keeps_input_feedback() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "filtered-output"
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=256,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=1,
                tie_word_embeddings=False,
            )
        )
        model.save_pretrained(source, safe_serialization=True, max_shard_size="5KB")
        package = filter_input_embedding_checkpoint(source, output, "lm_head.weight")

        filtered, measurement = load_filtered_output_backbone(output)

        assert package["omitted_tensor_bytes"] == 256 * 8 * 4
        assert measurement["output_module_parameter_count"] == 0
        assert measurement["meta_tensor_count"] == 0
        assert sum(parameter.numel() for parameter in cast(Any, filtered).get_input_embeddings().parameters()) == 256 * 8


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_deployment_spec_is_strict_and_requires_three_repetitions,
            test_public_command_materializes_strict_deployment_spec,
            test_deployment_runs_three_paired_fake_workers,
            test_deployment_baseline_and_candidate_use_fresh_spawn_workers,
            test_tied_deployment_is_inapplicable_without_workers,
            test_filtered_checkpoint_omits_input_table_and_loads_without_embedding_parameters,
            test_filtered_checkpoint_rejects_tied_embeddings,
            test_filtered_checkpoint_omits_output_head_but_keeps_input_feedback,
        )
    )
