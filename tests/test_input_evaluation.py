from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from input_model_fixtures import make_adapter
from torch import nn

import continuous_tokenizer.input.benchmark.prefill as prefill_module
import continuous_tokenizer.input.evaluation as evaluation_module
from continuous_tokenizer.backbone.assets import ModelAssets, load_model_assets
from continuous_tokenizer.codec.checkpoints import save_checkpoint
from continuous_tokenizer.codec.input import InputByteCodec, InputByteCodecConfig
from continuous_tokenizer.contracts.performance import prefill_performance_errors
from continuous_tokenizer.input.adapter import (
    InputEmbeddingAdapter,
)
from continuous_tokenizer.input.benchmark.prefill import (
    PrefillBenchmarkOptions,
    benchmark_model_prefill,
)
from continuous_tokenizer.input.evaluation import (
    EvaluationOptions,
    EvaluationRuntime,
    PromptSample,
    evaluate_input_replacement,
)
from continuous_tokenizer.input.evaluation_calibration import (
    CalibrationIdentity,
    CalibrationTolerance,
    load_or_build_calibration,
)
from continuous_tokenizer.input.evidence import input_source_identity
from continuous_tokenizer.input.generation import InputOnlyCausalLM
from continuous_tokenizer.runtime.device import resolve_model_device
from continuous_tokenizer.runtime.resume import ResumeManager
from continuous_tokenizer.runtime.tensors import parameter_fingerprint
from continuous_tokenizer.runtime.timing import TimingObservation


class TinyModel(nn.Module):
    def __init__(self, adapter: InputEmbeddingAdapter) -> None:
        super().__init__()
        self.embedding = nn.Embedding(258, 8)
        self.lm_head = nn.Linear(8, 258, bias=False)
        with torch.no_grad():
            self.embedding.weight[:256].copy_(adapter.codec.byte_embeddings)
            self.embedding.weight[256].copy_(adapter.control_embeddings[0])
        self.config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            layer_types=("full_attention",),
        )
        self.calls = 0

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls += 1
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        logits = self.lm_head(hidden)
        logits_to_keep = kwargs.get("logits_to_keep")
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits.index_select(1, logits_to_keep.to(logits.device))
        elif isinstance(logits_to_keep, int) and logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits, past_key_values=None)


def _synthetic_assets(adapter: InputEmbeddingAdapter) -> ModelAssets:
    return ModelAssets(
        model_id="synthetic/model",
        revision="synthetic-revision",
        tokenizer=SimpleNamespace(eos_token_id=256),
        config={},
        embedding_tensor_name="embedding.weight",
        embedding_shard=Path("synthetic.safetensors"),
        vocabulary=adapter.vocabulary,
        input_embeddings=torch.cat(
            (
                adapter.codec.byte_embeddings,
                adapter.control_embeddings,
                torch.randn(1, 8),
            )
        ),
    )


def test_frozen_model_compatibility_path_matches_native_embeddings() -> None:
    adapter = make_adapter()
    model = TinyModel(adapter)
    wrapper = InputOnlyCausalLM(model, adapter)
    token_ids = (65, 256, 66)

    native = model(input_ids=torch.tensor([token_ids])).logits
    plugin, encoding = wrapper.forward_token_ids(token_ids, mode="compatibility")

    assert torch.equal(plugin.logits, native)
    assert len(encoding.positions) == len(token_ids)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_requested_device_without_index_matches_loaded_device_zero() -> None:
    model = nn.Linear(1, 1)

    assert resolve_model_device(torch.device("cpu:0"), model) == torch.device("cpu")
    with unittest.TestCase().assertRaisesRegex(ValueError, "does not match"):
        resolve_model_device(torch.device("mps"), model)


def test_checkpoint_adapter_does_not_repeat_device_move() -> None:
    with tempfile.TemporaryDirectory() as directory:
        adapter = make_adapter()
        checkpoint = Path(directory) / "adapter.pt"
        assets = ModelAssets(
            model_id="synthetic/model",
            revision="synthetic-revision",
            tokenizer=SimpleNamespace(),
            config={},
            embedding_tensor_name="embedding.weight",
            embedding_shard=Path("synthetic.safetensors"),
            vocabulary=adapter.vocabulary,
            input_embeddings=torch.cat((adapter.codec.byte_embeddings, adapter.control_embeddings, torch.randn(1, 8))),
        )
        save_checkpoint(
            checkpoint,
            adapter.codec,
            {
                "model_revision": "synthetic-revision",
                "source_identity": input_source_identity(assets),
            },
            control_ids=adapter.control_ids,
            control_embeddings=adapter.control_embeddings,
        )

        with patch.object(
            InputEmbeddingAdapter,
            "to",
            side_effect=AssertionError("checkpoint adapter repeated the device move"),
        ):
            loaded = InputEmbeddingAdapter.from_checkpoint(
                assets,
                checkpoint,
                device=torch.device("cpu"),
            )

        assert loaded.adapter.device == torch.device("cpu")


class ScriptedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, inputs_embeds: torch.Tensor, **kwargs: Any) -> Any:
        del kwargs
        token_id = 257 if self.calls == 0 else 256
        self.calls += 1
        logits = torch.full((1, 1, 258), -1_000.0, device=inputs_embeds.device)
        logits[0, 0, token_id] = 1_000.0
        return SimpleNamespace(logits=logits, past_key_values=self.calls)


def test_generated_ordinary_token_is_reencoded_and_control_eos_stops() -> None:
    adapter = make_adapter()
    model = ScriptedModel()
    wrapper = InputOnlyCausalLM(model, adapter)
    before = parameter_fingerprint(model)

    result = wrapper.generate((65,), mode="compatibility", eos_token_ids=(256,), max_new_tokens=4)

    assert result.token_ids == (257, 256)
    assert result.positions_added == 2
    assert parameter_fingerprint(model) == before


def test_prefill_benchmark_pairs_rotated_registered_prompts() -> None:
    adapter = make_adapter()
    model = TinyModel(adapter)
    clock = 0

    def deterministic_timing(operation, _device):
        nonlocal clock
        clock += 1
        return operation(), clock / 1_000

    def direct_timing(operation, _device):
        return operation(), TimingObservation(
            schema_version=1,
            wall_seconds=0.5,
            synchronization_count=0,
            host_to_device_bytes=0,
            device_to_host_bytes=0,
            rss_before_bytes=1,
            rss_after_bytes=1,
            peak_rss_bytes=1,
            mps_allocated_before_bytes=0,
            mps_allocated_after_bytes=0,
            peak_mps_allocated_bytes=0,
            mps_driver_before_bytes=0,
            mps_driver_after_bytes=0,
            peak_mps_driver_bytes=0,
            mps_peak_method="not_applicable",
        )

    with (
        patch.object(
            prefill_module,
            "timed_call",
            side_effect=deterministic_timing,
        ),
        patch.object(
            prefill_module,
            "timed_observation",
            side_effect=direct_timing,
        ),
    ):
        performance = benchmark_model_prefill(
            model,
            adapter,
            ((65, 66), (66, 67)),
            PrefillBenchmarkOptions(
                warmups=1,
                repetitions=2,
            ),
        )

    measurement = performance["measurement"]
    assert measurement["prompt_count"] == 2
    assert measurement["expected_raw_pairs"] == 4
    assert measurement["recorded_raw_pairs"] == 4
    assert measurement["warmup_executions"] == 6
    assert len(measurement["prompt_set_sha256"]) == 64
    pairs = performance["raw_pairs"]
    assert [pair["prompt_index"] for pair in pairs] == [0, 1, 1, 0]
    assert [pair["path_order"] for pair in pairs] == [
        ["native", "compatibility", "segmented"],
        ["compatibility", "segmented", "native"],
        ["segmented", "native", "compatibility"],
        ["native", "compatibility", "segmented"],
    ]
    assert all(set(pair["paths"]) == {"native", "compatibility", "segmented"} for pair in pairs)
    assert all(path["time_to_first_logit_seconds"] == 0.5 for pair in pairs for path in pair["paths"].values())
    assert any(path["subphase_sum_seconds"] != path["time_to_first_logit_seconds"] for pair in pairs for path in pair["paths"].values())
    assert performance["native"]["positions"] == 4
    assert performance["native"]["timing_observations"] == 4
    assert not prefill_performance_errors(performance)
    performance["native"]["time_to_first_logit_median_seconds"] = -1.0
    assert "prefill native time_to_first_logit summary differs from raw observations" in prefill_performance_errors(performance)


def test_input_evaluation_boundaries_resume_with_equivalent_metrics() -> None:
    torch.manual_seed(17)
    adapter = make_adapter()
    assets = _synthetic_assets(adapter)
    model = TinyModel(adapter)
    samples = [
        PromptSample((65, 66), (67,)),
        PromptSample((66, 67), (68,)),
    ]
    options = EvaluationOptions(
        output_dir=Path(),
        samples=2,
        prompt_tokens=2,
        continuation_tokens=1,
        generation_samples=0,
        max_new_tokens=1,
        warmups=0,
        repetitions=1,
        performance_prompts=1,
    )
    policy = evaluation_module.TeacherForcedBatchPolicy(batch_size=2)
    baseline = evaluation_module._native_baseline(
        evaluation_module.EvaluationSession(),
        assets,
        model,
        samples,
        options,
        policy,
    )
    original_save = ResumeManager.save

    def save_then_interrupt(manager, phase, epoch, state):
        original_save(manager, phase, epoch, state)
        raise KeyboardInterrupt

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            patch.object(ResumeManager, "save", autospec=True, side_effect=save_then_interrupt),
            unittest.TestCase().assertRaises(KeyboardInterrupt),
        ):
            evaluation_module._teacher_forced_metrics(
                model,
                adapter,
                samples,
                "arbitrary",
                evaluation_module._TeacherForcedRuntime(
                    evaluation_module._EvaluationResume(
                        ResumeManager(root / "teacher", "experiment", "commit", "source", "lock", False),
                        "input-evaluation-teacher-forced",
                    ),
                    assets.input_embeddings,
                    baseline,
                    policy,
                ),
            )
        resumed_teacher = evaluation_module._teacher_forced_metrics(
            model,
            adapter,
            samples,
            "arbitrary",
            evaluation_module._TeacherForcedRuntime(
                evaluation_module._EvaluationResume(
                    ResumeManager(root / "teacher", "experiment", "commit", "source", "lock", True),
                    "input-evaluation-teacher-forced",
                ),
                assets.input_embeddings,
                baseline,
                policy,
            ),
        )
        uninterrupted_teacher = evaluation_module._teacher_forced_metrics(
            model,
            adapter,
            samples,
            "arbitrary",
            evaluation_module._TeacherForcedRuntime(
                evaluation_module._EvaluationResume(None, "input-evaluation-teacher-forced"),
                assets.input_embeddings,
                baseline,
                policy,
            ),
        )
        assert resumed_teacher == uninterrupted_teacher

        options = EvaluationOptions(
            output_dir=root,
            samples=2,
            prompt_tokens=2,
            continuation_tokens=1,
            generation_samples=2,
            max_new_tokens=1,
            warmups=0,
            repetitions=1,
            performance_prompts=2,
        )
        wrapper = InputOnlyCausalLM(model, adapter)

        def deterministic_timing(operation, _device):
            return operation(), 0.0

        with (
            patch.object(evaluation_module, "timed_call", side_effect=deterministic_timing),
            patch.object(ResumeManager, "save", autospec=True, side_effect=save_then_interrupt),
            unittest.TestCase().assertRaises(KeyboardInterrupt),
        ):
            evaluation_module._generation_metrics(
                wrapper,
                assets,
                samples,
                options,
                evaluation_module._EvaluationResume(
                    ResumeManager(root / "generation", "experiment", "commit", "source", "lock", False),
                    "input-evaluation-generation",
                ),
            )
        with patch.object(evaluation_module, "timed_call", side_effect=deterministic_timing):
            resumed_generation = evaluation_module._generation_metrics(
                wrapper,
                assets,
                samples,
                options,
                evaluation_module._EvaluationResume(
                    ResumeManager(root / "generation", "experiment", "commit", "source", "lock", True),
                    "input-evaluation-generation",
                ),
            )
            uninterrupted_generation = evaluation_module._generation_metrics(
                wrapper,
                assets,
                samples,
                options,
                evaluation_module._EvaluationResume(None, "input-evaluation-generation"),
            )
        assert resumed_generation == uninterrupted_generation


def test_native_baseline_rejects_identity_mismatch_and_tampering() -> None:
    identity = evaluation_module.NativeBaselineIdentity(
        model_id="synthetic/model",
        model_revision="model-revision",
        model_fingerprint="model-fingerprint",
        prompt_window_sha256="prompt-digest",
        sample_order_sha256="order-digest",
        seed=17,
        dtype="torch.float32",
        device="cpu",
        generation_samples=0,
        max_new_tokens=1,
        eos_token_ids=(256,),
        teacher_forced_batch_size=1,
    )
    bundle = evaluation_module.NativeBaselineBundle.create(
        identity,
        (torch.ones(1, 3),),
        (),
    )

    with unittest.TestCase().assertRaisesRegex(ValueError, "identity"):
        bundle.verify(replace(identity, seed=23))

    bundle.teacher_logits[0].zero_()
    with unittest.TestCase().assertRaisesRegex(ValueError, "modified"):
        bundle.verify(identity)


def test_native_baseline_reuse_avoids_model_forwards() -> None:
    adapter = make_adapter()
    assets = _synthetic_assets(adapter)
    model = TinyModel(adapter)
    samples = [
        PromptSample((65, 66), (67,)),
        PromptSample((66, 67), (68,)),
    ]
    options = EvaluationOptions(
        output_dir=Path(),
        samples=2,
        prompt_tokens=2,
        continuation_tokens=1,
        generation_samples=0,
        max_new_tokens=1,
        warmups=0,
        repetitions=1,
        performance_prompts=1,
    )
    session = evaluation_module.EvaluationSession()

    first = evaluation_module._native_baseline(
        session,
        assets,
        model,
        samples,
        options,
        evaluation_module.TeacherForcedBatchPolicy(),
    )
    calls = model.calls
    second = evaluation_module._native_baseline(
        session,
        assets,
        model,
        samples,
        options,
        evaluation_module.TeacherForcedBatchPolicy(),
    )

    assert first is second
    assert model.calls == calls
    assert session.telemetry()["native_baseline_reuses"] == 1
    assert session.telemetry()["native_model_forwards_avoided"] == 1


def test_teacher_batches_preserve_variable_length_order_and_denominators() -> None:
    adapter = make_adapter()
    assets = _synthetic_assets(adapter)
    model = TinyModel(adapter)
    samples = [
        PromptSample((65, 66), (67,)),
        PromptSample((66, 67, 68), (68, 69)),
        PromptSample((67,), (69,)),
    ]
    options = EvaluationOptions(
        output_dir=Path(),
        samples=3,
        prompt_tokens=3,
        continuation_tokens=2,
        generation_samples=0,
        max_new_tokens=1,
        warmups=0,
        repetitions=1,
        performance_prompts=1,
    )
    policy = evaluation_module.TeacherForcedBatchPolicy(batch_size=2)
    session = evaluation_module.EvaluationSession()
    baseline = evaluation_module._native_baseline(
        session,
        assets,
        model,
        samples,
        options,
        policy,
    )

    scalar_logits = evaluation_module._calibration_logits(
        model,
        adapter,
        samples,
        "arbitrary",
        assets.input_embeddings,
        batched=False,
    )
    batched_logits = evaluation_module._calibration_logits(
        model,
        adapter,
        samples,
        "arbitrary",
        assets.input_embeddings,
        batched=True,
    )
    for mode in scalar_logits:
        assert len(batched_logits[mode]) == len(samples)
        for scalar, batched in zip(
            scalar_logits[mode],
            batched_logits[mode],
            strict=True,
        ):
            assert torch.allclose(scalar, batched, atol=1e-6, rtol=1e-6)

    metrics, positions, _ = evaluation_module._teacher_forced_metrics(
        model,
        adapter,
        samples,
        "arbitrary",
        evaluation_module._TeacherForcedRuntime(
            evaluation_module._EvaluationResume(None, "batched"),
            assets.input_embeddings,
            baseline,
            policy,
        ),
    )
    assert metrics["compatibility"]["tokens"] == 4
    assert metrics["segmented"]["tokens"] == 4
    assert positions["native"] == 2.0


def test_calibration_repeats_variable_length_samples_to_static_production_batch() -> None:
    class RecordingTinyModel(TinyModel):
        def __init__(self, adapter: InputEmbeddingAdapter) -> None:
            super().__init__(adapter)
            self.batched_inputs: list[tuple[tuple[int, ...], torch.Tensor]] = []

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> Any:
            values = input_ids if input_ids is not None else inputs_embeds
            attention_mask = kwargs.get("attention_mask")
            if values is not None and values.shape[0] == 8:
                assert isinstance(attention_mask, torch.Tensor)
                self.batched_inputs.append(
                    (tuple(values.shape), attention_mask.detach().clone()),
                )
            return super().forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

    adapter = make_adapter()
    assets = _synthetic_assets(adapter)
    model = RecordingTinyModel(adapter)
    measurements = evaluation_module._calibration_measurements(
        model,
        adapter,
        [
            PromptSample((65, 66), (67,)),
            PromptSample((66, 67, 68), (68, 69)),
        ],
        "arbitrary",
        assets.input_embeddings,
    )

    assert measurements["unique_sample_count"] == 2
    assert measurements["calibration_rows"] == 8
    assert measurements["production_batch_size"] == 8
    assert measurements["batched_model_forwards"] == 4
    assert len(model.batched_inputs) == 4
    for shape, attention_mask in model.batched_inputs:
        assert shape[0] == 8
        assert attention_mask.shape[0] == 8
        for row in range(2, 8):
            assert torch.equal(
                attention_mask[row],
                attention_mask[row % 2],
            )
    native_mask = model.batched_inputs[0][1]
    assert torch.any(native_mask == 0)
    assert not torch.equal(native_mask[0], native_mask[1])


def _calibration_identity(**updates: Any) -> CalibrationIdentity:
    values: dict[str, Any] = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "qwen-revision",
        "tokenizer_revision": "qwen-revision",
        "model_fingerprint": "a" * 64,
        "codec_checkpoint_fingerprint": "e" * 64,
        "segmentation_alignment": "arbitrary",
        "source_commit": "source-commit",
        "source_state_sha256": "f" * 64,
        "dtype": "torch.bfloat16",
        "device": "mps",
        "production_batch_size": 8,
        "unique_sample_count": 2,
        "calibration_rows": 8,
        "implementation_sha256": "b" * 64,
        "unique_samples_sha256": "c" * 64,
        "calibration_rows_sha256": "6" * 64,
        "tolerance": CalibrationTolerance(),
        "dependency_lock_sha256": "d" * 64,
    }
    values.update(updates)
    return CalibrationIdentity(**values)


def _passing_calibration() -> dict[str, float | int]:
    return {
        "unique_sample_count": 2,
        "calibration_rows": 8,
        "production_batch_size": 8,
        "paths": 4,
        "tokens": 48,
        "scalar_model_forwards": 8,
        "batched_model_forwards": 4,
        "maximum_kl": 0.0,
        "maximum_nll_delta": 0.0,
        "top1_agreement": 1.0,
        "maximum_logit_error": 0.0,
    }


def test_evaluation_calibration_builds_once_and_reuses_durably() -> None:
    calls = 0

    def build() -> dict[str, float | int]:
        nonlocal calls
        calls += 1
        return _passing_calibration()

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        first = load_or_build_calibration(
            cache,
            _calibration_identity(),
            build,
        )
        second = load_or_build_calibration(
            cache,
            _calibration_identity(),
            build,
        )

    assert first.built
    assert not second.built
    assert first.sha256 == second.sha256
    assert calls == 1
    assert first.artifact["identity"]["production_batch_size"] == 8
    assert first.artifact["identity"]["unique_sample_count"] == 2
    assert first.artifact["identity"]["calibration_rows"] == 8
    assert first.artifact["measurements"]["production_batch_size"] == 8
    assert first.artifact["measurements"]["unique_sample_count"] == 2
    assert first.artifact["measurements"]["calibration_rows"] == 8
    assert "batch_size" not in first.artifact["identity"]


def test_evaluation_calibration_identity_prevents_stale_reuse() -> None:
    variants = (
        {"codec_checkpoint_fingerprint": "1" * 64},
        {"segmentation_alignment": "native_boundaries"},
        {"tokenizer_revision": "tokenizer-revision"},
        {"source_commit": "other-commit"},
        {"source_state_sha256": "2" * 64},
        {"implementation_sha256": "3" * 64},
        {"dtype": "torch.float32"},
        {"device": "cpu"},
        {"unique_samples_sha256": "4" * 64},
        {"calibration_rows_sha256": "7" * 64},
        {"tolerance": CalibrationTolerance(maximum_kl=2e-4)},
        {"dependency_lock_sha256": "5" * 64},
    )
    calls = 0

    def build() -> dict[str, float | int]:
        nonlocal calls
        calls += 1
        return _passing_calibration()

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        baseline = load_or_build_calibration(
            cache,
            _calibration_identity(),
            build,
        )
        for updates in variants:
            record = load_or_build_calibration(
                cache,
                _calibration_identity(**updates),
                build,
            )
            assert record.built
            assert record.cache_path != baseline.cache_path

    assert calls == len(variants) + 1
    with unittest.TestCase().assertRaisesRegex(ValueError, "batch size eight"):
        _calibration_identity(production_batch_size=16)


def test_evaluation_calibration_rejects_failure_tampering_and_stale_identity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        failed = _passing_calibration() | {"maximum_logit_error": 1.0}
        with unittest.TestCase().assertRaisesRegex(ValueError, "failed"):
            load_or_build_calibration(
                cache,
                _calibration_identity(model_id="google/gemma-3-270m-it"),
                lambda: failed,
            )
        for field, value in (
            ("unique_sample_count", 1),
            ("calibration_rows", 2),
            ("production_batch_size", 2),
        ):
            with unittest.TestCase().assertRaisesRegex(ValueError, "differs"):
                load_or_build_calibration(
                    cache,
                    _calibration_identity(),
                    lambda field=field, value=value: _passing_calibration() | {field: value},
                )

        identity = _calibration_identity()
        record = load_or_build_calibration(
            cache,
            identity,
            _passing_calibration,
        )
        tampered = json.loads(record.cache_path.read_text(encoding="utf-8"))
        tampered["passed"] = False
        record.cache_path.write_text(json.dumps(tampered), encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(ValueError, "modified"):
            load_or_build_calibration(
                cache,
                identity,
                _passing_calibration,
            )

        stale_identity = _calibration_identity(model_revision="stale-revision")
        stale_path = cache / f"{stale_identity.key}.json"
        stale_path.write_text(
            json.dumps(record.artifact, sort_keys=True),
            encoding="utf-8",
        )
        with unittest.TestCase().assertRaisesRegex(ValueError, "identity"):
            load_or_build_calibration(
                cache,
                stale_identity,
                _passing_calibration,
            )


def test_full_evaluation_builds_then_reuses_bound_calibration() -> None:
    adapter = make_adapter()
    assets = _synthetic_assets(adapter)
    model = TinyModel(adapter).eval()
    samples = [
        PromptSample((65, 66), (67,)),
        PromptSample((66, 67, 68), (69,)),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "adapter.pt"
        save_checkpoint(
            checkpoint,
            adapter.codec,
            {
                "model_revision": assets.revision,
                "source_identity": input_source_identity(assets),
            },
            control_ids=adapter.control_ids,
            control_embeddings=adapter.control_embeddings,
        )
        cache = root / "cache"
        original = evaluation_module._calibration_measurements
        with (
            patch.object(evaluation_module, "_samples", return_value=samples),
            patch.object(
                evaluation_module,
                "_calibration_measurements",
                wraps=original,
            ) as calibration,
        ):
            results = [
                evaluate_input_replacement(
                    assets,
                    checkpoint,
                    EvaluationOptions(
                        output_dir=root / name,
                        samples=2,
                        prompt_tokens=3,
                        continuation_tokens=1,
                        generation_samples=0,
                        max_new_tokens=1,
                        warmups=0,
                        repetitions=1,
                        performance_prompts=1,
                    ),
                    EvaluationRuntime(
                        frozen_model=model,
                        calibration_cache_directory=cache,
                        dependency_lock_sha256="d" * 64,
                    ),
                )
                for name in ("first", "second")
            ]

        first = results[0]["measurement"]["calibration"]
        second = results[1]["measurement"]["calibration"]
        assert first["built"]
        assert not second["built"]
        assert first["key"] == second["key"]
        assert calibration.call_count == 1
        telemetry = results[1]["measurement"]["evaluation_telemetry"]
        assert telemetry["teacher_forced_model_forwards"] == 4
        assert telemetry["scalar_teacher_forced_model_forwards"] == 8
        assert telemetry["teacher_forced_model_forwards_avoided"] == 4
        assert telemetry["compact_metric_transfers"] == 1
        record = json.loads((root / "second/evaluation-calibration.json").read_text(encoding="utf-8"))
        assert not record["cache"]["built"]
        assert record["cache"]["sha256"]
        assert parameter_fingerprint(model) == results[1]["model"]["parameter_fingerprint"]


def test_evaluation_session_detects_frozen_model_mutation() -> None:
    adapter = make_adapter()
    model = TinyModel(adapter)
    session = evaluation_module.EvaluationSession()
    session.bind_model(model)
    with torch.no_grad():
        model.lm_head.weight[0, 0].add_(1)

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "changed"):
        session.verify_model(model)


def test_qwen_frozen_input_evaluation_smoke() -> None:
    if os.getenv("RUN_MODEL_TESTS") != "1":
        raise unittest.SkipTest("set RUN_MODEL_TESTS=1")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = load_model_assets(
            "Qwen/Qwen3.5-0.8B",
            "2fc06364715b967f1860aea9cf38778875588b17",
        )
        byte_ids = torch.tensor(assets.vocabulary.byte_token_ids)
        control_ids = torch.tensor(assets.vocabulary.control_ids)
        codec = InputByteCodec(
            InputByteCodecConfig(
                embedding_dim=assets.input_embeddings.shape[1],
                local_dim=128,
                projection_dim=assets.input_embeddings.shape[1],
                max_span=max(assets.vocabulary.max_token_bytes, 32),
                query_heads=4,
                feedforward_dim=256,
                encoder_layers=1,
                decoder_layers=1,
            ),
            assets.input_embeddings[byte_ids].float(),
        )
        checkpoint = root / "qwen-smoke.pt"
        save_checkpoint(
            checkpoint,
            codec,
            {
                "model_revision": assets.revision,
                "source_identity": input_source_identity(assets),
            },
            control_ids=control_ids,
            control_embeddings=assets.input_embeddings[control_ids],
        )

        metrics = evaluate_input_replacement(
            assets,
            checkpoint,
            EvaluationOptions(
                output_dir=root,
                samples=2,
                prompt_tokens=4,
                continuation_tokens=2,
                generation_samples=1,
                max_new_tokens=1,
                warmups=0,
                repetitions=1,
                performance_prompts=1,
            ),
            EvaluationRuntime(device=torch.device("cpu")),
        )

        assert metrics["model"]["parameter_fingerprint"]
        assert metrics["positions"]["native"] == 4


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_frozen_model_compatibility_path_matches_native_embeddings,
            test_requested_device_without_index_matches_loaded_device_zero,
            test_checkpoint_adapter_does_not_repeat_device_move,
            test_generated_ordinary_token_is_reencoded_and_control_eos_stops,
            test_prefill_benchmark_pairs_rotated_registered_prompts,
            test_input_evaluation_boundaries_resume_with_equivalent_metrics,
            test_native_baseline_rejects_identity_mismatch_and_tampering,
            test_native_baseline_reuse_avoids_model_forwards,
            test_teacher_batches_preserve_variable_length_order_and_denominators,
            test_calibration_repeats_variable_length_samples_to_static_production_batch,
            test_evaluation_calibration_builds_once_and_reuses_durably,
            test_evaluation_calibration_identity_prevents_stale_reuse,
            test_evaluation_calibration_rejects_failure_tampering_and_stale_identity,
            test_full_evaluation_builds_then_reuses_bound_calibration,
            test_evaluation_session_detects_frozen_model_mutation,
            test_qwen_frozen_input_evaluation_smoke,
        )
    )
