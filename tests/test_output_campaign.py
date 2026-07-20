from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import torch
from torch import Tensor, nn

import continuous_tokenizer.output.benchmark as output_benchmark_module
from continuous_tokenizer.artifacts.store import RunDirectory
from continuous_tokenizer.backbone.assets import load_frozen_causal_lm, load_model_assets
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.synthetic import SyntheticCausalLM, synthetic_model_assets
from continuous_tokenizer.backbone.vocabulary import ByteVocabulary
from continuous_tokenizer.campaigns.dispatch import create_experiment_runner
from continuous_tokenizer.campaigns.output import OutputExperimentRunner
from continuous_tokenizer.codec.checkpoints import (
    load_checkpoint,
    load_output_checkpoint,
    save_output_checkpoint,
)
from continuous_tokenizer.codec.output import (
    OutputByteCodec,
    OutputByteCodecConfig,
)
from continuous_tokenizer.contracts.experiment import (
    ExperimentSpec,
    SearchSelectionSpec,
    StudySelectionSpec,
)
from continuous_tokenizer.contracts.output import OutputTrainingSpec
from continuous_tokenizer.contracts.performance import output_performance_errors
from continuous_tokenizer.output.benchmark import (
    OutputBenchmarkOptions,
    benchmark_output_generation,
)
from continuous_tokenizer.output.evaluation import (
    OutputRolloutOptions,
    evaluate_output_rollouts,
)
from continuous_tokenizer.output.generation import OutputOnlyGenerator


class _FixedOutputCodec(nn.Module):
    def __init__(self, generated: bytes, *, control: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.generated = generated
        self.control = control
        self.config = SimpleNamespace(control_count=1 if control else 0)
        self.max_span = max(len(generated), 1)

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    def decode_logits(
        self,
        hidden_state: Tensor,
        output_positions: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        del output_positions
        byte_logits = torch.full(
            (hidden_state.shape[0], self.max_span + 1, 257),
            -10.0,
            device=hidden_state.device,
        )
        for index, value in enumerate(self.generated):
            byte_logits[:, index, value] = 10.0
        byte_logits[:, len(self.generated), 256] = 10.0
        controls = torch.tensor(
            [[0.0, 10.0]] if self.control else [[10.0]],
            device=hidden_state.device,
        )
        return byte_logits, controls


class _FixedNativeHead(nn.Module):
    def __init__(self, vocabulary_size: int, token_id: int) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.token_id = token_id

    def forward(self, hidden: Tensor) -> Tensor:
        logits = torch.full(
            (*hidden.shape[:-1], self.vocabulary_size),
            -1.0,
            device=hidden.device,
        )
        logits[..., self.token_id] = 1.0
        return logits


class _ConditionalNativeHead(nn.Module):
    def forward(self, hidden: Tensor) -> Tensor:
        logits = torch.full((*hidden.shape[:-1], 257), -10.0, device=hidden.device)
        controls = hidden[..., 0] > 0
        logits[..., 65] = torch.where(controls, -10.0, 10.0)
        logits[..., 256] = torch.where(controls, 10.0, -10.0)
        return logits


class _ConditionalOutputCodec(nn.Module):
    config = SimpleNamespace(control_count=1)
    max_span = 1

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    def decode_logits(
        self,
        hidden_state: Tensor,
        output_positions: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        del output_positions
        controls = hidden_state[:, 0] > 0
        byte_logits = torch.full((hidden_state.shape[0], 2, 257), -10.0, device=hidden_state.device)
        byte_logits[:, 0, 65] = 10.0
        byte_logits[:, 1, 256] = 10.0
        control_logits = torch.stack((~controls, controls), dim=1).to(hidden_state.dtype) * 20 - 10
        return byte_logits, control_logits


class OutputModeTests(unittest.TestCase):
    def test_infeasible_final_lineage_completes_without_training(self) -> None:
        repository = Path(__file__).parents[1]
        baseline = ExperimentSpec.load(
            repository / "experiments/campaigns/output/qwen35-0.8b/seed-17.toml",
        )
        search = SearchSelectionSpec(
            search_kind="output",
            artifact="search.json",
            artifact_sha256="a" * 64,
            selected_trial=0,
            search_fingerprint="b" * 64,
            model_id=baseline.model.model_id,
            model_revision=baseline.model.revision,
            profile=baseline.training.profile,
            selected_parameters={
                "learning_rate": baseline.training.learning_rate,
                "weight_decay": baseline.training.weight_decay,
                "batch_size": baseline.training.batch_size,
            },
            feasible=False,
        )
        training = cast(OutputTrainingSpec, baseline.training)
        study = StudySelectionSpec(
            study_kind="output_oracle",
            artifact="result.json",
            artifact_sha256="c" * 64,
            study_fingerprint="d" * 64,
            model_id=baseline.model.model_id,
            model_revision=baseline.model.revision,
            selected_parameters={
                "max_span": training.max_span,
            },
            feasible=False,
        )
        spec = replace(
            baseline,
            evidence_scope="final",
            search_selections=(search,),
            study_selections=(study,),
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = object.__new__(OutputExperimentRunner)
            runner.spec = spec
            runner.pilot_corpus = None
            runner.device = torch.device("cpu")
            runner.verification = {"provided": False}
            runner.run_directory = RunDirectory(
                Path(directory) / "run",
            )
            runner._write_start_manifest = mock.Mock()
            runner._finalize_success = mock.Mock()

            self.assertTrue(runner._structurally_unrepresentable())
            result = runner._publish_structurally_unsupported()

        self.assertEqual(result["operational_status"], "completed")
        self.assertEqual(result["scientific_verdict"], "unsupported")
        self.assertFalse(result["output"]["training_performed"])
        verdicts = {claim["claim_id"]: claim["verdict"] for claim in result["claims"]}
        self.assertEqual(
            verdicts["output.semi_autoregressive_density"],
            "unsupported",
        )
        self.assertIn("incomplete", verdicts.values())
        runner._finalize_success.assert_called_once()

    def test_output_checkpoint_is_direction_tagged(self) -> None:
        codec = OutputByteCodec(
            OutputByteCodecConfig(
                embedding_dim=8,
                local_dim=8,
                max_span=2,
                feedforward_dim=16,
                decoder_layers=1,
                control_count=1,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.pt"
            save_output_checkpoint(
                path,
                codec,
                {"model_revision": "revision"},
                control_ids=torch.tensor([256]),
            )
            loaded = load_output_checkpoint(path)

            self.assertEqual(loaded.codec.config, codec.config)
            self.assertEqual(loaded.control_ids.tolist(), [256])
            with self.assertRaisesRegex(ValueError, "input codec"):
                load_checkpoint(path)

    def test_generation_never_calls_lm_head_and_feeds_all_native_positions(self) -> None:
        assets = synthetic_model_assets()
        model = SyntheticCausalLM(assets.input_embeddings)

        class FailingHead(nn.Module):
            def forward(self, value: Tensor) -> Tensor:
                del value
                raise AssertionError("native LM head was called")

        cast(Any, model).lm_head = FailingHead()
        codec = cast(OutputByteCodec, _FixedOutputCodec(b"\x00\xff"))
        generator = OutputOnlyGenerator(
            FrozenBackbone(model),
            codec,
            assets.vocabulary,
            torch.empty(0, dtype=torch.long),
        )

        result = generator.generate(
            (0,),
            stop_control_ids=frozenset(),
            max_macro_steps=1,
            max_bytes=4,
        )

        self.assertEqual(result.data, b"\x00\xff")
        self.assertEqual(result.native_tokens_represented, 2)
        self.assertEqual(result.native_head_invocations, 0)

    def test_generation_counts_native_head_invocations(self) -> None:
        assets = synthetic_model_assets()
        model = SyntheticCausalLM(assets.input_embeddings)
        generator = OutputOnlyGenerator(
            FrozenBackbone(model),
            cast(OutputByteCodec, _FixedOutputCodec(b"a")),
            assets.vocabulary,
            torch.empty(0, dtype=torch.long),
        )

        def invoke_native_head(
            _module: nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Any,
        ) -> None:
            model.lm_head(output.last_hidden_state)

        handle = model.base_model.register_forward_hook(invoke_native_head)
        try:
            result = generator.generate(
                (0,),
                stop_control_ids=frozenset(),
                max_macro_steps=1,
                max_bytes=1,
            )
        finally:
            handle.remove()

        self.assertEqual(result.native_head_invocations, 1)

    def test_control_event_stops_without_native_head_or_feedback(self) -> None:
        embeddings = torch.randn(257, 8)
        token_bytes = (*tuple(bytes([value]) for value in range(256)), None)
        vocabulary = ByteVocabulary(
            token_bytes,
            tuple(range(256)),
            (256,),
            tuple(range(256)),
            1,
        )
        model = SyntheticCausalLM(embeddings)
        codec = cast(OutputByteCodec, _FixedOutputCodec(b"", control=True))
        generator = OutputOnlyGenerator(
            FrozenBackbone(model),
            codec,
            vocabulary,
            torch.tensor([256]),
        )

        result = generator.generate(
            (0,),
            stop_control_ids=frozenset({256}),
            max_macro_steps=2,
            max_bytes=2,
        )

        self.assertEqual(result.data, b"")
        self.assertEqual(result.native_tokens_represented, 1)
        self.assertEqual(result.macro_steps, 1)
        self.assertEqual(result.termination_reason, "stop_control")

    def test_invalid_span_is_a_scientific_outcome_not_an_execution_error(self) -> None:
        assets = synthetic_model_assets()
        generator = OutputOnlyGenerator(
            FrozenBackbone(SyntheticCausalLM(assets.input_embeddings)),
            cast(OutputByteCodec, _FixedOutputCodec(b"")),
            assets.vocabulary,
            torch.empty(0, dtype=torch.long),
        )

        result = generator.generate(
            (0,),
            stop_control_ids=frozenset(),
            max_macro_steps=1,
            max_bytes=1,
        )

        self.assertEqual(result.events, ())
        self.assertEqual(result.invalid_events, 1)
        self.assertEqual(result.macro_steps, 1)
        self.assertEqual(result.termination_reason, "invalid_event")

    def test_multi_prompt_rollout_measures_real_control_coverage(self) -> None:
        embeddings = torch.randn(257, 8)
        token_bytes = (*tuple(bytes([value]) for value in range(256)), None)
        vocabulary = ByteVocabulary(
            token_bytes,
            tuple(range(256)),
            (256,),
            tuple(range(256)),
            1,
        )
        model = SyntheticCausalLM(embeddings)
        cast(Any, model).lm_head = _FixedNativeHead(257, 256)
        backbone = FrozenBackbone(model)
        generator = OutputOnlyGenerator(
            backbone,
            cast(OutputByteCodec, _FixedOutputCodec(b"", control=True)),
            vocabulary,
            torch.tensor([256]),
        )

        metrics = evaluate_output_rollouts(
            backbone,
            generator,
            vocabulary,
            ((0,), (1,)),
            OutputRolloutOptions(
                stop_control_ids=frozenset({256}),
                max_macro_steps=2,
                max_bytes=2,
            ),
        )

        self.assertEqual(metrics.prompts, 2)
        self.assertEqual(metrics.rollout_event_agreement, 1.0)
        self.assertEqual(metrics.control_prompt_coverage, 1.0)
        self.assertEqual(metrics.control_precision, 1.0)
        self.assertEqual(metrics.control_recall, 1.0)
        self.assertEqual(metrics.oracle_stop_control_events, 2)
        self.assertEqual(metrics.stop_prompt_coverage, 1.0)
        self.assertEqual(metrics.attempted_macro_steps, 2)
        self.assertEqual(metrics.native_tokens_per_attempted_macro_step, 1.0)
        self.assertEqual(metrics.termination_stop_control, 2)

    def test_multi_prompt_rollout_covers_mixed_byte_and_control_events(self) -> None:
        embeddings = torch.zeros(257, 8)
        embeddings[0, 0] = 1
        embeddings[1, 0] = -1
        token_bytes = (*tuple(bytes([value]) for value in range(256)), None)
        vocabulary = ByteVocabulary(
            token_bytes,
            tuple(range(256)),
            (256,),
            tuple(range(256)),
            1,
        )
        model = SyntheticCausalLM(embeddings)
        cast(Any, model).lm_head = _ConditionalNativeHead()
        backbone = FrozenBackbone(model)
        generator = OutputOnlyGenerator(
            backbone,
            cast(OutputByteCodec, _ConditionalOutputCodec()),
            vocabulary,
            torch.tensor([256]),
        )

        metrics = evaluate_output_rollouts(
            backbone,
            generator,
            vocabulary,
            ((0,), (1,)),
            OutputRolloutOptions(
                stop_control_ids=frozenset({256}),
                max_macro_steps=1,
                max_bytes=1,
            ),
        )

        self.assertEqual(metrics.prompts, 2)
        self.assertEqual(metrics.rollout_event_agreement, 1.0)
        self.assertEqual(metrics.rollout_byte_agreement, 1.0)
        self.assertEqual(metrics.rollout_token_agreement, 1.0)
        self.assertEqual(metrics.control_prompt_coverage, 0.5)
        self.assertEqual(metrics.control_precision, 1.0)
        self.assertEqual(metrics.control_recall, 1.0)
        self.assertEqual(metrics.attempted_macro_steps, 2)
        self.assertEqual(metrics.native_tokens_per_attempted_macro_step, 1.0)

    def test_rollout_invalid_attempts_remain_in_every_denominator(self) -> None:
        assets = synthetic_model_assets()
        model = SyntheticCausalLM(assets.input_embeddings)
        cast(Any, model).lm_head = _FixedNativeHead(
            assets.input_embeddings.shape[0],
            ord("a"),
        )
        backbone = FrozenBackbone(model)
        generator = OutputOnlyGenerator(
            backbone,
            cast(OutputByteCodec, _FixedOutputCodec(b"")),
            assets.vocabulary,
            torch.empty(0, dtype=torch.long),
        )

        metrics = evaluate_output_rollouts(
            backbone,
            generator,
            assets.vocabulary,
            ((0,), (1,)),
            OutputRolloutOptions(
                stop_control_ids=frozenset(),
                max_macro_steps=1,
                max_bytes=1,
            ),
        )

        self.assertEqual(metrics.attempted_macro_steps, 2)
        self.assertEqual(metrics.invalid_events, 2)
        self.assertEqual(metrics.invalid_fraction, 1.0)
        self.assertEqual(metrics.native_tokens_per_attempted_macro_step, 0.0)
        self.assertEqual(metrics.termination_invalid_event, 2)

    def test_control_and_stop_metrics_publish_false_positive_formulas(self) -> None:
        embeddings = torch.zeros(257, 8)
        embeddings[0, 0] = 1
        embeddings[1, 0] = -1
        vocabulary = ByteVocabulary(
            (*tuple(bytes([value]) for value in range(256)), None),
            tuple(range(256)),
            (256,),
            tuple(range(256)),
            1,
        )
        model = SyntheticCausalLM(embeddings)
        cast(Any, model).lm_head = _ConditionalNativeHead()
        backbone = FrozenBackbone(model)
        generator = OutputOnlyGenerator(
            backbone,
            cast(OutputByteCodec, _FixedOutputCodec(b"", control=True)),
            vocabulary,
            torch.tensor([256]),
        )

        metrics = evaluate_output_rollouts(
            backbone,
            generator,
            vocabulary,
            ((0,), (1,)),
            OutputRolloutOptions(
                stop_control_ids=frozenset({256}),
                max_macro_steps=1,
                max_bytes=1,
            ),
        )

        self.assertEqual(metrics.control_precision, 0.5)
        self.assertEqual(metrics.control_recall, 1.0)
        self.assertEqual(metrics.control_false_positives, 1)
        self.assertEqual(metrics.control_false_negatives, 0)
        self.assertEqual(metrics.stop_precision, 0.5)
        self.assertEqual(metrics.stop_recall, 1.0)
        self.assertEqual(metrics.stop_false_positives, 1)
        self.assertEqual(metrics.stop_false_negatives, 0)

    def test_generation_never_exceeds_byte_limit(self) -> None:
        assets = synthetic_model_assets()
        generator = OutputOnlyGenerator(
            FrozenBackbone(SyntheticCausalLM(assets.input_embeddings)),
            cast(OutputByteCodec, _FixedOutputCodec(b"ab")),
            assets.vocabulary,
            torch.empty(0, dtype=torch.long),
        )

        result = generator.generate(
            (0,),
            stop_control_ids=frozenset(),
            max_macro_steps=1,
            max_bytes=1,
        )

        self.assertEqual(result.data, b"")
        self.assertEqual(result.events, ())

    def test_paired_benchmark_retains_rotated_raw_repetitions(self) -> None:
        assets = synthetic_model_assets()
        generator = OutputOnlyGenerator(
            FrozenBackbone(SyntheticCausalLM(assets.input_embeddings)),
            cast(OutputByteCodec, _FixedOutputCodec(b"a")),
            assets.vocabulary,
            torch.empty(0, dtype=torch.long),
        )

        with mock.patch.object(
            output_benchmark_module,
            "_sha256_bytes",
            wraps=output_benchmark_module._sha256_bytes,
        ) as sha256_bytes:
            benchmark = benchmark_output_generation(
                generator,
                ((0,), (1,)),
                OutputBenchmarkOptions(
                    warmups=0,
                    repetitions=2,
                    stop_control_ids=frozenset(),
                    max_macro_steps=1,
                    max_bytes=1,
                ),
            )

        self.assertEqual(len(benchmark.raw_repetitions), 4)
        self.assertEqual(
            benchmark.raw_repetitions[0]["order"],
            ("native", "candidate"),
        )
        self.assertEqual(
            benchmark.raw_repetitions[2]["order"],
            ("candidate", "native"),
        )
        self.assertEqual(benchmark.schema_version, 1)
        self.assertEqual(
            benchmark.measurement["recorded_raw_repetitions"],
            4,
        )
        self.assertEqual(len(benchmark.measurement["setup_runs"]), 2)
        for prompt_index, setup in enumerate(
            benchmark.measurement["setup_runs"],
        ):
            self.assertEqual(setup["prompt_index"], prompt_index)
            self.assertEqual(setup["candidate_runs"], 1)
            self.assertEqual(setup["native_runs"], 1)
            self.assertFalse(setup["timed"])
        self.assertFalse(benchmark.speedup_claimable)
        self.assertFalse(output_performance_errors(benchmark.to_dict()))
        self.assertLessEqual(sha256_bytes.call_count, 4)
        for pair in benchmark.raw_repetitions:
            for variant in ("native", "candidate"):
                observation = pair[variant]
                self.assertEqual(
                    observation["latency_seconds"],
                    observation["timing"]["wall_seconds"],
                )
                self.assertEqual(
                    set(observation["subphases"]),
                    {
                        "schema_version",
                        "preparation_seconds",
                        "backbone_seconds",
                        "output_decode_seconds",
                        "feedback_seconds",
                        "cache_accounting_seconds",
                        "host_to_device_bytes",
                        "device_to_host_bytes",
                        "synchronization_count",
                        "backbone_calls",
                        "output_decode_calls",
                        "feedback_calls",
                        "cache_snapshots",
                        "graph_signature_counts",
                    },
                )
        tampered = benchmark.to_dict()
        cast(dict[str, Any], tampered["candidate"])["latency_median_seconds"] = -1.0
        self.assertIn(
            "output candidate latency summary differs from raw observations",
            output_performance_errors(tampered),
        )
        if not benchmark.exact_trajectory_equivalence:
            self.assertFalse(benchmark.speedup_claimable)
            self.assertIsNone(benchmark.latency_speedup)

    def test_output_setup_and_registered_calls_are_balanced_and_ordered(self) -> None:
        assets = synthetic_model_assets()
        generator = OutputOnlyGenerator(
            FrozenBackbone(SyntheticCausalLM(assets.input_embeddings)),
            cast(OutputByteCodec, _FixedOutputCodec(b"aa")),
            assets.vocabulary,
            torch.empty(0, dtype=torch.long),
        )
        events: list[str] = []
        original_candidate = output_benchmark_module._generate_candidate
        original_native = output_benchmark_module._generate_native

        def candidate(*args: Any, **kwargs: Any) -> Any:
            events.append("candidate")
            return original_candidate(*args, **kwargs)

        def native(*args: Any, **kwargs: Any) -> Any:
            events.append("native")
            return original_native(*args, **kwargs)

        with (
            mock.patch.object(
                output_benchmark_module,
                "_generate_candidate",
                side_effect=candidate,
            ) as candidate_call,
            mock.patch.object(
                output_benchmark_module,
                "_generate_native",
                side_effect=native,
            ) as native_call,
        ):
            benchmark = benchmark_output_generation(
                generator,
                ((0,),),
                OutputBenchmarkOptions(
                    warmups=2,
                    repetitions=2,
                    stop_control_ids=frozenset(),
                    max_macro_steps=1,
                    max_bytes=2,
                ),
            )

        self.assertEqual(candidate_call.call_count, 5)
        self.assertEqual(native_call.call_count, 5)
        self.assertEqual(
            events,
            [
                "candidate",
                "native",
                "native",
                "candidate",
                "native",
                "candidate",
                "native",
                "candidate",
                "candidate",
                "native",
            ],
        )
        setup = benchmark.measurement["setup_runs"][0]
        self.assertEqual(setup["native_token_horizon"], 2)
        self.assertEqual(benchmark.measurement["warmups"], 2)
        self.assertEqual(len(benchmark.raw_repetitions), 2)

    def test_compressed_benchmark_uses_one_semantic_horizon(self) -> None:
        embeddings = torch.randn(256, 16, generator=torch.Generator().manual_seed(4))
        vocabulary = ByteVocabulary(
            token_bytes=tuple(bytes([value]) for value in range(256)),
            ordinary_ids=tuple(range(256)),
            control_ids=(),
            byte_token_ids=tuple(range(256)),
            max_token_bytes=1,
        )
        model = SyntheticCausalLM(embeddings)
        cast(Any, model).lm_head = _FixedNativeHead(256, ord("a"))
        generator = OutputOnlyGenerator(
            FrozenBackbone(model),
            cast(OutputByteCodec, _FixedOutputCodec(b"aa")),
            vocabulary,
            torch.empty(0, dtype=torch.long),
        )

        reduced = benchmark_output_generation(
            generator,
            ((0,),),
            OutputBenchmarkOptions(
                warmups=1,
                repetitions=2,
                stop_control_ids=frozenset(),
                max_macro_steps=1,
                max_bytes=2,
            ),
        )
        complete = benchmark_output_generation(
            generator,
            ((0,),),
            OutputBenchmarkOptions(
                warmups=5,
                repetitions=20,
                stop_control_ids=frozenset(),
                max_macro_steps=1,
                max_bytes=2,
            ),
        )

        self.assertTrue(reduced.exact_trajectory_equivalence)
        self.assertFalse(reduced.speedup_claimable)
        self.assertTrue(complete.exact_trajectory_equivalence)
        self.assertTrue(complete.speedup_claimable)
        for pair in (*reduced.raw_repetitions, *complete.raw_repetitions):
            self.assertEqual(pair["native"]["native_tokens"], 2)
            self.assertEqual(pair["candidate"]["native_tokens"], 2)
            self.assertEqual(pair["candidate"]["attempted_macro_steps"], 1)
            self.assertEqual(pair["native"]["subphases"]["device_to_host_bytes"], 16)

        reduced_tampered = reduced.to_dict()
        reduced_tampered["speedup_claimable"] = True
        self.assertIn(
            "output speedup claimability verdict is inconsistent",
            output_performance_errors(reduced_tampered),
        )
        complete_tampered = complete.to_dict()
        complete_tampered["speedup_claimable"] = False
        self.assertIn(
            "output speedup claimability verdict is inconsistent",
            output_performance_errors(complete_tampered),
        )
        complete_tampered = complete.to_dict()
        complete_tampered["latency_speedup"] = 0.0
        self.assertIn(
            "output latency speedup is inconsistent",
            output_performance_errors(complete_tampered),
        )
        complete_tampered = complete.to_dict()
        cast(
            list[dict[str, Any]],
            cast(dict[str, Any], complete_tampered["measurement"])["setup_runs"],
        )[0]["native_token_horizon"] = 1
        self.assertIn(
            "output candidate horizon differs from setup run 0",
            output_performance_errors(complete_tampered),
        )

    @unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS") == "1", "set RUN_SLOW_TESTS=1")
    def test_synthetic_output_campaign_proves_end_to_end_path(self) -> None:
        repository = Path(__file__).parents[1]
        spec = ExperimentSpec.load(repository / "experiments/synthetic/output-smoke.toml")
        with tempfile.TemporaryDirectory() as directory:
            result = create_experiment_runner(
                spec,
                Path(directory) / "run",
                repository,
            ).run()

        self.assertEqual(result["scientific_verdict"], "unsupported")
        self.assertEqual(result["mode"], "output_only")
        self.assertEqual(result["evidence_scope"], "synthetic")
        self.assertEqual(result["operational_status"], "completed")
        self.assertFalse(result["output"]["speedup_claimable"])
        self.assertEqual(result["output"]["native_head_invocations"], 0)
        self.assertTrue(result["gates"]["direct_feedback"])
        self.assertTrue(result["gates"]["invalid_events"])
        self.assertTrue(result["gates"]["valid_non_empty_termination"])
        self.assertEqual(
            result["output"]["control_evidence"]["status"],
            "unsupported_zero_coverage",
        )
        self.assertFalse(result["gates"]["control_evidence"])


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "set RUN_MODEL_TESTS=1")
class RealOutputModelTests(unittest.TestCase):
    def test_qwen_hidden_state_path_never_calls_native_head(self) -> None:
        assets = load_model_assets(
            "Qwen/Qwen3.5-0.8B",
            "2fc06364715b967f1860aea9cf38778875588b17",
        )
        model = load_frozen_causal_lm(assets, torch.device("cpu"))
        output_embedding_getter = getattr(model, "get_output_embeddings", None)
        if not callable(output_embedding_getter):
            self.fail("causal language model has no output embedding accessor")
        head = output_embedding_getter()
        token_ids = assets.tokenizer.encode("output-only smoke", add_special_tokens=False)

        with mock.patch.object(
            head,
            "forward",
            side_effect=AssertionError("native vocabulary head was called"),
        ):
            output = FrozenBackbone(model).forward(
                input_ids=torch.tensor([token_ids], dtype=torch.long),
                use_cache=False,
            )

        self.assertEqual(output.last_hidden_state.shape[:2], (1, len(token_ids)))
