from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, final

import torch

from continuous_tokenizer.backbone.assets import (
    ModelAssets,
    load_model_assets,
)
from continuous_tokenizer.backbone.frozen import FrozenBackbone
from continuous_tokenizer.backbone.synthetic import SYNTHETIC_MODEL_ID
from continuous_tokenizer.campaigns.lifecycle import (
    ExperimentLifecycle,
    ProspectiveBudgetExhaustedError,
    ProspectivePolicy,
)
from continuous_tokenizer.codec.checkpoints import save_output_checkpoint
from continuous_tokenizer.codec.output import OutputByteCodec, OutputByteCodecConfig
from continuous_tokenizer.contracts.claim_derivation import (
    derive_output_claim_verdicts,
)
from continuous_tokenizer.contracts.claims import claim_records
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.output import (
    OutputCorpusSpec,
    OutputEvaluationSpec,
    OutputGateSpec,
    OutputTrainingSpec,
    registered_output_prompts,
)
from continuous_tokenizer.contracts.prospective import prospective_stage_records
from continuous_tokenizer.data.corpus import load_corpus_documents
from continuous_tokenizer.output.benchmark import (
    OutputBenchmark,
    OutputBenchmarkOptions,
    benchmark_output_generation,
)
from continuous_tokenizer.output.corpora import select_output_documents
from continuous_tokenizer.output.evaluation import (
    OutputEvaluationOptions,
    OutputMetrics,
    OutputRolloutMetrics,
    OutputRolloutOptions,
    evaluate_output_codec,
    evaluate_output_rollouts,
)
from continuous_tokenizer.output.generation import (
    OutputOnlyGenerator,
    output_stop_control_ids,
    output_stop_control_metadata,
)
from continuous_tokenizer.output.training import (
    OutputCodecTrainer,
    OutputTrainerContext,
    OutputTrainingOptions,
    OutputTrainingResult,
)
from continuous_tokenizer.output.trajectory_cache import (
    OutputCacheIdentity,
    OutputCorpusCacheInfo,
    OutputCorpusPreparation,
    OutputTrajectoryOptions,
    PreparedOutputCorpus,
    native_head_oracle_ceilings,
    oracle_ceiling_passes_gates,
    prepare_output_corpus,
)
from continuous_tokenizer.reporting.artifact_markdown import artifact_report
from continuous_tokenizer.reporting.prospective_markdown import (
    prospective_stop_markdown,
)
from continuous_tokenizer.runtime.environment import runtime_environment
from continuous_tokenizer.runtime.progress import log_event
from continuous_tokenizer.runtime.tensors import parameter_fingerprint

OUTPUT_PROMPT_NATIVE_TOKENS = 64


class OutputOracleCeilingError(RuntimeError):
    def __init__(
        self,
        ceilings: dict[str, dict[str, float | int | bool | None]],
        max_span: int,
    ) -> None:
        super().__init__(f"native-head oracle ceiling is infeasible at max span {max_span}")
        self.ceilings = ceilings
        self.max_span = max_span


def _require_search_oracle_feasibility(
    pilot_corpus: OutputPilotCorpus | None,
    feasible: bool,
    ceilings: dict[str, dict[str, float | int | bool | None]],
    max_span: int,
) -> None:
    if pilot_corpus is not None and not feasible:
        raise OutputOracleCeilingError(ceilings, max_span)


def _token_sequences(
    assets: ModelAssets,
    documents: Sequence[bytes],
    *,
    limit: int,
) -> tuple[tuple[int, ...], ...]:
    if assets.model_id == SYNTHETIC_MODEL_ID:
        start = 0
        if documents and documents[0].startswith(b"validation:"):
            start = 32
        elif documents and documents[0].startswith(b"test:"):
            start = 64
        return (tuple(range(start, start + OUTPUT_PROMPT_NATIVE_TOKENS)),)
    sequences: list[tuple[int, ...]] = []
    for document in documents[:limit]:
        text = document.decode("utf-8")
        token_ids = tuple(assets.tokenizer.encode(text, add_special_tokens=False))
        if token_ids:
            sequences.append(token_ids[:OUTPUT_PROMPT_NATIVE_TOKENS])
    if not sequences:
        raise ValueError("output campaign corpus produced no training sequences")
    return tuple(sequences)


def _registered_prompt_sequences(
    assets: ModelAssets,
    evaluation: OutputEvaluationSpec,
) -> tuple[tuple[int, ...], ...]:
    prompts = registered_output_prompts(
        evaluation.prompt_set,
        evaluation.prompt_set_sha256,
    )
    seed = evaluation.final_test_corpus.seed.to_bytes(8, "big")
    ordered = sorted(
        prompts,
        key=lambda prompt: hashlib.sha256(seed + prompt.encode()).digest(),
    )
    sequences = tuple(tuple(assets.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in ordered)
    if any(not sequence for sequence in sequences):
        raise ValueError("registered output prompt tokenized to an empty sequence")
    return sequences


def _corpus_token_sequences(
    assets: ModelAssets,
    spec: ExperimentSpec,
    corpus: OutputCorpusSpec,
    *,
    limit: int,
    excluded_sha256: frozenset[str] = frozenset(),
) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any], frozenset[str]]:
    documents = load_corpus_documents(
        corpus.split,
        dataset_id=spec.dataset.dataset_id,
        config=spec.dataset.config,
        revision=spec.dataset.revision,
        max_rows=min(spec.runtime.corpus_max_rows, max(limit * 4, limit)),
    )
    selected = select_output_documents(
        documents,
        count=limit,
        seed=corpus.seed,
        excluded_sha256=excluded_sha256,
    )
    return (
        _token_sequences(assets, selected.documents, limit=limit),
        {
            "split": corpus.split,
            "seed": corpus.seed,
            "documents": len(selected.documents),
            "sha256": selected.sha256,
        },
        frozenset(selected.document_sha256),
    )


def _rollout_agreement(generated: bytes, expected: bytes) -> float:
    width = max(len(generated), len(expected), 1)
    matches = sum(left == right for left, right in zip(generated, expected, strict=False))
    return matches / width


def _require_full_model(model: torch.nn.Module | None) -> torch.nn.Module:
    if model is None:
        raise RuntimeError("full output campaign preflight did not load a model")
    return model


def _seeded_output_codec(
    config: OutputByteCodecConfig,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> OutputByteCodec:
    torch.manual_seed(seed)
    return OutputByteCodec(config).to(device=device, dtype=dtype)


@final
@dataclass(frozen=True, slots=True)
class OutputPilotCorpus:
    training_documents: tuple[bytes, ...]
    checkpoint_selection_documents: tuple[bytes, ...]
    oracle_validation_documents: tuple[bytes, ...]


@final
@dataclass(frozen=True, slots=True)
class OutputRunnerOptions:
    verification_path: Path | None = None
    pilot_corpus: OutputPilotCorpus | None = None
    trajectory_cache_directory: Path | None = None
    resume: bool = False
    prospective_policy: ProspectivePolicy | None = None


DEFAULT_OUTPUT_RUNNER_OPTIONS: Final = OutputRunnerOptions()


@final
@dataclass(frozen=True, slots=True)
class OutputSequenceCorpora:
    training: tuple[tuple[int, ...], ...]
    checkpoint_selection: tuple[tuple[int, ...], ...]
    oracle_validation: tuple[tuple[int, ...], ...]
    final_test: tuple[tuple[int, ...], ...]
    training_metadata: dict[str, Any]
    checkpoint_selection_metadata: dict[str, Any]
    oracle_validation_metadata: dict[str, Any]
    final_test_metadata: dict[str, Any]


@final
@dataclass(frozen=True, slots=True)
class OutputTrajectoryPreparation:
    backbone: FrozenBackbone
    assets: ModelAssets
    stop_control_ids: frozenset[int]
    max_native_tokens: int
    max_bytes: int
    model_config_sha256: str
    frozen_backbone_fingerprint: str


@final
@dataclass(frozen=True, slots=True)
class PreparedOutputTrajectories:
    training: PreparedOutputCorpus
    checkpoint_selection: PreparedOutputCorpus
    final_test: PreparedOutputCorpus
    oracle_ceilings: dict[str, dict[str, float | int | bool | None]]
    oracle_feasible: bool
    cache_metadata: dict[str, Any]
    cache_artifact: str


@final
@dataclass(frozen=True, slots=True)
class OutputPublication:
    result: dict[str, Any]
    output_metrics: dict[str, Any]
    artifacts: dict[str, str]
    trainable_parameters: tuple[str, ...]
    assets: ModelAssets
    frozen_backbone_fingerprint: str | None


@final
@dataclass(frozen=True, slots=True)
class OutputCampaignTrainingContext:
    backbone: FrozenBackbone
    codec: OutputByteCodec
    assets: ModelAssets
    trajectories: PreparedOutputTrajectories
    spec: OutputTrainingSpec
    seed: int
    checkpoint_name: str
    artifacts: dict[str, str]


@final
@dataclass(frozen=True, slots=True)
class TrainedOutputCodec:
    training: OutputTrainingResult
    controls: torch.Tensor


@final
@dataclass(frozen=True, slots=True)
class OutputCampaignEvaluationContext:
    backbone: FrozenBackbone
    codec: OutputByteCodec
    assets: ModelAssets
    controls: torch.Tensor
    stop_control_ids: frozenset[int]
    sequences: OutputSequenceCorpora
    trajectories: PreparedOutputTrajectories
    training: OutputTrainingSpec
    evaluation: OutputEvaluationSpec
    gates: OutputGateSpec


@final
@dataclass(frozen=True, slots=True)
class OutputEvaluationResults:
    metrics: OutputMetrics
    benchmark: OutputBenchmark
    rollout: OutputRolloutMetrics
    native_head_invocations: int


def _output_sequence_corpora(
    assets: ModelAssets,
    spec: ExperimentSpec,
    pilot: OutputPilotCorpus | None,
    *,
    limit: int,
    validation_only: bool = False,
) -> OutputSequenceCorpora:
    evaluation = spec.evaluation
    if not isinstance(evaluation, OutputEvaluationSpec):
        raise ValueError("output sequence corpora require output evaluation settings")
    if pilot is not None:
        training = _token_sequences(
            assets,
            pilot.training_documents,
            limit=len(pilot.training_documents),
        )
        checkpoint_selection = _token_sequences(
            assets,
            pilot.checkpoint_selection_documents,
            limit=len(pilot.checkpoint_selection_documents),
        )
        oracle_validation = _token_sequences(
            assets,
            pilot.oracle_validation_documents,
            limit=len(pilot.oracle_validation_documents),
        )
        return OutputSequenceCorpora(
            training=training,
            checkpoint_selection=checkpoint_selection,
            oracle_validation=oracle_validation,
            final_test=oracle_validation,
            training_metadata={"scope": "search_training"},
            checkpoint_selection_metadata={"scope": "search_checkpoint_selection"},
            oracle_validation_metadata={"scope": "search_oracle_validation"},
            final_test_metadata={"scope": "not_executed_in_search"},
        )

    if spec.evidence_scope == "synthetic":
        final_test = _registered_prompt_sequences(assets, evaluation)
        metadata = {
            "scope": "synthetic_registered_prompt_software_validation",
            "prompts": len(final_test),
            "sha256": evaluation.prompt_set_sha256,
        }
        return OutputSequenceCorpora(
            training=final_test,
            checkpoint_selection=final_test,
            oracle_validation=final_test,
            final_test=final_test,
            training_metadata=metadata,
            checkpoint_selection_metadata=metadata,
            oracle_validation_metadata=metadata,
            final_test_metadata=metadata,
        )
    training, training_metadata, training_hashes = _corpus_token_sequences(
        assets,
        spec,
        evaluation.training_corpus,
        limit=limit,
    )
    checkpoint_selection, selection_metadata, _ = _corpus_token_sequences(
        assets,
        spec,
        evaluation.checkpoint_selection_corpus,
        limit=limit,
        excluded_sha256=training_hashes,
    )
    oracle_validation, oracle_metadata, _ = _corpus_token_sequences(
        assets,
        spec,
        evaluation.oracle_validation_corpus,
        limit=limit,
    )
    if validation_only:
        return OutputSequenceCorpora(
            training=training,
            checkpoint_selection=checkpoint_selection,
            oracle_validation=oracle_validation,
            final_test=oracle_validation,
            training_metadata=training_metadata,
            checkpoint_selection_metadata=selection_metadata,
            oracle_validation_metadata=oracle_metadata,
            final_test_metadata={
                "scope": "prospective_oracle_validation",
                "final_test_loaded": False,
            },
        )
    final_test = _registered_prompt_sequences(assets, evaluation)
    return OutputSequenceCorpora(
        training=training,
        checkpoint_selection=checkpoint_selection,
        oracle_validation=oracle_validation,
        final_test=final_test,
        training_metadata=training_metadata,
        checkpoint_selection_metadata=selection_metadata,
        oracle_validation_metadata=oracle_metadata,
        final_test_metadata={
            "split": evaluation.final_test_corpus.split,
            "seed": evaluation.final_test_corpus.seed,
            "prompts": len(final_test),
            "sha256": evaluation.prompt_set_sha256,
        },
    )


@final
class OutputExperimentRunner(ExperimentLifecycle):
    def __init__(
        self,
        spec: ExperimentSpec,
        output_dir: Path,
        project_root: Path,
        options: OutputRunnerOptions = DEFAULT_OUTPUT_RUNNER_OPTIONS,
    ) -> None:
        if spec.mode != "output_only":
            raise ValueError("output runner requires an output-only experiment")
        super().__init__(
            spec,
            output_dir,
            project_root,
            options.verification_path,
            resume=options.resume,
            prospective_policy=options.prospective_policy,
        )
        self.pilot_corpus = options.pilot_corpus
        self.trajectory_cache_directory = options.trajectory_cache_directory

    def _structurally_unrepresentable(self) -> bool:
        return self.pilot_corpus is None and any(
            not selection.feasible
            for selection in (
                *self.spec.search_selections,
                *self.spec.study_selections,
            )
        )

    def _publish_structurally_unsupported(self) -> dict[str, Any]:
        reason = "the sealed native-head oracle proves the registered output targets structurally unrepresentable; training and evaluation were not performed"
        verdicts = derive_output_claim_verdicts(
            (),
            (),
            complete=True,
            structurally_unrepresentable=True,
        )
        claims = claim_records("output_only", verdicts, reason=reason)
        output_metrics = {
            "oracle_feasible": False,
            "training_performed": False,
            "structurally_unrepresentable": True,
            "reason": reason,
            "native_head_invocations": None,
        }
        result = {
            "mode": "output_only",
            "evidence_scope": self.spec.evidence_scope,
            "operational_status": "completed",
            "scientific_verdict": "unsupported",
            "experiment": self.spec.to_dict(),
            "training": {
                "performed": False,
                "status": "not_started_structurally_unrepresentable",
                "reason": reason,
            },
            "output": output_metrics,
            "gates": {"oracle_feasible": False},
            "claims": claims,
            "verification": self.verification,
            "runtime": {
                "stages": [],
                "final": runtime_environment(self.device),
            },
        }
        self._write_start_manifest(None)
        self.run_directory.write_json("output-metrics.json", output_metrics)
        self.run_directory.write_json("result.json", result)
        self.run_directory.write_text(
            "artifact-report.md",
            artifact_report(result),
        )
        artifacts = {
            "output_metrics": "output-metrics.json",
            "result": "result.json",
            "report": "artifact-report.md",
        }
        self._finalize_success(
            artifacts,
            trainable_parameters=(),
            assets=None,
            frozen_backbone_fingerprint=None,
        )
        return result

    def _publish_prospective_stop(  # noqa: PLR0913 - Sealing context remains explicit.
        self,
        assets: ModelAssets,
        artifacts: dict[str, str],
        *,
        stop_reason: str,
        boundary: str,
        elapsed_seconds: float | None = None,
        training: dict[str, Any] | None = None,
        output_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        budget_exhausted = stop_reason == "budget"
        pending_status = "not_run_budget" if budget_exhausted else "not_run_futility"
        if stop_reason == "oracle":
            statuses = {
                "oracle": "failed",
                "training": pending_status,
                "exactness": pending_status,
                "density": pending_status,
                "behavior": pending_status,
                "generation": pending_status,
            }
            gates = {
                "oracle_feasible": {
                    "status": "failed",
                    "passed": False,
                },
                "direct_feedback": {
                    "status": pending_status,
                    "passed": None,
                },
                "macro_step_density": {
                    "status": pending_status,
                    "passed": None,
                },
                "behavioral_similarity": {
                    "status": pending_status,
                    "passed": None,
                },
            }
        elif stop_reason == "exactness":
            statuses = {
                "oracle": "passed",
                "training": "completed",
                "exactness": "failed",
                "density": "not_run_futility",
                "behavior": "not_run_futility",
                "generation": "not_run_futility",
            }
            gates = {
                "oracle_feasible": {
                    "status": "passed",
                    "passed": True,
                },
                "direct_feedback": {
                    "status": "failed",
                    "passed": False,
                },
                "macro_step_density": {
                    "status": "not_run_futility",
                    "passed": None,
                },
                "behavioral_similarity": {
                    "status": "not_run_futility",
                    "passed": None,
                },
            }
        else:
            statuses = {
                "oracle": "not_run_budget",
                "training": "stopped_budget",
                "exactness": "not_run_budget",
                "density": "not_run_budget",
                "behavior": "not_run_budget",
                "generation": "not_run_budget",
            }
            gates = {
                name: {
                    "status": "not_run_budget",
                    "passed": None,
                }
                for name in (
                    "oracle_feasible",
                    "direct_feedback",
                    "macro_step_density",
                    "behavioral_similarity",
                )
            }
        result = {
            "mode": "output_only",
            "evidence_scope": self.spec.evidence_scope,
            "operational_status": "completed",
            "scientific_verdict": "unsupported",
            "gates_passed": False,
            "experiment": self.spec.to_dict(),
            "training": training,
            "output": output_metrics,
            "gates": gates,
            "claims": [],
            "verification": self.verification,
            "runtime": {
                "stages": list(self.stage_timings),
                "final": runtime_environment(self.device),
                "recovery_snapshots": self.resume_manager.telemetry(),
            },
            "prospective_execution": {
                "budget_exhausted": budget_exhausted,
                "stop_reason": stop_reason,
                "boundary": boundary,
                "elapsed_seconds": elapsed_seconds,
                "stages": prospective_stage_records(statuses),
            },
        }
        self.run_directory.write_json("result.json", result)
        if output_metrics is not None:
            self.run_directory.write_json(
                "output-metrics.json",
                output_metrics,
            )
            artifacts["output_metrics"] = "output-metrics.json"
        self.run_directory.write_text(
            "artifact-report.md",
            prospective_stop_markdown(
                self.spec.name,
                stop_reason,
                boundary,
                statuses,
            ),
        )
        artifacts["result"] = "result.json"
        artifacts["report"] = "artifact-report.md"
        self._finalize_success(
            artifacts,
            trainable_parameters=(),
            assets=assets,
            frozen_backbone_fingerprint=None,
        )
        return result

    def _prepare_trajectory(
        self,
        preparation: OutputTrajectoryPreparation,
        sequences: tuple[tuple[int, ...], ...],
        *,
        split: str,
        max_span: int,
    ) -> tuple[PreparedOutputCorpus, OutputCorpusCacheInfo]:
        source_commit, source_dirty, source_state_sha256 = self.source_state
        return prepare_output_corpus(
            preparation.backbone,
            preparation.assets.vocabulary,
            sequences,
            OutputCorpusPreparation(
                identity=OutputCacheIdentity(
                    source_commit=source_commit,
                    source_dirty=source_dirty,
                    source_state_sha256=source_state_sha256,
                    dependency_lock_sha256=self.dependency_lock_sha256,
                    model_revision=preparation.assets.revision,
                    model_config_sha256=preparation.model_config_sha256,
                    frozen_backbone_fingerprint=preparation.frozen_backbone_fingerprint,
                    tokenizer_revision=preparation.assets.revision,
                ),
                split=split,
                trajectory=OutputTrajectoryOptions(
                    max_span=max_span,
                    stop_control_ids=preparation.stop_control_ids,
                    max_native_tokens=preparation.max_native_tokens,
                    max_bytes=preparation.max_bytes,
                ),
                cache_directory=self.trajectory_cache_directory,
            ),
        )

    def _prepare_final_trajectory(
        self,
        preparation: OutputTrajectoryPreparation,
        sequences: tuple[tuple[int, ...], ...],
        oracle_corpus: PreparedOutputCorpus,
        *,
        max_span: int,
    ) -> tuple[PreparedOutputCorpus, OutputCorpusCacheInfo | None]:
        if self.pilot_corpus is not None:
            return oracle_corpus, None
        return self._prepare_trajectory(
            preparation,
            sequences,
            split="final-test",
            max_span=max_span,
        )

    def _prepare_output_trajectories(
        self,
        preparation: OutputTrajectoryPreparation,
        sequences: OutputSequenceCorpora,
        codec: OutputByteCodec,
        evaluation: OutputEvaluationSpec,
        gates: OutputGateSpec,
    ) -> PreparedOutputTrajectories:
        assets = preparation.assets
        oracle_corpus, oracle_cache = self._prepare_trajectory(
            preparation,
            sequences.oracle_validation,
            split="oracle-validation",
            max_span=max(
                assets.vocabulary.max_token_bytes,
                *evaluation.oracle_span_limits,
            ),
        )
        oracle_ceilings = native_head_oracle_ceilings(
            oracle_corpus,
            assets.vocabulary,
            span_limits=evaluation.oracle_span_limits,
        )
        oracle_feasible = oracle_ceiling_passes_gates(
            oracle_ceilings[str(codec.max_span)],
            gates,
        )
        _require_search_oracle_feasibility(
            self.pilot_corpus,
            oracle_feasible,
            oracle_ceilings,
            codec.max_span,
        )
        training_corpus, training_cache = self._prepare_trajectory(
            preparation,
            sequences.training,
            split="training",
            max_span=codec.max_span,
        )
        selection_corpus, selection_cache = self._prepare_trajectory(
            preparation,
            sequences.checkpoint_selection,
            split="selection",
            max_span=codec.max_span,
        )
        final_corpus, final_cache = self._prepare_final_trajectory(
            preparation,
            sequences.final_test,
            oracle_corpus,
            max_span=codec.max_span,
        )
        cache_metadata = {
            "training": training_cache.to_dict() | {"corpus": sequences.training_metadata},
            "checkpoint_selection": selection_cache.to_dict() | {"corpus": sequences.checkpoint_selection_metadata},
            "oracle_validation": oracle_cache.to_dict() | {"corpus": sequences.oracle_validation_metadata},
            "final_test": (
                {"executed": False, "corpus": sequences.final_test_metadata}
                if final_cache is None
                else final_cache.to_dict() | {"corpus": sequences.final_test_metadata}
            ),
        }
        caches = (training_cache, selection_cache, oracle_cache)
        if final_cache is not None:
            caches = (*caches, final_cache)
        for cache in {cache.key: cache for cache in caches}.values():
            log_event("output_trajectory_cache", **cache.to_dict())
        cache_artifact = "output-trajectory-cache-resume.json" if self.resuming else "output-trajectory-cache.json"
        self.run_directory.write_json(cache_artifact, cache_metadata)
        return PreparedOutputTrajectories(
            training=training_corpus,
            checkpoint_selection=selection_corpus,
            final_test=final_corpus,
            oracle_ceilings=oracle_ceilings,
            oracle_feasible=oracle_feasible,
            cache_metadata=cache_metadata,
            cache_artifact=cache_artifact,
        )

    def _train_output_codec(
        self,
        context: OutputCampaignTrainingContext,
    ) -> TrainedOutputCodec:
        run = self.run_directory
        trainer = OutputCodecTrainer(
            context.codec,
            OutputTrainingOptions(
                epochs=context.spec.epochs,
                batch_size=context.spec.batch_size,
                learning_rate=context.spec.learning_rate,
                weight_decay=context.spec.weight_decay,
                seed=context.seed,
            ),
            OutputTrainerContext(
                backbone=context.backbone,
                vocabulary=context.assets.vocabulary,
                deployment_dtype=context.assets.input_embeddings.dtype,
                progress_directory=run.path("checkpoints/progress"),
                resume_manager=self.resume_manager,
                epoch_boundary=self._prospective_epoch_boundary,
            ),
        )
        with self._stage("train_output_codec"):
            training = trainer.run(
                context.trajectories.training,
                context.trajectories.checkpoint_selection,
            )
        context.codec.to(dtype=context.assets.input_embeddings.dtype)
        if self.device.type == "mps":
            context.codec.compile_neural_paths()
        checkpoint = run.path(f"checkpoints/{context.checkpoint_name}-output.pt")
        controls = torch.tensor(
            context.assets.vocabulary.control_ids,
            dtype=torch.long,
            device=self.device,
        )
        with self._stage("save_output_checkpoint"):
            save_output_checkpoint(
                checkpoint,
                context.codec,
                {
                    "model_id": context.assets.model_id,
                    "model_revision": context.assets.revision,
                    "mode": "output_only",
                    "training": training.to_dict(),
                },
                control_ids=controls,
            )
        context.artifacts["checkpoint"] = str(checkpoint.relative_to(run.root))
        context.artifacts["training_progress"] = "checkpoints/progress"
        return TrainedOutputCodec(training=training, controls=controls)

    def _evaluate_trained_output(
        self,
        context: OutputCampaignEvaluationContext,
    ) -> OutputEvaluationResults:
        with self._stage("evaluate_output_codec"):
            metrics = evaluate_output_codec(
                context.codec,
                context.trajectories.final_test,
                context.assets.vocabulary,
                OutputEvaluationOptions(
                    batch_size=context.evaluation.batch_size,
                    resume_manager=self.resume_manager,
                    resume_phase="output-selected-evaluation",
                ),
            )
            generator = OutputOnlyGenerator(
                context.backbone,
                context.codec,
                context.assets.vocabulary,
                context.controls,
            )
            benchmark = benchmark_output_generation(
                generator,
                context.sequences.final_test,
                OutputBenchmarkOptions(
                    warmups=context.evaluation.warmups,
                    repetitions=context.evaluation.repetitions,
                    stop_control_ids=context.stop_control_ids,
                    max_macro_steps=context.evaluation.max_macro_steps,
                    max_bytes=context.evaluation.max_output_bytes,
                ),
            )
            rollout = evaluate_output_rollouts(
                context.backbone,
                generator,
                context.assets.vocabulary,
                context.sequences.final_test,
                OutputRolloutOptions(
                    stop_control_ids=context.stop_control_ids,
                    max_macro_steps=context.evaluation.max_macro_steps,
                    max_bytes=context.evaluation.max_output_bytes,
                ),
            )
        return OutputEvaluationResults(
            metrics=metrics,
            benchmark=benchmark,
            rollout=rollout,
            native_head_invocations=generator.native_head_invocations,
        )

    def _output_evidence(
        self,
        context: OutputCampaignEvaluationContext,
        results: OutputEvaluationResults,
    ) -> tuple[dict[str, Any], dict[str, bool]]:
        metrics = results.metrics
        benchmark = results.benchmark
        rollout = results.rollout
        evaluation = context.evaluation
        gates = context.gates
        output_metrics = {
            **metrics.to_dict(),
            **benchmark.to_dict(),
            **rollout.to_dict(),
            "bytes_per_macro_step": rollout.output_bytes_per_macro_step,
            "direct_invalid_events": metrics.invalid_events,
            "rollout_invalid_events": rollout.invalid_events,
            "invalid_events": metrics.invalid_events + rollout.invalid_events,
            "direct_feedback_equality": min(
                metrics.direct_feedback_byte_equality,
                metrics.direct_feedback_token_equality,
            ),
            "prompt_set": {
                "name": (evaluation.prompt_set if self.pilot_corpus is None else "search-oracle-validation"),
                "sha256": (
                    evaluation.prompt_set_sha256 if self.pilot_corpus is None else context.trajectories.cache_metadata["oracle_validation"]["corpus_sha256"]
                ),
                "prompts": len(context.sequences.final_test),
            },
            "fidelity": {
                "direct_feedback_byte_equality": metrics.direct_feedback_byte_equality,
                "direct_feedback_token_equality": metrics.direct_feedback_token_equality,
                "rollout_event_agreement": rollout.rollout_event_agreement,
                "rollout_byte_agreement": rollout.rollout_byte_agreement,
                "rollout_token_agreement": rollout.rollout_token_agreement,
            },
            "output_density": {
                "bytes_per_macro_step": rollout.output_bytes_per_macro_step,
                "native_tokens_per_attempted_macro_step": (rollout.native_tokens_per_attempted_macro_step),
            },
            "control_evidence": {
                "coverage": rollout.control_prompt_coverage,
                "oracle_events": rollout.oracle_control_events,
                "predicted_events": rollout.predicted_control_events,
                "correct_events": rollout.correct_control_events,
                "precision": rollout.control_precision,
                "recall": rollout.control_recall,
                "false_positives": rollout.control_false_positives,
                "false_negatives": rollout.control_false_negatives,
                "status": ("unsupported_zero_coverage" if rollout.oracle_control_events == 0 else "measured"),
            },
            "native_head_oracle_ceilings": context.trajectories.oracle_ceilings,
            "oracle_feasible": context.trajectories.oracle_feasible,
            "stop_control": output_stop_control_metadata(
                context.assets.tokenizer,
                context.assets.vocabulary,
            )
            | {
                "oracle_events": rollout.oracle_stop_control_events,
                "predicted_events": rollout.predicted_stop_control_events,
                "correct_events": rollout.correct_stop_control_events,
                "coverage": rollout.stop_prompt_coverage,
                "precision": rollout.stop_precision,
                "recall": rollout.stop_recall,
                "false_positives": rollout.stop_false_positives,
                "false_negatives": rollout.stop_false_negatives,
            },
            "termination_reasons": {
                "stop_control": rollout.termination_stop_control,
                "invalid_event": rollout.termination_invalid_event,
                "max_bytes_truncated": rollout.termination_max_bytes_truncated,
                "max_bytes": rollout.termination_max_bytes,
                "max_macro_steps": rollout.termination_max_macro_steps,
            },
            "native_head_invocations": results.native_head_invocations,
            "deployment": None,
            "trajectory_cache": context.trajectories.cache_metadata,
        }
        gate_results = {
            "direct_feedback": (
                metrics.direct_feedback_byte_equality >= gates.minimum_direct_feedback_equality
                and metrics.direct_feedback_token_equality >= gates.minimum_direct_feedback_equality
            ),
            "invalid_events": (metrics.invalid_events + rollout.invalid_events <= gates.maximum_invalid_events),
            "valid_non_empty_termination": (metrics.valid_non_empty_termination >= gates.minimum_valid_non_empty_termination),
            "native_tokens_per_attempted_macro_step": (rollout.native_tokens_per_attempted_macro_step >= gates.minimum_native_tokens_per_attempted_macro_step),
            "rollout_event_agreement": (rollout.rollout_event_agreement >= gates.minimum_rollout_event_agreement),
            "control_evidence": (
                rollout.control_prompt_coverage >= gates.minimum_control_prompt_coverage
                and rollout.control_precision is not None
                and rollout.control_precision >= gates.minimum_control_precision
                and rollout.control_recall is not None
                and rollout.control_recall >= gates.minimum_control_recall
            ),
            "stop_control": (
                rollout.stop_precision is not None
                and rollout.stop_precision >= gates.minimum_stop_precision
                and rollout.stop_recall is not None
                and rollout.stop_recall >= gates.minimum_stop_recall
            ),
            "oracle_feasible": context.trajectories.oracle_feasible,
            "candidate_reference_state_ratio": (
                benchmark.candidate_reference_state_ratio is not None
                and benchmark.candidate_reference_state_ratio <= gates.maximum_candidate_reference_state_ratio
            ),
        }
        return output_metrics, gate_results

    def run(self) -> dict[str, Any]:  # noqa: C901, PLR0915 - Campaign stages remain explicit.
        spec = self.spec
        if not isinstance(spec.training, OutputTrainingSpec):
            raise ValueError("output runner requires output training settings")
        if not isinstance(spec.evaluation, OutputEvaluationSpec):
            raise ValueError("output runner requires output evaluation settings")
        if not isinstance(spec.gates, OutputGateSpec):
            raise ValueError("output runner requires output gates")
        self._write_experiment_contract()
        if self._structurally_unrepresentable():
            return self._publish_structurally_unsupported()
        artifacts: dict[str, str] = {}
        trainable: tuple[str, ...] = ()
        assets: ModelAssets | None = None
        frozen_fingerprint: str | None = None
        try:
            with self._stage("load_model_assets"):
                assets = load_model_assets(spec.model.model_id, spec.model.revision)
                self._write_start_manifest(assets)
            with self._stage("run_preflight"):
                _, source_model = self._run_preflight(assets)
                artifacts["preflight"] = "preflight.json"
            with self._stage("prepare_output_model"):
                backbone = FrozenBackbone(_require_full_model(source_model))
                profile = self.profile
                codec = _seeded_output_codec(
                    OutputByteCodecConfig(
                        embedding_dim=assets.input_embeddings.shape[1],
                        local_dim=profile.local_dim,
                        max_span=spec.training.max_span,
                        feedforward_dim=profile.feedforward_dim,
                        decoder_layers=profile.decoder_layers,
                        control_count=len(assets.vocabulary.control_ids),
                    ),
                    seed=spec.seed,
                    device=self.device,
                    dtype=torch.float32,
                )
                if self.device.type == "mps":
                    codec.compile_neural_paths()
            sequence_limit = max(spec.evaluation.samples, 1)
            with self._stage("load_output_corpus"):
                stop_control_ids = output_stop_control_ids(
                    assets.tokenizer,
                    assets.vocabulary,
                )
                sequence_corpora = _output_sequence_corpora(
                    assets,
                    spec,
                    self.pilot_corpus,
                    limit=sequence_limit,
                    validation_only=(self.prospective_policy is not None and self.prospective_policy.tier != "final_evidence"),
                )
            with self._stage("prepare_output_trajectories"):
                model_config_sha256 = hashlib.sha256(
                    json.dumps(
                        assets.config,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                trajectory_backbone_fingerprint = parameter_fingerprint(backbone.source_model)
                preparation = OutputTrajectoryPreparation(
                    backbone=backbone,
                    assets=assets,
                    stop_control_ids=stop_control_ids,
                    max_native_tokens=spec.evaluation.max_macro_steps,
                    max_bytes=spec.evaluation.max_output_bytes,
                    model_config_sha256=model_config_sha256,
                    frozen_backbone_fingerprint=trajectory_backbone_fingerprint,
                )
                trajectories = self._prepare_output_trajectories(
                    preparation,
                    sequence_corpora,
                    codec,
                    spec.evaluation,
                    spec.gates,
                )
                artifacts["trajectory_cache"] = trajectories.cache_artifact
            if self.prospective_policy is not None and self.prospective_policy.futility_enabled and not trajectories.oracle_feasible:
                return self._publish_prospective_stop(
                    assets,
                    artifacts,
                    stop_reason="oracle",
                    boundary="stage:prepare_output_trajectories",
                    output_metrics={
                        "oracle_feasible": False,
                        "native_head_oracle_ceilings": (trajectories.oracle_ceilings),
                        "training_performed": False,
                    },
                )
            trained = self._train_output_codec(
                OutputCampaignTrainingContext(
                    backbone=backbone,
                    codec=codec,
                    assets=assets,
                    trajectories=trajectories,
                    spec=spec.training,
                    seed=spec.seed,
                    checkpoint_name=profile.name,
                    artifacts=artifacts,
                ),
            )
            training = trained.training
            trainable = training.trainable_parameters
            frozen_fingerprint = training.backbone_fingerprint
            controls = trained.controls
            if self.prospective_policy is not None and self.prospective_policy.futility_enabled and training.exact_event_agreement < 1.0:
                return self._publish_prospective_stop(
                    assets,
                    artifacts,
                    stop_reason="exactness",
                    boundary="stage:train_output_codec",
                    training=training.to_dict(),
                    output_metrics={
                        "oracle_feasible": True,
                        "native_head_oracle_ceilings": (trajectories.oracle_ceilings),
                        "training_performed": True,
                        "exact_event_agreement": (training.exact_event_agreement),
                    },
                )
            evaluation_context = OutputCampaignEvaluationContext(
                backbone=backbone,
                codec=codec,
                assets=assets,
                controls=controls,
                stop_control_ids=stop_control_ids,
                sequences=sequence_corpora,
                trajectories=trajectories,
                training=spec.training,
                evaluation=spec.evaluation,
                gates=spec.gates,
            )
            evaluation_results = self._evaluate_trained_output(evaluation_context)
            output_metrics, gates = self._output_evidence(
                evaluation_context,
                evaluation_results,
            )
            claim_verdicts = derive_output_claim_verdicts(
                (output_metrics,),
                (asdict(spec.gates),),
                complete=spec.evidence_scope in {"final", "synthetic"},
            )
            result = {
                **self._result_metadata(
                    assets,
                    gates_passed=all(gates.values()),
                    search=self.pilot_corpus is not None,
                ),
                **self._reporting_context(assets),
                "experiment": spec.to_dict(),
                "training": training.to_dict(),
                "output": output_metrics,
                "gates": gates,
                "claims": claim_records("output_only", claim_verdicts),
            }
            with self._stage("publish_result"):
                publication = OutputPublication(
                    result=result,
                    output_metrics=output_metrics,
                    artifacts=artifacts,
                    trainable_parameters=trainable,
                    assets=assets,
                    frozen_backbone_fingerprint=frozen_fingerprint,
                )
                self._publish_result(publication)
            self._finalize_success(
                publication.artifacts,
                trainable_parameters=publication.trainable_parameters,
                assets=publication.assets,
                frozen_backbone_fingerprint=(publication.frozen_backbone_fingerprint),
            )
            return publication.result  # noqa: TRY300 - Successful publication exits the campaign.
        except ProspectiveBudgetExhaustedError as error:
            if assets is None:
                raise
            if (self.run_directory.root / "preflight.json").is_file():
                artifacts["preflight"] = "preflight.json"
            if (self.run_directory.root / "checkpoints/progress").is_dir():
                artifacts["training_progress"] = "checkpoints/progress"
            return self._publish_prospective_stop(
                assets,
                artifacts,
                stop_reason="budget",
                boundary=error.boundary,
                elapsed_seconds=error.elapsed_seconds,
            )
        except Exception as error:
            self._finalize_failure(
                error,
                artifacts,
                trainable_parameters=trainable,
                assets=assets,
                frozen_backbone_fingerprint=frozen_fingerprint,
            )
            raise

    def _publish_result(
        self,
        publication: OutputPublication,
    ) -> None:
        self.run_directory.write_json(
            "output-metrics.json",
            publication.output_metrics,
        )
        self.run_directory.write_json("result.json", publication.result)
        self.run_directory.write_text(
            "artifact-report.md",
            artifact_report(publication.result),
        )
        publication.artifacts.update(
            output_metrics="output-metrics.json",
            result="result.json",
            report="artifact-report.md",
        )
