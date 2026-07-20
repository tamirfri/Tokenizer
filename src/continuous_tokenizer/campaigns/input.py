from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast, final

from torch import nn

from continuous_tokenizer.artifacts.store import load_json_object
from continuous_tokenizer.backbone.assets import (
    ModelAssets,
    load_frozen_causal_lm,
    load_model_assets,
)
from continuous_tokenizer.backbone.config import input_table_is_removable
from continuous_tokenizer.campaigns.lifecycle import (
    ExperimentLifecycle,
    ProspectiveBudgetExhaustedError,
    ProspectivePolicy,
)
from continuous_tokenizer.codec.batches import span_bucket_width
from continuous_tokenizer.codec.checkpoints import load_checkpoint
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import (
    InputEvaluationSpec,
    InputGateSpec,
    InputTrainingSpec,
)
from continuous_tokenizer.contracts.prospective import prospective_stage_records
from continuous_tokenizer.contracts.prospective_subset import (
    PROSPECTIVE_INPUT_SUBSET_FILENAME,
    PROSPECTIVE_INPUT_SUBSET_KIND,
    prospective_vocabulary_subset_errors,
)
from continuous_tokenizer.data.corpus import load_corpus_documents, sample_content_windows
from continuous_tokenizer.input.adapter import InputEmbeddingAdapter
from continuous_tokenizer.input.benchmark.run import BenchmarkOptions, benchmark_experiment
from continuous_tokenizer.input.benchmark.tokenizer import (
    TokenizerMetricRequest,
    tokenizer_metrics,
)
from continuous_tokenizer.input.evaluation import (
    EvaluationOptions,
    EvaluationRuntime,
    EvaluationSession,
    TeacherForcedBatchPolicy,
    evaluate_input_replacement,
    evaluate_input_selection,
    evaluation_options_from_spec,
    teacher_forced_policy_from_spec,
)
from continuous_tokenizer.input.studies import (
    RegisteredVocabularySubsetRequest,
    VocabularySubset,
    input_behavior_gates,
    registered_vocabulary_subset,
    select_input_candidate,
)
from continuous_tokenizer.input.training.distillation import (
    DistillationOptions,
    DistillationRequest,
    distill_checkpoint,
)
from continuous_tokenizer.input.training.run import (
    TrainingOptions,
    TrainingResult,
    train_experiment,
    training_options_from_spec,
)
from continuous_tokenizer.reporting.artifact_markdown import artifact_report
from continuous_tokenizer.reporting.discovery import load_training_progress
from continuous_tokenizer.reporting.prospective_markdown import (
    prospective_stop_markdown,
)


@final
@dataclass(frozen=True, slots=True)
class _MeasurementRequest:
    checkpoint: Path
    output_dir: Path
    alignment: Literal["aligned", "arbitrary"]
    resume_phase: str


@final
@dataclass(frozen=True, slots=True)
class _TrainingState:
    result: TrainingResult
    reconstruction_checkpoint: Path
    trainable: tuple[str, ...]


@final
@dataclass(frozen=True, slots=True)
class _SelectionState:
    checkpoint: Path
    trainable: tuple[str, ...]
    distillation: dict[str, Any]
    input_selection: dict[str, Any] | None
    ablations: dict[str, Any]


def _required_frozen_model(model: nn.Module | None) -> nn.Module:
    if model is None:
        raise RuntimeError("input distillation selection requires a frozen model")
    return model


def _behavior_gates(
    metrics: dict[str, Any] | None,
    gates: InputGateSpec,
) -> dict[str, bool]:
    if metrics is None:
        return {}
    return input_behavior_gates(metrics, gates)


class InputExperimentRunner(ExperimentLifecycle):
    def __init__(  # noqa: PLR0913 - Runner dependencies remain explicit.
        self,
        spec: ExperimentSpec,
        output_dir: Path,
        project_root: Path,
        verification_path: Path | None = None,
        *,
        resume: bool = False,
        prospective_subset: RegisteredVocabularySubsetRequest | None = None,
        prospective_policy: ProspectivePolicy | None = None,
    ) -> None:
        self.prospective_subset_request = prospective_subset
        self.prospective_subset: VocabularySubset | None = None
        self.prospective_subset_descriptor: dict[str, Any] | None = None
        resume_fingerprint = spec.fingerprint() if prospective_subset is None else prospective_subset.execution_fingerprint(spec.fingerprint())
        if prospective_policy is not None:
            resume_fingerprint = prospective_policy.execution_fingerprint(
                resume_fingerprint,
            )
        super().__init__(
            spec,
            output_dir,
            project_root,
            verification_path,
            resume=resume,
            resume_fingerprint=resume_fingerprint,
            prospective_policy=prospective_policy,
        )

    def _validate_resume(self) -> None:
        super()._validate_resume()
        path = self.run_directory.root / PROSPECTIVE_INPUT_SUBSET_FILENAME
        request = self.prospective_subset_request
        if request is None:
            if path.exists():
                raise ValueError(
                    "interrupted run unexpectedly contains a prospective vocabulary subset",
                )
            return
        if not path.is_file():
            raise ValueError(
                "interrupted prospective run has no vocabulary subset artifact",
            )
        artifact = load_json_object(path)
        errors = prospective_vocabulary_subset_errors(artifact)
        expected = {
            "requested_rows": request.requested_rows,
            "subset_seed": request.subset_seed,
            "algorithm": request.algorithm,
            "subset_sha256": request.subset_sha256,
        }
        if errors or any(artifact.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "interrupted run has a different prospective vocabulary subset",
            )

    def _training_options(self) -> TrainingOptions:
        options = training_options_from_spec(
            self.spec,
            self.run_directory.path("checkpoints"),
        )
        if self.prospective_subset is None:
            return options
        return replace(
            options,
            vocabulary_token_ids=self.prospective_subset.token_ids,
        )

    def _prospective_work_units(self, selected_rows: int) -> dict[str, int]:
        training = self.spec.training
        evaluation = self.spec.evaluation
        if not isinstance(training, InputTrainingSpec) or not isinstance(
            evaluation,
            InputEvaluationSpec,
        ):
            raise ValueError("prospective input subset requires input-only settings")
        return {
            "behavior_samples": evaluation.samples,
            "distillation_windows": training.distillation_windows,
            "generation_samples": evaluation.generation_samples,
            "reconstruction_samples": training.reconstruction_samples,
            "validation_bytes": training.validation_bytes,
            "vocabulary_epochs": training.vocabulary_epochs,
            "vocabulary_rows": selected_rows,
        }

    def _prospective_subset_artifact(
        self,
        assets: ModelAssets,
        subset: VocabularySubset,
    ) -> dict[str, Any]:
        training = self.spec.training
        request = self.prospective_subset_request
        if not isinstance(training, InputTrainingSpec) or request is None:
            raise ValueError(
                "prospective input subset requires input-only training settings",
            )
        maximum_span = max(assets.vocabulary.max_token_bytes, 32)
        bucket_counts = Counter(
            span_bucket_width(
                len(assets.vocabulary.bytes_for(token_id)),
                max_span=maximum_span,
            )
            for token_id in subset.token_ids
        )
        width_buckets = [
            {
                "width": width,
                "rows": rows,
                "batches": (rows + training.batch_size - 1) // training.batch_size,
            }
            for width, rows in sorted(bucket_counts.items())
        ]
        return {
            "schema_version": 1,
            "artifact_kind": PROSPECTIVE_INPUT_SUBSET_KIND,
            "requested_rows": subset.requested_rows,
            "selected_rows": len(subset.token_ids),
            "subset_seed": request.subset_seed,
            "algorithm": subset.algorithm,
            "subset_sha256": subset.sha256,
            "batch_size": training.batch_size,
            "maximum_span": maximum_span,
            "width_buckets": width_buckets,
            "vocabulary_batches_per_epoch": sum(int(bucket["batches"]) for bucket in width_buckets),
            "rows": [
                {
                    "token_id": token_id,
                    "bytes": assets.vocabulary.bytes_for(token_id).hex(),
                }
                for token_id in subset.token_ids
            ],
        }

    def _prepare_prospective_subset(
        self,
        assets: ModelAssets,
        artifacts: dict[str, str],
    ) -> None:
        request = self.prospective_subset_request
        if request is None:
            return
        subset = registered_vocabulary_subset(
            assets,
            request.requested_rows,
            request.subset_seed,
        )
        compatibility_ids = set(assets.vocabulary.compatibility_ids)
        if (
            subset.requested_rows != request.requested_rows
            or len(subset.token_ids) != request.requested_rows
            or subset.algorithm != request.algorithm
            or subset.sha256 != request.subset_sha256
            or any(token_id not in compatibility_ids or len(assets.vocabulary.bytes_for(token_id)) <= 1 for token_id in subset.token_ids)
        ):
            raise ValueError(
                "computed prospective vocabulary subset differs from its registered contract",
            )
        if dict(request.work_units) != self._prospective_work_units(
            len(subset.token_ids),
        ):
            raise ValueError(
                "prospective work units differ from the bounded child experiment",
            )
        artifact = self._prospective_subset_artifact(assets, subset)
        if errors := prospective_vocabulary_subset_errors(artifact):
            raise ValueError("; ".join(errors))
        path = self.run_directory.root / PROSPECTIVE_INPUT_SUBSET_FILENAME
        if self.resuming:
            if dict(load_json_object(path)) != artifact:
                raise ValueError(
                    "interrupted run vocabulary subset differs from pinned model assets",
                )
        else:
            self.run_directory.write_json(
                PROSPECTIVE_INPUT_SUBSET_FILENAME,
                artifact,
            )
        self.inputs["prospective_vocabulary_subset"] = self._portable_identity(path)
        artifacts["prospective_vocabulary_subset"] = PROSPECTIVE_INPUT_SUBSET_FILENAME
        self.prospective_subset = subset
        self.prospective_subset_descriptor = {key: value for key, value in artifact.items() if key != "rows"} | {"artifact": PROSPECTIVE_INPUT_SUBSET_FILENAME}

    def _evaluation_options(
        self,
        output_dir: Path,
        alignment: Literal["aligned", "arbitrary"],
    ) -> EvaluationOptions:
        return evaluation_options_from_spec(
            self.spec,
            output_dir,
            segmentation_alignment=alignment,
        )

    def _teacher_forced_policy(self) -> TeacherForcedBatchPolicy:
        return teacher_forced_policy_from_spec(self.spec)

    def _evaluation_runtime(
        self,
        model: nn.Module | None,
        resume_phase: str,
        *,
        session: EvaluationSession | None = None,
    ) -> EvaluationRuntime:
        return EvaluationRuntime(
            frozen_model=model,
            resume_manager=self.resume_manager,
            resume_phase=resume_phase,
            session=session,
            teacher_forced_policy=self._teacher_forced_policy(),
            calibration_cache_directory=(self.project_root / ".cache" / "input-evaluation-calibration"),
            dependency_lock_sha256=self.dependency_lock_sha256,
        )

    def _measure(
        self,
        assets: ModelAssets,
        model: nn.Module | None,
        request: _MeasurementRequest,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        spec = self.spec
        if not isinstance(spec.evaluation, InputEvaluationSpec) or not isinstance(spec.training, InputTrainingSpec):
            raise ValueError("input runner requires input-only settings")
        tokenizer = benchmark_experiment(
            assets,
            request.checkpoint,
            request.output_dir,
            BenchmarkOptions(
                max_test_bytes=spec.evaluation.max_test_bytes,
                batch_size=spec.training.batch_size,
                retrieval_rows=spec.evaluation.retrieval_queries,
                repetitions=spec.evaluation.tokenizer_repetitions,
                dataset_id=spec.dataset.dataset_id,
                dataset_config=spec.dataset.config,
                dataset_revision=spec.dataset.revision,
                device=self.device,
            ),
        )
        llm = evaluate_input_replacement(
            assets,
            request.checkpoint,
            self._evaluation_options(request.output_dir, request.alignment),
            self._evaluation_runtime(model, request.resume_phase),
        )
        return tokenizer, llm

    def _distillation_options(
        self,
        alignment: Literal["aligned", "arbitrary"],
    ) -> DistillationOptions:
        training = self.spec.training
        if not isinstance(training, InputTrainingSpec):
            raise ValueError("input runner requires input-only training settings")
        return DistillationOptions(
            epochs=training.distillation_epochs,
            windows=training.distillation_windows,
            prompt_tokens=training.distillation_prompt_tokens,
            continuation_tokens=training.distillation_continuation_tokens,
            vocabulary_replay=training.batch_size,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            seed=self.spec.seed,
            alignment=alignment,
        )

    def _selection_measure(  # noqa: PLR0913 - Campaign context remains explicit.
        self,
        assets: ModelAssets,
        model: nn.Module,
        checkpoint: Path,
        output_dir: Path,
        *,
        name: str,
        session: EvaluationSession,
    ) -> dict[str, Any]:
        spec = self.spec
        training = spec.training
        evaluation = spec.evaluation
        if not isinstance(training, InputTrainingSpec) or not isinstance(
            evaluation,
            InputEvaluationSpec,
        ):
            raise ValueError("input runner requires input-only settings")
        documents = load_corpus_documents(
            "validation",
            dataset_id=spec.dataset.dataset_id,
            config=spec.dataset.config,
            revision=spec.dataset.revision,
            max_rows=spec.runtime.corpus_max_rows,
        )
        validation_windows = sample_content_windows(
            documents,
            maximum_bytes=training.validation_bytes,
        )
        loaded = InputEmbeddingAdapter.from_checkpoint(
            assets,
            checkpoint,
            device=self.device,
        )
        session.register_adapter(checkpoint, self.device, loaded)
        tokenizer = tokenizer_metrics(
            assets,
            checkpoint,
            loaded,
            TokenizerMetricRequest(
                test_windows=validation_windows,
                batch_size=min(
                    training.batch_size,
                    spec.runtime.cache_chunk_rows,
                ),
                retrieval_rows=evaluation.retrieval_queries,
                repetitions=1,
                dataset_id=spec.dataset.dataset_id,
                dataset_config=spec.dataset.config,
                dataset_revision=spec.dataset.revision,
                dataset_split="validation",
            ),
        )
        behavior = evaluate_input_selection(
            assets,
            checkpoint,
            replace(
                self._evaluation_options(output_dir, "arbitrary"),
                warmups=0,
                repetitions=1,
                dataset_split="validation",
            ),
            self._evaluation_runtime(
                model,
                f"input-candidate-selection-{name}",
                session=session,
            ),
        )
        return {
            "name": name,
            "checkpoint": str(checkpoint.relative_to(self.run_directory.root)),
            "tokenizer": tokenizer,
            "validation": behavior,
        }

    def _train_tokenizer(
        self,
        assets: ModelAssets,
        artifacts: dict[str, str],
    ) -> _TrainingState:
        run = self.run_directory
        with self._stage("train_input_tokenizer"):
            result = train_experiment(
                assets,
                self._training_options(),
                device=self.device,
                resume_manager=self.resume_manager,
                epoch_boundary=self._prospective_epoch_boundary,
            )
        run.write_json("training-result.json", asdict(result))
        artifacts["training"] = "training-result.json"
        reconstruction_checkpoint = Path(result.checkpoint)
        trainable: tuple[str, ...] = ()
        if self._training_futility_reason(assets, result) is None:
            with self._stage("load_selected_input_codec"):
                selected_codec = load_checkpoint(reconstruction_checkpoint).codec
            trainable = tuple(name for name, parameter in selected_codec.named_parameters() if parameter.requires_grad)
        if result.compatibility_checkpoint is not None:
            compatibility_checkpoint = Path(result.compatibility_checkpoint)
            artifacts["compatibility_checkpoint"] = str(
                compatibility_checkpoint.relative_to(run.root),
            )
        return _TrainingState(result, reconstruction_checkpoint, trainable)

    def _training_futility_reason(
        self,
        assets: ModelAssets,
        result: TrainingResult,
    ) -> Literal["exactness", "density"] | None:
        policy = self.prospective_policy
        if policy is None or not policy.futility_enabled:
            return None
        gates = self.spec.gates
        if not isinstance(gates, InputGateSpec):
            raise ValueError("input runner requires input-only gates")
        if not result.compatibility_passed or not result.alignment_preserved or not result.round_trip:
            return "exactness"
        compactness_passed = result.candidate_reference_state_ratio <= gates.maximum_candidate_reference_state_ratio
        density_passed = result.native_tokens_per_continuous_token >= gates.minimum_native_tokens_per_continuous_token
        if not density_passed or (input_table_is_removable(assets.config) and not compactness_passed):
            return "density"
        return None

    def _prospective_gate_rows(
        self,
        training: Mapping[str, Any] | None,
        *,
        stop_reason: Literal["exactness", "density", "budget"],
    ) -> dict[str, dict[str, Any]]:
        def measured(name: str) -> bool | None:
            value = None if training is None else training.get(name)
            return value if isinstance(value, bool) else None

        def gate_status(value: bool | None) -> str:
            if value is True:
                return "passed"
            if value is False:
                return "failed"
            return "not_run_budget" if stop_reason == "budget" else "not_run"

        exact_values = {
            "exact_compatibility": measured("compatibility_passed"),
            "embedding_alignment": measured("alignment_preserved"),
            "exact_round_trip": measured("round_trip"),
        }
        gates = {
            name: {
                "status": gate_status(value),
                "passed": value,
            }
            for name, value in exact_values.items()
        }
        if stop_reason == "exactness" and all(value is True for value in exact_values.values()):
            gates["selected_exactness"] = {
                "status": "failed",
                "passed": False,
            }
        exact_passed = all(value is True for value in exact_values.values()) and "selected_exactness" not in gates
        if stop_reason == "budget":
            density_status = "not_run_budget"
        elif exact_passed:
            density_status = "failed"
        else:
            density_status = "not_run_futility"
        gates["held_out_density"] = {
            "status": density_status,
            "passed": False if density_status == "failed" else None,
        }
        gates["behavioral_similarity"] = {
            "status": ("not_run_budget" if stop_reason == "budget" else "not_run_futility"),
            "passed": None,
        }
        return gates

    def _publish_prospective_stop(  # noqa: PLR0913 - Sealing context remains explicit.
        self,
        assets: ModelAssets,
        artifacts: dict[str, str],
        *,
        stop_reason: Literal["exactness", "density", "budget"],
        boundary: str,
        elapsed_seconds: float | None = None,
        training_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        training_progress = load_training_progress(self.run_directory.root)
        if training_progress:
            artifacts["training_progress"] = "checkpoints/progress"
        gates = self._prospective_gate_rows(
            training_result,
            stop_reason=stop_reason,
        )
        statuses = {
            "training": ("completed" if training_result is not None else "stopped_budget"),
            "exactness": next(
                (row["status"] for name, row in gates.items() if "exact" in name and row["status"] == "failed"),
                "passed" if training_result is not None else "not_run_budget",
            ),
            "density": gates["held_out_density"]["status"],
            "distillation": ("not_run_budget" if stop_reason == "budget" else "not_run_futility"),
            "behavior": gates["behavioral_similarity"]["status"],
            "generation": gates["behavioral_similarity"]["status"],
        }
        result = {
            "mode": "input_only",
            "evidence_scope": self.spec.evidence_scope,
            "operational_status": "completed",
            "scientific_verdict": "unsupported",
            "gates_passed": False,
            "experiment": self.spec.to_dict(),
            **(
                {}
                if self.prospective_subset_descriptor is None
                else {
                    "prospective_vocabulary_subset": (self.prospective_subset_descriptor),
                }
            ),
            "training": training_result,
            "training_progress": training_progress,
            "gates": gates,
            "claims": [],
            "prospective_execution": {
                "budget_exhausted": stop_reason == "budget",
                "stop_reason": stop_reason,
                "boundary": boundary,
                "elapsed_seconds": elapsed_seconds,
                "stages": prospective_stage_records(statuses),
            },
            **self._reporting_context(assets),
        }
        self.run_directory.write_json("result.json", result)
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

    def _select_tokenizer(
        self,
        assets: ModelAssets,
        model: nn.Module | None,
        training: _TrainingState,
        artifacts: dict[str, str],
    ) -> _SelectionState:
        training_spec = self.spec.training
        gates = self.spec.gates
        if not isinstance(training_spec, InputTrainingSpec) or not isinstance(
            gates,
            InputGateSpec,
        ):
            raise ValueError("input runner requires input-only training settings")
        checkpoint = training.reconstruction_checkpoint
        trainable = training.trainable
        distillation: dict[str, Any] = {}
        ablations: dict[str, Any] = {}
        input_selection: dict[str, Any] | None = None
        if "frozen_backbone_distillation" in self.spec.stages:
            model = _required_frozen_model(model)
            documents = load_corpus_documents(
                "train",
                dataset_id=self.spec.dataset.dataset_id,
                config=self.spec.dataset.config,
                revision=self.spec.dataset.revision,
                max_rows=self.spec.runtime.corpus_max_rows,
            )
            candidate_checkpoints = {
                "reconstruction_only": training.reconstruction_checkpoint,
            }
            distillation_trainable: dict[str, tuple[str, ...]] = {}
            for strategy, alignment, label in (
                ("token_aligned_distillation", "aligned", "token_aligned"),
                ("arbitrary_boundary_distillation", "arbitrary", "arbitrary_boundary"),
            ):
                if training_spec.strategy not in {"candidate_selection", strategy}:
                    continue
                strategy_checkpoint = self.run_directory.path(
                    f"checkpoints/{training.result.profile}-{alignment}.pt",
                )
                with self._stage(f"distill_{label}"):
                    distilled = distill_checkpoint(
                        DistillationRequest(
                            assets=assets,
                            checkpoint=training.reconstruction_checkpoint,
                            output=strategy_checkpoint,
                            documents=documents,
                            options=self._distillation_options(alignment),
                            frozen_model=model,
                            resume_manager=self.resume_manager,
                            epoch_boundary=self._prospective_epoch_boundary,
                        )
                    )
                candidate_checkpoints[strategy] = strategy_checkpoint
                distillation[alignment] = distilled.to_dict()
                distillation_trainable[strategy] = distilled.trainable_parameters

            selected_name = training_spec.strategy
            if selected_name == "candidate_selection":
                candidates = []
                evaluation_session = EvaluationSession()
                evaluation_session.bind_model(model)
                try:
                    for name, candidate_checkpoint in candidate_checkpoints.items():
                        with self._stage(f"select_{name}"):
                            candidate = self._selection_measure(
                                assets,
                                model,
                                candidate_checkpoint,
                                self.run_directory.path(f"ablations/{name}"),
                                name=name,
                                session=evaluation_session,
                            )
                        ablations[name] = candidate
                        candidates.append(candidate)
                finally:
                    evaluation_session.verify_model(model)
                input_selection = select_input_candidate(
                    candidates,
                    gates,
                )
                selected_name = str(input_selection["selected_candidate"])
                input_selection["selected_checkpoint"] = str(
                    candidate_checkpoints[selected_name].relative_to(
                        self.run_directory.root,
                    ),
                )
                self.run_directory.write_json(
                    "input-selection.json",
                    input_selection,
                )
                artifacts["input_selection"] = "input-selection.json"
            checkpoint = candidate_checkpoints[selected_name]
            trainable = distillation_trainable.get(selected_name, trainable)

        if distillation:
            self.run_directory.write_json("distillation.json", distillation)
            self.run_directory.write_json("ablations.json", ablations)
            artifacts["distillation"] = "distillation.json"
            artifacts["ablations"] = "ablations.json"
        return _SelectionState(
            checkpoint,
            trainable,
            distillation,
            input_selection,
            ablations,
        )

    def _measure_selected(
        self,
        assets: ModelAssets,
        model: nn.Module | None,
        selection: _SelectionState,
        artifacts: dict[str, str],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any] | None,
        tuple[dict[str, Any], ...],
        Literal["exactness", "density"] | None,
    ]:
        evaluation = self.spec.evaluation
        training = self.spec.training
        if not isinstance(evaluation, InputEvaluationSpec) or not isinstance(
            training,
            InputTrainingSpec,
        ):
            raise ValueError("input runner requires input-only settings")
        request = _MeasurementRequest(
            checkpoint=selection.checkpoint,
            output_dir=self.run_directory.root,
            alignment="arbitrary",
            resume_phase="input-selected-evaluation",
        )
        futility_reason: Literal["exactness", "density"] | None = None
        if self.prospective_policy is not None and self.prospective_policy.futility_enabled:
            with self._stage("measure_selected_input_exactness_density"):
                tokenizer = benchmark_experiment(
                    assets,
                    request.checkpoint,
                    request.output_dir,
                    BenchmarkOptions(
                        max_test_bytes=evaluation.max_test_bytes,
                        batch_size=training.batch_size,
                        retrieval_rows=evaluation.retrieval_queries,
                        repetitions=evaluation.tokenizer_repetitions,
                        dataset_id=self.spec.dataset.dataset_id,
                        dataset_config=self.spec.dataset.config,
                        dataset_revision=self.spec.dataset.revision,
                        device=self.device,
                    ),
                )
            acceptance = cast(
                Mapping[str, Any],
                tokenizer["acceptance"],
            )
            if acceptance.get("embedding_fit") is not True:
                futility_reason = "exactness"
            elif acceptance.get("density") is not True or (input_table_is_removable(assets.config) and acceptance.get("compactness") is not True):
                futility_reason = "density"
            llm = None
            if futility_reason is None:
                with self._stage("measure_selected_input_behavior"):
                    llm = evaluate_input_replacement(
                        assets,
                        request.checkpoint,
                        self._evaluation_options(
                            request.output_dir,
                            request.alignment,
                        ),
                        self._evaluation_runtime(model, request.resume_phase),
                    )
        else:
            with self._stage("measure_selected_input_tokenizer"):
                tokenizer, llm = self._measure(
                    assets,
                    model,
                    request,
                )
        artifacts.update(
            tokenizer_metrics="tokenizer-metrics.json",
            tokenizer_report="tokenizer-report.md",
            checkpoint=str(
                selection.checkpoint.relative_to(self.run_directory.root),
            ),
        )
        if llm is not None:
            artifacts.update(
                evaluation_calibration="evaluation-calibration.json",
                llm_metrics="llm-metrics.json",
                llm_report="llm-report.md",
                performance="performance-metrics.json",
                samples="samples.jsonl",
            )
        training_progress = load_training_progress(self.run_directory.root)
        if training_progress:
            artifacts["training_progress"] = "checkpoints/progress"
        return tokenizer, llm, training_progress, futility_reason

    def run(self) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915 - Campaign stages remain explicit.
        run = self.run_directory
        spec = self.spec
        training_spec = spec.training
        if not isinstance(training_spec, InputTrainingSpec):
            raise ValueError("input runner requires input-only training settings")
        if not isinstance(spec.gates, InputGateSpec):
            raise ValueError("input runner requires input-only gates")
        self._write_experiment_contract()
        artifacts: dict[str, str] = {}
        trainable: tuple[str, ...] = ()
        assets: ModelAssets | None = None
        model: nn.Module | None = None
        frozen_fingerprint: str | None = None
        try:
            with self._stage("load_model_assets"):
                assets = load_model_assets(spec.model.model_id, spec.model.revision)
                if assets.revision != spec.model.revision:
                    raise ValueError(  # noqa: TRY301 - Campaign failures must reach finalization.
                        "resolved model revision differs from the specification"
                    )
                self._prepare_prospective_subset(assets, artifacts)
                self._write_start_manifest(assets)
            with self._stage("run_preflight"):
                _, model = self._run_preflight(
                    assets,
                    load_full_model=not (self.prospective_policy is not None and self.prospective_policy.futility_enabled),
                )
                artifacts["preflight"] = "preflight.json"
            training = self._train_tokenizer(assets, artifacts)
            futility_reason = self._training_futility_reason(
                assets,
                training.result,
            )
            if futility_reason is not None:
                return self._publish_prospective_stop(
                    assets,
                    artifacts,
                    stop_reason=futility_reason,
                    boundary="stage:train_input_tokenizer",
                    training_result=asdict(training.result),
                )
            if model is None and spec.model.evaluation == "full":
                with self._stage("load_full_selection_model"):
                    model = load_frozen_causal_lm(assets, self.device)
            selection = self._select_tokenizer(
                assets,
                model,
                training,
                artifacts,
            )
            trainable = selection.trainable
            (
                tokenizer_metrics,
                llm_metrics,
                training_progress,
                measurement_futility,
            ) = self._measure_selected(
                assets,
                model,
                selection,
                artifacts,
            )
            if measurement_futility is not None:
                return self._publish_prospective_stop(
                    assets,
                    artifacts,
                    stop_reason=measurement_futility,
                    boundary=("stage:measure_selected_input_exactness_density"),
                    training_result=asdict(training.result),
                )
            if llm_metrics is not None:
                frozen_fingerprint = str(llm_metrics["model"]["parameter_fingerprint"])
            gates = {
                "exact_held_out_density": bool(
                    tokenizer_metrics["acceptance"]["density"],
                ),
                **_behavior_gates(llm_metrics, spec.gates),
            }
            result = {
                **self._result_metadata(
                    assets,
                    gates_passed=all(gates.values()),
                ),
                **self._reporting_context(assets),
                "experiment": spec.to_dict(),
                **(
                    {}
                    if self.prospective_subset_descriptor is None
                    else {
                        "prospective_vocabulary_subset": self.prospective_subset_descriptor,
                    }
                ),
                "training": asdict(training.result),
                "training_progress": training_progress,
                "distillation": selection.distillation or None,
                "input_selection": selection.input_selection,
                "ablations": selection.ablations or None,
                "tokenizer": tokenizer_metrics,
                "llm": llm_metrics,
                "gates": gates,
                "independent_findings": {
                    "embedding_alignment": tokenizer_metrics["acceptance"]["embedding_fit"],
                    "compactness": tokenizer_metrics["acceptance"]["compactness"],
                },
            }
            with self._stage("publish_result"):
                run.write_json("result.json", result)
                artifacts["result"] = "result.json"
                run.write_text("artifact-report.md", artifact_report(result))
                artifacts["report"] = "artifact-report.md"
            self._finalize_success(
                artifacts,
                trainable_parameters=trainable,
                assets=assets,
                frozen_backbone_fingerprint=frozen_fingerprint,
            )
        except ProspectiveBudgetExhaustedError as error:
            if assets is None:
                raise
            training_result: Mapping[str, Any] | None = None
            for relative in (
                "training-result.json",
                "checkpoints/training-result.json",
            ):
                path = run.root / relative
                if path.is_file():
                    training_result = load_json_object(path)
                    artifacts["training"] = relative
                    break
            if (run.root / "preflight.json").is_file():
                artifacts["preflight"] = "preflight.json"
            return self._publish_prospective_stop(
                assets,
                artifacts,
                stop_reason="budget",
                boundary=error.boundary,
                elapsed_seconds=error.elapsed_seconds,
                training_result=training_result,
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
        else:
            return result
