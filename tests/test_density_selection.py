from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from input_model_fixtures import make_adapter
from torch import nn

import continuous_tokenizer.campaigns.lifecycle as lifecycle_module
import continuous_tokenizer.input.evaluation as evaluation_module
from continuous_tokenizer.backbone.synthetic import synthetic_model_assets
from continuous_tokenizer.campaigns.lifecycle import ExperimentLifecycle
from continuous_tokenizer.contracts.input import InputGateSpec, InputTrainingSpec
from continuous_tokenizer.contracts.profiles import CAMPAIGN_PROFILE_NAME, profile_named
from continuous_tokenizer.input.studies import select_input_candidate


def _candidate(  # noqa: PLR0913 - independent gate dimensions
    name: str,
    *,
    density: float,
    mean_kl: float = 0.05,
    nll_delta: float = 0.05,
    top1: float = 0.95,
    generation_similarity: float = 0.75,
    exact_density: bool = True,
    alignment: bool = False,
    compactness: bool = False,
) -> dict[str, object]:
    return {
        "name": name,
        "tokenizer": {
            "density": {
                "round_trip": exact_density,
                "native_tokens_per_continuous_token": density,
            },
            "acceptance": {
                "density": exact_density and density >= 1.1,
                "embedding_fit": alignment,
                "compactness": compactness,
                "overall": False,
            },
        },
        "validation": {
            "teacher_forced": {
                "segmented": {
                    "teacher_nll": 1.0,
                    "student_nll": 1.0 + nll_delta,
                    "mean_kl": mean_kl,
                    "mean_js": mean_kl / 2,
                    "top1_agreement": top1,
                },
            },
            "generation": {
                "samples": 4,
                "segmented_mean_byte_similarity": generation_similarity,
            },
        },
    }


def _candidates(**overrides: dict[str, object]) -> list[dict[str, object]]:
    names = (
        "reconstruction_only",
        "token_aligned_distillation",
        "arbitrary_boundary_distillation",
    )
    candidates = []
    for index, name in enumerate(names):
        values: dict[str, Any] = {
            "density": 1.2 + index / 10,
            **overrides.get(name, {}),
        }
        candidates.append(_candidate(name, **values))
    return candidates


def test_selection_requires_exact_density_and_every_behavior_gate() -> None:
    candidates = _candidates(
        arbitrary_boundary_distillation={"generation_similarity": 0.49},
        token_aligned_distillation={"exact_density": False},
    )

    selection = select_input_candidate(candidates, InputGateSpec())

    assert selection["selected_candidate"] == "reconstruction_only"
    rows = {row["name"]: row for row in selection["ranked_candidates"]}
    assert rows["reconstruction_only"]["eligible"] is True
    assert rows["token_aligned_distillation"]["exact_held_out_density"] is False
    assert rows["arbitrary_boundary_distillation"]["behavioral_similarity_gates"]["minimum_segmented_generation_byte_similarity"] is False


def test_selection_is_independent_of_alignment_and_compactness() -> None:
    selection = select_input_candidate(_candidates(), InputGateSpec())

    assert selection["selection_feasible"] is True
    assert selection["selected_candidate"] == "arbitrary_boundary_distillation"
    selected = selection["ranked_candidates"][0]
    assert selected["embedding_alignment_passed"] is False
    assert selected["compactness_passed"] is False


def test_selection_ranks_density_before_behavioral_tiebreaks() -> None:
    candidates = _candidates(
        reconstruction_only={"density": 1.4, "mean_kl": 0.09},
        token_aligned_distillation={"density": 1.4, "mean_kl": 0.01},
        arbitrary_boundary_distillation={"density": 1.3, "mean_kl": 0.001},
    )

    selection = select_input_candidate(candidates, InputGateSpec())

    assert [row["name"] for row in selection["ranked_candidates"]] == [
        "token_aligned_distillation",
        "reconstruction_only",
        "arbitrary_boundary_distillation",
    ]


def test_infeasible_efficiency_provenance_is_authenticated_not_rejected() -> None:
    lifecycle = object.__new__(ExperimentLifecycle)
    training = InputTrainingSpec()
    lifecycle.spec = SimpleNamespace(
        efficiency_pilot="pilot.json",
        efficiency_pilot_sha256="a" * 64,
        model=SimpleNamespace(model_id="model", revision="revision"),
        training=training,
    )
    pilot = {
        "evidence_scope": "search",
        "operational_status": "completed",
        "selected_efficiency_passed": False,
        "selection_feasible": False,
        "model_id": "model",
        "model_revision": "revision",
        "selected_parameters": {
            "learning_rate": training.learning_rate,
            "batch_size": training.batch_size,
            "projection_multiplier": training.projection_multiplier,
            "muon_ns_steps": training.muon_ns_steps,
        },
    }

    with (
        patch.object(lifecycle_module, "sha256_file", return_value="a" * 64),
        patch.object(lifecycle_module, "load_artifact", return_value=pilot),
    ):
        assert lifecycle._validate_efficiency_pilot() is False


def test_infeasible_authenticated_provenance_makes_final_science_unsupported() -> None:
    lifecycle = object.__new__(ExperimentLifecycle)
    lifecycle.spec = SimpleNamespace(
        mode="input_only",
        evidence_scope="final",
        search_selections=(),
        study_selections=(),
    )
    lifecycle.profile = profile_named(CAMPAIGN_PROFILE_NAME)
    lifecycle.provenance_feasible = False
    assets = synthetic_model_assets()
    assets.model_id = "real/model"

    result = lifecycle._result_metadata(assets, gates_passed=True)

    assert result["operational_status"] == "completed"
    assert result["scientific_verdict"] == "unsupported"


def test_infeasible_search_and_study_provenance_remain_valid_inputs() -> None:
    lifecycle = object.__new__(ExperimentLifecycle)
    search = SimpleNamespace(
        search_kind="alignment",
        artifact="search.json",
        artifact_sha256="a" * 64,
        search_fingerprint="b" * 64,
        selected_trial=3,
        model_id="model",
        model_revision="revision",
        profile="large",
        selected_parameters={"learning_rate": 3e-4},
        feasible=False,
    )
    study = SimpleNamespace(
        study_kind="input_selection",
        artifact="study.json",
        artifact_sha256="c" * 64,
        study_fingerprint="d" * 64,
        model_id="model",
        model_revision="revision",
        selected_parameters={"selected_candidate": "reconstruction_only"},
        feasible=False,
    )
    lifecycle.spec = SimpleNamespace(
        prospective_selection=None,
        search_selections=(search,),
        study_selections=(study,),
    )
    artifacts = (
        {
            "operational_status": "completed",
            "search_fingerprint": search.search_fingerprint,
            "selected_trial": search.selected_trial,
            "model_id": search.model_id,
            "model_revision": search.model_revision,
            "profile": search.profile,
            "selected_parameters": search.selected_parameters,
            "selection_feasible": False,
        },
        {
            "operational_status": "completed",
            "study_fingerprint": study.study_fingerprint,
            "model_id": study.model_id,
            "model_revision": study.model_revision,
            "selection": {
                "selection_feasible": False,
                **study.selected_parameters,
            },
        },
    )

    with patch.object(
        lifecycle_module,
        "_load_selection_artifact",
        side_effect=artifacts,
    ):
        assert lifecycle._validate_selection_provenance() is False


class _TinyModel(nn.Module):
    def __init__(self, embeddings: torch.Tensor) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings,
            freeze=True,
        )
        self.head = nn.Linear(embeddings.shape[1], embeddings.shape[0], bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        logits_to_keep: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        logits = self.head(hidden)
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep]
        return SimpleNamespace(logits=logits)


def test_native_continuation_diagnostic_is_explicitly_excluded() -> None:
    adapter = make_adapter()
    embeddings = torch.cat(
        (
            adapter.codec.byte_embeddings,
            adapter.control_embeddings,
            torch.randn(1, adapter.codec.byte_embeddings.shape[1]),
        ),
    )
    model = _TinyModel(embeddings)
    samples = [evaluation_module.PromptSample((65, 66), (67,))]
    identity = evaluation_module.NativeBaselineIdentity(
        model_id="synthetic/model",
        model_revision="synthetic",
        model_fingerprint="a" * 64,
        prompt_window_sha256="b" * 64,
        sample_order_sha256="c" * 64,
        seed=17,
        dtype="torch.float32",
        device="cpu",
        generation_samples=0,
        max_new_tokens=1,
        eos_token_ids=(),
        teacher_forced_batch_size=8,
    )
    native = evaluation_module._native_batch_logits(
        model,
        list(enumerate(samples)),
    )
    baseline = evaluation_module.NativeBaselineBundle.create(
        identity,
        (native[0],),
        (),
    )

    teacher_forced, _, diagnostic = evaluation_module._teacher_forced_metrics(
        model,
        adapter,
        samples,
        "arbitrary",
        evaluation_module._TeacherForcedRuntime(
            evaluation_module._EvaluationResume(None, "diagnostic"),
            embeddings,
            baseline,
        ),
    )

    assert set(teacher_forced) == {"compatibility", "segmented"}
    assert diagnostic["label"] == "mechanism_only_native_continuation"
    assert diagnostic["acceptance_scope"] == "excluded"
    assert diagnostic["claims_scope"] == "excluded"
    assert diagnostic["performance_scope"] == "excluded"


def load_tests(
    _loader: unittest.TestLoader,
    _tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_selection_requires_exact_density_and_every_behavior_gate,
            test_selection_is_independent_of_alignment_and_compactness,
            test_selection_ranks_density_before_behavioral_tiebreaks,
            test_infeasible_efficiency_provenance_is_authenticated_not_rejected,
            test_infeasible_authenticated_provenance_makes_final_science_unsupported,
            test_infeasible_search_and_study_provenance_remain_valid_inputs,
            test_native_continuation_diagnostic_is_explicitly_excluded,
        )
    )
