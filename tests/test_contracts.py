from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from continuous_tokenizer.backbone.synthetic import SyntheticCausalLM, synthetic_model_assets
from continuous_tokenizer.campaigns.lifecycle import ExperimentLifecycle
from continuous_tokenizer.contracts.claim_derivation import (
    MINIMUM_LATENCY_REPETITIONS,
    MINIMUM_LATENCY_WARMUPS,
)
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import (
    InputEvaluationSpec,
    InputGateSpec,
    InputTrainingSpec,
)
from continuous_tokenizer.contracts.output import OutputGateSpec, OutputTrainingSpec
from continuous_tokenizer.contracts.output_study import OutputOracleStudySpec
from continuous_tokenizer.contracts.profiles import (
    CAMPAIGN_PROFILE_NAME,
    DIAGNOSTIC_PROFILE_NAME,
    PROFILES,
    PROJECTION_DIMENSION_CAP,
    profile_named,
)
from continuous_tokenizer.contracts.search import (
    EfficiencyPilotSpec,
    OutputSearchSpec,
    SearchSpec,
)


def valid_spec() -> str:
    return """
name = "synthetic"
mode = "input_only"
evidence_scope = "diagnostic"
device = "cpu"
seed = 17
stages = ["vocabulary", "reconstruction"]

[model]
id = "synthetic/model"
revision = "revision"
evaluation = "full"

[dataset]
id = "synthetic/data"
config = "default"
revision = "data-revision"

[runtime]
corpus_max_rows = 16
cache_chunk_rows = 8
snapshot_interval = 1
projected_run_bytes = 1024
storage_reserve_bytes = 1024
inductor_cache_estimate_bytes = 1024
minimum_mps_memory_bytes = 1024

[training]
profile = "small"
batch_size = 16
learning_rate = 0.001
vocabulary_epochs = 1
reconstruction_epochs = 1
reconstruction_samples = 1
reconstruction_vocabulary_fraction = 0.75
validation_bytes = 16
patience = 1
evaluation_interval = 1
distillation_epochs = 0
distillation_windows = 1
distillation_prompt_tokens = 2
distillation_continuation_tokens = 1

[evaluation]
batch_size = 8
samples = 2
prompt_tokens = 2
continuation_tokens = 1
generation_samples = 0
max_new_tokens = 1
warmups = 0
repetitions = 1
performance_prompts = 1
retrieval_queries = 16
max_test_bytes = 16

[gates]
maximum_normalized_rmse = 0.01
minimum_cosine_p01 = 0.999
minimum_cosine_p50 = 0.9999
minimum_native_tokens_per_continuous_token = 1.1
maximum_candidate_reference_state_ratio = 0.5
maximum_segmented_mean_kl = 0.1
maximum_segmented_nll_delta = 0.1
minimum_segmented_top1_agreement = 0.9
minimum_segmented_generation_byte_similarity = 0.5
"""


def write_spec(tmp_path: Path, value: str | None = None) -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(valid_spec() if value is None else value, encoding="utf-8")
    return path


def test_experiment_spec_is_strict_and_deterministic() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = write_spec(Path(directory))

        first = ExperimentSpec.load(path)
        second = ExperimentSpec.load(path)

        assert first == second
        assert first.fingerprint() == second.fingerprint()
        assert first.replication_fingerprint() == second.replication_fingerprint()
        replicated = replace(first, name="synthetic-seed-23", seed=23)
        assert first.fingerprint() != replicated.fingerprint()
        assert first.replication_fingerprint() == replicated.replication_fingerprint()
        assert first.device == "cpu"
        assert first.dataset.revision == "data-revision"
        assert first.stages == ("vocabulary", "reconstruction")
        assert isinstance(first.training, InputTrainingSpec)
        assert first.training.reconstruction_vocabulary_fraction == 0.75


def test_experiment_spec_rejects_invalid_contracts() -> None:
    cases = (
        ('name = "synthetic"', "mode must be"),
        ('name = "synthetic"\nunknown = true', "unknown experiment fields"),
        (
            'stages = ["reconstruction", "vocabulary"]',
            "stages must follow this order",
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for replacement, message in cases:
            original = valid_spec()
            if replacement.startswith("stages"):
                invalid = original.replace(
                    'stages = ["vocabulary", "reconstruction"]',
                    replacement,
                )
            elif "unknown" in replacement:
                invalid = original.replace('name = "synthetic"', replacement)
            else:
                invalid = original.replace(
                    'name = "synthetic"\nmode = "input_only"',
                    replacement,
                )

            with unittest.TestCase().assertRaisesRegex(ValueError, message):
                ExperimentSpec.load(write_spec(root, invalid))

        invalid_fraction = valid_spec().replace(
            "reconstruction_vocabulary_fraction = 0.75",
            "reconstruction_vocabulary_fraction = 1.0",
        )
        with unittest.TestCase().assertRaisesRegex(ValueError, "between zero and one"):
            ExperimentSpec.load(write_spec(root, invalid_fraction))


def test_checked_in_experiment_specs_parse() -> None:
    root = Path(__file__).parents[1]

    qwen = ExperimentSpec.load(root / "experiments/campaigns/input/qwen35-0.8b/seed-17.toml")
    gemma = ExperimentSpec.load(root / "experiments/campaigns/input/gemma3-270m/seed-17.toml")
    pilot = ExperimentSpec.load(root / "experiments/pilots/qwen35-0.8b-input.toml")
    alignment_search_base = ExperimentSpec.load(root / "experiments/searches/qwen35-0.8b-alignment-base.toml")
    output_search_base = ExperimentSpec.load(root / "experiments/searches/qwen35-0.8b-output-base.toml")
    gemma_alignment_search = SearchSpec.load(root / "experiments/searches/gemma3-270m-alignment.toml")
    gemma_output_search = OutputSearchSpec.load(root / "experiments/searches/gemma3-270m-output.toml")
    gemma_alignment_base = ExperimentSpec.load(root / "experiments/searches/gemma3-270m-alignment-base.toml")
    gemma_output_base = ExperimentSpec.load(root / "experiments/searches/gemma3-270m-output-base.toml")
    synthetic = ExperimentSpec.load(root / "experiments/synthetic/input-smoke.toml")
    synthetic_output = ExperimentSpec.load(root / "experiments/synthetic/output-smoke.toml")

    assert qwen.model.evaluation == "full"
    assert isinstance(qwen.evaluation, InputEvaluationSpec)
    assert qwen.evaluation.performance_prompts == 2
    assert qwen.evidence_scope == "candidate"
    assert qwen.stages == ("vocabulary", "reconstruction")
    assert gemma.model.evaluation == "full"
    assert {qwen.training.profile, gemma.training.profile} == {"large"}
    assert pilot.training.profile == "small"
    assert alignment_search_base.training.profile == "large"
    assert alignment_search_base.stages == ("vocabulary",)
    assert output_search_base.training.profile == "large"
    assert output_search_base.mode == "output_only"
    assert gemma_alignment_search.final_experiment.endswith("campaigns/input/gemma3-270m/seed-17.toml")
    assert gemma_output_search.final_experiment.endswith("campaigns/output/gemma3-270m/seed-17.toml")
    assert gemma_alignment_base.model.model_id == gemma_output_base.model.model_id == "google/gemma-3-270m-it"
    assert synthetic.dataset.dataset_id == "continuous-tokenizer/synthetic-bytes"
    assert synthetic.model.model_id == "continuous-tokenizer/synthetic-model"
    assert synthetic.model.revision == "synthetic"
    assert isinstance(synthetic.evaluation, InputEvaluationSpec)
    assert synthetic.evaluation.performance_prompts == 1
    assert synthetic_output.mode == "output_only"
    assert isinstance(synthetic_output.training, OutputTrainingSpec)
    assert synthetic_output.training.max_span == 2
    assert synthetic_output.stages == ("output_codec",)
    assert isinstance(output_search_base.training, OutputTrainingSpec)
    assert isinstance(gemma_output_base.training, OutputTrainingSpec)
    assert output_search_base.training.max_span == gemma_output_base.training.max_span == 8


def test_current_final_campaign_work_is_bounded_and_not_latency_claimable() -> None:
    root = Path(__file__).parents[1]
    qwen = ExperimentSpec.load(
        root / "experiments/campaigns/input/qwen35-0.8b/seed-17.toml",
    )
    assert isinstance(qwen.training, InputTrainingSpec)
    assert isinstance(qwen.evaluation, InputEvaluationSpec)
    assert (
        qwen.training.vocabulary_epochs,
        qwen.training.reconstruction_epochs,
        qwen.training.reconstruction_samples,
        qwen.training.distillation_epochs,
    ) == (10, 1, 2_048, 0)
    assert (
        qwen.runtime.corpus_max_rows,
        qwen.runtime.cache_chunk_rows,
        qwen.runtime.snapshot_interval,
    ) == (512, 32, 100)
    assert (
        qwen.evaluation.max_test_bytes,
        qwen.evaluation.retrieval_queries,
        qwen.evaluation.samples,
        qwen.evaluation.generation_samples,
    ) == (2_048, 128, 16, 2)
    assert (
        qwen.evaluation.warmups,
        qwen.evaluation.repetitions,
        qwen.evaluation.tokenizer_repetitions,
    ) == (1, 2, 2)
    assert MINIMUM_LATENCY_WARMUPS == 5
    assert MINIMUM_LATENCY_REPETITIONS == 20
    assert qwen.gates == InputGateSpec()
    output = ExperimentSpec.load(
        root / "experiments/campaigns/output/qwen35-0.8b/seed-17.toml",
    )
    assert output.gates == OutputGateSpec()


def test_current_search_contract_rejects_expanded_work() -> None:
    root = Path(__file__).parents[1]
    source_path = root / "experiments/searches/qwen35-0.8b-alignment.toml"
    source = source_path.read_text()
    efficiency_path = root / "experiments/searches/qwen35-0.8b-efficiency.toml"
    output_path = root / "experiments/searches/qwen35-0.8b-output.toml"
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "search.toml"
        for expanded in (
            source.replace("trials = 2", "trials = 3"),
            source.replace("vocabulary_rows = 512", "vocabulary_rows = 513"),
            source.replace("vocabulary_epochs = 2", "vocabulary_epochs = 4"),
        ):
            temporary.write_text(expanded)
            with unittest.TestCase().assertRaisesRegex(ValueError, "budget"):
                SearchSpec.load(temporary)
        temporary.write_text(
            efficiency_path.read_text().replace("trials = 2", "trials = 3"),
        )
        with unittest.TestCase().assertRaisesRegex(ValueError, "budget"):
            EfficiencyPilotSpec.load(temporary)
        temporary.write_text(
            output_path.read_text().replace("pilot_documents = 4", "pilot_documents = 5"),
        )
        with unittest.TestCase().assertRaisesRegex(ValueError, "budget"):
            OutputSearchSpec.load(temporary)


def test_every_campaign_candidate_is_strict_and_primary_models_have_seed_triplets() -> None:
    repository = Path(__file__).parents[1]
    campaign_root = repository / "experiments/campaigns"
    paths = sorted(campaign_root.glob("**/*.toml"))
    expected = {
        f"{direction}/{model}/seed-{seed}.toml" for direction in ("input", "output") for model in ("qwen35-0.8b", "gemma3-270m") for seed in (17, 23, 41)
    }
    assert {str(path.relative_to(campaign_root)) for path in paths} == expected

    specifications = [ExperimentSpec.load(path) for path in paths]
    assert all(spec.evidence_scope in {"candidate", "final"} for spec in specifications)
    assert all(spec.training.profile == "large" for spec in specifications)
    assert all(spec.model.evaluation == "full" for spec in specifications)
    assert all(spec.training.max_span == 8 for spec in specifications if isinstance(spec.training, OutputTrainingSpec))
    assert all(spec.device == "mps" for spec in specifications)
    assert all(spec.dataset.dataset_id == "Salesforce/wikitext" for spec in specifications)
    assert all(spec.dataset.revision == "b08601e04326c79dfdd32d625aee71d232d685c3" for spec in specifications)
    input_runs = [spec for spec in specifications if isinstance(spec.training, InputTrainingSpec)]
    input_training = [spec.training for spec in input_runs if isinstance(spec.training, InputTrainingSpec)]
    assert all(spec.stages == ("vocabulary", "reconstruction") for spec in input_runs)
    assert all(
        (
            training.vocabulary_epochs,
            training.reconstruction_epochs,
            training.reconstruction_samples,
            training.patience,
            training.evaluation_interval,
            training.distillation_epochs,
        )
        == (10, 1, 2_048, 2, 2, 0)
        for training in input_training
    )
    output_runs = [spec for spec in specifications if isinstance(spec.training, OutputTrainingSpec)]
    output_training = [spec.training for spec in output_runs if isinstance(spec.training, OutputTrainingSpec)]
    assert all(training.epochs == 2 for training in output_training)
    assert all(
        (
            spec.runtime.corpus_max_rows,
            spec.runtime.cache_chunk_rows,
            spec.runtime.snapshot_interval,
        )
        == (512, 32, 100)
        for spec in specifications
    )

    revisions = {
        "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
        "google/gemma-3-270m-it": "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
    }
    for direction in ("input_only", "output_only"):
        for model_id, revision in revisions.items():
            runs = [spec for spec in specifications if spec.mode == direction and spec.model.model_id == model_id]
            assert [spec.seed for spec in runs] == [17, 23, 41]
            assert {spec.model.revision for spec in runs} == {revision}
            assert len({spec.replication_fingerprint() for spec in runs}) == 1


def test_final_spec_requires_strict_search_provenance() -> None:
    repository = Path(__file__).parents[1]
    source = (repository / "experiments/campaigns/input/qwen35-0.8b/seed-17.toml").read_text()
    final = source.replace('evidence_scope = "candidate"', 'evidence_scope = "final"')
    selection = """

[[search_selections]]
search_kind = "alignment"
artifact = "search.json"
artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
selected_trial = 3
search_fingerprint = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
model_id = "Qwen/Qwen3.5-0.8B"
model_revision = "2fc06364715b967f1860aea9cf38778875588b17"
profile = "large"
feasible = false

[search_selections.selected_parameters]
learning_rate = 0.000014496048586998116
weight_decay = 0.0
batch_size = 64
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "final.toml"
        path.write_text(final + selection)
        parsed = ExperimentSpec.load(path)
        assert not parsed.search_selections[0].feasible

        path.write_text(final)
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "search-selection provenance",
        ):
            ExperimentSpec.load(path)


def test_result_metadata_distinguishes_all_evidence_scopes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        spec = ExperimentSpec.load(write_spec(Path(directory)))
    lifecycle = object.__new__(ExperimentLifecycle)
    lifecycle.spec = spec
    lifecycle.profile = profile_named(DIAGNOSTIC_PROFILE_NAME)
    lifecycle.provenance_feasible = True
    assets = synthetic_model_assets()

    assert lifecycle._result_metadata(assets, gates_passed=True) == {
        "mode": "input_only",
        "evidence_scope": "synthetic",
        "operational_status": "completed",
        "scientific_verdict": "supported",
    }
    assets.model_id = "real/model"
    assert lifecycle._result_metadata(assets, gates_passed=True)["scientific_verdict"] == ("not_applicable_diagnostic")
    lifecycle.profile = profile_named(CAMPAIGN_PROFILE_NAME)
    lifecycle.spec = replace(spec, evidence_scope="final")
    final = lifecycle._result_metadata(assets, gates_passed=False)
    assert final["evidence_scope"] == "final"
    assert final["scientific_verdict"] == "unsupported"
    search = lifecycle._result_metadata(assets, gates_passed=True, search=True)
    assert search["evidence_scope"] == "search"
    assert search["scientific_verdict"] == "not_applicable_search"


def test_synthetic_causal_model_exposes_native_and_embedding_logits() -> None:
    embeddings = torch.randn((256, 8))
    model = SyntheticCausalLM(embeddings)

    selected = torch.tensor([1])
    native = model(input_ids=torch.tensor([[1, 2]]), logits_to_keep=selected, use_cache=False)
    segmented = model(
        inputs_embeds=embeddings[torch.tensor([[1, 2]])],
        logits_to_keep=selected,
        use_cache=False,
    )

    assert native.logits.shape == (1, 1, 256)
    torch.testing.assert_close(native.logits, segmented.logits)


def test_output_spec_rejects_input_only_fields() -> None:
    repository = Path(__file__).parents[1]
    source = (repository / "experiments/synthetic/output-smoke.toml").read_text()
    invalid = source.replace("epochs = 10", "epochs = 10\nvocabulary_epochs = 1")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "invalid-output.toml"
        path.write_text(invalid)
        with unittest.TestCase().assertRaisesRegex(ValueError, "unknown training fields"):
            ExperimentSpec.load(path)

        path.write_text(source.replace("max_span = 2", "max_span = 16"))
        with unittest.TestCase().assertRaisesRegex(ValueError, "must be 1, 2, 4, or 8"):
            ExperimentSpec.load(path)


def test_checked_in_output_oracle_studies_are_symmetric_strict_contracts() -> None:
    repository = Path(__file__).parents[1]
    paths = sorted(
        (repository / "experiments/studies/output").glob("*-oracle.toml"),
    )
    specs = [OutputOracleStudySpec.load(path) for path in paths]

    assert len(specs) == 2
    assert {spec.span_limits for spec in specs} == {(1, 2, 4, 8)}
    assert len({spec.fingerprint() for spec in specs}) == 2
    assert {spec.load_experiment(path).model.model_id for spec, path in zip(specs, paths, strict=True)} == {"Qwen/Qwen3.5-0.8B", "google/gemma-3-270m-it"}

    source = (
        paths[0]
        .read_text()
        .replace(
            "span_limits = [1, 2, 4, 8]",
            "span_limits = [1, 2, 4]",
        )
    )
    with tempfile.TemporaryDirectory(dir=paths[0].parent) as directory:
        invalid = Path(directory) / "invalid.toml"
        invalid.write_text(source)
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "exact span limits",
        ):
            OutputOracleStudySpec.load(invalid)


def test_profiles_scale_only_supported_dimensions() -> None:
    assert [profile.name for profile in PROFILES] == ["small", "large"]
    assert [profile.local_dim for profile in PROFILES] == [128, 256]
    assert [profile.encoder_layers for profile in PROFILES] == [1, 2]
    assert [profile.decoder_layers for profile in PROFILES] == [1, 1]
    assert [profile.projection_multiplier for profile in PROFILES] == [4, 8]
    assert [profile.query_heads for profile in PROFILES] == [4, 4]
    assert [profile.key_value_heads for profile in PROFILES] == [2, 2]
    assert all(profile.feedforward_dim == 2 * profile.local_dim for profile in PROFILES)
    assert PROFILES[1].projection_dim(640) == 5_120
    assert PROFILES[1].projection_dim(1_024) == 8_192
    assert PROFILES[1].projection_dim(8_192) == PROJECTION_DIMENSION_CAP
    with unittest.TestCase().assertRaisesRegex(ValueError, "unknown profile"):
        profile_named("medium")


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_experiment_spec_is_strict_and_deterministic,
            test_experiment_spec_rejects_invalid_contracts,
            test_checked_in_experiment_specs_parse,
            test_every_campaign_candidate_is_strict_and_primary_models_have_seed_triplets,
            test_final_spec_requires_strict_search_provenance,
            test_result_metadata_distinguishes_all_evidence_scopes,
            test_synthetic_causal_model_exposes_native_and_embedding_logits,
            test_output_spec_rejects_input_only_fields,
            test_checked_in_output_oracle_studies_are_symmetric_strict_contracts,
            test_profiles_scale_only_supported_dimensions,
        )
    )
