from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import cast

from continuous_tokenizer.contracts.claims import (
    CURRENT_DESIGN_NOTICE,
    DEFAULT_PERFORMANCE_LIMITATION,
    INPUT_HEADLINE,
    directional_claims,
)
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.contracts.input import InputEvaluationSpec, InputTrainingSpec
from continuous_tokenizer.contracts.output import OutputEvaluationSpec, OutputTrainingSpec
from continuous_tokenizer.contracts.profiles import profile_named
from continuous_tokenizer.reporting.shared import current_design_lines

ROOT = Path(__file__).parents[1]
DOCUMENTS = (ROOT / "README.md", ROOT / "AGENTS.md")


class CurrentDesignEvidenceTests(unittest.TestCase):
    def test_documents_describe_only_current_profile_and_span_limits(self) -> None:
        text = "\n".join(path.read_text(encoding="utf-8") for path in DOCUMENTS)
        for stale in (
            "320-wide",
            "1,024-wide",
            "64 bytes plus one `CLS`",
            "maximum output span `16`",
            "prospective/v1",
            "historical compatibility",
            "backward-compatible",
        ):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, text)
        for current in (
            "local width `128`",
            "local width `256`",
            "at most 32 bytes",
            "maximum output span `8`",
            "experiments/prospective/input/qwen35-0.8b/feasibility-screen.toml",
        ):
            with self.subTest(current=current):
                self.assertIn(current, text)

    def test_profiles_and_final_budgets_match_documented_design(self) -> None:
        small = profile_named("small")
        large = profile_named("large")
        self.assertEqual(
            (
                small.local_dim,
                small.encoder_layers,
                small.decoder_layers,
                small.query_heads,
                small.key_value_heads,
                small.feedforward_dim,
                small.projection_multiplier,
            ),
            (128, 1, 1, 4, 2, 256, 4),
        )
        self.assertEqual(
            (
                large.local_dim,
                large.encoder_layers,
                large.decoder_layers,
                large.query_heads,
                large.key_value_heads,
                large.feedforward_dim,
                large.projection_multiplier,
            ),
            (256, 2, 1, 4, 2, 512, 8),
        )
        for path in sorted((ROOT / "experiments/campaigns/input").glob("*/*.toml")):
            spec = ExperimentSpec.load(path)
            self.assertIsInstance(spec.training, InputTrainingSpec)
            self.assertIsInstance(spec.evaluation, InputEvaluationSpec)
            training = cast(InputTrainingSpec, spec.training)
            evaluation = cast(InputEvaluationSpec, spec.evaluation)
            self.assertEqual(spec.stages, ("vocabulary", "reconstruction"))
            self.assertEqual(
                (training.vocabulary_epochs, training.patience, training.evaluation_interval, training.reconstruction_epochs, training.reconstruction_samples),
                (10, 2, 2, 1, 2_048),
            )
            self.assertEqual(
                (evaluation.batch_size, evaluation.samples, evaluation.generation_samples, evaluation.retrieval_queries, evaluation.max_test_bytes),
                (8, 16, 2, 128, 2_048),
            )
            self.assertEqual((evaluation.warmups, evaluation.repetitions, evaluation.tokenizer_repetitions), (1, 2, 2))
        for path in sorted((ROOT / "experiments/campaigns/output").glob("*/*.toml")):
            spec = ExperimentSpec.load(path)
            self.assertIsInstance(spec.training, OutputTrainingSpec)
            self.assertIsInstance(spec.evaluation, OutputEvaluationSpec)
            training = cast(OutputTrainingSpec, spec.training)
            evaluation = cast(OutputEvaluationSpec, spec.evaluation)
            self.assertEqual((training.max_span, training.epochs), (8, 2))
            self.assertEqual((evaluation.batch_size, evaluation.samples, evaluation.warmups, evaluation.repetitions), (8, 16, 1, 2))

    def test_headline_and_limitations_are_single_and_visible(self) -> None:
        readme = DOCUMENTS[0].read_text(encoding="utf-8")
        self.assertEqual(readme.count(INPUT_HEADLINE), 1)
        self.assertIn("Old artifacts are\nunsupported", readme)
        self.assertIn(CURRENT_DESIGN_NOTICE, current_design_lines())
        self.assertIn("5` warmups and `20` repetitions", readme)
        self.assertIn("no new real-model results", readme.lower())
        self.assertIn("lower research power", readme.lower())
        self.assertIn("not comparable", readme.lower())
        self.assertIn("1 warmup and 2 repetitions", DEFAULT_PERFORMANCE_LIMITATION)

    def test_claim_denominators_match_reduced_work(self) -> None:
        claims = {
            claim.claim_id: claim.denominator_context
            for claim in (
                *directional_claims("input_only"),
                *directional_claims("output_only"),
            )
        }
        self.assertIn("128-, 256-, and 512-row", claims["input.fixed_subset_alignment_feasibility"])
        self.assertIn("2,048 held-out WikiText bytes", claims["input.held_out_position_compression"])
        self.assertIn("16 teacher-forced samples and 2 generation samples", claims["input.registered_behavioral_similarity_tolerances"])
        self.assertIn("at least 5 warmups and 20 raw repetitions", claims["input.tokenizer_latency_improvement"])
        self.assertIn("16 evaluation samples", claims["output.direct_feedback_exactness"])
        self.assertIn("spans capped at 8 bytes", claims["output.semi_autoregressive_density"])

    def test_current_artifact_kinds_have_no_suffix(self) -> None:
        paths = (
            ROOT / "src/continuous_tokenizer/contracts/prospective.py",
            ROOT / "src/continuous_tokenizer/artifacts/evidence.py",
            ROOT / "src/continuous_tokenizer/reporting/discovery.py",
        )
        for path in paths:
            contents = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIsNone(re.search(r'artifact_kind.{0,80}"[^"]+_v1"', contents))


if __name__ == "__main__":
    unittest.main()
