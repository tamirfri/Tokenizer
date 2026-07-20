from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch

import continuous_tokenizer.output.study as study_module
from continuous_tokenizer.backbone.synthetic import (
    SyntheticCausalLM,
    synthetic_model_assets,
)
from continuous_tokenizer.contracts.output import OutputGateSpec
from continuous_tokenizer.contracts.output_study import (
    OUTPUT_ORACLE_SELECTION_RULE,
    OutputOracleStudySpec,
)
from continuous_tokenizer.output.study import _selection


class OutputOracleStudyTests(unittest.TestCase):
    def test_checked_in_oracle_studies_are_strict_model_specific_contracts(self) -> None:
        root = Path(__file__).parents[1]
        paths = sorted((root / "experiments/studies/output").glob("*.toml"))
        specs = [OutputOracleStudySpec.load(path) for path in paths]

        self.assertEqual(len(specs), 2)
        self.assertEqual({spec.selection_rule for spec in specs}, {OUTPUT_ORACLE_SELECTION_RULE})
        self.assertEqual({spec.corpus_role for spec in specs}, {"oracle_validation"})
        self.assertEqual({spec.span_limits for spec in specs}, {(1, 2, 4, 8)})
        self.assertEqual(
            {spec.load_experiment(path).model.model_id for spec, path in zip(specs, paths, strict=True)},
            {"Qwen/Qwen3.5-0.8B", "google/gemma-3-270m-it"},
        )

    def test_oracle_selects_largest_span_that_can_pass_before_training(self) -> None:
        ceilings = {
            str(span): {
                "feasible": True,
                "exact_native_sequence_rate_ceiling": 1.0 if span <= 8 else 0.8,
                "native_tokens_per_attempted_macro_step_ceiling": (1.2 if span >= 2 else 1.0),
            }
            for span in (1, 2, 4, 8)
        }
        study_path = Path(__file__).parents[1] / "experiments/studies/output/qwen35-0.8b-oracle.toml"
        experiment = OutputOracleStudySpec.load(study_path).load_experiment(study_path)

        selected = _selection(ceilings, cast(OutputGateSpec, experiment.gates))

        self.assertEqual(selected["selected_max_span"], 8)
        self.assertEqual(selected["feasible_spans"], [2, 4, 8])
        self.assertTrue(selected["selection_feasible"])

    def test_infeasible_oracle_completes_with_best_registered_span(self) -> None:
        ceilings = {
            "1": {
                "feasible": True,
                "exact_native_sequence_rate_ceiling": 1.0,
                "native_tokens_per_attempted_macro_step_ceiling": 1.0,
            },
            "2": {
                "feasible": True,
                "exact_native_sequence_rate_ceiling": 0.9,
                "native_tokens_per_attempted_macro_step_ceiling": 1.2,
            },
        }
        study_path = Path(__file__).parents[1] / "experiments/studies/output/qwen35-0.8b-oracle.toml"
        experiment = OutputOracleStudySpec.load(study_path).load_experiment(
            study_path,
        )

        selected = _selection(
            ceilings,
            cast(OutputGateSpec, experiment.gates),
        )

        self.assertFalse(selected["selection_feasible"])
        self.assertEqual(selected["selected_max_span"], 1)
        self.assertEqual(
            selected["selection_policy"],
            "best_registered_oracle_ceiling",
        )

    def test_output_study_resume_is_equivalent_and_identity_bound(self) -> None:
        study_path = Path(__file__).parents[1] / "experiments/studies/output/qwen35-0.8b-oracle.toml"
        experiment = OutputOracleStudySpec.load(study_path).load_experiment(
            study_path,
        )
        assets = replace(
            synthetic_model_assets(),
            model_id=experiment.model.model_id,
            revision=experiment.model.revision,
        )
        ceilings = {
            str(span): {
                "feasible": True,
                "exact_native_sequence_rate_ceiling": 1.0,
                "native_tokens_per_attempted_macro_step_ceiling": 1.2,
            }
            for span in (1, 2, 4, 8)
        }
        package = {
            "name": "continuous-byte-tokenizer",
            "version": "0.1.0",
            "content_sha256": "c" * 64,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                study_module,
                "source_state",
                return_value=("commit", False, "a" * 64),
            ),
            patch.object(
                study_module,
                "sha256_file",
                return_value="b" * 64,
            ),
            patch.object(
                study_module,
                "installed_distribution_identity",
                return_value=package,
            ),
            patch.object(
                study_module,
                "load_model_assets",
                return_value=assets,
            ),
            patch.object(
                study_module,
                "declared_device",
                return_value=torch.device("cpu"),
            ),
            patch.object(
                study_module,
                "load_frozen_causal_lm",
                return_value=SyntheticCausalLM(
                    assets.input_embeddings,
                ),
            ),
            patch.object(
                study_module,
                "output_stop_control_ids",
                return_value=frozenset(),
            ),
            patch.object(
                study_module,
                "_prompt_sequences",
                return_value=(((0,),), {"sha256": "d" * 64}),
            ),
            patch.object(
                study_module,
                "build_prepared_output_corpus",
                return_value=object(),
            ),
            patch.object(
                study_module,
                "native_head_oracle_ceilings",
                return_value=ceilings,
            ),
            patch.object(
                study_module,
                "dependency_environment",
                return_value={"device": "cpu"},
            ),
        ):
            output = Path(directory) / "study"
            initial = study_module.run_output_oracle_study(
                study_path,
                output,
            )
            resumed = study_module.run_output_oracle_study(
                study_path,
                output,
                resume=True,
            )
            self.assertEqual(initial, resumed)
            with (
                patch.object(
                    study_module,
                    "installed_distribution_identity",
                    return_value={**package, "content_sha256": "e" * 64},
                ),
                self.assertRaisesRegex(ValueError, "sealed identity"),
            ):
                study_module.run_output_oracle_study(
                    study_path,
                    output,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
