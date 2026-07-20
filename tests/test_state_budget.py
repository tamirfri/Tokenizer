from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from continuous_tokenizer.campaigns.state_budget import (
    _paired_identity,
    run_state_budget,
)
from continuous_tokenizer.contracts.state_budget import (
    CONTROL_DEDUPLICATION_POLICY,
    REFERENCE_DEDUPLICATION_POLICY,
    STATE_BUDGET_CONCLUSION,
    STATE_BUDGET_SCOPE,
    STATE_BUDGET_VERSION,
    StateBudgetArithmetic,
    StateBudgetConfig,
    StateBudgetIdentity,
    StateBudgetNonClaims,
    StateBudgetResult,
    StateBudgetSeedResult,
    StateBudgetTensor,
    inventory_sha256,
)
from continuous_tokenizer.reporting.state_budget_markdown import (
    state_budget_report,
)


def _tensor(
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    byte_count: int,
    digest: str,
) -> StateBudgetTensor:
    return StateBudgetTensor(name, shape, dtype, byte_count, digest * 64)


def _seed_result(
    model_id: str,
    revision: str,
    seed: int,
    *,
    tied: bool,
) -> StateBudgetSeedResult:
    input_inventory = (
        _tensor("codec.byte_embeddings", (2, 2), "torch.float32", 16, "1"),
        _tensor("codec.weight", (1,), "torch.float32", 4, "2"),
        _tensor("controls.ids", (1,), "torch.int64", 8, "3"),
        _tensor("controls.embeddings", (1, 2), "torch.float32", 8, "4"),
    )
    output_inventory = (
        _tensor("codec.weight", (2,), "torch.float32", 8, "5"),
        _tensor("controls.ids", (1,), "torch.int64", 8, "3"),
    )
    input_reference = _tensor(
        "native.tied_vocabulary" if tied else "native.input_embedding",
        (20,),
        "torch.float32",
        80,
        "6",
    )
    reference_inventory = (
        (input_reference,)
        if tied
        else (
            input_reference,
            _tensor(
                "native.output_head",
                (20,),
                "torch.float32",
                80,
                "7",
            ),
        )
    )
    reference_bytes = 80 if tied else 160
    candidate_bytes = 44
    return StateBudgetSeedResult(
        model_id=model_id,
        model_revision=revision,
        seed=seed,
        tie_word_embeddings=tied,
        identity=StateBudgetIdentity(
            source_commit="commit",
            source_dirty=False,
            source_state_sha256="a" * 64,
            dependency_lock_sha256="b" * 64,
            installed_package_sha256="c" * 64,
            claim_vocabulary_sha256="d" * 64,
            model_config_sha256="e" * 64,
            input_embedding_sha256="f" * 64,
            tokenizer_vocabulary_sha256="0" * 64,
            input_contract_sha256="1" * 64,
            output_contract_sha256="2" * 64,
        ),
        input_checkpoint_sha256="3" * 64,
        output_checkpoint_sha256="4" * 64,
        input_inventory=input_inventory,
        output_inventory=output_inventory,
        reference_inventory=reference_inventory,
        input_inventory_sha256=inventory_sha256(input_inventory),
        output_inventory_sha256=inventory_sha256(output_inventory),
        reference_inventory_sha256=inventory_sha256(reference_inventory),
        reference_deduplication_policy=REFERENCE_DEDUPLICATION_POLICY,
        control_deduplication_policy=CONTROL_DEDUPLICATION_POLICY,
        arithmetic=StateBudgetArithmetic(
            input_codec_bytes=20,
            output_codec_bytes=8,
            atomic_byte_rows_bytes=16,
            shared_control_id_bytes=8,
            shared_control_row_bytes=8,
            candidate_tensor_state_bytes=candidate_bytes,
            reference_input_table_bytes=80,
            reference_output_head_bytes=80,
            reference_tensor_state_bytes=reference_bytes,
        ),
        ratio=candidate_bytes / reference_bytes,
    )


def _result() -> StateBudgetResult:
    rows = tuple(
        _seed_result(
            model_id,
            revision,
            seed,
            tied=model_id.startswith("Qwen/"),
        )
        for model_id, revision in (
            ("Qwen/Qwen3.5-0.8B", "qwen-revision"),
            ("google/gemma-3-270m-it", "gemma-revision"),
        )
        for seed in (17, 23, 41)
    )
    worst = max(row.ratio for row in rows)
    return StateBudgetResult(
        version=STATE_BUDGET_VERSION,
        evidence_scope=STATE_BUDGET_SCOPE,
        operational_status="completed",
        config=StateBudgetConfig(),
        conclusion=STATE_BUDGET_CONCLUSION,
        verdict="supported",
        non_claims=StateBudgetNonClaims(),
        per_seed=rows,
        worst_case_ratio=worst,
    )


class StateBudgetTests(unittest.TestCase):
    def test_tied_reference_and_shared_controls_are_counted_once(self) -> None:
        parsed = StateBudgetResult.from_mapping(_result().to_dict())
        tied = parsed.per_seed[0]

        self.assertEqual(len(tied.reference_inventory), 1)
        self.assertEqual(
            tied.arithmetic.candidate_tensor_state_bytes,
            20 + 8 + 8 + 8,
        )
        self.assertEqual(tied.arithmetic.reference_tensor_state_bytes, 80)
        self.assertEqual(
            set(parsed.non_claims.to_dict()),
            {
                "combined_runtime_tested",
                "continuous_feedback_tested",
                "physical_omission_tested",
                "resident_memory_reduction_tested",
                "peak_memory_reduction_tested",
            },
        )
        report = state_budget_report(parsed)
        self.assertIn("Future prerequisite verdict", report)
        self.assertIn("not evidence of a combined tokenizer", report)
        self.assertEqual(report.count(": `false`"), 5)

    def test_untied_references_are_counted_separately(self) -> None:
        parsed = StateBudgetResult.from_mapping(_result().to_dict())
        untied = parsed.per_seed[3]

        self.assertEqual(len(untied.reference_inventory), 2)
        self.assertEqual(untied.arithmetic.reference_tensor_state_bytes, 160)

    def test_strict_fields_flags_duplicate_rows_and_tampering(self) -> None:
        canonical = _result().to_dict()
        extra = json.loads(json.dumps(canonical))
        extra["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "not canonical"):
            StateBudgetResult.from_mapping(extra)

        flags = json.loads(json.dumps(canonical))
        flags["non_claims"]["combined_runtime_tested"] = True
        with self.assertRaisesRegex(ValueError, "must all be false"):
            StateBudgetResult.from_mapping(flags)

        duplicate = json.loads(json.dumps(canonical))
        duplicate["per_seed"][0]["input_inventory"].append(
            duplicate["per_seed"][0]["input_inventory"][0],
        )
        with self.assertRaisesRegex(ValueError, "unique rows"):
            StateBudgetResult.from_mapping(duplicate)

        arithmetic = json.loads(json.dumps(canonical))
        arithmetic["per_seed"][0]["arithmetic"]["candidate_tensor_state_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            StateBudgetResult.from_mapping(arithmetic)

        inventory_hash = json.loads(json.dumps(canonical))
        inventory_hash["per_seed"][0]["input_inventory_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "inventory hash"):
            StateBudgetResult.from_mapping(inventory_hash)

    def test_ratio_failure_supports_only_registered_unsupported_conclusion(
        self,
    ) -> None:
        canonical = _result()
        rows = list(canonical.per_seed)
        failed = replace(
            rows[0],
            arithmetic=replace(
                rows[0].arithmetic,
                candidate_tensor_state_bytes=88,
                input_codec_bytes=64,
            ),
            ratio=1.1,
        )
        failed_input = list(failed.input_inventory)
        failed_input[1] = _tensor(
            "codec.weight",
            (12,),
            "torch.float32",
            48,
            "2",
        )
        failed = replace(
            failed,
            input_inventory=tuple(failed_input),
            input_inventory_sha256=inventory_sha256(tuple(failed_input)),
        )
        rows[0] = failed
        unsupported = replace(
            canonical,
            per_seed=tuple(rows),
            verdict="unsupported",
            worst_case_ratio=1.1,
        )

        parsed = StateBudgetResult.from_mapping(unsupported.to_dict())

        self.assertEqual(parsed.conclusion, STATE_BUDGET_CONCLUSION)
        self.assertEqual(parsed.verdict, "unsupported")

    def test_pairing_rejects_identity_mismatch(self) -> None:
        shared = {
            "model_id": "model",
            "model_revision": "revision",
            "dataset_id": "dataset",
            "dataset_revision": "revision",
            "embedding_tensor": "embed.weight",
            "source_dtype": "torch.float32",
            "seed": 17,
            "source_commit": "commit",
            "source_dirty": False,
            "source_state_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "installed_package": {"content_sha256": "c" * 64},
            "claim_vocabulary_sha256": "d" * 64,
            "source_assets": {"model_config": {"sha256": "e" * 64}},
        }
        input_manifest = SimpleNamespace(**shared)
        output_manifest = SimpleNamespace(
            **(shared | {"dependency_lock_sha256": "f" * 64}),
        )

        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            _paired_identity(input_manifest, output_manifest)

    def test_output_directory_overwrite_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "budget"
            output.mkdir()
            with (
                patch(
                    "continuous_tokenizer.campaigns.state_budget.calculate_state_budget",
                    return_value=_result(),
                ),
                self.assertRaises(FileExistsError),
            ):
                run_state_budget(
                    Path("input-project"),
                    Path("output-project"),
                    output,
                )


if __name__ == "__main__":
    unittest.main()
