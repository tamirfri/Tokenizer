from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from typing import Literal
from unittest import mock

import continuous_tokenizer.commands.freeze as freeze_module
from continuous_tokenizer.artifacts.evidence import (
    EvidenceIdentity,
    EvidenceManifest,
    write_evidence_manifest,
)
from continuous_tokenizer.artifacts.store import (
    load_json_object,
    write_json_atomic,
)
from continuous_tokenizer.commands.freeze import freeze
from continuous_tokenizer.contracts.claims import CLAIM_VOCABULARY_SHA256


def _seal(
    directory: Path,
    artifact_kind: Literal[
        "prospective_candidate_selection",
        "prospective_feasibility_screen",
    ],
) -> None:
    result = directory / "prospective.json"
    write_json_atomic(result, {})
    write_evidence_manifest(
        directory,
        EvidenceManifest(
            artifact_kind=artifact_kind,
            mode="input_only",
            status="completed",
            identity=EvidenceIdentity(
                source_commit="commit",
                source_dirty=False,
                source_state_sha256="a" * 64,
                dependency_lock_sha256="b" * 64,
                installed_package={
                    "name": "continuous-byte-tokenizer",
                    "version": "0.1.0",
                    "content_sha256": "c" * 64,
                },
                claim_vocabulary_sha256=CLAIM_VOCABULARY_SHA256,
                source_assets={},
                verification={"provided": False},
                model_id="Qwen/Qwen3.5-0.8B",
                model_revision="revision",
            ),
            parents={},
            inputs={},
            artifacts={"result": result},
        ),
    )


class FreezeTests(unittest.TestCase):
    def test_freeze_dispatches_only_current_candidate_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "selection"
            _seal(artifact, "prospective_candidate_selection")
            expected = {"status": "completed", "specifications": []}
            with (
                mock.patch.object(
                    freeze_module,
                    "source_state",
                    return_value=("commit", False, "a" * 64),
                ),
                mock.patch.object(
                    freeze_module,
                    "_freeze_prospective",
                    return_value=expected,
                ) as materialize,
            ):
                result = freeze(
                    argparse.Namespace(
                        artifacts=[artifact],
                        output_dir=root / "frozen",
                    ),
                )
            self.assertEqual(result, expected)
            materialize.assert_called_once()

    def test_freeze_rejects_suffixed_candidate_selection_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "selection"
            _seal(artifact, "prospective_candidate_selection")
            manifest_path = artifact / "evidence-manifest.json"
            manifest = dict(load_json_object(manifest_path))
            manifest["artifact_kind"] = "prospective_candidate_selection_v1"
            write_json_atomic(manifest_path, manifest)
            with (
                mock.patch.object(
                    freeze_module,
                    "source_state",
                    return_value=("commit", False, "a" * 64),
                ),
                self.assertRaisesRegex(ValueError, "invalid evidence artifact kind"),
            ):
                freeze(
                    argparse.Namespace(
                        artifacts=[artifact],
                        output_dir=root / "frozen",
                    ),
                )

    def test_freeze_rejects_other_current_artifact_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "screen"
            _seal(artifact, "prospective_feasibility_screen")
            with (
                mock.patch.object(
                    freeze_module,
                    "source_state",
                    return_value=("commit", False, "a" * 64),
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "only current prospective candidate-selection",
                ),
            ):
                freeze(
                    argparse.Namespace(
                        artifacts=[artifact],
                        output_dir=root / "frozen",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
