from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from itertools import pairwise
from pathlib import Path

from continuous_tokenizer.artifacts.evidence import (
    verify_artifact,
)
from continuous_tokenizer.commands.evidence import aggregate
from continuous_tokenizer.contracts.claim_derivation import GEMMA_MODEL, QWEN_MODEL
from continuous_tokenizer.contracts.claims import directional_claims
from continuous_tokenizer.contracts.statements import (
    SoftwareValidationInputs,
    SourceBinding,
    SyntheticCampaignEvidence,
    VerificationEvidence,
)
from continuous_tokenizer.reporting.project import assemble_project_artifact
from continuous_tokenizer.reporting.readme import (
    README_LEDGER_BEGIN,
    README_LEDGER_END,
    update_readme,
)
from tests.test_replication import write_completed_run


def _write_project(directory: Path) -> None:
    root = directory.parent
    replications = []
    for model in (QWEN_MODEL, GEMMA_MODEL):
        runs = tuple(
            write_completed_run(
                root / f"{model[0].split('/')[-1]}-{seed}",
                seed=seed,
                density=1.25,
                model=model,
            )
            for seed in (17, 23, 41)
        )
        replication = root / f"{model[0].split('/')[-1]}-replication"
        aggregate(Namespace(runs=runs, output_dir=replication))
        replications.append(replication)
    assemble_project_artifact(replications, directory)
    assert verify_artifact(directory)["valid"]


def _readme(path: Path) -> None:
    path.write_text(
        f"# Paper\n\n{README_LEDGER_BEGIN}\nold\n{README_LEDGER_END}\n\n## Methods\n",
        encoding="utf-8",
    )


def _software_validation() -> SoftwareValidationInputs:
    source = SourceBinding("d" * 64, "e" * 64)
    return SoftwareValidationInputs(
        verification=VerificationEvidence(source, True, "verification/verification.json"),
        synthetic_campaigns=(
            SyntheticCampaignEvidence(
                "input_only",
                source,
                "completed",
                True,
                "synthetic-input/result.json",
            ),
            SyntheticCampaignEvidence(
                "output_only",
                source,
                "completed",
                True,
                "synthetic-output/result.json",
            ),
        ),
    )


def test_readme_ledger_defaults_to_incomplete_and_check_is_pure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "README.md"
        _readme(path)

        assert update_readme(path)
        generated = path.read_text(encoding="utf-8")
        assert generated.count(README_LEDGER_BEGIN) == 1
        assert generated.count(README_LEDGER_END) == 1
        assert generated.count("Status: **PROVED**") == 3
        assert generated.count("Status: **NOT VALIDATED**") == 4
        assert "Status: **VALIDATED**" not in generated
        assert "**No final semantically verified sealed project evidence was supplied.**" in generated
        assert generated.count("Verdict: **INCOMPLETE**") == len(directional_claims("input_only")) + len(directional_claims("output_only"))
        assert "Verdict: **UNSUPPORTED**" not in generated
        input_ledger = generated.split("### Input-only claims", 1)[1].split(
            "### Output-only claims",
            1,
        )[0]
        assert (
            input_ledger.index("#### Primary claims")
            < input_ledger.index("#### Prerequisite claims")
            < input_ledger.index("#### Secondary claims")
            < input_ledger.index("#### Applicability claims")
        )
        assert update_readme(path, check=True) is False

        drifted = generated.replace("Input-only claims", "Drifted claims", 1)
        path.write_text(drifted, encoding="utf-8")
        before = path.read_bytes()
        with unittest.TestCase().assertRaisesRegex(ValueError, "out of date"):
            update_readme(path, check=True)
        assert path.read_bytes() == before


def test_readme_ledger_accepts_only_verified_sealed_directional_projects() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "README.md"
        project = root / "input-project"
        _readme(path)
        _write_project(project)

        update_readme(path, input_project=project)

        generated = path.read_text(encoding="utf-8")
        assert "Full Vocabulary Embedding Compatibility" in generated
        assert "Verdict: **SUPPORTED**" in generated
        assert "project.json#/claims/0" in generated
        assert "Evidence manifest SHA-256" in generated
        assert "Decisive metric or policy" in generated
        assert "Scope:" in generated
        assert generated.count("Status: **PROVED**") == 3
        assert "Status: **VALIDATED**" not in generated

        (project / "project.json").write_text("{}", encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(
            ValueError,
            "verification failed|hash mismatch",
        ):
            update_readme(path, input_project=project)


def test_readme_ledger_validates_only_with_explicit_source_bound_inputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "README.md"
        _readme(path)

        update_readme(path, software_validation=_software_validation())

        generated = path.read_text(encoding="utf-8")
        assert generated.count("Status: **PROVED**") == 3
        assert generated.count("Status: **VALIDATED**") == 4
        assert "Status: **NOT VALIDATED**" not in generated
        assert "`verification/verification.json`" in generated
        assert generated.count("Verdict: **INCOMPLETE**") == len(directional_claims("input_only")) + len(directional_claims("output_only"))


def test_checked_in_readme_ledger_has_no_generated_drift() -> None:
    repository = Path(__file__).parents[1]
    assert update_readme(repository / "README.md", check=True) is False


def test_checked_in_readme_is_linear_evidence_first_paper() -> None:
    repository = Path(__file__).parents[1]
    readme = (repository / "README.md").read_text(encoding="utf-8")
    headings = (
        "## Abstract",
        "## Research question and contributions",
        "## Method",
        "## Formal guarantees",
        "## Empirical hypotheses",
        "## Experiment design",
        "## Current results",
        "## Limitations",
        "## Reproducibility",
    )

    assert all(readme.index(first) < readme.index(second) for first, second in pairwise(headings))
    assert (
        readme.count(
            "usable input compression = exact held-out position compression + registered behavioral similarity",
        )
        == 1
    )
    assert "No completed three-seed Qwen or Gemma final replication is present." in readme
    assert "Performance cannot substitute for either operand." in readme
    assert "Runtime or cache speedups do not support it." in readme.replace(
        "\n  ",
        " ",
    )
    assert "Qwen/Qwen3.5-0.8B" in readme
    assert "google/gemma-3-270m-it" in readme
    assert "confirmation model" not in readme.lower()
    assert "lossless dense" not in readme.lower()
    assert all(f"**{verdict}**" in readme for verdict in ("INCOMPLETE",))


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_readme_ledger_defaults_to_incomplete_and_check_is_pure,
            test_readme_ledger_accepts_only_verified_sealed_directional_projects,
            test_readme_ledger_validates_only_with_explicit_source_bound_inputs,
            test_checked_in_readme_ledger_has_no_generated_drift,
            test_checked_in_readme_is_linear_evidence_first_paper,
        )
    )
