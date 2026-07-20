from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from continuous_tokenizer.campaigns.input import InputExperimentRunner
from continuous_tokenizer.campaigns.output import OutputExperimentRunner, OutputRunnerOptions
from continuous_tokenizer.contracts.experiment import ExperimentSpec
from continuous_tokenizer.input.studies import RegisteredVocabularySubsetRequest

if TYPE_CHECKING:
    from continuous_tokenizer.input.prospective import ProspectiveExecutionPolicy


def create_experiment_runner(  # noqa: PLR0913 - Runner dependencies remain explicit.
    spec: ExperimentSpec,
    output_dir: Path,
    project_root: Path,
    verification_path: Path | None = None,
    *,
    resume: bool = False,
    prospective_input_subset: RegisteredVocabularySubsetRequest | None = None,
    prospective_execution_policy: ProspectiveExecutionPolicy | None = None,
) -> InputExperimentRunner | OutputExperimentRunner:
    if spec.mode == "output_only":
        if prospective_input_subset is not None:
            raise ValueError(
                "output runs cannot receive an input vocabulary subset request",
            )
        return OutputExperimentRunner(
            spec,
            output_dir,
            project_root,
            OutputRunnerOptions(
                verification_path=verification_path,
                resume=resume,
                prospective_policy=prospective_execution_policy,
            ),
        )
    return InputExperimentRunner(
        spec,
        output_dir,
        project_root,
        verification_path,
        resume=resume,
        prospective_subset=prospective_input_subset,
        prospective_policy=prospective_execution_policy,
    )
