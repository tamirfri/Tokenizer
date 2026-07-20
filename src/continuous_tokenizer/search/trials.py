from __future__ import annotations

import optuna
from optuna.trial import FrozenTrial, TrialState

_FINISHED_STATES = frozenset(
    {
        TrialState.COMPLETE,
        TrialState.FAIL,
        TrialState.PRUNED,
    }
)


def _finished_trials(trials: list[FrozenTrial]) -> int:
    return sum(trial.state in _FINISHED_STATES for trial in trials)


def _remaining_trials(trials: list[FrozenTrial], requested: int) -> int:
    return max(0, requested - _finished_trials(trials))


def _reconcile_running_trials(study: optuna.Study) -> None:
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
