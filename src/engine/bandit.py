"""The steering bandit over discovery veins, an Optuna ask-and-tell study. A durable
RDBStorage (SQLite locally, Postgres on the cluster) lets a resumed campaign keep its
search state. Depth/breadth floors and the value-of-information split live above this;
the bandit is the standard acquisition the design deliberately did not hand-roll."""
from __future__ import annotations

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


class BanditError(Exception):
    """A bandit query with no valid answer (e.g. best_arm before any trial completed)."""


class Bandit:
    def __init__(self, n_arms: int, seed: int = 0, storage: str | None = None,
                 study_name: str | None = None):
        self.n_arms = n_arms
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )

    def ask(self):
        trial = self.study.ask()
        arm = trial.suggest_categorical("arm", list(range(self.n_arms)))
        return trial, arm

    def tell(self, trial, reward: float) -> None:
        self.study.tell(trial, reward)

    def best_arm(self) -> int:
        # A resumed-but-empty study (no COMPLETE trials) has no best; fail with a clear engine
        # error instead of Optuna's opaque ValueError. best_params considers only COMPLETE trials,
        # so a study with only pruned/failed trials must trip this guard too.
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in self.study.trials):
            raise BanditError("no completed trials: best_arm is undefined until a trial is told")
        return self.study.best_params["arm"]

    def n_trials(self) -> int:
        return len(self.study.trials)
