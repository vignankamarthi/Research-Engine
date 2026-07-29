"""The steering bandit (Optuna ask-and-tell). Discovery steers compute toward the
higher-reward veins on dev data. The store is durable (SQLite/RDBStorage) so a
resumed campaign keeps its search state."""
import numpy as np
import pytest

from engine import Bandit, BanditError


def test_best_arm_on_empty_study_raises_bandit_error():
    b = Bandit(n_arms=3, seed=0)
    with pytest.raises(BanditError):
        b.best_arm()


def test_bandit_steers_toward_the_best_arm():
    rng = np.random.default_rng(0)
    b = Bandit(n_arms=4, seed=0)
    true_reward = [0.1, 0.2, 0.9, 0.3]  # arm 2 is best
    counts = [0, 0, 0, 0]
    for _ in range(80):
        trial, arm = b.ask()
        b.tell(trial, true_reward[arm] + rng.normal(0, 0.05))
        counts[arm] += 1
    assert int(np.argmax(counts)) == 2   # most compute spent on the best vein
    assert b.best_arm() == 2


def test_bandit_state_is_durable(tmp_path):
    url = f"sqlite:///{tmp_path/'study.db'}"
    b = Bandit(n_arms=3, seed=1, storage=url, study_name="camp")
    for _ in range(10):
        trial, arm = b.ask()
        b.tell(trial, 1.0 if arm == 1 else 0.0)
    # a fresh bandit on the same store sees the prior trials
    b2 = Bandit(n_arms=3, seed=1, storage=url, study_name="camp")
    assert b2.n_trials() >= 10
