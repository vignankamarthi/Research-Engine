"""The VEIN SET as the bandit's arm space (PLAN 57b). The bandit can be constructed directly over
named veins, so its arm space IS the diversity axis (six derivative + three generative) rather than a
set of anonymous integer arms. The integer-arm construction (used elsewhere) keeps working."""
import pytest

from engine import Bandit, BanditError
from engine.generation import VEINS


def test_bandit_over_veins_labels_arms():
    b = Bandit(arms=VEINS, seed=0)
    assert b.n_arms == len(VEINS) == 9
    trial, arm = b.ask()
    assert b.arm_label(arm) in VEINS
    b.tell(trial, 1.0)


def test_integer_arm_construction_still_works():
    b = Bandit(n_arms=3, seed=0)
    assert b.arm_label(2) == 2  # falls back to the integer label
    with pytest.raises(BanditError):
        b.best_arm()


def test_bandit_requires_arms_or_n_arms():
    with pytest.raises(ValueError):
        Bandit()
