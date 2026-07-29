"""G0 positive-control detectability. No verdict is admissible unless G0 first
proves the pipeline can detect an MDE-sized planted effect at the target power.
A broken pipeline (one that ignores the planted effect or collapses an arm) must
FAIL G0, so an upstream fault reads as INELIGIBLE, never a clean negative."""
import numpy as np
from scipy.stats import ttest_ind

from gatelib import g0_detectable


def good_pipeline(effect, rng, n=200):
    """A sound measurement path: plant `effect` in the treatment arm, one-sided t."""
    control = rng.normal(0.0, 1.0, n)
    treatment = rng.normal(effect, 1.0, n)
    return float(ttest_ind(treatment, control, alternative="greater").pvalue)


def ignores_effect_pipeline(effect, rng, n=200):
    """A broken loader: both arms drawn identically, the planted effect never lands."""
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 1.0, n)
    return float(ttest_ind(b, a, alternative="greater").pvalue)


def collapsed_arm_pipeline(effect, rng, n=200):
    """A broken loader returning a single clip: degenerate, no power."""
    return 0.9


def test_good_pipeline_passes_g0():
    rng = np.random.default_rng(1)
    res = g0_detectable(good_pipeline, mde=0.4, alpha=0.05, power_target=0.8,
                        n_trials=200, rng=rng)
    assert res.passed
    assert res.empirical_power >= 0.8


def test_pipeline_that_ignores_the_effect_fails_g0():
    rng = np.random.default_rng(2)
    res = g0_detectable(ignores_effect_pipeline, mde=0.4, alpha=0.05, power_target=0.8,
                        n_trials=200, rng=rng)
    assert not res.passed
    assert res.empirical_power < 0.8


def test_collapsed_arm_fails_g0():
    rng = np.random.default_rng(3)
    res = g0_detectable(collapsed_arm_pipeline, mde=0.4, alpha=0.05, power_target=0.8,
                        n_trials=100, rng=rng)
    assert not res.passed
    assert res.empirical_power == 0.0
