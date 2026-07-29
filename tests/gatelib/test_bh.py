"""Standard Benjamini-Hochberg over a pre-registered confirmatory set, provisional
at score time. The design forbids bespoke e-BH and requires a differential test
against a named third-party reference (statsmodels)."""
import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from gatelib import benjamini_hochberg


def test_empty_input():
    res = benjamini_hochberg([], alpha=0.05)
    assert res.rejected.size == 0
    assert res.qvalues.size == 0


def test_all_null_rejects_nothing():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 1.0, size=50)  # no signal
    res = benjamini_hochberg(p, alpha=0.05)
    assert not res.rejected.any()


def test_strong_signal_is_rejected():
    p = np.array([1e-6, 2e-6, 0.9, 0.8, 0.95])
    res = benjamini_hochberg(p, alpha=0.05)
    assert res.rejected[0] and res.rejected[1]
    assert not res.rejected[2:].any()


@pytest.mark.parametrize("seed", range(8))
def test_differential_against_statsmodels(seed):
    rng = np.random.default_rng(seed)
    # a mix of nulls (uniform) and signals (tiny) to exercise the step-up boundary
    p = np.concatenate([rng.uniform(0, 1, 40), rng.uniform(0, 1e-3, 10)])
    rng.shuffle(p)
    alpha = 0.05
    res = benjamini_hochberg(p, alpha)
    rej_ref, q_ref, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    assert np.array_equal(res.rejected, rej_ref)
    assert np.allclose(res.qvalues, q_ref, atol=1e-12)


def test_qvalues_bounded_and_monotone_in_rank():
    rng = np.random.default_rng(3)
    p = np.sort(rng.uniform(0, 1, 30))
    res = benjamini_hochberg(p, alpha=0.1)
    assert np.all(res.qvalues >= 0) and np.all(res.qvalues <= 1)
    # adjusted values are monotone non-decreasing along ascending raw p
    assert np.all(np.diff(res.qvalues) >= -1e-12)
