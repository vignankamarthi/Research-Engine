"""Effect-size confidence intervals feeding the magnitude and FLOOR gates."""
import numpy as np
import pytest

from gatelib import mean_ci, paired_diff_ci


def test_constant_array_has_zero_width_ci():
    lo, hi = mean_ci(np.full(20, 0.3))
    assert lo == hi  # zero variance -> zero-width interval
    assert np.isclose(lo, 0.3)


def test_ci_brackets_the_true_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 2000)
    lo, hi = mean_ci(x, alpha=0.05)
    assert lo < 0.5 < hi
    assert hi - lo < 0.1  # tight at n=2000


def test_too_few_points_rejected():
    with pytest.raises(ValueError):
        mean_ci([1.0])


def test_paired_diff_centered_on_constant_shift():
    a = np.array([0.1, 0.2, 0.3, 0.4])
    b = a + 0.25
    mean, lo, hi = paired_diff_ci(a, b)  # mean(a - b) = -0.25, zero variance
    assert np.isclose(mean, -0.25) and np.isclose(lo, -0.25) and np.isclose(hi, -0.25)


def test_paired_diff_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        paired_diff_ci([1, 2, 3], [1, 2])
