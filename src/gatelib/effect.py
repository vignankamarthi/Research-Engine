"""Effect-size confidence intervals (Student-t) for the magnitude and FLOOR gates."""
from __future__ import annotations

import numpy as np
from scipy.stats import t as _t


def mean_ci(x, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need at least two observations for a CI")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        return mean, mean
    crit = float(_t.ppf(1.0 - alpha / 2.0, n - 1))
    return mean - crit * se, mean + crit * se


def paired_diff_ci(a, b, alpha: float = 0.05) -> tuple[float, float, float]:
    """CI of mean(a - b), paired element-wise. Returns (mean, lo, hi)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match shape: {a.shape} vs {b.shape}")
    d = a - b
    lo, hi = mean_ci(d, alpha)
    return float(d.mean()), lo, hi
