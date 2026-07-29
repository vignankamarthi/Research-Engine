"""Standard Benjamini-Hochberg FDR control. Deliberately the textbook procedure,
no bespoke e-BH: the disjoint-box discipline removes adaptivity, so plain BH is
valid and is differential-tested against statsmodels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BHResult:
    rejected: np.ndarray  # bool mask, in input order
    qvalues: np.ndarray   # BH-adjusted p-values, in input order


def benjamini_hochberg(pvalues, alpha: float) -> BHResult:
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return BHResult(np.zeros(0, dtype=bool), np.zeros(0, dtype=float))

    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # BH-adjusted p-values: running minimum of (m/i) * p_(i) from the largest rank down.
    q_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Step-up rejection: reject ranks 1..k for the largest k with p_(k) <= (k/m) * alpha.
    below = ranked <= (ranks / m) * alpha
    rej_sorted = np.zeros(m, dtype=bool)
    if below.any():
        kmax = int(np.nonzero(below)[0].max())
        rej_sorted[: kmax + 1] = True

    qvalues = np.empty(m, dtype=float)
    rejected = np.empty(m, dtype=bool)
    qvalues[order] = q_sorted
    rejected[order] = rej_sorted
    return BHResult(rejected=rejected, qvalues=qvalues)
