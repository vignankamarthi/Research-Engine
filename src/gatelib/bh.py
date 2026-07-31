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


def benjamini_hochberg(pvalues, alpha: float, n_tests: int | None = None) -> BHResult:
    p = np.asarray(pvalues, dtype=float)
    k = p.size  # p-value-bearing looks being ranked
    if k == 0:
        return BHResult(np.zeros(0, dtype=bool), np.zeros(0, dtype=float))

    # The correction denominator is the FULL selection-family size m (>= k). A box-touching look with
    # no scalar p-value (law_shape, a functional-form fit) is counted in m via n_tests so the ranked
    # looks stay honestly conservative for the true number of looks. n_tests=None -> plain BH (m = k).
    m = k if n_tests is None else max(int(n_tests), k)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, k + 1)

    # BH-adjusted p-values: running minimum of (m/i) * p_(i) from the largest rank down.
    q_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Step-up rejection: reject ranks 1..j for the largest j with p_(j) <= (j/m) * alpha.
    below = ranked <= (ranks / m) * alpha
    rej_sorted = np.zeros(k, dtype=bool)
    if below.any():
        jmax = int(np.nonzero(below)[0].max())
        rej_sorted[: jmax + 1] = True

    qvalues = np.empty(k, dtype=float)
    rejected = np.empty(k, dtype=bool)
    qvalues[order] = q_sorted
    rejected[order] = rej_sorted
    return BHResult(rejected=rejected, qvalues=qvalues)
