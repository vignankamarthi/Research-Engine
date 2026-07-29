"""The untrained-weights FLOOR separation gate. The trained-minus-untrained residual
must exceed the MIE at power, measured paired on the same box against the worst-case
(highest-effect) of K pre-registered untrained inits, so a geometry artifact an
untrained model reproduces fails and a lucky-low untrained draw cannot pass one."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .effect import paired_diff_ci


@dataclass(frozen=True, slots=True)
class FloorResult:
    passed: bool
    residual: float
    ci_lo: float
    ci_hi: float
    worst_init: int


def floor_separation(trained, untrained_runs, mie: float, alpha: float = 0.05) -> FloorResult:
    trained = np.asarray(trained, dtype=float)
    if len(untrained_runs) == 0:
        raise ValueError("need at least one untrained init")
    runs = [np.asarray(u, dtype=float) for u in untrained_runs]
    for u in runs:
        if u.shape != trained.shape:
            raise ValueError("each untrained run must be paired on the same box items")

    worst_init = int(np.argmax([u.mean() for u in runs]))
    residual, lo, hi = paired_diff_ci(trained, runs[worst_init], alpha)
    return FloorResult(
        passed=lo > mie,
        residual=residual,
        ci_lo=lo,
        ci_hi=hi,
        worst_init=worst_init,
    )
