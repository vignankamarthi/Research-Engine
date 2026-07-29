"""G0 positive-control detectability. A pipeline is a callable pipeline(effect, rng)
-> p_value that runs the whole measurement path with a planted effect. G0 plants an
MDE-sized effect and requires the empirical detection power to clear the target, so
a pipeline that ignores the effect or collapses an arm FAILS and the verdict is
INELIGIBLE rather than a false negative."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class G0Result:
    passed: bool
    empirical_power: float


def g0_detectable(
    pipeline: Callable[[float, np.random.Generator], float],
    mde: float,
    alpha: float,
    power_target: float,
    n_trials: int,
    rng: np.random.Generator,
) -> G0Result:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    rejections = 0
    for _ in range(n_trials):
        p = pipeline(mde, rng)
        if p <= alpha:
            rejections += 1
    power = rejections / n_trials
    return G0Result(passed=power >= power_target, empirical_power=power)
