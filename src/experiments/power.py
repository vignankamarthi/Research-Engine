"""Box-power sizing. Given the gap a claim must clear (the MIE for an effect or a mechanism
contrast, the incumbent gap for a capability separation) and the per-item standard deviation of
the scored quantity, return the number of items a holdout box needs so a true effect of that size
is detected at the target power. This SIZES the boxes a campaign carves. It never gates a verdict,
so it lives outside the signed gate library. Sizing the box BEFORE fixing the maturation count is
what keeps a task like TOMATO (1,484 items) from being asked for more powered boxes than it holds."""
from __future__ import annotations

import math

from scipy.stats import norm


def required_n(delta: float, sd: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items per box to detect a one-sided mean shift `delta` at per-item SD `sd`, at significance
    `alpha` and `power` (the standard z-based normal approximation). Raises on a non-positive delta
    or sd, since a zero-or-negative target has no finite sample size."""
    if not (delta > 0.0):
        raise ValueError(f"delta must be a positive detectable gap, got {delta}")
    if not (sd > 0.0):
        raise ValueError(f"sd must be a positive per-item standard deviation, got {sd}")
    z_alpha = float(norm.ppf(1.0 - alpha))
    z_power = float(norm.ppf(power))
    n = ((z_alpha + z_power) * sd / delta) ** 2
    return int(math.ceil(n))


def proportion_sd(p: float) -> float:
    """The per-item SD of a Bernoulli success rate p. Sizing on the worst case (p near 0.5) never
    under-powers a box."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be a probability, got {p}")
    return math.sqrt(p * (1.0 - p))
