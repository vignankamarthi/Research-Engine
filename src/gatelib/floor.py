"""The untrained-weights FLOOR separation gate. The trained-minus-untrained residual
must exceed the MIE at power, measured paired on the same box against the worst-case
(highest-effect) of K pre-registered untrained inits, so a geometry artifact an
untrained model reproduces fails and a lucky-low untrained draw cannot pass one."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .effect import paired_diff_ci


class DegenerateFloorError(Exception):
    """The untrained FLOOR arm is degenerate (every init identically zero), i.e. a stub stood in for
    a scored random-init model. The geometry-artifact catcher is one of the two protections the whole
    design rests on, and it cannot run on a stand-in, so this HALTs rather than passing trivially (a
    zeros arm makes the residual equal the whole trained score). A genuine chance-level untrained
    model has variance and some non-zero items, so this catches the stub without rejecting a real
    control."""


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

    # HALT on a stub untrained arm: every init identically zero means no random-init model was scored.
    if all(float(np.max(np.abs(u))) == 0.0 for u in runs):
        raise DegenerateFloorError(
            "untrained FLOOR arm is identically zero across all inits -- a real random-init model was "
            "not scored, so the geometry-artifact catcher cannot run")

    worst_init = int(np.argmax([u.mean() for u in runs]))
    residual, lo, hi = paired_diff_ci(trained, runs[worst_init], alpha)
    return FloorResult(
        passed=lo > mie,
        residual=residual,
        ci_lo=lo,
        ci_hi=hi,
        worst_init=worst_init,
    )
