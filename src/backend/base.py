"""The Backend abstraction: a video model that can score a holdout box, either
trained or from a weights-randomized untrained init (for the FLOOR gate). A box is
scored per-item so the gates can form confidence intervals."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Box:
    id: str
    n: int
    origin_date: Optional[date] = None  # earliest public-availability date of the data


@runtime_checkable
class Backend(Protocol):
    cutoff_date: date  # the backbone's training cutoff, for the backbone gate

    def score_box(self, box: Box, untrained_init: Optional[int] = None) -> np.ndarray:
        """Per-item scores on the box. untrained_init=None scores the trained model;
        an int scores a weights-randomized untrained model at that init seed.

        UNIT CONTRACT: these are ABSOLUTE per-item scores (e.g. 0/1 MCQ correctness), NOT a
        pre-formed effect delta. A CAPABILITY / PHENOMENON claim compares this absolute level against
        its own signed threshold (the incumbent / null rate), which is unit-correct. An EFFECT claim's
        magnitude must be a DELTA on the MIE's scale, so the referee forms it as the trained-minus-
        untrained CONTRAST (the FLOOR residual) rather than comparing the absolute level to the MIE
        (that would be a tautology on any real benchmark). An untrained arm scored as an identically
        zero constant is a stub, not a model, and HALTs at the FLOOR (gatelib.floor.DegenerateFloorError)."""
        ...
