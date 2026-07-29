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
        an int scores a weights-randomized untrained model at that init seed."""
        ...
