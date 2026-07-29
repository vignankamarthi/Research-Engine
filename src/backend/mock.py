"""A deterministic MockBackend for validating the confirmatory core on the Mac.
Determinism is process-stable (hashlib, not the salted builtin hash) so resume and
reruns reproduce the same box scores. A clean backend (untrained effect ~ 0) passes
the FLOOR; an artifact backend (untrained ~ trained) fails it."""
from __future__ import annotations

import hashlib
from datetime import date
from typing import Optional

import numpy as np

from .base import Box


def _seed(seed: int, box_id: str, tag: int) -> int:
    digest = hashlib.sha256(f"{seed}|{box_id}|{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class MockBackend:
    def __init__(self, trained_effect: float, untrained_effect: float, noise: float,
                 seed: int, cutoff_date: date = date(2024, 1, 1)):
        self.trained_effect = float(trained_effect)
        self.untrained_effect = float(untrained_effect)
        self.noise = float(noise)
        self.seed = int(seed)
        self.cutoff_date = cutoff_date

    def score_box(self, box: Box, untrained_init: Optional[int] = None) -> np.ndarray:
        tag = -1 if untrained_init is None else untrained_init
        rng = np.random.default_rng(_seed(self.seed, box.id, tag))
        mean = self.trained_effect if untrained_init is None else self.untrained_effect
        return rng.normal(mean, self.noise, box.n)
