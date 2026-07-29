"""The real-checkpoint backend. Cluster-fenced: torch/transformers are imported LAZILY inside
load(), so this module imports cleanly on the Mac and the class type-checks against the Backend
protocol without any heavy dependency. Actually loading a checkpoint and scoring on a GPU runs
only on the cluster.

The adapter stays GENERIC: it loads the model and delegates the per-item metric to an injected
`scorer(model, box, untrained_init) -> per-item scores`. The experiment-specific logic (what to
measure, and how the untrained-FLOOR control re-initializes the weights when `untrained_init` is
set) lives in the scorer, which ships with a hypothesis. The adapter only loads the checkpoint and
validates the shape, so its dispatch logic is testable on the Mac without torch."""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np

from .base import Box


class HFBackend:
    def __init__(self, model_id: str, revision: str, cutoff_date: date, scorer=None):
        self.model_id = model_id
        self.revision = revision  # pinned commit SHA for provenance
        self.cutoff_date = cutoff_date
        self._scorer = scorer     # (model, box, untrained_init) -> np.ndarray of length box.n
        self._model = None

    def load(self) -> "HFBackend":
        # Lazy, cluster-only imports. Never imported at module load on the Mac.
        import torch  # noqa: F401
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(self.model_id, revision=self.revision)
        return self

    def score_box(self, box: Box, untrained_init: Optional[int] = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HFBackend.load() must be called before scoring (cluster-only step)")
        if self._scorer is None:
            raise RuntimeError("HFBackend needs a scorer, the experiment's per-item metric")
        # untrained_init is None -> score the trained model; an int -> the scorer re-initializes
        # the weights from that seed for the untrained-FLOOR control (worst-of-K inits).
        scores = np.asarray(self._scorer(self._model, box, untrained_init), dtype=float)
        if scores.shape != (box.n,):
            raise ValueError(f"scorer must return {box.n} per-item scores, got shape {scores.shape}")
        return scores
