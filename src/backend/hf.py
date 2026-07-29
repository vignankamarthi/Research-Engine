"""The real-checkpoint loader. Cluster-fenced: transformers/torch are imported
LAZILY inside load(), so this module imports cleanly on the Mac and the class
type-checks against the Backend protocol without any heavy dependency. Actually
loading a checkpoint and scoring on a GPU is a human-intervention point."""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np

from .base import Box


class HFBackend:
    def __init__(self, model_id: str, revision: str, cutoff_date: date):
        self.model_id = model_id
        self.revision = revision  # pinned commit SHA for provenance
        self.cutoff_date = cutoff_date
        self._model = None

    def load(self) -> "HFBackend":
        # Lazy, cluster-only imports. Never imported at module load on the Mac.
        import torch  # noqa: F401
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(self.model_id, revision=self.revision)
        return self

    def score_box(self, box: Box, untrained_init: Optional[int] = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "HFBackend.load() must be called before scoring (cluster-only step)"
            )
        raise NotImplementedError("real checkpoint scoring runs only on the cluster")
