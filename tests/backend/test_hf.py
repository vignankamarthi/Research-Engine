"""The real-checkpoint loader is written but cluster-fenced. It must satisfy the
Backend protocol WITHOUT importing transformers or loading a model (a lazy import
inside load()), so the local suite can type-check it. Actually loading a 7B
checkpoint on a B200 is a human-intervention point and is marked `cluster`."""
from datetime import date

import pytest

from backend import Backend
from backend.hf import HFBackend


def test_hf_backend_satisfies_protocol_without_loading():
    hb = HFBackend(model_id="fake/model", revision="deadbeef", cutoff_date=date(2024, 1, 1))
    assert isinstance(hb, Backend)


def test_hf_backend_scoring_before_load_is_an_error():
    hb = HFBackend(model_id="fake/model", revision="deadbeef", cutoff_date=date(2024, 1, 1))
    from backend import Box
    with pytest.raises(RuntimeError):
        hb.score_box(Box(id="x", n=4))


@pytest.mark.cluster
def test_hf_backend_loads_and_scores_on_cluster():
    # Human-intervention point: real checkpoint on a real GPU. Never run locally.
    hb = HFBackend(model_id="OpenGVLab/InternVideo2", revision="main",
                   cutoff_date=date(2024, 1, 1))
    hb.load()
    from backend import Box
    scores = hb.score_box(Box(id="real", n=8))
    assert scores.shape == (8,)
