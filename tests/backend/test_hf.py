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


def test_hf_backend_requires_a_scorer():
    from backend import Box
    hb = HFBackend("fake/model", "deadbeef", date(2024, 1, 1))  # no scorer
    hb._model = object()  # bypass load(), so no torch is needed to test the dispatch logic
    with pytest.raises(RuntimeError):
        hb.score_box(Box(id="x", n=4))


def test_hf_backend_dispatches_to_scorer_and_passes_untrained_init():
    import numpy as np

    from backend import Box
    seen = []

    def scorer(model, box, untrained_init):
        seen.append(untrained_init)
        return np.full(box.n, 0.5)

    hb = HFBackend("fake/model", "deadbeef", date(2024, 1, 1), scorer=scorer)
    hb._model = object()
    box = Box(id="x", n=6)
    assert hb.score_box(box).shape == (6,) and seen[-1] is None
    hb.score_box(box, untrained_init=3)
    assert seen[-1] == 3  # the FLOOR control's init seed reaches the scorer


def test_hf_backend_validates_scorer_output_shape():
    import numpy as np

    from backend import Box
    hb = HFBackend("fake/model", "deadbeef", date(2024, 1, 1),
                   scorer=lambda model, box, ui: np.zeros(3))  # wrong length
    hb._model = object()
    with pytest.raises(ValueError):
        hb.score_box(Box(id="x", n=5))


@pytest.mark.cluster
def test_hf_backend_loads_and_scores_on_cluster():
    # Human-intervention point: real cached checkpoint on a real GPU. Never run locally.
    import numpy as np

    from backend import Box

    def scorer(model, box, untrained_init):
        return np.zeros(box.n)  # trivial metric standing in for the real experiment

    hb = HFBackend(model_id="Qwen/Qwen2.5-VL-7B-Instruct", revision="main",
                   cutoff_date=date(2024, 1, 1), scorer=scorer)
    hb.load()
    scores = hb.score_box(Box(id="real", n=8))
    assert scores.shape == (8,)
