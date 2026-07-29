"""The Backend abstraction and its deterministic MockBackend. The confirmatory core
is validated on the Mac by driving the gates with a MockBackend; the real checkpoint
loader (HFBackend) is written but cluster-fenced. A box is scored reproducibly here
for testing, though the live loop touches each box exactly once."""
from datetime import date

import numpy as np

from backend import Backend, Box, MockBackend
from gatelib import floor_separation


def test_mock_conforms_to_backend_protocol():
    mb = MockBackend(trained_effect=0.2, untrained_effect=0.0, noise=0.1, seed=0)
    assert isinstance(mb, Backend)


def test_score_box_is_deterministic_per_init():
    mb = MockBackend(0.2, 0.0, 0.1, seed=7)
    box = Box(id="b1", n=100)
    assert np.array_equal(mb.score_box(box), mb.score_box(box))
    assert np.array_equal(mb.score_box(box, untrained_init=0), mb.score_box(box, untrained_init=0))
    # a different untrained init is a different draw
    assert not np.array_equal(mb.score_box(box, untrained_init=0), mb.score_box(box, untrained_init=1))
    # trained is distinct from any untrained init
    assert not np.array_equal(mb.score_box(box), mb.score_box(box, untrained_init=0))


def test_clean_backend_passes_floor():
    mb = MockBackend(trained_effect=0.25, untrained_effect=0.0, noise=0.1, seed=1)
    box = Box(id="clean", n=600)
    trained = mb.score_box(box)
    untrained = [mb.score_box(box, untrained_init=k) for k in range(4)]
    assert floor_separation(trained, untrained, mie=0.05).passed


def test_artifact_backend_fails_floor():
    mb = MockBackend(trained_effect=0.25, untrained_effect=0.25, noise=0.1, seed=2)
    box = Box(id="artifact", n=600)
    trained = mb.score_box(box)
    untrained = [mb.score_box(box, untrained_init=k) for k in range(4)]
    assert not floor_separation(trained, untrained, mie=0.05).passed


def test_cutoff_date_exposed_for_backbone_gate():
    mb = MockBackend(0.2, 0.0, 0.1, seed=0, cutoff_date=date(2023, 6, 1))
    assert mb.cutoff_date == date(2023, 6, 1)
