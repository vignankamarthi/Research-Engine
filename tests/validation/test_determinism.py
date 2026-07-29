"""Determinism and teeth (step 25). The referee returns the identical verdict on a rerun with
the same inputs (a prerequisite for reproducibility across reruns and nodes), and the gates
actually discriminate: a boundary case flips the verdict, so an inverted comparison would be
caught rather than silently passing."""
from datetime import date

import numpy as np

from backend import Box, MockBackend
from engine.agents import Bundle
from gateconfig import validate_config
from gatelib import library_digest
from gatelib import EXCEEDS_MIE, POWERED_NULL, classify_magnitude
from referee import normalize_schema
from referee.lineage import control_catalog_digest
from referee.runner import confirm

BOX = Box(id="b", n=800, origin_date=date(2024, 6, 1))


def _cfg():
    return validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": control_catalog_digest(), "key_id": "t",
    })


def _schema():
    return normalize_schema({"claim": "x improves recognition", "claim_type": "effect",
                             "backbone": "iv2", "dataset": "ssv2", "scale": "7b",
                             "measure": "accuracy", "prior_claim": False})


def test_confirm_is_deterministic_across_reruns():
    def run():
        return confirm(MockBackend(0.25, 0.0, 0.1, seed=7), BOX, _schema(), _cfg(), Bundle.passing())

    a, b = run(), run()
    assert (a.status, a.reason, a.pvalue) == (b.status, b.reason, b.pvalue)


def test_mock_backend_scores_are_reproducible():
    s1 = MockBackend(0.25, 0.0, 0.1, seed=7).score_box(BOX)
    s2 = MockBackend(0.25, 0.0, 0.1, seed=7).score_box(BOX)
    assert np.allclose(s1, s2)


def test_magnitude_gate_discriminates_at_the_boundary():
    # just above the MIE is a positive, just below is a powered null. An inverted comparison
    # would flip both of these, so the suite has teeth here.
    assert classify_magnitude(0.031, 0.05, mie=0.03) == EXCEEDS_MIE
    assert classify_magnitude(0.01, 0.029, mie=0.03) == POWERED_NULL
