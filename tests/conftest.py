"""Shared test fixtures. The confirmatory tests all need the same building blocks (a valid
signed-style config with the real control-catalog hash, an EFFECT schema, a post-cutoff box, a
passing bundle). These live here so a test does not re-derive them and a change lands in one
place. This also puts the flat `cluster/` dir on sys.path so the pure cluster-scorer math
(cluster/scoring_math.py) is importable in tests without torch."""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "cluster"))

from backend import Box
from engine.agents import Bundle
from gateconfig import validate_config
from gatelib import library_digest
from referee import normalize_schema
from referee.lineage import control_catalog_digest


@pytest.fixture
def make_cfg():
    """Factory for a valid GateConfig. Pass control_catalog_hash=... to override the (correct)
    default, or any other field, e.g. to exercise a tampered-catalog HALT."""
    def _make(control_catalog_hash=None, **over):
        data = {
            "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
            "claim_types": ["effect"], "gate_library_digest": library_digest(),
            "control_catalog_hash": control_catalog_hash or control_catalog_digest(),
            "key_id": "test",
        }
        data.update(over)
        return validate_config(data)
    return _make


@pytest.fixture
def effect_cfg(make_cfg):
    return make_cfg()


@pytest.fixture
def effect_schema():
    return normalize_schema({
        "claim": "x improves recognition", "claim_type": "effect", "backbone": "iv2",
        "dataset": "ssv2", "scale": "7b", "measure": "accuracy", "prior_claim": False,
    })


@pytest.fixture
def post_cutoff_box():
    return Box(id="b", n=800, origin_date=date(2024, 6, 1))


@pytest.fixture
def passing_bundle():
    return Bundle.passing()
