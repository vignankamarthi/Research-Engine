"""Per-task MIE resolution. The interest bar is field-anchored PER TASK (the mie_distribution
catalog's top-quartile for that task), not one flat floor. A task with no signed entry falls back
to the config's mie_floor. A resolved MIE at or below the detectability floor (mde) raises, and a
tampered catalog raises. The runner then gates on the substrate-resolved bundle.mie, so a multi-task
campaign gates each task on its own bar."""
from datetime import date

import pytest

from backend import Box, MockBackend
from engine.agents import Bundle
from gateconfig import validate_config
from gatelib import library_digest
from referee import normalize_schema
from referee.catalog import CatalogError, catalog_digest, resolve_mie
from referee.lineage import control_catalog_digest
from referee.runner import confirm

MIE = {
    "ssv2_recognition_top1": {"mie_value": 0.01},
    "kinetics400_recognition_top1": {"mie_value": 0.011},
}
DIG = catalog_digest(MIE)


def test_per_task_mie_is_read_from_the_catalog():
    assert resolve_mie("ssv2_recognition_top1", MIE, DIG, fallback=0.03, mde=0.005) == 0.01
    assert resolve_mie("kinetics400_recognition_top1", MIE, DIG, fallback=0.03, mde=0.005) == 0.011


def test_unlisted_task_falls_back_to_the_floor():
    assert resolve_mie("epic_kitchens", MIE, DIG, fallback=0.03, mde=0.005) == 0.03


def test_mie_at_or_below_detectability_raises():
    below = {"weird_task": {"mie_value": 0.004}}
    with pytest.raises(CatalogError):
        resolve_mie("weird_task", below, catalog_digest(below), fallback=0.03, mde=0.005)


def test_fallback_below_detectability_raises():
    with pytest.raises(CatalogError):
        resolve_mie("unlisted", MIE, DIG, fallback=0.003, mde=0.005)


def test_tampered_catalog_raises():
    with pytest.raises(CatalogError):
        resolve_mie("ssv2_recognition_top1", MIE, "sha256:wrong", fallback=0.03, mde=0.005)


def test_entry_without_mie_value_raises():
    bad = {"t": {"note": "no value"}}
    with pytest.raises(CatalogError):
        resolve_mie("t", bad, catalog_digest(bad), fallback=0.03, mde=0.005)


# --- the runner honors bundle.mie over the config floor ---

def _cfg():
    return validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.005, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": control_catalog_digest(), "key_id": "test",
    })


def _schema():
    return normalize_schema({"claim": "x improves recognition", "claim_type": "effect",
                             "backbone": "iv2", "dataset": "ssv2", "scale": "7b",
                             "measure": "accuracy", "prior_claim": False})


_BOX = Box(id="b", n=800, origin_date=date(2024, 6, 1))


def test_confirm_uses_bundle_mie_when_present():
    # effect 0.25 clears the config floor (0.03) but NOT a per-task MIE of 0.5, so setting bundle.mie
    # high must change the verdict, proving the runner gates on the per-task bar, not the flat floor.
    be = MockBackend(trained_effect=0.25, untrained_effect=0.0, noise=0.1, seed=1)
    confirmed = confirm(be, _BOX, _schema(), _cfg(), Bundle.passing())  # mie None -> floor 0.03
    gated = confirm(be, _BOX, _schema(), _cfg(), Bundle.passing(mie=0.5))  # per-task bar 0.5
    assert confirmed.status == "CONFIRMED"
    assert gated.status != "CONFIRMED"
