"""The remaining adversarial vectors, each caught by a distinct guard, not by luck: leaked
labels (eval data that was in training), and in-process referee subversion (fabricated score
artifacts, pickle payloads, and a tampered control catalog)."""
from datetime import date

import pytest

from backend import Box, MockBackend
from engine.agents import Bundle
from gateconfig import validate_config
from gatelib import library_digest
from referee import normalize_schema
from referee.lineage import ControlCatalogError, control_catalog_digest
from referee.provenance import RunProvenance, artifact_checksum, executed_not_fabricated
from referee.runner import confirm
from referee.safeio import SafeFormatError, safe_load

BOX = Box(id="b", n=800, origin_date=date(2024, 6, 1))


def _cfg(catalog_hash=None):
    return validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": catalog_hash or control_catalog_digest(), "key_id": "t",
    })


def _schema():
    return normalize_schema({"claim": "x improves recognition", "claim_type": "effect",
                             "backbone": "iv2", "dataset": "ssv2", "scale": "7b",
                             "measure": "accuracy", "prior_claim": False})


def test_leaked_labels_are_caught_by_the_membership_check():
    # a strong effect, but the eval data leaked into training (membership not clean).
    be = MockBackend(trained_effect=0.25, untrained_effect=0.0, noise=0.1, seed=1)
    v = confirm(be, BOX, _schema(), _cfg(), Bundle.passing(membership_clean=False))
    assert v.status == "INELIGIBLE" and "backbone" in v.reason


def test_fabricated_scores_fail_provenance():
    prov = RunProvenance(artifact_checksum(b"real-scores"), fingerprint_digest="fp", seeds=(1,))
    assert executed_not_fabricated(b"fabricated-scores", prov) is False


def test_pickle_artifact_is_refused_by_extension(tmp_path):
    import pickle
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps({"x": 1}))
    with pytest.raises(SafeFormatError):
        safe_load(str(p), "json")


def test_pickle_magic_byte_is_refused_even_disguised_as_json(tmp_path):
    import pickle
    p = tmp_path / "scores.json"
    p.write_bytes(pickle.dumps([1, 2, 3]))  # a pickle stream starts with 0x80
    with pytest.raises(SafeFormatError):
        safe_load(str(p), "json")


def test_tampered_control_catalog_halts_the_referee():
    be = MockBackend(0.25, 0.0, 0.1, seed=1)
    with pytest.raises(ControlCatalogError):
        confirm(be, BOX, _schema(), _cfg("sha256:" + "9" * 64), Bundle.passing())
