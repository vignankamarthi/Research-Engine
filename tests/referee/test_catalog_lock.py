"""The control-catalog-hash lock. The set of controls the schema-normal-form derives (the
mandatory controls, the floor-by-claim mapping, the magnitude-gate-by-type mapping) is a
CATALOG. Its canonical digest is signed into the GateConfig. Recomputing it at use and
comparing against the signed value means a weakening edit to the control set (dropping a
control, remapping a claim-type to a softer gate) is caught, not silently accepted."""
import pytest

from referee.lineage import (
    ControlCatalogError,
    control_catalog_digest,
    derive,
    normalize_schema,
    verify_control_catalog,
)


def _schema(prior=False):
    return normalize_schema({
        "claim": "temporal band k carries motion",
        "claim_type": "effect",
        "backbone": "videomae",
        "dataset": "ssv2",
        "scale": "base",
        "measure": "acc",
        "prior_claim": prior,
    })


def test_control_of_the_control_in_every_control_set():
    for prior in (True, False):
        assert "control_of_the_control" in derive(_schema(prior)).control_set


def test_catalog_digest_is_deterministic():
    assert control_catalog_digest() == control_catalog_digest()


def test_verify_accepts_the_matching_digest():
    verify_control_catalog(control_catalog_digest())  # must not raise


def test_verify_rejects_a_tampered_digest():
    with pytest.raises(ControlCatalogError):
        verify_control_catalog("deadbeef")
