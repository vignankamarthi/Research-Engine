"""Signed catalogs for the anti-HARKing wiring. The consequence a positive must discharge
is drawn from a signed TEMPLATE keyed to the claim-type, and the incumbent it must beat is
drawn from a signed per-task catalog. Both are verified against their digest at use, so
neither can be authored at handoff to fit the result. The incumbent separation is COMPUTED
from claimed vs incumbent values at the MIE, not asserted."""
import pytest

from referee.catalog import (
    CatalogError,
    catalog_digest,
    incumbent_separated,
    resolve_consequence_template,
    resolve_incumbent,
    verify_catalog,
)

CONSEQ = {
    "effect": "downstream accuracy rises by >= MIE on held-out task T",
    "capability": "solves held-out instances the incumbent fails",
}
INCUMB = {"ssv2_recognition": 0.70}


def test_catalog_digest_is_deterministic():
    assert catalog_digest(CONSEQ) == catalog_digest(CONSEQ)


def test_verify_accepts_matching_digest():
    verify_catalog(CONSEQ, catalog_digest(CONSEQ))


def test_verify_rejects_tampered_digest():
    with pytest.raises(CatalogError):
        verify_catalog(CONSEQ, "bad")


def test_resolve_consequence_template_returns_the_signed_template():
    got = resolve_consequence_template("effect", CONSEQ, catalog_digest(CONSEQ))
    assert got.startswith("downstream")


def test_resolve_consequence_template_refuses_unknown_claim_type():
    # anti-HARKing: a claim-type with no pre-registered template cannot get one at handoff.
    with pytest.raises(CatalogError):
        resolve_consequence_template("law_shape", CONSEQ, catalog_digest(CONSEQ))


def test_resolve_consequence_template_refuses_tampered_catalog():
    with pytest.raises(CatalogError):
        resolve_consequence_template("effect", CONSEQ, "bad")


def test_resolve_incumbent_returns_the_signed_rate():
    assert resolve_incumbent("ssv2_recognition", INCUMB, catalog_digest(INCUMB)) == 0.70


def test_resolve_incumbent_refuses_unknown_task():
    with pytest.raises(CatalogError):
        resolve_incumbent("unknown_task", INCUMB, catalog_digest(INCUMB))


def test_resolve_incumbent_reads_the_rich_catalog_record():
    # the real incumbent_catalog.json is {task: {value, source, protocol, ...}}, not a bare float
    rich = {"ssv2_recognition_top1": {"value": 0.773, "source": "MVD ViT-H", "protocol": "val top-1"}}
    assert resolve_incumbent("ssv2_recognition_top1", rich, catalog_digest(rich)) == 0.773


def test_resolve_incumbent_rejects_a_record_with_no_value():
    bad = {"t": {"source": "x"}}
    with pytest.raises(CatalogError):
        resolve_incumbent("t", bad, catalog_digest(bad))


def test_incumbent_separated_at_mie():
    assert incumbent_separated(claimed_value=0.80, incumbent_value=0.70, mie=0.05) is True
    assert incumbent_separated(claimed_value=0.72, incumbent_value=0.70, mie=0.05) is False
