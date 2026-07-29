"""The end-to-end campaign finalize: one call that threads depth-completion, the selection
correction, the second-box replication conjunction, and overlap/family-wise reporting into a
single human-facing dossier. The GO/NO-GO decision itself stays the human's; finalize only
assembles what the human decides on."""
from types import SimpleNamespace

from engine.pool import DELIVERABLE, NO_ARC, finalize_campaign


def _result(lineage, status, pvalue):
    verdict = SimpleNamespace(status=status, pvalue=pvalue)
    return SimpleNamespace(verdict=verdict, lineage=lineage, narrative="n", schema=None)


def test_deliverable_with_a_replicated_finding():
    results = [_result("k1", "CONFIRMED", 0.001)]
    reps = {"k1": ("CONFIRMED", "exceeds_mie")}
    close = finalize_campaign(results, reps, {"k1": None}, lead_arc_confirmed=True)
    assert close.depth_status == DELIVERABLE
    assert len(close.submitted) == 1
    assert "GO/NO-GO is Vignan's" in close.dossier


def test_finding_that_fails_replication_is_dropped():
    results = [_result("k1", "CONFIRMED", 0.001)]
    reps = {"k1": ("FAILED", "exceeds_mie")}  # the second fresh box did not replicate
    close = finalize_campaign(results, reps, {"k1": None}, lead_arc_confirmed=False)
    assert len(close.submitted) == 0


def test_no_foundational_arc_labels_breadth_only():
    results = [_result("k1", "CONFIRMED", 0.001)]
    reps = {"k1": ("CONFIRMED", "exceeds_mie")}
    close = finalize_campaign(results, reps, {"k1": None}, lead_arc_confirmed=False)
    assert close.depth_status == NO_ARC
    assert "breadth-only" in close.dossier


def test_arc_findings_reported_as_one_family():
    results = [_result("k1", "CONFIRMED", 0.001), _result("k2", "STRONG", 0.001)]
    reps = {"k1": ("CONFIRMED", "exceeds_mie"), "k2": ("STRONG", "exceeds_mie")}
    close = finalize_campaign(results, reps, {"k1": "arcA", "k2": "arcA"}, lead_arc_confirmed=True)
    assert len(close.submitted) == 2
    assert len(close.family_report.families) == 1  # both under arcA
