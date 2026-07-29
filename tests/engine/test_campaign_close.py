"""The fuller campaign-close wiring on top of the selection correction. Depth-completion asks
whether the lead arc's joint-prediction claim confirmed on its own box (else there is no
foundational arc this campaign, breadth-only). The replication conjunction gate requires BOTH
the primary and a second fresh-box to confirm, with the magnitude read from the unbiased
replication box. Overlap grouping ships independent findings as separate families and merges an
arc into one family under a family-wise count. Reporting is per-lineage, per-family, and
campaign-wide. The final GO/NO-GO stays the human's."""
from pytest import approx

from engine.pool import (
    DELIVERABLE,
    NO_ARC,
    depth_completion,
    group_and_report,
    replication_conjunction,
)


def test_depth_completion_deliverable_when_lead_arc_confirms():
    assert depth_completion(True) == DELIVERABLE


def test_depth_completion_no_foundational_arc_otherwise():
    assert depth_completion(False) == NO_ARC


def test_replication_conjunction_requires_both_boxes_and_magnitude():
    assert replication_conjunction("CONFIRMED", "CONFIRMED", "exceeds_mie") is True


def test_replication_conjunction_fails_when_second_box_does_not_replicate():
    assert replication_conjunction("CONFIRMED", "FAILED", "exceeds_mie") is False


def test_replication_conjunction_fails_when_replication_magnitude_is_not_exceeding():
    assert replication_conjunction("STRONG", "STRONG", "inconclusive") is False


def test_independent_findings_are_separate_families():
    fr = group_and_report([("k1", None), ("k2", None)], alpha=0.05)
    assert len(fr.families) == 2
    assert fr.campaign_expected_false == approx(0.10)
    assert fr.per_lineage_expected_false["k1"] == 0.05


def test_arc_findings_merge_into_one_family():
    fr = group_and_report([("k1", "arcA"), ("k2", "arcA"), ("k3", None)], alpha=0.05)
    assert len(fr.families) == 2  # arcA (2) + independent k3 (1)
    assert fr.per_family_expected_false["arcA"] == approx(0.10)
    assert fr.campaign_expected_false == approx(0.15)
