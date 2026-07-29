"""The backbone contamination gate, HARD on every positive. The box's data ORIGIN
date (earliest public availability) must post-date the backbone cutoff, AND a
membership-verified clean split. Where membership is unverifiable (the usual FM
case) the finding passes but is labeled 'origin-date-verified only', never as
fully backbone-clean."""
from datetime import date

import pytest

from gatelib import backbone_check


def test_origin_before_cutoff_is_contaminated():
    r = backbone_check(box_origin=date(2022, 1, 1), backbone_cutoff=date(2023, 1, 1),
                       membership_clean=None)
    assert not r.passed and r.label == "contaminated"


def test_post_cutoff_and_membership_clean_is_clean():
    r = backbone_check(box_origin=date(2024, 1, 1), backbone_cutoff=date(2023, 1, 1),
                       membership_clean=True)
    assert r.passed and r.label == "clean"


def test_post_cutoff_membership_unverifiable_is_origin_only():
    r = backbone_check(box_origin=date(2024, 1, 1), backbone_cutoff=date(2023, 1, 1),
                       membership_clean=None)
    assert r.passed and r.label == "origin_date_verified_only"


def test_post_cutoff_but_membership_flagged_dirty_is_contaminated():
    r = backbone_check(box_origin=date(2024, 1, 1), backbone_cutoff=date(2023, 1, 1),
                       membership_clean=False)
    assert not r.passed and r.label == "contaminated"


def test_origin_within_margin_is_contaminated():
    # a stricter origin margin: origin must clear cutoff by margin_days
    r = backbone_check(box_origin=date(2023, 1, 15), backbone_cutoff=date(2023, 1, 1),
                       membership_clean=None, margin_days=90)
    assert not r.passed and r.label == "contaminated"


def test_missing_origin_is_ineligible():
    with pytest.raises(ValueError):
        backbone_check(box_origin=None, backbone_cutoff=date(2023, 1, 1), membership_clean=None)
