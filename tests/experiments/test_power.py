"""Box-power sizing. Sizes a holdout box so a true effect of the claim's gap is detected at the
target power, and is run BEFORE the maturation count is fixed so a task is never asked for more
powered boxes than its item pool holds (the TOMATO under-powering the audit surfaced)."""
import pytest

from experiments.power import proportion_sd, required_n


def test_required_n_grows_as_the_gap_shrinks():
    assert required_n(0.03, 0.5) > required_n(0.10, 0.5) > 0


def test_required_n_matches_the_z_based_closed_form():
    # z_0.95 = 1.645, z_0.8 = 0.842 -> n = ((1.645 + 0.842) * 0.5 / 0.05) ** 2 ~ 619
    assert 600 <= required_n(0.05, 0.5, alpha=0.05, power=0.8) <= 640


def test_required_n_rejects_a_non_positive_gap_or_sd():
    with pytest.raises(ValueError):
        required_n(0.0, 0.5)
    with pytest.raises(ValueError):
        required_n(0.05, 0.0)


def test_tomato_capability_box_is_decidable():
    # A capability claim separating a ~0.09 gap above the 0.379 incumbent needs a modest box that
    # 1,484 items can supply many disjoint copies of, unlike an effect claim at the 0.03 MIE.
    n_capability = required_n(0.09, proportion_sd(0.47), alpha=0.05, power=0.8)
    n_effect_at_mie = required_n(0.03, proportion_sd(0.47), alpha=0.05, power=0.8)
    assert n_capability < 300
    assert n_effect_at_mie > 1000  # why TOMATO cannot power an effect claim at the signed MIE


def test_proportion_sd_peaks_near_one_half():
    assert proportion_sd(0.5) >= proportion_sd(0.3) >= proportion_sd(0.05)
