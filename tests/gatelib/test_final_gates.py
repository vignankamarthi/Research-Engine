"""The remaining confirmatory gates: mechanism (ablation + specificity), novelty
(fail-closed, positive-delta), and the importance-consequence gate (templated
consequence confirmed AND incumbent separated at MIE)."""
from gatelib import consequence_check, mechanism_check, novelty_check


# --- mechanism: the paired (full minus ablated) CONTRAST exceeds the MIE, plus specificity ---
def test_mechanism_supported():
    # ablating the mechanism drops the metric by more than the MIE, and specificity holds
    assert mechanism_check(contrast_lo=0.06, mie=0.03, specificity_ok=True)


def test_mechanism_fails_when_the_ablation_drop_is_below_the_mie():
    assert not mechanism_check(contrast_lo=0.02, mie=0.03, specificity_ok=True)


def test_mechanism_fails_on_a_non_positive_contrast():
    # ablation did not reduce the metric (an MCQ chance floor no longer makes this unsatisfiable)
    assert not mechanism_check(contrast_lo=0.0, mie=0.03, specificity_ok=True)


def test_mechanism_fails_without_specificity():
    assert not mechanism_check(contrast_lo=0.06, mie=0.03, specificity_ok=False)


# --- novelty: fail-closed on a collision, positive-delta over named priors ---
def test_novelty_collision_is_rejected():
    assert not novelty_check(collision_found=True, k_nearest=["a", "b"], advance_argued=True).passed


def test_novelty_passes_with_named_priors_and_an_argued_advance():
    r = novelty_check(collision_found=False, k_nearest=["a", "b", "c"], advance_argued=True)
    assert r.passed


def test_novelty_needs_named_nearest_priors():
    assert not novelty_check(collision_found=False, k_nearest=[], advance_argued=True).passed


def test_novelty_needs_an_argued_advance():
    assert not novelty_check(collision_found=False, k_nearest=["a"], advance_argued=False).passed


# --- importance-consequence: discharged only if confirmed AND incumbent-separated ---
def test_consequence_discharged():
    assert consequence_check(consequence_confirmed=True, incumbent_separated_at_mie=True)


def test_consequence_not_confirmed():
    assert not consequence_check(consequence_confirmed=False, incumbent_separated_at_mie=True)


def test_consequence_incumbent_not_separated():
    assert not consequence_check(consequence_confirmed=True, incumbent_separated_at_mie=False)
