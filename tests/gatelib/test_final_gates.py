"""The remaining confirmatory gates: mechanism (ablation + specificity), novelty
(fail-closed, positive-delta), and the importance-consequence gate (templated
consequence confirmed AND incumbent separated at MIE)."""
from gatelib import consequence_check, mechanism_check, novelty_check


# --- mechanism: the full effect is present and the ablation removes it ---
def test_mechanism_supported():
    assert mechanism_check(full_lo=0.06, ablated_hi=0.02, mie=0.03, specificity_ok=True)


def test_mechanism_fails_if_ablation_does_not_remove_the_effect():
    assert not mechanism_check(full_lo=0.06, ablated_hi=0.05, mie=0.03, specificity_ok=True)


def test_mechanism_fails_if_full_effect_absent():
    assert not mechanism_check(full_lo=0.01, ablated_hi=0.005, mie=0.03, specificity_ok=True)


def test_mechanism_fails_without_specificity():
    assert not mechanism_check(full_lo=0.06, ablated_hi=0.02, mie=0.03, specificity_ok=False)


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
