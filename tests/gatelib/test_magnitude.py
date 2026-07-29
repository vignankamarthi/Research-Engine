"""The claim-type magnitude gate, on a confidence interval versus the signed MIE.
Exceeds-MIE feeds a positive, CI-excludes-MIE-from-above is a powered null
(CONFIRMED NEGATIVE), a CI that includes the MIE is INCONCLUSIVE (underpowered),
never a null. MDE (detectability) is a separate, lower floor and is not this gate."""
import pytest

from gatelib import classify_magnitude


def test_effect_clearly_above_mie():
    assert classify_magnitude(ci_lo=0.05, ci_hi=0.09, mie=0.03) == "exceeds_mie"


def test_powered_null_ci_excludes_mie_from_above():
    assert classify_magnitude(ci_lo=0.001, ci_hi=0.02, mie=0.03) == "powered_null"


def test_ci_straddling_mie_is_inconclusive():
    assert classify_magnitude(ci_lo=0.01, ci_hi=0.05, mie=0.03) == "inconclusive"


def test_boundary_touching_mie_is_inconclusive_not_exceeds():
    # strict: the CI must EXCLUDE the MIE to be a positive or a powered null
    assert classify_magnitude(ci_lo=0.03, ci_hi=0.09, mie=0.03) == "inconclusive"
    assert classify_magnitude(ci_lo=0.001, ci_hi=0.03, mie=0.03) == "inconclusive"


def test_inverted_ci_rejected():
    with pytest.raises(ValueError):
        classify_magnitude(ci_lo=0.09, ci_hi=0.01, mie=0.03)


def test_nonpositive_mie_rejected():
    with pytest.raises(ValueError):
        classify_magnitude(ci_lo=0.01, ci_hi=0.09, mie=0.0)
