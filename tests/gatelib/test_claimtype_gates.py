"""Per-claim-type magnitude gates, each its own disposal stage. EFFECT is MIE-at-power
(the CI-vs-MIE classifier). PHENOMENON separates over a null/baseline rate. CAPABILITY
separates over a pre-registered incumbent's held-out success rate. LAW-SHAPE checks a
functional-form prediction across held-out scales. Multi-benchmark superiority is DEFERRED
and must stay fenced out of the campaign-one coverage invariant."""
import pytest

from gatelib.magnitude import (
    FAIL,
    PASS,
    capability_gate,
    law_shape_gate,
    magnitude_gate_for,
    phenomenon_gate,
)


def test_phenomenon_passes_when_ci_strictly_above_baseline():
    assert phenomenon_gate(0.30, 0.40, baseline_rate=0.20) == PASS


def test_phenomenon_fails_when_ci_reaches_baseline():
    assert phenomenon_gate(0.18, 0.40, baseline_rate=0.20) == FAIL


def test_capability_passes_when_above_incumbent():
    assert capability_gate(0.75, 0.85, incumbent_rate=0.70) == PASS


def test_capability_fails_when_not_separated_from_incumbent():
    assert capability_gate(0.65, 0.85, incumbent_rate=0.70) == FAIL


def test_law_shape_passes_within_tolerance():
    assert law_shape_gate([1.0, 2.0, 3.0], [1.02, 1.98, 3.03], tol=0.1) == PASS


def test_law_shape_fails_outside_tolerance():
    assert law_shape_gate([1.0, 2.0, 3.0], [1.5, 2.0, 3.0], tol=0.1) == FAIL


def test_dispatch_routes_effect():
    assert magnitude_gate_for("mie_at_power", ci_lo=0.05, ci_hi=0.09, mie=0.03) == "exceeds_mie"


def test_dispatch_routes_phenomenon():
    assert magnitude_gate_for("phenomenon_vs_null", ci_lo=0.3, ci_hi=0.4, baseline_rate=0.2) == PASS


def test_dispatch_routes_capability():
    assert magnitude_gate_for("capability_separation", ci_lo=0.75, ci_hi=0.85, incumbent_rate=0.7) == PASS


def test_dispatch_routes_law_shape():
    assert magnitude_gate_for("law_shape_fit", predicted=[1, 2], observed=[1.01, 2.01], tol=0.1) == PASS


def test_dispatch_multi_benchmark_is_deferred():
    with pytest.raises(NotImplementedError):
        magnitude_gate_for("sota_margin")


def test_dispatch_unknown_gate_raises():
    with pytest.raises(ValueError):
        magnitude_gate_for("bogus")
