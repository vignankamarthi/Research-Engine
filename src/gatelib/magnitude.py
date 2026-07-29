"""The claim-type magnitude gate. A confidence interval is classified against the
signed MIE (interest floor). The CI must strictly EXCLUDE the MIE to be a positive
(exceeds) or a powered null; a CI that includes the MIE is INCONCLUSIVE, never a
proven null, because under-detection is the dominant residual risk."""
from __future__ import annotations

import numpy as np
from scipy.stats import t as _t

from .verdicts import EXCEEDS_MIE, FAIL, INCONCLUSIVE, PASS, POWERED_NULL


def classify_magnitude(ci_lo: float, ci_hi: float, mie: float) -> str:
    _require_ci(ci_lo, ci_hi)
    if not (mie > 0.0):
        raise ValueError(f"mie must be a positive interest floor, got {mie}")
    if ci_lo > mie:
        return EXCEEDS_MIE
    if ci_hi < mie:
        return POWERED_NULL
    return INCONCLUSIVE


def _require_ci(ci_lo: float, ci_hi: float) -> None:
    if ci_lo > ci_hi:
        raise ValueError(f"inverted CI: lo {ci_lo} > hi {ci_hi}")


def phenomenon_gate(ci_lo: float, ci_hi: float, baseline_rate: float) -> str:
    """QUALITATIVE-PHENOMENON: the observed-rate CI must sit STRICTLY above a signed
    null/baseline rate (separated at the CI's confidence level)."""
    _require_ci(ci_lo, ci_hi)
    return PASS if ci_lo > baseline_rate else FAIL


def capability_gate(ci_lo: float, ci_hi: float, incumbent_rate: float) -> str:
    """CAPABILITY: the observed success-rate CI must separate above the pre-registered
    incumbent's held-out success rate (the strongest provenance-verified prior result)."""
    _require_ci(ci_lo, ci_hi)
    return PASS if ci_lo > incumbent_rate else FAIL


def law_shape_gate(predicted, observed, tol: float) -> str:
    """LAW-SHAPE: a functional-form prediction across held-out scales holds when the worst
    per-scale residual stays within tolerance."""
    p = np.asarray(predicted, dtype=float)
    o = np.asarray(observed, dtype=float)
    if p.shape != o.shape or p.size == 0:
        raise ValueError("predicted and observed must be non-empty and the same shape")
    return PASS if float(np.max(np.abs(p - o))) <= tol else FAIL


def _deferred_gate(kw) -> str:
    raise NotImplementedError(
        "multi-benchmark superiority is deferred and fenced out of the campaign-one coverage invariant"
    )


# The claim-type magnitude registry. Adding a claim-type is one entry here, not an edit to a
# dispatch chain; the schema-normal-form's magnitude_gate name keys straight into it.
_MAGNITUDE_GATES = {
    "mie_at_power": lambda kw: classify_magnitude(kw["ci_lo"], kw["ci_hi"], kw["mie"]),
    "phenomenon_vs_null": lambda kw: phenomenon_gate(kw["ci_lo"], kw["ci_hi"], kw["baseline_rate"]),
    "capability_separation": lambda kw: capability_gate(kw["ci_lo"], kw["ci_hi"], kw["incumbent_rate"]),
    "law_shape_fit": lambda kw: law_shape_gate(kw["predicted"], kw["observed"], kw["tol"]),
    "sota_margin": _deferred_gate,  # registered stub, not a special case
}


def magnitude_gate_for(gate_name: str, **kw) -> str:
    """Route a claim to its magnitude gate by the name the schema-normal-form derived."""
    try:
        gate = _MAGNITUDE_GATES[gate_name]
    except KeyError:
        raise ValueError(f"unknown magnitude gate: {gate_name!r}") from None
    return gate(kw)


def magnitude_pvalue(scores, mie: float) -> float:
    """One-sided p-value for H0: effect <= MIE vs H1: effect > MIE. Feeds the
    campaign-close selection correction (BH over the matured-and-scored family)."""
    x = np.asarray(scores, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need at least two observations for a p-value")
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        return 0.0 if x.mean() > mie else 1.0
    tstat = (float(x.mean()) - mie) / se
    return float(_t.sf(tstat, n - 1))
