"""The mechanism gate: the full effect is present (exceeds MIE at power) and a causal
ablation removes it (the ablated effect no longer exceeds MIE), plus a specificity
control. This certifies ablation-and-specificity, not the drafted causal story."""
from __future__ import annotations


def mechanism_check(full_lo: float, ablated_hi: float, mie: float, specificity_ok: bool) -> bool:
    return bool(full_lo > mie and ablated_hi < mie and specificity_ok)
