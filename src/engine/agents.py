"""The discovery-tier agent interface and a deterministic MockAgent. The real
generative agents (framing, adversaries, scouts, via LLM calls) implement the same
`Agent` protocol and swap in at runtime; the MockAgent lets the whole pipeline run
fast and free so the plumbing is CI-green first. The maturation carries the gate
inputs the confirmatory runner needs (G0, mechanism, consequence, novelty, backbone)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Bundle:
    """The gate inputs a matured hypothesis carries into confirmation. Defaults are FAIL-CLOSED:
    an unpopulated gate input cannot pass its gate, so a caller that forgets a field is caught by
    the referee instead of silently sailing through. Opt into a satisfied bundle explicitly with
    Bundle.passing(**overrides), which the substrate (or a test) does deliberately."""
    g0_passed: bool = False
    backbone_cutoff: date = date.max  # fail-closed: any real box origin then reads as contaminated
    membership_clean: Optional[bool] = None
    mech_full_lo: float = 0.0
    mech_ablated_hi: float = 1.0
    specificity_ok: bool = False
    consequence_confirmed: bool = False
    incumbent_separated: bool = False
    novelty_collision: bool = False
    novelty_k_nearest: list = field(default_factory=list)
    novelty_advance: bool = False
    ood_holds: bool = False
    believed_claim: bool = False
    # The PER-TASK MIE the substrate resolved from the signed mie_distribution catalog. None means
    # the substrate did not resolve one, so the runner falls back to the config's signed mie_floor.
    mie: float | None = None

    @classmethod
    def passing(cls, **overrides) -> "Bundle":
        """A bundle with every gate input at its SATISFIED value. Callers opt INTO passing
        explicitly; a bare Bundle() fails closed."""
        satisfied = dict(
            g0_passed=True, backbone_cutoff=date(2023, 1, 1), membership_clean=True,
            mech_full_lo=0.06, mech_ablated_hi=0.01, specificity_ok=True,
            consequence_confirmed=True, incumbent_separated=True,
            novelty_collision=False, novelty_k_nearest=["a", "b"], novelty_advance=True,
            ood_holds=False, believed_claim=True,
        )
        satisfied.update(overrides)
        return cls(**satisfied)


@dataclass
class Maturation:
    matured: bool
    bundle: Bundle


@runtime_checkable
class Agent(Protocol):
    def propose(self, context: dict) -> list[dict]: ...
    def mature(self, schema_raw: dict) -> Maturation: ...
    def frame(self, schema_raw: dict, verdict) -> str: ...


class MockAgent:
    """A deterministic stand-in for the generative agents."""

    def propose(self, context: dict) -> list[dict]:
        return [{
            "claim": "freq-domain temporal modeling improves recognition",
            "claim_type": "effect", "backbone": "iv2", "dataset": "ssv2",
            "mechanism": "temporal_frequency",  # what the ablation removes; read from the claim
            "scale": "7b", "measure": "top-1 accuracy", "prior_claim": False,
        }]

    def mature(self, schema_raw: dict) -> Maturation:
        # The mock represents a hypothesis whose substrate gates are all satisfied, so it opts
        # into a passing bundle explicitly (a real agent leaves substrate-owned gates fail-closed).
        return Maturation(matured=True, bundle=Bundle.passing())

    def frame(self, schema_raw: dict, verdict) -> str:
        status = verdict.status if verdict is not None else "pending"
        return f"Thesis: {schema_raw['claim']}. Confirmatory verdict: {status}."
