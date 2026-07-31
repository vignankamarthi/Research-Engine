"""The substrate producer, the piece that makes a real run eligible. The referee gates on a
Bundle of measured gate inputs. The substrate is what RUNS the experiments and produces that
Bundle: the G0 detectability probe, the mechanism ablation, the novelty audit, the backbone
check, the consequence experiment. This closes the trust boundary: the agent PROPOSES and
matures a hypothesis and contributes its one legitimate opinion (believed_claim), the substrate
MEASURES the evidence, and the referee JUDGES the measurements. No party grades its own work.

Each experiment is an injected callable so the substrate is testable on the Mac with mocks; the
real G0/mechanism/consequence experiments (which score through the backend, HFBackend on the
cluster) and the real novelty audit (research MCPs) swap in without touching this assembly."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .agents import Bundle


@runtime_checkable
class Substrate(Protocol):
    def produce(self, schema, backend, believed_claim: bool = False) -> Bundle: ...


class MockSubstrate:
    """Deterministic stand-in: represents every gate experiment having run and passed. The Mac
    pipeline uses this so the plumbing is exercised without a real model or MCP calls."""

    def produce(self, schema, backend, believed_claim: bool = False) -> Bundle:
        return Bundle.passing(believed_claim=believed_claim)


class ExperimentSubstrate:
    """Assembles a Bundle from real (or injected) gate experiments. Every field comes from a
    measurement, never from the agent. A failed experiment produces the failing value, so the
    referee reads it as such rather than the substrate papering over it."""

    def __init__(self, g0_fn, mechanism_fn, novelty_fn, backbone_fn, consequence_fn, mie_fn=None,
                 magnitude_fn=None):
        self._g0_fn = g0_fn                  # (backend, schema) -> bool
        self._mechanism_fn = mechanism_fn    # (backend, schema) -> (contrast_lo, specificity_ok)
        self._novelty_fn = novelty_fn        # (schema) -> (collision, k_nearest, advance)
        self._backbone_fn = backbone_fn      # (schema) -> (cutoff_date, membership_clean)
        self._consequence_fn = consequence_fn  # (schema) -> (consequence_confirmed, incumbent_separated)
        self._mie_fn = mie_fn                # (schema) -> per-task MIE float, or None to use the floor
        self._magnitude_fn = magnitude_fn    # (schema) -> the per-type magnitude thresholds dict

    def produce(self, schema, backend, believed_claim: bool = False) -> Bundle:
        g0 = self._g0_fn(backend, schema)
        mech_contrast_lo, spec = self._mechanism_fn(backend, schema)
        collision, k_nearest, advance = self._novelty_fn(schema)
        cutoff, clean = self._backbone_fn(schema)
        consequence_ok, incumbent_sep = self._consequence_fn(schema)
        mie = self._mie_fn(schema) if self._mie_fn is not None else None
        mag = self._magnitude_fn(schema) if self._magnitude_fn is not None else {}
        return Bundle(
            mie=mie,
            g0_passed=g0,
            backbone_cutoff=cutoff,
            membership_clean=clean,
            mech_contrast_lo=mech_contrast_lo,
            specificity_ok=spec,
            consequence_confirmed=consequence_ok,
            incumbent_separated=incumbent_sep,
            incumbent_rate=mag.get("incumbent_rate"),
            baseline_rate=mag.get("baseline_rate"),
            law_predicted=mag.get("law_predicted"),
            law_observed=mag.get("law_observed"),
            law_tol=mag.get("law_tol"),
            law_within_envelope=bool(mag.get("law_within_envelope", False)),
            novelty_collision=collision,
            novelty_k_nearest=k_nearest,
            novelty_advance=advance,
            believed_claim=believed_claim,
        )
