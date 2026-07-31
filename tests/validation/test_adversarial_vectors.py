"""The two Milestone-4 adversarial vectors PLAN step 23 had left REMAINING: LEAKED LABELS
(the answer leaks into the scored input) and IN-PROCESS REFEREE SUBVERSION (a mutation of
the referee's gates/config/derivation while it runs). Both are shown caught by the machinery
that already exists, so these tests DOCUMENT the guard rather than adding one.

LEAKED LABELS splits into the two ways an answer can reach the input:
  - model-AGNOSTIC leak (statistics of the input carry the label): the untrained-weights FLOOR
    reads the same leak, so the trained-minus-untrained residual collapses and no false
    positive is admitted. This is the mandatory RoPE-catcher doing its job (ANTIPATTERNS 4/5).
  - backbone-SPECIFIC leak (only the trained backbone decodes it): the effect clears the floor
    and the magnitude bar, but it SURVIVES the mechanism ablation (the drop is not attributable
    to the claimed mechanism), so the specificity/confound gate fails it (spec 3c).

IN-PROCESS REFEREE SUBVERSION is the stronger cousin of the tampered-config case: the config
hash is offline-signed and cannot be re-signed in-process (the key lives only on Vignan's Mac),
so mutating the live derivation constants (the mandatory controls, the magnitude-gate mapping)
or the gate-library surface makes the referee's recomputed digest disagree with the signed
config, and confirm() HALTs at Gate 0 before a box is ever scored.
"""
import pytest

import gatelib
import referee.lineage as lineage
from backend import MockBackend
from engine.agents import Bundle
from gatelib import GateLibraryError, floor_separation
from referee.lineage import ControlCatalogError
from referee.runner import confirm

_POSITIVE_VERDICTS = {"CONFIRMED", "STRONG", "CONFIRMED_EFFECT"}


# ---------------------------------------------------------------------------
# Vector 1: LEAKED LABELS
# ---------------------------------------------------------------------------

def test_model_agnostic_label_leak_collapses_the_floor(
        effect_schema, effect_cfg, post_cutoff_box, passing_bundle):
    """The label leaks into the input in a way the untrained arm reads too, so trained and
    untrained score the same. The mandatory FLOOR is the catcher: the residual is a powered
    null, so the manufactured positive is refused (never a CONFIRMED positive)."""
    be = MockBackend(trained_effect=0.30, untrained_effect=0.30, noise=0.1, seed=3)

    # unit: the trained-minus-untrained residual does not clear the MIE.
    trained = be.score_box(post_cutoff_box)
    untrained = [be.score_box(post_cutoff_box, untrained_init=k) for k in range(4)]
    assert not floor_separation(trained, untrained, mie=0.03).passed

    # end to end: no false positive survives the gauntlet.
    v = confirm(be, post_cutoff_box, effect_schema, effect_cfg, passing_bundle)
    assert v.status not in _POSITIVE_VERDICTS


def test_backbone_specific_label_leak_fails_the_specificity_gate(
        effect_schema, effect_cfg, post_cutoff_box):
    """The leak is decodable only by the trained backbone, so it clears the floor and the
    magnitude bar. It is caught downstream because it SURVIVES the mechanism ablation: the
    full-minus-ablated contrast is ~0, so the effect is not attributable to the claimed
    mechanism and the specificity/confound gate fails it."""
    be = MockBackend(trained_effect=0.30, untrained_effect=0.0, noise=0.1, seed=3)
    leak_bundle = Bundle.passing(mech_contrast_lo=0.0)  # effect does not drop when ablated
    v = confirm(be, post_cutoff_box, effect_schema, effect_cfg, leak_bundle)
    assert v.status == "FAILED" and v.reason == "mechanism_unsupported"


# ---------------------------------------------------------------------------
# Vector 2: IN-PROCESS REFEREE SUBVERSION
# ---------------------------------------------------------------------------

def test_in_process_drop_of_a_mandatory_control_halts(
        monkeypatch, effect_schema, effect_cfg, post_cutoff_box, passing_bundle):
    """An attacker weakens the derivation in memory by dropping the control-of-the-control from
    the mandatory set. The config's control-catalog hash was signed over the ORIGINAL set and
    cannot be re-signed in-process, so the recomputed digest no longer matches and the referee
    HALTs before scoring."""
    monkeypatch.setattr(lineage, "_MANDATORY_CONTROLS", ("arch_control",))
    with pytest.raises(ControlCatalogError):
        confirm(MockBackend(0.25, 0.0, 0.1, seed=1), post_cutoff_box,
                effect_schema, effect_cfg, passing_bundle)


def test_in_process_remap_of_the_magnitude_gate_halts(
        monkeypatch, effect_schema, effect_cfg, post_cutoff_box, passing_bundle):
    """An attacker remaps the EFFECT claim-type to a weaker magnitude gate in memory to draw a
    softer gauntlet. The magnitude-gate mapping is inside the signed control catalog, so the
    recomputed digest diverges from the signed config and the referee HALTs."""
    weakened = {**lineage._MAGNITUDE_GATE_BY_TYPE, "effect": "phenomenon_vs_null"}
    monkeypatch.setattr(lineage, "_MAGNITUDE_GATE_BY_TYPE", weakened)
    with pytest.raises(ControlCatalogError):
        confirm(MockBackend(0.25, 0.0, 0.1, seed=1), post_cutoff_box,
                effect_schema, effect_cfg, passing_bundle)


def test_in_process_mutation_of_the_gate_library_surface_halts(
        monkeypatch, effect_schema, effect_cfg, post_cutoff_box, passing_bundle):
    """An attacker mutates the gate-library public surface in memory (drops a gate symbol). The
    config pins the library digest, so the recomputed digest diverges and the referee HALTs at
    the gate-library check, again before any box is scored."""
    reduced = [name for name in gatelib.__all__ if name != "floor_separation"]
    monkeypatch.setattr(gatelib, "__all__", reduced)
    with pytest.raises(GateLibraryError):
        confirm(MockBackend(0.25, 0.0, 0.1, seed=1), post_cutoff_box,
                effect_schema, effect_cfg, passing_bundle)
