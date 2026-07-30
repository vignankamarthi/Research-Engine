"""The RealSubstrate assembly. It wires the real experiment callables + per-task MIE + the three
digest-verified signed catalogs into one live ExperimentSubstrate, with the heavy cluster deps (the
model scorer, the MCP novelty audit, the G0 pipeline, the held-out and membership checks) INJECTED
so the wiring is testable on the Mac. The substrate operates on the RAW hypothesis dict (the full
proposal, which carries the mechanism and the claimed value), not the referee's normalized Schema
(which drops them). These tests prove the assembly produces a coherent Bundle: the per-task MIE
reaches bundle.mie, the catalogs are digest-verified, and the incumbent separation uses the rich
signed record."""
from datetime import date

import numpy as np
import pytest

from engine.real_substrate import assemble_substrate
from gateconfig import validate_config
from gatelib import library_digest
from referee.catalog import CatalogError, catalog_digest
from referee.lineage import control_catalog_digest

CONS = {"effect": "downstream accuracy rises by >= MIE on the held-out task"}
INC = {"ssv2_recognition_top1": {"value": 0.773, "source": "MVD ViT-H"},
       "kinetics400_recognition_top1": {"value": 0.921, "source": "InternVideo2-6B"}}
# k400 has an incumbent but NO signed MIE, so it exercises the per-task MIE fallback in isolation.
MIE = {"ssv2_recognition_top1": {"mie_value": 0.01}}


def _cfg(**over):
    data = {
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.005, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": control_catalog_digest(), "key_id": "test",
        "consequence_catalog_digest": catalog_digest(CONS),
        "incumbent_catalog_digest": catalog_digest(INC),
        "mie_distribution_digest": catalog_digest(MIE),
    }
    data.update(over)
    return validate_config(data)


def _raw(**over):
    # the RAW hypothesis dict the substrate measures on (carries mechanism + claimed_value)
    raw = {"claim": "freq modeling helps", "claim_type": "effect", "backbone": "iv2",
           "dataset": "ssv2_recognition_top1", "mechanism": "temporal_frequency", "scale": "7b",
           "measure": "top-1 accuracy", "prior_claim": False, "claimed_value": 0.80}
    raw.update(over)
    return raw


def _assemble(cfg, **over):
    kw = dict(
        config=cfg, consequence_catalog=CONS, incumbent_catalog=INC, mie_catalog=MIE,
        score_task=lambda backend, task, ablation: np.full(50, 0.10 if ablation is None else 0.0),
        novelty_audit=lambda schema: (False, ["Paper A", "Paper B"]),
        g0_pipeline=lambda effect, rng: 0.001,  # always significant -> G0 passes
        specificity_check=lambda backend, schema, task: True,
        membership_check=lambda schema: True,
        held_out_check=lambda schema: True,
        backbone_cutoff=date(2023, 1, 1),
        g0_rng=np.random.default_rng(0),
    )
    kw.update(over)
    return assemble_substrate(**kw)


def test_assembled_substrate_produces_a_coherent_bundle():
    b = _assemble(_cfg()).produce(_raw(), backend=None, believed_claim=True)
    assert b.g0_passed is True
    assert b.mech_full_lo > 0.01 and b.mech_ablated_hi < 0.01 and b.specificity_ok is True
    assert b.novelty_collision is False and b.novelty_k_nearest == ["Paper A", "Paper B"]
    assert b.consequence_confirmed is True and b.incumbent_separated is True
    assert b.believed_claim is True


def test_per_task_mie_reaches_the_bundle():
    b = _assemble(_cfg()).produce(_raw(), backend=None, believed_claim=True)
    assert b.mie == 0.01  # the signed SSv2 per-task MIE, not the config floor (0.03)


def test_task_with_no_signed_mie_falls_back_to_the_config_floor():
    # k400 has an incumbent but no MIE entry -> mie falls back to the floor (0.03)
    b = _assemble(_cfg()).produce(
        _raw(dataset="kinetics400_recognition_top1", claimed_value=0.95),
        backend=None, believed_claim=True)
    assert b.mie == 0.03


def test_tampered_mie_digest_raises():
    with pytest.raises(CatalogError):
        _assemble(_cfg(mie_distribution_digest="sha256:wrong")).produce(_raw(), backend=None)


def test_incumbent_separation_uses_the_rich_record_and_mie():
    # claimed 0.80 beats the signed incumbent 0.773 by 0.027 >= MIE 0.01 -> separated
    b = _assemble(_cfg()).produce(_raw(), backend=None, believed_claim=True)
    assert b.incumbent_separated is True
    # a claim only 0.005 above the incumbent does NOT clear the 0.01 MIE
    b2 = _assemble(_cfg()).produce(_raw(claimed_value=0.778), backend=None, believed_claim=True)
    assert b2.incumbent_separated is False
