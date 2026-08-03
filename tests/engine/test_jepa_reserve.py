"""The JEPA standing reserve (PLAN 78 / Core-purpose "JEPA reserve"): across the ~50-idea campaign the
JEPA-tagged maturations stay >= 5 (a FLOOR, a guaranteed slot like the high-patience binding) and <= 10
(a CAP, past which JEPA candidates are deprioritized so the other 40+ ideas stay diverse). These tests
cover the tag, the floor/cap arithmetic, the wave-level reordering, the WaveAgent integration, and the
end-to-end enforcement through the assembled loop."""
from datetime import date

from backend import Box, MockBackend
from engine.agents import Bundle, Maturation
from engine.discovery_roles import MockReviewerAdversary
from engine.generation import (
    JEPA_CAP,
    JEPA_FLOOR,
    JepaReserve,
    RankedCandidate,
    WaveAgent,
    build_generation_context,
    is_jepa_candidate,
    stamp_candidate,
)
from engine.handoff import accept_as_proposed
from engine.health import HealthGate
from engine.orchestrator import run_loop
from engine.steering import DEPTH
from engine.substrate import MockSubstrate
from engine.supervisor import BASE_CASE, Budget, HaltFlag
from gateconfig import validate_config
from gatelib import library_digest
from referee.lease import LeaseStore
from referee.lineage import control_catalog_digest


# --- (a) the JEPA tag ---

def test_is_jepa_candidate_tags_from_mechanism_measure_or_claim():
    assert is_jepa_candidate({"mechanism": "a V-JEPA joint-embedding predictive signal"})
    assert is_jepa_candidate({"claim": "an I-JEPA world model acquires intuitive physics"})
    assert is_jepa_candidate({"measure": "masked-latent prediction accuracy"})
    assert is_jepa_candidate({"mechanism": "predictive coding of future latents"})


def test_is_jepa_candidate_false_for_non_jepa():
    assert not is_jepa_candidate({"mechanism": "temporal_frequency", "claim": "motion helps"})
    assert not is_jepa_candidate({"measure": "spatial appearance fidelity"})
    assert not is_jepa_candidate({})


def test_stamp_candidate_adds_jepa_flag():
    j = stamp_candidate({"mechanism": "v-jepa latent prediction"}, "cross_domain_analogy")
    n = stamp_candidate({"mechanism": "temporal_frequency"}, "limitations")
    assert j["jepa"] is True and j["vein"] == "cross_domain_analogy"
    assert n["jepa"] is False


# --- (b) the floor/cap arithmetic ---

def test_reserve_defaults_are_5_and_10():
    r = JepaReserve()
    assert (r.floor, r.cap) == (5, 10) == (JEPA_FLOOR, JEPA_CAP)


def test_below_floor_and_at_cap_boundaries():
    r = JepaReserve()
    assert r.below_floor(4) and not r.below_floor(5)          # floor is inclusive-satisfied at 5
    assert not r.at_cap(9) and r.at_cap(10) and r.at_cap(11)  # cap blocks from the 10th onward


def _ranked(*specs):
    """Build a ranked list from (claim, jepa, score) tuples, preserving order."""
    return [RankedCandidate({"claim": c, "jepa": j}, s, 0.0, s) for c, j, s in specs]


# --- (b) wave-level reordering ---

def test_below_floor_lifts_a_jepa_candidate_to_the_top():
    # the non-JEPA candidate outranks the JEPA one, but under floor pressure the reserve guarantees
    # the JEPA slot by lifting it to index 0 (the candidate run_campaign matures).
    ranked = _ranked(("nonjepa", False, 0.9), ("jepa", True, 0.4))
    out = JepaReserve().apply(ranked, jepa_matured=0)
    assert out[0].candidate["claim"] == "jepa"


def test_at_cap_deprioritizes_jepa_so_non_jepa_matures():
    # once the cap is reached the 11th JEPA idea must not take the slot; a non-JEPA candidate does.
    ranked = _ranked(("jepa", True, 0.9), ("nonjepa", False, 0.4))
    out = JepaReserve().apply(ranked, jepa_matured=10)
    assert out[0].candidate["claim"] == "nonjepa"


def test_at_cap_keeps_jepa_only_when_nothing_else_exists():
    # never empties a wave: an all-JEPA wave at the cap still yields its candidates (last resort).
    ranked = _ranked(("jepa1", True, 0.9), ("jepa2", True, 0.4))
    out = JepaReserve().apply(ranked, jepa_matured=10)
    assert [r.candidate["claim"] for r in out] == ["jepa1", "jepa2"]


def test_between_floor_and_cap_leaves_ranking_untouched():
    # diversity is preserved: between the floor and the cap the natural quality ranking is unchanged.
    ranked = _ranked(("nonjepa", False, 0.9), ("jepa", True, 0.4))
    out = JepaReserve().apply(ranked, jepa_matured=7)
    assert [r.candidate["claim"] for r in out] == ["nonjepa", "jepa"]


# --- floor + cap over a simulated selection stream ---

def _simulate(reserve, wave_specs, ticks):
    """Run `ticks` waves, maturing the index-0 candidate each tick, counting JEPA maturations."""
    jepa = 0
    selected = []
    for _ in range(ticks):
        ranked = _ranked(*wave_specs)
        out = reserve.apply(ranked, jepa_matured=jepa)
        top = out[0].candidate
        selected.append(bool(top["jepa"]))
        if top["jepa"]:
            jepa += 1
    return jepa, selected


def test_floor_forces_at_least_five_when_jepa_is_under_proposed():
    # a JEPA candidate is present each wave but ALWAYS out-ranked by a non-JEPA one (under-proposed).
    # Without the reserve none would mature; the floor guarantees exactly 5 before diversity resumes.
    r = JepaReserve()
    jepa, selected = _simulate(r, [("nonjepa", False, 0.9), ("jepa", True, 0.4)], ticks=20)
    assert jepa >= 5
    assert selected[:5] == [True] * 5           # the first five slots went to JEPA (the floor)
    assert selected[5] is False                 # then the natural ranking resumes (diversity)


def test_cap_blocks_the_eleventh_jepa_and_preserves_diversity():
    # a JEPA-biased scout (JEPA out-ranks non-JEPA every wave). Without the cap all 20 would be JEPA.
    r = JepaReserve()
    jepa, selected = _simulate(r, [("jepa", True, 0.9), ("nonjepa", False, 0.4)], ticks=20)
    assert jepa == 10                            # capped at 10, the 11th JEPA is blocked
    assert selected[:10] == [True] * 10
    assert all(s is False for s in selected[10:])  # every slot after the cap is non-JEPA (diverse)


# --- (b) WaveAgent integration: the reserve reorders what run_campaign matures ---

class _DualScout:
    """A scout whose wave carries one non-JEPA candidate (higher natural rank) and one JEPA candidate,
    each grounded and well-formed, so the reserve decides which one reaches index 0."""

    def propose(self, context):
        return [
            {"claim": "spatial appearance idea", "claim_type": "effect", "backbone": "iv2",
             "dataset": "ssv2", "scale": "7b", "measure": "accuracy",
             "mechanism": "spatial texture", "prior_claim": False, "grounding": "arXiv:2401.12345"},
            {"claim": "a v-jepa predictive idea", "claim_type": "effect", "backbone": "iv2",
             "dataset": "ssv2", "scale": "7b", "measure": "accuracy",
             "mechanism": "v-jepa latent prediction", "prior_claim": False,
             "grounding": "arXiv:2401.12346"},
        ]

    def mature(self, schema_raw):
        return Maturation(matured=True, bundle=Bundle(believed_claim=True))

    def frame(self, schema_raw, verdict):
        return "narrative"


def test_wave_agent_below_floor_surfaces_jepa_first():
    agent = WaveAgent(_DualScout(), resolver=lambda i: True, wired_claim_types={"effect"},
                      reserve=JepaReserve())
    ctx = build_generation_context(vein="cross_domain_analogy", mode=DEPTH, extra={"jepa_matured": 0})
    candidates = agent.propose(ctx)
    assert candidates[0]["jepa"] is True                       # floor pressure lifted JEPA to the top
    assert agent.last_selected["jepa"] is True                 # exposed for the orchestrator's count


def test_wave_agent_at_cap_surfaces_non_jepa_first():
    agent = WaveAgent(_DualScout(), resolver=lambda i: True, wired_claim_types={"effect"},
                      reserve=JepaReserve())
    ctx = build_generation_context(vein="cross_domain_analogy", mode=DEPTH, extra={"jepa_matured": 10})
    candidates = agent.propose(ctx)
    assert candidates[0]["jepa"] is False                      # cap deprioritized JEPA
    assert agent.last_selected["jepa"] is False


def test_wave_agent_without_reserve_is_unchanged():
    agent = WaveAgent(_DualScout(), resolver=lambda i: True, wired_claim_types={"effect"})
    ctx = build_generation_context(vein="cross_domain_analogy", mode=DEPTH, extra={"jepa_matured": 0})
    candidates = agent.propose(ctx)
    assert candidates[0]["claim"] == "spatial appearance idea"   # natural ranking, no JEPA steering


# --- end-to-end enforcement through the assembled loop ---

def _cfg():
    return validate_config({
        "version": "1", "alpha": 0.05, "power": 0.8, "mde": 0.01, "mie_floor": 0.03,
        "claim_types": ["effect"], "gate_library_digest": library_digest(),
        "control_catalog_hash": control_catalog_digest(), "key_id": "test",
    })


class _JepaBiasedScout:
    """A WaveAgent scout that proposes a JEPA idea (higher rank) AND a non-JEPA idea each tick, with a
    UNIQUE claim per tick so successive maturations hash to distinct lineages (and distinct boxes)."""

    def __init__(self):
        self._n = 0

    def propose(self, context):
        self._n += 1
        return [
            {"claim": f"v-jepa predictive idea {self._n}", "claim_type": "effect", "backbone": "iv2",
             "dataset": "ssv2", "scale": "7b", "measure": "accuracy",
             "mechanism": "v-jepa latent prediction", "prior_claim": False,
             "grounding": "arXiv:2401.10000"},
            {"claim": f"spatial appearance idea {self._n}", "claim_type": "effect", "backbone": "iv2",
             "dataset": "ssv2", "scale": "7b", "measure": "accuracy",
             "mechanism": "spatial texture", "prior_claim": False, "grounding": "arXiv:2401.20000"},
        ]

    def mature(self, schema_raw):
        return Maturation(matured=True, bundle=Bundle(believed_claim=True))

    def frame(self, schema_raw, verdict):
        return "narrative"


def _box_factory(box_id):
    return Box(id=box_id, n=800, origin_date=date(2024, 6, 1))


def test_loop_caps_jepa_maturations_and_still_scores_non_jepa(tmp_path):
    # a small reserve (floor 1, cap 2) drives the whole assembled loop: the JEPA-biased scout would
    # mature JEPA every tick, but the reserve caps JEPA at 2 and the remaining slots go to non-JEPA,
    # so the campaign's JEPA count is bounded AND the other ideas still get scored (diversity holds).
    agent = WaveAgent(_JepaBiasedScout(), resolver=lambda i: True, wired_claim_types={"effect"},
                      reserve=JepaReserve(floor=1, cap=2))
    ls = LeaseStore(str(tmp_path / "loop.db"))
    ls.add_boxes([f"b{i}" for i in range(12)])
    reason, report, campaign = run_loop(
        agent=agent, backend=MockBackend(0.25, 0.0, 0.1, seed=1), config=_cfg(),
        lease_store=ls, box_factory=_box_factory, substrate=MockSubstrate(),
        triage=accept_as_proposed, reviewer=MockReviewerAdversary(),
        budget=Budget.default(max_gpu_hours=100, max_boxes=10, max_maturations=6),
        halt_flag=HaltFlag(str(tmp_path / "halt")), health_gate=HealthGate([]), seed=0)

    assert reason == BASE_CASE
    assert campaign.jepa_matured == 2                          # capped, the 3rd JEPA idea was blocked
    scored = [r for r in campaign.results if r.verdict is not None]
    non_jepa_scored = [r for r in scored if r.schema is not None
                       and "spatial appearance" in r.schema.claim]
    assert non_jepa_scored                                     # non-JEPA ideas still matured (diverse)


def test_loop_floor_matures_jepa_when_scout_under_proposes(tmp_path):
    # the mirror case: a scout that ranks non-JEPA first every tick would never mature a JEPA idea, but
    # the floor guarantees the reserved slots, so the campaign reaches its JEPA floor.
    class _NonJepaFirst(_JepaBiasedScout):
        def propose(self, context):
            self._n += 1
            return [
                {"claim": f"spatial appearance idea {self._n}", "claim_type": "effect",
                 "backbone": "iv2", "dataset": "ssv2", "scale": "7b", "measure": "accuracy",
                 "mechanism": "spatial texture", "prior_claim": False, "grounding": "arXiv:2401.20000"},
                {"claim": f"v-jepa predictive idea {self._n}", "claim_type": "effect",
                 "backbone": "iv2", "dataset": "ssv2", "scale": "7b", "measure": "accuracy",
                 "mechanism": "v-jepa latent prediction", "prior_claim": False,
                 "grounding": "arXiv:2401.10000"},
            ]

    agent = WaveAgent(_NonJepaFirst(), resolver=lambda i: True, wired_claim_types={"effect"},
                      reserve=JepaReserve(floor=2, cap=5))
    ls = LeaseStore(str(tmp_path / "loop.db"))
    ls.add_boxes([f"b{i}" for i in range(12)])
    _, _, campaign = run_loop(
        agent=agent, backend=MockBackend(0.25, 0.0, 0.1, seed=1), config=_cfg(),
        lease_store=ls, box_factory=_box_factory, substrate=MockSubstrate(),
        triage=accept_as_proposed, reviewer=MockReviewerAdversary(),
        budget=Budget.default(max_gpu_hours=100, max_boxes=10, max_maturations=6),
        halt_flag=HaltFlag(str(tmp_path / "halt")), health_gate=HealthGate([]), seed=0)

    assert campaign.jepa_matured >= 2                          # the floor was met despite under-proposal
