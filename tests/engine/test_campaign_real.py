"""The generalized campaign driver's CONSTRUCTION SEAM (PLAN 73), Mac-testable with a fake scout, no
cluster and no `claude -p`. Proves the driver builds the GROUNDED, vein-diverse WaveAgent (with the JEPA
reserve and the significance-adversary's advisory penalty) and the UNATTENDED classifier triage, and
that it stamps only the signed TASK key onto a candidate, NEVER re-pinning a mechanism (the drift fix).
The RPC/service/substrate/run wiring in `main()` is cluster-gated and exercised only by a cluster smoke
run; it is not imported here."""
import sys
from datetime import date
from pathlib import Path

# the driver lives under cluster/; add it to the path (its module-level imports are Mac-safe: numpy +
# engine, no cluster service code runs at import time).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "cluster"))

import campaign_real as cr  # noqa: E402

from engine.generation import JepaReserve, WaveAgent  # noqa: E402
from engine.handoff import Dossier  # noqa: E402


def _cfg(**over):
    base = dict(task="intphys2_physics_binary", claim_types=("effect",),
                origin_date=date(2025, 6, 1), rpc_dir="/x/rpc", svc_name="svc", svc_dir="/x")
    base.update(over)
    return cr.TaskConfig(**base)


class _FakeScout:
    """A stand-in blind scout: returns a JEPA candidate (lower natural rank) and a non-JEPA candidate,
    each grounded + well-formed, WITHOUT dataset/task (the stamper adds them) and with a real mechanism
    (which must survive unchanged, proving no re-pin)."""

    def propose(self, context):
        return [
            {"claim": "a spatial appearance idea", "claim_type": "effect", "backbone": "iv2",
             "scale": "7b", "measure": "accuracy", "prior_claim": False,
             "mechanism": "spatial texture", "grounding": "arXiv:2401.12345"},
            {"claim": "a v-jepa predictive idea", "claim_type": "effect", "backbone": "iv2",
             "scale": "7b", "measure": "accuracy", "prior_claim": False,
             "mechanism": "v-jepa latent prediction", "grounding": "arXiv:2401.12346"},
        ]


# --- the task registry + config ---

def test_task_registry_has_intphys2_effect_no_incumbent():
    c = cr.TASKS["intphys2"]
    assert c.claim_types == ("effect",)
    assert c.incumbent_tasks == ()               # no signed incumbent -> classifier routes to EFFECT
    assert (c.jepa_floor, c.jepa_cap) == (5, 10)  # the standing JEPA reserve


# --- task stamping: stamp the task key, NEVER the mechanism (the drift fix) ---

def test_task_stamping_stamps_task_not_mechanism():
    stamped = cr.task_stamping_scout(_FakeScout(), task="intphys2_physics_binary")
    out = stamped.propose({"vein": "limitations"})
    assert all(c["dataset"] == "intphys2_physics_binary" for c in out)
    assert all(c["task"] == "intphys2_physics_binary" for c in out)
    # mechanisms are the scout's own, untouched: nothing is pinned to temporal_frequency.
    mechs = {c["mechanism"] for c in out}
    assert mechs == {"spatial texture", "v-jepa latent prediction"}
    assert "temporal_frequency" not in mechs


def test_task_stamping_does_not_inject_a_mechanism_when_absent():
    class _NoMech:
        def propose(self, context):
            return [{"claim": "c", "claim_type": "effect", "backbone": "b", "scale": "7b",
                     "measure": "accuracy", "prior_claim": False, "grounding": "arXiv:2401.1"}]

    out = cr.task_stamping_scout(_NoMech(), task="t").propose({})
    assert "mechanism" not in out[0]             # no mechanism pinned when the scout gives none


# --- build_wave_agent: the grounded WaveAgent with the JEPA reserve wired into the steering ---

def test_build_wave_agent_wires_reserve_envelope_and_grounding():
    config = _cfg()
    agent = cr.build_wave_agent(config, scout=_FakeScout(), resolver=lambda i: True)
    assert isinstance(agent, WaveAgent)
    assert isinstance(agent._reserve, JepaReserve)
    assert (agent._reserve.floor, agent._reserve.cap) == (config.jepa_floor, config.jepa_cap)
    assert agent._wired == config.claim_types      # the wired claim-type envelope

    # below the floor, the reserve lifts the JEPA candidate to index 0 (what run_campaign matures),
    # and every candidate is stamped with the signed task key. The mechanism is the scout's own.
    candidates = agent.propose({"vein": "cross_domain_analogy", "mode": "depth", "jepa_matured": 0})
    assert candidates[0]["jepa"] is True
    assert candidates[0]["dataset"] == config.task
    assert candidates[0]["mechanism"] == "v-jepa latent prediction"


def test_build_wave_agent_cap_deprioritizes_jepa():
    agent = cr.build_wave_agent(_cfg(), scout=_FakeScout(), resolver=lambda i: True)
    candidates = agent.propose({"vein": "cross_domain_analogy", "mode": "depth", "jepa_matured": 10})
    assert candidates[0]["jepa"] is False          # cap reached -> the other ideas take the slot


# --- build_triage: the unattended classifier handoff, forced by the signed incumbent catalog ---

def _dossier(proposed, task):
    return Dossier(claim="c", proposed_claim_type=proposed, believed_claim=True,
                   form={"task": task, "dataset": task})


def test_build_triage_no_incumbent_commits_effect():
    triage = cr.build_triage(_cfg())                       # incumbent_tasks=()
    d = triage(_dossier("effect", "intphys2_physics_binary"))
    assert d.accept and d.claim_type == "effect"
    assert d.consequence_template_id == "effect" and d.seeds == (0, 1)


def test_build_triage_incumbent_forces_capability_and_rejects_mismatch():
    config = _cfg(incumbent_tasks=("intphys2_physics_binary",), claim_types=("capability",))
    triage = cr.build_triage(config)
    # the agent advises effect, but a signed incumbent forces the strict CAPABILITY bar -> mismatch,
    # shelved fail-closed (no box spent), the loop never dodges to a weaker gauntlet.
    assert triage(_dossier("effect", "intphys2_physics_binary")).accept is False
    # advising capability agrees with the strictest-consistent gate -> accepted.
    ok = triage(_dossier("capability", "intphys2_physics_binary"))
    assert ok.accept and ok.claim_type == "capability"


# --- significance_penalty: the advisory importance adapter, fail-soft ---

def test_significance_penalty_none_when_no_adversary():
    assert cr.significance_penalty(None) is None


def test_significance_penalty_reads_the_adversary_and_fails_soft():
    class _Sig:
        def challenge(self, candidate):
            from engine.discovery_roles import SignificanceVerdict
            return SignificanceVerdict(incremental=True, case="already known", importance_penalty=0.4)

    pen = cr.significance_penalty(_Sig())
    assert abs(pen({"claim": "x"}) - 0.4) < 1e-9

    class _Boom:
        def challenge(self, candidate):
            raise RuntimeError("adversary offline")

    assert cr.significance_penalty(_Boom())({"claim": "x"}) == 0.0   # advisory -> 0.0, never a kill


def test_novelty_cascade_tiers_down_then_uses_a_live_source(monkeypatch):
    # tier 1 down (S2 429) must not fail-close: it tiers down to the next source and uses its answer.
    calls = []
    monkeypatch.setattr(cr, "_NOVELTY_TIERS", (
        ("s2", lambda q: (_ for _ in ()).throw(RuntimeError("429"))),
        ("arxiv", lambda q: calls.append("arxiv") or ["some unrelated title"]),
    ))
    collision, titles, novel = cr._real_novelty_audit({"mechanism": "a brand new mechanism"})
    assert calls == ["arxiv"]          # the fallback tier actually ran
    assert titles == ["some unrelated title"]
    assert novel is True               # prior work returned, no collision -> novel


def test_novelty_cascade_hard_stops_when_all_sources_down(monkeypatch):
    # every tier down -> HARD STOP (red flag), never a silent not-novel reject.
    monkeypatch.setattr(cr, "_NOVELTY_TIERS", (
        ("s2", lambda q: (_ for _ in ()).throw(RuntimeError("down"))),
        ("arxiv", lambda q: (_ for _ in ()).throw(RuntimeError("down"))),
    ))
    import pytest
    with pytest.raises(cr.NoveltySourcesExhausted):
        cr._real_novelty_audit({"mechanism": "x"})


def test_novelty_collision_marks_not_novel(monkeypatch):
    # a returned title that contains the query text is a collision -> not novel.
    monkeypatch.setattr(cr, "_NOVELTY_TIERS", (
        ("s2", lambda q: ["A study of my exact mechanism and more"]),
    ))
    collision, _, novel = cr._real_novelty_audit({"mechanism": "my exact mechanism"})
    assert collision is True and novel is False
