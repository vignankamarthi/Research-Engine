"""Grounded, vein-diverse, cumulative, blind generation (PLAN 57 b/c/d/f). The generation stage sits
between the blind scout and the campaign: it stamps each candidate with its vein + mechanism-family,
enforces per-wave distinctness, checks campaign concentration, constrains candidates to the wired
claim-type envelope (dropping malformed ones without ever going fatal), conditions the prompt context
on campaign state, and ranks by quality plus advisory importance (never a silent kill). `generate_wave`
composes the whole stage into one orchestrator hook."""
from engine.generation import (
    GENERATIVE_VEINS,
    VEINS,
    apply_importance_penalty,
    bind_high_patience_slot,
    build_generation_context,
    choose_mode,
    concentration_check,
    constrain_candidates,
    enforce_distinct,
    exclude_dead_ends,
    generate_wave,
    human_seed,
    mechanism_family,
    rank_candidates,
    stamp_candidate,
)
from engine.steering import BREADTH, DEPTH, SteeringPolicy


# --- (b) vein set + stamping + distinctness + concentration ---

def test_vein_set_is_six_derivative_plus_three_generative():
    assert len(VEINS) == 9 and len(GENERATIVE_VEINS) == 3


def test_choose_mode_is_the_depth_breadth_chooser():
    # choose_mode is the correctly-named depth/breadth chooser (was choose_vein, a misnomer).
    p = SteeringPolicy()
    assert choose_mode(BREADTH, depth_count=0, breadth_count=0, policy=p) == DEPTH  # depth floor first


def test_mechanism_family_classifies_and_defaults_other():
    assert mechanism_family({"mechanism": "temporal_frequency"}) == "temporal"
    assert mechanism_family({"measure": "spatial appearance fidelity"}) == "spatial"
    assert mechanism_family({"claim": "something unrelated"}) == "other"


def test_stamp_candidate_adds_vein_and_family():
    c = stamp_candidate({"mechanism": "motion cues"}, "limitations")
    assert c["vein"] == "limitations" and c["mechanism_family"] == "temporal"


def test_enforce_distinct_drops_duplicate_vein_family():
    cands = [
        {"vein": "limitations", "mechanism_family": "temporal"},
        {"vein": "limitations", "mechanism_family": "temporal"},  # dup
        {"vein": "future_work", "mechanism_family": "temporal"},
    ]
    kept, dropped = enforce_distinct(cands)
    assert len(kept) == 2 and len(dropped) == 1


def test_concentration_check_pages_on_monoculture():
    mats = [{"mechanism_family": "temporal"}] * 4 + [{"mechanism_family": "spatial"}]
    v = concentration_check(mats)
    assert v.page is True and v.dominant_family == "temporal"


def test_concentration_check_ok_when_diverse():
    mats = [{"mechanism_family": f} for f in ("temporal", "spatial", "attention", "scaling")]
    assert concentration_check(mats).page is False


def test_concentration_check_ignores_small_waves():
    assert concentration_check([{"mechanism_family": "temporal"}]).page is False


# --- (c) generation conditioned on campaign state + dead-end exclusion ---

def test_build_generation_context_carries_campaign_state():
    ctx = build_generation_context(
        vein="limitations", mode=DEPTH, negative_bank={"lk2", "lk1"},
        prior_results=[{"status": "FAILED"}], surprises=["odd partial"],
        reviewer_objections=["confound X"], framing_draft="thesis draft")
    assert ctx["vein"] == "limitations" and ctx["mode"] == DEPTH
    assert ctx["negative_bank"] == ["lk1", "lk2"]  # sorted, deterministic
    assert ctx["surprises"] == ["odd partial"] and ctx["framing_draft"] == "thesis draft"


def test_exclude_dead_ends_uses_is_dead_end():
    cands = [{"claim": "a"}, {"claim": "b"}]
    lineage = {"a": "lk_dead", "b": "lk_live"}
    kept, dropped = exclude_dead_ends(cands, {"lk_dead"},
                                      lineage_of=lambda c: lineage[c["claim"]])
    assert [c["claim"] for c in kept] == ["b"] and len(dropped) == 1


# --- (d) wired claim-type envelope; malformed dropped and logged, never fatal ---

def test_constrain_drops_malformed_and_logs():
    log = []
    cands = [
        {"claim": "ok", "claim_type": "effect", "backbone": "iv2", "dataset": "ssv2",
         "scale": "7b", "measure": "acc"},
        {"claim": "missing fields", "claim_type": "effect"},                 # malformed
        {"claim": "bad type", "claim_type": "vibes", "backbone": "iv2", "dataset": "ssv2",
         "scale": "7b", "measure": "acc"},                                   # not wired
    ]
    kept = constrain_candidates(cands, wired_claim_types={"effect", "capability"}, log=log)
    assert len(kept) == 1 and len(log) == 2  # dropped, not fatal


# --- (f) rank + high-patience slot + importance_penalty consumer ---

def test_apply_importance_penalty_lowers_but_never_removes():
    assert apply_importance_penalty(1.0, 0.5) == 0.5
    assert apply_importance_penalty(1.0, 0.0) == 1.0
    assert apply_importance_penalty(1.0, 2.0) == 0.0  # clamped, still a number (no drop)


def test_rank_never_drops_and_orders_by_score():
    scored = [
        ({"claim": "hi_q_low_imp"}, 0.9, 0.0),
        ({"claim": "hi_q_hi_pen"}, 0.9, 0.8),
        ({"claim": "mid"}, 0.5, 0.0),
    ]
    ranked = rank_candidates(scored)
    assert len(ranked) == 3                       # never a silent kill
    assert ranked[0].candidate["claim"] == "hi_q_low_imp"
    assert ranked[-1].candidate["claim"] == "hi_q_hi_pen"  # importance penalty sinks it, not drops


def test_bind_high_patience_slot_picks_generative():
    from engine.generation import RankedCandidate
    ranked = [
        RankedCandidate({"vein": "limitations"}, 0.9, 0.0, 0.9),
        RankedCandidate({"vein": "cross_domain_analogy"}, 0.6, 0.0, 0.6),  # generative
    ]
    slot = bind_high_patience_slot(ranked)
    assert slot.candidate["vein"] in GENERATIVE_VEINS


def test_human_seed_entry_point():
    c = human_seed("a bold claim", claim_type="effect")
    assert c["origin"] == "human_seed" and c["claim"] == "a bold claim"


# --- generate_wave: the composed orchestrator hook ---

class _Scout:
    def propose(self, context):
        return [
            {"claim": "temporal effect", "claim_type": "effect", "backbone": "iv2",
             "dataset": "ssv2", "scale": "7b", "measure": "acc",
             "mechanism": "temporal_frequency", "grounding": "arXiv:2401.12345"},
            {"claim": "malformed", "claim_type": "effect"},  # dropped by envelope
        ]


def test_generate_wave_composes_stages():
    ctx = build_generation_context(vein="limitations", mode=DEPTH)
    result = generate_wave(
        scout=_Scout(), context=ctx, wired_claim_types={"effect"},
        resolver=lambda i: True, negative_bank=set(),
        lineage_of=lambda c: c["claim"])
    assert len(result.ranked) == 1                       # the grounded, well-formed one survived
    assert len(result.malformed) == 1                    # the bad one was logged, not fatal
    assert result.grounding.rate() == 1.0                # the survivor grounded
    assert result.ranked[0].candidate["vein"] == "limitations"


# --- WaveAgent: the adapter that puts generate_wave behind run_campaign's agent contract ---

class _FullScout(_Scout):
    """A scout that also matures + frames, so it can stand in as the campaign's agent. Its well-formed
    candidate carries `prior_claim` so the default lineage key (which normalizes the schema) resolves."""

    def propose(self, context):
        return [
            {"claim": "temporal effect", "claim_type": "effect", "backbone": "iv2",
             "dataset": "ssv2", "scale": "7b", "measure": "acc", "prior_claim": False,
             "mechanism": "temporal_frequency", "grounding": "arXiv:2401.12345"},
            {"claim": "malformed", "claim_type": "effect"},  # dropped by the envelope pre-lineage
        ]

    def mature(self, schema_raw):
        return ("MATURED", schema_raw)

    def frame(self, schema_raw, verdict):
        return "narrative"


def test_wave_agent_propose_runs_the_wave_and_returns_ranked_candidates():
    from engine.generation import WaveAgent

    agent = WaveAgent(_FullScout(), resolver=lambda i: True, wired_claim_types={"effect"})
    ctx = build_generation_context(vein="limitations", mode=DEPTH)
    candidates = agent.propose(ctx)
    assert len(candidates) == 1                           # only the grounded, well-formed one
    assert candidates[0]["vein"] == "limitations"         # stamped by the wave
    assert agent.last_wave.grounding.rate() == 1.0        # WaveResult exposed for the orchestrator
    assert len(agent.last_wave.malformed) == 1            # the drop is visible, not fatal


def test_wave_agent_delegates_mature_and_frame_to_the_scout():
    from engine.generation import WaveAgent

    agent = WaveAgent(_FullScout(), resolver=lambda i: True, wired_claim_types={"effect"})
    assert agent.mature({"claim": "c"}) == ("MATURED", {"claim": "c"})
    assert agent.frame({"claim": "c"}, verdict=None) == "narrative"


def test_wave_agent_empty_wave_returns_the_scout_fallback():
    from engine.generation import WaveAgent

    class _Empty(_FullScout):
        def propose(self, context):
            return []

    agent = WaveAgent(_Empty(), resolver=lambda i: True, wired_claim_types={"effect"})
    assert agent.propose(build_generation_context(vein="limitations", mode=DEPTH)) == [{}]
