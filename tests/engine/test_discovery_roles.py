"""The discovery-tier adversary and synthesis roles. The reviewer-adversary raises the
strongest CORRECTNESS objections, and surviving it is a maturity CONDITION (a fatal objection
blocks maturity). The significance-adversary raises the strongest incremental/already-known
case, ADVISORY only (it lowers an importance rank and feeds the dossier, never a silent kill).
The synthesizer clusters findings under one thesis with a joint prediction, and an arc ships as
one paper only if that joint prediction is NOT already entailed by a single finding."""
from engine.discovery_roles import (
    ClaudeReviewerAdversary,
    ClaudeSignificanceAdversary,
    ClaudeSynthesizer,
    MockReviewerAdversary,
    MockSignificanceAdversary,
    ReviewerVerdict,
    Synthesis,
    decide_arc,
    is_mature,
)


def test_mock_reviewer_survives_by_default():
    assert MockReviewerAdversary().review({"claim": "c"}, {}).survives is True


def test_mock_significance_is_not_incremental_by_default():
    assert MockSignificanceAdversary().challenge({"claim": "c"}).incremental is False


def test_is_mature_requires_both_agent_and_reviewer():
    ok = ReviewerVerdict(survives=True, objections=[])
    fatal = ReviewerVerdict(survives=False, objections=["confound"])
    assert is_mature(agent_matured=True, reviewer=ok) is True
    assert is_mature(agent_matured=True, reviewer=fatal) is False   # reviewer can block
    assert is_mature(agent_matured=False, reviewer=ok) is False


def test_arc_ships_when_joint_prediction_is_not_entailed():
    assert decide_arc(Synthesis("t", "jp", entailed_by_single=False)) is True


def test_arc_ships_separately_when_joint_prediction_is_entailed():
    # if a single finding already entails the joint prediction, it is not a real arc.
    assert decide_arc(Synthesis("t", "jp", entailed_by_single=True)) is False


def test_claude_reviewer_parses_a_fatal_objection():
    r = ClaudeReviewerAdversary(runner=lambda p: '{"survives": false, "objections": ["confound X"]}')
    v = r.review({"claim": "c"}, {})
    assert v.survives is False and "confound X" in v.objections


def test_claude_significance_parses_incremental_case():
    a = ClaudeSignificanceAdversary(
        runner=lambda p: '{"incremental": true, "case": "shown in 2021", "importance_penalty": 0.5}')
    v = a.challenge({"claim": "c"})
    assert v.incremental is True and v.importance_penalty == 0.5


def test_claude_synthesizer_parses():
    s = ClaudeSynthesizer(
        runner=lambda p: '{"thesis": "T", "joint_prediction": "JP", "entailed_by_single": false}')
    out = s.synthesize([{"claim": "a"}, {"claim": "b"}])
    assert out.thesis == "T" and out.entailed_by_single is False
