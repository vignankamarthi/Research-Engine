"""The red/blue ablation-CONSTRUCTION loop (spec 3c, step 45). A BLUE role proposes an ablation as
a bounded primitive composition, a RED panel attacks it, and blue revises until the panel cannot
refute, or the idea is marked NO CLEAN ABLATION and the mechanism gate fails closed. The loop is
Python; blue/red are claude -p roles spawned by it, mocked here. These tests prove the convergence,
the fail-closed no-convergence, the revise-on-refutation feedback, and that only VETTED primitives
resolve."""
import numpy as np
import pytest

from engine.ablation_construction import (
    AblationSpec,
    BlueProposal,
    ClaudeBlueBuilder,
    ClaudeRedAttacker,
    MockBlueBuilder,
    MockRedAttacker,
    NoCleanAblation,
    RedVerdict,
    construct_ablation,
    resolve_spec,
)

SPEC = AblationSpec("spectral_mask", {"axis": 0, "keep_fraction": 0.25})


def test_converges_when_the_red_panel_cannot_refute():
    got = construct_ablation(
        "temporal_frequency", "ssv2", blue=MockBlueBuilder(SPEC),
        red_panel=[MockRedAttacker(), MockRedAttacker(), MockRedAttacker()], rounds=4)
    assert got.spec == SPEC and got.rounds == 1
    # the returned ablation actually runs (a real primitive parameterization)
    x = np.random.default_rng(0).random((8, 4))
    assert got.apply(x).shape == x.shape


def test_fails_closed_when_red_always_refutes():
    with pytest.raises(NoCleanAblation):
        construct_ablation(
            "temporal_frequency", "ssv2", blue=MockBlueBuilder(SPEC),
            red_panel=[MockRedAttacker(refute_rounds=99)], rounds=4)


def test_blue_revises_until_red_concedes():
    # red refutes the first two rounds then concedes -> converges at round 3
    red = MockRedAttacker(refute_rounds=2)
    got = construct_ablation(
        "temporal_frequency", "ssv2", blue=MockBlueBuilder(SPEC), red_panel=[red], rounds=5)
    assert got.rounds == 3


def test_refutations_are_fed_back_to_blue():
    seen = {}

    class SpyBlue:
        def build(self, mechanism, task, feedback):
            seen["feedback_len"] = len(feedback)
            return BlueProposal(SPEC)

    construct_ablation("m", "t", blue=SpyBlue(),
                       red_panel=[MockRedAttacker(refute_rounds=1)], rounds=3)
    assert seen["feedback_len"] == 1  # the last round's refutation was passed back


def test_resolve_spec_rejects_an_unvetted_primitive():
    with pytest.raises(NoCleanAblation):
        resolve_spec(AblationSpec("nonexistent_primitive", {}))


def test_resolved_spec_applies_the_named_primitive():
    from engine.ablation_primitives import spectral_mask
    fn = resolve_spec(SPEC)
    x = np.random.default_rng(1).random((8, 4))
    assert np.allclose(fn(x), spectral_mask(x, axis=0, keep_fraction=0.25))


def test_a_single_refuter_in_the_panel_forces_a_revision():
    # one accepter + one refuter -> the round is refuted (any refutation forces a revise)
    red_panel = [MockRedAttacker(), MockRedAttacker(refute_rounds=1)]
    got = construct_ablation("m", "t", blue=MockBlueBuilder(SPEC), red_panel=red_panel, rounds=3)
    assert got.rounds == 2  # round 1 refuted by the second attacker, round 2 clean


def test_red_verdict_carries_the_axis():
    v = RedVerdict(refuted=True, axis="anti_collusion", reason="effect died without the mechanism")
    assert v.refuted and v.axis == "anti_collusion"


def test_construction_loop_plugs_into_build_mechanism_fn():
    # the construction loop replaces the static resolve_ablation seam: build_mechanism_fn accepts a
    # resolve_ablation_fn that constructs + verifies the ablation, score_task applies it via .apply
    from engine.ablations import build_mechanism_fn

    def resolve_via_construction(mechanism, task):
        return construct_ablation(mechanism, task, blue=MockBlueBuilder(SPEC),
                                  red_panel=[MockRedAttacker()], rounds=4)

    def score_task(backend, task, ablation):
        # full model scores 0.10; the constructed ablation (spectral_mask) is applied, scores ~0
        base = np.full((40, 6), 0.10)
        return base.mean(axis=1) if ablation is None else ablation.apply(base).mean(axis=1)

    fn = build_mechanism_fn(score_task=score_task, specificity_check=lambda b, s, t: True,
                            alpha=0.05, resolve_ablation_fn=resolve_via_construction)
    lo, hi, spec = fn(backend=None, schema={"mechanism": "temporal_frequency", "dataset": "ssv2"})
    assert lo > 0.05 and spec is True  # full effect present, ablation was constructed + applied


# --- the real claude -p roles (parse-tested with a canned runner, like the discovery roles) ---

def _canned(text):
    def runner(prompt):
        return text
    return runner


def test_claude_blue_builder_parses_a_proposal():
    reply = '{"primitive": "spectral_mask", "params": {"axis": 0}, "rationale": "removes fast"}'
    prop = ClaudeBlueBuilder(runner=_canned(reply)).build("temporal_frequency", "ssv2", [])
    assert prop.spec.primitive == "spectral_mask" and prop.spec.params["axis"] == 0
    assert "removes" in prop.rationale


def test_claude_red_attacker_parses_a_verdict():
    reply = '{"refuted": true, "axis": "anti_collusion", "reason": "effect dies without it"}'
    v = ClaudeRedAttacker(runner=_canned(reply)).attack("temporal_frequency", SPEC)
    assert v.refuted is True and v.axis == "anti_collusion"


def test_claude_red_attacker_fails_closed_on_missing_verdict():
    # an ambiguous reply with no 'refuted' key reads as REFUTED, so a bad attacker lets none through
    v = ClaudeRedAttacker(runner=_canned('{"reason": "unsure"}')).attack("m", SPEC)
    assert v.refuted is True


def test_real_roles_drive_the_loop_to_convergence():
    blue = ClaudeBlueBuilder(runner=_canned('{"primitive": "spectral_mask", "params": {}}'))
    red = ClaudeRedAttacker(runner=_canned('{"refuted": false, "axis": "none", "reason": "clean"}'))
    got = construct_ablation("temporal_frequency", "ssv2", blue=blue, red_panel=[red], rounds=3)
    assert got.spec.primitive == "spectral_mask" and got.rounds == 1
