"""Milestone-4 adversarial suite. The trust-concentration point is the single
schema-normal-form (it derives the control set, lineage key, and magnitude gate), so
it is attacked directly: no phrasing draws a weaker gauntlet, a reworded-but-equivalent
claim cannot escape the negative bank into a fresh lineage, dataset aliases collide,
and a prior claim still carries an artifact catcher. Plus the geometry-artifact catch
(the Temporal-RoPE failure) end to end."""
import numpy as np

from backend import Box, MockBackend
from gatelib import floor_separation
from referee import derive, lineage_key, normalize_schema


def s(**over):
    raw = {"claim": "x improves recognition", "claim_type": "effect", "backbone": "iv2",
           "dataset": "ssv2", "scale": "7b", "measure": "top-1 accuracy", "prior_claim": False}
    raw.update(over)
    return normalize_schema(raw)


def test_no_phrasing_draws_a_weaker_gauntlet():
    for over in [{}, {"measure": "acc"}, {"backbone": "videomae"}, {"scale": "1b"},
                 {"claim_type": "capability"}]:
        sch = s(**over)
        d = derive(sch)
        floor = "prior_ablated_baseline" if sch.prior_claim else "untrained_floor"
        assert floor in d.control_set
        assert "control_of_the_control" in d.control_set


def test_reworded_equivalent_claim_cannot_escape_the_bank():
    original = s()
    reworded = s(measure="acc", dataset="something-something v2", backbone="videomae", scale="1b")
    assert lineage_key(original) == lineage_key(reworded)


def test_dataset_alias_collision_is_canonicalized():
    assert s(dataset="ssv2").dataset == s(dataset="something-something v2").dataset


def test_prior_claim_still_carries_an_artifact_catcher():
    assert "prior_ablated_baseline" in derive(s(prior_claim=True)).control_set


def test_geometry_artifact_is_caught_and_names_the_worst_init():
    rng = np.random.default_rng(12)
    trained = rng.normal(0.25, 0.1, 500)
    untrained = [rng.normal(0.0, 0.1, 500), rng.normal(0.0, 0.1, 500), rng.normal(0.25, 0.1, 500)]
    r = floor_separation(trained, untrained, mie=0.03)
    assert not r.passed and r.worst_init == 2


def test_degenerate_control_end_to_end_fails_the_floor():
    be = MockBackend(trained_effect=0.25, untrained_effect=0.25, noise=0.1, seed=9)
    box = Box(id="deg", n=500)
    trained = be.score_box(box)
    untrained = [be.score_box(box, untrained_init=k) for k in range(4)]
    assert not floor_separation(trained, untrained, mie=0.03).passed
