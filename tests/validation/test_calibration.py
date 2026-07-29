"""Milestone-4 trust validation (Monte Carlo). The confirmatory core is trusted only
if it is calibrated: a true effect below the MIE is called a positive at most ~alpha,
a clearly-above-MIE effect is detected at high power, and the FLOOR almost never
passes a geometry artifact while reliably passing a genuine effect. These gate trust
in the referee before any discovery output is believed."""

from backend import Box, MockBackend
from gatelib import EXCEEDS_MIE, classify_magnitude, floor_separation, mean_ci

MIE = 0.03
ALPHA = 0.05


def magnitude_accept_rate(true_effect, n_boxes, box_n, noise, seed):
    accepts = 0
    for i in range(n_boxes):
        be = MockBackend(trained_effect=true_effect, untrained_effect=0.0, noise=noise, seed=seed + i)
        scores = be.score_box(Box(id=f"b{i}", n=box_n))
        lo, hi = mean_ci(scores, ALPHA)
        if classify_magnitude(lo, hi, MIE) == EXCEEDS_MIE:
            accepts += 1
    return accepts / n_boxes


def floor_pass_rate(trained_effect, untrained_effect, n_boxes, box_n, noise, seed, k=4):
    passes = 0
    for i in range(n_boxes):
        be = MockBackend(trained_effect, untrained_effect, noise, seed=seed + i)
        box = Box(id=f"f{i}", n=box_n)
        trained = be.score_box(box)
        untrained = [be.score_box(box, untrained_init=j) for j in range(k)]
        if floor_separation(trained, untrained, MIE, ALPHA).passed:
            passes += 1
    return passes / n_boxes


def test_null_below_mie_false_accept_at_or_below_alpha():
    rate = magnitude_accept_rate(true_effect=0.0, n_boxes=400, box_n=400, noise=0.1, seed=1000)
    assert rate <= ALPHA


def test_clearly_above_mie_is_detected_at_high_power():
    rate = magnitude_accept_rate(true_effect=0.08, n_boxes=300, box_n=400, noise=0.1, seed=2000)
    assert rate >= 0.8


def test_small_subthreshold_effect_is_mostly_rejected():
    rate = magnitude_accept_rate(true_effect=0.015, n_boxes=300, box_n=400, noise=0.1, seed=3000)
    assert rate < 0.2


def test_floor_rarely_passes_a_geometry_artifact():
    rate = floor_pass_rate(trained_effect=0.20, untrained_effect=0.20,
                           n_boxes=300, box_n=400, noise=0.1, seed=4000)
    assert rate <= ALPHA


def test_floor_reliably_passes_a_genuine_effect():
    rate = floor_pass_rate(trained_effect=0.20, untrained_effect=0.0,
                           n_boxes=300, box_n=400, noise=0.1, seed=5000)
    assert rate >= 0.8
