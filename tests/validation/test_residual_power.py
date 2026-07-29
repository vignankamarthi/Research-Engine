"""Milestone-4 residual power curve. The FLOOR gate lives on the trained-minus-untrained
RESIDUAL, so trust in it requires the residual itself to be powered: an effect planted into
the residual at 1x the MIE is detected at >= 0.8 power, at 0.5x it mostly is not, and at 2x it
is detected nearly always. This is the residual analogue of the raw-effect sensitivity curve,
sizing the box against the residual as step 13 requires."""
from backend import Box, MockBackend
from gatelib import paired_diff_ci

MIE = 0.03
ALPHA = 0.05


def residual_detect_rate(residual, base, n_boxes, box_n, noise, seed):
    """Plant a residual of the given size on top of a shared geometric `base`, then measure
    how often the paired trained-minus-untrained CI excludes zero."""
    hits = 0
    for i in range(n_boxes):
        be = MockBackend(trained_effect=base + residual, untrained_effect=base, noise=noise, seed=seed + i)
        box = Box(id=f"r{i}", n=box_n)
        trained = be.score_box(box)
        untrained = be.score_box(box, untrained_init=0)
        _, lo, _ = paired_diff_ci(trained, untrained, ALPHA)
        if lo > 0:
            hits += 1
    return hits / n_boxes


def test_residual_at_half_mie_is_mostly_undetected():
    rate = residual_detect_rate(0.5 * MIE, base=0.2, n_boxes=300, box_n=300, noise=0.1, seed=6000)
    assert rate < 0.6


def test_residual_at_one_mie_is_detected_at_high_power():
    rate = residual_detect_rate(1.0 * MIE, base=0.2, n_boxes=300, box_n=300, noise=0.1, seed=7000)
    assert rate >= 0.8


def test_residual_at_two_mie_is_detected_almost_always():
    rate = residual_detect_rate(2.0 * MIE, base=0.2, n_boxes=300, box_n=300, noise=0.1, seed=8000)
    assert rate >= 0.95
