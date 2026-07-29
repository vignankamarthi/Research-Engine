"""The untrained-weights FLOOR separation gate. A genuine effect must separate from
a weights-randomized untrained model by more than the MIE, measured PAIRED on the
same box against the WORST-CASE of K untrained inits, so a geometry artifact an
untrained model reproduces (the Temporal-RoPE lesson) fails, and a lucky-low
untrained draw cannot let one through."""
import numpy as np
import pytest

from gatelib import floor_separation


def test_genuine_effect_passes_floor():
    rng = np.random.default_rng(0)
    n = 500
    trained = rng.normal(0.20, 0.1, n)
    untrained_runs = [rng.normal(0.0, 0.1, n) for _ in range(4)]
    res = floor_separation(trained, untrained_runs, mie=0.05)
    assert res.passed
    assert res.ci_lo > 0.05


def test_geometry_artifact_fails_floor():
    # trained effect is fully reproduced by the untrained model -> residual ~ 0
    rng = np.random.default_rng(1)
    n = 500
    trained = rng.normal(0.20, 0.1, n)
    untrained_runs = [rng.normal(0.20, 0.1, n) for _ in range(4)]
    res = floor_separation(trained, untrained_runs, mie=0.05)
    assert not res.passed


def test_worst_case_init_governs():
    # three low untrained inits and one that matches the trained effect;
    # the worst-case (matching) init must drive the residual to ~0 -> fail.
    rng = np.random.default_rng(2)
    n = 500
    trained = rng.normal(0.20, 0.1, n)
    untrained_runs = [
        rng.normal(0.0, 0.1, n),
        rng.normal(0.0, 0.1, n),
        rng.normal(0.0, 0.1, n),
        rng.normal(0.20, 0.1, n),  # the artifact-reproducing init
    ]
    res = floor_separation(trained, untrained_runs, mie=0.05)
    assert not res.passed
    assert res.worst_init == 3


def test_empty_untrained_runs_rejected():
    with pytest.raises(ValueError):
        floor_separation(np.zeros(10), [], mie=0.05)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        floor_separation(np.zeros(10), [np.zeros(9)], mie=0.05)
