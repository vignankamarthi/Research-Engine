"""The idea-AGNOSTIC ablation primitive library (spec 3c, step 44). A small vetted set of general
removal ops with no domain in them, the building blocks the blue team composes into an idea-specific
ablation. Each primitive carries a SPECIFICITY self-test: on a synthetic control it removes its
target structure and moves nothing where the target is absent. A primitive that fails its self-test
is excluded fail-closed, so the blue team never builds from an op that does not cleanly isolate."""
import numpy as np

from engine.ablation_primitives import (
    PRIMITIVES,
    Primitive,
    spectral_mask,
    subspace_project_out,
    vetted_primitives,
    zero_channels,
)


def test_spectral_mask_removes_high_frequency_preserves_low():
    # slow ramp (low temporal freq) + fast flicker (high temporal freq) along axis 0
    t = np.arange(16)[:, None]
    ramp = (t / 16.0) * np.ones((16, 5))
    flicker = np.where(t % 2 == 0, 1.0, -1.0) * np.ones((16, 5))
    out = spectral_mask(ramp + flicker, axis=0, keep_fraction=0.25)
    # the flicker (high-freq target) is gone, the ramp (low-freq off-target) survives
    assert np.var(out - ramp) < 0.05 * np.var(flicker)


def test_spectral_mask_is_axis_general():
    # the same primitive ablates along ANY axis, not just time (nothing domain-specific)
    x = np.random.default_rng(0).random((5, 16))
    out = spectral_mask(x, axis=1, keep_fraction=0.25)
    assert out.shape == x.shape


def test_subspace_project_out_removes_the_named_directions():
    basis = np.array([[1.0, 0.0, 0.0]])  # remove the e0 direction only
    x = np.array([[3.0, 2.0, 5.0], [1.0, 7.0, 4.0]])
    out = subspace_project_out(x, basis)
    assert np.allclose(out[:, 0], 0.0)              # the e0 component is gone
    assert np.allclose(out[:, 1:], x[:, 1:])        # the orthogonal components are untouched


def test_zero_channels_zeros_only_the_named_channels():
    x = np.arange(12, dtype=float).reshape(3, 4)
    out = zero_channels(x, channels=[1], axis=-1)
    assert np.allclose(out[:, 1], 0.0)
    assert np.allclose(np.delete(out, 1, axis=1), np.delete(x, 1, axis=1))


def test_every_registered_primitive_passes_its_specificity_self_test():
    assert PRIMITIVES  # non-empty
    for name, prim in PRIMITIVES.items():
        assert prim.self_test() is True, f"{name} failed its specificity self-test"


def test_vetted_primitives_excludes_a_primitive_that_fails_its_self_test():
    broken = Primitive("broken", "always fails", lambda x: x, lambda: False)
    vetted = vetted_primitives({**PRIMITIVES, "broken": broken})
    assert "broken" not in vetted and set(vetted) == set(PRIMITIVES)


def test_temporal_frequency_is_now_a_spectral_mask_parameterization():
    # the old domain-specific op is just spectral_mask on the time axis, no privileged status
    from engine.ablations import remove_temporal_frequency
    x = np.random.default_rng(1).random((12, 4, 4, 3))
    assert np.allclose(remove_temporal_frequency(x, keep_fraction=0.5),
                       spectral_mask(x, axis=0, keep_fraction=0.5))
