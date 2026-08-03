"""Unit tests for the pure cluster-scorer math (cluster/scoring_math.py): the logit-lean scoring
(step 71) and the temporal-frequency keep-fraction (step 72). numpy-only, so it runs on a laptop
with no torch / transformers / decord. The GPU forward pass and the option-token resolution need a
cluster smoke; the arithmetic here does not. conftest.py puts cluster/ on sys.path."""
import numpy as np
import pytest
import scoring_math as sm

# ---- step 72: the ablation keep-fraction (the DC-only bug and its fix) ----

def test_keep_bins_old_dc_only_bug_is_fixed():
    # 8 frames -> rfft has 5 bins. Old code: max(1, round(0.25*5)) = 1 => DC-only, all motion gone.
    # Fixed: keep DC + at least one AC band.
    assert sm.spectral_keep_bins(5, 0.25) == 2


def test_keep_bins_16_frame_regime():
    # 16 frames -> 9 bins. round(0.25*9)=2 (DC + 1 AC band); round(0.5*9)=4.
    assert sm.spectral_keep_bins(9, 0.25) == 2
    assert sm.spectral_keep_bins(9, 0.5) == 4


def test_keep_bins_lower_bound_preserves_motion():
    # Even keep_fraction 0 keeps DC + 1 AC band (never DC-only), so real motion always survives.
    assert sm.spectral_keep_bins(9, 0.0) == 2


def test_keep_bins_always_removes_a_band():
    # keep_fraction 1.0 must still drop the top band, so it is a genuine ablation not a no-op.
    assert sm.spectral_keep_bins(9, 1.0) == 8


def test_keep_bins_none_when_too_few_bins():
    # < 3 bins can't both keep DC+AC and remove a band; caller then leaves frames untouched.
    assert sm.spectral_keep_bins(2, 0.5) is None


def test_spectral_mask_preserves_low_freq_motion_and_removes_high():
    # A slow temporal ramp (low band) plus fast alternating jitter (top band). After keep=0.25 the
    # ramp must survive (per-pixel temporal variance stays > 0, i.e. motion NOT collapsed to one
    # frame) and the output must differ from the input (a band was actually removed).
    t, h, w, c = 16, 4, 4, 3
    ramp = np.linspace(0, 120, t).reshape(t, 1, 1, 1)          # low temporal frequency
    jitter = 40 * (np.arange(t) % 2).reshape(t, 1, 1, 1)       # Nyquist (top band)
    frames = np.broadcast_to(ramp + jitter, (t, h, w, c)).astype(np.uint8)

    out = sm.spectral_mask_time(frames, keep_fraction=0.25)

    assert out.shape == frames.shape
    assert out.std(axis=0).mean() > 0            # motion preserved (not DC-only)
    assert not np.array_equal(out, frames)       # a band was removed


def test_spectral_mask_short_clip_is_noop():
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    assert np.array_equal(sm.spectral_mask_time(frames), frames)


# ---- step 71: the logit-lean option scoring ----

def _logits(overrides, vocab=100):
    v = np.full(vocab, -5.0)
    for idx, val in overrides.items():
        v[idx] = val
    return v


def test_option_probs_is_a_distribution():
    probs = sm.option_probs(_logits({10: 3.0, 20: 1.0}), [10, 20])
    assert probs.shape == (2,)
    assert probs.sum() == pytest.approx(1.0)
    assert np.all((probs > 0) & (probs < 1))     # continuous [0,1], never a hard {0,1}


def test_option_probs_leans_toward_higher_logit():
    probs = sm.option_probs(_logits({10: 4.0, 20: 0.0}), [10, 20])
    assert probs[0] > probs[1]
    assert probs[0] == pytest.approx(1.0 / (1.0 + np.exp(-4.0)))  # exact 2-way softmax


def test_option_probs_uniform_is_the_honest_floor():
    # Equal logits (an untrained model's near-flat lean) -> chance, NOT a coin flip in code.
    probs = sm.option_probs(_logits({10: 0.7, 20: 0.7}), [10, 20])
    assert probs == pytest.approx([0.5, 0.5])

    mcq = sm.option_probs(_logits({1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0}), [1, 2, 3, 4])
    assert mcq == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_option_probs_mcq_scores_correct_index():
    probs = sm.option_probs(_logits({1: 0.0, 2: 5.0, 3: 0.0, 4: 0.0}), [1, 2, 3, 4])
    assert np.argmax(probs) == 1                 # the high-logit option index
    assert probs.sum() == pytest.approx(1.0)


def test_option_probs_duplicate_first_token_splits_mass_uniformly():
    # SSv2 free-label case: two labels share a first token id (5). The shared token's mass is split
    # equally between them (option_probs allows duplicate ids), a correct uniform tie-break.
    probs = sm.option_probs(_logits({5: 3.0, 9: 3.0}), [5, 5, 9])
    assert probs[0] == pytest.approx(probs[1])
    assert probs == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_is_numerically_stable():
    probs = sm.softmax([1000.0, 1000.0, 1000.0])   # would overflow without the max-shift
    assert probs == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_continuation_probs_is_a_distribution():
    probs = sm.continuation_probs([-0.5, -2.0])
    assert probs.sum() == pytest.approx(1.0)
    assert (probs >= 0).all()


def test_continuation_probs_leans_toward_higher_loglik():
    # -0.3 mean log-prob (more likely word) beats -3.0; P must favor option 0.
    probs = sm.continuation_probs([-0.3, -3.0])
    assert probs[0] > probs[1]
    assert probs[0] > 0.9


def test_continuation_probs_equal_loglik_is_the_honest_floor():
    # An untrained model gives near-equal option log-likelihoods -> ~0.5 each (the chance FLOOR).
    probs = sm.continuation_probs([-4.2, -4.2])
    assert probs == pytest.approx([0.5, 0.5])
