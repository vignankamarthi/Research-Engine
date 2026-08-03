"""Pure scoring math for the cluster scorers, numpy-only so it imports and unit-tests on a laptop
with no torch / transformers / decord. The GPU scorers (ssv2/tomato/intphys2_qwen) call these on the
logits they read from Qwen, so the validity-critical arithmetic (the logit-lean softmax and the
temporal-frequency keep-fraction) is testable off the cluster while the model plumbing stays on it."""
import numpy as np


def softmax(x):
    a = np.asarray(x, dtype=np.float64)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def option_probs(next_token_logits, option_token_ids):
    """The LOGIT-LEAN score. One forward pass gives `next_token_logits`, the model's logit row over
    the whole vocab at the answer position. `option_token_ids` is one token id per answer option
    (each option's first / single answer token). Gather just those option logits and softmax over
    them, so the return is a probability vector over the options summing to 1, a continuous [0,1]
    lean toward each option.

    No generation and no word-parsing, so there is no yes-bias and no unparseable-fallthrough. An
    untrained (weights-randomized) model's near-flat token geometry yields a near-uniform vector,
    which is the HONEST FLOOR (its real logit lean), not a filename coin flip.

    Duplicate ids are allowed: for a free-label set (SSv2's 174 classes) two labels can share a
    first token, and softmax over the gathered row splits that token's mass equally between them,
    which is the correct uniform tie-break. For a small option set where a collision would be a bug
    (binary possible/impossible, MCQ indices) the caller resolves the ids with require_unique=True."""
    logits = np.asarray(next_token_logits, dtype=np.float64)
    ids = np.asarray(option_token_ids, dtype=np.int64)
    return softmax(logits[ids])


def continuation_probs(option_logprobs):
    """Turn per-option CONTINUATION log-likelihoods into a probability vector over the options.

    `option_logprobs[i]` is the length-normalized mean log-prob of option i (from
    `ssv2_qwen.continuation_logprobs`), i.e. the model's mean log-likelihood of writing that whole
    option word. Softmax over them gives a [0,1] lean toward each option summing to 1. This replaces
    `option_probs` for the BINARY possible/impossible task, where a single-token read was invalid
    because the model's surface form ("Possible") did not match the bare option token. The math is a
    plain softmax; it lives here so the option-vs-option normalization is unit-tested off the cluster
    while the teacher-forcing forward passes stay in the GPU module. An untrained model yields nearly
    equal option log-probs, so this returns a near-uniform vector, the honest FLOOR."""
    return softmax(option_logprobs)


def spectral_keep_bins(n_bins, keep_fraction):
    """How many low temporal-frequency bins to KEEP (indices [0, n_keep)) so the ablation removes a
    real band without destroying all motion. `n_bins` is the rfft bin count (t // 2 + 1).

    The fix for the old DC-only bug: keep the DC bin PLUS at least one AC band (n_keep >= 2) so real
    motion survives, and always drop at least the top band (n_keep <= n_bins - 1) so it stays a
    genuine ablation. Returns None when there are too few bins (n_bins < 3) to do both, so the caller
    leaves the frames untouched rather than silently zeroing all motion.

    Old behaviour it replaces: max(1, round(0.25 * 5)) = 1 on 8 frames kept only the DC bin, which
    removed ALL motion. With 16 frames (n_bins = 9) round(0.25 * 9) = 2 keeps DC + 1 AC band."""
    if n_bins < 3:
        return None
    n_keep = int(round(keep_fraction * n_bins))
    return max(2, min(n_bins - 1, n_keep))


def spectral_mask_time(frames, keep_fraction=0.5):
    """Zero the high temporal-frequency bands along the time axis (the temporal_frequency ablation).
    Keeps DC + >= 1 AC band so real motion survives and removes >= 1 top band so it is a genuine
    ablation (see spectral_keep_bins). `frames` is (T, H, W, C) uint8."""
    x = np.asarray(frames).astype(np.float64)
    t = x.shape[0]
    if t < 2:
        return frames
    spec = np.fft.rfft(x, axis=0)
    n_keep = spectral_keep_bins(spec.shape[0], keep_fraction)
    if n_keep is None:
        return frames
    spec[n_keep:] = 0.0
    out = np.fft.irfft(spec, n=t, axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)
