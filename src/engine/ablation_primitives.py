"""The idea-AGNOSTIC ablation primitive library (spec 3c). A small vetted set of general removal
operations with no domain in them, the building blocks the blue team composes and parameterizes into
an idea-specific ablation. None of them mention temporal-frequency, video, or any idea: they are
mathematical operations on an array (an input clip or a representation tensor).

Each primitive carries a SPECIFICITY self-test. On a synthetic control it builds itself, the
primitive must remove its target structure and move nothing where the target is absent. A primitive
that fails its self-test is excluded fail-closed (`vetted_primitives`), so the blue team never
composes from an op that does not cleanly isolate. Temporal-frequency ablation is no longer special,
it is `spectral_mask` on the time axis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Primitive:
    name: str
    description: str
    apply: Callable       # (x, **params) -> x with the target structure removed
    self_test: Callable   # () -> bool, the specificity self-check on a synthetic control


# --- the primitive operations (all domain-free) ---

def spectral_mask(x, axis: int = 0, keep_fraction: float = 0.5):
    """Zero the high-frequency bands of `x` along `axis`, keeping the lowest `keep_fraction` of the
    rfft bands. Removes fast structure along that axis (temporal-frequency ablation is this on the
    time axis; spatial-frequency ablation is this on a spatial axis)."""
    a = np.asarray(x, dtype=float)
    n = a.shape[axis]
    if n < 2:
        return a
    spectrum = np.fft.rfft(a, axis=axis)
    n_bands = spectrum.shape[axis]
    n_keep = max(1, min(n_bands, int(round(keep_fraction * n_bands))))
    cut = [slice(None)] * a.ndim
    cut[axis] = slice(n_keep, None)
    spectrum[tuple(cut)] = 0.0
    return np.fft.irfft(spectrum, n=n, axis=axis)


def subspace_project_out(x, basis):
    """Remove the projection of `x` onto the row space of `basis` (each row a direction over the
    LAST axis). Orthogonalizes the basis first, so overlapping directions do not over-subtract."""
    a = np.asarray(x, dtype=float)
    b = np.atleast_2d(np.asarray(basis, dtype=float))
    q, _ = np.linalg.qr(b.T)                 # orthonormal columns spanning the same subspace
    coeffs = a @ q                           # (..., k) projections onto each basis direction
    return a - coeffs @ q.T


def zero_channels(x, channels, axis: int = -1):
    """Zero the named channels (indices) along `axis`, leaving every other channel untouched."""
    a = np.array(x, dtype=float)             # copy, so the caller's array is not mutated
    idx = [slice(None)] * a.ndim
    idx[axis] = list(channels)
    a[tuple(idx)] = 0.0
    return a


# --- per-primitive specificity self-tests (a clean target removed, an off-target preserved) ---

def _spectral_mask_self_test() -> bool:
    t = np.arange(16)[:, None] * np.ones((16, 4))
    low = t / 16.0
    high = np.where(np.arange(16)[:, None] % 2 == 0, 1.0, -1.0) * np.ones((16, 4))
    out = spectral_mask(low + high, axis=0, keep_fraction=0.25)
    removed_target = np.var(out - low) < 0.05 * np.var(high)
    preserved_offtarget = np.var(out - low) < 0.05 * np.var(low + high)
    return bool(removed_target and preserved_offtarget)


def _subspace_self_test() -> bool:
    basis = np.array([[1.0, 0.0, 0.0]])
    x = np.array([[4.0, 3.0, 2.0], [1.0, 5.0, 6.0]])
    out = subspace_project_out(x, basis)
    return bool(np.allclose(out[:, 0], 0.0) and np.allclose(out[:, 1:], x[:, 1:]))


def _zero_channels_self_test() -> bool:
    x = np.arange(12, dtype=float).reshape(3, 4)
    out = zero_channels(x, [2], axis=-1)
    others_ok = np.allclose(np.delete(out, 2, 1), np.delete(x, 2, 1))
    return bool(np.allclose(out[:, 2], 0.0) and others_ok)


PRIMITIVES: dict[str, Primitive] = {
    "spectral_mask": Primitive(
        "spectral_mask", "zero high-frequency bands along an axis", spectral_mask,
        _spectral_mask_self_test),
    "subspace_project_out": Primitive(
        "subspace_project_out", "project out named directions over the last axis",
        subspace_project_out, _subspace_self_test),
    "zero_channels": Primitive(
        "zero_channels", "zero named channels along an axis", zero_channels,
        _zero_channels_self_test),
}


def vetted_primitives(primitives: dict[str, Primitive] | None = None) -> dict[str, Primitive]:
    """Only the primitives whose specificity self-test passes. Fail-closed: a primitive that does
    not cleanly isolate is excluded, so the blue team never composes from a bad building block."""
    pool = PRIMITIVES if primitives is None else primitives
    return {name: p for name, p in pool.items() if p.self_test()}
