"""Mechanism-AGNOSTIC ablation. The mechanism gate is a GENERAL causal test: the effect is present
with the mechanism and gone without it. WHICH mechanism gets removed comes from the hypothesis,
never hardcoded. Each registered ablation is the operation `score_ablated` applies to remove its
named mechanism; temporal-frequency band removal is ONE entry, not the only one, so an idea can name
any mechanism that has a registered ablation. The TASK the experiment scores on is read from the
CLAIM, so an idea can mature OUT of a default realm (SSv2 is only a default motion-rich test-bed,
never a fixed target). This module keeps the engine general per its source of truth, rather than
pinned to one domain-specific experiment.

The GPU tensor op that actually removes a mechanism ships on the cluster (in each Ablation's
`apply`, called by the backend scorer). On the Mac only the dispatch is exercised: the right
ablation is chosen from the schema, the right task from the claim, and both fail closed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ablation_primitives import spectral_mask
from .real_experiments import real_mechanism


class MechanismError(ValueError):
    """A hypothesis named no mechanism or no task, or named a mechanism with no registered
    ablation. Fail-closed: an unresolvable mechanism cannot get a passing mechanism measurement,
    so a claim cannot skip the causal test by omission."""


@dataclass(frozen=True)
class Ablation:
    """The operation that removes a named mechanism. `apply(model, box, untrained_init)` is what
    the cluster scorer calls to produce the mechanism-ablated model variant; on the Mac it is only
    dispatched, never invoked."""
    name: str
    description: str
    apply: Callable


_ABLATIONS: dict[str, Ablation] = {}


def register_ablation(ablation: Ablation) -> Ablation:
    """Register (and return) an ablation so a hypothesis naming its mechanism can be causal-tested.
    Returning it lets a module-level `X = register_ablation(Ablation(...))` bind a handle."""
    _ABLATIONS[ablation.name] = ablation
    return ablation


def resolve_ablation(mechanism: str) -> Ablation:
    if not mechanism:
        raise MechanismError("the hypothesis names no mechanism, so the ablation cannot be chosen")
    try:
        return _ABLATIONS[mechanism]
    except KeyError:
        raise MechanismError(
            f"no registered ablation for mechanism '{mechanism}'; register one or the hypothesis "
            f"cannot be mechanism-tested"
        ) from None


def task_from_claim(schema: dict) -> str:
    """The task comes from the CLAIM, not a default. An explicit `task` wins, else the `dataset`
    the hypothesis claims on. Missing both fails closed, so the experiment never silently defaults
    to a domain the idea did not name."""
    task = schema.get("task") or schema.get("dataset")
    if not task:
        raise MechanismError("the hypothesis names no task; the ablation task comes from the claim")
    return task


def build_mechanism_fn(*, score_task, specificity_check, alpha: float, resolve_ablation_fn=None):
    """Build the mechanism experiment for the `ExperimentSubstrate` seam. The returned
    `mechanism_fn(backend, schema)` reads the mechanism and task the hypothesis NAMES, resolves the
    ablation from the registry, scores the task full vs mechanism-ablated through the backend, and
    delegates the CI logic to `real_mechanism`. Nothing is domain-pinned: both the mechanism and the
    task come from the schema.

    `score_task(backend, task, ablation) -> per-item scores` (ablation=None -> the full model; an
    ablation with an `.apply` transform -> that mechanism removed). `specificity_check(backend,
    schema, task) -> bool`.

    `resolve_ablation_fn(mechanism, task) -> ablation` obtains the ablation. It defaults to the
    static registry (`resolve_ablation`), but the red/blue construction loop (spec 3c,
    `ablation_construction`) is injected here to make the ablation idea-agnostic and adversarially
    verified. A registry `Ablation` and a `ConstructedAblation` both expose `.apply`, so the
    `score_task` contract is uniform either way."""

    if resolve_ablation_fn is None:
        def resolve_ablation_fn(mechanism, task):
            return resolve_ablation(mechanism)

    def mechanism_fn(backend, schema):
        task = task_from_claim(schema)
        ablation = resolve_ablation_fn(schema.get("mechanism", ""), task)
        spec = specificity_check(backend, schema, task)
        return real_mechanism(
            score_full=lambda: score_task(backend, task, None),
            score_ablated=lambda: score_task(backend, task, ablation),
            specificity_ok=spec,
            alpha=alpha,
        )

    return mechanism_fn


def remove_temporal_frequency(frames, keep_fraction: float = 0.5):
    """Temporal-frequency ablation, now just the `spectral_mask` PRIMITIVE parameterized to the time
    axis (axis 0). Kept as a named convenience for the SSv2 experiment, with no privileged status:
    it is one parameterization of a general, domain-free op, not a special-cased ablation."""
    return spectral_mask(frames, axis=0, keep_fraction=keep_fraction)


TEMPORAL_FREQUENCY = register_ablation(Ablation(
    name="temporal_frequency",
    description="remove high temporal-frequency bands from the input clip (DFT along time)",
    apply=remove_temporal_frequency,
))
