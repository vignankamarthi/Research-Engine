"""Mechanism-AGNOSTIC ablation. The mechanism gate is a general causal test: the effect is present
with the mechanism and gone without it. WHICH mechanism gets removed comes from the HYPOTHESIS, not
a hardcoded temporal-DCT. WHICH task it scores on is read from the CLAIM, so an idea can mature out
of a default realm (SSv2 is only a default motion-rich test-bed). These tests prove the dispatch:
the named ablation is selected from a registry, the task from the claim, and both fail closed when
the hypothesis names neither. The GPU op ships on the cluster; here only the dispatch is tested."""
from datetime import date

import numpy as np
import pytest

from engine.ablations import (
    TEMPORAL_FREQUENCY,
    Ablation,
    MechanismError,
    build_mechanism_fn,
    register_ablation,
    resolve_ablation,
    task_from_claim,
)


def test_temporal_frequency_is_one_registered_ablation():
    assert resolve_ablation("temporal_frequency") is TEMPORAL_FREQUENCY


def test_unknown_mechanism_fails_closed():
    # a hypothesis naming a mechanism with no registered ablation cannot get a mechanism measurement
    with pytest.raises(MechanismError):
        resolve_ablation("phlogiston_flux")


def test_empty_mechanism_fails_closed():
    with pytest.raises(MechanismError):
        resolve_ablation("")


def test_registry_is_general_not_temporal_only():
    # register a SECOND, unrelated mechanism: temporal-frequency is one entry, never special-cased
    spatial = register_ablation(
        Ablation("spatial_attention", "remove spatial attn", lambda *a: None))
    assert resolve_ablation("spatial_attention") is spatial
    assert resolve_ablation("temporal_frequency") is TEMPORAL_FREQUENCY


def test_task_read_from_explicit_task_then_dataset():
    assert task_from_claim({"task": "epic_kitchens"}) == "epic_kitchens"
    assert task_from_claim({"dataset": "ssv2"}) == "ssv2"  # falls back to the claimed dataset
    assert task_from_claim({"task": "diving48", "dataset": "ssv2"}) == "diving48"  # explicit


def test_task_missing_fails_closed():
    with pytest.raises(MechanismError):
        task_from_claim({"claim": "c"})


def test_build_mechanism_fn_dispatches_the_named_ablation():
    # score_full is scored with ablation=None, score_ablated with the schema-named Ablation
    calls = []

    def score_task(backend, task, ablation):
        calls.append((task, ablation))
        return np.full(50, 0.10 if ablation is None else 0.0)

    fn = build_mechanism_fn(
        score_task=score_task, specificity_check=lambda b, s, t: True, alpha=0.05)
    contrast_lo, spec = fn(backend=None, schema={"mechanism": "temporal_frequency", "dataset": "ssv2"})
    assert calls[0] == ("ssv2", None)  # full model first
    assert calls[1][0] == "ssv2" and calls[1][1] is TEMPORAL_FREQUENCY  # then the named ablation
    assert contrast_lo > 0.05 and spec is True  # paired (0.10 - 0.0) contrast clears the MIE


def test_task_is_read_from_the_claim_not_a_default():
    seen = {}

    def score_task(backend, task, ablation):
        seen["task"] = task
        return np.full(20, 0.10 if ablation is None else 0.0)

    fn = build_mechanism_fn(
        score_task=score_task, specificity_check=lambda b, s, t: True, alpha=0.05)
    fn(backend=None, schema={"mechanism": "temporal_frequency", "task": "epic_kitchens"})
    assert seen["task"] == "epic_kitchens"  # matured out of the SSv2 default realm


def test_mechanism_fn_fails_closed_when_no_mechanism_named():
    # a hypothesis naming NO mechanism cannot get a passing mechanism measurement: the gate fails
    # CLOSED with a -inf contrast. It no longer RAISES and crashes the campaign (SPEC 3c) -- the
    # referee reads the failing contrast and returns FAILED, and the run continues to the next idea.
    fn = build_mechanism_fn(
        score_task=lambda b, t, a: np.zeros(1), specificity_check=lambda b, s, t: True, alpha=0.05)
    contrast_lo, spec = fn(backend=None, schema={"dataset": "ssv2"})  # names a task but no mechanism
    assert contrast_lo == float("-inf") and spec is False


def test_built_mechanism_fn_composes_into_the_substrate():
    from engine.substrate import ExperimentSubstrate

    fn = build_mechanism_fn(
        score_task=lambda b, t, a: np.full(50, 0.10 if a is None else 0.0),
        specificity_check=lambda b, s, t: True, alpha=0.05)
    sub = ExperimentSubstrate(
        g0_fn=lambda backend, schema: True,
        mechanism_fn=fn,
        novelty_fn=lambda schema: (False, ["a"], True),
        backbone_fn=lambda schema: (date(2023, 1, 1), True),
        consequence_fn=lambda schema: (True, True))
    b = sub.produce(
        {"claim": "c", "mechanism": "temporal_frequency", "dataset": "ssv2"},
        backend=None, believed_claim=True)
    assert b.mech_contrast_lo > 0.05 and b.specificity_ok is True


def test_mechanism_fn_fails_closed_on_no_clean_ablation():
    # SPEC 3c: a non-converging red/blue construction (NoCleanAblation) FAILS the mechanism gate
    # closed with a -inf contrast, it never propagates and crashes the campaign.
    from engine.ablation_construction import NoCleanAblation

    def raising(mechanism, task):
        raise NoCleanAblation("no clean ablation")

    fn = build_mechanism_fn(score_task=lambda b, t, a: np.full(10, 0.5),
                            specificity_check=lambda b, s, t: True, alpha=0.05,
                            resolve_ablation_fn=raising)
    contrast_lo, spec = fn(backend=None, schema={"mechanism": "x", "dataset": "ssv2"})
    assert contrast_lo == float("-inf") and spec is False
