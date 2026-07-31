"""The frozen referee-side claim-type classifier: the STRICTEST magnitude gate consistent with the
schema + the signed catalogs (RESEARCH-LOOP-SPEC line 416), not the agent's chosen label. The
distinguishing signal is WHICH externally-anchored threshold judges the claim (spec 249-251); the
mechanism gate is UNIVERSAL and never a signal. These tests pin the anti-shopping guarantee: a task
with a signed incumbent forces CAPABILITY (un-dodgeable), a pre-registered type that disagrees with the
derivation is REJECTED, and a lineage locked to one type cannot re-mature under another."""
import pytest

from referee.claim_type import (
    ClaimTypeMismatch,
    commit_claim_type,
    strictest_consistent,
)

# the signed incumbent catalog covers this task; a performance claim on it is capability
INCUMBENT_TASKS = ("tomato_temporal_mcq",)


# --- strictest_consistent: the threshold decides the type -----------------------------------------

def test_signed_incumbent_task_is_capability():
    assert strictest_consistent(
        {"task": "tomato_temporal_mcq"}, incumbent_tasks=INCUMBENT_TASKS) == "capability"


def test_law_series_off_an_incumbent_task_is_law_shape():
    s = {"task": "kinetics_scaling", "law_predicted": 0.5, "law_observed": 0.48, "law_tol": 0.05}
    assert strictest_consistent(s, incumbent_tasks=INCUMBENT_TASKS) == "law_shape"


def test_baseline_off_an_incumbent_task_is_qualitative_phenomenon():
    s = {"task": "some_probe", "baseline_rate": 0.25}
    assert strictest_consistent(s, incumbent_tasks=INCUMBENT_TASKS) == "qualitative_phenomenon"


def test_no_threshold_no_incumbent_is_effect():
    # the default: the bar is the always-present MIE, magnitude is the trained-minus-untrained contrast
    assert strictest_consistent({"task": "novel_task"}, incumbent_tasks=INCUMBENT_TASKS) == "effect"


# --- the mechanism is UNIVERSAL, never a type signal ----------------------------------------------

def test_a_named_mechanism_does_not_change_the_type():
    # every positive claim needs a mechanism (runner Gate 5); naming one cannot make a claim "effect"
    with_mech = {"task": "tomato_temporal_mcq", "mechanism": "temporal_frequency"}
    assert strictest_consistent(with_mech, incumbent_tasks=INCUMBENT_TASKS) == "capability"


# --- strictest-consistent: a signed incumbent cannot be dodged ------------------------------------

def test_incumbent_task_forces_capability_over_a_planted_baseline():
    # the shopping attack: add a baseline to an incumbent-bearing task to escape the SOTA as a mere
    # phenomenon. The signed incumbent dominates: still capability.
    s = {"task": "tomato_temporal_mcq", "baseline_rate": 0.25}
    assert strictest_consistent(s, incumbent_tasks=INCUMBENT_TASKS) == "capability"


def test_partial_law_series_is_not_a_law():
    s = {"task": "novel_task", "law_predicted": 0.5, "law_observed": 0.48}  # no tol
    assert strictest_consistent(s, incumbent_tasks=INCUMBENT_TASKS) == "effect"


def test_dataset_falls_back_for_the_task_key():
    assert strictest_consistent(
        {"dataset": "tomato_temporal_mcq"}, incumbent_tasks=INCUMBENT_TASKS) == "capability"


# --- commit_claim_type: the derivation decides, the proposal is only cross-checked -----------------

def test_matching_proposal_commits_the_derived_gate():
    assert commit_claim_type(
        {"task": "tomato_temporal_mcq"}, "capability", incumbent_tasks=INCUMBENT_TASKS) == "capability"


def test_empty_proposal_lets_the_derivation_decide():
    assert commit_claim_type({"task": "novel_task"}, "", incumbent_tasks=INCUMBENT_TASKS) == "effect"


def test_cannot_downgrade_an_incumbent_task_to_a_phenomenon():
    # the core shopping attempt: pre-register the weakest bar on a task that has a signed incumbent
    with pytest.raises(ClaimTypeMismatch):
        commit_claim_type(
            {"task": "tomato_temporal_mcq"}, "qualitative_phenomenon", incumbent_tasks=INCUMBENT_TASKS)


def test_cannot_relabel_an_incumbent_task_as_effect():
    with pytest.raises(ClaimTypeMismatch):
        commit_claim_type(
            {"task": "tomato_temporal_mcq"}, "effect", incumbent_tasks=INCUMBENT_TASKS)


# --- the one-type-per-lineage lock ----------------------------------------------------------------

def test_lock_matching_derived_type_is_allowed():
    assert commit_claim_type(
        {"task": "tomato_temporal_mcq"}, "capability",
        incumbent_tasks=INCUMBENT_TASKS, locked="capability") == "capability"


def test_lock_conflicting_with_derived_type_is_rejected():
    # a lineage committed to `capability` cannot re-mature as `effect` on a now-incumbent-free task
    with pytest.raises(ClaimTypeMismatch):
        commit_claim_type({"task": "novel_task"}, "effect",
                          incumbent_tasks=INCUMBENT_TASKS, locked="capability")
