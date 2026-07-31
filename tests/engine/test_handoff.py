"""The discovery -> confirmation handoff. `build_dossier` extracts the OBJECTIVE classifier-input fields
a party other than the agent can read, and `classifier_triage` is the UNATTENDED gate: the claim-type
is the strictest gate consistent with the schema + signed catalogs (never the agent's label) and locked
per lineage, so the loop cannot shop for a gauntlet without a human sitting per maturation."""
from dataclasses import dataclass

from engine.handoff import (
    Dossier,
    accept_as_proposed,
    build_dossier,
    classifier_triage,
)

INCUMBENT_TASKS = ("tomato_temporal_mcq",)


@dataclass
class _Bundle:
    believed_claim: bool = True


@dataclass
class _Maturation:
    bundle: _Bundle


def _mat(believed=True):
    return _Maturation(bundle=_Bundle(believed_claim=believed))


# --- build_dossier extracts the classifier-input fields objectively -------------------------------

def test_dossier_extracts_the_form_keys():
    raw = {"claim": "the model beats the TOMATO incumbent", "claim_type": "capability",
           "task": "tomato_temporal_mcq", "baseline_rate": 0.25, "unrelated": "ignore me"}
    d = build_dossier(raw, _mat())
    assert d.form == {"task": "tomato_temporal_mcq", "baseline_rate": 0.25}
    assert "unrelated" not in d.form
    assert d.proposed_claim_type == "capability"


def test_dossier_form_carries_only_present_keys():
    d = build_dossier({"claim": "c", "claim_type": "effect", "dataset": "novel_task"}, _mat())
    assert d.form == {"dataset": "novel_task"}


# --- classifier_triage: unattended, the derivation decides ----------------------------------------

def test_classifier_triage_rejects_a_downgrade_on_an_incumbent_task():
    # agent pre-registers the weakest bar on a task that has a signed incumbent -> shelved, no box
    d = Dossier(claim="c", proposed_claim_type="qualitative_phenomenon", believed_claim=True,
                form={"task": "tomato_temporal_mcq", "baseline_rate": 0.25})
    decision = classifier_triage(incumbent_tasks=INCUMBENT_TASKS)(d)
    assert decision.accept is False


def test_classifier_triage_accepts_an_honest_capability_claim():
    d = Dossier(claim="c", proposed_claim_type="capability", believed_claim=True,
                form={"task": "tomato_temporal_mcq"})
    decision = classifier_triage(
        incumbent_tasks=INCUMBENT_TASKS, consequence_template_id="capability", seeds=(0, 1))(d)
    assert decision.accept is True
    assert decision.claim_type == "capability"
    assert decision.seeds == (0, 1)


def test_classifier_triage_commits_effect_off_an_incumbent_free_task():
    d = Dossier(claim="c", proposed_claim_type="effect", believed_claim=True,
                form={"task": "novel_motion_task"})
    decision = classifier_triage(incumbent_tasks=INCUMBENT_TASKS)(d)
    assert decision.accept is True and decision.claim_type == "effect"


def test_classifier_triage_enforces_the_lineage_lock():
    d = Dossier(claim="c", proposed_claim_type="effect", believed_claim=True,
                form={"task": "novel_motion_task"})
    triage = classifier_triage(incumbent_tasks=INCUMBENT_TASKS,
                               committed_type_for=lambda dossier: "capability")
    assert triage(d).accept is False


def test_accept_as_proposed_still_trusts_the_label_for_the_mock_spine():
    d = Dossier(claim="c", proposed_claim_type="effect", believed_claim=True, form={})
    assert accept_as_proposed(d).claim_type == "effect"
