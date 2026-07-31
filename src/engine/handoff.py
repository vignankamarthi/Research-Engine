"""The discovery -> confirmation HANDOFF. The human triages a NEUTRAL dossier (assembled by a party
OTHER than the framing agent, so the handoff is not the storyteller judging its own story), PICKS the
claim-type, commits the templated consequence, and freezes the seeds. The committed claim-type drives
the confirmation Schema, so the loop NEVER selects its own gauntlet (the agent's proposed type is
advisory only). This closes the audit finding that discovery could pick the gate that judges it."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dossier:
    """The neutral decision packet triaged at the handoff. Carries the raw claim, the agent's ADVISORY
    proposed claim-type, its believed value, and the OBJECTIVE form-signals the frozen referee-side
    classifier reads (`form`: mechanism / incumbent / law fields), plus (enriched at Milestone 8 step
    58) the k closest priors verbatim, the significance-adversary's strongest case, the MIE percentile,
    the commensurability line, and the embedding-nearest negative-bank neighbors for the SEMANTIC
    lineage-collision review (not a spent-key equality check a paraphrase games). The dossier is built
    by a party OTHER than the framing agent, so the `form` signals are extracted objectively, never
    taken from the agent's narrative."""
    claim: str
    proposed_claim_type: str            # ADVISORY only; the committed type is authoritative
    believed_claim: bool
    form: dict = field(default_factory=dict)  # objective type-signals for the frozen classifier
    k_nearest_priors: tuple = ()
    nearest_bank_neighbors: tuple = ()  # for the semantic lineage-collision check
    significance_case: str = ""         # the significance-adversary's strongest incremental case (58)
    mie_percentile: float | None = None
    commensurability: str = ""


@dataclass(frozen=True)
class TriageDecision:
    """The human's FROZEN pre-registration at the handoff. The human PICKS the claim-type AND the
    prior-claim flag (both select the gauntlet, so neither may come from the agent), commits the
    templated consequence, and freezes the seeds. accept=False shelves it without spending a box.
    Every field is threaded into the committed schema in `run_campaign`, so none is a frozen-looking
    field that nothing reads."""
    accept: bool
    claim_type: str = ""
    prior_claim: bool = False
    consequence_template_id: str = ""
    seeds: tuple = ()

    def __post_init__(self):
        # an accepting decision that names no claim-type would crash normalize_schema; the human
        # MUST pick the type, so this is a triage-completeness error, caught here not mid-campaign.
        if self.accept and not self.claim_type:
            raise ValueError("an accepting TriageDecision must name a claim_type (the human picks it)")


def build_dossier(schema_raw: dict, maturation, *, k_nearest_priors=(),
                  nearest_bank_neighbors=(), significance_case: str = "") -> Dossier:
    """Assemble the neutral dossier from the matured proposal, by a party other than the framing
    agent. The proposed claim-type is carried ADVISORY (triage recomputes it from the form). The
    OBJECTIVE form-signals the classifier reads are extracted straight off the raw schema. The
    significance-adversary's strongest incremental case and the embedding-nearest negative-bank
    neighbors are folded in so triage rests on evidence, not the storyteller's story alone."""
    from referee.claim_type import FORM_KEYS
    return Dossier(
        claim=str(schema_raw.get("claim", "")),
        proposed_claim_type=str(schema_raw.get("claim_type", "")),
        believed_claim=bool(maturation.bundle.believed_claim),
        form={k: schema_raw[k] for k in FORM_KEYS if k in schema_raw},
        k_nearest_priors=tuple(k_nearest_priors),
        nearest_bank_neighbors=tuple(nearest_bank_neighbors),
        significance_case=str(significance_case),
    )


def accept_as_proposed(dossier: Dossier) -> TriageDecision:
    """The default triage for tests and the Mac spine: accept and commit the proposal's ADVISORY
    type unchanged. Not for a real campaign -- it trusts the agent's label, which is the very thing
    the frozen classifier (`classifier_triage`) exists to stop."""
    return TriageDecision(accept=True, claim_type=dossier.proposed_claim_type)


def classifier_triage(*, incumbent_tasks=(), consequence_template_id: str = "", seeds: tuple = (),
                      prior_claim: bool = False, committed_type_for=None):
    """Build the UNATTENDED handoff: a triage callback that assigns the claim-type from the FROZEN
    referee-side classifier (the STRICTEST gate consistent with the schema + signed catalogs, not the
    agent's label) and enforces the one-type-per-lineage lock. No human sits per maturation, yet the
    loop still cannot pick its own gauntlet -- the gate is recomputed from the derivation and locked
    per lineage, so a claim can never shop for an easier bar. The single human scientific gate moves
    to the final go/no-go at submit.

    `incumbent_tasks` is the set of task keys the signed incumbent catalog covers; a claim on such a
    task is forced to CAPABILITY and cannot drop to a weaker bar. `committed_type_for(dossier) -> str |
    None` reports the type a lineage already committed to earlier (None if fresh), so a re-maturation
    under a different type is rejected. The lock is optional -- the semantic negative-bank already
    blocks re-proposing a banked idea, so it is a belt-and-suspenders second guard against a paraphrase
    slipping through under a new type.

    A mismatch (the derivation disagrees with the agent's pre-registration, or with the lineage lock)
    FAILS CLOSED: the handoff is shelved with accept=False and no box is spent (spec: mismatch ->
    INELIGIBLE)."""
    from referee.claim_type import ClaimTypeMismatch, commit_claim_type

    def triage(dossier: Dossier) -> TriageDecision:
        locked = committed_type_for(dossier) if committed_type_for is not None else None
        try:
            claim_type = commit_claim_type(
                dossier.form, dossier.proposed_claim_type,
                incumbent_tasks=incumbent_tasks, locked=locked)
        except ClaimTypeMismatch:
            return TriageDecision(accept=False)  # fail-closed: shelve, spend no box
        return TriageDecision(
            accept=True, claim_type=claim_type, prior_claim=prior_claim,
            consequence_template_id=consequence_template_id, seeds=seeds)

    return triage
