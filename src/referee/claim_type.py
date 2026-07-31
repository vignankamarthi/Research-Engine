"""Frozen, referee-side claim-type classifier: the STRICTEST magnitude gate consistent with a
hypothesis, computed in the trusted process from the signed catalogs, never a label the framing agent
chooses. This IS the schema-normal-form property the spec states (RESEARCH-LOOP-SPEC line 416: the
derivation yields the magnitude gate STRICTEST consistent with the schema, mismatch -> INELIGIBLE) and
the audit found unenforced (`derive()` trusted `schema.claim_type`). It closes agent-controlled
gauntlet selection: the agent PRE-REGISTERS a type, the classifier recomputes the strictest consistent
gate, and REJECTS any disagreement. With the one-type-per-lineage lock, an idea can never shop for an
easier bar across maturations.

The distinguishing signal is WHICH externally-anchored threshold the claim is judged against (spec
lines 249-251; runner `_missing_threshold`). The mechanism gate is UNIVERSAL (every positive claim
must pass it), so a named mechanism does NOT distinguish the types and is not a signal here.

  - capability          : a signed per-task incumbent exists  -> separation over the incumbent
  - law_shape           : the claim carries predicted + observed + tolerance (a law series)
  - qualitative_phenomenon : the claim carries a pre-registered null/baseline rate
  - effect              : none of the above; the bar is the always-present MIE

STRICTEST-CONSISTENT ordering: a task with a signed incumbent forces CAPABILITY (the strict,
un-dodgeable performance bar), so a performance claim cannot drop to the MIE (effect) or a null
(phenomenon) to escape a published SOTA. Only on a task with NO signed incumbent are effect / phenomenon
/ law reachable, chosen by the claim's own structure. This is why a task's available claim-types are
largely fixed by its signed catalogs, and why exercising all four needs diverse datasets, not one."""
from __future__ import annotations


# The fields the classifier reads off a matured hypothesis. The dossier-builder (a party other than
# the framing agent) extracts exactly these into Dossier.form. `task`/`dataset` decide capability (via
# the signed incumbent catalog); the law triple and `baseline_rate` decide law_shape / phenomenon.
FORM_KEYS = ("task", "dataset", "baseline_rate", "law_predicted", "law_observed", "law_tol")


class ClaimTypeMismatch(ValueError):
    """The agent's pre-registered claim-type is not the strictest gate consistent with the schema, or
    a lineage already committed to one type is re-maturing under a different one. Fail-closed: the
    handoff is REJECTED (spec: mismatch -> INELIGIBLE), no box spent, so a claim can never reach a
    weaker gauntlet than its form and the signed catalogs warrant."""


def _present(schema: dict, key: str) -> bool:
    """A field COUNTS only if it carries a real value. An empty string, None, or a missing key is
    absent, so an agent cannot conjure (or dodge) a threshold with a blank field it never filled."""
    v = schema.get(key, None)
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    return True


def _task_of(schema: dict) -> str:
    return str(schema.get("task") or schema.get("dataset") or "").strip().lower()


def _has_law_series(schema: dict) -> bool:
    return (_present(schema, "law_predicted") and _present(schema, "law_observed")
            and _present(schema, "law_tol"))


def strictest_consistent(schema: dict, *, incumbent_tasks=()) -> str:
    """The strictest magnitude gate consistent with the schema and the signed catalogs. `incumbent_tasks`
    is the set of task keys the signed incumbent catalog has an entry for; a claim on such a task is a
    CAPABILITY claim and cannot be graded under a weaker bar. Off an incumbent-bearing task the type
    follows the claim's own structure (a law series -> law_shape, a null/baseline -> phenomenon, else
    the MIE-anchored effect)."""
    incumbent_tasks = {str(t).strip().lower() for t in incumbent_tasks}
    if _task_of(schema) in incumbent_tasks:
        return "capability"          # a signed published incumbent exists -> the strict performance bar
    if _has_law_series(schema):
        return "law_shape"
    if _present(schema, "baseline_rate"):
        return "qualitative_phenomenon"
    return "effect"                  # the default: bar is the always-present MIE


def commit_claim_type(schema: dict, proposed: str, *, incumbent_tasks=(), locked: str | None = None) -> str:
    """Validate and return the claim-type for a matured hypothesis. The strictest-consistent gate
    decides; `proposed` (the agent's advisory pre-registration) is only cross-checked, so a lie or a
    confusion is caught rather than honoured. A `locked` type (this lineage already committed at an
    earlier maturation) must equal the computed gate, or the re-maturation is a gauntlet-shopping
    attempt. Fail-closed on every mismatch: the caller shelves the handoff without spending a box."""
    gate = strictest_consistent(schema, incumbent_tasks=incumbent_tasks)
    proposed_canon = (proposed or "").strip().lower()
    if proposed_canon and proposed_canon != gate:
        raise ClaimTypeMismatch(
            f"agent pre-registered claim_type {proposed_canon!r} but the strictest gate consistent "
            f"with the schema + signed catalogs is {gate!r}; the gate follows the derivation, not the "
            f"agent's label (spec: mismatch -> INELIGIBLE)"
        )
    if locked is not None and locked != gate:
        raise ClaimTypeMismatch(
            f"lineage already committed to claim_type {locked!r}; it cannot re-mature under {gate!r} "
            f"(one-type-per-lineage lock, anti gauntlet-shopping)"
        )
    return gate
