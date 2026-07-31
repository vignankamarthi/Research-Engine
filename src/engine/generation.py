"""Grounded, vein-diverse, cumulative, blind generation (PLAN 57 b/c/d/f). This is the stage between
the blind scout (scout_isolation) and the campaign spine. It:

  (b) makes the VEIN SET the diversity axis: stamps each candidate with its vein + mechanism-family,
      enforces per-wave distinctness, and flags a campaign that concentrates in one family;
  (c) conditions the scout's context on campaign state (prior results, surprises, live reviewer
      objections, the framing draft, the negative-bank dead-end exclusion via `is_dead_end`);
  (d) constrains candidates to the WIRED claim-type envelope, dropping+logging a malformed one
      without ever going fatal;
  (f) ranks by quality plus advisory importance (never a silent kill), binds the high-patience slot to
      a generative-vein candidate, and gives `importance_penalty` a consumer.

`choose_mode` is the correctly-named depth/breadth chooser (the old `choose_vein` is a misnomer: it
returns DEPTH/BREADTH, not a vein). `generate_wave` composes the stage into one orchestrator hook."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .grounding import ground_wave
from .steering import choose_vein, is_dead_end

# The VEIN SET (SPEC 3): six DERIVATIVE veins mined from the frontier literature + three GENERATIVE
# veins. Mirrors the orchestrator's VEINS so the arm space + the diversity axis are one and the same.
DERIVATIVE_VEINS = ("limitations", "future_work", "contradictions", "ablation_surprises",
                    "assumption_relaxation", "method_transplant")
GENERATIVE_VEINS = ("problem_restatement", "cross_domain_analogy", "new_framework_assumption")
VEINS = DERIVATIVE_VEINS + GENERATIVE_VEINS

# Coarse mechanism families, matched in priority order against the candidate's mechanism/measure/claim
# text. The family is the axis the concentration check watches, so a campaign cannot quietly collapse
# onto one mechanism (the temporal monoculture the audit flagged).
_FAMILY_KEYWORDS = (
    ("temporal", ("temporal", "motion", "frequency", "rope", "time", "dynamic")),
    ("spatial", ("spatial", "appearance", "texture", "static", "pixel")),
    ("attention", ("attention", "token", "head", "context length", "window")),
    ("scaling", ("scale", "scaling", "law", "capacity", "parameter")),
    ("data", ("dataset", "augmentation", "sampling", "curriculum", "label")),
    ("objective", ("loss", "objective", "contrastive", "reconstruction", "distillation")),
)

_REQUIRED_KEYS = ("claim", "claim_type", "backbone", "dataset", "scale", "measure")


def choose_mode(bandit_pick, depth_count, breadth_count, policy):
    """The depth/breadth chooser, correctly named (the steering helper is still called `choose_vein`,
    a misnomer: it returns DEPTH/BREADTH, not a vein). Kept as the single canonical entry point so
    call sites read honestly; the underlying floor/cap logic is unchanged."""
    return choose_vein(bandit_pick, depth_count, breadth_count, policy)


def mechanism_family(candidate: dict) -> str:
    """Classify a candidate into a coarse mechanism family, defaulting to 'other'."""
    text = " ".join(str(candidate.get(k, "")) for k in ("mechanism", "measure", "claim")).lower()
    for family, keywords in _FAMILY_KEYWORDS:
        if any(kw in text for kw in keywords):
            return family
    return "other"


def stamp_candidate(candidate: dict, vein: str) -> dict:
    """Stamp a candidate with its vein and derived mechanism-family (the two diversity keys)."""
    return {**candidate, "vein": vein, "mechanism_family": mechanism_family(candidate)}


def enforce_distinct(candidates):
    """Per-wave distinctness: keep the first candidate for each (vein, mechanism-family) signature,
    drop later collisions so one wave cannot be N copies of the same mechanism. Returns (kept,
    dropped); non-fatal."""
    seen, kept, dropped = set(), [], []
    for c in candidates:
        key = (c.get("vein"), c.get("mechanism_family"))
        if key in seen:
            dropped.append(c)
        else:
            seen.add(key)
            kept.append(c)
    return kept, dropped


@dataclass(frozen=True)
class ConcentrationVerdict:
    page: bool
    dominant_family: str
    fraction: float


def concentration_check(maturations, *, min_count: int = 3,
                        threshold: float = 0.6) -> ConcentrationVerdict:
    """Page when the matured hypotheses cluster in one mechanism family past `threshold` (once at
    least `min_count` have matured, so an early small wave is not flagged). The orchestrator wires
    the page into the escalation channel; this only computes the verdict."""
    fams = [m.get("mechanism_family", "other") for m in maturations]
    if len(fams) < min_count:
        return ConcentrationVerdict(False, "", 0.0)
    family, n = Counter(fams).most_common(1)[0]
    fraction = n / len(fams)
    return ConcentrationVerdict(fraction > threshold, family, fraction)


def build_generation_context(*, vein, mode, negative_bank=(), prior_results=(), surprises=(),
                             reviewer_objections=(), framing_draft="", extra=None) -> dict:
    """Assemble the scout's context, conditioned on campaign state (PLAN 57c). The negative bank is
    sorted for determinism; the scout is told to avoid it (dead-end exclusion is also enforced
    structurally by `exclude_dead_ends` after proposal)."""
    ctx = {
        "vein": vein,
        "mode": mode,
        "negative_bank": sorted(negative_bank),
        "prior_results": list(prior_results),
        "surprises": list(surprises),
        "reviewer_objections": list(reviewer_objections),
        "framing_draft": framing_draft,
    }
    if extra:
        ctx.update(extra)
    return ctx


def default_lineage_of(candidate: dict) -> str:
    """Trusted-process lineage key for a candidate (referee normal form). Injected/overridable so
    the generation tests need no full referee schema."""
    from referee import lineage_key, normalize_schema
    return lineage_key(normalize_schema(candidate))


def exclude_dead_ends(candidates, negative_bank, *, lineage_of=default_lineage_of):
    """Drop candidates whose lineage is already a banked dead end (PLAN 57c, via `is_dead_end`).
    Returns (kept, dropped); non-fatal."""
    kept, dropped = [], []
    for c in candidates:
        if is_dead_end(lineage_of(c), negative_bank):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


def _malformed_reason(c, wired_claim_types) -> str:
    if not isinstance(c, dict):
        return "not a dict"
    missing = [k for k in _REQUIRED_KEYS if not c.get(k)]
    if missing:
        return f"missing keys: {missing}"
    if c.get("claim_type") not in wired_claim_types:
        return f"claim_type {c.get('claim_type')!r} not in wired envelope"
    return ""


def constrain_candidates(candidates, wired_claim_types, *, log=None):
    """Keep only candidates that fit the WIRED claim-type envelope (PLAN 57d). A malformed candidate
    is DROPPED and logged, never fatal, so a single bad JSON reply cannot sink the wave."""
    kept = []
    for c in candidates:
        reason = _malformed_reason(c, wired_claim_types)
        if reason:
            if log is not None:
                log.append({"candidate": c, "reason": reason})
            continue
        kept.append(c)
    return kept


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def apply_importance_penalty(quality: float, importance_penalty: float) -> float:
    """The consumer for the significance-adversary's advisory `importance_penalty`: it LOWERS a
    candidate's effective rank but never removes it (importance is advisory, never a silent gate)."""
    return quality * (1.0 - _clamp01(importance_penalty))


@dataclass
class RankedCandidate:
    candidate: dict
    quality: float
    importance_penalty: float
    score: float


def rank_candidates(scored) -> list:
    """Rank candidates by quality tempered by advisory importance. `scored` is an iterable of
    (candidate, quality, importance_penalty). NEVER drops a candidate (a silent kill is forbidden);
    it only reorders, so an ambitious idea still reaches the human at triage."""
    ranked = [RankedCandidate(c, q, ip, apply_importance_penalty(q, ip)) for c, q, ip in scored]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def bind_high_patience_slot(ranked, *, generative_veins=GENERATIVE_VEINS):
    """Bind the high-patience, high-variance slot to the top-ranked GENERATIVE-vein candidate
    (ambition protection, SPEC 3). Returns None if the wave produced no generative candidate."""
    for r in ranked:
        if r.candidate.get("vein") in generative_veins:
            return r
    return None


def human_seed(claim: str, **fields) -> dict:
    """The human-seed entry point: a candidate the human injects directly, stamped so its origin is
    auditable. It flows through the same envelope, grounding, and rank stages as a scout candidate."""
    seed = {"claim": claim, "origin": "human_seed"}
    seed.update(fields)
    return seed


@dataclass
class WaveResult:
    ranked: list = field(default_factory=list)
    high_patience: object = None
    grounding: object = None
    malformed: list = field(default_factory=list)
    dead_ends: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)


def generate_wave(*, scout, context, wired_claim_types, resolver, negative_bank=(),
                  lineage_of=default_lineage_of, quality_of=None, importance_of=None,
                  distinct=True) -> WaveResult:
    """The composed generation stage, the single orchestrator hook. Blind scout proposes -> stamp
    (b) -> claim-type envelope (d) -> dead-end exclusion (c) -> grounding (e) -> per-wave distinctness
    (b) -> rank + high-patience bind (f). Every drop is recorded and non-fatal; only the survivors
    reach the campaign. `quality_of` / `importance_of` are injected scorers (importance is the
    significance-adversary's advisory penalty)."""
    raw = scout.propose(context)
    vein = context.get("vein")
    stamped = [stamp_candidate(c, vein) for c in raw if isinstance(c, dict)]

    malformed_log: list = []
    valid = constrain_candidates(stamped, wired_claim_types, log=malformed_log)
    alive, dead = exclude_dead_ends(valid, negative_bank, lineage_of=lineage_of)

    grec = ground_wave(alive, resolver)
    grounded = grec.grounded
    duplicates: list = []
    if distinct:
        grounded, duplicates = enforce_distinct(grounded)

    quality_of = quality_of or (lambda c: 1.0)
    importance_of = importance_of or (lambda c: 0.0)
    ranked = rank_candidates((c, quality_of(c), importance_of(c)) for c in grounded)

    return WaveResult(ranked=ranked, high_patience=bind_high_patience_slot(ranked),
                      grounding=grec, malformed=malformed_log, dead_ends=dead,
                      duplicates=duplicates)
