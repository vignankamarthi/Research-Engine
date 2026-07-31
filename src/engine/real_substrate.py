"""Assemble a REAL ExperimentSubstrate from the signed config, the loaded signed catalogs, and the
cluster-side deps. This is the wiring that turns the tested seams into a live substrate. Every
signed value is resolved through a digest-verified catalog, so the substrate MEASURES from anchored
references and the agent authors nothing the referee gates on.

The substrate operates on the RAW hypothesis dict (the full proposal), because that is where the
mechanism to ablate and the claimed held-out value live. The referee keeps the normalized Schema for
lineage and control derivation; the two representations are used for their two different jobs.

The heavy deps are INJECTED so the wiring is Mac-testable: `score_task` scores through the backend
(HFBackend on the cluster), `novelty_audit` queries the research MCPs, `g0_pipeline` runs the
planted-effect detection path, and `held_out_check` / `membership_check` are closures bound by
the backend (so the held-out consequence experiment scores through the real model)."""
from __future__ import annotations

import functools

from referee.catalog import resolve_incumbent, resolve_mie

from .ablations import build_mechanism_fn
from .real_experiments import real_g0, real_novelty, resolve_consequence
from .substrate import ExperimentSubstrate


class StubSubstrateError(RuntimeError):
    """A REAL campaign was assembled with an unmarked (stub) substrate measurement. Raised by the
    no-stub preflight so a constant stub can never silently satisfy a validity gate."""


def measured(fn, *, name: str):
    """Tag a substrate measurement callable as a REAL executed measurement. The no-stub preflight
    (`require_measured=True`) refuses a real campaign whose measurements are not tagged, so an
    accidental constant stub (`lambda ...: True`) cannot silently satisfy a validity gate. Wrapping
    a genuine constant in `measured` would be an explicit lie, a far higher bar than a forgotten
    stub, and the per-Bundle-field executed-not-fabricated provenance still catches that at the real
    run. Wraps rather than setattr so any callable (lambda, partial, method) can carry the tag."""
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)
    _wrapped._executed_measurement = name
    return _wrapped


def is_measured(fn) -> bool:
    return getattr(fn, "_executed_measurement", None) is not None


# The measurement callables a REAL run must not stub. Each stands in a validity gate the referee
# reads (the untrained FLOOR arm is scored through `score_task` and is checked separately).
_REAL_RUN_MEASUREMENTS = ("score_task", "novelty_audit", "g0_pipeline",
                          "specificity_check", "membership_check", "held_out_check")


def _task_of(schema: dict) -> str:
    """The task key, read from the claim. Must match the catalogs' task keys."""
    return schema.get("task") or schema["dataset"]


def assemble_substrate(*, config, consequence_catalog, incumbent_catalog, mie_catalog,
                       score_task, novelty_audit, g0_pipeline, specificity_check,
                       membership_check, held_out_check, backbone_cutoff, g0_rng,
                       resolve_ablation_fn=None, require_measured=False):
    """Wire the real callables into an ExperimentSubstrate. `config` is the SIGNED gate config (its
    catalog digests pin the three catalogs). The returned substrate's `.produce(raw_schema, backend,
    believed_claim=...)` yields the referee's Bundle, with the per-task MIE resolved onto bundle.mie
    and every catalog verified against the signed digest at resolve time.

    `require_measured=True` is the no-stub preflight for a REAL run: every measurement callable must
    be wrapped in `measured(...)`, so an accidental constant stub cannot silently pass a gate. The
    Mac test suite leaves it False (mocks are the point there); the real driver sets it True."""
    if require_measured:
        deps = dict(score_task=score_task, novelty_audit=novelty_audit, g0_pipeline=g0_pipeline,
                    specificity_check=specificity_check, membership_check=membership_check,
                    held_out_check=held_out_check)
        unmarked = sorted(n for n in _REAL_RUN_MEASUREMENTS if not is_measured(deps[n]))
        if unmarked:
            raise StubSubstrateError(
                f"real run requires executed measurements; unmarked stubs: {unmarked}")
    alpha, mde, floor = config.alpha, config.mde, config.mie_floor

    def _mie(schema):
        return resolve_mie(_task_of(schema), mie_catalog, config.mie_distribution_digest,
                           fallback=floor, mde=mde)

    def g0_fn(backend, schema):
        return real_g0(g0_pipeline, mde=mde, alpha=alpha, rng=g0_rng)

    mechanism_fn = build_mechanism_fn(
        score_task=score_task, specificity_check=specificity_check, alpha=alpha,
        resolve_ablation_fn=resolve_ablation_fn)

    def novelty_fn(schema):
        # advance_argued is FAIL-CLOSED inside real_novelty and comes from the audit party's return,
        # never from schema.get("advance_argued") (which the agent authors).
        return real_novelty(schema, audit_fn=novelty_audit)

    def magnitude_fn(schema):
        # The per-type magnitude threshold the referee compares the box score against. A CAPABILITY
        # claim separates above the SIGNED incumbent's held-out rate, resolved here from the pinned
        # catalog (never authored by the agent). Phenomenon (a signed null rate) and law_shape (a
        # scale sweep) supply their own thresholds when those runs exist; absent, they stay None and
        # the runner reads the claim INELIGIBLE rather than passing it on a missing bar.
        out = {}
        if schema.get("claim_type") == "capability":
            out["incumbent_rate"] = resolve_incumbent(
                _task_of(schema), incumbent_catalog, config.incumbent_catalog_digest)
        return out

    def backbone_fn(schema):
        return backbone_cutoff, bool(membership_check(schema))

    def consequence_fn(schema):
        task = _task_of(schema)
        # the consequence experiment MEASURES both its confirmation and the held-out VALUE used for
        # the incumbent separation. The agent's claimed_value never enters the gate.
        confirmed, measured_value = held_out_check(schema)
        return resolve_consequence(
            schema["claim_type"], task, float(measured_value), _mie(schema),
            consequence_catalog=consequence_catalog,
            consequence_digest=config.consequence_catalog_digest,
            incumbent_catalog=incumbent_catalog,
            incumbent_digest=config.incumbent_catalog_digest,
            held_out_confirmed=bool(confirmed))

    return ExperimentSubstrate(
        g0_fn=g0_fn, mechanism_fn=mechanism_fn, novelty_fn=novelty_fn,
        backbone_fn=backbone_fn, consequence_fn=consequence_fn, mie_fn=_mie,
        magnitude_fn=magnitude_fn)
