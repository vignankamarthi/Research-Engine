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

from referee.catalog import resolve_mie

from .ablations import build_mechanism_fn
from .real_experiments import real_g0, real_novelty, resolve_consequence
from .substrate import ExperimentSubstrate


def _task_of(schema: dict) -> str:
    """The task key, read from the claim. Must match the catalogs' task keys."""
    return schema.get("task") or schema["dataset"]


def assemble_substrate(*, config, consequence_catalog, incumbent_catalog, mie_catalog,
                       score_task, novelty_audit, g0_pipeline, specificity_check,
                       membership_check, held_out_check, backbone_cutoff, g0_rng,
                       resolve_ablation_fn=None):
    """Wire the real callables into an ExperimentSubstrate. `config` is the SIGNED gate config (its
    catalog digests pin the three catalogs). The returned substrate's `.produce(raw_schema, backend,
    believed_claim=...)` yields the referee's Bundle, with the per-task MIE resolved onto bundle.mie
    and every catalog verified against the signed digest at resolve time."""
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
        return real_novelty(schema, audit_fn=novelty_audit,
                            advance_argued=bool(schema.get("advance_argued", True)))

    def backbone_fn(schema):
        return backbone_cutoff, bool(membership_check(schema))

    def consequence_fn(schema):
        task = _task_of(schema)
        return resolve_consequence(
            schema["claim_type"], task, float(schema.get("claimed_value", 0.0)), _mie(schema),
            consequence_catalog=consequence_catalog,
            consequence_digest=config.consequence_catalog_digest,
            incumbent_catalog=incumbent_catalog,
            incumbent_digest=config.incumbent_catalog_digest,
            held_out_confirmed=bool(held_out_check(schema)))

    return ExperimentSubstrate(
        g0_fn=g0_fn, mechanism_fn=mechanism_fn, novelty_fn=novelty_fn,
        backbone_fn=backbone_fn, consequence_fn=consequence_fn, mie_fn=_mie)
