"""The substrate producer. It MEASURES the gate inputs (runs the G0 probe, the mechanism
ablation, the novelty audit, the backbone check, the consequence experiment) and assembles a
Bundle from those measurements, so the referee gates on evidence the substrate produced, not on
numbers the untrusted agent authored. The agent keeps only its one legitimate input,
believed_claim. Each experiment is an injected callable: mocked here, real on the cluster."""
from datetime import date

from engine.agents import Bundle
from engine.substrate import ExperimentSubstrate, MockSubstrate, Substrate


def test_mock_substrate_produces_a_passing_bundle():
    s = MockSubstrate()
    b = s.produce(schema={"claim": "c"}, backend=None, believed_claim=True)
    assert isinstance(b, Bundle) and b.g0_passed is True and b.believed_claim is True


def test_mock_substrate_satisfies_the_protocol():
    assert isinstance(MockSubstrate(), Substrate)


def _fixed_experiments(**over):
    exp = dict(
        g0_fn=lambda backend, schema: True,
        mechanism_fn=lambda backend, schema: (0.06, 0.01, True),
        novelty_fn=lambda schema: (False, ["a", "b"], True),
        backbone_fn=lambda schema: (date(2023, 1, 1), True),
        consequence_fn=lambda schema: (True, True),
    )
    exp.update(over)
    return exp


def test_experiment_substrate_assembles_from_measurements():
    s = ExperimentSubstrate(**_fixed_experiments())
    b = s.produce({"claim": "c"}, backend=None, believed_claim=False)
    assert b.g0_passed is True
    assert b.mech_full_lo == 0.06 and b.mech_ablated_hi == 0.01 and b.specificity_ok is True
    assert b.novelty_collision is False and b.novelty_k_nearest == ["a", "b"]
    assert b.backbone_cutoff == date(2023, 1, 1) and b.membership_clean is True
    assert b.consequence_confirmed is True and b.incumbent_separated is True
    assert b.believed_claim is False


def test_real_callables_compose_into_a_working_substrate():
    # The Mac-side proof of the whole real path: the four real experiment callables wired into an
    # ExperimentSubstrate produce a Bundle whose measured fields are exactly what the referee needs.
    # On the cluster the deps (HFBackend scoring, live MCP audit, real catalogs) swap in unchanged.
    import numpy as np

    from engine.real_experiments import (
        real_g0, real_mechanism, real_novelty, resolve_consequence,
    )
    from referee.catalog import catalog_digest

    cons, inc = {"effect": "downstream acc rises >= MIE"}, {"ssv2": 0.70}
    cd, idg = catalog_digest(cons), catalog_digest(inc)
    rng = np.random.default_rng(0)

    sub = ExperimentSubstrate(
        g0_fn=lambda backend, schema: real_g0(lambda e, r: 0.001, mde=0.01, alpha=0.05,
                                              n_trials=30, rng=rng),
        mechanism_fn=lambda backend, schema: real_mechanism(
            score_full=lambda: np.full(50, 0.10), score_ablated=lambda: np.full(50, 0.0),
            specificity_ok=True, alpha=0.05),
        novelty_fn=lambda schema: real_novelty(schema, audit_fn=lambda s: (False, ["a", "b"]),
                                               advance_argued=True),
        backbone_fn=lambda schema: (date(2023, 1, 1), True),
        consequence_fn=lambda schema: resolve_consequence(
            "effect", "ssv2", claimed_value=0.80, mie=0.05, held_out_confirmed=True,
            consequence_catalog=cons, consequence_digest=cd,
            incumbent_catalog=inc, incumbent_digest=idg),
    )
    b = sub.produce({"claim": "c"}, backend=None, believed_claim=True)
    assert b.g0_passed is True
    assert b.mech_full_lo > 0.05 and b.mech_ablated_hi < 0.05
    assert b.novelty_collision is False and b.novelty_k_nearest == ["a", "b"]
    assert b.consequence_confirmed is True and b.incumbent_separated is True


def test_a_failed_experiment_propagates_to_a_fail_closed_field():
    # the G0 probe genuinely failed on the pipeline -> the produced bundle carries g0_passed=False,
    # which the referee will read as INELIGIBLE. The substrate does not paper over a failed measure.
    s = ExperimentSubstrate(**_fixed_experiments(g0_fn=lambda backend, schema: False))
    b = s.produce({"claim": "c"}, backend=None, believed_claim=True)
    assert b.g0_passed is False
