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


def test_a_failed_experiment_propagates_to_a_fail_closed_field():
    # the G0 probe genuinely failed on the pipeline -> the produced bundle carries g0_passed=False,
    # which the referee will read as INELIGIBLE. The substrate does not paper over a failed measure.
    s = ExperimentSubstrate(**_fixed_experiments(g0_fn=lambda backend, schema: False))
    b = s.produce({"claim": "c"}, backend=None, believed_claim=True)
    assert b.g0_passed is False
