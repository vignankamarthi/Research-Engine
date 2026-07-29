"""The single trusted schema-normal-form: it derives the control set, the lineage
key, and the magnitude gate as the strictest consistent with the schema. The lineage
key hashes the CLAIM only, so a condition-only change (backbone/scale/measure/dataset)
stays one lineage and only a genuinely different claim hashes fresh. Measures and
dataset identities are alias-canonicalized so a reworded-but-equivalent claim does
not slip a fresh lineage. Computed in the trusted process, never by the generative tier."""
import pytest

from referee import derive, lineage_key, normalize_schema


def raw(**kw):
    d = {
        "claim": "freq-domain temporal modeling improves recognition",
        "claim_type": "effect",
        "backbone": "internvideo2",
        "dataset": "ssv2",
        "scale": "7b",
        "measure": "top-1 accuracy",
        "prior_claim": False,
    }
    d.update(kw)
    return d


def test_condition_only_change_stays_one_lineage():
    a = normalize_schema(raw())
    b = normalize_schema(raw(backbone="videomae", scale="1b", measure="acc",
                             dataset="something-something v2"))
    assert lineage_key(a) == lineage_key(b)


def test_different_claim_gets_a_fresh_lineage():
    a = normalize_schema(raw())
    b = normalize_schema(raw(claim="wavelet pooling reduces flops"))
    assert lineage_key(a) != lineage_key(b)


def test_measure_and_dataset_aliases_are_canonicalized():
    a = normalize_schema(raw(measure="top-1 accuracy", dataset="ssv2"))
    assert a.measure == "accuracy"
    assert a.dataset == "something-something-v2"


def test_control_set_always_carries_the_floor():
    assert "untrained_floor" in derive(normalize_schema(raw())).control_set


def test_prior_claim_swaps_in_the_prior_ablated_baseline():
    d = derive(normalize_schema(raw(prior_claim=True)))
    assert "prior_ablated_baseline" in d.control_set
    assert "untrained_floor" not in d.control_set  # replaced, not added


def test_magnitude_gate_follows_claim_type():
    assert derive(normalize_schema(raw(claim_type="capability"))).magnitude_gate == "capability_separation"
    assert derive(normalize_schema(raw())).magnitude_gate == "mie_at_power"


def test_unknown_claim_type_rejected():
    with pytest.raises(ValueError):
        normalize_schema(raw(claim_type="leaderboard_bump"))


def test_lineage_key_is_deterministic_and_stable():
    a = lineage_key(normalize_schema(raw()))
    b = lineage_key(normalize_schema(raw()))
    assert a == b and isinstance(a, str) and len(a) == 64  # sha256 hex
