"""The SSv2 experiment pieces that are testable without a model or the dataset: the temporal
-frequency ablation transform (the concrete mechanism op) and the label parsing. The Qwen inference
itself is validated on the cluster. The ablation removes fast temporal structure from a clip by
zeroing high temporal-frequency bands along the time axis, so a model relying on that structure
loses its effect while a spatial/appearance model does not."""
import numpy as np
import pytest

from engine.ablations import resolve_ablation
from experiments.ssv2 import (
    annotation_class_id,
    debracket_template,
    parse_label_index,
    remove_temporal_frequency,
)


def test_temporal_frequency_registry_op_is_the_real_transform():
    # the registry's temporal_frequency ablation applies the real DFT transform, not a stub
    assert resolve_ablation("temporal_frequency").apply is remove_temporal_frequency


def _clip(t, h=4, w=4, c=3):
    return np.zeros((t, h, w, c), dtype=float)


def test_ablation_preserves_shape():
    frames = np.random.default_rng(0).random((16, 8, 8, 3))
    out = remove_temporal_frequency(frames, keep_fraction=0.5)
    assert out.shape == frames.shape


def test_a_temporally_constant_clip_is_unchanged():
    # no temporal variation -> nothing in the high bands to remove
    frames = np.tile(np.random.default_rng(1).random((1, 4, 4, 3)), (12, 1, 1, 1))
    out = remove_temporal_frequency(frames, keep_fraction=0.25)
    assert np.allclose(out, frames, atol=1e-9)


def test_high_frequency_flicker_is_strongly_attenuated():
    # a clip that flips every frame is pure highest temporal frequency; removing high bands kills it
    base = np.random.default_rng(2).random((4, 4, 3))
    frames = np.stack([base if i % 2 == 0 else 1.0 - base for i in range(16)], axis=0)
    out = remove_temporal_frequency(frames, keep_fraction=0.25)
    # the per-pixel temporal variance collapses toward the mean
    assert out.var(axis=0).mean() < 0.25 * frames.var(axis=0).mean()


def test_keep_fraction_one_is_near_identity():
    frames = np.random.default_rng(3).random((10, 4, 4, 3))
    out = remove_temporal_frequency(frames, keep_fraction=1.0)
    assert np.allclose(out, frames, atol=1e-9)


def test_parse_label_index_maps_class_to_id():
    # SSv2 labels.json is {class: "index"}; we want an ordered class list + a class -> int lookup
    raw = {"Pushing something from left to right": "0", "Moving something up": "1",
           "Covering something with something": "2"}
    classes, index = parse_label_index(raw)
    assert classes[0] == "Pushing something from left to right"
    assert index["Moving something up"] == 1
    assert len(classes) == 3


def test_debracket_matches_the_real_ssv2_format():
    # train/val templates are bracketed, labels.json keys are not (verified against the real files)
    assert debracket_template("Spinning [something] that quickly stops spinning") == \
        "Spinning something that quickly stops spinning"


def test_annotation_class_id_resolves_a_bracketed_template():
    # the real mapping: this template resolves to class 140 in the shipped labels.json
    label_index = {"Spinning something that quickly stops spinning": 140}
    entry = {"id": "74225", "template": "Spinning [something] that quickly stops spinning"}
    assert annotation_class_id(entry, label_index) == 140


def test_annotation_class_id_fails_loud_on_unknown_template():
    with pytest.raises(KeyError):
        annotation_class_id({"template": "Doing [something] impossible"}, {"real class": 0})


def _val_entries(n):
    return [{"id": str(1000 + i), "template": "Moving [something] up", "label": "moving cup up"}
            for i in range(n)]


def _idx():
    return {"Moving something up": 42}


def test_carve_boxes_are_disjoint_and_touch_once():
    from datetime import date

    import numpy as np

    from experiments.ssv2 import SSV2_ORIGIN, carve_boxes
    out = carve_boxes(_val_entries(100), _idx(), n_boxes=3, box_size=10, dev_size=20,
                      rng=np.random.default_rng(0))
    all_box_ids = [c["id"] for b in out.boxes for c in out.manifest[b.id]]
    dev_ids = [c["id"] for c in out.dev]
    assert len(all_box_ids) == 30 and len(dev_ids) == 20
    # no clip appears twice across boxes, and no box clip is in the dev split (touch-once)
    assert len(set(all_box_ids)) == 30
    assert set(all_box_ids).isdisjoint(set(dev_ids))
    assert all(b.n == 10 and b.origin_date == SSV2_ORIGIN == date(2018, 6, 1) for b in out.boxes)
    assert out.manifest[out.boxes[0].id][0]["class_id"] == 42


def test_carve_boxes_is_deterministic_under_a_seed():
    import numpy as np

    from experiments.ssv2 import carve_boxes
    a = carve_boxes(_val_entries(50), _idx(), n_boxes=2, box_size=5, dev_size=10,
                    rng=np.random.default_rng(7))
    b = carve_boxes(_val_entries(50), _idx(), n_boxes=2, box_size=5, dev_size=10,
                    rng=np.random.default_rng(7))
    assert [c["id"] for c in a.dev] == [c["id"] for c in b.dev]


def test_carve_boxes_refuses_when_not_enough_clips():
    import numpy as np

    from experiments.ssv2 import carve_boxes
    with pytest.raises(ValueError):
        carve_boxes(_val_entries(10), _idx(), n_boxes=3, box_size=10, dev_size=20,
                    rng=np.random.default_rng(0))
