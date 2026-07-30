"""SSv2 recognition experiment. The dataset-specific scorer the substrate injects for a
Something-Something-V2 recognition claim. Split into pieces that are testable on the Mac (the
temporal-frequency ablation transform, the label parsing) and the Qwen inference, which lazy-imports
torch and is validated on the cluster.

The temporal-frequency ablation is an INPUT transform, not a model edit: it removes fast temporal
structure from the clip so a model whose effect depends on that structure loses it, while a
spatial/appearance model is unaffected. That keeps the causal test model-agnostic."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from backend import Box

# The temporal-frequency ablation op is generic signal processing and lives with the engine's
# ablation registry (the one `temporal_frequency` entry). Re-exported here so the SSv2 scorer and
# its tests read it from the experiment module they live in.
from engine.ablations import remove_temporal_frequency

__all__ = ["remove_temporal_frequency", "parse_label_index", "debracket_template",
           "annotation_class_id", "carve_boxes", "SSv2Boxes", "SSV2_ORIGIN"]

# SSv2 v2 public release. It PRE-dates any modern backbone cutoff (Qwen ~2024), so the backbone gate
# will correctly flag contamination on SSv2 val, honest, and a real thing to observe in a debug run.
SSV2_ORIGIN = date(2018, 6, 1)


def parse_label_index(raw: dict) -> tuple[list[str], dict[str, int]]:
    """Parse the SSv2 labels file ({class: "index"}) into an ordered class list and a class -> int
    lookup. The order follows the integer indices, so the list is the canonical class order. Note
    labels.json keys are DE-BRACKETED ('Spinning something ...'), unlike the train/val templates."""
    pairs = sorted(((cls, int(idx)) for cls, idx in raw.items()), key=lambda p: p[1])
    classes = [cls for cls, _ in pairs]
    index = {cls: int(idx) for cls, idx in pairs}
    return classes, index


def debracket_template(template: str) -> str:
    """Map a train/val bracketed template ('... [something] ...') to the de-bracketed form used as
    the labels.json key ('... something ...'). Without this the template never matches the index."""
    return template.replace("[", "").replace("]", "")


def annotation_class_id(entry: dict, label_index: dict[str, int]) -> int:
    """The integer class id for a train/val annotation entry, via its de-bracketed template. Raises
    if the template has no matching class, so a mislabeled entry fails loud, not scoring wrong."""
    cls = debracket_template(entry["template"])
    if cls not in label_index:
        raise KeyError(f"template {entry['template']!r} -> {cls!r} not in the label index")
    return label_index[cls]


@dataclass
class SSv2Boxes:
    boxes: list       # list[Box], one per holdout box (touch-once)
    manifest: dict    # box_id -> list of {"id": clip_id, "class_id": int, "template": str}
    dev: list         # the dev-split clips (same dict shape), disjoint from every box


def carve_boxes(entries, label_index, *, n_boxes, box_size, dev_size, rng, origin=SSV2_ORIGIN):
    """Carve DISJOINT touch-once holdout boxes + a dev split from SSv2 val entries. `entries` are
    validation.json dicts, `rng` is a numpy Generator for a deterministic shuffle. Each box gets a
    unique id, `box_size` clips, and SSv2's honest origin date. No clip appears in two boxes or in
    the dev split, so a scored holdout box is never reused (the touch-once invariant)."""
    pool = [{"id": e["id"], "class_id": annotation_class_id(e, label_index),
             "template": debracket_template(e["template"])} for e in entries]
    ordered = [pool[i] for i in rng.permutation(len(pool))]
    need = dev_size + n_boxes * box_size
    if need > len(ordered):
        raise ValueError(f"need {need} clips (dev {dev_size} + {n_boxes}x{box_size}) but only "
                         f"{len(ordered)} available")
    dev = ordered[:dev_size]
    rest = ordered[dev_size:]
    boxes, manifest = [], {}
    for b in range(n_boxes):
        chunk = rest[b * box_size:(b + 1) * box_size]
        bid = f"ssv2_holdout_{b:03d}"
        boxes.append(Box(id=bid, n=box_size, origin_date=origin))
        manifest[bid] = chunk
    return SSv2Boxes(boxes=boxes, manifest=manifest, dev=dev)
