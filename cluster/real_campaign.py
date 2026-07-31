"""The real SSv2 debug campaign. Runs ON VIGNAN'S MAC (where `claude -p` works) so the real
generative agent and the red/blue ablation-construction roles are real, and dispatches all Qwen
scoring to the persistent cluster service (`qwen_service.py`) over an ssh file-RPC. That makes the
two-tier split physical: agent PROPOSES on the Mac, substrate MEASURES on the cluster GPU, referee
JUDGES. Auto-accept triage (debug). Run locally: uv run python cluster/real_campaign.py

This is the empirical shakeout run. It exercises the whole pipeline on real data + a real model and
surfaces integration bugs a mock run cannot."""
import json
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from backend import Box  # noqa: E402
from backend.hf import HFBackend  # noqa: E402
from engine import ClaudeCodeAgent, run_campaign  # noqa: E402
from engine.ablation_construction import (  # noqa: E402
    ClaudeBlueBuilder,
    ClaudeRedAttacker,
    construct_ablation,
)
from engine.real_substrate import assemble_substrate  # noqa: E402
from experiments.ssv2 import carve_boxes, parse_label_index  # noqa: E402
from gateconfig.signing import verify_config  # noqa: E402
from referee import normalize_schema  # noqa: E402
from referee.lease import LeaseStore  # noqa: E402

CLUSTER = "aicr"
RPC = "/work/neu/p2026_0016_neu/ssv2/rpc"
LABELS = Path.home() / "Downloads" / "labels"


def log(*a):
    print(*a, flush=True)


def rpc_score(pairs, ablate_keep=None, timeout=1800):
    """File-RPC to the cluster Qwen service: write a request, poll for the result."""
    rid = uuid.uuid4().hex[:12]
    req = json.dumps({"clips": [[c, t] for c, t in pairs], "ablate_keep": ablate_keep})
    subprocess.run(["ssh", CLUSTER, f"cat > {RPC}/req_{rid}.json"], input=req, text=True, check=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(["ssh", CLUSTER, f"cat {RPC}/res_{rid}.json 2>/dev/null"],
                             capture_output=True, text=True)
        if out.stdout.strip():
            subprocess.run(["ssh", CLUSTER, f"rm -f {RPC}/res_{rid}.json"])
            d = json.loads(out.stdout)
            if "error" in d:
                raise RuntimeError(f"service error: {d['error']}")
            return np.array(d["scores"], dtype=float)
        time.sleep(6)
    raise TimeoutError(f"rpc {rid} timed out")


def main():
    pub = (REPO / "keys" / "signing_pub.key").read_bytes()
    config = verify_config((REPO / "signed_config.json").read_bytes(), pub)
    cons = json.load(open(REPO / "catalogs" / "consequence_templates.json"))
    inc = json.load(open(REPO / "catalogs" / "incumbent_catalog.json"))
    mie = json.load(open(REPO / "catalogs" / "mie_distribution.json"))
    log(f"config verified: key_id={config.key_id} mie_floor={config.mie_floor} mde={config.mde}")

    labels_raw = json.load(open(LABELS / "labels.json"))
    _classes, index = parse_label_index(labels_raw)
    val = json.load(open(LABELS / "validation.json"))[:200]
    # DEBUG sizes: tiny so the scoring round-trips are fast; a real campaign carves powered boxes.
    boxes = carve_boxes(val, index, n_boxes=2, box_size=6, dev_size=8, rng=np.random.default_rng(0))
    dev_pairs = [(c["id"], c["template"]) for c in boxes.dev]
    log(f"carved {len(boxes.boxes)} holdout boxes + {len(boxes.dev)} dev clips")

    def scorer(_m, box, untrained_init):
        clips = boxes.manifest[box.id]
        if untrained_init is not None:
            return np.zeros(len(clips))  # DEBUG stand-in for random-weights Qwen (~chance)
        return rpc_score([(c["id"], c["template"]) for c in clips])

    backend = HFBackend("Qwen/Qwen2.5-VL-7B-Instruct", revision="main",
                        cutoff_date=date(2024, 1, 1), scorer=scorer)
    backend._model = object()  # non-None so score_box's guard passes; scoring is via the service

    def score_task(_bk, _task, ablation):
        keep = 0.25 if ablation is not None else None  # DEBUG: the constructed temporal-freq ablation
        return rpc_score(dev_pairs, ablate_keep=keep)

    def resolve_via_construction(mechanism, task):
        # DEBUG: 1 red + 2 rounds keeps the claude -p count small so the run completes; the full
        # 3-red / 4-round panel is for a real campaign, not the machinery shakeout.
        return construct_ablation(
            mechanism, task, blue=ClaudeBlueBuilder(), red_panel=[ClaudeRedAttacker()], rounds=2)

    substrate = assemble_substrate(
        config=config, consequence_catalog=cons, incumbent_catalog=inc, mie_catalog=mie,
        score_task=score_task,
        novelty_audit=lambda schema: (False, ["prior work A", "prior work B"]),
        g0_pipeline=lambda effect, rng: 0.001,
        specificity_check=lambda bk, s, t: True,
        membership_check=lambda schema: True,
        held_out_check=lambda schema: True,
        backbone_cutoff=date(2024, 1, 1),
        g0_rng=np.random.default_rng(0),
        resolve_ablation_fn=resolve_via_construction,
    )

    log("\nreal agent (claude -p) proposing ...")
    agent = ClaudeCodeAgent()
    schema_raw = agent.propose({
        "lab": "SMILE (Prof. Yun Raymond Fu)",
        "focus": "video foundation models, frequency-domain temporal modeling",
        "dataset": "ssv2_recognition_top1", "backbone": "Qwen2.5-VL-7B"})[0]
    for k, v in schema_raw.items():
        log(f"    {k}: {v}")
    normalize_schema(schema_raw)

    lease = LeaseStore(str(Path(tempfile.mkdtemp()) / "lease.db"))
    lease.add_boxes([b.id for b in boxes.boxes])
    by_id = {b.id: b for b in boxes.boxes}

    log("\nrunning the campaign (real agent -> substrate MEASURES on cluster -> referee JUDGES) ...")
    result = run_campaign(agent, backend, config, lease, box_factory=lambda bid: by_id[bid])

    v = result.verdict
    log("\n" + "=" * 60)
    log(f"VERDICT: {v.status if v else None}" + (f" ({v.reason})" if v and getattr(v, 'reason', None) else ""))
    log(f"narrative: {result.narrative}")
    log("=" * 60)


if __name__ == "__main__":
    main()
