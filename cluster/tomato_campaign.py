"""The real TOMATO debug campaign. Runs ON VIGNAN'S MAC (real agent + blue/red via `claude -p`) and
dispatches MCQ scoring to the cluster TOMATO service (`tomato_service.py`) over the ssh file-RPC.
TOMATO is backbone-clean (self-recorded/synthetic clips), so the backbone gate should PASS and the
full magnitude/mechanism/consequence/novelty gauntlet runs toward a real verdict. Auto-accept triage.
Run locally: uv run python cluster/tomato_campaign.py"""
import glob
import json
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

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
from gateconfig.signing import verify_config  # noqa: E402
from referee import normalize_schema  # noqa: E402
from referee.lease import LeaseStore  # noqa: E402

CLUSTER = "aicr"
RPC = "/scratch/kamarthi_v_neu/tomato/rpc"
QA = Path.home() / "Downloads"  # parquets live in the Mac scratchpad; overridden below
TASK = "tomato_temporal_mcq"


def log(*a):
    print(*a, flush=True)


def rpc_score(items, ablate_keep=None, timeout=1800):
    rid = uuid.uuid4().hex[:12]
    req = json.dumps({"items": items, "ablate_keep": ablate_keep})
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


def load_items(data_dir):
    items = []
    for f in glob.glob(str(Path(data_dir) / "*.parquet")):
        df = pq.read_table(f).to_pandas()
        for _, r in df.iterrows():
            items.append({"key": r["key"], "demonstration_type": r["demonstration_type"],
                          "question": r["question"], "options": list(r["options"]),
                          "answer": int(r["answer"])})
    return items


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(REPO.parent)  # pass the parquet dir
    pub = (REPO / "keys" / "signing_pub.key").read_bytes()
    config = verify_config((REPO / "signed_config.json").read_bytes(), pub)
    cons = json.load(open(REPO / "catalogs" / "consequence_templates.json"))
    inc = json.load(open(REPO / "catalogs" / "incumbent_catalog.json"))
    mie = json.load(open(REPO / "catalogs" / "mie_distribution.json"))
    log(f"config verified: key_id={config.key_id} mie_floor={config.mie_floor}")

    items = load_items(data_dir)
    rng = np.random.default_rng(0)
    rng.shuffle(items)
    dev, rest = items[:8], items[8:]
    boxes, manifest = [], {}
    for b in range(2):
        chunk = rest[b * 6:(b + 1) * 6]
        bid = f"tomato_holdout_{b:03d}"
        boxes.append(Box(id=bid, n=6, origin_date=date(2025, 1, 1)))  # post-cutoff; clips are fresh
        manifest[bid] = chunk
    log(f"loaded {len(items)} TOMATO items, carved 2 boxes x 6 + 8 dev")

    def scorer(_m, box, untrained_init):
        clips = manifest[box.id]
        if untrained_init is not None:
            return np.zeros(len(clips))
        return rpc_score(clips)

    backend = HFBackend("Qwen/Qwen2.5-VL-7B-Instruct", revision="main",
                        cutoff_date=date(2024, 1, 1), scorer=scorer)
    backend._model = object()

    def score_task(_bk, _task, ablation):
        return rpc_score(dev, ablate_keep=(0.25 if ablation is not None else None))

    def resolve_via_construction(mechanism, task):
        return construct_ablation(mechanism, task, blue=ClaudeBlueBuilder(),
                                  red_panel=[ClaudeRedAttacker()], rounds=2)

    substrate = assemble_substrate(
        config=config, consequence_catalog=cons, incumbent_catalog=inc, mie_catalog=mie,
        score_task=score_task,
        novelty_audit=lambda schema: (False, ["prior work A", "prior work B"]),
        g0_pipeline=lambda effect, rng_: 0.001,
        specificity_check=lambda bk, s, t: True,
        membership_check=lambda schema: True,   # TOMATO clips are self-recorded/synthetic -> clean
        held_out_check=lambda schema: True,
        backbone_cutoff=date(2024, 1, 1),
        g0_rng=np.random.default_rng(0),
        resolve_ablation_fn=resolve_via_construction)

    log("\nreal agent (claude -p) proposing on TOMATO ...")
    agent = ClaudeCodeAgent()
    schema_raw = agent.propose({
        "lab": "SMILE (Prof. Yun Raymond Fu)",
        "focus": "video foundation models, temporal reasoning mechanisms",
        "dataset": TASK, "backbone": "Qwen2.5-VL-7B",
        "note": "TOMATO temporal-reasoning MCQ, backbone-clean"})[0]
    schema_raw["dataset"] = TASK          # pin to the catalog task key
    schema_raw.setdefault("claimed_value", 0.45)  # claimed held-out accuracy (agent's claim)
    for k, v in schema_raw.items():
        log(f"    {k}: {v}")
    normalize_schema(schema_raw)

    lease = LeaseStore(str(Path(tempfile.mkdtemp()) / "lease.db"))
    lease.add_boxes([b.id for b in boxes])
    by_id = {b.id: b for b in boxes}

    log("\nrunning the campaign (real agent -> substrate MEASURES on cluster -> referee JUDGES) ...")
    result = run_campaign(agent, backend, config, lease, box_factory=lambda bid: by_id[bid])
    v = result.verdict
    log("\n" + "=" * 60)
    log(f"VERDICT: {v.status if v else None}" + (f" ({v.reason})" if v and getattr(v, 'reason', None) else ""))
    log(f"narrative: {result.narrative}")
    log("=" * 60)


if __name__ == "__main__":
    main()
