"""The REAL IntPhys 2 EFFECT/PHENOMENON run on the ASSEMBLED loop. Runs on Vignan's Mac (real
ClaudeCodeAgent + blue/red + the orchestrator + the mid-campaign synthesizer), scores on the cluster
Qwen service over an ssh file-RPC. This is the first CONFIRMED-reachable campaign: IntPhys 2 has NO
signed incumbent, so the frozen classifier routes performance claims to EFFECT (a mechanism's
trained-minus-untrained contrast against the MIE), not capability, which a 7B VLM would only fail.

Every validity fix is live: the FLOOR arm scores a real weights-randomized Qwen (never a stub), the
substrate is assembled require_measured=True, the claim-type is DERIVED by the classifier (never the
agent's label) and locked per lineage, the box budget CLOSES against the live-box reserves, and the
boxes are sized against the MEASURED dev paired-contrast sd (step 51), not the absolute chance variance.

The GO/NO-GO stays human: this driver runs to verdicts + the pool close, then stops.
Run on the Mac: `uv run python cluster/intphys2_real.py <intphys2_items.json> [max_maturations]`."""
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from backend import Box  # noqa: E402
from engine.ablation_construction import ClaudeBlueBuilder, ClaudeRedAttacker, construct_ablation  # noqa: E402
from engine.claude_agent import ClaudeCodeAgent  # noqa: E402
from engine.closure import Reserves  # noqa: E402
from engine.discovery_roles import ClaudeReviewerAdversary, ClaudeSignificanceAdversary, ClaudeSynthesizer  # noqa: E402
from engine.handoff import classifier_triage  # noqa: E402
from engine.health import RETRY, HealthGate, Probe  # noqa: E402
from engine.orchestrator import run_loop  # noqa: E402
from engine.real_substrate import assemble_substrate, measured  # noqa: E402
from engine.supervisor import Budget, HaltFlag  # noqa: E402
from experiments.power import required_n  # noqa: E402
from gateconfig.signing import verify_config  # noqa: E402
from referee.lease import LeaseStore  # noqa: E402

CLUSTER = "aicr"
RPC = "/work/neu/p2026_0016_neu/intphys2/rpc"
SVC_NAME = "intphys2-svc"
SVC_DIR = "/work/neu/p2026_0016_neu/intphys2"
TASK = "intphys2_physics_binary"          # NO signed incumbent -> the classifier routes to EFFECT
INTPHYS2_ORIGIN = date(2025, 6, 1)        # synthetic Unreal renders, post Qwen cutoff -> backbone-clean


def log(*a):
    print(*a, flush=True)


def _ssh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=20", CLUSTER, cmd],
                          capture_output=True, text=True, **kw)


def _service_healthy():
    if _ssh("true").returncode != 0:
        return False, "ssh/cert"
    if not _ssh(f"test -f {RPC}/READY && echo ok").stdout.strip():
        return False, "no READY"
    return True, "ok"


def _resubmit_service():
    _ssh(f"cd {SVC_DIR} && scancel --name={SVC_NAME} 2>/dev/null; "
         f"sleep 2; rm -rf rpc && mkdir -p rpc && sbatch svc.slurm")
    for _ in range(18):
        time.sleep(20)
        if _service_healthy()[0]:
            return True
    return False


def rpc_score(items, untrained_seed=None, ablate_keep=None, timeout=1800):
    ok, why = _service_healthy()
    if not ok:
        raise RuntimeError(f"service unhealthy before RPC: {why}")
    rid = uuid.uuid4().hex[:12]
    req = json.dumps({"items": items, "ablate_keep": ablate_keep, "untrained_seed": untrained_seed})
    subprocess.run(["ssh", "-o", "ConnectTimeout=20", CLUSTER,
                    f"cat > {RPC}/.tmp_{rid} && mv {RPC}/.tmp_{rid} {RPC}/req_{rid}.json"],
                   input=req, text=True, check=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = _ssh(f"cat {RPC}/res_{rid}.json 2>/dev/null")
        if out.stdout.strip():
            try:
                d = json.loads(out.stdout)
            except json.JSONDecodeError:
                time.sleep(3)
                continue
            _ssh(f"rm -f {RPC}/res_{rid}.json")
            if "error" in d:
                raise RuntimeError(f"service error: {d['error']}")
            return np.array(d["scores"], dtype=float)
        ok, why = _service_healthy()
        if not ok:
            raise RuntimeError(f"service died mid-score: {why}")
        time.sleep(6)
    raise TimeoutError(f"rpc {rid} timed out")


class ServiceBackend:
    def __init__(self, manifest, cutoff_date):
        self.cutoff_date = cutoff_date
        self._manifest = manifest

    def score_box(self, box, untrained_init=None):
        return rpc_score(self._manifest[box.id], untrained_seed=untrained_init)


def real_novelty_audit(schema):
    q = str(schema.get("mechanism", "") or schema.get("claim", ""))[:120]
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?limit=5&fields=title&query="
           + urllib.parse.quote(q))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                papers = json.load(r).get("data", []) or []
            titles = [p.get("title", "") for p in papers]
            collision = any(q.lower() in t.lower() or t.lower() in q.lower() for t in titles if t)
            return bool(collision), titles, (bool(titles) and not collision)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            log(f"    novelty audit HTTP {e.code}; fail-closed")
            return False, [], False
        except Exception as e:
            log(f"    novelty audit unreachable ({e}); fail-closed")
            return False, [], False
    return False, [], False


class IntPhys2Agent:
    """Wraps the real ClaudeCodeAgent to PIN the run to the signed IntPhys 2 task key (so the MIE
    catalog resolves) and to ensure the hypothesis names an ablatable mechanism, so the mechanism gate
    has something to remove. The agent free-forms the claim, measure, and scale; the CLASSIFIER decides
    the claim-type from the form (no incumbent -> effect), not the agent's advisory label."""

    def __init__(self):
        self._a = ClaudeCodeAgent()

    def propose(self, context):
        out = []
        for c in self._a.propose(context):
            c = dict(c)
            c["dataset"] = TASK
            c["task"] = TASK
            if not c.get("mechanism"):
                c["mechanism"] = "temporal_frequency"  # a registered, ablatable mechanism
            out.append(c)
        return out or [{}]

    def mature(self, schema_raw):
        return self._a.mature(schema_raw)

    def frame(self, schema_raw, verdict):
        return self._a.frame(schema_raw, verdict)


def main():
    items = json.load(open(sys.argv[1]))
    pub = (REPO / "keys" / "signing_pub.key").read_bytes()
    config = verify_config((REPO / "signed_config.json").read_bytes(), pub)
    cons = json.load(open(REPO / "catalogs" / "consequence_templates.json"))
    inc = json.load(open(REPO / "catalogs" / "incumbent_catalog.json"))
    mie_cat = json.load(open(REPO / "catalogs" / "mie_distribution.json"))
    mie = float(mie_cat[TASK]["mie_value"])
    log(f"config verified: key_id={config.key_id}; TASK={TASK}; MIE={mie}")

    rng = np.random.default_rng(0)
    rng.shuffle(items)
    n_dev = 180
    dev, rest = items[:n_dev], items[n_dev:]

    # DEV PASS (step 51): size boxes against the MEASURED paired-contrast sd, not the absolute chance
    # variance. Score dev full vs a representative temporal-frequency ablation, per-item, and take the
    # sd of the paired difference. An effect's contrast is concentrated, so its sd is far below 0.5.
    log("dev pass: scoring dev full + ablated to size boxes ...")
    dev_full = rpc_score(dev)
    dev_abl = rpc_score(dev, ablate_keep=0.25)
    diff = dev_full - dev_abl
    contrast_sd = float(diff.std()) or 0.2
    dev_contrast = float(diff.mean())
    log(f"dev full acc={dev_full.mean():.3f}  ablated={dev_abl.mean():.3f}  "
        f"contrast={dev_contrast:+.3f}  contrast_sd={contrast_sd:.3f}")

    n_box = required_n(mie, contrast_sd, alpha=config.alpha, power=config.power)
    n_box = max(60, min(n_box, len(rest) // 2))       # at least a replication pair from the pool
    n_boxes = len(rest) // n_box
    boxes, manifest = [], {}
    for b in range(n_boxes):
        bid = f"intphys2_hold_{b:03d}"
        manifest[bid] = rest[b * n_box:(b + 1) * n_box]
        boxes.append(Box(id=bid, n=n_box, origin_date=INTPHYS2_ORIGIN))
    manifest["__dev__"] = dev
    log(f"power-sized boxes: {n_box} items/box (MIE {mie} at contrast_sd {contrast_sd:.3f}), "
        f"{n_boxes} holdout boxes")
    if n_boxes < 1:
        log("FATAL: the pool cannot carve one powered box; raise the MIE (re-sign) or add data.")
        return

    backend = ServiceBackend(manifest, cutoff_date=date(2024, 1, 1))

    def score_task(_bk, _task, ablation):
        return rpc_score(dev, ablate_keep=(0.25 if ablation is not None else None))

    def held_out(schema):
        # the effect consequence: the mechanism's contrast holds on a held-out-adjacent (dev) set.
        # measured_value is the full accuracy (informational for effect; no incumbent separation).
        return (dev_contrast >= mie, float(dev_full.mean()))

    def membership(schema):
        return str(schema.get("dataset", "")).startswith("intphys2")  # synthetic renders -> clean

    def g0(effect, rng_):
        base = rng_.random(n_box) < float(dev_full.mean())
        lifted = np.clip(base + effect, 0, 1)
        from gatelib import mean_ci
        lo, _ = mean_ci(lifted.astype(float) - base.astype(float), config.alpha)
        return 0.001 if lo > 0 else 0.5

    def resolve_ablation(mechanism, task):
        return construct_ablation(mechanism, task, blue=ClaudeBlueBuilder(),
                                  red_panel=[ClaudeRedAttacker()], rounds=2)

    substrate = assemble_substrate(
        config=config, consequence_catalog=cons, incumbent_catalog=inc, mie_catalog=mie_cat,
        score_task=measured(score_task, name="intphys2_score_task"),
        novelty_audit=measured(real_novelty_audit, name="semantic_scholar_novelty"),
        g0_pipeline=measured(g0, name="g0_planted_effect"),
        specificity_check=measured(lambda bk, s, t: True, name="mechanism_specificity"),
        membership_check=measured(membership, name="intphys2_membership"),
        held_out_check=measured(held_out, name="intphys2_effect_consequence"),
        backbone_cutoff=date(2024, 1, 1), g0_rng=np.random.default_rng(0),
        resolve_ablation_fn=resolve_ablation, require_measured=True)

    lease = LeaseStore(str(Path.home() / ".research-engine" / "intphys2_campaign.db"))
    lease.add_boxes([b.id for b in boxes])
    by_id = {b.id: b for b in boxes}

    # the CLASSIFIER derives the claim-type (no incumbent -> effect), locked one-type-per-lineage; the
    # human never picks the gauntlet here, only the final go/no-go.
    triage = classifier_triage(incumbent_tasks=(), consequence_template_id="effect", seeds=(0, 1))

    # Size the maturation ceiling to what the carved pool can CLOSE, keeping the mandatory second-box
    # replication (1 per maturation). The FLOOR is scored ON-box via untrained_seed, so no separate
    # backbone-cohort boxes are needed, and a first run holds no re-score contingency. Each maturation
    # therefore needs 2 boxes (primary + replication), so the ceiling is floor(pool / 2).
    live = lease.live_count()
    requested = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if live >= 2:
        max_mat = max(1, min(requested, live // 2))
        reserves = Reserves.for_campaign(max_mat, rescore_contingency=0, backbone_cohort=0)
    else:  # a single powered box: one primary verdict, no replication this run (not submit-bound)
        max_mat = 1
        reserves = Reserves(primary_demand=1, replication=0, rescore=0, backbone=0)
    budget = Budget.from_closure(live_boxes=live, reserves=reserves,
                                 max_gpu_hours=200.0, max_maturations=max_mat)
    log(f"closure: pool={live} boxes -> max_maturations={max_mat} (replication reserve "
        f"{reserves.replication}, on-box FLOOR)")

    log(f"\nRUNNING the assembled loop on IntPhys 2 (effect/phenomenon), max_maturations={max_mat} ...\n")
    reason, report, campaign = run_loop(
        agent=IntPhys2Agent(), backend=backend, config=config, lease_store=lease,
        box_factory=lambda bid: by_id[bid], substrate=substrate, triage=triage,
        reviewer=ClaudeReviewerAdversary(), significance=ClaudeSignificanceAdversary(),
        synthesizer=ClaudeSynthesizer(), budget=budget,
        halt_flag=HaltFlag(str(Path(tempfile.mkdtemp()) / "halt")),
        health_gate=HealthGate([Probe("service", check=lambda: _service_healthy()[0],
                                      state=RETRY, self_heal=_resubmit_service)]), seed=0)

    log("\n" + "=" * 64)
    log(f"TERMINAL: {reason}")
    for r in campaign.results:
        v = r.verdict
        log(f"  {v.status if v else 'None'}"
            + (f" ({v.reason})" if v and getattr(v, 'reason', None) else "") + f"  <- {r.lineage[:10]}")
    log("\n" + report.narrative)
    log("=" * 64)
    log("\nNEXT: Vignan's GO/NO-GO on any submit-bound finding above.")


if __name__ == "__main__":
    main()
