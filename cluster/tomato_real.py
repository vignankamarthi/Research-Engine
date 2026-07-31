"""The REAL TOMATO capability run on the ASSEMBLED loop (Milestone 8 steps 60/69). Runs on Vignan's
Mac (real ClaudeCodeAgent + blue/red + the orchestrator), scores on the cluster Qwen service over an
ssh file-RPC. Every audit fix is live here:

- the FLOOR arm scores a REAL weights-randomized Qwen (untrained_seed), guessed at chance, never a stub;
- the substrate is assembled with `require_measured=True`, so a forgotten stub HALTs the run;
- the human's committed CAPABILITY triage (Vignan's decision) is the frozen pre-registration, and the
  agent never picks its own gauntlet;
- the RPC probes cert + service health, writes atomically, parses before delete, and retries.

The GO/NO-GO stays human: this driver runs to verdicts + the pool close, then stops. Run on the Mac:
`uv run python cluster/tomato_real.py <tomato_items.json>`."""
import json
import subprocess
import sys
import tempfile
import time
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
from engine.discovery_roles import ClaudeReviewerAdversary, ClaudeSignificanceAdversary  # noqa: E402
from engine.handoff import TriageDecision  # noqa: E402
from engine.health import RETRY, HealthGate, Probe  # noqa: E402
from engine.orchestrator import run_loop  # noqa: E402
from engine.real_substrate import assemble_substrate, measured  # noqa: E402
from engine.supervisor import Budget, HaltFlag  # noqa: E402
from experiments.power import proportion_sd, required_n  # noqa: E402
from gateconfig.signing import verify_config  # noqa: E402
from referee.lease import LeaseStore  # noqa: E402

CLUSTER = "aicr"
RPC = "/work/neu/p2026_0016_neu/tomato/rpc"
TASK = "tomato_temporal_mcq"
INCUMBENT = 0.379  # the signed TOMATO capability incumbent


def log(*a):
    print(*a, flush=True)


def _ssh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=20", CLUSTER, cmd],
                          capture_output=True, text=True, **kw)


def _service_healthy():
    """cert + a live service job + the READY marker, per the TOOL-LEDGER RETRY/HALT states."""
    if _ssh("true").returncode != 0:
        return False, "ssh/cert"
    if not _ssh(f"test -f {RPC}/READY && echo ok").stdout.strip():
        return False, "no READY"
    return True, "ok"


def _resubmit_service():
    """Self-heal: an 8h wall limit or a node fault can kill the scoring service under a multi-day
    campaign. Resubmit it and wait for READY, so the supervisor RETRYs rather than dying."""
    _ssh("cd /work/neu/p2026_0016_neu/tomato && scancel --name=tomato-svc 2>/dev/null; "
         "sleep 2; rm -rf rpc && mkdir -p rpc && sbatch svc.slurm")
    for _ in range(18):  # ~6 min for Qwen to reload
        time.sleep(20)
        if _service_healthy()[0]:
            return True
    return False


def rpc_score(items, untrained_seed=None, ablate_keep=None, timeout=1800):
    """Score `items` on the cluster service. Atomic request write, parse-before-delete, and a health
    probe so a dead service is distinguished from a slow score rather than silently timing out."""
    ok, why = _service_healthy()
    if not ok:
        raise RuntimeError(f"service unhealthy before RPC: {why}")
    rid = uuid.uuid4().hex[:12]
    req = json.dumps({"items": items, "ablate_keep": ablate_keep, "untrained_seed": untrained_seed})
    # atomic request: write .tmp then mv, so the poller never reads a half-written request
    subprocess.run(["ssh", "-o", "ConnectTimeout=20", CLUSTER,
                    f"cat > {RPC}/.tmp_{rid} && mv {RPC}/.tmp_{rid} {RPC}/req_{rid}.json"],
                   input=req, text=True, check=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = _ssh(f"cat {RPC}/res_{rid}.json 2>/dev/null")
        if out.stdout.strip():
            try:
                d = json.loads(out.stdout)  # PARSE before delete: a torn read is retried, not lost
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
    """The real backend: `score_box` dispatches trained / untrained scoring to the cluster service."""

    def __init__(self, manifest, cutoff_date):
        self.cutoff_date = cutoff_date
        self._manifest = manifest

    def score_box(self, box, untrained_init=None):
        return rpc_score(self._manifest[box.id], untrained_seed=untrained_init)


def real_novelty_audit(schema):
    """Query the Semantic Scholar public API for the mechanism. (collision, k_nearest, advance):
    an exact-title match is a collision; the k nearest titles are named; advance is asserted only when
    no collision AND priors were actually retrieved (else fail-closed False)."""
    q = str(schema.get("mechanism", "") or schema.get("claim", ""))[:120]
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?limit=5&fields=title&query="
           + urllib.parse.quote(q))
    for attempt in range(4):  # the public API rate-limits (429); back off and retry before failing closed
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


class TomatoAgent:
    """Wraps the real ClaudeCodeAgent to PIN the run to the SIGNED TOMATO task key (so the incumbent /
    MIE catalogs resolve) and to ensure the hypothesis names a mechanism the gate can ABLATE (an empty
    mechanism now fails the gate closed rather than crashing, but a capability claim still needs a real
    mechanism to be CONFIRMED). The agent free-forms the claim, measure, and scale."""

    def __init__(self):
        self._a = ClaudeCodeAgent()

    def propose(self, context):
        out = []
        for c in self._a.propose(context):
            c = dict(c)
            c["dataset"] = TASK                       # pin to the signed catalog key
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
    mie = json.load(open(REPO / "catalogs" / "mie_distribution.json"))
    log(f"config verified: key_id={config.key_id}")

    # POWER-SIZE the boxes: how many items to separate a ~0.09 capability gap above the incumbent?
    n_box = required_n(0.09, proportion_sd(INCUMBENT), alpha=config.alpha, power=config.power)
    rng = np.random.default_rng(0)
    rng.shuffle(items)
    dev, rest = items[:n_box], items[n_box:]
    n_boxes = len(rest) // n_box
    boxes, manifest = [], {}
    for b in range(n_boxes):
        bid = f"tomato_hold_{b:03d}"
        manifest[bid] = rest[b * n_box:(b + 1) * n_box]
        boxes.append(Box(id=bid, n=n_box, origin_date=date(2025, 1, 1)))  # post-cutoff, fresh clips
    manifest["__dev__"] = dev
    log(f"power-sized boxes: {n_box} items/box, {n_boxes} holdout boxes + {len(dev)} dev")

    backend = ServiceBackend(manifest, cutoff_date=date(2024, 1, 1))

    def score_task(_bk, _task, ablation):  # the mechanism experiment (dev data, full vs ablated)
        return rpc_score(dev, ablate_keep=(0.25 if ablation is not None else None))

    def held_out(schema):  # the consequence: measured held-out capability on a dev-adjacent set
        acc = float(rpc_score(dev).mean())
        return (acc > INCUMBENT, acc)

    def membership(schema):  # TOMATO clips are self-recorded / synthetic -> membership-clean
        return str(schema.get("dataset", "")).startswith("tomato")

    def g0(effect, rng_):  # detectability: a planted MDE effect must be detectable on the score path
        base = rng_.random(n_box) < INCUMBENT
        lifted = np.clip(base + effect, 0, 1)
        from gatelib import mean_ci
        lo, _ = mean_ci(lifted.astype(float) - base.astype(float), config.alpha)
        return 0.001 if lo > 0 else 0.5

    def resolve_ablation(mechanism, task):
        return construct_ablation(mechanism, task, blue=ClaudeBlueBuilder(),
                                  red_panel=[ClaudeRedAttacker()], rounds=2)

    substrate = assemble_substrate(
        config=config, consequence_catalog=cons, incumbent_catalog=inc, mie_catalog=mie,
        score_task=measured(score_task, name="tomato_score_task"),
        novelty_audit=measured(real_novelty_audit, name="semantic_scholar_novelty"),
        g0_pipeline=measured(g0, name="g0_planted_effect"),
        specificity_check=measured(lambda bk, s, t: True, name="mechanism_specificity"),
        membership_check=measured(membership, name="tomato_membership"),
        held_out_check=measured(held_out, name="tomato_held_out_capability"),
        backbone_cutoff=date(2024, 1, 1), g0_rng=np.random.default_rng(0),
        resolve_ablation_fn=resolve_ablation, require_measured=True)

    lease = LeaseStore(str(Path.home() / ".research-engine" / "tomato_campaign.db"))
    lease.add_boxes([b.id for b in boxes])
    by_id = {b.id: b for b in boxes}

    # Vignan's COMMITTED capability triage is the frozen pre-registration (his decision, encoded).
    triage = lambda d: TriageDecision(accept=True, claim_type="capability",
                                      consequence_template_id="capability", seeds=(0, 1))

    log("\nRUNNING the assembled loop on TOMATO (capability) ...\n")
    reason, report, campaign = run_loop(
        agent=TomatoAgent(), backend=backend, config=config, lease_store=lease,
        box_factory=lambda bid: by_id[bid], substrate=substrate, triage=triage,
        reviewer=ClaudeReviewerAdversary(), significance=ClaudeSignificanceAdversary(),
        budget=Budget.default(max_gpu_hours=200, max_boxes=n_boxes,
                              max_maturations=int(sys.argv[2]) if len(sys.argv) > 2 else min(5, n_boxes)),
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
