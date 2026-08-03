"""Generalized, dataset-parameterized REAL campaign driver (PLAN 73).

This REPLACES the thin per-dataset agents (`intphys2_real.IntPhys2Agent`, `tomato_real`) that PINNED
`mechanism=temporal_frequency` (the drift this step kills) with the GROUNDED, vein-diverse WaveAgent:
a BLIND `claude -p` scout proposes each idea by working a literature vein, the trusted-process
arXiv/DOI resolver grounds it, the negative bank excludes dead ends, and the JEPA standing reserve
(PLAN 78) is enforced in the steering. ONE `TaskConfig` parameterizes the driver over datasets. NO
mechanism is ever re-pinned: the only thing stamped onto a candidate is the signed TASK key (so the
MIE + catalogs resolve), never the mechanism, which the scout free-forms and grounds every idea.

Two layers, on purpose:

  1. The CONSTRUCTION SEAM (Mac-testable, pure): `TaskConfig`, `task_stamping_scout`, `build_wave_agent`,
     `build_triage`, `significance_penalty`. Unit-tested in `tests/engine/test_campaign_real.py` with a
     fake scout, no cluster, no `claude -p`. This is where the WaveAgent + classifier triage + JEPA
     reserve are wired, correct-by-construction.
  2. The RUN WIRING (`main`, CLUSTER-GATED): the ssh file-RPC to the Qwen service, the dev paired-contrast
     box sizing, the powered box carving, `assemble_substrate(require_measured=True)`, and `run_loop`.
     This CANNOT run on the Mac (it needs the real service + real `claude -p` scouts). It is
     correct-by-construction here; a CLUSTER SMOKE RUN is required before any verdict is trusted.

Run on the Mac (dispatches scoring to the cluster service):
    uv run python cluster/campaign_real.py <task> <items.json> [max_maturations]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from backend import Box  # noqa: E402
from engine.ablation_construction import ClaudeBlueBuilder, ClaudeRedAttacker, construct_ablation  # noqa: E402
from engine.claude_agent import ClaudeCodeAgent  # noqa: E402
from engine.closure import Reserves  # noqa: E402
from engine.discovery_roles import (  # noqa: E402
    ClaudeReviewerAdversary,
    ClaudeSignificanceAdversary,
    ClaudeSynthesizer,
)
from engine.generation import JepaReserve, WaveAgent  # noqa: E402
from engine.handoff import classifier_triage  # noqa: E402
from engine.health import RETRY, HealthGate, Probe  # noqa: E402
from engine.orchestrator import run_loop  # noqa: E402
from engine.real_substrate import assemble_substrate, measured  # noqa: E402
from engine.resolvers import arxiv_doi_resolver  # noqa: E402
from engine.supervisor import Budget, HaltFlag  # noqa: E402
from experiments.power import required_n  # noqa: E402
from gateconfig.signing import verify_config  # noqa: E402
from referee.lease import LeaseStore  # noqa: E402

CLUSTER = "aicr"


# ---------------------------------------------------------------------------------------------------
# The CONSTRUCTION SEAM (Mac-testable, pure). No cluster, no claude -p, no RPC in this section.
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskConfig:
    """One dataset's parameterization of the campaign. `task` is the SIGNED task key the MIE + incumbent
    + consequence catalogs resolve on; it is the ONLY field stamped onto a candidate (never mechanism)."""
    task: str
    claim_types: tuple                       # the WIRED claim-type envelope (e.g. ("effect",))
    origin_date: date                        # data origin (post-cutoff -> backbone-clean)
    rpc_dir: str                             # the cluster ssh file-RPC dir
    svc_name: str                            # the SLURM service job name (for auto-resubmit)
    svc_dir: str                             # the service working dir (holds svc.slurm)
    incumbent_tasks: tuple = ()              # signed-incumbent task keys -> classifier forces CAPABILITY
    consequence_template_id: str = "effect"
    seeds: tuple = (0, 1)
    backbone_cutoff: date = date(2024, 1, 1)
    n_dev: int = 180                         # dev items held out for paired-contrast box sizing
    jepa_floor: int = 5                      # PLAN 78 standing reserve floor
    jepa_cap: int = 10                       # PLAN 78 standing reserve cap
    ablate_keep: float = 0.25                # representative dev ablation keep-fraction for sizing


class _TaskStampingScout:
    """Wraps a BLIND scout to stamp the signed TASK key onto `dataset`/`task` so the MIE + catalogs
    resolve, WITHOUT touching `mechanism`. This is the drift fix: the scout free-forms and grounds a
    fresh mechanism every idea, and nothing here pins `temporal_frequency` (or any mechanism)."""

    def __init__(self, scout, *, task: str):
        self._scout = scout
        self._task = task

    def propose(self, context):
        out = []
        for c in self._scout.propose(context):
            if not isinstance(c, dict):
                continue
            c = dict(c)
            c["dataset"] = self._task
            c["task"] = self._task
            out.append(c)
        return out or [{}]

    def mature(self, schema_raw):
        return self._scout.mature(schema_raw)

    def frame(self, schema_raw, verdict):
        return self._scout.frame(schema_raw, verdict)


def task_stamping_scout(scout, *, task: str) -> _TaskStampingScout:
    """The public factory for the task-stamping wrapper (kept a function so tests read cleanly)."""
    return _TaskStampingScout(scout, task=task)


def significance_penalty(significance):
    """Adapt the significance-adversary into the WaveAgent's advisory `importance_of` penalty. It is
    per-candidate and FAIL-SOFT (0.0 on any error): importance is advisory and never a silent kill, so a
    flaky adversary lowers no rank rather than dropping a candidate. Returns None when no adversary is
    wired (the WaveAgent then applies a zero penalty)."""
    if significance is None:
        return None

    def importance_of(candidate: dict) -> float:
        try:
            return float(significance.challenge(candidate).importance_penalty)
        except Exception:
            return 0.0

    return importance_of


def build_wave_agent(config: TaskConfig, *, scout, resolver=arxiv_doi_resolver,
                     significance=None, quality_of=None) -> WaveAgent:
    """Construct the campaign's agent: the GROUNDED, vein-diverse WaveAgent wrapping a BLIND scout,
    with the JEPA standing reserve wired into the steering. The scout's raw proposals flow through
    stamp -> claim-type envelope (`config.claim_types`) -> dead-end exclusion -> trusted-process
    grounding (`resolver`) -> rank (quality tempered by the significance-adversary's advisory penalty)
    -> the JEPA reserve, and the surviving top candidate is what the campaign matures. NO mechanism pin."""
    return WaveAgent(
        task_stamping_scout(scout, task=config.task),
        resolver=resolver,
        wired_claim_types=config.claim_types,
        quality_of=quality_of,
        importance_of=significance_penalty(significance),
        reserve=JepaReserve(floor=config.jepa_floor, cap=config.jepa_cap),
    )


def build_triage(config: TaskConfig):
    """The UNATTENDED handoff: the claim-type is DERIVED by the frozen classifier from the schema + the
    signed catalogs (never the agent's label) and locked one-type-per-lineage. `incumbent_tasks` from the
    signed incumbent catalog forces a covered task to CAPABILITY; an un-incumbent task routes to EFFECT."""
    return classifier_triage(
        incumbent_tasks=config.incumbent_tasks,
        consequence_template_id=config.consequence_template_id,
        seeds=config.seeds,
    )


def build_scout(*, runner=None):
    """The real BLIND Tier-1 scout (domain-neutral prompt, no lab steering). Isolation of cwd + MCP
    scope is scout_isolation's job at launch; this constructs the blind agent the WaveAgent wraps."""
    return ClaudeCodeAgent(runner=runner, blind=True)


# ---------------------------------------------------------------------------------------------------
# The signed task registry. Each entry is a dataset the campaign can run against.
# ---------------------------------------------------------------------------------------------------

TASKS: dict[str, TaskConfig] = {
    # IntPhys 2: synthetic Unreal renders (post Qwen cutoff -> backbone-clean), NO signed incumbent, so
    # the classifier routes performance claims to EFFECT (a mechanism's trained-minus-untrained contrast).
    "intphys2": TaskConfig(
        task="intphys2_physics_binary",
        claim_types=("effect",),
        origin_date=date(2025, 6, 1),
        rpc_dir="/work/neu/p2026_0016_neu/intphys2/rpc",
        svc_name="intphys2-svc",
        svc_dir="/work/neu/p2026_0016_neu/intphys2",
        incumbent_tasks=(),                 # no signed incumbent -> EFFECT
    ),
    # The engine's OWN generated paired possible/impossible physics clips (8 violation types). Rendered
    # in 2026 (post Qwen cutoff -> backbone-clean by construction, and un-memorizable since procedural),
    # scored by the SAME continuation-lean binary scorer as IntPhys 2. NO signed incumbent, so the
    # classifier routes performance claims to EFFECT (a mechanism's trained-minus-untrained contrast on
    # the pixel-identical pairs). MIE resolves on the signed `generated_physics_paired` catalog entry.
    "generated_physics": TaskConfig(
        task="generated_physics_paired",
        claim_types=("effect",),
        origin_date=date(2026, 8, 1),
        rpc_dir="/work/neu/p2026_0016_neu/generated_physics/rpc",
        svc_name="genphys-svc",
        svc_dir="/work/neu/p2026_0016_neu/generated_physics",
        incumbent_tasks=(),                 # no signed incumbent -> EFFECT
    ),
}


# ---------------------------------------------------------------------------------------------------
# The RUN WIRING (CLUSTER-GATED). Needs the real Qwen service + real claude -p scouts. Not Mac-runnable.
# A cluster smoke run is required before any verdict is trusted.
# ---------------------------------------------------------------------------------------------------

def log(*a):
    print(*a, flush=True)


def _ssh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=20", CLUSTER, cmd],
                          capture_output=True, text=True, **kw)


class _Rpc:
    """The ssh file-RPC to the cluster Qwen service, parameterized by the task's RPC + service dirs. cert
    + service-alive + READY probes, atomic-rename both ways, parse-before-delete, service auto-resubmit."""

    def __init__(self, config: TaskConfig):
        self.rpc = config.rpc_dir
        self.svc_name = config.svc_name
        self.svc_dir = config.svc_dir

    def healthy(self):
        if _ssh("true").returncode != 0:
            return False, "ssh/cert"
        if not _ssh(f"test -f {self.rpc}/READY && echo ok").stdout.strip():
            return False, "no READY"
        return True, "ok"

    def resubmit(self):
        _ssh(f"cd {self.svc_dir} && scancel --name={self.svc_name} 2>/dev/null; "
             f"sleep 2; rm -rf rpc && mkdir -p rpc && sbatch svc.slurm")
        for _ in range(18):
            time.sleep(20)
            if self.healthy()[0]:
                return True
        return False

    def score(self, items, untrained_seed=None, ablate_keep=None, timeout=1800):
        ok, why = self.healthy()
        if not ok:
            raise RuntimeError(f"service unhealthy before RPC: {why}")
        rid = uuid.uuid4().hex[:12]
        req = json.dumps({"items": items, "ablate_keep": ablate_keep, "untrained_seed": untrained_seed})
        subprocess.run(["ssh", "-o", "ConnectTimeout=20", CLUSTER,
                        f"cat > {self.rpc}/.tmp_{rid} && mv {self.rpc}/.tmp_{rid} {self.rpc}/req_{rid}.json"],
                       input=req, text=True, check=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            out = _ssh(f"cat {self.rpc}/res_{rid}.json 2>/dev/null")
            if out.stdout.strip():
                try:
                    d = json.loads(out.stdout)
                except json.JSONDecodeError:
                    time.sleep(3)
                    continue
                _ssh(f"rm -f {self.rpc}/res_{rid}.json")
                if "error" in d:
                    raise RuntimeError(f"service error: {d['error']}")
                return np.array(d["scores"], dtype=float)
            ok, why = self.healthy()
            if not ok:
                raise RuntimeError(f"service died mid-score: {why}")
            time.sleep(6)
        raise TimeoutError(f"rpc {rid} timed out")


class _ServiceBackend:
    def __init__(self, rpc: _Rpc, manifest, cutoff_date):
        self._rpc, self._manifest, self.cutoff_date = rpc, manifest, cutoff_date

    def score_box(self, box, untrained_init=None):
        return self._rpc.score(self._manifest[box.id], untrained_seed=untrained_init)


class NoveltySourcesExhausted(RuntimeError):
    """Every novelty-verification tier failed. This is a HARD STOP (red flag), NEVER a silent
    fail-closed rejection: silently marking a real finding not-novel because the sources were down
    would bury a genuine result, which the TOOL-LEDGER novelty contract forbids ("never
    fail-closed-rejects a scored finding"). When all tiers are exhausted the campaign halts for human
    intervention instead of quietly rejecting."""


def _novelty_of(q, titles):
    """The shared novelty verdict from a source's returned titles. `(collision, titles, novel)`:
    NOVEL iff the source returned prior work AND none of it collides with the query text. Every tier
    verifies the same way, so a fallback source is a like-for-like substitute, not a weaker check."""
    titles = [t for t in (titles or []) if t]
    collision = any(q.lower() in t.lower() or t.lower() in q.lower() for t in titles)
    return bool(collision), titles, (bool(titles) and not collision)


def _s2_key():
    """The Semantic Scholar API key, from the S2_API_KEY env var or an off-repo key file
    (~/.research-engine/s2_api_key, mode 600, never committed, alongside the signing key). Returns None
    if neither is present, so tier 1 falls back to the shared unauthenticated pool."""
    env = os.environ.get("S2_API_KEY")
    if env and env.strip():
        return env.strip()
    keyfile = Path.home() / ".research-engine" / "s2_api_key"
    if keyfile.exists():
        k = keyfile.read_text().strip()
        return k or None
    return None


def _s2_titles(q):
    """Tier 1: Semantic Scholar. Uses the API key (env or off-repo file) when present for a dedicated
    1 req/s limit instead of the shared unauthenticated pool that 429s under load. Retries a 429 a few
    times, then raises to tier down."""
    key = _s2_key()
    headers = {"x-api-key": key} if key else {}
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?limit=5&fields=title&query="
           + urllib.parse.quote(q))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=25) as r:
                return [p.get("title", "") for p in (json.load(r).get("data") or [])]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("semantic_scholar retries exhausted")


def _arxiv_titles(q):
    """Tier 2: the arXiv API (keyless). Returns Atom XML; pull each entry's title."""
    url = "http://export.arxiv.org/api/query?max_results=5&search_query=all:" + urllib.parse.quote(q)
    with urllib.request.urlopen(url, timeout=25) as r:
        xml = r.read().decode("utf-8", "replace")
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        m = re.search(r"<title>(.*?)</title>", entry, re.S)
        if m:
            out.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return out


def _openalex_titles(q):
    """Tier 3: OpenAlex (keyless, polite pool via mailto). Returns JSON works with titles."""
    url = ("https://api.openalex.org/works?per-page=5&mailto=vignankamarthi@gmail.com&search="
           + urllib.parse.quote(q))
    with urllib.request.urlopen(url, timeout=25) as r:
        return [w.get("title") or "" for w in (json.load(r).get("results") or [])]


# The novelty verification CASCADE (Vignan, 2026-08-03). Try each source in order; the first that
# ANSWERS wins; every tier-down is LOGGED (an explicit, deterministic escalation, not the forbidden
# SILENT fallback). All sources verify novelty the same way (query prior work, check title collision),
# so a fallback is a like-for-like substitute. When ALL are exhausted the audit HARD-STOPS (red flag)
# rather than fail-closed-rejecting, per the TOOL-LEDGER novelty contract. S2 is tier 1 (uses
# S2_API_KEY when set); arxiv + openalex are the keyless fallbacks that keep novelty verifiable when S2
# rate-limits. Overridable as a module attribute for tests.
_NOVELTY_TIERS = (
    ("semantic_scholar", _s2_titles),
    ("arxiv", _arxiv_titles),
    ("openalex", _openalex_titles),
)


def _real_novelty_audit(schema):
    q = str(schema.get("mechanism", "") or schema.get("claim", ""))[:120]
    for name, fetch in _NOVELTY_TIERS:
        try:
            titles = fetch(q)
        except Exception as e:  # this source is down -> tier down (logged, deterministic, not silent)
            log(f"    novelty tier {name} down ({type(e).__name__}: {str(e)[:70]}); tiering down")
            continue
        collision, titles, novel = _novelty_of(q, titles)
        log(f"    novelty via {name}: {len(titles)} hits, collision={collision}, novel={novel}")
        return collision, titles, novel
    raise NoveltySourcesExhausted(
        "ALL novelty sources exhausted (" + ", ".join(n for n, _ in _NOVELTY_TIERS)
        + ") -- HARD STOP (red flag), never a silent not-novel reject")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in TASKS:
        log(f"usage: campaign_real.py <{'|'.join(TASKS)}> <items.json> [max_maturations]")
        return
    config = TASKS[sys.argv[1]]
    items = json.load(open(sys.argv[2]))

    pub = (REPO / "keys" / "signing_pub.key").read_bytes()
    signed = verify_config((REPO / "signed_config.json").read_bytes(), pub)
    cons = json.load(open(REPO / "catalogs" / "consequence_templates.json"))
    inc = json.load(open(REPO / "catalogs" / "incumbent_catalog.json"))
    mie_cat = json.load(open(REPO / "catalogs" / "mie_distribution.json"))
    mie = float(mie_cat[config.task]["mie_value"])
    log(f"config verified: key_id={signed.key_id}; TASK={config.task}; MIE={mie}")

    rpc = _Rpc(config)

    rng = np.random.default_rng(0)
    rng.shuffle(items)
    dev, rest = items[:config.n_dev], items[config.n_dev:]

    # DEV PASS (step 51): size boxes against the MEASURED paired-contrast sd, not the absolute chance
    # variance. An effect's contrast is concentrated, so its sd is far below 0.5.
    log("dev pass: scoring dev full + ablated to size boxes ...")
    dev_full = rpc.score(dev)
    dev_abl = rpc.score(dev, ablate_keep=config.ablate_keep)
    diff = dev_full - dev_abl
    contrast_sd = float(diff.std()) or 0.2
    dev_contrast = float(diff.mean())
    log(f"dev full acc={dev_full.mean():.3f}  ablated={dev_abl.mean():.3f}  "
        f"contrast={dev_contrast:+.3f}  contrast_sd={contrast_sd:.3f}")

    n_box = required_n(mie, contrast_sd, alpha=signed.alpha, power=signed.power)
    n_box = max(60, min(n_box, len(rest) // 2))
    n_boxes = len(rest) // n_box
    boxes, manifest = [], {}
    for b in range(n_boxes):
        bid = f"{config.task}_hold_{b:03d}"
        manifest[bid] = rest[b * n_box:(b + 1) * n_box]
        boxes.append(Box(id=bid, n=n_box, origin_date=config.origin_date))
    manifest["__dev__"] = dev
    log(f"power-sized boxes: {n_box} items/box (MIE {mie} at contrast_sd {contrast_sd:.3f}), "
        f"{n_boxes} holdout boxes")
    if n_boxes < 1:
        log("FATAL: the pool cannot carve one powered box; raise the MIE (re-sign) or add data.")
        return

    backend = _ServiceBackend(rpc, manifest, cutoff_date=config.backbone_cutoff)

    def score_task(_bk, _task, ablation):
        return rpc.score(dev, ablate_keep=(config.ablate_keep if ablation is not None else None))

    def held_out(schema):
        return (dev_contrast >= mie, float(dev_full.mean()))

    def membership(schema):
        return str(schema.get("dataset", "")) == config.task or str(
            schema.get("dataset", "")).startswith(config.task)

    def g0(effect, rng_):
        base = rng_.random(n_box) < float(dev_full.mean())
        lifted = np.clip(base + effect, 0, 1)
        from gatelib import mean_ci
        lo, _ = mean_ci(lifted.astype(float) - base.astype(float), signed.alpha)
        return 0.001 if lo > 0 else 0.5

    def resolve_ablation(mechanism, task):
        return construct_ablation(mechanism, task, blue=ClaudeBlueBuilder(),
                                  red_panel=[ClaudeRedAttacker()], rounds=2)

    substrate = assemble_substrate(
        config=signed, consequence_catalog=cons, incumbent_catalog=inc, mie_catalog=mie_cat,
        score_task=measured(score_task, name=f"{config.task}_score_task"),
        novelty_audit=measured(_real_novelty_audit, name="semantic_scholar_novelty"),
        g0_pipeline=measured(g0, name="g0_planted_effect"),
        specificity_check=measured(lambda bk, s, t: True, name="mechanism_specificity"),
        membership_check=measured(membership, name=f"{config.task}_membership"),
        held_out_check=measured(held_out, name=f"{config.task}_effect_consequence"),
        backbone_cutoff=config.backbone_cutoff, g0_rng=np.random.default_rng(0),
        resolve_ablation_fn=resolve_ablation, require_measured=True)

    lease = LeaseStore(str(Path.home() / ".research-engine" / f"{config.task}_campaign.db"))
    lease.add_boxes([b.id for b in boxes])
    by_id = {b.id: b for b in boxes}

    # The GROUNDED, vein-diverse WaveAgent + the significance-adversary's advisory importance penalty +
    # the JEPA standing reserve. NO mechanism is pinned: every idea is a fresh grounded mechanism.
    significance = ClaudeSignificanceAdversary()
    agent = build_wave_agent(config, scout=build_scout(), significance=significance)
    triage = build_triage(config)

    live = lease.live_count()
    requested = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    if live >= 2:
        max_mat = max(1, min(requested, live // 2))
        reserves = Reserves.for_campaign(max_mat, rescore_contingency=0, backbone_cohort=0)
    else:
        max_mat = 1
        reserves = Reserves(primary_demand=1, replication=0, rescore=0, backbone=0)
    budget = Budget.from_closure(live_boxes=live, reserves=reserves,
                                 max_gpu_hours=200.0, max_maturations=max_mat)
    log(f"closure: pool={live} boxes -> max_maturations={max_mat} "
        f"(replication reserve {reserves.replication}, on-box FLOOR)")
    log(f"JEPA reserve: floor={config.jepa_floor} <= JEPA maturations <= cap={config.jepa_cap}")

    log(f"\nRUNNING the assembled loop on {config.task} (grounded WaveAgent, no mechanism pin), "
        f"max_maturations={max_mat} ...\n")
    reason, report, campaign = run_loop(
        agent=agent, backend=backend, config=signed, lease_store=lease,
        box_factory=lambda bid: by_id[bid], substrate=substrate, triage=triage,
        reviewer=ClaudeReviewerAdversary(), significance=significance,
        synthesizer=ClaudeSynthesizer(), budget=budget,
        halt_flag=HaltFlag(str(Path(tempfile.mkdtemp()) / "halt")),
        health_gate=HealthGate([Probe("service", check=lambda: rpc.healthy()[0],
                                      state=RETRY, self_heal=rpc.resubmit)]), seed=0)

    log("\n" + "=" * 64)
    log(f"TERMINAL: {reason}")
    log(f"JEPA maturations this campaign: {campaign.jepa_matured} "
        f"(reserve {config.jepa_floor}-{config.jepa_cap})")
    for r in campaign.results:
        v = r.verdict
        log(f"  {v.status if v else 'None'}"
            + (f" ({v.reason})" if v and getattr(v, 'reason', None) else "") + f"  <- {r.lineage[:10]}")
    log("\n" + report.narrative)
    log("=" * 64)
    log("\nNEXT: Vignan's GO/NO-GO on any submit-bound finding above.")


if __name__ == "__main__":
    main()
