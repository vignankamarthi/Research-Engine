# Tool Ledger

> Every runtime dependency the loop has, each with a health probe and a failure STATE. A construction
> artifact, built alongside `PLAN.md`. Generated from a machine-readable signed ledger config (the
> source of truth). Criticality states and the heartbeat interval live in the signed gate config.
> A build rule enforces completeness -- any code path touching an unregistered dependency is a
> build-time error. Status column: LIVE (now), CORE (built during the confirmatory core, PLAN
> Milestones 0-4), DISCOVERY (built with the discovery tier, PLAN Milestones 5-6). CORE/DISCOVERY are
> build PHASES, not PLAN's milestone numbers.

> IN-CODE TODAY vs PRODUCTION-DEPLOYMENT. The Mac build's actual dependencies are `numpy`, `scipy`,
> `statsmodels`, `optuna`, and `cryptography`, plus the durable stores backed by SQLite (`common.sqlite`)
> and local checksummed files, and the Claude Code CLI for the discovery subagents. MLflow, Postgres,
> Langfuse, DVC, Apptainer, and SLURM are the production-deployment targets for the same roles, not
> current in-code deps. Rows below name the production tool; the Mac stand-in is noted where it differs.

## The four failure states

"No silent fallback" forbids a substitute tool, a skipped step, or continuing past a broken disposer.
A bounded retry against the SAME tool is not a fallback. The response is DETERMINISTIC. Every fault
first attempts a bounded self-heal, and if that does not clear the health probe it HALTs and pages
Vignan. The system either fixes itself or stops, it never diverges.

- **HALT** -- disposers + integrity. Stop launching, checkpoint, snapshot, escalate. Data/liveness resume on a green re-probe + fingerprint match. Integrity/tamper HALTs need human acknowledgment.
- **QUARANTINE** -- a data/provenance break. Everything since the last green probe of this entry is SUSPECT and recomputed, capped at N recompute cycles. If the same probe re-trips after N, or the break is one a recompute cannot fix (egress cut, data purged), it PROMOTES to HALT + escalate rather than spinning. Unaffected arms continue.
- **RETRY** -- a transient blip on the same tool. Bounded backoff, escalate after N failures.
- **DEGRADE-WITH-BUFFER** -- observability only. Buffer to disk, warn, continue.

## Kill / resume policy

- Preflight probes every entry. A heartbeat re-probes on the signed-config interval (faster for the disposer/integrity tier).
- A HALT snapshot is the full environment fingerprint + resumeFromRunId + durable Optuna/MLflow state, written to a checksummed append-only file on `/work`. The two-phase box records (marker, staged score, spend, bank entry) live in ONE ACID store (SQLite today via `common.sqlite`, Postgres in production), and the `/work` snapshot is advisory reconstruction only, never a two-phase record, so the resume decision is never split across stores.
- On resume the fingerprint is re-verified byte-for-byte. A mismatch quarantines the affected pre-registrations.
- The self-chaining supervisor carries the campaign across cert-expiry, maintenance-window, and compaction gaps.

## Integrity self-checks (HALT)

Verified INSIDE the trusted process at every use (a preflight check is a TOCTOU window given arbitrary code execution on the node).

| Entry | Purpose | Probe (at use) | Break signature | Status |
|---|---|---|---|---|
| Frozen gate library tests | all disposal | full suite green + mutation score above threshold | any gate test red | CORE |
| Gate-config signature | goalpost-lock | signature verifies against the offline public key baked into the container | not signed by Vignan's Mac key | CORE |
| Control-code write-lock | no self-modification | running code matches the committed baseline SHA | any unauthorized diff | CORE |
| Control catalog hash | no free-authored controls | catalog hash unchanged and signed | hash changed without a signed edit | CORE |
| Environment-fingerprint equality | no pooling incomparable runs | two runs share the node-invariant canonical fingerprint (container digest, lib versions, GPU model + compute capability, determinism flags), confirmatory scoring is SINGLE-GPU by default so a once-scored box needs no NCCL/GPU-count pin and a cluster driver bump does not strand a resume, any unavoidable multi-GPU confirmation pins those and a resume mismatch on an un-scored box PAGES rather than quarantining | fingerprint mismatch | CORE |

## The referee's data channels (HALT / QUARANTINE)

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| Holdout boxes (fresh disjoint, lease) | the confirmatory held-out score | reachable, pre-split into disjoint powered boxes, label read DENIED, the allocator ATOMICALLY claims the BOX (serialization key) + a per-lineage reservation, staged in ONE ACID store (reserved pre-launch, label-read committed AND read-back-verified BEFORE the first label byte else HALT, staged score, atomic spend + bank entry, nothing bandit/MLflow-visible before the spend), the label-read commit a CAS against the box's monotonic lease GENERATION (a reclaim bumps it, so a partitioned orphan the controller falsely marked terminal fails the CAS and HALTs before reading), on resume staged->re-commit, label-read->burn + a durable burned-re-score-pending lineage record, reserved-only->reclaim ONLY after sacct/squeue shows the reserving job TERMINAL, none->reclaim | a box readable outside its score, TWO maturations of ANY lineage on one live box, a re-scored box, a reclaim of a box whose job is not provably dead, or a leased box in no defined state | HALT | CORE |
| Dataset signed manifest | executed-not-fabricated | manifest signature verifies and its hash matches the one pinned in the signed gate config | checksum or signature mismatch (a fabrication or swap signal) | QUARANTINE | CORE |
| Pre-registration store + clock anchor | no backdated pre-registration | the pre-reg hash exists in the remote append-only store with a server-side timestamp before any run | local timestamp with no remote anchor | HALT | CORE |

## Cluster and execution

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| AICR SSH cert | agent setup and read | `ssh aicr true` returns 0 | Permission denied (cert expired or 3rd-Tuesday window) | RETRY (wait through window, supervisor carries progress) | LIVE |
| SLURM scheduler | job submission | `sinfo` responds and the account can submit | scheduler unreachable | HALT | LIVE |
| SLURM account state | no stuck arms | fairshare, held-job, drained-node, and concurrency limits queried, plus a per-arm wall-clock timeout | an arm hangs vs merely queued | QUARANTINE the arm | CORE |
| SLURM-layer blast radius | fail-closed guard | submit filter and QOS caps reject a known-bad script (time, GPU, and write-path limits enforced by SLURM, not the agent) | caps missing or bypassable | HALT | CORE |
| Apptainer .sif digest | frozen environment | running container digest matches the fingerprint | digest drift | HALT | CORE |
| HF Hub + token | model/backbone downloads | `huggingface-cli whoami`, a cached-model load, and every model pinned by revision SHA | 401 gated, or a tag/branch that moved | RETRY (download), HALT (SHA mismatch) | LIVE |
| GPU node health | correct numerics | job lands on the assumed partition, device name and compute capability asserted in-job | node failure (RETRY requeue) vs an ECC or XID error (HALT, wrong numbers) | RETRY / HALT | CORE |
| torch Blackwell build | model runs | in-job numerical known-answer test against a golden tensor, plus device and version asserts | silent CPU or PTX-JIT fallback, or a numeric drift | HALT | CORE |
| transformers | model loading | import plus pinned version | API contract break | HALT | CORE |
| Network egress (compute nodes) | pulls and writes reach out | a canary fetch and a canary MLflow write from inside a job | egress blocked, silent cache use | QUARANTINE | CORE |
| Scratch purge | data not silently deleted | ALL datasets + cluster ops live under `/work/neu/p2026_0016_neu` (snapshot-backed, ratified 2026-07-30), NOT `/scratch` which AICR purges on inactivity; probe that no work-product path resolves under `/scratch` and that `/work` free space is above a floor | a dataset / checkpoint / manifest found on `/scratch`, or a `/work` path aged out (should not happen) | QUARANTINE | CORE |
| Home disk quota | no truncated writes | free space above a floor before a job writes | quota full, truncated checkpoint | HALT | CORE |

## Data and persistence

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| SQLite ledger (Postgres/MLflow in production) | the ledger's backing store | write-then-read-back canary, connection headroom, disk headroom, backup freshness | intermittent write failure (fewer logged seeds, reads as a legitimate result) | HALT | CORE |
| MLflow | provenance, dedup | an in-job canary run with a unique token, asserting the tracking URI, experiment id, and artifact writability | `MLFLOW_TRACKING_URI` unset so jobs write to local `./mlruns` | QUARANTINE | CORE |
| Optuna + RDBStorage | bandit search state | study loads from the DB | DB down | HALT | CORE |
| Negative bank | precise failure memory, gates a fresh box + steers discovery | durable append-only store loads, each entry carries the exact claim + failure mode + conditions + box spent, the LINEAGE KEY is a deterministic function of the frozen schema computed IN the trusted process (never the generative tier), the bank entry and box-spend written in ONE transaction | store unreachable, an entry missing its structured fields, lineage judged outside the trusted process, or a bank/allocator mismatch | HALT | CORE |
| Langfuse | tracing, visibility | ingestion endpoint healthy, on a SEPARATE instance from the ledger | endpoint down | DEGRADE-WITH-BUFFER | LIVE |
| Backbone cache | assay + experiments | backbones present with hashes bound to the HF revision SHA recorded at first download | missing or SHA mismatch | HALT | CORE |
| uv.lock + DVC | reproducibility roots | lock hash and DVC artifact hashes match the fingerprint | drift | HALT | CORE |

## Research and literature MCPs

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| Semantic Scholar MCP | occupancy, novelty re-audit (PRE-allocation), AND grounded generation (scouts mine the derivative veins: limitations / future-work / contradictions), Milestone 8 step 51 | a freshness check (a known-recent item returns, corpus release id recorded) | stale cache returns old results, or rate-limited | RETRY (novelty runs before the box is leased, so a stale corpus defers the check, never fail-closed-rejects a scored finding; a stale corpus at GENERATION time only narrows breadth, never a validity risk) | LIVE |
| arXiv MCP | paper mining for grounded generation (vein sourcing, step 51) + novelty | freshness check returns hits | brief outage | RETRY | LIVE |
| Scite MCP | believed-claim check | auth valid, a test query returns | OAuth expired or monthly cap hit | park THAT ONE finding in a provisional "null pending certification" bucket for human triage, NEVER blocks the pool (every other finding routes and assembles normally), a persistent cap ESCALATES | LIVE |
| Parallel Research MCP | deep research | `task_status` ok | down | DEGRADE-WITH-BUFFER | LIVE |
| HF Papers + Hub cards | paper-to-artifact bridge, trending recency, backbone cutoff dates | a known-recent daily paper returns and a model-card cutoff is readable | stale or down | RETRY | LIVE |

## The brain and the campaign

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| Python engine (`src/engine`) | orchestration | `uv run pytest` green | tests red | HALT | LIVE |
| Claude Code subagents (`claude -p`) | discovery agents: the grounded-generation scouts (51), the reviewer + significance adversaries (52), and the synthesizer (53), plus the propose / mature / frame roles already live | a real proposal returns and parses | CLI failure or unparseable output | RETRY then HALT | LIVE (propose / mature / frame; scouts + adversaries + synthesizer land in Milestone 8) |
| Ablation-construction subagents (`claude -p` blue / red) | build + adversarially verify the per-idea mechanism ablation (3c) | blue returns a parseable primitive composition, the red panel returns verdicts, the loop converges within K rounds | CLI failure, unparseable output, or no convergence in K rounds | RETRY on CLI failure then HALT; NO-CONVERGENCE marks the idea "no clean ablation found" and fails the mechanism gate closed | LIVE (`engine.ablation_construction`, red's concrete control-experiment backing is a cluster enhancement) |
| Ablation primitive library | the vetted idea-AGNOSTIC removal ops blue composes (spectral mask, subspace projection, channel zeroing) | each primitive passes its specificity self-test on a synthetic control (removes its target, moves nothing where the target is absent) | a primitive fails its self-test, or the library cannot express a named mechanism | the failing primitive is disabled (fail-closed); an inexpressible mechanism goes to blue for a new vetted primitive, else "no clean ablation found" | LIVE (`engine.ablation_primitives`, 3 vetted primitives) |
| Stats libs (scipy, statsmodels, pingouin) | acceptance math (standard BH, no bespoke e-BH) | import plus known-answer tests, differential-tested against a named third-party reference | version drift or a KAT mismatch | HALT | CORE |
| MIE effect-size distribution | the external interest anchor | a signed distribution of recent accepted-paper effect sizes per task with verified provenance + signature, a re-sign/refresh policy per cycle, a HIGH percentile (top-quartile) and a stated fallback for a task with no distribution | missing, unsigned, stale beyond policy, or a no-distribution task with no fallback | HALT | CORE |
| Incumbent catalog | the foundational-comparison bar | a signed per-task catalog whose entry is the STRONGEST provenance-verified published held-out result at campaign start, verified provenance + signature, per-cycle re-sign/refresh, not a strawman | missing, unsigned, stale, or a strawman-flagged entry | HALT | CORE |
| Self-chaining supervisor | carries the campaign + guarantees termination | EXACTLY ONE queued successor (two-sided), the campaign GPU-hour cap enforced at SLURM (`GrpTRESMins`), the base-case halt evaluated on every wake (fires when GPU-hours OR LIVE boxes OR max-maturations are spent, with the canonical `live_boxes >= per-maturation demand + per-family replication reserve + correlated re-score-and-burn contingency` held), and a durable human-clearable HALT flag honored on wake | zero or more-than-one successor, the cap not scheduler-enforced, the base case not evaluated on wake, the coverage invariant violated, or a HALT flag raced | HALT | CORE |
| External dead-man's-switch | catches a silently-dead supervisor | a supervisor heartbeat seen by a genuinely ALWAYS-ON VM (never the laptop, never a preemptible scavenger, since a preemption would masquerade as death), independent of the chain, the miss-alert routed through TWO independent transports DISJOINT from the escalation path plus a positive alerting-OK heartbeat Vignan watches | no heartbeat in X hours (the two-sided probe cannot self-observe zero) | HALT + alert | CORE |
| Anthropic API (the driving agent) | setup and read | availability, rate headroom, token-budget remaining | outage or budget exhausted | RETRY | LIVE |
| Context compaction | agent state survives a compaction | campaign state is fully reconstructable from durable stores, nothing load-bearing lives only in agent context | in-context-only state lost at compaction | survivable (supervisor is the source of truth) | LIVE |
| Escalation channel | a halt reaches Vignan | a test escalation is delivered and acknowledged over TWO independent transports, with a periodic positive alerting-OK beat so absence of the beat is itself the signal | unmonitored channel, a single transport, or a halt becomes an indefinite stall | HALT if unverified | CORE |
| Running-code git SHA | provenance | the SHA and a clean/dirty flag recorded on every run | dirty tree or an uncommitted run | HALT | CORE |
| Campaign budget + box count | bounded runtime, the base case | wall-clock, GPU-hours, and LIVE boxes under the signed ceilings, CANDIDATE-INCLUSIVE admission on GPU-hours AND wallclock (spent + drain(running) + candidate walltime x GPUs <= hard cap where drain(running) = sum of running jobs' SLURM TIME-LIMIT reservations, candidate walltime <= 24h) so no confirmatory job dies mid-score, `live_boxes >= per-maturation demand + per-family replication reserve + correlated re-score-and-burn contingency + backbone-cohort reserve` AND a matching GPU-HOUR reserve for the held-back scores (the GPU-hour base case fires at hard_cap minus that reserve, admission checked against the reduced ceiling) config-validated to CLOSE before the campaign AND re-validated on any ceiling raise (reserves held back so the base case fires while they remain) | any ceiling hit | base-case halt, drain in-flight, selection-correct (threshold + replication gate) + route the pool per lineage/family, resumable only when the ceiling is raised in the signed config AND the SLURM QOS AND the closure validator re-passes | CORE |

## Notes

- A checksum or integrity failure is a fabrication/tamper signal, and it flags every result since the last green probe of that entry, not only the most recent row.
- Every CORE entry gets its probe written test-first. Correctness-shaped probes (in-job canaries, known-answer tests, freshness checks) are the pattern. Liveness alone is not enough for anything a disposer depends on.
- The mechanical-invariant guards are the gate-library's own invariant checks (built test-first with the gate library, PLAN Milestone 2), verified in the trusted process at use, not a separate subsystem.
- The one schema-normal-form (deriving control set + semantic lineage key + magnitude gate) is the trust-concentration point, so the Milestone-4 adversarial suite must attack IT directly (phrasing-to-weaker-gauntlet, equivalent-claim-to-different-lineage, dataset-alias collisions), not only the gates it derives.
- The referee's bundle of gate inputs is produced by the SUBSTRATE (`engine.substrate`), not by the discovery agent. The agent contributes only `believed_claim`. Every other input (G0 detectability, mechanism ablation, novelty audit, backbone check, consequence) is a measured experiment stamped with executed-not-fabricated provenance, so the CONFIRMED / FAILED verdict is judged on evidence no party graded for itself. The real experiment callables score through `HFBackend` and query the research MCPs on the cluster.
