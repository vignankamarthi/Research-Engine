# Tool Ledger

> Every runtime dependency the loop has, each with a health probe and a failure STATE. A construction
> artifact, built alongside `PLAN.md`. Generated from a machine-readable signed ledger config (the
> source of truth). Criticality states and the heartbeat interval live in the signed gate config.
> A build rule enforces completeness -- any code path touching an unregistered dependency is a
> build-time error. Status column: LIVE (now), CORE (built during the confirmatory core, PLAN
> Milestones 0-4), DISCOVERY (built with the discovery tier, PLAN Milestones 5-6). CORE/DISCOVERY are
> build PHASES, not PLAN's milestone numbers.

## The four failure states

"No silent fallback" forbids a substitute tool, a skipped step, or continuing past a broken disposer.
A bounded retry against the SAME tool is not a fallback.

- **HALT** -- disposers + integrity. Stop launching, checkpoint, snapshot, escalate. Data/liveness resume on a green re-probe + fingerprint match. Integrity/tamper HALTs need human acknowledgment.
- **QUARANTINE** -- a data/provenance break. Everything since the last green probe of this entry is SUSPECT and recomputed. Unaffected arms continue.
- **RETRY** -- a transient blip on the same tool. Bounded backoff, escalate after N failures.
- **DEGRADE-WITH-BUFFER** -- observability only. Buffer to disk, warn, continue.

## Kill / resume policy

- Preflight probes every entry. A heartbeat re-probes on the signed-config interval (faster for the disposer/integrity tier).
- A HALT snapshot is the full environment fingerprint + resumeFromRunId + durable Optuna/MLflow state, written to a checksummed append-only file on `/work`, INDEPENDENT of Postgres.
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
| Environment-fingerprint equality | no pooling incomparable runs | two runs share the NODE-INVARIANT canonical fingerprint (container digest, lib versions, GPU model + compute capability, determinism flags) before aggregation, so parallel multi-node seeds can pool | fingerprint mismatch | CORE |

## The referee's data channels (HALT / QUARANTINE)

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| Holdout scoring (touch-once) | the confirmatory held-out score | reachable, a direct label read from the experiment identity is DENIED, each matured hypothesis scores the holdout at most once (logged, append-only) | label file readable, or a hypothesis re-scores a spent holdout | HALT | CORE |
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
| Scratch purge | data not silently deleted | files touched within the 30-day window, critical artifacts mirrored to `/work` | a checkpoint or manifest aged out | QUARANTINE | CORE |
| Home disk quota | no truncated writes | free space above a floor before a job writes | quota full, truncated checkpoint | HALT | CORE |

## Data and persistence

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| Postgres (MLflow) | the ledger's backing store | write-then-read-back canary, connection headroom, disk headroom, backup freshness | intermittent write failure (fewer logged seeds, reads as a legitimate result) | HALT | CORE |
| MLflow | provenance, dedup | an in-job canary run with a unique token, asserting the tracking URI, experiment id, and artifact writability | `MLFLOW_TRACKING_URI` unset so jobs write to local `./mlruns` | QUARANTINE | CORE |
| Optuna + RDBStorage | bandit search state | study loads from the DB | DB down | HALT | CORE |
| Langfuse | tracing, visibility | ingestion endpoint healthy, on a SEPARATE instance from the ledger | endpoint down | DEGRADE-WITH-BUFFER | LIVE |
| Backbone cache | assay + experiments | backbones present with hashes bound to the HF revision SHA recorded at first download | missing or SHA mismatch | HALT | CORE |
| uv.lock + DVC | reproducibility roots | lock hash and DVC artifact hashes match the fingerprint | drift | HALT | CORE |

## Research and literature MCPs

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| Semantic Scholar MCP | occupancy, novelty re-audit | a freshness check (a known-recent item returns, corpus release id recorded) | stale cache returns old results, or rate-limited | RETRY | LIVE |
| arXiv MCP | paper mining | freshness check returns hits | brief outage | RETRY | LIVE |
| Scite MCP | believed-claim check | auth valid, a test query returns | OAuth expired or monthly cap hit | defer the NEGATIVE gate for that arm, positive arms continue | LIVE |
| Parallel Research MCP | deep research | `task_status` ok | down | DEGRADE-WITH-BUFFER | LIVE |
| HF Papers + Hub cards | paper-to-artifact bridge, trending recency, backbone cutoff dates | a known-recent daily paper returns and a model-card cutoff is readable | stale or down | RETRY | LIVE |

## The brain and the campaign

| Entry | Purpose | Probe | Break signature | State | Status |
|---|---|---|---|---|---|
| Bun/JS engine | orchestration | `bun test` green in `engine/` | tests red or build drift | HALT | LIVE |
| Generated workflow | run entry | matches a fresh `bun build-workflow.js` | hand-edited or stale | HALT | LIVE |
| Stats libs (scipy, statsmodels, pingouin) | acceptance math (standard BH, no bespoke e-BH) | import plus known-answer tests, differential-tested against a named third-party reference | version drift or a KAT mismatch | HALT | CORE |
| Self-chaining supervisor | carries the campaign | EXACTLY ONE queued successor (two-sided), the campaign GPU-hour cap enforced at SLURM (`GrpTRESMins`), and a durable human-clearable HALT flag honored on wake | zero or more-than-one successor, the cap not scheduler-enforced, or a HALT flag raced | HALT | CORE |
| External dead-man's-switch | catches a silently-dead supervisor | a supervisor heartbeat seen by a Mac cron or SLURM scavenger, independent of the chain | no heartbeat in X hours (the two-sided probe cannot self-observe zero) | HALT + alert | CORE |
| Anthropic API (the driving agent) | setup and read | availability, rate headroom, token-budget remaining | outage or budget exhausted | RETRY | LIVE |
| Context compaction | agent state survives a compaction | campaign state is fully reconstructable from durable stores, nothing load-bearing lives only in agent context | in-context-only state lost at compaction | survivable (supervisor is the source of truth) | LIVE |
| Escalation channel | a halt reaches Vignan | a test escalation is delivered and acknowledged | unmonitored channel, a halt becomes an indefinite stall | HALT if unverified | CORE |
| Running-code git SHA | provenance | the SHA and a clean/dirty flag recorded on every run | dirty tree or an uncommitted run | HALT | CORE |
| Campaign budget | bounded runtime | wall-clock and GPU-hours under the ceiling in the signed config | budget exhausted | end campaign as INCONCLUSIVE | CORE |

## Notes

- A checksum or integrity failure is a fabrication/tamper signal, and it flags every result since the last green probe of that entry, not only the most recent row.
- Every M0/M1 entry gets its probe written test-first. Correctness-shaped probes (in-job canaries, known-answer tests, freshness checks) are the pattern. Liveness alone is not enough for anything a disposer depends on.
