# Research-Engine

A two-tier discovery to confirmation loop that automates the research process for the SMILE lab
(Prof. Yun Raymond Fu), heavily parallel, to a publishable finding. A generative discovery tier
(Claude Code subagents) proposes and matures hypotheses; a frozen out-of-process referee scores
each once against a touch-once holdout; a human triages the drafted narrative and submits.

## Architecture and design docs

`RESEARCH-LOOP-SPEC.md` is the architecture reference, covering the diagram, the four disposers,
the acceptance gate, and the parameters. `TOOL-LEDGER.md` is the runtime dependency and
failure-state manifest. `ANTIPATTERNS.md` is the hard-stop rules. `PLAN.md` is the build plan.

## Module map (`src/`)

| Package | Role |
|---|---|
| `common/` | Shared primitives, the canonical-digest trust hash and the hardened SQLite helper |
| `gateconfig/` | The offline-signed gate-config trust root (Ed25519 sign/verify, schema invariants) |
| `gatelib/` | The frozen statistical gate library (G0, BH, magnitude + claim-type registry, FLOOR, backbone, mechanism, novelty, consequence) |
| `backend/` | The model-backend abstraction (`Backend` protocol, deterministic `MockBackend`, cluster-fenced `HFBackend`) |
| `referee/` | The trusted confirmatory core (safe deserialization, the schema-normal-form, signed catalogs, the ACID box lease, the runner) |
| `engine/` | The discovery tier + orchestration (Claude Code subagent roles, steering, bandit, supervisor, health gate, ledger, campaign pool) |

## Testing

```
uv run pytest              # the full Mac suite (cluster-marked tests excluded by default)
uv run pytest -m cluster   # cluster-only tests: need AICR + a real checkpoint, run on the cluster
```

The Mac build stays torch-free. `torch`/`transformers` are imported lazily inside
`HFBackend.load()`, behind the `cluster` pytest marker, so nothing GPU-only reaches the local
suite.
