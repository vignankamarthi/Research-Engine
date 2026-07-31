"""BLIND scout isolation (PLAN 57a, ANTIPATTERNS 12). A formation scout works ONLY from the field
via the research stack. If it reads CLAUDE/MEMORY/landscape it inherits the convergence and is no
longer blind, and that vault-inheritance leak (not an ungrounded prompt) is what drove the temporal
monoculture. This module makes the isolation STRUCTURAL: a neutral cwd outside the vault (so no
project CLAUDE.md is inherited), an MCP config filtered to research servers only, and a hard check
that no vault/project text reaches the scout context. The spawn is injected so the contract is
tested without shelling out to `claude`; the default shells to `claude -p` with the scoped config."""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The ONLY MCP servers a blind scout may reach: the research stack. Anything else (browser, supabase,
# the vault's own memory tools) could re-open a path to the convergence and is refused.
RESEARCH_MCP_SERVERS = frozenset({
    "arxiv", "scite", "semantic-scholar", "pubmed", "parallel-research",
})

# Distinctive substrings that betray vault/project inheritance. Deliberately whole tokens (never bare
# "Fu", which collides with "future_work") so a blind prompt is not false-flagged.
VAULT_MARKERS = (
    "SMILE", "Yun Raymond", "Raymond Fu", "Radhakrishnan", "Vignan",
    "CLAUDE.md", "MEMORY.md", "ANTIPATTERNS", "Research-Engine",
    "Northeastern", "video foundation model", "Temporal-RoPE", "RIVeR", "AI4Pain",
)


class ScoutIsolationError(Exception):
    """A blind-scout isolation invariant was violated (vault text, or a cwd inside the vault)."""


@dataclass(frozen=True)
class ScoutSpec:
    """The isolated spawn the scout runs under: a blind prompt, a neutral cwd, the research-only MCP
    allowlist actually granted, and strict scoping so no ambient server is inherited."""
    prompt: str
    cwd: str
    mcp_servers: tuple
    strict_mcp: bool = True
    mcp_config_path: str | None = None


def is_within(path, root) -> bool:
    """True if `path` is at or under `root` (resolved). The vault-inheritance guard rests on this."""
    p, r = Path(path).resolve(), Path(root).resolve()
    return p == r or r in p.parents


def neutral_scratch_dir(vault_root, base=None) -> Path:
    """Create a fresh scratch dir GUARANTEED outside the vault, so the spawned scout inherits no
    project CLAUDE.md/MEMORY.md. A `base` that itself sits inside the vault is refused."""
    if base is not None and is_within(base, vault_root):
        raise ScoutIsolationError(f"scratch base {base!r} is inside the vault {vault_root!r}")
    d = Path(tempfile.mkdtemp(prefix="blind-scout-", dir=base))
    if is_within(d, vault_root):
        d.rmdir()
        raise ScoutIsolationError(f"scratch dir {d} resolved inside the vault {vault_root!r}")
    return d


def assert_no_vault_leak(text, markers=VAULT_MARKERS) -> None:
    """Raise if any vault/project marker appears (case-insensitive). This is the load-bearing blind
    check: the scout's prompt AND its context must carry none of the convergence."""
    low = text.lower()
    hits = [m for m in markers if m.lower() in low]
    if hits:
        raise ScoutIsolationError(f"vault text leaked into scout context: {hits}")


def filter_mcp_config(mcp_config: dict, allow=RESEARCH_MCP_SERVERS) -> dict:
    """Return an MCP config keeping only the research servers. A non-research server (browser, db,
    the vault's memory tools) is dropped so the scout cannot reach back into the convergence."""
    servers = mcp_config.get("mcpServers", {})
    return {"mcpServers": {k: v for k, v in servers.items() if k in allow}}


def write_scout_mcp_config(scratch_dir, mcp_config: dict) -> Path:
    """Filter `mcp_config` to research servers and write it into the scratch dir; return its path.
    The default spawn passes this with `--mcp-config ... --strict-mcp-config`."""
    path = Path(scratch_dir) / "scout-mcp.json"
    path.write_text(json.dumps(filter_mcp_config(mcp_config), indent=2))
    return path


def build_scout_runner(scratch_dir, *, vault_root, spawn=None, mcp_config_path=None,
                       mcp_servers=RESEARCH_MCP_SERVERS, model=None, timeout_s=240.0,
                       claude_bin="claude"):
    """Return a `runner(prompt) -> str` that spawns a BLIND scout: cwd in the neutral scratch dir,
    MCP scoped to research servers only, and a pre-spawn guard that refuses any prompt (template OR
    context) carrying vault text. `spawn` is injected for tests; the default shells to `claude -p`."""
    if is_within(scratch_dir, vault_root):
        raise ScoutIsolationError(f"scout cwd {scratch_dir} is inside the vault {vault_root!r}")
    spawn = spawn or _default_scout_spawn
    allow = tuple(sorted(mcp_servers))

    def runner(prompt: str) -> str:
        assert_no_vault_leak(prompt)  # the enforcement point: no convergence reaches the scout
        spec = ScoutSpec(prompt=prompt, cwd=str(scratch_dir), mcp_servers=allow,
                         strict_mcp=True, mcp_config_path=mcp_config_path)
        return spawn(spec, model=model, timeout_s=timeout_s, claude_bin=claude_bin)

    return runner


def _default_scout_spawn(spec: ScoutSpec, *, model, timeout_s, claude_bin) -> str:
    """Shell to `claude -p` in the neutral cwd with the research-only MCP config strictly scoped."""
    cmd = [claude_bin, "-p", spec.prompt, "--output-format", "text"]
    if spec.mcp_config_path:
        cmd += ["--mcp-config", spec.mcp_config_path]
    if spec.strict_mcp:
        cmd += ["--strict-mcp-config"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, cwd=spec.cwd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ScoutIsolationError(f"blind scout spawn failed: {e}") from e
    if proc.returncode != 0:
        raise ScoutIsolationError(f"blind scout spawn failed (rc={proc.returncode}): "
                                  f"{proc.stderr.strip()}")
    return proc.stdout.strip()
