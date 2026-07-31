"""BLIND scout isolation (PLAN 57a, ANTIPATTERNS 12). A formation scout must work only from the
field via the research stack: a neutral cwd OUTSIDE the vault, MCP scoped to research servers only,
and a prompt stripped of the domain-steering line. The vault-inheritance leak (not an ungrounded
prompt) is the real cause of the temporal drift, so the load-bearing test asserts no vault/project
text reaches the scout context. The spawn is injected, so the suite validates the isolation contract
without shelling out to `claude`."""
import json

import pytest

from engine.claude_agent import ClaudeCodeAgent
from engine.scout_isolation import (
    RESEARCH_MCP_SERVERS,
    ScoutIsolationError,
    ScoutSpec,
    assert_no_vault_leak,
    build_scout_runner,
    filter_mcp_config,
    is_within,
    neutral_scratch_dir,
)

VAULT_ROOT = "/Users/x/Desktop/Work/Personal Projects/2025-2026/2026 Co-Op/SMILE Lab Work"


def test_neutral_scratch_dir_is_outside_vault():
    d = neutral_scratch_dir(VAULT_ROOT)
    try:
        assert not is_within(d, VAULT_ROOT)
    finally:
        d.rmdir()


def test_scratch_dir_inside_vault_is_rejected(tmp_path):
    # asking for a scratch dir based INSIDE the vault is a hard error (it would inherit CLAUDE.md)
    with pytest.raises(ScoutIsolationError):
        neutral_scratch_dir(str(tmp_path), base=str(tmp_path))


def test_assert_no_vault_leak_raises_on_vault_text():
    with pytest.raises(ScoutIsolationError):
        assert_no_vault_leak("propose a hypothesis for the SMILE lab video models")
    with pytest.raises(ScoutIsolationError):
        assert_no_vault_leak("work led by Prof. Yun Raymond Fu")


def test_assert_no_vault_leak_passes_on_blind_text():
    assert_no_vault_leak("Generate one falsifiable hypothesis from the retrieved literature.")


def test_filter_mcp_config_keeps_only_research_servers():
    full = {"mcpServers": {
        "arxiv": {"command": "uvx"}, "scite": {"command": "x"},
        "supabase": {"command": "y"}, "claude-in-chrome": {"command": "z"},
    }}
    scoped = filter_mcp_config(full)
    assert set(scoped["mcpServers"]) <= RESEARCH_MCP_SERVERS
    assert "arxiv" in scoped["mcpServers"] and "supabase" not in scoped["mcpServers"]


def test_scout_runner_spawns_outside_vault_with_scoped_mcp():
    captured = {}

    def spawn(spec: ScoutSpec, **kw):
        captured["spec"] = spec
        return "{}"

    d = neutral_scratch_dir(VAULT_ROOT)
    try:
        runner = build_scout_runner(d, spawn=spawn, vault_root=VAULT_ROOT)
        runner("Generate one falsifiable hypothesis from the retrieved literature.")
        spec = captured["spec"]
        assert not is_within(spec.cwd, VAULT_ROOT)
        assert set(spec.mcp_servers) <= RESEARCH_MCP_SERVERS
        assert spec.strict_mcp is True
    finally:
        d.rmdir()


def test_scout_runner_rejects_prompt_carrying_vault_text():
    d = neutral_scratch_dir(VAULT_ROOT)
    try:
        runner = build_scout_runner(d, spawn=lambda spec, **kw: "{}", vault_root=VAULT_ROOT)
        with pytest.raises(ScoutIsolationError):
            runner("propose for the SMILE lab")
    finally:
        d.rmdir()


def test_blind_agent_prompt_has_no_vault_text_but_default_does():
    # the blind scout's propose prompt is clean; the default (lab-named) prompt is NOT -- that is
    # exactly the domain-steering line the blind path strips.
    captured = {}
    blind = ClaudeCodeAgent(runner=lambda p: captured.setdefault("p", p) or "{}", blind=True)
    blind.propose({"vein": "limitations", "negative_bank": []})
    assert_no_vault_leak(captured["p"])

    captured.clear()
    default = ClaudeCodeAgent(runner=lambda p: captured.setdefault("p", p) or "{}")
    default.propose({"vein": "limitations", "negative_bank": []})
    with pytest.raises(ScoutIsolationError):
        assert_no_vault_leak(captured["p"])


def test_blind_agent_context_leak_is_caught_by_runner():
    # even a clean prompt template is guarded: if vault text rides in via the context, the runner
    # refuses to spawn (the enforcement point for ANTIPATTERNS 12).
    d = neutral_scratch_dir(VAULT_ROOT)
    try:
        runner = build_scout_runner(d, spawn=lambda spec, **kw: "{}", vault_root=VAULT_ROOT)
        leaky = "Generate a hypothesis. Context: " + json.dumps({"framing_draft": "SMILE lab thesis"})
        with pytest.raises(ScoutIsolationError):
            runner(leaky)
    finally:
        d.rmdir()
