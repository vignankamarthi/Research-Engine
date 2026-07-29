"""Neutral Claude Code subagent plumbing, shared by every subagent-backed role (the Tier-1
agent and the discovery adversary/synthesis roles). One place owns: shelling to `claude -p`
(Max subscription, no API key), tolerant JSON extraction, and the runner-injection + `_ask_json`
scaffold. Roles subclass `ClaudeRoleBase`; tests inject a canned runner for a free, deterministic
suite."""
from __future__ import annotations

import json
import subprocess


class ClaudeAgentError(Exception):
    pass


def default_cli_runner(prompt: str, model: str | None = None, timeout_s: float = 240.0,
                       claude_bin: str = "claude") -> str:
    """Shell out to `claude -p` headless (Max subscription, no API key)."""
    cmd = [claude_bin, "-p", prompt, "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ClaudeAgentError(f"claude CLI invocation failed: {e}") from e
    if proc.returncode != 0:
        raise ClaudeAgentError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout.strip()


def extract_json(text: str):
    """Parse a JSON object/array out of an LLM reply, tolerating ``` fences and surrounding
    prose. Raises ClaudeAgentError if nothing parses."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ClaudeAgentError(f"no JSON found in agent output: {text[:200]!r}")


class ClaudeRoleBase:
    """Shared scaffold for a Claude Code subagent role: runner injection + JSON asking. The
    default runner shells to `claude -p`; a test injects a canned runner."""

    def __init__(self, runner=None, model: str | None = None, timeout_s: float = 240.0,
                 claude_bin: str = "claude"):
        self._model = model
        self._timeout = timeout_s
        self._bin = claude_bin
        self._runner = runner or self._default_runner

    def _default_runner(self, prompt: str) -> str:
        return default_cli_runner(prompt, self._model, self._timeout, self._bin)

    def _ask_json(self, prompt: str):
        return extract_json(self._runner(prompt))
