"""Shared SQLite connection helper. One place owns the concurrency hardening so the durable
stores (the experiment ledger and the ACID box-lease) can never drift apart: WAL journaling so
readers do not block a writer, a long busy_timeout so a contended write waits instead of raising
'database is locked', and autocommit (isolation_level=None) so callers manage transactions
explicitly (the lease relies on this for its two-phase protocol)."""
from __future__ import annotations

import sqlite3

_BUSY_TIMEOUT_MS = 30000


def connect(path, timeout: float = 30.0) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=timeout, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return conn
