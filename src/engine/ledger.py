"""The durable experiment ledger. A thin, JSON-only key/value provenance store the
campaign writes metrics and artifacts into. The real deployment backs this with MLflow
on Postgres; `SQLiteLedger` is the Mac-buildable durable implementation and the one the
tests exercise. `canary_probe` is the two-sided write-read check the health gate runs to
prove the store is alive before trusting it."""
from __future__ import annotations

import contextlib
import json
from typing import Any, Protocol, runtime_checkable

from common.sqlite import connect as _connect

_CANARY_RUN = "__canary__"
_CANARY_KEY = "probe"
_CANARY_TOKEN = "canary-ok"


@runtime_checkable
class Ledger(Protocol):
    def log(self, run_id: str, key: str, value: Any) -> None: ...
    def read(self, run_id: str, key: str) -> Any: ...


class SQLiteLedger:
    """Durable, JSON-serialized (never pickle), last-write-wins per (run_id, key)."""

    def __init__(self, path):
        self._path = str(path)
        # closing() so a raising statement can't leak a WAL connection handle. The shared helper
        # opens in autocommit, so single-statement writes persist with no explicit commit().
        with contextlib.closing(_connect(self._path)) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS ledger "
                "(run_id TEXT, key TEXT, value TEXT, PRIMARY KEY (run_id, key))"
            )

    def log(self, run_id: str, key: str, value: Any) -> None:
        with contextlib.closing(_connect(self._path)) as con:
            con.execute(
                "INSERT OR REPLACE INTO ledger (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, json.dumps(value)),
            )

    def read(self, run_id: str, key: str) -> Any:
        with contextlib.closing(_connect(self._path)) as con:
            row = con.execute(
                "SELECT value FROM ledger WHERE run_id = ? AND key = ?", (run_id, key)
            ).fetchone()
        return json.loads(row[0]) if row is not None else None


def canary_probe(ledger: Ledger) -> bool:
    """True iff the ledger round-trips a written sentinel AND returns None for a key that
    was never written. The second leg catches a store that fabricates reads."""
    ledger.log(_CANARY_RUN, _CANARY_KEY, _CANARY_TOKEN)
    if ledger.read(_CANARY_RUN, _CANARY_KEY) != _CANARY_TOKEN:
        return False
    if ledger.read(_CANARY_RUN, "__never_written__") is not None:
        return False
    return True
