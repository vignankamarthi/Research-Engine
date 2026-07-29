"""The holdout box lease in an ACID store (SQLite locally, Postgres on the cluster).

The box is the serialization key, so two different-lineage maturations cannot both
claim one live box. A per-lineage bank entry enforces the one-grant. Scoring is
two-marker (reserved -> label_read) and two-phase: the verdict-bearing result is
durably staged, then one atomic spend commits it and writes the bank entry. The
lease carries a monotonic generation; the label_read/stage/commit writes are a CAS
against it, so a partitioned orphan whose lease was superseded is fenced out. Resume
is decidable: staged -> re-commit (no re-touch), label_read-no-stage -> burn (a
durable one-grant record), reserved-only -> reclaim."""
from __future__ import annotations

from common.sqlite import connect as _connect
from dataclasses import dataclass


class FenceError(Exception):
    """A write lost the lease-generation CAS (a stale/orphaned holder) or the box is
    not in the expected state."""


@dataclass(frozen=True, slots=True)
class ClaimResult:
    box_id: str
    generation: int


class LeaseStore:
    def __init__(self, path: str):
        self.conn = _connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS boxes("
            " box_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'live',"
            " holder TEXT, lineage TEXT, generation INTEGER NOT NULL DEFAULT 0,"
            " label_read INTEGER NOT NULL DEFAULT 0,"
            " staged_verdict TEXT, staged_score BLOB)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS bank("
            " lineage TEXT PRIMARY KEY, box_id TEXT, verdict TEXT, kind TEXT)"
        )

    def add_boxes(self, box_ids) -> None:
        for b in box_ids:
            self.conn.execute("INSERT OR IGNORE INTO boxes(box_id) VALUES(?)", (b,))

    def claim(self, hypothesis: str, lineage: str):
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            if c.execute("SELECT 1 FROM bank WHERE lineage=?", (lineage,)).fetchone():
                c.execute("COMMIT")
                return None  # one-grant: this lineage already spent/burned a box
            row = c.execute(
                "SELECT box_id FROM boxes WHERE status='live' ORDER BY box_id LIMIT 1"
            ).fetchone()
            if row is None:
                c.execute("COMMIT")
                return None
            box_id = row[0]
            cur = c.execute(
                "UPDATE boxes SET status='reserved', holder=?, lineage=?,"
                " generation=generation+1 WHERE box_id=? AND status='live'",
                (hypothesis, lineage, box_id),
            )
            if cur.rowcount != 1:  # lost the race inside the transaction (should not happen)
                c.execute("ROLLBACK")
                return None
            gen = c.execute("SELECT generation FROM boxes WHERE box_id=?", (box_id,)).fetchone()[0]
            c.execute("COMMIT")
            return ClaimResult(box_id=box_id, generation=gen)
        except Exception:
            c.execute("ROLLBACK")
            raise

    def mark_label_read(self, box_id: str, generation: int) -> None:
        cur = self.conn.execute(
            "UPDATE boxes SET label_read=1 WHERE box_id=? AND generation=? AND status='reserved'",
            (box_id, generation),
        )
        if cur.rowcount != 1:
            raise FenceError(f"stale lease generation for {box_id} (orphan fenced out)")

    def stage(self, box_id: str, generation: int, verdict: str, score: bytes) -> None:
        cur = self.conn.execute(
            "UPDATE boxes SET staged_verdict=?, staged_score=? WHERE box_id=? AND"
            " generation=? AND status='reserved' AND label_read=1",
            (verdict, score, box_id, generation),
        )
        if cur.rowcount != 1:
            raise FenceError(f"cannot stage on {box_id}: not label-read or generation stale")

    def commit(self, box_id: str, generation: int) -> None:
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute(
                "SELECT lineage, staged_verdict FROM boxes WHERE box_id=? AND"
                " generation=? AND status='reserved'",
                (box_id, generation),
            ).fetchone()
            if row is None or row[1] is None:
                c.execute("ROLLBACK")
                raise FenceError(f"cannot commit {box_id}: not staged or generation stale")
            lineage, verdict = row
            c.execute("UPDATE boxes SET status='spent' WHERE box_id=? AND generation=?",
                      (box_id, generation))
            c.execute("INSERT OR REPLACE INTO bank(lineage,box_id,verdict,kind) VALUES(?,?,?, 'spent')",
                      (lineage, box_id, verdict))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    def resume(self) -> dict[str, str]:
        """Reconcile every reserved box, GENERATION-FENCED. Each write predicates on the box's
        snapshot generation AND status='reserved' (like every other CAS in this store), so a box
        a concurrent reclaim/claim moved between the snapshot and the write is skipped, never
        clobbered, and a stale lineage is never written into the bank. Returns {box_id: action}."""
        actions: dict[str, str] = {}
        rows = self.conn.execute(
            "SELECT box_id, generation, lineage, label_read, staged_verdict"
            " FROM boxes WHERE status='reserved'"
        ).fetchall()
        for box_id, generation, lineage, label_read, staged_verdict in rows:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                if staged_verdict is not None:
                    moved = c.execute(
                        "UPDATE boxes SET status='spent'"
                        " WHERE box_id=? AND generation=? AND status='reserved'",
                        (box_id, generation),
                    ).rowcount == 1
                    if moved:
                        c.execute(
                            "INSERT OR REPLACE INTO bank(lineage,box_id,verdict,kind) VALUES(?,?,?, 'spent')",
                            (lineage, box_id, staged_verdict),
                        )
                    actions[box_id] = "recommitted" if moved else "skipped_moved"
                elif label_read:
                    moved = c.execute(
                        "UPDATE boxes SET status='burned'"
                        " WHERE box_id=? AND generation=? AND status='reserved'",
                        (box_id, generation),
                    ).rowcount == 1
                    if moved:
                        c.execute(
                            "INSERT OR REPLACE INTO bank(lineage,box_id,verdict,kind) VALUES(?,?,NULL,'burned')",
                            (lineage, box_id),
                        )
                    actions[box_id] = "burned" if moved else "skipped_moved"
                else:
                    # reserved-only: on the cluster this also requires the SLURM job to be
                    # provably terminal; locally we reclaim directly.
                    moved = c.execute(
                        "UPDATE boxes SET status='live', holder=NULL, lineage=NULL"
                        " WHERE box_id=? AND generation=? AND status='reserved'",
                        (box_id, generation),
                    ).rowcount == 1
                    actions[box_id] = "reclaimed" if moved else "skipped_moved"
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        return actions

    def box_status(self, box_id: str) -> str:
        return self.conn.execute("SELECT status FROM boxes WHERE box_id=?", (box_id,)).fetchone()[0]

    def bank_verdict(self, lineage: str):
        row = self.conn.execute(
            "SELECT box_id, verdict, kind FROM bank WHERE lineage=?", (lineage,)
        ).fetchone()
        return None if row is None else (row[0], row[1], row[2])
