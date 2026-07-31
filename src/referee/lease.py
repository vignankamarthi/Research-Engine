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
            " holder TEXT, lineage TEXT, purpose TEXT NOT NULL DEFAULT 'primary',"
            " generation INTEGER NOT NULL DEFAULT 0,"
            " label_read INTEGER NOT NULL DEFAULT 0,"
            " staged_verdict TEXT, staged_score BLOB)"
        )
        # The one-grant is keyed on (lineage, PURPOSE), so a lineage's mandatory second-box REPLICATION
        # and its one guarded crash RE-SCORE can each draw a FRESH box, while a second PRIMARY (or a
        # second replication / rescore of the same purpose) is still barred. Without the purpose the
        # lineage was a PRIMARY KEY and no finding could ever replicate, so none could become submit-bound.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS bank("
            " lineage TEXT, purpose TEXT NOT NULL DEFAULT 'primary', box_id TEXT, verdict TEXT,"
            " kind TEXT, PRIMARY KEY(lineage, purpose))"
        )

    def add_boxes(self, box_ids) -> None:
        for b in box_ids:
            self.conn.execute("INSERT OR IGNORE INTO boxes(box_id) VALUES(?)", (b,))

    def claim(self, hypothesis: str, lineage: str, purpose: str = "primary"):
        """Claim a FRESH live box for (lineage, purpose). The one-grant is per PURPOSE, so a
        `replication` or a `rescore` draws a fresh box even after the `primary` spent one, while a
        second grant of the SAME purpose is barred. A `rescore` is allowed ONLY when a prior burned
        record exists for the lineage (a supervisor-attested crash), so the one guarded re-score
        stays exactly one."""
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            if c.execute("SELECT 1 FROM bank WHERE lineage=? AND purpose=?",
                         (lineage, purpose)).fetchone():
                c.execute("COMMIT")
                return None  # one-grant: this (lineage, purpose) already spent/burned a box
            if purpose == "rescore" and not c.execute(
                    "SELECT 1 FROM bank WHERE lineage=? AND kind='burned'", (lineage,)).fetchone():
                c.execute("COMMIT")
                return None  # a re-score is only granted against a prior burned (crashed) box
            row = c.execute(
                "SELECT box_id FROM boxes WHERE status='live' ORDER BY box_id LIMIT 1"
            ).fetchone()
            if row is None:
                c.execute("COMMIT")
                return None
            box_id = row[0]
            cur = c.execute(
                "UPDATE boxes SET status='reserved', holder=?, lineage=?, purpose=?,"
                " generation=generation+1 WHERE box_id=? AND status='live'",
                (hypothesis, lineage, purpose, box_id),
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
                "SELECT lineage, purpose, staged_verdict FROM boxes WHERE box_id=? AND"
                " generation=? AND status='reserved'",
                (box_id, generation),
            ).fetchone()
            if row is None or row[2] is None:
                c.execute("ROLLBACK")
                raise FenceError(f"cannot commit {box_id}: not staged or generation stale")
            lineage, purpose, verdict = row
            c.execute("UPDATE boxes SET status='spent' WHERE box_id=? AND generation=?",
                      (box_id, generation))
            c.execute(
                "INSERT OR REPLACE INTO bank(lineage,purpose,box_id,verdict,kind) VALUES(?,?,?,?, 'spent')",
                (lineage, purpose, box_id, verdict))
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
            "SELECT box_id, generation, lineage, purpose, label_read, staged_verdict"
            " FROM boxes WHERE status='reserved'"
        ).fetchall()
        for box_id, generation, lineage, purpose, label_read, staged_verdict in rows:
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
                            "INSERT OR REPLACE INTO bank(lineage,purpose,box_id,verdict,kind)"
                            " VALUES(?,?,?,?, 'spent')",
                            (lineage, purpose, box_id, staged_verdict),
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
                            "INSERT OR REPLACE INTO bank(lineage,purpose,box_id,verdict,kind)"
                            " VALUES(?,?,?,NULL,'burned')",
                            (lineage, purpose, box_id),
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

    def live_count(self) -> int:
        """Boxes still 'live' (unclaimed). The closure validator reads this as ground truth for the
        pool size at campaign start AND on a ceiling raise, rather than trusting a passed-in count."""
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM boxes WHERE status='live'").fetchone()[0])

    def counts(self):
        """(boxes_spent, maturations) reconstructed from the durable bank, for a crash-resume: every
        bank row spent (or burned) a box, and a PRIMARY row is a matured hypothesis. The supervisor
        rebuilds its SupervisorState from this so a restart neither re-scores a spent box (the
        (lineage, purpose) one-grant bars it) nor exceeds the maturation budget across restarts."""
        boxes = self.conn.execute("SELECT COUNT(*) FROM bank").fetchone()[0]
        matured = self.conn.execute(
            "SELECT COUNT(*) FROM bank WHERE purpose='primary'").fetchone()[0]
        return int(boxes), int(matured)

    def bank_verdict(self, lineage: str, purpose: str = "primary"):
        row = self.conn.execute(
            "SELECT box_id, verdict, kind FROM bank WHERE lineage=? AND purpose=?", (lineage, purpose)
        ).fetchone()
        return None if row is None else (row[0], row[1], row[2])
