"""The shared SQLite connection helper. One place sets the concurrency hardening (WAL +
busy_timeout + autocommit) so the ledger and the box-lease store can never drift apart, which
matters for a heavily parallel campaign."""
from common.sqlite import connect


def test_connect_enables_wal_and_autocommit(tmp_path):
    conn = connect(tmp_path / "t.db")
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        assert conn.isolation_level is None  # autocommit; callers manage txns explicitly
    finally:
        conn.close()


def test_connect_roundtrips(tmp_path):
    conn = connect(tmp_path / "t.db")
    try:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        assert conn.execute("SELECT x FROM t").fetchone()[0] == 1
    finally:
        conn.close()
