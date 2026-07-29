"""The durable experiment ledger + its canary write-read probe. The ledger is where
provenance and per-run metrics land; the canary is what the health gate calls to prove
the store round-trips before the campaign trusts it. Values are JSON, never pickle."""
from engine.ledger import SQLiteLedger, canary_probe


def test_log_read_roundtrip(tmp_path):
    led = SQLiteLedger(tmp_path / "l.db")
    led.log("run1", "metric", {"acc": 0.9})
    assert led.read("run1", "metric") == {"acc": 0.9}


def test_read_missing_returns_none(tmp_path):
    led = SQLiteLedger(tmp_path / "l.db")
    assert led.read("run1", "nope") is None


def test_last_write_wins(tmp_path):
    led = SQLiteLedger(tmp_path / "l.db")
    led.log("r", "k", 1)
    led.log("r", "k", 2)
    assert led.read("r", "k") == 2


def test_durable_across_reopen(tmp_path):
    p = tmp_path / "l.db"
    SQLiteLedger(p).log("r", "k", "v")
    assert SQLiteLedger(p).read("r", "k") == "v"


def test_canary_passes_on_healthy_ledger(tmp_path):
    assert canary_probe(SQLiteLedger(tmp_path / "l.db")) is True


def test_canary_fails_when_writes_are_dropped():
    class Broken:
        def log(self, *a):
            pass

        def read(self, *a):
            return None

    assert canary_probe(Broken()) is False


def test_canary_fails_when_store_returns_garbage_for_unwritten_key():
    class Garbage:
        def log(self, *a):
            pass

        def read(self, *a):
            return "garbage"

    assert canary_probe(Garbage()) is False
