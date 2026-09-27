import sqlite3
import pytest

from src.web.store import Store
from src.data.rate_limit import AdaptivePacer


def test_store_closes_connections_and_rolls_back_failures(tmp_path):
    store = Store(tmp_path)
    with store.connect() as conn:
        conn.execute("INSERT INTO labels VALUES ('ok','committed')")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    with pytest.raises(RuntimeError):
        with store.connect() as conn:
            conn.execute("INSERT INTO labels VALUES ('bad','rollback')")
            raise RuntimeError("abort")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    assert store.labels() == {"ok": "committed"}


def test_pacer_closes_every_connection(tmp_path, monkeypatch):
    original = sqlite3.connect
    opened = []
    def track(*args, **kwargs):
        conn = original(*args, **kwargs)
        opened.append(conn)
        return conn
    monkeypatch.setattr(sqlite3, "connect", track)
    pacer = AdaptivePacer(tmp_path)
    pacer.acquire()
    pacer.success()
    pacer.failure(0)
    assert len(opened) == 4
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
