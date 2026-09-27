"""SQLite transactions with deterministic connection ownership."""

from contextlib import contextmanager
import sqlite3


@contextmanager
def connection(path, *, rows=False):
    conn = sqlite3.connect(path, timeout=10)
    try:
        if rows:
            conn.row_factory = sqlite3.Row
        with conn:
            yield conn
    finally:
        conn.close()
