"""Small durable job and exposure ledger; no model or price data in SQLite."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = self.directory / "state.sqlite3"
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                    created TEXT NOT NULL, updated TEXT NOT NULL, payload TEXT NOT NULL,
                    result TEXT, error TEXT, idempotency TEXT UNIQUE);
                CREATE TABLE IF NOT EXISTS labels (id TEXT PRIMARY KEY, label TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS exposures (id INTEGER PRIMARY KEY, created TEXT,
                    kind TEXT, dataset TEXT, start TEXT, end TEXT, source TEXT,
                    UNIQUE(kind, dataset, start, end, source));
            """)

    def connect(self):
        conn = sqlite3.connect(self.db, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def decode(row):
        value = dict(row)
        for key in ["payload", "result"]:
            value[key] = json.loads(value[key]) if value[key] else None
        return value

    def enqueue(self, kind, payload, key=None):
        identifier = "j_" + uuid.uuid4().hex[:16]
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if key:
                prior = conn.execute("SELECT * FROM jobs WHERE idempotency=?", (key,)).fetchone()
                if prior:
                    decoded = self.decode(prior)
                    if decoded["kind"] != kind or decoded["payload"] != payload:
                        raise ValueError("Idempotency key was already used for different input")
                    return decoded
            timestamp = now()
            conn.execute("INSERT INTO jobs(id,kind,status,created,updated,payload,idempotency) VALUES(?,?,?,?,?,?,?)",
                         (identifier, kind, "queued", timestamp, timestamp, json.dumps(payload), key))
        return self.get(identifier)

    def get(self, identifier):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Unknown job")
        return self.decode(row)

    def claim_next(self):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM jobs WHERE status IN ('running','stop_requested') LIMIT 1").fetchone():
                return None
            row = conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            conn.execute("UPDATE jobs SET status='running',updated=? WHERE id=?", (now(), row["id"]))
            identifier = row["id"]
        return self.get(identifier)

    def jobs(self):
        with self.connect() as conn:
            return [self.decode(row) for row in conn.execute("SELECT * FROM jobs ORDER BY created DESC")]

    def update(self, identifier, status, result=None, error=None):
        with self.connect() as conn:
            conn.execute("UPDATE jobs SET status=?,updated=?,result=?,error=? WHERE id=?",
                         (status, now(), json.dumps(result) if result is not None else None, error, identifier))

    def labels(self):
        with self.connect() as conn:
            return dict(conn.execute("SELECT id,label FROM labels"))

    def label(self, identifier, label):
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO labels VALUES (?,?)", (identifier, label))

    def expose(self, kind, dataset, start, end, source):
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO exposures(created,kind,dataset,start,end,source) VALUES(?,?,?,?,?,?)",
                         (now(), kind, dataset, start, end, source))

    def exposures(self):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM exposures ORDER BY id DESC")]
