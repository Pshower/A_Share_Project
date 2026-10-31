"""Small durable job and exposure ledger; no model or price data in SQLite."""

from datetime import datetime, timezone
import json
from pathlib import Path
from src.runtime.sqlite_connection import connection
import uuid
import time

NETWORK_KINDS = {"online_history", "online_quotes", "online_supplement"}
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def lane_for(kind):
    return "history" if kind in {"online_history", "online_supplement"} else "quotes" if kind == "online_quotes" else "research"


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
                CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created TEXT, recipe TEXT);
                CREATE TABLE IF NOT EXISTS experiment_jobs (experiment_id TEXT, stage TEXT, job_id TEXT UNIQUE, fingerprint TEXT);
                CREATE TABLE IF NOT EXISTS exposures (id INTEGER PRIMARY KEY, created TEXT,
                    kind TEXT, dataset TEXT, start TEXT, end TEXT, source TEXT,
                    UNIQUE(kind, dataset, start, end, source));
                CREATE TABLE IF NOT EXISTS monitors (id TEXT PRIMARY KEY, created TEXT, payload TEXT,
                    status TEXT, expires REAL, lease REAL, next_due REAL, last_job TEXT);
            """)

    def connect(self):
        return connection(self.db, rows=True)

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

    def claim_next(self, lane="research"):
        if lane not in {"research", "history", "quotes"}:
            raise ValueError("Unknown execution lane")
        clause = "kind NOT IN ('online_history','online_quotes','online_supplement')" if lane == "research" else "kind IN ('online_history','online_supplement')" if lane == "history" else "kind='online_quotes'"
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(f"SELECT 1 FROM jobs WHERE status IN ('running','stop_requested') AND {clause} LIMIT 1").fetchone():
                return None
            row = conn.execute(f"SELECT * FROM jobs WHERE status='queued' AND {clause} ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            conn.execute("UPDATE jobs SET status='running',updated=? WHERE id=?", (now(), row["id"]))
            identifier = row["id"]
        return self.get(identifier)

    def jobs(self):
        with self.connect() as conn:
            return [self.decode(row) for row in conn.execute("SELECT * FROM jobs ORDER BY created DESC")]

    def update(self, identifier, status, result=None, error=None):
        if status not in TERMINAL | {"queued", "running", "stop_requested"}:
            raise ValueError("Unknown job status")
        with self.connect() as conn:
            conn.execute("UPDATE jobs SET status=?,updated=?,result=?,error=? WHERE id=? AND status NOT IN ('succeeded','failed','cancelled','interrupted')",
                         (status, now(), json.dumps(result) if result is not None else None, error, identifier))

    def request_stop(self, identifier):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("Unknown job")
            if row["status"] not in TERMINAL | {"stop_requested"}:
                directory = self.directory / "jobs" / row["id"]
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "stop").touch()
                conn.execute("UPDATE jobs SET status=?,updated=? WHERE id=?",
                             ("cancelled" if row["status"] == "queued" else "stop_requested", now(), identifier))
        return self.get(identifier)

    def labels(self):
        with self.connect() as conn:
            return dict(conn.execute("SELECT id,label FROM labels"))

    def create_experiment(self, recipe):
        if recipe.get("parent_id"):
            self.experiment(recipe["parent_id"])
        identifier = "e_" + uuid.uuid4().hex[:16]
        with self.connect() as conn:
            conn.execute("INSERT INTO experiments VALUES(?,?,?)", (identifier, now(), json.dumps(recipe)))
        return self.experiment(identifier)

    def experiment(self, identifier):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Unknown experiment")
        return dict(id=row["id"], created=row["created"], recipe=json.loads(row["recipe"]))

    def experiments(self):
        with self.connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM experiments ORDER BY created DESC")]
        return [self.experiment(identifier) for identifier in ids]

    def experiment_jobs(self, identifier):
        self.experiment(identifier)
        with self.connect() as conn:
            return [self.decode(r) for r in conn.execute("SELECT jobs.*, stage, fingerprint FROM jobs JOIN experiment_jobs ON job_id=jobs.id WHERE experiment_id=? ORDER BY jobs.created DESC", (identifier,))]

    def enqueue_experiment(self, identifier, stage, kind, payload, fingerprint):
        self.experiment(identifier)
        payload = dict(payload, experiment_id=identifier, experiment_fingerprint=fingerprint)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute("SELECT jobs.*, fingerprint FROM jobs JOIN experiment_jobs ON job_id=jobs.id WHERE experiment_id=? AND stage=? ORDER BY jobs.created DESC LIMIT 1", (identifier, stage)).fetchone()
            reusable = prior and prior["status"] == "succeeded" and prior["fingerprint"] == fingerprint
            if reusable and stage == "download":
                reusable = json.loads(prior["result"] or "{}").get("capture_status") == "ready"
            if prior and (prior["status"] in {"queued", "running", "stop_requested"} or reusable):
                return self.decode(prior)
            job_id, stamp = "j_" + uuid.uuid4().hex[:16], now()
            conn.execute("INSERT INTO jobs(id,kind,status,created,updated,payload) VALUES(?,?,?,?,?,?)",
                         (job_id, kind, "queued", stamp, stamp, json.dumps(payload)))
            conn.execute("INSERT INTO experiment_jobs VALUES(?,?,?,?)", (identifier, stage, job_id, fingerprint))
        return self.get(job_id)

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

    def create_monitor(self, payload):
        identifier = "m_" + uuid.uuid4().hex[:16]
        stamp = time.time()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE monitors SET status='expired' WHERE status='active' AND (lease<? OR expires<?)", (stamp, stamp))
            if conn.execute("SELECT 1 FROM monitors WHERE status='active'").fetchone():
                raise ValueError("Stop the existing monitor before starting another")
            conn.execute("INSERT INTO monitors VALUES(?,?,?,?,?,?,?,?)", (identifier, now(), json.dumps(payload), "active", stamp + payload["duration_seconds"], stamp + 45, stamp, None))
        return self.monitor(identifier)

    def monitor(self, identifier):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM monitors WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Unknown monitor")
        record = dict(row)
        record["payload"] = json.loads(record["payload"])
        return record

    def renew_monitor(self, identifier):
        record = self.monitor(identifier)
        stamp = time.time()
        if record["status"] != "active" or min(record["lease"], record["expires"]) < stamp:
            raise ValueError("Monitor lease expired; explicitly start a new session")
        with self.connect() as conn:
            conn.execute("UPDATE monitors SET lease=? WHERE id=? AND status='active'", (min(stamp + 45, record["expires"]), identifier))
        return self.monitor(identifier)

    def stop_monitor(self, identifier):
        record = self.monitor(identifier)
        with self.connect() as conn:
            conn.execute("UPDATE monitors SET status='stopped' WHERE id=? AND status='active'", (identifier,))
        return record

    def tick_monitors(self):
        stamp = time.time()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expired = conn.execute("SELECT last_job FROM monitors WHERE status='active' AND (lease<? OR expires<?)", (stamp, stamp)).fetchall()
            conn.execute("UPDATE monitors SET status='expired' WHERE status='active' AND (lease<? OR expires<?)", (stamp, stamp))
            for old in expired:
                if old["last_job"]:
                    directory = self.directory / "jobs" / old["last_job"]
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "stop").touch()
                    conn.execute("UPDATE jobs SET status=CASE WHEN status='queued' THEN 'cancelled' ELSE 'stop_requested' END WHERE id=? AND status IN ('queued','running')", (old["last_job"],))
            for row in conn.execute("SELECT * FROM monitors WHERE status='active' AND next_due<=?", (stamp,)).fetchall():
                if row["last_job"] and conn.execute("SELECT 1 FROM jobs WHERE id=? AND status IN ('queued','running','stop_requested')", (row["last_job"],)).fetchone():
                    continue
                payload = json.loads(row["payload"])
                request = {key: payload[key] for key in ["codes", "allow_network", "plan_id"]}
                request["source"] = payload.get("source", "sina")
                request["monitor_id"] = row["id"]
                identifier = "j_" + uuid.uuid4().hex[:16]
                conn.execute("INSERT INTO jobs(id,kind,status,created,updated,payload,idempotency) VALUES(?,?,?,?,?,?,?)", (identifier, "online_quotes", "queued", now(), now(), json.dumps(request), "monitor:" + row["id"] + ":" + identifier))
                conn.execute("UPDATE monitors SET last_job=?, next_due=? WHERE id=?", (identifier, stamp + payload["interval_seconds"], row["id"]))
