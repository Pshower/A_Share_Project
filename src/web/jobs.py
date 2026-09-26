"""One owned subprocess slot, cooperative stop and persistent recovery."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from src.data.preprocess import ROOT, write_json
from .store import lane_for

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def read_events(directory, after=0):
    path = Path(directory) / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record["sequence"] > after:
                events.append(record)
        except (ValueError, KeyError):
            continue
    return events


class JobManager:
    def __init__(self, store, root=ROOT, lane="research"):
        self.store, self.root = store, Path(root)
        self.stop_event = threading.Event()
        self.process = None
        self.active_id = None
        self.log_stream = None
        self.thread = None
        self.lane = lane

    def directory(self, identifier):
        self.store.get(identifier)
        return self.store.directory / "jobs" / identifier

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def request_stop(self, identifier):
        job = self.store.get(identifier)
        if job["status"] in TERMINAL:
            return job
        directory = self.directory(identifier)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stop").touch()
        self.store.update(identifier, "cancelled" if job["status"] == "queued" else "stop_requested")
        return self.store.get(identifier)

    def reconcile(self, job):
        directory = self.directory(job["id"])
        result = directory / "result.json"
        if result.exists():
            try:
                record = json.loads(result.read_text(encoding="utf-8"))
                if record["status"] not in TERMINAL:
                    raise ValueError("Invalid completion status")
                self.store.update(job["id"], record["status"], record.get("result"), record.get("error"))
            except (ValueError, OSError, KeyError, TypeError):
                self.store.update(job["id"], "failed", error="Invalid worker completion record")
            return False
        heartbeat = directory / "heartbeat"
        alive = heartbeat.exists() and time.time() - heartbeat.stat().st_mtime < 60
        if not heartbeat.exists():
            alive = (datetime.now(timezone.utc) - datetime.fromisoformat(job["updated"])).total_seconds() < 60
        if self.active_id == job["id"] and self.process is not None:
            alive = self.process.poll() is None
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(job["updated"])).total_seconds()
            limit = 660 if self.lane == "history" else 150
            timed_out = self.lane != "research" and elapsed > limit
            stopped_late = self.lane != "research" and job["status"] == "stop_requested" and elapsed > 20
            if alive and (timed_out or stopped_late):
                self.process.terminate()
                self.process.wait(timeout=5)
                self.store.update(job["id"], "cancelled" if stopped_late else "failed", error="Owned network worker reached cancellation/deadline limit")
                return False
        if alive:
            return True
        self.store.update(job["id"], "interrupted", error="Worker exited without a complete result; no automatic restart")
        return False

    def tick(self):
        if self.lane == "quotes":
            self.store.tick_monitors()
        active = [j for j in self.store.jobs() if j["status"] in {"running", "stop_requested"} and lane_for(j["kind"]) == self.lane]
        if any(self.reconcile(job) for job in active):
            return
        if self.process is not None and self.process.poll() is None:
            return
        if self.log_stream:
            self.log_stream.close()
            self.log_stream = None
        job = self.store.claim_next(self.lane)
        if job is None:
            return
        self.active_id = job["id"]
        directory = self.directory(job["id"])
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "stop").exists():
            self.store.update(job["id"], "cancelled")
            return
        write_json(directory / "request.json", job)
        self.log_stream = (directory / "worker.log").open("w", encoding="utf-8")
        args = [sys.executable, "-B", "-u", "-m", "src.web.worker", "--job", job["id"]]
        self.process = subprocess.Popen(args, cwd=self.root, stdout=self.log_stream, stderr=subprocess.STDOUT,
                                        shell=False, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception as error:
                if self.active_id:
                    self.store.update(self.active_id, "failed", error=str(error))
            self.stop_event.wait(0.4)

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.process is not None and self.process.poll() is None:
            self.request_stop(self.active_id)
            try:
                self.process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
                self.store.update(self.active_id, "interrupted", error="Worker did not stop before local service shutdown")
        if self.log_stream:
            self.log_stream.close()
