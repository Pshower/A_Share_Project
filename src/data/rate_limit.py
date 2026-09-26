"""Cross-process adaptive pacing without pre-booked retry bursts."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import sqlite3
import time

import requests

from src.runtime import events


def retry_delay(error):
    response = getattr(error, "response", None)
    raw = response.headers.get("Retry-After") if response is not None else None
    if raw is None:
        return 0.0
    try:
        delay = float(raw)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(raw)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            delay = (date - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


def transient(error):
    response = getattr(error, "response", None)
    if response is not None:
        return response.status_code in {408, 429, 500, 502, 503, 504}
    return isinstance(error, (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError))


def wait(seconds, deadline=None):
    remaining = seconds
    while remaining > 0:
        events.check_cancel()
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Download retry/time budget exhausted")
        duration = min(remaining, 0.2)
        if deadline is not None:
            duration = min(duration, max(0, deadline - time.monotonic()))
        time.sleep(duration)
        remaining -= duration


class AdaptivePacer:
    def __init__(self, root):
        self.path = root / ".workbench/network.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS clock (id INTEGER PRIMARY KEY, next REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS throttle (id INTEGER PRIMARY KEY, interval REAL, cooldown REAL, successes INTEGER)")
            conn.execute("INSERT OR IGNORE INTO throttle VALUES(1,1,0,0)")

    def acquire(self, deadline=None):
        announced = False
        while True:
            events.check_cancel()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Download retry/time budget exhausted")
            with sqlite3.connect(self.path, timeout=10) as conn:
                conn.execute("BEGIN IMMEDIATE")
                interval, cooldown, _ = conn.execute("SELECT interval,cooldown,successes FROM throttle WHERE id=1").fetchone()
                row = conn.execute("SELECT next FROM clock WHERE id=1").fetchone()
                now = time.time()
                delay = max(cooldown, row[0] if row else 0) - now
                if delay <= 0:
                    conn.execute("INSERT OR REPLACE INTO clock VALUES(1,?)", (now + interval,))
                    return
            if not announced and delay > 1:
                events.emit("download_throttle", wait_seconds=delay, interval_seconds=interval)
                announced = True
            # Re-read shared state so another process can extend the cooldown.
            wait(min(delay, 0.2), deadline)

    def failure(self, delay):
        with sqlite3.connect(self.path, timeout=10) as conn:
            conn.execute("BEGIN IMMEDIATE")
            interval, cooldown, _ = conn.execute("SELECT interval,cooldown,successes FROM throttle WHERE id=1").fetchone()
            interval = min(30.0, interval * 2)
            conn.execute("UPDATE throttle SET interval=?,cooldown=?,successes=0 WHERE id=1",
                         (interval, max(cooldown, time.time() + max(delay, interval))))
        return interval

    def success(self):
        with sqlite3.connect(self.path, timeout=10) as conn:
            conn.execute("BEGIN IMMEDIATE")
            interval, cooldown, successes = conn.execute("SELECT interval,cooldown,successes FROM throttle WHERE id=1").fetchone()
            # An in-flight success must not undo another worker's newer failure.
            if time.time() < cooldown:
                return
            successes += 1
            if successes >= 5:
                interval, successes = max(1.0, interval * 0.8), 0
            conn.execute("UPDATE throttle SET interval=?,successes=? WHERE id=1", (interval, successes))
