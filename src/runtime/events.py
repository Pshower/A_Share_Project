"""Append-only worker events and cooperative cancellation."""

from datetime import datetime, timezone
import json
from pathlib import Path

_directory = None
_sequence = 0


class Cancelled(RuntimeError):
    pass


def configure(directory):
    global _directory, _sequence
    _directory, _sequence = Path(directory) if directory is not None else None, 0


def check_cancel():
    if _directory is not None and (_directory / "stop").exists():
        raise Cancelled("Stop requested by the user")


def emit(kind, **values):
    global _sequence
    if _directory is None:
        return
    _sequence += 1
    event = dict(sequence=_sequence, timestamp=datetime.now(timezone.utc).isoformat(), type=kind, **values)
    with (_directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def phase(name, **values):
    check_cancel()
    emit("phase", phase=name, **values)
