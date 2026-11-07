"""Fetch the code-basis pairs an earlier batch never reached.

`download_history` stops a batch after three consecutive failures and marks every
remaining pair "Batch circuit/time limit reached".  Those pairs were never attempted:
in the smallest-capitalisation chunks the three failures are recently listed names
that genuinely have no 2017-2024 history, so the breaker fires on permanent gaps and
the rest of the chunk is skipped.  Retrying the chunk as a whole reproduces the same
split, because the same three names come first.

Small batches avoid it.  Each call starts with the failure counter at zero, so a batch
of five codes tolerates two permanent gaps and still reaches the rest.  Pairs that keep
failing are the ones the provider really does not have.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from src.data.online import download_history

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY = PROJECT_ROOT / "data" / "online" / "history"
UNIVERSE = PROJECT_ROOT / "data" / "online" / "universe" / "a_share_spot.json"
START = "20170101"
END = "20241231"
BASES = ("hfq", "unadjusted")


def load_universe():
    payload = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    rows = payload.get("codes") or payload.get("rows") or []
    return [row["code"] for row in rows]


def ready_pairs():
    """Every (code, basis) that some snapshot already holds as ready."""
    found = set()
    for manifest in HISTORY.glob("h_*/manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for entry in payload.get("entries", []):
            if entry.get("status") == "ready":
                found.add((entry["code"], entry["basis"]))
    return found


def missing_codes(codes, ready):
    """Universe codes that are missing at least one basis."""
    return [code for code in codes if any((code, basis) not in ready for basis in BASES)]


def batches(items, size):
    for offset in range(0, len(items), size):
        yield items[offset:offset + size]


def run(batch_size=5, rounds=4, limit=None, log=print):
    universe = load_universe()
    for attempt in range(1, rounds + 1):
        ready = ready_pairs()
        pending = missing_codes(universe, ready)
        if limit is not None:
            pending = pending[:limit]
        if not pending:
            log("nothing missing")
            return dict(status="complete", attempt=attempt)
        before = len(pending)
        log("round %d: %d codes missing at least one basis" % (attempt, before))
        for group in batches(pending, batch_size):
            result = download_history(group, START, END, list(BASES), allow_network=True,
                                      root=PROJECT_ROOT, source="sina")
            got = sum(1 for row in result.get("entries", []) if row.get("status") == "ready")
            log("  batch %s..%s  ready %d/%d" % (group[0], group[-1], got, len(group) * len(BASES)))
        after = len(missing_codes(universe, ready_pairs()))
        log("round %d done: missing %d -> %d" % (attempt, before, after))
        if after >= before:
            log("no progress; remaining codes have no history")
            return dict(status="stalled", remaining=after, attempt=attempt)
    return dict(status="finished", remaining=len(missing_codes(universe, ready_pairs())))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch code-basis pairs earlier batches skipped.")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_network:
        raise SystemExit("Explicit network consent required: pass --allow-network")
    result = run(batch_size=args.batch, rounds=args.rounds, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
