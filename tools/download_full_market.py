"""Download the whole A-share universe, largest float cap first, resumably.

The CAPM test needs two things this repository does not have: a market portfolio
that is not the 300 largest listings, and a cross-section wide enough to estimate
betas away from those 300.  Both require the full ~5,578-name universe, so the
download is ordered by current free-float capitalisation and executed in fixed
chunks: **any prefix of the order is already a usable cap-weighted market proxy**,
which makes the run degradable and lets the coverage itself become a robustness
check instead of an assumption.

Each chunk becomes its own immutable history snapshot.  A progress file records the
snapshot id per chunk, so an interrupted run resumes at the first missing chunk and
never re-fetches a completed one.  ``source_bundle.compose_bundle`` joins the chunk
snapshots into one bundle at the end.

The range stops at 2024-12-31 on purpose: the reserved and protected segments are
not downloaded again.  Only ``hfq`` and ``unadjusted`` are requested — ``hfq`` for
returns, ``unadjusted`` because market capitalisation needs the traded price times
``outstanding_share``.
"""

import argparse
import json
import time
from pathlib import Path

from src.data.online import download_history

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = PROJECT_ROOT / "data" / "online" / "universe" / "a_share_spot.json"
PROGRESS = PROJECT_ROOT / "data" / "online" / "universe" / "download_progress.json"
START = "20170101"
END = "20241231"
BASES = ("hfq", "unadjusted")
CHUNK = 150
SOURCE = "sina"


def load_universe(path=UNIVERSE):
    """Codes ordered by free-float capitalisation, largest first."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    codes = [str(row["code"]) for row in rows if str(row.get("code", "")).isdigit()]
    if not codes:
        raise ValueError("Universe file contains no usable code")
    return codes


def load_progress(path=PROGRESS):
    path = Path(path)
    if not path.exists():
        return dict(start=START, end=END, bases=list(BASES), chunk=CHUNK, source=SOURCE, chunks={})
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(progress, path=PROGRESS):
    Path(path).write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def plan(codes, progress, chunk_size=CHUNK):
    """Chunks that still need a snapshot, as ``(index, codes)`` pairs.

    Only a chunk whose every code-basis pair arrived counts as done.  A chunk that
    finished with failures is re-planned: recording it as complete would silently lock
    the missing names out of the universe for good, which is exactly what happened to
    one chunk that hit a rate limit.
    """
    done = {int(index) for index, row in progress.get("chunks", {}).items()
            if row.get("status") == "ready"}
    pending = []
    for index, offset in enumerate(range(0, len(codes), chunk_size)):
        if index in done:
            continue
        pending.append((index, codes[offset:offset + chunk_size]))
    return pending


PERMANENT_ERRORS = ("History is empty", "404 Client Error", "Invalid OHLC")


def classify_failures(snapshot_directory):
    """Split a snapshot's failures into permanent gaps and transient ones.

    "History is empty" means the provider has no bars for that code in the range (a
    listing that post-dates the window), "404" means it does not serve the code, and
    "Invalid OHLC" means the bars failed validation.  None of those improve on a
    retry.  A circuit-breaker or time-limit message does, and that is the whole reason
    this distinction exists: one chunk lost 169 pairs to a rate limit while fifteen
    others simply carry codes that never had data.
    """
    path = Path(snapshot_directory) / "manifest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    permanent = transient = ready = 0
    for entry in payload.get("entries", []):
        if entry.get("status") == "ready":
            ready += 1
            continue
        text = str(entry.get("error"))
        if any(marker in text for marker in PERMANENT_ERRORS):
            permanent += 1
        else:
            transient += 1
    return dict(ready=ready, permanent=permanent, transient=transient)


def repair(progress, codes, chunk_size=CHUNK, history_root=None):
    """Settle every recorded chunk into ``ready`` or ``partial`` and return the change count.

    A chunk whose failures are all permanent (the provider simply has no bars for those
    codes) is finished work and is closed, so it is never fetched again.  A chunk with a
    transient failure such as a circuit breaker is reopened so the resume logic retries
    it.  Without the closing direction, one accidental reopen would re-download fifteen
    chunks for nothing.
    """
    changed = 0
    for index, offset in enumerate(range(0, len(codes), chunk_size)):
        key = str(index)
        row = progress.get("chunks", {}).get(key)
        if not row:
            continue
        expected = len(codes[offset:offset + chunk_size]) * len(BASES)
        before = row.get("status")
        detail = None
        if history_root and row.get("snapshot"):
            detail = classify_failures(Path(history_root) / row["snapshot"])
        if detail is not None:
            # Trust the snapshot over the stored counters: a failed retry can leave the
            # record blank while the manifest still holds every entry.
            row["ready"] = detail["ready"]
            row["failed"] = detail["permanent"] + detail["transient"]
        if row.get("ready", 0) >= expected:
            row["status"] = "ready"
        else:
            row["failure_detail"] = detail
            # An unclassifiable chunk stays open: guessing "permanent" would lose data.
            row["status"] = "partial" if detail is None or detail["transient"] > 0 else "ready"
        if row.get("status") != before:
            changed += 1
    return changed


def run(*, limit=None, chunk_size=CHUNK, universe=UNIVERSE, progress_path=PROGRESS,
        start=START, end=END, allow_network=False, dry_run=False):
    """Download every pending chunk, recording each snapshot before moving on."""
    if not allow_network and not dry_run:
        raise ValueError("Explicit network consent required")
    codes = load_universe(universe)
    progress = load_progress(progress_path)
    if progress.get("start") != start or progress.get("end") != end:
        raise ValueError("Progress file was written for a different date range")
    pending = plan(codes, progress, chunk_size)
    print("universe %d codes, %d chunks, %d pending" % (
        len(codes), (len(codes) + chunk_size - 1) // chunk_size, len(pending)), flush=True)
    if dry_run:
        for index, group in pending[:5]:
            print("  chunk %-3d %d codes  e.g. %s" % (index, len(group), group[:4]))
        return dict(pending=len(pending), dry_run=True)

    done_now = 0
    for index, group in pending:
        if limit is not None and done_now >= limit:
            break
        started = time.time()
        existing = progress.get("chunks", {}).get(str(index)) or {}
        resume = existing.get("snapshot") if existing.get("status") == "partial" else None
        try:
            # A list, not the BASES tuple: the stored request round-trips through JSON,
            # where a tuple becomes a list, and the resume check compares for equality.
            result = download_history(group, start, end, list(BASES), allow_network=True,
                                      root=PROJECT_ROOT, source=SOURCE, resume_id=resume)
            ready = sum(1 for row in result["entries"] if row["status"] == "ready")
            expected = len(group) * len(BASES)
            entry = dict(status="ready" if ready == expected else "partial",
                         snapshot=result.get("id") or result.get("snapshot_id"),
                         codes=len(group), ready=ready, failed=expected - ready,
                         seconds=round(time.time() - started, 1),
                         first=group[0], last=group[-1], resumed=bool(resume))
        except Exception as error:  # noqa: BLE001 - the chunk record is the durable evidence
            entry = dict(status="failed", error=str(error)[:400], codes=len(group),
                         seconds=round(time.time() - started, 1), first=group[0], last=group[-1])
        progress.setdefault("chunks", {})[str(index)] = entry
        save_progress(progress, progress_path)
        done_now += 1
        print("chunk %-3d codes=%-4d ready=%-4s failed=%-3s %5.1fs  %s -> %s" % (
            index, entry["codes"], entry.get("ready", "-"), entry.get("failed", "-"),
            entry["seconds"], entry.get("first"), entry.get("snapshot", entry.get("error"))), flush=True)
    total_ready = sum(row.get("ready", 0) for row in progress["chunks"].values())
    print("done: %d chunks recorded, %d code-basis pairs ready" % (
        sum(1 for row in progress["chunks"].values() if row.get("status") == "ready"), total_ready),
        flush=True)
    return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download the full A-share history in cap order.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many chunks.")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repair", action="store_true",
                        help="Mark every chunk with missing pairs as pending, then continue.")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.repair:
        codes = load_universe()
        progress = load_progress()
        changed = repair(progress, codes, args.chunk,
                         history_root=PROJECT_ROOT / "data" / "online" / "history")
        if args.dry_run:
            print("would repair %d chunk(s)" % changed, flush=True)
        else:
            save_progress(progress)
            print("repaired %d chunk(s)" % changed, flush=True)
    run(limit=args.limit, chunk_size=args.chunk, dry_run=args.dry_run,
        allow_network=args.allow_network or args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
