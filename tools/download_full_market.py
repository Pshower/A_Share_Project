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
    """Chunks that still need a snapshot, as ``(index, codes)`` pairs."""
    done = {int(index) for index, row in progress.get("chunks", {}).items()
            if row.get("status") == "ready"}
    pending = []
    for index, offset in enumerate(range(0, len(codes), chunk_size)):
        if index in done:
            continue
        pending.append((index, codes[offset:offset + chunk_size]))
    return pending


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
        try:
            result = download_history(group, start, end, BASES, allow_network=True,
                                      root=PROJECT_ROOT, source=SOURCE)
            ready = sum(1 for row in result["entries"] if row["status"] == "ready")
            entry = dict(status="ready", snapshot=result.get("id") or result.get("snapshot_id"),
                         codes=len(group), ready=ready, failed=len(group) * len(BASES) - ready,
                         seconds=round(time.time() - started, 1),
                         first=group[0], last=group[-1])
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
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    run(limit=args.limit, chunk_size=args.chunk, dry_run=args.dry_run,
        allow_network=args.allow_network or args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
