"""Extend the E1 pool to the present so the shadow run can decide on live data.

``research_v3`` stops at 2024-12-31 and must not be rewritten -- ``run_suite`` verifies
its artifact hashes.  The E1 construction is frozen, so the only thing that needs to
move is the data underneath it.  This tool moves that data: download the same pool's
bars to the requested end date, then rebuild the panel with the repository's own
``DataPreprocessor`` under a new name.

Two deliberate choices:

* the whole history is downloaded again rather than the missing tail appended, because
  the forward-adjusted series is rebased to the newest date and splicing two captures
  leaves a dividend gap in the middle of the feature window;
* the split dates are copied from ``research_v3`` so the twenty features are defined
  identically -- a different split would mean a different dataset, not an extension.

This tool never touches the shadow ledger or the recorder state.  Opening the forward
book is a protocol step, taken by hand, not a build step.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from src.data.online import download_history
from src.data.preprocess import DataPreprocessor, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_MANIFEST = PROJECT_ROOT / "data" / "clean" / "research_v3" / "manifest.json"
STOCK_LIST_DIR = PROJECT_ROOT / "data" / "stock_list"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
HISTORY_DIR = PROJECT_ROOT / "data" / "online" / "history"
START = "2017-01-01"
TRAIN_END, VAL_END = "2022-08-05", "2023-10-20"


def pool_codes(path=POOL_MANIFEST):
    """The frozen E1 pool, in the order ``research_v3`` recorded it."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    codes = [str(code) for code in manifest["stock_codes"]]
    if len(codes) != len(set(codes)):
        raise ValueError("Pool contains duplicate codes")
    return codes


def write_stock_list(codes, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(codes) + "\n", encoding="utf-8")
    return path


def capture(codes, end, log=print):
    """Download one clean capture of the pool and return its snapshot directory."""
    result = download_history(codes, START, end, ["hfq"], allow_network=True)
    identifier = result["id"]
    directory = HISTORY_DIR / identifier
    ready = [entry["code"] for entry in result["entries"] if entry["status"] == "ready"]
    missing = [code for code in codes if code not in set(ready)]
    log("captured %s: %d ready, %d missing" % (identifier, len(ready), len(missing)))
    if missing:
        log("missing: %s" % ",".join(missing))
    return directory, set(ready), missing


def build(snapshot, codes, version, end, log=print):
    """Rebuild the panel from one capture, restricted to the codes it returned."""
    clean = CLEAN_DIR / version
    if clean.exists():
        raise ValueError("Refusing to rewrite an existing dataset: %s" % clean)
    stock_list = write_stock_list(codes, STOCK_LIST_DIR / ("%s.txt" % version))
    processor = DataPreprocessor(
        raw_dir=Path(snapshot) / "hfq", clean_dir=clean, stock_list_path=stock_list,
        start_date=START, end_date=end, train_end=TRAIN_END, val_end=VAL_END,
        train_codes=None, volume_multiplier=1, raw_pattern="{code}.csv", version=version)
    processor.process_all_stocks_and_save_scaler()
    manifest_path = clean / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["forward"] = dict(
        purpose="forward extension of the frozen E1 pool for the S2 shadow run",
        pool_source=str(POOL_MANIFEST.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        snapshot=Path(snapshot).name, capture_end=end, rebuilt_full_history=True,
        construction="e1 reversal v1 (unchanged)")
    write_json(manifest_path, manifest)
    log("built %s: %d codes, %d sessions, ends %s" % (
        version, len(manifest["stock_codes"]), manifest["n_dates"], manifest["end_date"]))
    return manifest


def prune(keep=5, root=CLEAN_DIR, log=print):
    """Delete all but the newest ``keep`` forward panels.

    A daily panel is 488 MB, most of it per-code and split files the recorder never reads.
    Keeping every day would cost ~180 GB a year for data that is reproducible from the
    capture snapshot in ``data/online/history``, which is the immutable one.  The panels
    are caches; the captures are the record.
    """
    root = Path(root)
    directories = sorted((d for d in root.glob("forward_pool_*") if d.is_dir()), key=lambda d: d.name)
    doomed = directories[:-keep] if keep > 0 else []
    for directory in doomed:
        if directory.resolve().parent != root.resolve() or not directory.name.startswith("forward_pool_"):
            raise ValueError("Refusing to delete outside the forward panel root: %s" % directory)
        shutil.rmtree(directory)
    if doomed:
        log("pruned %d old panel(s): %s" % (len(doomed), ", ".join(d.name for d in doomed)))
    return [d.name for d in doomed]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild the E1 pool panel up to a given date.")
    parser.add_argument("--end", default=str(date.today()), help="Last date to capture (YYYY-MM-DD).")
    parser.add_argument("--version", default=None, help="Dataset name; defaults to forward_pool_<end>.")
    parser.add_argument("--snapshot", default=None, help="Reuse an existing capture instead of downloading.")
    parser.add_argument("--keep", type=int, default=5, help="How many dated panels to keep; 0 keeps all.")
    args = parser.parse_args(argv)
    version = args.version or "forward_pool_%s" % args.end.replace("-", "")
    codes = pool_codes()
    if args.snapshot:
        snapshot, ready = Path(args.snapshot), set()
        directory = HISTORY_DIR / snapshot.name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        ready = {e["code"] for e in manifest["entries"] if e["status"] == "ready"}
        missing = [code for code in codes if code not in ready]
    else:
        directory, ready, missing = capture(codes, args.end)
    kept = [code for code in codes if code in ready]
    manifest = build(directory, kept, version, args.end)
    pruned = prune(args.keep)
    print(json.dumps(dict(version=version, codes=len(kept), skipped=missing,
                          end=manifest["end_date"], dataset=str(CLEAN_DIR / version),
                          pruned=pruned),
                     ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
