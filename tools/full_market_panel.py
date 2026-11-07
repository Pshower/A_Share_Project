"""Consolidate the chunked full-market downloads into one wide panel.

The universe download creates one immutable history snapshot per chunk, so the
data arrives as dozens of directories with two price bases each.  This tool joins
them into a single date-by-code panel with the three fields the CAPM work needs:

``hfq_close``        total-return price, used for returns;
``unadj_close``      traded price, used with ``shares`` for market capitalisation;
``shares``           ``outstanding_share``, the free-float share count.

A code can appear in more than one snapshot (an early probe, a re-run chunk); the
copy with the most valid closes wins, and the winning snapshot is recorded in the
manifest so the panel stays traceable.  Snapshots whose request falls outside the
declared range are ignored rather than silently mixed.

The output is a data product, not a research result: no returns are computed here
and no window is consumed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import file_hash, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = PROJECT_ROOT / "data" / "online" / "history"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "clean" / "full_market_v1"
START = "2017-01-03"
END = "2024-12-31"
FIELDS = ("hfq_close", "unadj_close", "shares")


def snapshot_requests(root=HISTORY_ROOT):
    """Every history snapshot with its request, newest first."""
    found = []
    for manifest in Path(root).glob("h_*/manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        request = payload.get("request") or {}
        found.append(dict(id=payload.get("id", manifest.parent.name), directory=manifest.parent,
                          codes=list(request.get("codes") or []),
                          start=str(request.get("start_date")), end=str(request.get("end_date")),
                          bases=list(request.get("bases") or []),
                          created=str(payload.get("created_at")),
                          entries=payload.get("entries") or []))
    return sorted(found, key=lambda row: row["created"], reverse=True)


def _as_date(value):
    """Parse the two date spellings the downloader accepts."""
    text = str(value)
    return pd.Timestamp(text if "-" in text else f"{text[:4]}-{text[4:6]}-{text[6:]}")


def usable(snapshot, start, end, universe=None):
    """A snapshot participates when its request covers the panel range."""
    if not snapshot["codes"]:
        return False
    try:
        covers = _as_date(snapshot["start"]) <= pd.Timestamp(start) and _as_date(snapshot["end"]) >= pd.Timestamp(end)
    except (ValueError, TypeError):
        return False
    if not covers:
        return False
    if not {"hfq", "unadjusted"} <= set(snapshot["bases"]):
        return False
    return universe is None or set(snapshot["codes"]) <= set(universe)


def _read_one(directory, basis, code, start, end, extra=()):
    """One code from one basis, restricted to the panel range."""
    path = Path(directory) / basis / f"{code}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["date", "close", *extra])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
    if frame.empty:
        return None
    return frame.set_index("date").sort_index()


def float_shares(price):
    """Recover the free-float share count from volume and turnover.

    The saved history files do not carry ``outstanding_share`` even though the
    provider returns it, but they do carry ``volume`` (shares) and ``turnover``
    (percent of free float).  ``shares = volume * 100 / turnover`` reproduces the
    provider's own value: for 600519 on 2017-01-03 it gives 1.2562e9, matching the
    ``outstanding_share`` of 1.256198e9, and it is constant to a coefficient of
    variation around 1e-16.  Suspended days report zero turnover, which is not a
    share count; those are dropped and filled inside the code only.
    """
    if "volume" not in price or "turnover" not in price:
        raise ValueError("Unadjusted rows must carry volume and turnover")
    derived = price["volume"].astype(float) * 100.0 / price["turnover"].astype(float).replace(0.0, np.nan)
    return derived.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def capitalisation_covered(columns, universe_file):
    """Share of total free-float capitalisation the panel actually contains.

    Coverage is the number that decides whether a cap-weighted market portfolio is
    already a market portfolio or still a large-cap proxy, so it is recorded with
    the panel rather than recomputed later.
    """
    path = Path(universe_file) if universe_file else None
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    total = sum(float(row.get("nmc") or 0.0) for row in rows)
    if total <= 0:
        return None
    present = set(columns)
    covered = sum(float(row.get("nmc") or 0.0) for row in rows if str(row.get("code")) in present)
    return dict(universe_codes=len(rows), panel_codes=len(present), share_of_float_cap=covered / total)


def build(codes, *, start=START, end=END, root=HISTORY_ROOT, output=DEFAULT_OUTPUT,
          universe_file=None, log=print):
    """Join every usable snapshot into three wide frames plus a manifest."""
    snapshots = [row for row in snapshot_requests(root) if usable(row, start, end, codes)]
    if not snapshots:
        raise ValueError("No history snapshot matches the requested codes and range")
    wanted = set(codes)
    chosen = {}
    for snapshot in snapshots:
        for code in snapshot["codes"]:
            if code in wanted and code not in chosen:
                chosen[code] = snapshot
    log("snapshots usable: %d, codes covered: %d / %d" % (len(snapshots), len(chosen), len(wanted)))

    hfq, unadj, shares, provenance, missing = {}, {}, {}, {}, []
    for code in codes:
        snapshot = chosen.get(code)
        if snapshot is None:
            missing.append(code)
            continue
        total = _read_one(snapshot["directory"], "hfq", code, start, end)
        price = _read_one(snapshot["directory"], "unadjusted", code, start, end,
                          extra=("volume", "turnover"))
        if total is None or price is None or price.empty:
            missing.append(code)
            continue
        hfq[code] = total["close"].reindex(price.index)
        unadj[code] = price["close"]
        shares[code] = float_shares(price)
        provenance[code] = snapshot["id"]

    if not hfq:
        raise ValueError("No code could be read from the selected snapshots")
    panel = pd.DataFrame({code: frame for code, frame in hfq.items()}).sort_index()
    traded = pd.DataFrame({code: frame for code, frame in unadj.items()}).sort_index()
    counts = pd.DataFrame({code: frame for code, frame in shares.items()}).sort_index()
    panel = panel.reindex(index=panel.index.union(traded.index).union(counts.index))
    traded = traded.reindex(panel.index)
    counts = counts.reindex(panel.index)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("hfq_close", panel), ("unadj_close", traded), ("shares", counts)):
        frame.to_parquet(output / f"{name}.parquet")
    manifest = dict(schema_version=1, version=output.name, start=start, end=end,
                    codes=list(panel.columns), codes_missing=missing,
                    fields=list(FIELDS), shape=list(panel.shape),
                    provenance=provenance,
                    capitalisation=capitalisation_covered(panel.columns, universe_file),
                    snapshots=[dict(id=row["id"], codes=len(row["codes"]), created=row["created"])
                               for row in snapshots],
                    coverage=dict(
                        hfq_nonnull=int(np.isfinite(panel.to_numpy()).sum()),
                        unadj_nonnull=int(np.isfinite(traded.to_numpy()).sum()),
                        shares_nonnull=int(np.isfinite(counts.to_numpy()).sum()),
                        rows=int(len(panel)), columns=int(panel.shape[1])),
                    artifacts={name: file_hash(output / f"{name}.parquet")
                               for name in FIELDS})
    write_json(output / "manifest.json", manifest)
    log("panel %s: %d dates x %d codes, %d missing codes" % (
        output.name, panel.shape[0], panel.shape[1], len(missing)))
    return manifest


def load(output=DEFAULT_OUTPUT):
    """Read a built panel and its manifest."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    frames = {name: pd.read_parquet(output / f"{name}.parquet") for name in FIELDS}
    return manifest, frames["hfq_close"], frames["unadj_close"], frames["shares"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the wide full-market panel.")
    parser.add_argument("--codes-from", default=str(PROJECT_ROOT / "data" / "online" / "universe" / "a_share_spot.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--lenient", action="store_true", help="Use every code present instead of the universe list.")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.codes_from).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    universe = [str(row["code"]) for row in rows if str(row.get("code", "")).isdigit()]
    if args.lenient:
        universe = [code for row in snapshot_requests() for code in row["codes"]]
        universe = list(dict.fromkeys(universe))
    manifest = build(universe, output=args.output, universe_file=args.codes_from)
    print(json.dumps(manifest["coverage"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
