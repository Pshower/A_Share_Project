"""Derive point-in-time availability dates from an immutable PIT snapshot.

The first PIT capture (``f_2f8ef1efe22c430a``) stored the vendor's ``最新公告日期`` as
``announcement_date``. That field is the latest revision date, not the first disclosure
date: for the 2024 Q1 table the median value lands in April 2025, and every matched
``(code, period)`` pair is later than the same period's earnings forecast. Nothing was
look-ahead, but a row only became usable about thirteen months late.

Snapshots are immutable, so this tool never rewrites one. It reads an existing snapshot
through :class:`src.data.snapshots.Snapshots` (which verifies the saved file hashes),
recomputes the availability rule used by the collector, and writes a new derived artifact
plus an audit of the lag distribution.

Financials use the statutory disclosure deadline, which can only delay a row and therefore
cannot leak future information. Forecasts and shareholder counts carry an observed
announcement date and keep it. No network access, no training, no market simulation.
"""

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.data.pit import STATUTORY_DEADLINES, statutory_deadline
from src.data.preprocess import ROOT, write_json
from src.data.snapshots import Snapshots

TABLES = ("financials", "forecasts", "shareholders")
BASIS = {"financials": "statutory_deadline", "forecasts": "vendor_announcement",
         "shareholders": "vendor_announcement"}


def availability(frame, kind):
    """Return ``available_from`` and its basis for one normalized PIT table."""
    if kind not in BASIS:
        raise ValueError("Unknown PIT table")
    if BASIS[kind] == "statutory_deadline":
        periods = pd.to_datetime(frame["period"].astype(str))
        deadlines = periods.map(lambda value: statutory_deadline(value.strftime("%Y%m%d")))
        return deadlines.dt.strftime("%Y-%m-%d"), pd.Series(BASIS[kind], index=frame.index)
    return frame["announcement_date"], pd.Series(BASIS[kind], index=frame.index)


def lag_profile(frame):
    """Describe how late both date columns are relative to the report period end."""
    period = pd.to_datetime(frame["period"].astype(str))
    result = {}
    for column in ("announcement_date", "available_from"):
        days = (pd.to_datetime(frame[column]) - period).dt.days.dropna()
        result[column] = dict(count=int(len(days)),
                              median_days=int(days.median()) if len(days) else None,
                              p10_days=int(days.quantile(.1)) if len(days) else None,
                              p90_days=int(days.quantile(.9)) if len(days) else None,
                              over_365_days=int((days > 365).sum()),
                              before_period_end=int((days < 0).sum()))
    return result


def listing_check(frame, history):
    """Count rows whose availability date precedes the stock's first traded bar."""
    first = {code: pd.to_datetime(value.date.min())
             for code, value in ((code, pd.read_parquet(path, columns=["date"]))
                                 for code, path in history.items())}
    listed = frame["code"].map(first)
    available = pd.to_datetime(frame["available_from"])
    known = listed.notna()
    before = known & (available < listed)
    return dict(codes_without_history=int((~known).sum()), rows_before_listing=int(before.sum()),
                codes_before_listing=int(frame.loc[before, "code"].nunique()))


def derive(snapshot_id, output, *, history_id=None, root=ROOT):
    store = Snapshots(root)
    saved, directory = store.get("fundamentals", snapshot_id)
    if saved.get("status") not in {"ready", "partial"}:
        raise ValueError("Only a completed PIT snapshot can be derived from")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.parent / ("." + output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    history = {}
    if history_id:
        _, bars = store.get("history", history_id)
        history = {path.stem: path for path in sorted((bars / "unadjusted").glob("*.parquet"))}
    tables, audit = {}, {"snapshot": snapshot_id, "history_snapshot": history_id}
    try:
        for kind in TABLES:
            path = directory / f"{kind}.parquet"
            if not path.exists():
                tables[kind] = dict(rows=0, status="missing")
                continue
            frame = pd.read_parquet(path)
            frame["available_from"], frame["availability_basis"] = availability(frame, kind)
            frame = frame.sort_values(["period", "code"]).reset_index(drop=True)
            frame.to_parquet(staging / f"{kind}.parquet", index=False)
            entry = dict(rows=len(frame), codes=int(frame["code"].nunique()),
                         periods=int(frame["period"].nunique()), basis=BASIS[kind],
                         lag=lag_profile(frame))
            if history:
                entry["listing"] = listing_check(frame, history)
            tables[kind] = entry
        audit["tables"] = tables
        audit["statutory_deadlines"] = {key: "%02d-%02d (year %+d)" % (value[1], value[2], value[0])
                                        for key, value in STATUTORY_DEADLINES.items()}
        audit["rule"] = ("Financials: available_from is the statutory disclosure deadline, because the "
                         "vendor exposes only the latest revision date. Forecasts and shareholders keep "
                         "their observed announcement date. Date-only values must still be shifted to the "
                         "next trading day before use.")
        audit["model_ready"] = False
        write_json(staging / "audit.json", audit)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="fundamentals snapshot id, for example f_2f8ef1efe22c430a")
    parser.add_argument("--history", help="optional history snapshot id used for the listing-date check")
    parser.add_argument("--output", required=True, help="new directory for the derived tables")
    arguments = parser.parse_args()
    audit = derive(arguments.snapshot, arguments.output, history_id=arguments.history)
    for kind, entry in audit["tables"].items():
        if not entry.get("rows"):
            print(f"{kind}: missing")
            continue
        lag = entry["lag"]
        print(f"{kind}: rows={entry['rows']} codes={entry['codes']} basis={entry['basis']} "
              f"vendor_median_lag={lag['announcement_date']['median_days']}d "
              f"available_median_lag={lag['available_from']['median_days']}d "
              f"vendor_over_365={lag['announcement_date']['over_365_days']}")
        if "listing" in entry:
            print(f"  rows before listing: {entry['listing']['rows_before_listing']}")


if __name__ == "__main__":
    main()
