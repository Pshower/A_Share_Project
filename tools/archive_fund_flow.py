"""Archive a rolling-window feed before it rolls away.

``stock_individual_fund_flow`` serves roughly the last 120 sessions and takes no date
range, so anything not captured while it is inside the window is gone.  The feed is not
worth a daily job: each pull reaches back about six months, so a weekly pass overlaps the
previous one by roughly 115 sessions and no trading day can slip between two runs.

The store is append-only and deduplicated on (code, date), so a rerun after a partial
failure cannot double-count and a later revision cannot silently overwrite an earlier
observation.
"""
from __future__ import annotations

from datetime import date

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = PROJECT_ROOT / "data" / "online" / "universe" / "a_share_spot.json"
STORE = PROJECT_ROOT / "data" / "online" / "fund_flow"
KEEP = ["date", "code", "close", "pct_change", "main_net", "main_pct"]


def universe_codes(limit=None):
    rows = json.loads(UNIVERSE.read_text(encoding="utf-8"))["rows"]
    codes = [str(row["code"]).zfill(6) for row in rows]
    return codes[:limit] if limit else codes


def market_of(code):
    return "sh" if code.startswith(("6", "9")) else "sz"


def existing(index_path):
    if index_path.exists():
        return pd.read_parquet(index_path)
    return pd.DataFrame(columns=KEEP)


def fetch(code, pause=0.0):
    import time
    if pause:
        time.sleep(pause)
    import akshare as ak
    frame = ak.stock_individual_fund_flow(stock=code, market=market_of(code))
    if frame is None or frame.empty:
        return None
    frame = frame.rename(columns={"日期": "date", "收盘价": "close", "涨跌幅": "pct_change",
                                  "主力净流入-净额": "main_net", "主力净流入-净占比": "main_pct"})
    missing = [name for name in KEEP if name != "code" and name not in frame.columns]
    if missing:
        raise ValueError(f"{code} missing {missing}")
    frame = frame[[name for name in KEEP if name != "code"]].copy()
    frame["code"] = code
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("close", "pct_change", "main_net", "main_pct"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def last_trading_day(today=None):
    """The most recent session at or before today; None when the calendar cannot answer."""
    from tools import trading_calendar

    today = pd.Timestamp(today or date.today()).normalize()
    try:
        dates = trading_calendar.load()
    except Exception:                                  # noqa: BLE001 - offline is not fatal
        return None
    past = dates[dates <= today]
    return past.max() if len(past) else None


def run(limit=None, codes=None, pause=0.2, store=STORE, log=print):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    index_path = store / "flow.parquet"
    current = existing(index_path)
    target = last_trading_day()
    targets = codes or universe_codes(limit)
    fresh, failed, skipped = [], [], []
    for position, code in enumerate(targets, 1):
        have = current.loc[current["code"] == code, "date"]
        # Skip only a code that already reaches the *target date*.  Using the store's own maximum as
        # the target made the second weekly run a no-op by construction: every healthy code was
        # already at that maximum, so nothing was fetched and the archive froze on whatever date the
        # first run happened to capture (2026-11-06 in the real log).
        if target is not None and len(have) and have.max() >= target:
            skipped.append(code)
            continue
        try:
            frame = fetch(code, pause)
        except Exception as error:                     # noqa: BLE001 - provider varies
            failed.append(dict(code=code, error=type(error).__name__))
            continue
        if frame is None or frame.empty:
            failed.append(dict(code=code, error="empty"))
            continue
        fresh.append(frame)
        if position % 200 == 0:
            log("fetched %d/%d" % (position, len(targets)))
    added = 0
    if fresh:
        # pandas 2.1 rejects a concat whose datetime columns differ in unit (ns vs s); normalise
        # both sides so the merge does not depend on which pandas the machine happens to have.
        def dated(frame):
            return frame.assign(date=pd.to_datetime(frame["date"]).astype("datetime64[ns]"))

        merged = pd.concat([dated(current)] + [dated(frame) for frame in fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=["code", "date"], keep="first").sort_values(["code", "date"])
        merged.to_parquet(index_path, index=False)
        added = int(len(merged) - len(current))
        current = merged
    summary = dict(archived_at=datetime.now(timezone.utc).isoformat(),
                   requested=len(targets), fetched=len(fresh), skipped=len(skipped),
                   failed=len(failed), rows_added=added, rows_total=int(len(current)),
                   codes_total=int(current["code"].nunique()),
                   first_date=str(current["date"].min().date()) if len(current) else None,
                   last_date=str(current["date"].max().date()) if len(current) else None,
                   failures=failed[:10])
    (store / "status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps({k: summary[k] for k in ("requested", "fetched", "skipped", "failed",
                                            "rows_total", "last_date")}, ensure_ascii=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Archive the rolling fund-flow feed.")
    parser.add_argument("--limit", type=int, default=None, help="First N universe codes, for a smoke run.")
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between requests.")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_network:
        print("Refusing to hit the network without --allow-network")
        return 2
    summary = run(args.limit, pause=args.pause, store=args.store)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
