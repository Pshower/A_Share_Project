"""Official A-share trading days, cached so the daily jobs can tell a holiday from a fault.

The exchange publishes the calendar a year ahead, so a scheduled job can ask the question in
advance rather than inferring it from whether data arrived.  That distinction matters: a job
that cannot tell "today was a holiday" from "the data feed is broken" will report a fault on
every public holiday.

The cache is refreshed when it no longer reaches far enough past today.  The exchange publishes
up to the next year end, not a rolling year, so the test is "reaches a week past today": asking for
a full year ahead could never be satisfied and made every call download the file again.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "calendar"
COVER = 7             # days of slack past today before the cache counts as "not far enough"


def _covers(dates, today, days=COVER) -> bool:
    return not len(dates) == 0 and dates.max() >= today + pd.Timedelta(days=days)


def load(refresh=False, cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "trade_dates.csv"
    today = pd.Timestamp(date.today())
    cached = None
    if path.exists():
        cached = pd.to_datetime(pd.read_csv(path)["trade_date"])
        if not refresh:
            if _covers(cached, today):
                return cached
            if _covers(cached, today, days=0) and _refreshed_today(path):
                # The exchange publishes to the next year end, so in the last days of December the
                # week of slack can never be met.  One attempt per day is enough; the cache still
                # answers today's question.
                return cached
    import akshare as ak
    try:
        frame = ak.tool_trade_date_hist_sina()
    except Exception:
        if cached is not None and _covers(cached, today, days=0):
            # The feed is down but the cache still answers today's question; a refresh failure
            # must not turn "is today a trading day" into an exception.
            return cached
        raise
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)          # a half-written calendar would be worse than a stale one
    return pd.to_datetime(frame["trade_date"])


def _refreshed_today(path: Path) -> bool:
    stamp = datetime.fromtimestamp(path.stat().st_mtime).date()
    return stamp == date.today()


def is_trading_day(when=None, calendar=None):
    """True/False for a date the calendar covers; refuse beyond it instead of answering "holiday".

    The cache ends on a date, and past that date every day looks like a public holiday.  Both daily
    jobs treat False as "do nothing today", so a stale calendar would make them go quiet for a year
    without a single error -- the exact confusion this module exists to prevent.
    """
    stamp = pd.Timestamp(when or date.today()).normalize()
    dates = calendar if calendar is not None else load()
    if stamp > dates.max():
        raise ValueError("trading calendar ends %s, before %s; refresh it"
                         % (dates.max().date(), stamp.date()))
    return bool((dates == stamp).any())


def describe(when=None):
    stamp = pd.Timestamp(when or date.today()).normalize()
    dates = load()
    return dict(date=str(stamp.date()), weekday=stamp.day_name(),
                trading=bool((dates == stamp).any()),
                calendar_first=str(dates.min().date()), calendar_last=str(dates.max().date()),
                trading_days=int(len(dates)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="A-share trading calendar.")
    parser.add_argument("--date", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.refresh:
        load(refresh=True)
    print(json.dumps(describe(args.date), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
