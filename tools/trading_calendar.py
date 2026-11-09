"""Official A-share trading days, cached so the daily jobs can tell a holiday from a fault.

The exchange publishes the calendar a year ahead, so a scheduled job can ask the question in
advance rather than inferring it from whether data arrived.  That distinction matters: a job
that cannot tell "today was a holiday" from "the data feed is broken" will report a fault on
every public holiday.

The cache is refreshed when it no longer reaches far enough into the future, which is once a
year in practice.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "calendar"
HORIZON = 30          # refresh when fewer than this many days of calendar remain


def load(refresh=False, cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "trade_dates.csv"
    today = pd.Timestamp(date.today())
    if path.exists() and not refresh:
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame["trade_date"])
        if dates.max() >= today + pd.Timedelta(days=HORIZON):
            return dates
    import akshare as ak
    frame = ak.tool_trade_date_hist_sina()
    frame.to_csv(path, index=False)
    return pd.to_datetime(frame["trade_date"])


def is_trading_day(when=None, calendar=None):
    stamp = pd.Timestamp(when or date.today()).normalize()
    dates = calendar if calendar is not None else load()
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
