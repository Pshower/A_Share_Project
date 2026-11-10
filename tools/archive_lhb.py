"""Archive the 龙虎榜 (top-trader) disclosures, one month at a time.

The list is an **event** source: it only exists for stocks that moved enough to be posted,
so its coverage is a property of the data, not a choice.  That is exactly why it has to be
archived before it is tested -- a signal whose denominator is "who happened to be on the
list" cannot be evaluated from whatever a single interface call returns today.

Storage follows ``archive_fund_flow``: one parquet, append-only, deduplicated on
``(code, date, reason)``, with a status file that records what was asked for and what came
back.  Nothing here interprets the numbers.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "online" / "lhb"
COLUMNS = {"代码": "code", "上榜日": "date", "龙虎榜净买额": "net_buy", "龙虎榜买入额": "buy",
           "龙虎榜卖出额": "sell", "龙虎榜成交额": "amount", "市场总成交额": "market_amount",
           "换手率": "turnover", "流通市值": "float_cap", "上榜原因": "reason", "解读": "note"}
KEYS = ["code", "date", "reason"]


def normalize(frame):
    data = frame.rename(columns=COLUMNS)
    missing = [name for name in ("code", "date", "net_buy", "amount") if name not in data]
    if missing:
        raise ValueError("Unexpected LHB schema, missing %s" % ",".join(missing))
    data = data[[name for name in COLUMNS.values() if name in data]].copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"]).dt.date
    for name in ("net_buy", "buy", "sell", "amount", "market_amount", "turnover", "float_cap"):
        if name in data:
            data[name] = pd.to_numeric(data[name], errors="coerce")
    return data.dropna(subset=KEYS)


def fetch(start, end, log=print):
    """One akshare call per month: the interface is chunked internally and keeps failing
    on multi-year ranges."""
    import akshare as ak
    import calendar
    pieces = []
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            if year == end.year and month > end.month:
                break
            first = max(start, pd.Timestamp(year=year, month=month, day=1).date())
            last = min(end, pd.Timestamp(year=year, month=month,
                                         day=calendar.monthrange(year, month)[1]).date())
            if first > last:
                continue
            frame = ak.stock_lhb_detail_em(start_date=first.strftime("%Y%m%d"),
                                           end_date=last.strftime("%Y%m%d"))
            if frame is not None and len(frame):
                pieces.append(normalize(frame))
                log("%s-%02d: %d rows" % (year, month, len(frame)))
    if not pieces:
        raise ValueError("No LHB rows returned")
    return pd.concat(pieces, ignore_index=True)


def run(start="2019-01-01", end="2024-12-31", log=print):
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    fresh = fetch(start, end, log=log)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "lhb.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        before = len(existing)
        combined = pd.concat([existing, fresh], ignore_index=True)
    else:
        before, combined = 0, fresh
    combined = combined.drop_duplicates(subset=KEYS, keep="last").sort_values(KEYS)
    combined.to_parquet(path, index=False)
    status = dict(archived_at=datetime.now(timezone.utc).isoformat(),
                  requested=dict(start=str(start), end=str(end)), rows_fetched=int(len(fresh)),
                  rows_before=before, rows_total=int(len(combined)),
                  codes_total=int(combined["code"].nunique()),
                  first_date=str(combined["date"].min()), last_date=str(combined["date"].max()))
    (OUTPUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    log("archived %d rows (%d new), %d codes, %s..%s" % (
        status["rows_total"], len(combined) - before, status["codes_total"],
        status["first_date"], status["last_date"]))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="Archive 龙虎榜 disclosures.")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.start, args.end), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
