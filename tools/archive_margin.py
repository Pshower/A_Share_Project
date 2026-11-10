"""Archive the per-stock 融资融券 balances, one exchange-day at a time.

Both exchanges publish a per-stock table for each trading day and nothing else: there is no
bulk endpoint, so this is ~2,900 small requests and about two hours.  The archive is
therefore built to be **resumable** -- every fetched day is appended and the status file
records the days already present, so an interrupted run continues where it stopped instead
of starting over.

Only three quantities per name are kept: the financing balance, the day's financing
purchases, and the short-selling balance.  The first is the cumulative leverage position,
the second is the flow that moves it.  Nothing here interprets either.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "online" / "margin"
KEEP = {"融资余额": "balance", "融资买入额": "buy", "融资偿还额": "repay",
        "融券余量": "short_balance", "融券卖出量": "short_sell", "融券偿还量": "short_repay"}
CODES = ("标的证券代码", "证券代码")
DATES = ("信用交易日期", "交易日期")


def pick(frame, names):
    for name in names:
        if name in frame.columns:
            return name
    return None


def normalize(frame, date):
    """One exchange table to ``(date, code, ...)`` in yuan and shares."""
    code_col = pick(frame, CODES)
    if code_col is None:
        raise ValueError("No stock-code column in %s" % list(frame.columns)[:6])
    data = pd.DataFrame({"code": frame[code_col].astype(str).str.zfill(6)})
    date_col = pick(frame, DATES)
    data["date"] = pd.to_datetime(frame[date_col]).dt.date if date_col else date
    for source, target in KEEP.items():
        column = pick(frame, (source,))
        if column is not None:
            data[target] = pd.to_numeric(frame[column], errors="coerce")
    return data.dropna(subset=["code"])


def trading_days(start, end):
    from tools import trading_calendar
    dates = trading_calendar.load()
    window = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    return [stamp.date() for stamp in window]


def call_with_timeout(fn, timeout, **kwargs):
    """Run one request in a daemon thread so a hung endpoint costs ``timeout``, not the run.

    ``socket.setdefaulttimeout`` does not cover a name lookup or a server that accepts the
    connection and then says nothing, and this archive has already frozen twice on exactly
    that.  The thread is abandoned rather than joined: a daemon thread cannot block exit.
    """
    import threading
    box = {}

    def target():
        try:
            box["value"] = fn(**kwargs)
        except BaseException as error:              # noqa: BLE001 - reported to the caller
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("no response within %ss" % timeout)
    if "error" in box:
        raise box["error"]
    return box["value"]


def run(start="2019-01-02", end="2024-12-31", pause=0.0, timeout=45, log=print):
    import akshare as ak
    import time
    # Two ways this loop dies quietly, both fixed here.  akshare draws a tqdm bar on stderr
    # for every request, and under a captured pipe nobody drains it, so the child eventually
    # blocks on the write; and a request can hang with no timeout at all.  The first is
    # silenced at the source, the second is bounded by ``call_with_timeout``.
    os.environ.setdefault("TQDM_DISABLE", "1")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "margin.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["code", "date"])
    done = set(existing["date"]) if len(existing) else set()
    days = [day for day in trading_days(start, end) if day not in done]
    log("margin: %d days already archived, %d to fetch" % (len(done), len(days)))
    frames = [existing] if len(existing) else []
    failures = []
    for position, day in enumerate(days, 1):
        started = time.monotonic()
        collected = []
        for name, fetch in (("sse", ak.stock_margin_detail_sse), ("szse", ak.stock_margin_detail_szse)):
            try:
                with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
                    frame = call_with_timeout(fetch, timeout, date=day.strftime("%Y%m%d"))
                if frame is not None and len(frame):
                    collected.append(normalize(frame, day))
            except Exception as error:                    # noqa: BLE001 - one exchange, one day
                failures.append(dict(date=str(day), exchange=name, error=str(error)[:200]))
        if collected:
            frames.append(pd.concat(collected, ignore_index=True))
        if position % 10 == 0:
            merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code", "date"], keep="last")
            merged.to_parquet(path, index=False)
            frames = [merged]
            log("  %d/%d days, %d rows" % (position, len(days), len(merged)))
        if time.monotonic() - started > 15:
            log("  slow day %s: %.0fs" % (day, time.monotonic() - started))
        if pause:
            time.sleep(pause)
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code", "date"], keep="last")
    merged = merged.sort_values(["date", "code"])
    merged.to_parquet(path, index=False)
    status = dict(archived_at=datetime.now(timezone.utc).isoformat(),
                  requested=dict(start=str(start), end=str(end)), rows_total=int(len(merged)),
                  codes_total=int(merged["code"].nunique()), days=int(merged["date"].nunique()),
                  first_date=str(merged["date"].min()), last_date=str(merged["date"].max()),
                  failures=failures[-20:], failure_count=len(failures))
    (OUTPUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    log("archived %d rows, %d days, %d codes" % (status["rows_total"], status["days"], status["codes_total"]))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="Archive per-stock margin balances.")
    parser.add_argument("--start", default="2019-01-02")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.start, args.end), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
