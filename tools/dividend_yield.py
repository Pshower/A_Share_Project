"""How much dividend does a price index leave out?

An ETF holds the shares and receives their cash dividends.  A price index does not, so
comparing an ETF's expected return against a price index understates it by roughly the
dividend yield -- a measurement gap, not a strategy.  This measures the gap directly
instead of quoting a rule of thumb.

The yield is computed on **unadjusted** prices, because a dividend is paid in real yuan
against a real quote.  Using the forward-adjusted series would silently shrink every
historical dividend relative to the price it is divided by, and the error grows the further
back the sample goes.

The sample is the largest names by median float capitalisation, which is a good proxy for
the constituents of a large-cap index but is not its official membership.  So the answer
is an estimate of the dividend yield of a large-cap A-share portfolio, not of CSI 300
itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.full_market_panel import load as load_panel_full

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "dividend"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "dividend_yield"
PANEL = PROJECT_ROOT / "data" / "clean" / "panel_full"
COUNT = 300
START, END = "2019-01-02", "2024-12-31"


def sample_codes(panel_dir=PANEL, count=COUNT, skip=0):
    manifest, prices, traded, shares = load_panel_full(panel_dir)
    capitalisation = (traded * shares).median().sort_values(ascending=False)
    return list(capitalisation.index[skip:skip + count]), capitalisation


def dividends(code, cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{code}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        import akshare as ak
        frame = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
        frame.to_csv(path, index=False)
    if frame is None or frame.empty:
        return None
    frame = frame.rename(columns={"派息": "cash_per_ten", "除权除息日": "ex_date", "进度": "status"})
    if "cash_per_ten" not in frame.columns or "ex_date" not in frame.columns:
        return None
    frame = frame[frame.get("status", "实施").astype(str).str.contains("实施", na=False)]
    frame["ex_date"] = pd.to_datetime(frame["ex_date"], errors="coerce")
    frame["cash"] = pd.to_numeric(frame["cash_per_ten"], errors="coerce") / 10.0
    return frame.dropna(subset=["ex_date", "cash"])[["ex_date", "cash"]]


def measure(count=COUNT, start=START, end=END, panel_dir=PANEL, skip=0):
    manifest, prices, traded, shares = load_panel_full(panel_dir)
    codes, capitalisation = sample_codes(panel_dir, count, skip)
    unadjusted = pd.read_parquet(Path(panel_dir) / "unadj_close.parquet")
    unadjusted = unadjusted.loc[start:end]
    rows, missing, events = [], [], 0
    for code in codes:
        table = dividends(code)
        if table is None or code not in unadjusted.columns:
            missing.append(code)
            continue
        series = unadjusted[code].dropna()
        if series.empty:
            missing.append(code)
            continue
        for row in table.itertuples(index=False):
            if row.ex_date < series.index[0] or row.ex_date > series.index[-1]:
                continue
            price = float(series.asof(row.ex_date))
            if not np.isfinite(price) or price <= 0:
                continue
            rows.append(dict(code=code, year=row.ex_date.year, yield_=row.cash / price,
                             weight=float(capitalisation.get(code, np.nan))))
            events += 1
    if not rows:
        return dict(status="empty", codes=len(codes), missing=len(missing))
    frame = pd.DataFrame(rows)
    by_year = {}
    years = sorted(frame["year"].unique())
    for year in years:
        block = frame[frame["year"] == year]
        weights = block["weight"].fillna(block["weight"].median())
        by_year[int(year)] = float((block["yield_"] * weights).sum() / weights.sum())
    mean_yield = float(np.mean(list(by_year.values())))
    return dict(status="ready", codes=len(codes), skip=skip, covered=len(codes) - len(missing),
                missing=len(missing), events=int(events), by_year=by_year,
                mean_annual_yield=mean_yield, window=dict(start=start, end=end),
                basis="unadjusted price, cap-weighted across the largest names",
                note="estimate for a large-cap portfolio, not the official index yield")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the dividend a price index omits.")
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(args.count, skip=args.skip)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "dividend_yield.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
