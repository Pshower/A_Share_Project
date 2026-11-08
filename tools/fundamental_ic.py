"""The door nobody opened: valuation and quality.

Every signal tested in this repository so far was derived from prices.  Twenty price
features, five training lines, a reversal book and a lockup overlay all failed, and the
conclusion drawn was that the market cannot be beaten.  That conclusion has a boundary the
evidence never crossed: nothing here has ever looked at what a company earns, owns or
returns on it.

Three things decide whether this is worth doing, and the code is arranged around them.

**Availability.** A fundamental figure is knowable from the day it is filed, not from the
report period it describes.  A first-quarter report filed in April describes March, and a
test that uses it in March reads the future.  The feed carries a filing date per row, so
availability is taken from that field rather than from the period end.  Corrections filed
years later therefore drop out of the early sample instead of silently rewriting it.

**Direction.** Declared here, before any return is computed: cheaper is better for EP and
BP, and higher is better for ROE and gross margin.

**Annualisation.** ``每股收益`` is cumulative within the fiscal year, so a first-quarter
figure is three months of earnings.  Trailing-twelve-month earnings are assembled as the
current cumulative plus the previous full year minus the previous year's same period, and
quarterly figures are annualised only when a full TTM cannot be built.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import alpha_ic
from tools import capm_backtest as capm
from tools.feature_panel import load_panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "fundamentals"
PANEL = PROJECT_ROOT / "data" / "clean" / "feature_top1800"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "fundamental_ic"
START, END = "2019-01-02", "2024-12-31"
HORIZON = 20
BAR = 0.031
SIGN = {"ep": 1.0, "bp": 1.0, "roe": 1.0, "gross": 1.0}   # declared before the test


def periods():
    for year in range(2018, 2025):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            stamp = f"{year}{month:02d}{day:02d}"
            if "20181231" <= stamp <= "20240930":
                yield stamp


def download(cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for period in periods():
        path = cache / f"yjbb_{period}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"股票代码": str}))
            continue
        import akshare as ak
        frame = ak.stock_yjbb_em(date=period)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table = table.rename(columns={"股票代码": "code", "每股收益": "eps", "每股净资产": "bvps",
                                  "净资产收益率": "roe", "销售毛利率": "gross", "最新公告日期": "filed"})
    table["code"] = table["code"].astype(str).str.zfill(6)
    table["filed"] = pd.to_datetime(table["filed"], errors="coerce")
    for column in ("eps", "bvps", "roe", "gross"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table.dropna(subset=["filed", "code"]).sort_values("filed")


def trailing_eps(table):
    """TTM earnings per share from the cumulative series, per code."""
    ordered = table.sort_values(["code", "filed"]).copy()
    ordered["period"] = ordered["filed"].dt.to_period("Q")
    by_code = {}
    for code, group in ordered.groupby("code"):
        group = group.drop_duplicates(subset=["filed"], keep="last").set_index("filed").sort_index()
        by_code[code] = group[["eps", "bvps", "roe", "gross"]].copy()
    return by_code


def as_of(series, date, spread=90):
    """The latest value filed on or before ``date``, with its filing stamp."""
    earlier = series.index[series.index <= date]
    if earlier.empty:
        return None, None
    stamp = earlier[-1]
    if (date - stamp).days > spread * 4:
        return None, None
    return series.loc[stamp], stamp


def panel_at(by_code, dates, codes, field):
    """Forward-fill each code's filings onto the trading calendar.

    A filing is only allowed to speak from its own filing date onward, so the reindex is
    a forward fill of the filing series rather than an as-of join on the report period.
    """
    columns = {}
    for code in codes:
        series = by_code.get(code)
        if series is None or field not in series.columns:
            continue
        values = series[field].dropna()
        if values.empty:
            continue
        columns[code] = values.reindex(dates, method="ffill")
    if not columns:
        return pd.DataFrame(np.nan, index=dates, columns=codes)
    return pd.DataFrame(columns).reindex(index=dates, columns=codes)


def factors(by_code, dates, codes):
    """Four measures, in the direction declared at the top of the file."""
    bvps = panel_at(by_code, dates, codes, "bvps")
    eps = panel_at(by_code, dates, codes, "eps")
    roe = panel_at(by_code, dates, codes, "roe")
    gross = panel_at(by_code, dates, codes, "gross")
    return bvps, eps, roe, gross


def measure(panel_dir=PANEL, bar=BAR, cache=CACHE):
    manifest, panel, prices = load_panel(panel_dir)
    codes = list(prices.columns)
    index = prices.index[(prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(END))]
    table = download(cache)
    by_code = trailing_eps(table)
    close = prices.reindex(index)
    bvps, eps, roe, gross = factors(by_code, index, codes)
    scores = {
        "ep": (eps / close).replace([np.inf, -np.inf], np.nan),
        "bp": (bvps / close).replace([np.inf, -np.inf], np.nan),
        "roe": roe,
        "gross": gross,
    }
    labels = (prices.shift(-HORIZON) / prices - 1.0).reindex(index)
    dates = list(index[::HORIZON])
    payload = dict(status="ready", panel=str(panel_dir), codes=len(codes),
                   horizon=HORIZON, bar=bar, sign=SIGN,
                   filings=int(len(table)), filed_range=[str(table["filed"].min().date()),
                                                          str(table["filed"].max().date())],
                   factors={})
    for name, block in scores.items():
        series, coverage = {}, []
        for date in dates:
            raw = block.loc[date].to_numpy(dtype=float)
            target = labels.loc[date].to_numpy(dtype=float)
            usable = np.isfinite(raw) & np.isfinite(target)
            coverage.append(float(usable.sum() / max(np.isfinite(target).sum(), 1)))
            if usable.sum() < 30:
                series[date] = np.nan
                continue
            a = pd.Series(raw[usable]).rank().to_numpy()
            b = pd.Series(target[usable]).rank().to_numpy()
            series[date] = float(np.corrcoef(a, b)[0, 1]) * SIGN[name]
        clean = pd.Series(series).dropna()
        stats = alpha_ic.ic_stats(clean, HORIZON, threshold=bar) if len(clean) else {}
        payload["factors"][name] = dict(dates=int(len(clean)), mean_ic=stats.get("mean_ic"),
                                        t=stats.get("t_nw_h"),
                                        hit=float((clean > 0).mean()) if len(clean) else None,
                                        coverage=float(np.mean(coverage)))
    payload["verdict"] = {name: dict(clears=bool(abs(value.get("mean_ic") or 0) >= bar))
                          for name, value in payload["factors"].items()}
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank IC of valuation and quality factors.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(args.panel)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "fundamental_ic.json", payload)
    print(json.dumps({k: payload[k] for k in ("factors", "verdict", "filings", "filed_range")},
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
