"""Whether the safe leg can still pay what the strategy was built on.

The drawdown-capped strategy needs the bond leg to return roughly four percent a year.  That
figure came from the treasury index over 2019-2024, a period in which Chinese yields fell,
so part of the return was a one-off capital gain rather than carry.  A bond index return is
approximately the average yield plus the price change from yields moving, and only the first
half repeats.

The distinction decides whether the published result is a description of a mechanism or a
description of a bond bull market.  Yields at the end of the sample are the best available
guide to what carry can be collected next, since nobody can forecast the next move in yields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools.drawdown_target import fetch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "yields"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "bond_yield_context"
WINDOWS = {"2019-2024": ("2019-01-02", "2024-12-31"),
           "2005-2018": ("2005-01-04", "2018-12-31"),
           "2005-2012": ("2005-01-04", "2012-12-31"),
           "2013-2018": ("2013-01-04", "2018-12-31")}


def yields(start="2005-01-01", cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "china_treasury.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        import akshare as ak
        frame = ak.bond_zh_us_rate(start_date=start.replace("-", ""))
        frame.to_csv(path, index=False)
    frame = frame.rename(columns={"日期": "date", "中国国债收益率2年": "y2",
                                  "中国国债收益率5年": "y5", "中国国债收益率10年": "y10",
                                  "中国国债收益率30年": "y30"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    for column in ("y2", "y5", "y10", "y30"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def bond_returns(start, end):
    series = fetch("000012", "sh000012").pct_change().dropna()
    return series.loc[start:end]


def decompose(table, start, end):
    window = table.loc[start:end].dropna(subset=["y10"])
    if window.empty:
        return None
    returns = bond_returns(start, end)
    years = len(returns) / capm.TRADING_DAYS
    total = float((1 + returns).prod() ** (1 / years) - 1) if len(returns) else float("nan")
    first, last = float(window["y10"].iloc[0]), float(window["y10"].iloc[-1])
    average = float(window["y10"].mean()) / 100.0
    change = last - first
    return dict(start=start, end=end, years=float(years), index_return=total,
                average_ten_year=average, first_ten_year=first, last_ten_year=last,
                change_bp=float(change * 100),
                carry=average,
                capital=total - average,
                note="a fall in yields turns carry plus a capital gain; a rise turns it into a loss")


def measure():
    table = yields()
    payload = dict(status="ready", windows={}, latest=None)
    for label, (start, end) in WINDOWS.items():
        block = decompose(table, start, end)
        if block:
            payload["windows"][label] = block
    last = table.dropna(subset=["y10"]).iloc[-1]
    payload["latest"] = dict(date=str(table.dropna(subset=["y10"]).index[-1].date()),
                             y2=float(last["y2"]), y5=float(last["y5"]),
                             y10=float(last["y10"]), y30=float(last["y30"]))
    payload["verdict"] = dict(
        strategy_needs=0.040,
        carry_available=float(last["y10"]) / 100.0,
        shortfall=0.040 - float(last["y10"]) / 100.0,
        feasible=bool(float(last["y10"]) / 100.0 >= 0.040))
    return payload


def _markdown(payload):
    lines = ["# 安全腿还能给多少：国债收益率给的答案", "",
             "债券指数的收益 ≈ **平均收益率（carry）** + **收益率变动带来的价差**。只有前半部分会重复。", "",
             "| 窗口 | 债券指数年化 | 平均 10 年期 | 期初 → 期末 | 变动 | carry | 价差 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, block in payload["windows"].items():
        lines.append("| %s | %+.2f%% | %.2f%% | %.2f%% → %.2f%% | %+.0fbp | %.2f%% | %+.2f%% |" % (
            label, block["index_return"] * 100, block["average_ten_year"] * 100,
            block["first_ten_year"], block["last_ten_year"], block["change_bp"],
            block["carry"] * 100, block["capital"] * 100))
    latest = payload["latest"]
    verdict = payload["verdict"]
    lines += ["", "## 当前收益率（%s）" % latest["date"], "",
              "| 2 年 | 5 年 | 10 年 | 30 年 |", "| ---: | ---: | ---: | ---: |",
              "| %.2f%% | %.2f%% | **%.2f%%** | %.2f%% |" % (
                  latest["y2"], latest["y5"], latest["y10"], latest["y30"]), "",
              "**策略需要安全腿给到 %.1f%%，当前 10 年期是 %.2f%%，缺口 %.2fpp。**" % (
                  verdict["strategy_needs"] * 100, verdict["carry_available"] * 100,
                  verdict["shortfall"] * 100), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="What the bond leg can carry.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "bond_yield_context.json", payload)
    (target / "bond_yield_context.md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
