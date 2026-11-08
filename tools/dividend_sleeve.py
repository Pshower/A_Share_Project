"""Does adding a dividend sleeve to the blend help, on a total-return basis?

The fundamental test found book-to-price carries a positive rank IC, but concentrated in
2022-2024, which makes it a style exposure rather than a forecast.  A style exposure can be
bought.  中证红利 is the standard vessel for the value-and-yield end of the A-share market
and has large ETFs tracking it, so the question stops being "can I rank stocks" and becomes
"does putting part of the equity sleeve there help".

Every series is reported twice: as a price index, which is what the repository stores, and
as price plus measured dividend, which is what a fund actually receives.  Reporting only
the first is what produced the earlier five percent figure and the objection it drew.

Yields come from the per-stock measurement: each name's mean annual dividend over
2019-2024 divided by its ex-date price.  The dividend sleeve's yield is estimated from the
hundred highest-yielding names in that sample, since the index selects on yield and the
official constituent list is not in the repository.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools import dividend_yield as dy
from tools.allocation_study import returns_of, stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "dividend_sleeve"
CACHE = PROJECT_ROOT / "data" / "online" / "index_long"
START, END = "2019-01-02", "2024-12-31"
DIV_CODE = "000015"          # 上证红利：2005 起有全历史，且 510880 追踪它
DIV_SYMBOL = "sh000015"
# A second vessel for the same exposure, used only to check that the result is a property
# of the style rather than the history of one index.
SIBLINGS = {"000015": "sh000015", "399324": "sz399324"}
BASE = {"000300": 0.5, "000905": 0.3, "000852": 0.2}
SLEEVES = (0.0, 0.1, 0.2, 0.3, 0.4)
COST = 0.0005


def fetch_dividend_index(code=None, cache=CACHE):
    code = code or DIV_CODE
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{code}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        import akshare as ak
        frame = ak.stock_zh_index_daily(symbol=SIBLINGS[code])
        frame.to_csv(path, index=False)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["close"].astype(float).sort_index()


def per_stock_yields(count=1500):
    """Mean annual dividend yield per stock, reusing the dividend cache."""
    codes, capitalisation = dy.sample_codes(dy.PANEL, count)
    rows = []
    for code in codes:
        table = dy.dividends(code)
        if table is None or table.empty:
            continue
        frame = pd.read_parquet(Path(dy.PANEL) / "unadj_close.parquet", columns=[code])
        series = frame[code].dropna()
        if series.empty:
            continue
        values = []
        for row in table.itertuples(index=False):
            if row.ex_date < series.index[0] or row.ex_date > series.index[-1]:
                continue
            price = float(series.asof(row.ex_date))
            if np.isfinite(price) and price > 0:
                values.append(row.cash / price)
        if values:
            years = max((series.index[-1] - series.index[0]).days / 365.25, 1.0)
            rows.append(dict(code=code, mean_yield=float(np.sum(values) / years)))
    return pd.DataFrame(rows)


def daily_from_annual(annual, index):
    return pd.Series(annual / capm.TRADING_DAYS, index=index)


def measure(dividend_share=None, index=None):
    code_used = index or DIV_CODE
    frame = pd.DataFrame({code: returns_of(code, START, END) for code in BASE}).dropna()
    prices = fetch_dividend_index(code_used).loc[START:END]
    dividend = prices.pct_change().dropna()
    shared = frame.index.intersection(dividend.index)
    frame, dividend = frame.loc[shared], dividend.loc[shared]

    yields = per_stock_yields()
    top = yields.nlargest(100, "mean_yield")["mean_yield"].mean()
    base_yields = {"000300": 0.0258, "000905": 0.0183, "000852": 0.0168}
    payload = dict(status="ready", window=dict(start=START, end=END), sessions=int(len(shared)),
                   dividend_index=code_used, dividend_yield_estimate=float(top),
                   base_yields=base_yields, rows=[], top_yield_names=int(len(yields)))
    for share in SLEEVES:
        blended = None
        for code, weight in BASE.items():
            piece = frame[code] * weight * (1 - share)
            blended = piece if blended is None else blended + piece
        blended = blended + dividend * share
        price_stats = stats(blended)
        total = blended.copy()
        for code, weight in BASE.items():
            total = total + daily_from_annual(base_yields[code] * weight * (1 - share), shared)
        total = total + daily_from_annual(top * share, shared)
        total_stats = stats(total)
        payload["rows"].append(dict(dividend_share=share,
                                    price=price_stats, total=total_stats))
    return payload


def _markdown(payload):
    lines = ["# 加一个红利袖套：价格口径 vs 全收益口径", "",
             "红利指数 `%s`，股息率估计 **%.2f%%**（样本里股息率最高的 100 只均值）。"
             "其余袖套沿用实测股息：大盘 2.58%%、中盘 1.83%%、小盘 1.68%%。" % (
                 payload["dividend_index"], payload["dividend_yield_estimate"] * 100), "",
             "| 红利占比 | 年化（价格） | 回撤（价格） | Sharpe（价格） | 年化（全收益） | 波动（全收益） | 回撤（全收益） | **Sharpe（全收益）** |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["rows"]:
        price, total = row["price"], row["total"]
        lines.append("| %.0f%% | %+.2f%% | %.2f%% | %+.3f | %+.2f%% | %.2f%% | %.2f%% | **%+.3f** |" % (
            row["dividend_share"] * 100, price["annual"] * 100, price["drawdown"] * 100,
            price["sharpe"], total["annual"] * 100, total["vol"] * 100,
            total["drawdown"] * 100, total["sharpe"]))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add a dividend sleeve to the blend.")
    parser.add_argument("--index", default=None, choices=sorted(SIBLINGS))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(index=args.index)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "dividend_sleeve.json", payload)
    (target / "dividend_sleeve.md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
