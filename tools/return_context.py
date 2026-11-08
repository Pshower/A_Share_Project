"""Put one window's return in the context of every other window.

A five percent annual figure from 2019-2024 invites the objection that equity is not worth
holding.  The number is real but it is a property of the window, not of the asset: the
period ends near a multi-year low after a long decline, and a start or end date moved by
two years changes the answer by more than any allocation decision discussed so far.

Two corrections belong with any quoted figure and both are reported here.  The indices
carried by the repository are **price** indices, so the dividend is simply missing; A-share
large caps yield roughly two to three percent and that is a measurement gap rather than a
strategy.  And a single start date is one draw, so the distribution of rolling five-year
returns says more than the point estimate does.

The rolling figures are descriptive.  Choosing the start date that looks best is exactly
the fitting this repository keeps refusing to do.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "index_long"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "return_context"
SYMBOLS = {"000300": "sh000300", "000905": "sh000905", "399317": "sz399317"}
LABELS = {"000300": "沪深300", "000905": "中证500", "399317": "国证A指"}
WINDOWS = [("全部历史", None, None), ("2019-2024", "2019-01-02", "2024-12-31"),
           ("2010-2026", "2010-01-04", "2026-11-06"), ("2015-2026", "2015-01-05", "2026-11-06"),
           ("2021-2024", "2021-01-04", "2024-12-31")]
ROLL = 243 * 5


def fetch(code, cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{code}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        import akshare as ak
        frame = ak.stock_zh_index_daily(symbol=SYMBOLS[code])
        frame.to_csv(path, index=False)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["close"].astype(float).sort_index()


def window_stats(nav, start=None, end=None):
    series = nav.loc[start:end].dropna()
    if len(series) < 200:
        return None
    years = len(series) / capm.TRADING_DAYS
    annual = float((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1)
    returns = series.pct_change().dropna()
    vol = float(returns.std(ddof=1) * math.sqrt(capm.TRADING_DAYS))
    drawdown = float((series / series.cummax() - 1).min())
    return dict(start=str(series.index[0].date()), end=str(series.index[-1].date()),
                years=float(years), annual=annual, vol=vol, drawdown=drawdown,
                total=float(series.iloc[-1] / series.iloc[0] - 1))


def rolling(nav, length=ROLL):
    if len(nav) <= length:
        return None
    cagr = (nav.shift(-length) / nav) ** (capm.TRADING_DAYS / length) - 1
    values = cagr.dropna()
    if values.empty:
        return None
    return dict(observations=int(len(values)), median=float(values.median()),
                p10=float(values.quantile(0.10)), p90=float(values.quantile(0.90)),
                share_negative=float((values < 0).mean()))


def measure(code="000300"):
    nav = fetch(code)
    payload = dict(index=code, label=LABELS.get(code, code),
                   first=str(nav.index[0].date()), last=str(nav.index[-1].date()),
                   basis="price index, dividends excluded", windows={}, rolling=None)
    for label, start, end in WINDOWS:
        block = window_stats(nav, start, end)
        if block:
            payload["windows"][label] = block
    payload["rolling"] = rolling(nav)
    return payload


def _markdown(payloads):
    lines = ["# 把 5% 放回它的窗口里", "",
             "**所有数字都是价格指数，不含分红。** 这是一个度量缺口，不是策略结论。", ""]
    for payload in payloads:
        lines += ["## %s（%s）" % (payload["label"], payload["index"]), "",
                  "区间 %s ~ %s" % (payload["first"], payload["last"]), "",
                  "| 窗口 | 年数 | 年化 | 波动 | 最大回撤 | 累计 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for label, block in payload["windows"].items():
            lines.append("| %s | %.1f | %+.2f%% | %.2f%% | %.2f%% | %+.1f%% |" % (
                label, block["years"], block["annual"] * 100, block["vol"] * 100,
                block["drawdown"] * 100, block["total"] * 100))
        roll = payload.get("rolling")
        if roll:
            lines += ["", "**滚动 5 年年化的分布**（%d 个观测）：中位 %+.2f%%，p10 %+.2f%%，p90 %+.2f%%，"
                          "负值占 %.0f%%。" % (
                              roll["observations"], roll["median"] * 100, roll["p10"] * 100,
                              roll["p90"] * 100, roll["share_negative"] * 100), ""]
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Window sensitivity of index returns.")
    parser.add_argument("--index", default="000300")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payloads = [measure(code) for code in SYMBOLS]
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "return_context.json", payloads)
    (target / "return_context.md").write_text(_markdown(payloads), encoding="utf-8")
    print(_markdown(payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
