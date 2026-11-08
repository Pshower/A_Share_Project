"""Can assets outside A-shares make ten percent drawdown and seven percent return coexist?

Inside A-shares the answer was no: the same construction that returns 7.05 percent over
2019-2024 returns 3.94 over 2010-2018, because the equity premium itself was near zero in
that stretch and no set of weights invents one.  The remaining untried lever is a different
asset, so this adds the S&P 500, the Hang Seng and Shanghai gold, and asks whether a mix
reaches the target in **both** windows rather than in a chosen one.

Two things make this less clean than it looks and both are reported rather than buried.

Foreign returns are earned in dollars or Hong Kong dollars, so a mainland investor's return
includes the currency.  Daily fixing rates are converted per calendar year and the series is
rebuilt in yuan; a flat conversion would misprice the 2015 devaluation badly.

The comparison also risks becoming "buy whatever just went up".  The United States had an
unusually strong decade while A-shares did not, so a global mix will look excellent for
reasons that are about one decade rather than about diversification.  That is why the target
is tested on the earlier window as well, and why a mix that only works in the later one is
reported as a failure.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools.allocation_study import returns_of
from tools.drawdown_target import fetch, drawdown_curve, evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "global"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "global_frontier"
WINDOWS = {"2019-2024": ("2019-01-02", "2024-12-31"),
           "2010-2018": ("2010-01-04", "2018-12-31")}
EQUITY = {"000300": 0.5, "000905": 0.3, "000852": 0.2}
EQUITY_YIELD = 0.0218
# Foreign series are quoted without dividends; the S&P yields about two percent and the
# Hang Seng about three and a half.  Gold yields nothing.
YIELDS = {"sp500": 0.020, "hsi": 0.035, "gold": 0.0}
STEP = 0.05
TARGET_RETURN, TARGET_DRAWDOWN = 0.07, 0.10


def cached(name, builder):
    cache = CACHE
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = builder()
        frame.to_csv(path, index=False)
    return frame


def foreign_close(name, symbol, kind):
    def builder():
        import akshare as ak
        if kind == "us":
            return ak.index_us_stock_sina(symbol=symbol)
        if kind == "hk":
            return ak.stock_hk_index_daily_sina(symbol=symbol)
        return ak.spot_hist_sge(symbol=symbol)
    frame = cached(name, builder)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["close"].astype(float).sort_index()


def usdcny():
    def builder():
        import akshare as ak
        frames = []
        for year in range(2005, 2025):
            try:
                block = ak.currency_boc_sina(symbol="美元", start_date=f"{year}0101",
                                             end_date=f"{year}1231")
                frames.append(block)
            except Exception:                       # noqa: BLE001 - provider is uneven
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    frame = cached("usdcny", builder)
    if frame.empty:
        return None
    frame = frame.rename(columns={"日期": "date"})
    column = [c for c in frame.columns if "中行" in str(c)][:1] or [frame.columns[-1]]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["rate"] = pd.to_numeric(frame[column[0]], errors="coerce")
    return frame.dropna(subset=["date", "rate"]).set_index("date")["rate"].sort_index()


def yuan_returns(close, fx=None):
    """Foreign price return, converted to yuan when a fixing series is supplied."""
    local = close.pct_change()
    if fx is None:
        return local.dropna()
    aligned = fx.reindex(local.index, method="ffill")
    moved = aligned.pct_change().fillna(0.0)
    return ((1 + local) * (1 + moved) - 1).dropna()


def assets(start, end):
    equity = pd.DataFrame({code: returns_of(code, start, end) for code in EQUITY}).dropna()
    blended = sum(equity[code] * weight for code, weight in EQUITY.items())
    blended = blended + EQUITY_YIELD / capm.TRADING_DAYS
    bond = fetch("000012", "sh000012").pct_change().dropna()
    fx = usdcny()
    sp = yuan_returns(foreign_close("sp500", ".INX", "us"), fx) + YIELDS["sp500"] / capm.TRADING_DAYS
    hk = yuan_returns(foreign_close("hsi", "HSI", "hk"), fx) + YIELDS["hsi"] / capm.TRADING_DAYS
    gold = foreign_close("gold", "Au99.99", "gold").pct_change().dropna()
    frame = pd.DataFrame({"ashare": blended, "bond": bond, "sp500": sp, "hsi": hk, "gold": gold})
    return frame.dropna().loc[start:end]


def grid(names, step=STEP):
    points = int(round(1 / step))
    for combination in itertools.product(range(points + 1), repeat=len(names) - 1):
        last = points - sum(combination)
        if last < 0:
            continue
        weights = list(combination) + [last]
        yield dict(zip(names, [w * step for w in weights]))


def frontier(frame, names):
    rows = []
    for weights in grid(names):
        series = sum(frame[name] * weight for name, weight in weights.items())
        block = evaluate(series)
        block["weights"] = weights
        rows.append(block)
    return rows


def best_under(rows, limit=TARGET_DRAWDOWN):
    usable = [row for row in rows if abs(row["drawdown"]) <= limit]
    return max(usable, key=lambda row: row["annual"]) if usable else None


def measure():
    payload = dict(status="ready", targets=dict(return_=TARGET_RETURN, drawdown=TARGET_DRAWDOWN),
                   step=STEP, yields=YIELDS, windows={})
    for label, (start, end) in WINDOWS.items():
        frame = assets(start, end)
        rows = frontier(frame, list(frame.columns))
        best = best_under(rows)
        payload["windows"][label] = dict(
            sessions=int(len(frame)),
            best=dict(annual=best["annual"], vol=best["vol"], drawdown=best["drawdown"],
                      sharpe=best["sharpe"], weights=best["weights"]) if best else None,
            meets=bool(best and best["annual"] >= TARGET_RETURN))
    left = payload["windows"]["2019-2024"]["best"]
    right = payload["windows"]["2010-2018"]["best"]
    payload["verdict"] = dict(
        meets_both=bool(payload["windows"]["2019-2024"]["meets"]
                        and payload["windows"]["2010-2018"]["meets"]),
        later=left["annual"] if left else None,
        earlier=right["annual"] if right else None)
    return payload


def _markdown(payload):
    lines = ["# 加入 A 股以外的资产：能不能稳定拿到 7% / 10%", "",
             "目标：年化 ≥ 7%、最大回撤 ≤ 10%。权重按 5% 步长穷举，取满足约束的最大收益。", ""]
    for label, block in payload["windows"].items():
        best = block["best"]
        lines += ["## %s（%d 个交易日）" % (label, block["sessions"]), ""]
        if not best:
            lines += ["没有组合满足约束。", ""]
            continue
        weights = "、".join("%s %.0f%%" % (name, value * 100) for name, value in best["weights"].items() if value)
        lines += ["**最优：%s**" % weights, "",
                  "| 年化 | 波动 | 最大回撤 | Sharpe | 达标 |",
                  "| ---: | ---: | ---: | ---: | --- |",
                  "| **%+.2f%%** | %.2f%% | %.2f%% | %+.3f | %s |" % (
                      best["annual"] * 100, best["vol"] * 100, best["drawdown"] * 100,
                      best["sharpe"], "是" if block["meets"] else "否"), ""]
    verdict = payload["verdict"]
    lines += ["## 结论", "",
              "两个窗口都达标：**%s**（2019-2024 %s / 2010-2018 %s）。" % (
                  "是" if verdict["meets_both"] else "否",
                  "%+.2f%%" % (verdict["later"] * 100) if verdict["later"] is not None else "—",
                  "%+.2f%%" % (verdict["earlier"] * 100) if verdict["earlier"] is not None else "—"), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Global assets against a drawdown target.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "global_frontier.json", payload)
    (target / "global_frontier.md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
