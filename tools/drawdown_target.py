"""Is seven percent a year with a ten percent drawdown reachable at all?

Those two numbers together are a demanding pair.  Over 2019-2024 the A-share blend returned
about 7.5 percent including dividends while losing 37 percent at its worst, so a portfolio
meeting both constraints needs something the equity blend cannot supply on its own: either a
second asset with a much flatter path, or a rule that steps aside before the fall.

Two assets are tested.  Bond indices are available back to 2003 and, unlike the equity
indices here, are quoted as wealth series that already accrue coupon, so no dividend
adjustment is applied to them.  The equity series is the three-ETF blend with its measured
dividend added, since a price index would otherwise report a return nobody receives.

Three questions, in order: what is the best return achievable subject to a ten percent
drawdown using stocks and bonds alone; does a trend rule on the equity sleeve do better; and
if neither reaches seven percent, what the actual frontier looks like, so that the target can
be priced instead of merely missed.

Maximum drawdown is a path statistic, not a variance statistic, so it is computed on the
realised path rather than approximated from volatility.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools.allocation_study import returns_of, stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "online" / "index_long"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "drawdown_target"
START, END = "2019-01-02", "2024-12-31"
EQUITY = {"000300": 0.5, "000905": 0.3, "000852": 0.2}
EQUITY_YIELD = 0.0218        # measured: 2.58 / 1.83 / 1.68 weighted by the blend
BONDS = {"399481": "sz399481", "000012": "sh000012"}
TARGET_RETURN, TARGET_DRAWDOWN = 0.07, 0.10
GRID = np.round(np.arange(0.0, 1.0001, 0.02), 4)
TREND_WINDOWS = (100, 150, 200)


def fetch(code, symbol, cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{code}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        import akshare as ak
        frame = ak.stock_zh_index_daily(symbol=symbol)
        frame.to_csv(path, index=False)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["close"].astype(float).sort_index()


def equity_series():
    frame = pd.DataFrame({code: returns_of(code, START, END) for code in EQUITY}).dropna()
    blended = sum(frame[code] * weight for code, weight in EQUITY.items())
    return blended + EQUITY_YIELD / capm.TRADING_DAYS


def drawdown_curve(returns):
    """Worst peak-to-trough fall, with the opening value counted as a peak.

    Taking the running maximum of the compounded series alone misses the case where the
    loss starts on the first day: the first NAV is already below the one it was measured
    from, so the peak would be taken from a level the account had already lost.  Prepending
    the opening value fixes that and makes the first observation eligible as a trough.
    """
    nav = (1 + returns).cumprod().to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.concatenate(([1.0], nav)))[1:]
    return float((nav / peak - 1).min())


def evaluate(returns):
    block = stats(returns)
    return dict(annual=block.get("annual"), vol=block.get("vol"),
                drawdown=drawdown_curve(returns), sharpe=block.get("sharpe"))


def frontier(equity, bond):
    shared = equity.index.intersection(bond.index)
    e, b = equity.loc[shared], bond.loc[shared]
    rows = []
    for weight in GRID:
        blended = weight * e + (1 - weight) * b
        block = evaluate(blended)
        block["equity"] = float(weight)
        rows.append(block)
    return rows, shared


def best_under(rows, limit=TARGET_DRAWDOWN):
    usable = [row for row in rows if abs(row["drawdown"]) <= limit]
    return max(usable, key=lambda row: row["annual"]) if usable else None


def trend_rule(equity, bond, price, window):
    """Hold equity while its own price is above a moving average, otherwise hold bonds."""
    moving = price.rolling(window).mean()
    signal = (price > moving).astype(float).shift(1).fillna(0.0)
    shared = equity.index.intersection(bond.index)
    signal = signal.reindex(shared).fillna(0.0)
    blended = signal * equity.loc[shared] + (1 - signal) * bond.loc[shared]
    block = evaluate(blended)
    block["window"] = window
    block["average_equity"] = float(signal.mean())
    block["switches"] = int((signal.diff().abs() > 0).sum())
    return block, blended


def measure():
    equity = equity_series()
    payload = dict(status="ready", window=dict(start=START, end=END),
                   target=dict(return_=TARGET_RETURN, drawdown=TARGET_DRAWDOWN),
                   equity_yield=EQUITY_YIELD, bonds={}, frontier={}, trend={})
    for code, symbol in BONDS.items():
        bond = fetch(code, symbol).pct_change().dropna()
        bond = bond.loc[START:END]
        rows, shared = frontier(equity, bond)
        payload["bonds"][code] = evaluate(bond.loc[shared])
        payload["frontier"][code] = rows
        payload["best_under_" + code] = best_under(rows)
    price = fetch("000300", "sh000300").loc[START:END]
    for window in TREND_WINDOWS:
        bond = fetch("399481", "sz399481").pct_change().dropna().loc[START:END]
        block, series = trend_rule(equity, bond, price, window)
        payload["trend"][window] = block
    payload["verdict"] = check(payload)
    return payload


def check(payload):
    best = payload.get("best_under_399481") or {}
    trend_hits = [block for block in payload.get("trend", {}).values()
                  if block["annual"] >= TARGET_RETURN and abs(block["drawdown"]) <= TARGET_DRAWDOWN]
    return dict(
        stocks_and_bonds=dict(annual=best.get("annual"), drawdown=best.get("drawdown"),
                              equity=best.get("equity")),
        stocks_and_bonds_meets=bool(best.get("annual", 0) >= TARGET_RETURN),
        trend_meets=bool(trend_hits),
        achievable=bool(best.get("annual", 0) >= TARGET_RETURN or trend_hits))


def _markdown(payload):
    lines = ["# 回撤 10% 以内、年化 7%：能不能做到", "",
             "窗口 %s ~ %s。股票袖套为三 ETF 混合加实测股息 %.2f%%。"
             "债券指数视为财富序列（已计息），不再加股息。" % (
                 payload["window"]["start"], payload["window"]["end"],
                 payload["equity_yield"] * 100), ""]
    for code, block in payload["bonds"].items():
        lines.append("债券 `%s` 单独：年化 %+.2f%%，波动 %.2f%%，最大回撤 %.2f%%。" % (
            code, block["annual"] * 100, block["vol"] * 100, block["drawdown"] * 100))
    lines.append("")
    for code, rows in payload["frontier"].items():
        lines += ["## 股债前沿（债券 `%s`）" % code, "",
                  "| 股票占比 | 年化 | 波动 | 最大回撤 | Sharpe |", "| ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            if row["equity"] % 0.1 < 1e-9:
                lines.append("| %.0f%% | %+.2f%% | %.2f%% | %.2f%% | %+.3f |" % (
                    row["equity"] * 100, row["annual"] * 100, row["vol"] * 100,
                    row["drawdown"] * 100, row["sharpe"]))
        best = payload.get("best_under_" + code)
        if best:
            lines.append("")
            lines.append("**回撤 ≤ 10%% 时能拿到的最好收益：%.0f%% 股票 → 年化 %+.2f%%，回撤 %.2f%%。**" % (
                best["equity"] * 100, best["annual"] * 100, best["drawdown"] * 100))
        lines.append("")
    lines += ["## 趋势规则", "",
              "| 均线窗口 | 年化 | 波动 | 最大回撤 | Sharpe | 平均股票仓位 | 切换次数 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for window, block in payload["trend"].items():
        lines.append("| %s | %+.2f%% | %.2f%% | %.2f%% | %+.3f | %.0f%% | %d |" % (
            window, block["annual"] * 100, block["vol"] * 100, block["drawdown"] * 100,
            block["sharpe"], block["average_equity"] * 100, block["switches"]))
    lines += ["", "**结论：%s**" % ("达标" if payload["verdict"]["achievable"] else "达不到"), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Can ten percent drawdown coexist with seven percent?")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "drawdown_target.json", payload)
    (target / "drawdown_target.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
