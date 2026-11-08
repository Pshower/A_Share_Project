"""Rebalancing, risk scaling and volatility targeting -- none of which predicts anything.

The allocation study settled the weights and left three questions that need no forecast:

* how often to reset them, given that every reset costs money and doing nothing costs drift;
* how much to hold in cash, which is the only honest way to promise a smaller drawdown;
* whether scaling exposure to recent realised volatility helps, which is a mechanical rule
  but still a form of timing and is reported as such.

Every rule here uses only information available at the time it trades.  Nothing is fitted
to the sample and nothing is ranked by return afterwards; the point of the exercise is to
show the spread of outcomes across rules, so that a reader can see whether the choice
matters at all rather than being told which one happened to win.

Costs: the three funds are exchange-traded, so the round trip is charged at the same
commission and slippage the equity work uses.  Fund management fees are **not** modelled.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools.allocation_study import FIT, CHECK, returns_of, stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "allocation_robustness"
WEIGHTS = {"000300": 0.5, "000905": 0.3, "000852": 0.2}
START, END = "2019-01-02", "2024-12-31"
COST = 0.0005          # one round trip on the weight that changes
BAND = 0.05            # absolute deviation that triggers a band rebalance
TARGETS = (0.15, 0.20)
CASH_EXPOSURES = (1.0, 0.8, 0.6, 0.4)


def load_window(start=START, end=END):
    codes = sorted(WEIGHTS)
    frame = pd.DataFrame({code: returns_of(code, start, end) for code in codes}).dropna()
    return frame


def blend(frame, weights, *, period=None, band=None, cost=COST):
    """Fixed weights, either reset every ``period`` sessions or when drift exceeds ``band``.

    Returns the daily return series and the annualised one-way turnover actually traded.
    """
    codes = list(weights)
    target = np.array([weights[code] for code in codes], dtype=float)
    values = frame[codes].to_numpy(dtype=float)
    out = np.zeros(len(values))
    current = target.copy()
    traded = 0.0
    for position in range(len(values)):
        out[position] = float(current @ values[position])
        grown = current * (1.0 + values[position])
        total = grown.sum()
        current = grown / total if total > 0 else current
        trigger = False
        if period and (position + 1) % period == 0:
            trigger = True
        elif band and np.abs(current - target).max() > band:
            trigger = True
        if trigger:
            move = 0.5 * float(np.abs(target - current).sum())
            if move > 0:
                out[position] -= cost * move
                traded += move
                current = target.copy()
    years = len(values) / capm.TRADING_DAYS
    return pd.Series(out, index=frame.index), traded / years if years else float("nan")


def rebalance_table(frame):
    rules = [("不再平衡", dict(period=None, band=None)),
             ("按 5% 偏离", dict(period=None, band=BAND)),
             ("每季度", dict(period=61)),
             ("每半年", dict(period=121)),
             ("每年", dict(period=243))]
    rows = []
    for label, options in rules:
        series, turnover = blend(frame, WEIGHTS, **options)
        block = stats(series)
        rows.append(dict(rule=label, annual=block["annual"], vol=block["vol"],
                         drawdown=block["drawdown"], sharpe=block["sharpe"],
                         turnover=turnover, cost=turnover * COST))
    return rows


def cash_ladder(frame):
    """The blend diluted with cash at 2%, which is a promise about drawdown, not return."""
    base, _ = blend(frame, WEIGHTS, period=None, band=BAND)
    rows = []
    for exposure in CASH_EXPOSURES:
        series = exposure * base + (1 - exposure) * (capm.CASH_RATE / capm.TRADING_DAYS)
        block = stats(series)
        rows.append(dict(exposure=exposure, annual=block["annual"], vol=block["vol"],
                         drawdown=block["drawdown"], sharpe=block["sharpe"]))
    return rows


def volatility_target(frame, target, window=60):
    """Scale exposure to the inverse of trailing realised volatility, capped at one."""
    base, _ = blend(frame, WEIGHTS, period=243)
    realised = base.rolling(window).std(ddof=1) * math.sqrt(capm.TRADING_DAYS)
    scale = (target / realised).clip(upper=1.0).shift(1).fillna(1.0)
    scaled = base * scale + (1 - scale) * (capm.CASH_RATE / capm.TRADING_DAYS)
    turnover = float(scale.diff().abs().sum() / (len(scale) / capm.TRADING_DAYS))
    scaled = scaled - scale.diff().abs().fillna(0.0) * COST
    block = stats(scaled)
    return dict(target=target, window=window, annual=block["annual"], vol=block["vol"],
                drawdown=block["drawdown"], sharpe=block["sharpe"],
                mean_exposure=float(scale.mean()), turnover=turnover,
                worst_exposure=float(scale.min()))


def measure():
    frame = load_window()
    payload = dict(status="ready", window=dict(start=START, end=END), weights=WEIGHTS,
                   sessions=int(len(frame)), cost_round_trip=COST, band=BAND)
    payload["rebalancing"] = rebalance_table(frame)
    payload["cash"] = cash_ladder(frame)
    payload["vol_target"] = [volatility_target(frame, target) for target in TARGETS]
    static, _ = blend(frame, WEIGHTS, period=243)
    payload["static"] = stats(static)
    payload["spread"] = dict(
        annual=float(max(row["annual"] for row in payload["rebalancing"])
                     - min(row["annual"] for row in payload["rebalancing"])),
        drawdown=float(max(row["drawdown"] for row in payload["rebalancing"])
                       - min(row["drawdown"] for row in payload["rebalancing"])))
    return payload


def _markdown(payload):
    lines = ["# 再平衡、现金与波动目标", "",
             "窗口 %s ~ %s，%d 个交易日。三 ETF 权重 50/30/20。单次往返成本 %.0fbp。" % (
                 payload["window"]["start"], payload["window"]["end"],
                 payload["sessions"], payload["cost_round_trip"] * 1e4), "",
             "## 再平衡规则", "",
             "| 规则 | 年化 | 波动 | 最大回撤 | Sharpe | 年换手 | 年成本 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["rebalancing"]:
        lines.append("| %s | %+.2f%% | %.2f%% | %.2f%% | %+.3f | %.2f | %.2fbp |" % (
            row["rule"], row["annual"] * 100, row["vol"] * 100, row["drawdown"] * 100,
            row["sharpe"], row["turnover"], row["cost"] * 1e4))
    lines += ["", "**规则之间的差距：年化 %.2fpp、回撤 %.2fpp。**" % (
        payload["spread"]["annual"] * 100, payload["spread"]["drawdown"] * 100), "",
              "## 现金比例", "",
              "| 股票暴露 | 年化 | 波动 | 最大回撤 | Sharpe |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in payload["cash"]:
        lines.append("| %.0f%% | %+.2f%% | %.2f%% | %.2f%% | %+.3f |" % (
            row["exposure"] * 100, row["annual"] * 100, row["vol"] * 100,
            row["drawdown"] * 100, row["sharpe"]))
    lines += ["", "## 波动目标", "",
              "| 目标波动 | 年化 | 波动 | 最大回撤 | Sharpe | 平均暴露 | 最低暴露 | 年换手 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["vol_target"]:
        lines.append("| %.0f%% | %+.2f%% | %.2f%% | %.2f%% | %+.3f | %.0f%% | %.0f%% | %.2f |" % (
            row["target"] * 100, row["annual"] * 100, row["vol"] * 100, row["drawdown"] * 100,
            row["sharpe"], row["mean_exposure"] * 100, row["worst_exposure"] * 100, row["turnover"]))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rebalancing, cash and volatility targeting.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "allocation_robustness.json", payload)
    (target / "allocation_robustness.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(dict(spread=payload["spread"], static=payload["static"],
                          rebalancing=payload["rebalancing"], cash=payload["cash"],
                          vol_target=payload["vol_target"]), ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
