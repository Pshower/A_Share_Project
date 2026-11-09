"""Track a broad all-share exposure with ETFs that actually exist.

The menu collapses faster than it looks.  Over 2019-2024 the daily correlation between
CSI 500, CSI 1000 and the equal-weight 500 is 0.96 to 0.99, so "mid, small and equal
weight" are one exposure, not three.  What is genuinely different is large caps against
everything else, and the broadest published index -- 399317 -- beat every narrower one on
return, drawdown and Sharpe at once.

The question this answers is therefore small: given a few investable funds, what weights
reproduce the broad index, and what does the tracking cost?  Weights are chosen to
minimise tracking error, not return, and they are fitted on one half and checked on the
other, because a weight vector chosen by maximising historical return is exactly the
mistake this repository keeps finding.
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

FIT = ("2019-01-02", "2021-12-31")
CHECK = ("2022-01-03", "2024-12-31")
# Every code here has a large, liquid tracking ETF; 399317 is the target, not a holding.
FUNDS = {"000300": "沪深300ETF", "000905": "中证500ETF", "000852": "中证1000ETF"}
TARGET = "399317"
REBALANCE_DAYS = 243
COST = 0.0005          # one round trip on the weight that changes


def returns_of(code, start, end):
    series = capm.index_returns(code, start, end)
    if series is None or series.empty:
        raise ValueError(f"index {code} unavailable")
    return series


def frame(codes, start, end):
    columns = {code: returns_of(code, start, end) for code in codes}
    return pd.DataFrame(columns).dropna()


def stats(series, start=None, end=None):
    window = series.loc[start:end] if start else series
    if len(window) < 60:
        return dict(status="insufficient")
    nav = (1 + window).cumprod()
    years = len(window) / capm.TRADING_DAYS
    annual = float(nav.iloc[-1] ** (1 / years) - 1)
    vol = float(window.std(ddof=1) * math.sqrt(capm.TRADING_DAYS))
    drawdown = float((nav / nav.cummax() - 1).min())
    return dict(status="ready", annual=annual, vol=vol, drawdown=drawdown,
                sharpe=(annual - capm.CASH_RATE) / vol if vol > 0 else float("nan"))


def blend(frame_, weights, rebalance=REBALANCE_DAYS, cost=COST):
    """Fixed weights, reset on a schedule, with the one-way turnover charged."""
    codes = list(weights)
    weight = np.array([weights[code] for code in codes], dtype=float)
    values = frame_[codes].to_numpy(dtype=float)
    out = np.zeros(len(values))
    current = weight.copy()
    for position in range(len(values)):
        out[position] = float(current @ values[position])
        grown = current * (1.0 + values[position])
        total = grown.sum()
        current = grown / total if total > 0 else current
        if rebalance and (position + 1) % rebalance == 0:
            out[position] -= cost * 0.5 * float(np.abs(weight - current).sum())
            current = weight.copy()
    return pd.Series(out, index=frame_.index)


def tracking(series, target):
    joined = pd.concat([series.rename("book"), target.rename("target")], axis=1, join="inner").dropna()
    if len(joined) < 60:
        return dict(status="insufficient")
    active = joined["book"] - joined["target"]
    return dict(status="ready", observations=int(len(joined)),
                tracking_error=float(active.std(ddof=1) * math.sqrt(capm.TRADING_DAYS)),
                correlation=float(joined["book"].corr(joined["target"])),
                annual_gap=float(((1 + joined["book"]).prod() ** (capm.TRADING_DAYS / len(joined)) - 1)
                                 - ((1 + joined["target"]).prod() ** (capm.TRADING_DAYS / len(joined)) - 1)))


def grid(step=0.1):
    """Every non-negative weight vector on a 0.1 grid that sums to one."""
    points = int(round(1 / step))
    for a, b in itertools.product(range(points + 1), repeat=2):
        c = points - a - b
        if c < 0:
            continue
        yield {"000300": a * step, "000905": b * step, "000852": c * step}


def fit_weights(train, target_train, step=0.1):
    rows = []
    for weights in grid(step):
        series = blend(train, weights)
        detail = tracking(series, target_train)
        if detail.get("status") != "ready":
            continue
        rows.append(dict(weights=dict(weights), tracking_error=detail["tracking_error"],
                         annual_gap=detail["annual_gap"],
                         summary=stats(series, train.index[0], train.index[-1])))
    rows.sort(key=lambda row: row["tracking_error"])
    return rows


def measure(funds=FUNDS, target=TARGET, step=0.1):
    codes = list(funds) + [target]
    full = frame(codes, "2019-01-02", "2024-12-31")
    train = full.loc[FIT[0]:FIT[1]]
    check = full.loc[CHECK[0]:CHECK[1]]
    payload = dict(status="ready", funds=funds, target=target,
                   fit=dict(start=FIT[0], end=FIT[1]), check=dict(start=CHECK[0], end=CHECK[1]),
                   components={}, notes=[])
    for code in codes:
        payload["components"][code] = dict(
            label=funds.get(code, "目标指数（不可直接持有）"),
            full=stats(full[code], FIT[0], CHECK[1]),
            train=stats(train[code], FIT[0], FIT[1]),
            check=stats(check[code], CHECK[0], CHECK[1]))
    ranked = fit_weights(train, train[target], step)
    payload["fitted"] = ranked[:8]
    best = ranked[0]["weights"] if ranked else None
    if best:
        payload["best"] = dict(weights=best)
        payload["best"]["train"] = tracking(blend(train, best), train[target])
        payload["best"]["check"] = tracking(blend(check, best), check[target])
        payload["best"]["full"] = stats(blend(full, best), FIT[0], CHECK[1])
        payload["best"]["target_full"] = stats(full[target], FIT[0], CHECK[1])
        payload["notes"].append("weights minimise tracking error against %s on %s..%s and are "
                                "checked on %s..%s" % (target, FIT[0], FIT[1], CHECK[0], CHECK[1]))
    return payload


def _markdown(payload):
    lines = ["# 用 ETF 逼近全市场暴露", "",
             "口径：日收益，权重按 %d 个交易日重置，单边换手计 %.0fbp。" % (REBALANCE_DAYS, COST * 1e4), "",
             "| 代码 | 说明 | 全样本年化 | 波动 | 最大回撤 | Sharpe |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for code, block in payload["components"].items():
        s = block["full"]
        if s.get("status") != "ready":
            continue
        lines.append("| `%s` | %s | %+.2f%% | %.2f%% | %.2f%% | %+.3f |" % (
            code, block["label"], s["annual"] * 100, s["vol"] * 100, s["drawdown"] * 100, s["sharpe"]))
    best = payload.get("best")
    if best:
        weights = "、".join("%s %.0f%%" % (code, value * 100) for code, value in best["weights"].items())
        lines += ["", "**最优跟踪组合**：" + weights, "",
                  "| 区间 | 跟踪误差 | 相关 | 年化差 |", "| --- | ---: | ---: | ---: |"]
        for label in ("train", "check"):
            block = best[label]
            if block.get("status") == "ready":
                lines.append("| %s | %.2f%% | %.4f | %+.2fpp |" % (
                    label, block["tracking_error"] * 100, block["correlation"], block["annual_gap"] * 100))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Track the all-share exposure with ETFs.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    payload = measure()
    if args.output:
        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        from src.data.preprocess import write_json
        write_json(target / "allocation_study.json", payload)
        (target / "allocation_study.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("best",)}, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
