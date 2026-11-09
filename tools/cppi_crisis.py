"""What the floor actually looks like inside a named crisis.

A backtest summary says the maximum drawdown was nine percent.  It does not say what
happened in October 2008, whether the strategy kept its exposure through the panic, how long
it stayed out, or when the account got back to its previous high.  For a rule whose entire
claim is about behaviour during a fall, those are the numbers that matter, and they are the
ones a summary hides.

The state is carried in rather than reset.  Starting fresh at each crisis would assume a
full ten percent cushion on the eve of the crash, which is the most favourable possible
position and not the one an investor would have held after a bull market.  Running one
continuous path from 2005 and slicing it keeps the cushion honest.

The opposite question is asked as well, because it is the one a new investor faces: if you
started this rule the day before each crash, with a full cushion and no history, what would
the first year have looked like?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools import global_frontier as gf
from tools.allocation_study import returns_of
from tools.cppi_strategy import BAND, COST, EQUITY_YIELD, FLOOR, SP_YIELD
from tools.drawdown_target import fetch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "cppi_crisis"
START, END = "2005-01-04", "2024-12-31"
MULTIPLIER = 6
EPISODES = {
    "2008 金融危机": ("2008-01-02", "2008-12-31"),
    "2015 股灾": ("2015-06-01", "2016-02-29"),
    "2018 熊市": ("2018-01-02", "2018-12-28"),
    "2020 疫情": ("2020-01-20", "2020-03-31"),
    "2022 调整": ("2022-01-04", "2022-12-30"),
}


def full_frame():
    local = pd.DataFrame({"000300": returns_of("000300", START, END),
                          "000905": returns_of("000905", START, END)}).dropna()
    ashare = local.mean(axis=1) + EQUITY_YIELD / capm.TRADING_DAYS
    bond = fetch("000012", "sh000012").pct_change().dropna()
    sp500 = gf.yuan_returns(gf.foreign_close("sp500", ".INX", "us"), gf.usdcny())
    sp500 = sp500 + SP_YIELD / capm.TRADING_DAYS
    frame = pd.DataFrame({"ashare": ashare, "sp500": sp500, "bond": bond}).dropna()
    return frame.loc[START:END]


def walk(frame, multiplier=MULTIPLIER, floor=FLOOR, band=BAND, cost=COST, warm=False):
    """One continuous path; optionally restart from a full cushion on the first day."""
    risky = 0.5 * frame["ashare"] + 0.5 * frame["sp500"]
    safe = frame["bond"]
    nav, peak, held = 1.0, 1.0, (0.0 if not warm else min(multiplier * floor, 1.0))
    rows = []
    for date, r_risky, r_safe in zip(frame.index, risky.to_numpy(), safe.to_numpy()):
        cushion = nav - floor * peak
        target = min(multiplier * max(cushion, 0.0) / nav, 1.0)
        if band and abs(target - held) < band:
            target = held
        move = abs(target - held)
        nav *= (1 - move * cost)
        nav *= (1 + target * r_risky + (1 - target) * r_safe)
        peak = max(peak, nav)
        held = target
        rows.append(dict(date=date, nav=nav, peak=peak, exposure=target,
                         r_risky=r_risky, cushion=cushion))
    return pd.DataFrame(rows).set_index("date")


def episode_block(state, risky, start, end):
    window = state.loc[start:end]
    if window.empty:
        return None
    entered = float(window["nav"].iloc[0])
    exited = float(window["nav"].iloc[-1])
    curve = window["nav"] / window["nav"].cummax()
    path = risky.loc[start:end]
    risky_total = float((1 + path).prod() - 1)
    # how long until the account is back above where it stood when the episode began
    after = state.loc[end:]
    above = after[after["nav"] >= entered]
    recovery = int((above.index[0] - window["nav"].idxmin()).days) if len(above) else None
    return dict(start=str(window.index[0].date()), end=str(window.index[-1].date()),
                sessions=int(len(window)),
                strategy_return=float(exited / entered - 1),
                risky_return=risky_total,
                strategy_drawdown=float(curve.min() - 1),
                min_exposure=float(window["exposure"].min()),
                mean_exposure=float(window["exposure"].mean()),
                days_at_floor=int((window["exposure"] <= 0.01).sum()),
                recovery_days=recovery)


def measure():
    frame = full_frame()
    risky = 0.5 * frame["ashare"] + 0.5 * frame["sp500"]
    continuous = walk(frame)
    payload = dict(status="ready", multiplier=MULTIPLIER, floor=FLOOR, band=BAND,
                   sessions=int(len(frame)), carried={}, fresh={},
                   full=dict(annual=float(continuous["nav"].iloc[-1] ** (252.0 / len(continuous)) - 1),
                             drawdown=float((continuous["nav"] / continuous["nav"].cummax() - 1).min())))
    for label, (start, end) in EPISODES.items():
        payload["carried"][label] = episode_block(continuous, risky, start, end)
        fresh = walk(frame.loc[start:end], warm=True)
        payload["fresh"][label] = episode_block(fresh, risky, start, end)
    return payload


def _markdown(payload):
    lines = ["# 危机分段：楼层在真正的崩盘里长什么样", "",
             "倍数 %d、楼层 %.2f、不交易带 %.0f%%。路径从 2005 连续跑到 2024，缓冲垫**承接前文**，不每天重置。" % (
                 payload["multiplier"], payload["floor"], payload["band"] * 100), "",
             "## 承接历史状态（真实情形）", "",
             "| 事件 | 区间 | 风险腿 | **策略** | 期间最大回撤 | 平均暴露 | 最低暴露 | 贴地板天数 | 恢复天数 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, block in payload["carried"].items():
        if not block:
            continue
        lines.append("| %s | %s~%s | %+.1f%% | **%+.2f%%** | %.2f%% | %.0f%% | %.0f%% | %d | %s |" % (
            label, block["start"][2:], block["end"][2:], block["risky_return"] * 100,
            block["strategy_return"] * 100, block["strategy_drawdown"] * 100,
            block["mean_exposure"] * 100, block["min_exposure"] * 100,
            block["days_at_floor"], block["recovery_days"] if block["recovery_days"] is not None else "未恢复"))
    lines += ["", "## 崩盘前一天才开始（满缓冲垫）", "",
              "| 事件 | 风险腿 | 策略 | 期间最大回撤 | 最低暴露 | 贴地板天数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, block in payload["fresh"].items():
        if not block:
            continue
        lines.append("| %s | %+.1f%% | **%+.2f%%** | %.2f%% | %.0f%% | %d |" % (
            label, block["risky_return"] * 100, block["strategy_return"] * 100,
            block["strategy_drawdown"] * 100, block["min_exposure"] * 100, block["days_at_floor"]))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPPI through named crises.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "cppi_crisis.json", payload)
    (target / "cppi_crisis.md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
