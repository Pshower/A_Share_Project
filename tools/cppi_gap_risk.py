"""Does the floor actually hold, or does it only hold because the market was kind?

Constant proportion portfolio insurance promises to keep the drawdown under the floor by
selling the risky leg as the cushion shrinks.  It assumes the sale can happen before the
fall.  If the risky leg drops further in one session than the cushion can absorb, the floor
is breached on that day and the promise fails -- quietly, in a single print, rather than
gradually.

The arithmetic is cleaner than it looks.  With exposure set to ``m`` times the cushion, a
one-day fall of the risky leg of ``1/m`` wipes the cushion exactly: at seven times, any
session worse than minus fourteen percent breaks the floor regardless of position size.  So
the question is not "did the strategy do well" but "how close did the worst session come to
that number", and that is what this measures.

Two other things are reported because they are what makes the answer usable: the worst
single-day fall of the risky leg in each window, and how much further it could have fallen
on its worst day before the floor broke.  The second is the actual safety margin, expressed
in market terms rather than in portfolio terms.

A no-trade band makes exposure lag the rule, so the effective multiplier on the day of a
fall can be higher than the nominal one.  The check therefore uses the exposure that was
actually held, not the exposure the formula would have wanted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools.cppi_strategy import BAND, COST, FLOOR, MULTIPLIERS, WINDOWS, build

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "cppi_gap_risk"


def walk(frame, multiplier, floor=FLOOR, band=BAND, cost=COST):
    """Replay CPPI and record, each day, the state that decides the next day's risk."""
    risky = 0.5 * frame["ashare"] + 0.5 * frame["sp500"]
    safe = frame["bond"]
    nav, peak, held = 1.0, 1.0, 0.0
    rows = []
    for date, r_risky, r_safe in zip(frame.index, risky.to_numpy(), safe.to_numpy()):
        cushion = nav - floor * peak
        target = min(multiplier * max(cushion, 0.0) / nav, 1.0)
        if band and abs(target - held) < band:
            target = held
        move = abs(target - held)
        nav_after_cost = nav * (1 - move * cost)
        # the floor can only be breached between today's decision and tomorrow's return
        rows.append(dict(date=date, nav=nav, peak=peak, cushion=cushion,
                         exposure=target, r_risky=r_risky,
                         headroom=(nav_after_cost - floor * peak) / max(target, 1e-9) / nav_after_cost
                         if target > 0 else np.inf))
        nav = nav_after_cost * (1 + target * r_risky + (1 - target) * r_safe)
        peak = max(peak, nav)
        held = target
    return pd.DataFrame(rows)


def summarise(state, multiplier):
    """A breach is a session where the risky leg fell further than the cushion allowed."""
    breach = state[state["r_risky"] < -state["headroom"]]
    worst = state["r_risky"].min()
    worst_day = state.loc[state["r_risky"].idxmin()]
    return dict(
        multiplier=multiplier,
        days=int(len(state)),
        floor_breaches=int(len(breach)),
        worst_risky_day=float(worst),
        worst_day_date=str(worst_day["date"].date()),
        exposure_that_day=float(worst_day["exposure"]),
        headroom_that_day=float(worst_day["headroom"]),
        margin=float(worst_day["headroom"] + worst),
        min_exposure=float(state["exposure"].min()),
        mean_exposure=float(state["exposure"].mean()),
        time_at_floor=float((state["exposure"] <= 0.01).mean()),
        min_cushion=float(state["cushion"].min()),
    )


def measure():
    payload = dict(status="ready", floor=FLOOR, band=BAND, cost=COST, windows={}, thresholds={})
    for name in WINDOWS:
        frame = build(name)
        payload["windows"][name] = {}
        for multiplier in MULTIPLIERS:
            state = walk(frame, multiplier)
            block = summarise(state, multiplier)
            risky = 0.5 * frame["ashare"] + 0.5 * frame["sp500"]
            block["risky_worst_day"] = float(risky.min())
            block["risky_worst_date"] = str(risky.idxmin().date())
            payload["windows"][name][multiplier] = block
        payload["thresholds"][name] = float(1.0 / min(MULTIPLIERS))
    payload["verdict"] = {
        name: dict(breaches=sum(payload["windows"][name][m]["floor_breaches"] for m in MULTIPLIERS),
                   tightest_margin=min(payload["windows"][name][m]["margin"] for m in MULTIPLIERS))
        for name in WINDOWS}
    return payload


def _markdown(payload):
    lines = ["# 跳空风险：楼层是真的还是运气", "",
             "楼层 %.2f、不交易带 %.0f%%、成本 %.0fbp。" % (
                 payload["floor"], payload["band"] * 100, payload["cost"] * 1e4), "",
             "CPPI 的击穿条件是**单日风险腿跌幅超过缓冲垫**。把风险暴露写成 `m × 缓冲垫 / 净值`，",
             "条件化简为 **单日跌幅 > 1/m**：倍数 5 → 20%、倍数 8 → 12.5%。", ""]
    for name, block in payload["windows"].items():
        worst = min(block[m]["risky_worst_day"] for m in block)
        lines += ["## %s" % name, "",
                  "风险腿单日最差 **%.2f%%**（%s）。" % (
                      worst * 100, block[MULTIPLIERS[0]]["risky_worst_date"]), "",
                  "| 倍数 | 1/m 阈值 | 击穿次数 | 最差单日 | 当日暴露 | 剩余余量 | 最低暴露 | 贴地板时间 |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for multiplier in MULTIPLIERS:
            row = block[multiplier]
            lines.append("| %d | %.1f%% | **%d** | %+.2f%% | %.0f%% | %+.2f%% | %.0f%% | %.1f%% |" % (
                multiplier, 100.0 / multiplier, row["floor_breaches"],
                row["risky_worst_day"] * 100, row["exposure_that_day"] * 100,
                row["margin"] * 100, row["min_exposure"] * 100, row["time_at_floor"] * 100))
        lines.append("")
    lines += ["## 结论", ""]
    total = sum(value["breaches"] for value in payload["verdict"].values())
    tightest = min(value["tightest_margin"] for value in payload["verdict"].values())
    lines.append("四段窗口、五个倍数共 20 组里，**击穿 %d 次**；最紧的一次余量 **%.2f%%**。" % (
        total, tightest * 100))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the gap risk of the CPPI floor.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "cppi_gap_risk.json", payload)
    (target / "cppi_gap_risk.md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
