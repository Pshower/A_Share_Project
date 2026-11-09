"""How much of the result came from the window rather than the mechanism.

The strategy was first reported on four sub-periods and all four cleared the target.  Then a
window that began at the end of 2016 returned 4.35 rather than 7.21, which is a large enough
gap to make the four-period claim look like a selection rather than a result.  Four windows
chosen by hand cannot settle that; every start date can.

So the rule is run from every quarter end with at least three years of runway, always ending
at the end of 2024, and the resulting distribution of annualised returns is reported in full
rather than as a single figure.  What the mechanism controls is the drawdown, and that should
be stable across starts.  What it does not control is the equity return over the window, and
the spread of outcomes measures exactly how much that matters.

Two bond carries are shown.  Four point six percent is what the treasury index delivered when
yields were falling; one point six is what the ten-year now offers.  The difference between
the two columns is the part of the result that has already disappeared.
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
from tools.cppi_strategy import run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "cppi_start_dates"
START, END = "2005-01-04", "2024-12-31"
MULTIPLIER = 6
MIN_YEARS = 3
CARRIES = (0.046, 0.016)
EQUITY_YIELD, SP_YIELD = 0.021, 0.020


def panel():
    local = pd.DataFrame({"000300": returns_of("000300", START, END),
                          "000905": returns_of("000905", START, END)}).dropna()
    ashare = local.mean(axis=1) + EQUITY_YIELD / capm.TRADING_DAYS
    sp500 = gf.yuan_returns(gf.foreign_close("sp500", ".INX", "us"), gf.usdcny())
    sp500 = sp500 + SP_YIELD / capm.TRADING_DAYS
    frame = pd.DataFrame({"ashare": ashare, "sp500": sp500}).dropna()
    return frame.loc[START:END]


def starts(index, minimum=MIN_YEARS):
    """Quarter-end starts with at least ``minimum`` years of runway to the end of 2024."""
    cutoff = index[-1] - pd.DateOffset(years=minimum)
    quarters = pd.Series(index, index=index).groupby([index.year, index.quarter]).last()
    return [date for date in quarters if date <= cutoff]


def measure():
    frame = panel()
    dates = starts(frame.index)
    payload = dict(status="ready", multiplier=MULTIPLIER, min_years=MIN_YEARS,
                   sessions=int(len(frame)), starts=len(dates),
                   first=str(dates[0].date()), last=str(dates[-1].date()), results={})
    for carry in CARRIES:
        bond = pd.Series(carry / capm.TRADING_DAYS, index=frame.index)
        rows = []
        for date in dates:
            window = frame.loc[date:]
            sub = pd.DataFrame({"ashare": window["ashare"], "sp500": window["sp500"],
                                "bond": bond.loc[date:]})
            block = run(sub, MULTIPLIER)
            rows.append(dict(start=str(date.date()), years=float(len(sub) / capm.TRADING_DAYS),
                             annual=block["annual"], drawdown=block["drawdown"],
                             sharpe=block["sharpe"]))
        annual = np.array([row["annual"] for row in rows])
        payload["results"][carry] = dict(
            rows=rows,
            annual=dict(min=float(annual.min()), p10=float(np.percentile(annual, 10)),
                        median=float(np.median(annual)), p90=float(np.percentile(annual, 90)),
                        max=float(annual.max()), share_over_7=float((annual >= 0.07).mean())),
            drawdown=dict(min=float(min(row["drawdown"] for row in rows)),
                          median=float(np.median([row["drawdown"] for row in rows])),
                          share_within_10=float(np.mean([abs(row["drawdown"]) <= 0.10 for row in rows]))))
    return payload


def _markdown(payload):
    lines = ["# 每一个起点：机制控制的和它控制不了的", "",
             "倍数 %d，起点为每个季末、至少 %d 年跑道，终点固定 2024-12-31。**共 %d 个起点。**" % (
                 payload["multiplier"], payload["min_years"], payload["starts"]), "",
             "| 债券 carry | 最差 | p10 | **中位** | p90 | 最好 | ≥7% 的起点占比 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for carry, block in payload["results"].items():
        a = block["annual"]
        lines.append("| %.1f%% | %+.2f%% | %+.2f%% | **%+.2f%%** | %+.2f%% | %+.2f%% | **%.0f%%** |" % (
            carry * 100, a["min"] * 100, a["p10"] * 100, a["median"] * 100,
            a["p90"] * 100, a["max"] * 100, a["share_over_7"] * 100))
    lines += ["", "| 债券 carry | 回撤最差 | 回撤中位 | 回撤 ≤10% 的起点占比 |",
              "| ---: | ---: | ---: | ---: |"]
    for carry, block in payload["results"].items():
        d = block["drawdown"]
        lines.append("| %.1f%% | %.2f%% | %.2f%% | **%.0f%%** |" % (
            carry * 100, d["min"] * 100, d["median"] * 100, d["share_within_10"] * 100))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Every start date of the drawdown-capped rule.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "cppi_start_dates.json", payload)
    (target / "cppi_start_dates.md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
