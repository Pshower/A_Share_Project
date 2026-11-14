"""Two questions the plan's parameters were never asked: how much capital it needs, and how wide
the no-trade band should be.

The plan ships with a 90% floor, a multiplier of 6 and a 5% no-trade band, validated on 129 monthly
start dates at 1,000,000 yuan on real ETF prices with 100-unit lots.  Two knobs never moved:

* **Capital.** One lot of 511010 is about 14,100 yuan, so at small capital the safe leg cannot be
  bought in the size the rule asks for.  The lot-granularity experiment already showed this
  matters (the seven-year result moved 25% when lots were removed), but the plan has only ever
  been run at 1,000,000.
* **Band.** 5% was chosen once, against 0%, and never swept.  It is also the only number in the
  daily gate a person could legitimately want to change.

Both sweeps reuse ``plan_backtest.simulate_book`` unchanged -- same engine, same prices, same fee
model -- so the numbers stay comparable with the published +5.3% median.  Neither sweep adopts a
parameter: the output is a table and a recommendation, and changing the plan is a separate decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import plan_backtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPITALS = (300_000.0, 500_000.0, 1_000_000.0, 2_000_000.0, 3_000_000.0, 5_000_000.0)
BANDS = (0.00, 0.03, 0.05, 0.08, 0.10, 0.15)
TARGET_RETURN = 0.07
DRAWDOWN_CAP = 0.10


def series_metrics(series, capital, fills, fees):
    years = len(series) / 242.0
    final = float(series.iloc[-1])
    return dict(final=round(final, 2),
                annual=float((final / capital) ** (1 / years) - 1) if years > 0 else float("nan"),
                drawdown=float((series / series.cummax() - 1).min()),
                trades=int(len(fills)), fees=round(float(fees), 2),
                years=round(years, 2))


def run_config(prices, payouts, start, capital, band, cadence="daily", step_months=1):
    """Every monthly start for one (capital, band) cell."""
    starts = plan_backtest.month_starts(prices.loc[pd.Timestamp(start):].index)
    rows = []
    for stamp in starts[::max(int(step_months), 1)]:
        result = plan_backtest.simulate_book(prices, payouts, stamp, capital=capital, band=band,
                                             cadence=cadence)
        if not result:
            continue
        rows.append(series_metrics(result["series"], capital, result["fills"], result["fees"]))
    return rows


def summarize(rows, target=TARGET_RETURN, cap=DRAWDOWN_CAP):
    if not rows:
        return dict(runs=0)
    annual = np.array([row["annual"] for row in rows], dtype=float)
    draw = np.array([row["drawdown"] for row in rows], dtype=float)
    fees = np.array([row["fees"] for row in rows], dtype=float)
    trades = np.array([row["trades"] for row in rows], dtype=float)
    return dict(runs=len(rows), annual_median=float(np.median(annual)),
                annual_min=float(annual.min()), annual_max=float(annual.max()),
                drawdown_median=float(np.median(draw)), drawdown_worst=float(draw.min()),
                share_under_cap=float((draw >= -cap).mean()),
                share_over_target=float((annual >= target).mean()),
                share_both=float(((draw >= -cap) & (annual >= target)).mean()),
                fees_median=float(np.median(fees)), trades_median=float(np.median(trades)))


def ladder(prices, payouts, start, capitals=CAPITALS, band=None, cadence="daily", log=print):
    """Capital sweep: the same rule at six account sizes."""
    band = plan_backtest.BAND if band is None else band
    out = []
    for capital in capitals:
        summary = summarize(run_config(prices, payouts, start, capital, band, cadence))
        summary.update(capital=capital, band=band)
        out.append(summary)
        log("本金 %9.0f 年化中位 %6.2f%% 最差回撤 %6.2f%% 达标率 %4.0f%% 费用中位 %8.0f"
            % (capital, 100 * summary["annual_median"], 100 * summary["drawdown_worst"],
               100 * summary["share_both"], summary["fees_median"]))
    return out


def band_sweep(prices, payouts, start, bands=BANDS, capital=plan_backtest.CAPITAL,
               cadence="daily", log=print):
    """Band sweep at the plan's own capital."""
    out = []
    for band in bands:
        summary = summarize(run_config(prices, payouts, start, capital, band, cadence))
        summary.update(capital=capital, band=band)
        out.append(summary)
        log("带宽 %4.0f%%  年化中位 %6.2f%% 最差回撤 %6.2f%% 成交中位 %5.0f 笔 费用中位 %8.0f"
            % (100 * band, 100 * summary["annual_median"], 100 * summary["drawdown_worst"],
               summary["trades_median"], summary["fees_median"]))
    return out


def run(start=None, output=None, cadence="daily", log=print):
    start = start or str(plan_backtest.START)
    prices, payouts, payout_dates = plan_backtest.universe_book(start)
    payload = dict(start=start, cadence=cadence, codes=plan_backtest.CODES,
                   floor=plan_backtest.FLOOR, multiplier=plan_backtest.MULTIPLIER,
                   payout_dates=payout_dates,
                   window=dict(first=str(prices.index[0].date()), last=str(prices.index[-1].date())),
                   ladder=ladder(prices, payouts, start, cadence=cadence, log=log),
                   bands=band_sweep(prices, payouts, start, cadence=cadence, log=log))
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        (path / "plan_sensitivity.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        log("wrote %s" % (path / "plan_sensitivity.json"))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capital ladder and band sweep for the plan.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--cadence", default="daily", choices=["daily", "monthly"])
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "runs" / "plan_sensitivity"))
    args = parser.parse_args(argv)
    payload = run(args.start, args.output, args.cadence)
    print(json.dumps(dict(ladder=payload["ladder"], bands=payload["bands"]),
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
