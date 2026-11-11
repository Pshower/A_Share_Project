"""Run the actual final plan on the actual ETFs, from many possible start dates.

Every earlier CPPI result here used index proxies.  This uses the four products the plan names
-- 510300, 510500, 513500, 511010 -- at their traded prices, with the plan's own parameters
(floor 0.90, multiplier 6, no-trade band 5%) and the real commission floor and lot size.

**The Sina price history is not usable as it stands**, and that is the first thing this file
fixes.  Two of the four series contain unit splits recorded as zero-amount "distributions":
513500 halves on 2022-03-30, 510500 drops 12.7% on 2022-08-29 for a 0.92 payout the interface
reports as 0.000.  Left alone they read as losses and every drawdown below is inflated by
roughly the size of the split.  So each series is turned into a **total-return** index first:

* a move of 30% or more carrying no recorded payout is a split, and history before it is
  rescaled;
* a drop of 10% or more with no recorded payout is a distribution the interface lost, and the
  implied amount (previous close minus close) is added back;
* recorded payouts are used as given.

The result is a series that can be held as an index.  Cash dividends in the real account are
handled by the ledger, not here; the difference is that this reinvests them.

Why many start dates: CPPI exposure depends on the gap between NAV and the peak, so the day you
start decides the risk you carry for the first months.  One start proves nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import plan_orders
from tools.account_ledger import distributions, fee_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "plan_backtest"
CODES = ["510300", "510500", "513500", "511010"]
RISKY = ["510300", "510500", "513500"]
SAFE = "511010"
FLOOR, MULTIPLIER, BAND = plan_orders.FLOOR, plan_orders.MULTIPLIER, plan_orders.BAND
CAPITAL = 1_000_000.0
LOT = plan_orders.LOT
SLIPPAGE = 0.001
START, END = "2016-01-04", "2026-11-11"
SPLIT, PAYOUT = 0.30, 0.10


def raw_series(code, start=START, end=END):
    import akshare as ak
    frame = ak.fund_etf_hist_sina(symbol=plan_orders.prefix_for(code) + code)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()["close"].astype(float)
    frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
    paid = {pd.Timestamp(stamp): float(per_unit) for stamp, per_unit in distributions(code, start, end)}
    return frame, paid


def total_return(frame, paid, split=SPLIT, payout=PAYOUT):
    """Price series to a reinvested index, repairing splits and lost payouts."""
    prices = frame.copy()
    events = []
    for date in list(prices.index)[1:]:
        prior = prices.index[prices.index < date]
        if not len(prior):
            continue
        before, now = float(prices.loc[prior[-1]]), float(prices.loc[date])
        move = now / before
        recorded = float(paid.get(date, 0.0))
        if recorded == 0.0 and move <= 1 - split:
            prices.loc[prior] = prices.loc[prior] * move
            events.append(dict(date=str(date.date()), kind="split", factor=round(move, 4)))
        elif recorded == 0.0 and move <= 1 - payout:
            events.append(dict(date=str(date.date()), kind="payout",
                               amount=round(before - now, 4)))
    index = pd.Series(1.0, index=prices.index)
    for position in range(1, len(prices.index)):
        date, previous = prices.index[position], prices.index[position - 1]
        now, before = float(prices.loc[date]), float(prices.loc[previous])
        cash = float(paid.get(date, 0.0))
        if cash == 0.0 and (now / before - 1.0) <= -payout:
            cash = before - now
        index.iloc[position] = index.iloc[position - 1] * (now + cash) / before
    return index, events


def universe(start=START, end=END, log=print):
    index, repairs = {}, {}
    for code in CODES:
        frame, paid = raw_series(code, start, end)
        index[code], repairs[code] = total_return(frame, paid)
        if repairs[code]:
            log("%s repaired: %s" % (code, repairs[code]))
    frame = pd.DataFrame(index).sort_index().dropna()
    return frame.loc[frame.index >= pd.Timestamp(start)], repairs


def month_starts(index):
    grouped = pd.Series(index, index=index).groupby([index.year, index.month]).min()
    return [pd.Timestamp(value) for value in grouped.values]


def simulate(index, start, capital=CAPITAL, floor=FLOOR, multiplier=MULTIPLIER,
             band=BAND, end=None, cadence="monthly"):
    """CPPI on a reinvested index; ``cadence`` is how often the cushion is checked."""
    window = index.loc[pd.Timestamp(start):]
    if end:
        window = window.loc[:pd.Timestamp(end)]
    if len(window) < 30:
        return None
    units = {code: 0 for code in CODES}
    cash = float(capital)
    peak = float(capital)
    equity, trades, fees = [], 0, 0.0
    marks = set(month_starts(window.index)) if cadence == "monthly" else set(window.index)
    for day in window.index:
        row = window.loc[day]
        value = cash + sum(units[code] * float(row[code]) for code in CODES)
        peak = max(peak, value)
        if day in marks and value > 0:
            target = min(max(multiplier * (value - floor * peak) / value, 0.0), 1.0)
            held = sum(units[code] * float(row[code]) for code in RISKY) / value
            if abs(target - held) >= band:
                wanted = {code: target * plan_orders.LEGS[code] * value for code in RISKY}
                wanted[SAFE] = (1.0 - target) * value
                for code in CODES:                       # sells fund the buys
                    price = float(row[code])
                    goal = int(wanted[code] // price // LOT * LOT)
                    delta = goal - units[code]
                    if delta < 0:
                        amount = -delta * price
                        fee = fee_for(amount)
                        cash += amount * (1 - SLIPPAGE) - fee
                        fees += fee
                        units[code] += delta
                        trades += 1
                for code in CODES:
                    price = float(row[code])
                    goal = int(wanted[code] // price // LOT * LOT)
                    delta = goal - units[code]
                    if delta > 0:
                        amount = delta * price
                        fee = fee_for(amount)
                        if amount * (1 + SLIPPAGE) + fee <= cash:
                            cash -= amount * (1 + SLIPPAGE) + fee
                            fees += fee
                            units[code] += delta
                            trades += 1
                value = cash + sum(units[code] * float(row[code]) for code in CODES)
        equity.append((day, value))
    series = pd.Series(dict(equity)).sort_index()
    returns = series.pct_change().dropna()
    years = len(series) / 242.0
    final = float(series.iloc[-1])
    return dict(start=str(pd.Timestamp(start).date()), end=str(series.index[-1].date()),
                years=round(years, 2), final=round(final, 2),
                annual=float((final / capital) ** (1 / years) - 1) if years > 0 else float("nan"),
                drawdown=float((series / series.cummax() - 1).min()),
                trades=int(trades), fees=round(fees, 2),
                vol=float(returns.std(ddof=1) * np.sqrt(242)) if len(returns) > 2 else None)


def sweep(index, capital=CAPITAL, start=START, step_months=1, cadence="monthly"):
    starts = month_starts(index.loc[pd.Timestamp(start):].index)
    rows = []
    for stamp in starts[::max(int(step_months), 1)]:
        result = simulate(index, stamp, capital=capital, cadence=cadence)
        if result:
            rows.append(result)
    return rows


def summarize(rows, target_return=0.07, drawdown_cap=0.10):
    if not rows:
        return dict(status="no_runs")
    annual = np.array([row["annual"] for row in rows], dtype=float)
    draw = np.array([row["drawdown"] for row in rows], dtype=float)
    return dict(status="ready", runs=len(rows), first=rows[0]["start"], last=rows[-1]["start"],
                annual_median=float(np.median(annual)), annual_min=float(annual.min()),
                annual_max=float(annual.max()), drawdown_median=float(np.median(draw)),
                drawdown_worst=float(draw.min()),
                share_under_cap=float((draw >= -drawdown_cap).mean()),
                share_over_target=float((annual >= target_return).mean()),
                share_both=float(((draw >= -drawdown_cap) & (annual >= target_return)).mean()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="The final plan on the real ETFs, many start dates.")
    parser.add_argument("--capital", type=float, default=CAPITAL)
    parser.add_argument("--start", default=START)
    parser.add_argument("--cadence", default="monthly", choices=["monthly", "daily"])
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    index, repairs = universe(args.start)
    rows = sweep(index, capital=args.capital, start=args.start, cadence=args.cadence)
    summary = summarize(rows)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    payload = dict(status="ready", capital=args.capital, codes=CODES, cadence=args.cadence,
                   floor=FLOOR, multiplier=MULTIPLIER, band=BAND, repairs=repairs,
                   window=dict(first=str(index.index[0].date()), last=str(index.index[-1].date())),
                   summary=summary, runs=rows)
    (target / f"plan_backtest_{args.cadence}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
