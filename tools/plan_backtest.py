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


def price_and_payouts(frame, paid, split=SPLIT, payout=PAYOUT):
    """A split-repaired price series plus the cash each unit paid, kept separate.

    ``total_return`` reinvests distributions, which is right for measuring an index but wrong
    for a ledger: the account holds units and receives cash.  This is the same repair with the
    cash left as cash, so the simulator can mark a book the way the account does.
    """
    prices = frame.astype(float).copy()
    payouts = {}
    for date in list(prices.index)[1:]:
        prior = prices.index[prices.index < date]
        if not len(prior):
            continue
        before, now = float(prices.loc[prior[-1]]), float(prices.loc[date])
        move = now / before
        recorded = float(paid.get(date, 0.0))
        if recorded > 0:
            payouts[date] = recorded
        elif move <= 1 - split:
            prices.loc[prior] = prices.loc[prior] * move          # unit split
        elif move <= 1 - payout:
            payouts[date] = round(before - now, 4)                # interface lost the payout
    return prices, payouts


def simulate_book(prices, payouts, start, capital=CAPITAL, floor=FLOOR, multiplier=MULTIPLIER,
                  band=BAND, end=None, cadence="daily"):
    """The plan as a book: units, cash, distributions and a daily mark, like the live ledger.

    Returns the NAV series and every fill, so a backtest can be written into an account and
    read back with the same ``curve`` and ``report`` the real book uses.
    """
    window = prices.loc[pd.Timestamp(start):]
    if end:
        window = window.loc[:pd.Timestamp(end)]
    if len(window) < 30:
        return None
    units = {code: 0 for code in CODES}
    cash = float(capital)
    peak = float(capital)
    marks = set(month_starts(window.index)) if cadence == "monthly" else set(window.index)
    nav_rows, fills, fees = [], [], 0.0
    for day in window.index:
        row = window.loc[day]
        for code in CODES:
            if day in payouts.get(code, {}) and units[code] > 0:
                cash += units[code] * float(payouts[code][day])
        value = cash + sum(units[code] * float(row[code]) for code in CODES)
        peak = max(peak, value)
        if day in marks and value > 0:
            target = min(max(multiplier * (value - floor * peak) / value, 0.0), 1.0)
            held = sum(units[code] * float(row[code]) for code in RISKY) / value
            if abs(target - held) >= band:
                wanted = {code: target * plan_orders.LEGS[code] * value for code in RISKY}
                wanted[SAFE] = (1.0 - target) * value
                for side in ("sell", "buy"):
                    for code in CODES:
                        price = float(row[code])
                        goal = int(wanted[code] // price // LOT * LOT)
                        delta = goal - units[code]
                        if (side == "sell" and delta >= 0) or (side == "buy" and delta <= 0):
                            continue
                        amount, fee = abs(delta) * price, fee_for(abs(delta) * price)
                        if side == "sell":
                            cash += amount * (1 - SLIPPAGE) - fee
                        else:
                            if amount * (1 + SLIPPAGE) + fee > cash:
                                continue
                            cash -= amount * (1 + SLIPPAGE) + fee
                        fees += fee
                        units[code] += delta
                        fills.append(dict(date=str(day.date()), code=code, action=side,
                                          shares=abs(int(delta)), price=round(price, 4),
                                          fee=round(fee, 2)))
                value = cash + sum(units[code] * float(row[code]) for code in CODES)
        nav_rows.append((day, value, cash, dict(units), {code: float(row[code])
                                                         for code in CODES if units[code]}))
    series = pd.Series({day: value for day, value, _, _, _ in nav_rows}).sort_index()
    return dict(series=series, rows=nav_rows, fills=fills, fees=round(fees, 2),
                final=float(series.iloc[-1]))


def export_account(start, name="backtest", capital=CAPITAL, cadence="daily", end=None):
    """Write a simulated book into reports/account/<name> so curve/report read it like a real one.

    **Not reconciled yet.**  Exporting 2019-01-02 gives 1,687,443 where ``simulate`` on the
    total-return index gives 1,342,379 for the same start and the same rule -- a 34-point gap
    that cannot be explained by the dividend treatment (crediting them as cash accounts for
    about 173k of it).  One of the two simulators is wrong and until that is settled the
    export must not be presented as a track record.  The code is kept so the reconciliation
    has something to test against.
    """
    from tools import account_ledger
    prices, payouts, index = {}, {}, {}
    for code in CODES:
        frame, paid = raw_series(code, start, end or END)
        prices[code], payouts[code] = price_and_payouts(frame, paid)
    frame = pd.DataFrame(prices).sort_index().dropna()
    frame = frame.loc[frame.index >= pd.Timestamp(start)]
    result = simulate_book(frame, payouts, start, capital=capital, cadence=cadence, end=end)
    if result is None:
        raise ValueError("Window too short to simulate")
    base = account_ledger.ACCOUNT_DIR / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "account.json").write_text(json.dumps(dict(capital=capital, start_date=str(start)),
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    with (base / "trades.jsonl").open("w", encoding="utf-8") as stream:
        for fill in result["fills"]:
            row = dict(fill, recorded_at="backtest", source="backtest")
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (base / "snapshots.jsonl").open("w", encoding="utf-8") as stream:
        for day, value, cash, units, held_prices in result["rows"]:
            row = dict(date=str(day.date()), nav=round(value, 2), cash=round(cash, 2),
                       positions={code: count for code, count in units.items() if count},
                       prices=held_prices)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return dict(name=name, start=str(start), days=len(result["rows"]),
                fills=len(result["fills"]), fees=result["fees"], final=result["final"])


def universe_book(start=START, end=END, log=print):
    """Real traded prices plus the cash distributions, for a book that holds units.

    This is the honest price source for a 1,000,000 account: 511010 trades near 140, so one
    100-unit lot is 1.2% of NAV.  The normalised total-return index starts at 1.0, which makes
    a lot worth 100 yuan -- a fractional-share account in disguise.  Running the same rule on
    the two changes the seven-year outcome by 25%, entirely through when the band fires.
    """
    prices, payouts, repairs = {}, {}, {}
    for code in CODES:
        frame, paid = raw_series(code, start, end)
        prices[code], payouts[code] = price_and_payouts(frame, paid)
        if payouts[code]:
            repairs[code] = sorted(str(stamp.date()) for stamp in payouts[code])[:4]
    frame = pd.DataFrame(prices).sort_index().dropna()
    return frame.loc[frame.index >= pd.Timestamp(start)], payouts, repairs


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
                trades=int(trades), fees=round(fees, 2), series=series,
                vol=float(returns.std(ddof=1) * np.sqrt(242)) if len(returns) > 2 else None)


def sweep(prices, payouts, capital=CAPITAL, start=START, step_months=1, cadence="monthly"):
    """Run every monthly start on real prices, so the 100-unit lot is the real one."""
    starts = month_starts(prices.loc[pd.Timestamp(start):].index)
    rows = []
    for stamp in starts[::max(int(step_months), 1)]:
        result = simulate_book(prices, payouts, stamp, capital=capital, cadence=cadence)
        if result:
            rows.append(dict(start=str(pd.Timestamp(stamp).date()),
                             end=str(result["series"].index[-1].date()),
                             years=round(len(result["series"]) / 242.0, 2),
                             final=round(result["final"], 2),
                             annual=float((result["final"] / capital) **
                                          (242.0 / len(result["series"])) - 1),
                             drawdown=float((result["series"] /
                                             result["series"].cummax() - 1).min()),
                             trades=len(result["fills"]), fees=result["fees"]))
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
    prices, payouts, payout_dates = universe_book(args.start)
    rows = sweep(prices, payouts, capital=args.capital, start=args.start, cadence=args.cadence)
    summary = summarize(rows)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    payload = dict(status="ready", capital=args.capital, codes=CODES, cadence=args.cadence,
                   floor=FLOOR, multiplier=MULTIPLIER, band=BAND, payout_dates=payout_dates,
                   prices="traded prices with 100-unit lots",
                   window=dict(first=str(prices.index[0].date()), last=str(prices.index[-1].date())),
                   summary=summary, runs=rows)
    (target / f"plan_backtest_{args.cadence}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
