"""Backtest the frozen E1 reversal book at different top-K sizes.

The shadow recorder holds a fixed 30-name book; the question here is what the same rule would have
earned holding only the best 5, 10 or 20 of them.  That is a **concentration diagnostic on the spent
sample**, not a search for a better K: the panel ends 2024-12-31, the P window (2026-07 onward) is
never read, and no K reported here is adopted by the live rule.

Generality the first version lacked, and why each piece is now explicit:

* **Layout.** Datasets in this repo are not shaped alike -- ``research_v3`` and the forward panels
  carry ``(code, feature)`` frames, ``feature_top1800`` ships *flat* price columns (close only), and
  ``e2_dynamic1800`` has unnamed levels.  ``load`` normalises all three and raises a clear error on
  anything else, instead of failing somewhere inside ``xs``.
* **Window.** Presets for V and the design window, but ``--start/--end`` overrides them, which is
  what a 2025-2026 forward panel needs.
* **Costs and sizing.** Capital, lot, commission floor, stamp duty, slippage and the buffer multiple
  are arguments, not constants: the answer moves by more than the K values do.
* **One rule.** The score and the buffer rule come from ``tools.e1_rule``, the same module the
  recorder uses, so a backtest cannot quietly test a different rule than the one being recorded.
* **Audit.** Every run writes its holdings per rebalance date; a NAV curve alone cannot say whether
  a K was holding thirty names or the same five all year.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import e1_rule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "clean" / "research_v3"
WINDOWS = {"v": ("2019-01-01", "2024-12-31"), "full": ("2017-01-03", "2024-12-31")}
HORIZONS = (5, 10, 20, 30)


class Costs:
    """The fee schedule a trade pays, as one object so it can be printed and stored with a run."""

    def __init__(self, commission=0.0003, minimum=5.0, sell_tax=0.0005, slippage=0.0005,
                 lot=100, capital=1_000_000.0):
        self.commission, self.minimum, self.sell_tax = commission, minimum, sell_tax
        self.slippage, self.lot, self.capital = slippage, int(lot), float(capital)

    def commission_for(self, amount):
        return max(self.minimum, amount * self.commission)

    def as_dict(self):
        return dict(commission=self.commission, minimum=self.minimum, sell_tax=self.sell_tax,
                    slippage=self.slippage, lot=self.lot, capital=self.capital)


def _normalise_panel(panel):
    """(code, feature) with named levels, whatever the file called them."""
    if not isinstance(panel.columns, pd.MultiIndex) or panel.columns.nlevels != 2:
        raise ValueError("panel must have a two-level (code, feature) column index; got %r"
                         % (panel.columns.nlevels if isinstance(panel.columns, pd.MultiIndex) else 1))
    codes = [str(value) for value in panel.columns.get_level_values(0)]
    features = [str(value) for value in panel.columns.get_level_values(1)]
    if any(code in features for code in codes[:1]):
        raise ValueError("panel levels look swapped: level 0 should be the code")
    panel = panel.copy()
    panel.columns = pd.MultiIndex.from_arrays([codes, features], names=["code", "feature"])
    return panel


def _normalise_prices(prices, codes):
    """(open, close) frames from either a (code, field) frame or a flat close-only frame."""
    if isinstance(prices.columns, pd.MultiIndex) and prices.columns.nlevels == 2:
        frame = prices.copy()
        level = [str(value) for value in frame.columns.get_level_values(1)]
        if not set(level) & {"open", "close"}:
            raise ValueError("price frame has neither an 'open' nor a 'close' field")
        code_level = [str(value) for value in frame.columns.get_level_values(0)]
        frame.columns = pd.MultiIndex.from_arrays([code_level, level], names=["code", "feature"])
        close = frame.xs("close", axis=1, level=1) if "close" in set(level) else None
        open_ = frame.xs("open", axis=1, level=1) if "open" in set(level) else close
        if close is None:
            raise ValueError("price frame has no close")
        return open_.reindex(columns=codes), close.reindex(columns=codes)
    # Flat frame: one level of codes, treated as the close -- trades then fill at the close.
    flat = prices.copy()
    flat.columns = [str(value) for value in flat.columns]
    missing = [code for code in codes if code not in flat.columns]
    if missing and len(missing) > len(codes) // 2:
        raise ValueError("flat price frame does not look like codes: missing %d of %d"
                         % (len(missing), len(codes)))
    close = flat.reindex(columns=codes)
    return close.copy(), close


def load(dataset=DATASET):
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    panel = _normalise_panel(pd.read_parquet(dataset / "panel_data.parquet"))
    prices = pd.read_parquet(dataset / "prices.parquet")
    codes = [str(code) for code in manifest["stock_codes"]]
    features = [str(name) for name in manifest.get("feature_cols") or
                panel.columns.get_level_values(1).unique()]
    open_, close = _normalise_prices(prices, codes)
    return dict(dataset=dataset.name, panel=panel, open=open_, close=close, codes=codes,
                features=features)


def backtest(panel, score, open_, close, top_k, start, end, costs=None, interval=1, buffer=None):
    """Rebalance the frozen rule every ``interval`` sessions, priced at the next open.

    ``interval=1`` is what the shadow recorder does (it re-decides daily and lets names fall out of
    the buffer); ``interval=20`` is what E1's engine did (a decision every twenty sessions).  Both
    are reported because the difference is bigger than the difference between any two K values.
    """
    costs = costs or Costs()
    buffer = 2 * top_k if buffer is None else int(buffer)
    days = [day for day in close.index if pd.Timestamp(start) <= day <= pd.Timestamp(end)]
    if len(days) < 30:
        raise ValueError("window too short: %d sessions between %s and %s" % (len(days), start, end))
    held, units, cash = [], {}, cost_capital(costs)
    last_close, carried = {}, 0
    nav_rows, trades, holdings = [], [], []
    for index, day in enumerate(days[:-1]):
        row = score.loc[day] if day in score.index else None
        if row is None or row.dropna().empty:
            continue
        wanted = e1_rule.decide(row, held, top_k=top_k, buffer=buffer) if index % max(int(interval), 1) == 0 else list(held)
        next_day = days[index + 1]
        prices = open_.loc[next_day].reindex(close.columns)
        for code in list(units):
            if code not in wanted:
                price = float(prices.get(code, np.nan))
                if not np.isfinite(price):
                    continue
                shares = units[code]
                amount = shares * price * (1 - costs.slippage)
                fee = costs.commission_for(amount) + amount * costs.sell_tax
                cash += amount - fee
                trades.append(dict(date=str(next_day.date()), code=code, action="sell",
                                   shares=shares, price=price * (1 - costs.slippage),
                                   fee=round(fee, 2)))
                units.pop(code)
                held.remove(code)
        free = [code for code in wanted if code not in units]
        if free:
            budget = cash / len(free)
            for code in free:
                price = float(prices.get(code, np.nan))
                if not np.isfinite(price) or price <= 0:
                    continue
                shares = int(budget // (price * (1 + costs.slippage)) // costs.lot * costs.lot)
                if shares <= 0:
                    continue
                amount = shares * price * (1 + costs.slippage)
                fee = costs.commission_for(amount)
                if amount + fee > cash:
                    shares -= costs.lot
                    if shares <= 0:
                        continue
                    amount = shares * price * (1 + costs.slippage)
                    fee = costs.commission_for(amount)
                cash -= amount + fee
                units[code] = units.get(code, 0) + shares
                held.append(code)
                trades.append(dict(date=str(next_day.date()), code=code, action="buy",
                                   shares=shares, price=price * (1 + costs.slippage),
                                   fee=round(fee, 2)))
        # A suspended name has no close today but is still owned: marking it at zero (or letting a
        # NaN through, which is what ``nan or 0.0`` does -- NaN is truthy) would print a fake crash.
        for code, price in close.loc[next_day].items():
            if np.isfinite(price):
                last_close[code] = float(price)
        missing_mark = [code for code in units if code not in last_close]
        if missing_mark:
            raise ValueError("no price has ever been seen for %s" % ", ".join(sorted(missing_mark)))
        carried += sum(1 for code in units if not np.isfinite(close.loc[next_day].get(code, np.nan)))
        value = cash + sum(units[code] * last_close[code] for code in units)
        nav_rows.append((next_day, value, cash, len(units)))
        holdings.append(dict(date=str(next_day.date()), names=" ".join(sorted(held))))
    series = pd.Series({day: value for day, value, _, _ in nav_rows}).sort_index()
    return dict(top_k=top_k, interval=int(interval), buffer=int(buffer), series=series,
                trades=trades, rows=nav_rows, holdings=holdings)


def cost_capital(costs):
    return costs.capital


def benchmark(close, start, end, costs=None):
    """Equal weight of the whole pool, bought once and held: the reference E1 was judged against.

    Deliberately costless -- it is the optimistic reference, so any strategy that needs help from
    the benchmark's fees to win has not won.
    """
    costs = costs or Costs()
    days = [day for day in close.index if pd.Timestamp(start) <= day <= pd.Timestamp(end)]
    prices = close.loc[days[0]]
    usable = [code for code in close.columns if np.isfinite(prices.get(code, np.nan))]
    series = close.loc[days, usable].div(close.loc[days[0], usable]).mean(axis=1) * costs.capital
    return series, len(usable)


def metrics(series, trades, days_per_year=242):
    if len(series) < 2:
        return {}
    returns = series.pct_change().dropna()
    years = len(series) / days_per_year
    total = float(series.iloc[-1] / series.iloc[0] - 1)
    drawdown = float((series / series.cummax() - 1).min())
    volatility = float(returns.std() * np.sqrt(days_per_year))
    traded = sum(float(row["shares"]) * float(row["price"]) for row in trades)
    return dict(total=total, annualised=float((1 + total) ** (1 / years) - 1) if years > 0 else None,
                volatility=volatility,
                sharpe=float(returns.mean() / returns.std() * np.sqrt(days_per_year)) if returns.std() else None,
                worst_drawdown=drawdown, turnover_per_year=float(traded / series.mean() / years),
                trades=len(trades),
                fees=round(sum(float(row["fee"]) for row in trades), 2),
                days=int(len(series)))


def run(window="v", horizons=HORIZONS, dataset=DATASET, output=None, interval=1,
        start=None, end=None, costs=None, buffer_mult=2, log=print):
    costs = costs or Costs()
    if start is None or end is None:
        start, end = WINDOWS[window] if window in WINDOWS else (start, end)
    data = load(dataset)
    score = e1_rule.composite_frame(data["panel"], data["codes"], data["features"])
    equal, named = benchmark(data["close"], start, end, costs)
    result = dict(window=window, interval=int(interval), start=start, end=end,
                  dataset=data["dataset"], codes=named, costs=costs.as_dict(),
                  features=len(data["features"]), benchmark=metrics(equal, []), runs=[])
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        equal.rename("nav").to_frame().to_csv(Path(output) / ("benchmark_%s.csv" % window))
    for top_k in horizons:
        book = backtest(data["panel"], score, data["open"], data["close"], top_k, start, end,
                        costs=costs, interval=interval,
                        buffer=buffer_mult * top_k if buffer_mult else None)
        row = dict(top_k=top_k, interval=int(interval), **metrics(book["series"], book["trades"]))
        row["excess_vs_benchmark"] = (row.get("annualised") or 0) - (result["benchmark"].get("annualised") or 0)
        result["runs"].append(row)
        if output:
            tag = "%s_k%d_i%d" % (window, top_k, interval)
            book["series"].rename("nav").to_frame().to_csv(Path(output) / ("nav_%s.csv" % tag))
            pd.DataFrame(book["holdings"]).to_csv(Path(output) / ("holdings_%s.csv" % tag), index=False)
            pd.DataFrame(book["trades"]).to_csv(Path(output) / ("trades_%s.csv" % tag), index=False)
        log("k=%2d 年化 %6.2f%%  回撤 %6.2f%%  换手 %.1fx  费用 %8.0f"
            % (top_k, 100 * (row["annualised"] or 0), 100 * row["worst_drawdown"],
               row["turnover_per_year"], row["fees"]))
    if output:
        (Path(output) / ("summary_%s_i%d.json" % (window, interval))).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Top-K sweep of the frozen E1 book.")
    parser.add_argument("--window", default="v", help="Preset name, or anything with --start/--end.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--topk", default="5,10,20,30")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--interval", type=int, default=1,
                        help="Sessions between decisions: 1 = shadow, 20 = E1 engine.")
    parser.add_argument("--buffer-mult", type=float, default=2.0,
                        help="Exit below rank buffer_mult*K; 0 keeps the rule's own default.")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--lot", type=int, default=100)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--sell-tax", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    horizons = tuple(int(part) for part in args.topk.split(",") if part.strip())
    costs = Costs(args.commission, args.min_commission, args.sell_tax, args.slippage,
                  args.lot, args.capital)
    window = args.window if (args.start is None or args.end is None) else "custom"
    result = run(window, horizons, args.dataset, args.output, interval=args.interval,
                 start=args.start, end=args.end, costs=costs, buffer_mult=args.buffer_mult)
    print(json.dumps(dict(window=result["window"], interval=result["interval"], start=result["start"],
                          end=result["end"], costs=result["costs"],
                          benchmark=result["benchmark"], runs=result["runs"]),
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
