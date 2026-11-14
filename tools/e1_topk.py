"""Backtest the frozen E1 reversal book at different top-K sizes.

The shadow recorder holds a fixed 30-name book; the question here is what the same rule would have
earned holding only the best 5, 10 or 20 of them.  That is a **concentration diagnostic on the spent
sample**, not a search for a better K: the panel ends 2024-12-31, the P window (2026-07 onward) is
never read, and no K reported here is adopted by the live rule.  The K=30 row is the frozen
construction, so it doubles as a check that this engine agrees with what E1 published.

Construction, unchanged from ``shadow_run``:

* score = mean cross-sectional percentile of the twenty features (all negated, so lower is better);
* keep whatever still ranks below ``2 * K``; fill the remaining slots from the top of the ranking;
* decide on the close of day t, trade at the open of t+1, in 100-share lots;
* commission 0.03% (5 yuan floor), 0.05% stamp duty on sells, 0.05% slippage each way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "clean" / "research_v3"
LOT = 100
CAPITAL = 1_000_000.0
COMMISSION = 0.0003
MIN_COMMISSION = 5.0
SELL_TAX = 0.0005
SLIPPAGE = 0.0005
HORIZONS = (5, 10, 20, 30)
WINDOWS = {"v": ("2019-01-01", "2024-12-31"), "full": ("2017-01-03", "2024-12-31")}


def load(dataset=DATASET):
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(dataset / "panel_data.parquet")
    prices = pd.read_parquet(dataset / "prices.parquet")
    return manifest, panel, prices


def opens_and_closes(prices, codes):
    """Split the price frame into the two fields the rule needs, in the panel's column order."""
    levels = prices.columns.get_level_values(1).unique() if isinstance(prices.columns, pd.MultiIndex) else []
    fields = {}
    for field in ("open", "close"):
        if field in levels:
            frame = prices.xs(field, axis=1, level=1)
        else:
            frame = prices[field] if field in prices.columns else None
        if frame is None:
            raise ValueError("price frame has no %r field" % field)
        fields[field] = frame.reindex(columns=codes)
    return fields["open"], fields["close"]


def scores(panel, codes, features):
    """Mean cross-sectional percentile of the negated features; lower is a stronger buy."""
    ranks = []
    for feature in features:
        block = panel.xs(feature, axis=1, level=1).reindex(columns=codes)
        ranks.append(block.rank(axis=1, pct=True))
    stacked = np.stack([frame.to_numpy() for frame in ranks], axis=0)
    usable = np.isfinite(stacked).sum(axis=0)
    total = np.nansum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(usable > 0, total / np.maximum(usable, 1), np.nan)
    return pd.DataFrame(mean, index=panel.index, columns=codes)


def decide(score_row, held, top_k, buffer=None):
    """Keep what still ranks inside the buffer, then fill the rest from the top."""
    buffer = buffer or 2 * top_k
    order = score_row.dropna().sort_values()
    rank_of = {code: position for position, code in enumerate(order.index)}
    keep = [code for code in held if rank_of.get(code, 10 ** 9) < buffer]
    keep = sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))[:top_k]
    for code in order.index:
        if len(keep) >= top_k:
            break
        if code not in keep:
            keep.append(code)
    return keep


def fee_for(amount):
    return max(MIN_COMMISSION, amount * COMMISSION)


def backtest(panel, score, open_, close, top_k, start, end, capital=CAPITAL, interval=1):
    """Rebalance the frozen rule every ``interval`` sessions, priced at the next open.

    ``interval=1`` is what the shadow recorder does (it re-decides daily and lets names fall out of
    the buffer); ``interval=20`` is what E1's engine did (a decision every twenty sessions).  Both
    are reported because the difference is bigger than the difference between any two K values.
    """
    days = [day for day in close.index if pd.Timestamp(start) <= day <= pd.Timestamp(end)]
    if len(days) < 30:
        raise ValueError("window too short")
    held, units, cash = [], {}, float(capital)
    nav_rows, trades = [], []
    for index, day in enumerate(days[:-1]):
        row = score.loc[day] if day in score.index else None
        if row is None or row.dropna().empty:
            continue
        wanted = decide(row, held, top_k) if index % max(int(interval), 1) == 0 else list(held)
        next_day = days[index + 1]
        prices = open_.loc[next_day].reindex(close.columns)
        for code in list(units):
            if code not in wanted:
                price = float(prices.get(code, np.nan))
                if not np.isfinite(price):
                    continue
                shares = units[code]
                amount = shares * price * (1 - SLIPPAGE)
                fee = fee_for(amount) + amount * SELL_TAX
                cash += amount - fee
                trades.append(dict(date=str(next_day.date()), code=code, action="sell",
                                   shares=shares, price=price * (1 - SLIPPAGE), fee=round(fee, 2)))
                units.pop(code)
                held.remove(code)
        free = [code for code in wanted if code not in units]
        if free:
            budget = cash / len(free)
            for code in free:
                price = float(prices.get(code, np.nan))
                if not np.isfinite(price) or price <= 0:
                    continue
                shares = int(budget // (price * (1 + SLIPPAGE)) // LOT * LOT)
                if shares <= 0:
                    continue
                amount = shares * price * (1 + SLIPPAGE)
                fee = fee_for(amount)
                if amount + fee > cash:
                    shares -= LOT
                    if shares <= 0:
                        continue
                    amount = shares * price * (1 + SLIPPAGE)
                    fee = fee_for(amount)
                cash -= amount + fee
                units[code] = units.get(code, 0) + shares
                held.append(code)
                trades.append(dict(date=str(next_day.date()), code=code, action="buy",
                                   shares=shares, price=price * (1 + SLIPPAGE), fee=round(fee, 2)))
        value = cash + sum(units[code] * float(close.loc[next_day].get(code, np.nan) or 0.0)
                           for code in units)
        nav_rows.append((next_day, value, cash, len(units)))
    series = pd.Series({day: value for day, value, _, _ in nav_rows}).sort_index()
    return dict(top_k=top_k, series=series, trades=trades, rows=nav_rows)


def benchmark(close, start, end, capital=CAPITAL):
    """Equal weight of the whole pool, bought once and held: the reference E1 was judged against."""
    days = [day for day in close.index if pd.Timestamp(start) <= day <= pd.Timestamp(end)]
    prices = close.loc[days[0]]
    usable = [code for code in close.columns if np.isfinite(prices.get(code, np.nan))]
    series = close.loc[days, usable].div(close.loc[days[0], usable]).mean(axis=1) * capital
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


def run(window="v", horizons=HORIZONS, dataset=DATASET, output=None, interval=1, log=print):
    start, end = WINDOWS[window]
    manifest, panel, prices = load(dataset)
    codes = manifest["stock_codes"]
    open_, close = opens_and_closes(prices, codes)
    score = scores(panel, list(close.columns), list(panel.columns.get_level_values(1).unique()))
    equal, named = benchmark(close, start, end)
    result = dict(window=window, interval=int(interval), start=start, end=end, dataset=str(Path(dataset).name),
                  codes=named, benchmark=metrics(equal, []), runs=[])
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        equal.rename("nav").to_frame().to_csv(Path(output) / ("benchmark_%s.csv" % window))
    for top_k in horizons:
        book = backtest(panel, score, open_, close, top_k, start, end, interval=interval)
        row = dict(top_k=top_k, interval=int(interval), **metrics(book["series"], book["trades"]))
        row["excess_vs_benchmark"] = (row.get("annualised") or 0) - (result["benchmark"].get("annualised") or 0)
        result["runs"].append(row)
        if output:
            path = Path(output) / ("nav_%s_k%d.csv" % (window, top_k))
            book["series"].rename("nav").to_frame().to_csv(path)
        log("k=%2d 年化 %6.2f%%  回撤 %6.2f%%  换手 %.1fx  费用 %8.0f"
            % (top_k, 100 * (row["annualised"] or 0), 100 * row["worst_drawdown"],
               row["turnover_per_year"], row["fees"]))
    if output:
        (Path(output) / ("summary_%s_i%d.json" % (window, interval))).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Top-K sweep of the frozen E1 book.")
    parser.add_argument("--window", default="v", choices=sorted(WINDOWS))
    parser.add_argument("--topk", default="5,10,20,30")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--interval", type=int, default=1,
                        help="Sessions between decisions: 1 = shadow, 20 = E1 engine.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    horizons = tuple(int(part) for part in args.topk.split(",") if part.strip())
    result = run(args.window, horizons, args.dataset, args.output, interval=args.interval)
    print(json.dumps(dict(window=result["window"], interval=result["interval"], benchmark=result["benchmark"], runs=result["runs"]),
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
