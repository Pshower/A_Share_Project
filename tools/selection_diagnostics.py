"""Benchmarks that can actually be bought, and the attribution of what a strategy picked.

Two different questions keep getting answered with the same object, which is how a
look-ahead-selected equal-weight book ended up standing in for a deployable beta base.

**In-pool attribution.** When the question is "did the selection add anything", the
benchmark has to share the pool, the rebalance dates, the capital and the 100-share lot.
Otherwise a strategy that quietly tilts to small caps is credited with the tilt.

**Deployable beta.** When the question is "what would I hold instead", the benchmark has
to be something purchasable: a published index, or a lot-feasible basket of a size a
retail account can actually hold. A 4,200-name equal-weight book is neither.

This module produces both, plus the null the selection has to beat: random n-name baskets
at the same cadence and capital.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import baseline_capm
from tools import capm_backtest as capm
from tools.full_market_panel import load

DEFAULT_PANEL = capm.PROJECT_ROOT / "data" / "clean" / "panel_full"
COMMISSION_RATE = 0.0003
MINIMUM_COMMISSION = 5.0
STAMP_DUTY = 0.0005
SLIPPAGE = 0.001
LOT = 100


def buy_cost(notional):
    """Commission with the five-yuan floor, plus slippage. A-share buy side."""
    if notional <= 0:
        return 0.0
    return max(notional * COMMISSION_RATE, MINIMUM_COMMISSION) + notional * SLIPPAGE


def sell_cost(notional):
    """Commission with the floor, stamp duty and slippage. A-share sell side."""
    if notional <= 0:
        return 0.0
    return (max(notional * COMMISSION_RATE, MINIMUM_COMMISSION)
            + notional * STAMP_DUTY + notional * SLIPPAGE)


def affordable_shares(budget, price, lot=LOT):
    """Whole lots only, and never more than the budget allows once costs are paid."""
    if not np.isfinite(price) or price <= 0 or budget <= 0:
        return 0
    lots = int(budget // (price * lot))
    while lots > 0 and buy_cost(lots * lot * price) + lots * lot * price > budget:
        lots -= 1
    return lots * lot


def simulate(books, prices, capital=3_000_000.0, lot=LOT, cost=True):
    """Hold each book from its date until the next one, whole lots only.

    ``books`` maps a rebalance date to the list of codes bought that day.  The book is
    liquidated and rebuilt at each rebalance, so the cash that lot indivisibility leaves
    behind is real rather than assumed away.  Returns the daily equity curve and the
    average share of capital sitting in cash.
    """
    dates = prices.index
    stamps = sorted(pd.Timestamp(d) for d in books)
    equity = pd.Series(np.nan, index=dates, dtype=float)
    cash_share = []
    cash = capital
    holdings = {}
    position = {stamp: index for index, stamp in enumerate(stamps)}
    for date in dates:
        if date in position:
            value = cash
            for code, shares in holdings.items():
                price = prices.at[date, code] if code in prices.columns else np.nan
                if np.isfinite(price):
                    value += shares * price
            if holdings:
                for code, shares in list(holdings.items()):
                    price = prices.at[date, code] if code in prices.columns else np.nan
                    if not np.isfinite(price):
                        continue
                    notional = shares * price
                    cash += notional - (sell_cost(notional) if cost else 0.0)
                    del holdings[code]
            target = books[date]
            live = [code for code in target
                    if code in prices.columns and np.isfinite(prices.at[date, code])]
            if live:
                budget = cash / len(live)
                for code in live:
                    price = float(prices.at[date, code])
                    shares = affordable_shares(budget, price, lot)
                    if not shares:
                        continue
                    notional = shares * price
                    outflow = notional + (buy_cost(notional) if cost else 0.0)
                    if outflow > cash:
                        shares = affordable_shares(cash - (buy_cost(cash) if cost else 0.0), price, lot)
                        if not shares:
                            continue
                        notional = shares * price
                        outflow = notional + (buy_cost(notional) if cost else 0.0)
                    cash -= outflow
                    holdings[code] = shares
        value = cash
        for code, shares in holdings.items():
            price = prices.at[date, code] if code in prices.columns else np.nan
            if np.isfinite(price):
                value += shares * price
        equity.at[date] = value
        cash_share.append(cash / value if value > 0 else 1.0)
    equity = equity.ffill().dropna()
    returns = equity.pct_change().dropna()
    return dict(equity=equity, returns=returns, cash_share=float(np.mean(cash_share)) if cash_share else 1.0)


def cap_percentile(books, cap_frame, pool):
    """Average within-pool capitalisation percentile of what was bought.

    Zero means the selection sits in the smallest names of the pool, one in the largest.
    A strategy whose percentile drifts towards zero is a size bet whatever it says it
    is doing.
    """
    ranked = {}
    for date in sorted(pd.Timestamp(d) for d in books):
        available = cap_frame.loc[:date]
        if available.empty:
            continue
        asof = available.iloc[-1]
        ordered = asof[asof.index.intersection(pool)].dropna().sort_values(ascending=True)
        if len(ordered) < 10:
            continue
        rank = {code: index / max(len(ordered) - 1, 1) for index, code in enumerate(ordered.index)}
        picked = [rank[code] for code in books[date] if code in rank]
        if picked:
            ranked[date] = float(np.mean(picked))
    if not ranked:
        return dict(status="empty")
    values = np.array(list(ranked.values()))
    first, last = values[:max(len(values) // 3, 1)], values[-max(len(values) // 3, 1):]
    return dict(status="ready", rebalances=len(values), mean=float(values.mean()),
                median=float(np.median(values)), first_third=float(first.mean()),
                last_third=float(last.mean()))


def random_books(pool, size, dates, seed=0, draws=200):
    """``draws`` random baskets of ``size`` names from ``pool`` on the same dates."""
    rng = np.random.default_rng(seed)
    pool = list(pool)
    size = min(size, len(pool))
    return [{date: list(rng.choice(pool, size=size, replace=False)) for date in dates}
            for _ in range(draws)]


def factor_attribution(returns, start=None, end=None, market_code=baseline_capm.MARKET_CODE):
    """Market and size loadings for a strategy's daily returns."""
    window = returns.loc[start:end]
    if len(window) < 120:
        return dict(status="insufficient", observations=int(len(window)))
    market = baseline_capm.market_returns(window.index[0].date().isoformat(),
                                          window.index[-1].date().isoformat(), market_code)
    size = baseline_capm.market_returns(window.index[0].date().isoformat(),
                                        window.index[-1].date().isoformat(), baseline_capm.SIZE_CODE)
    joined = pd.concat([window.rename("strategy"), market.rename("market"), size.rename("size")],
                       axis=1, join="inner").dropna()
    if len(joined) < 120:
        return dict(status="insufficient", observations=int(len(joined)))
    single = baseline_capm.regress_strategy(joined["strategy"], joined["market"])
    size_factor = (joined["size"] - joined["market"]).to_numpy(dtype=float)
    double = baseline_capm.ols_two_factor(joined["strategy"].to_numpy(dtype=float),
                                          joined["market"].to_numpy(dtype=float), size_factor)
    return dict(status="ready", observations=int(len(joined)),
                alpha=double.get("alpha"), alpha_t=double.get("alpha_t"),
                beta_market=double.get("beta_market"), beta_size=double.get("beta_size"),
                beta_size_t=double.get("beta_size_t"),
                alpha_single=single.get("alpha"), beta_single=single.get("beta"),
                r_squared=double.get("r_squared"))


def measure(books, panel_dir=DEFAULT_PANEL, capital=3_000_000.0, draws=200, seed=0):
    """Executable benchmark, random null and selection diagnostics for one book."""
    books = {pd.Timestamp(date): codes for date, codes in books.items()}
    manifest, prices, traded, shares = load(panel_dir)
    pool = sorted({code for codes in books.values() for code in codes})
    dates = sorted(pd.Timestamp(d) for d in books)
    equal = simulate({date: pool for date in dates}, prices, capital=capital)
    chosen = simulate({date: sorted(books[date]) for date in dates}, prices, capital=capital)
    size = max((len(books[date]) for date in dates), default=1)
    sample = random_books(pool, size, dates, seed=seed, draws=draws)
    random_annual = []
    for drawn in sample:
        run = simulate(drawn, prices, capital=capital)
        series = run["returns"]
        years = len(series) / capm.TRADING_DAYS
        random_annual.append(float((1 + series).prod() ** (1 / years) - 1) if years > 0 else float("nan"))
    random_annual = np.array([value for value in random_annual if np.isfinite(value)])
    selected = chosen["returns"]
    years = len(selected) / capm.TRADING_DAYS
    selected_annual = float((1 + selected).prod() ** (1 / years) - 1) if years > 0 else float("nan")
    return dict(
        status="ready", capital=capital, pool=len(pool), rebalances=len(dates),
        selected=dict(annual=selected_annual, cash_share=chosen["cash_share"]),
        executable_equal_weight=dict(
            annual=float((1 + equal["returns"]).prod() ** (1 / (len(equal["returns"]) / capm.TRADING_DAYS)) - 1),
            cash_share=equal["cash_share"]),
        random=dict(draws=int(len(random_annual)), median=float(np.median(random_annual)),
                    p10=float(np.percentile(random_annual, 10)),
                    p90=float(np.percentile(random_annual, 90)),
                    share_beaten=float(np.mean(random_annual < selected_annual))),
        size_percentile=cap_percentile(books, traded * shares, pool),
        attribution=factor_attribution(selected),
    )


def load_books(path):
    """Read ``{"rebalances": [{"date": ..., "codes": [...]}, ...]}``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    books = {}
    for row in payload["rebalances"]:
        books[pd.Timestamp(row["date"])] = [str(code) for code in row["codes"]]
    return books


def main(argv=None):
    parser = argparse.ArgumentParser(description="Executable benchmark and selection diagnostics.")
    parser.add_argument("--books", required=True, help="JSON of rebalance dates and codes.")
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--capital", type=float, default=3_000_000.0)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    payload = measure(load_books(args.books), args.panel, args.capital, args.draws)
    if args.output:
        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        from src.data.preprocess import write_json
        write_json(target / "selection_diagnostics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
