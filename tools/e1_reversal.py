"""E1: the mechanical cross-sectional reversal book, run once.

The protocol is fixed in ``doc/research_direction_20261107.md`` and is not searched:

* pool ``data/clean/research_v3``, all of its codes;
* score = the equally weighted rank of **all twenty price features, all negated**;
* hold the top thirty names at equal weight for twenty sessions;
* a name is only replaced when it falls out of the top sixty (a fixed buffer);
* 3,000,000 of capital through the repository's own broker, so lot indivisibility,
  the five-yuan commission floor, stamp duty and slippage all apply;
* report the V and full windows; pass requires **net excess positive in both** and a
  **combined t of at least two**, where t is the information ratio times the square
  root of the years.

The benchmark is the executable equal weight of the same pool under the same capital
and cost model, which is the standard recorded in ``doc/selection_diagnostics_20261108.md``.
Using the pool's cap-weighted or a published index here would change what the test means,
because the pool is only 298 names and every one of them is a survivor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.run import run_policy
from src.envs.broker import BrokerConfig
from src.envs.market import MarketDataProvider
from tools import capm_backtest as capm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "clean" / "research_v3"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "e1_reversal"
CAPITAL = 3_000_000
TOP_K = 30
BUFFER = 60
HOLD = 20
BROKER = dict(commission=0.0003, min_commission=5.0, sell_tax=0.0005,
              slippage=0.001, lot_size=100)
WINDOWS = {"V": ("2019-01-01", "2024-12-31"), "full": ("2017-01-02", "2024-12-31")}


class ReversalBook:
    """Rank-mean of the negated features, top thirty with a sixty-name buffer."""

    def __init__(self, feature_cols, codes=None, top_k=TOP_K, buffer=BUFFER, interval=HOLD):
        self.feature_cols = list(feature_cols)
        self.codes = list(codes) if codes else None
        self.top_k = top_k
        self.buffer = buffer
        self.interval = interval
        self.held = set()
        self.books = {}

    def scores(self, observation):
        block = np.asarray(observation["stock_features"], dtype=float)[:, -1, :]
        valid = np.asarray(observation["valid_mask"], dtype=bool)
        count = int(valid.sum())
        if count == 0:
            return None
        ranks = np.zeros(count, dtype=float)
        usable = 0
        for column in range(block.shape[1]):
            values = block[valid, column]
            if not np.isfinite(values).any():
                continue
            order = pd.Series(values).rank(method="average", pct=True).to_numpy()
            ranks += np.nan_to_num(order, nan=0.5)
            usable += 1
        if not usable:
            return None
        composite = ranks / usable                      # mean percentile, 0 = smallest
        return composite, np.flatnonzero(valid)          # negated: the smallest ranks win

    def act(self, observation, step):
        if step % self.interval:
            return None
        scored = self.scores(observation)
        if scored is None:
            return None
        composite, selected = scored
        order = selected[np.argsort(composite, kind="stable")]
        rank_of = {int(code): position for position, code in enumerate(order)}
        keep = [code for code in self.held if rank_of.get(code, 10 ** 9) < self.buffer]
        keep = sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))[:self.top_k]
        for code in order:
            if len(keep) >= self.top_k:
                break
            if int(code) not in keep:
                keep.append(int(code))
        self.held = set(keep)
        # Record positions as codes so the diagnostics can read them.
        self.books[step] = [self.codes[i] for i in keep] if self.codes else list(keep)
        weights = np.zeros(len(observation["valid_mask"]))
        if keep:
            weights[keep] = 1.0 / len(keep)
        return weights


class EqualWeightBook:
    """The pool bought at equal weight on the same dates, through the same broker."""

    def __init__(self, interval=HOLD):
        self.interval = interval

    def act(self, observation, step):
        if step % self.interval:
            return None
        valid = np.asarray(observation["valid_mask"], dtype=bool)
        buyable = np.asarray(observation["buy_mask"], dtype=bool) & valid
        weights = np.zeros(len(valid))
        count = int(buyable.sum())
        if count:
            weights[buyable] = 1.0 / count
        return weights


def build_market(manifest, panel, prices, start, end, lookback=1):
    return MarketDataProvider(panel, manifest["feature_cols"], price_path=prices,
                              price_basis=manifest["price_basis"],
                              start_date=pd.Timestamp(start), end_date=pd.Timestamp(end),
                              lookback=lookback)


def curve(daily):
    """Annual return, volatility and the equity curve from a ``run_policy`` table."""
    frame = pd.DataFrame(daily)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    equity = frame["equity"].astype(float)
    returns = equity.pct_change().dropna()
    years = len(returns) / capm.TRADING_DAYS
    annual = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 and equity.iloc[0] > 0 else float("nan")
    vol = float(returns.std(ddof=1) * math.sqrt(capm.TRADING_DAYS))
    return frame, equity, returns, dict(annual=annual, vol=vol, years=years,
                                        final_cash_share=float(frame["cash"].iloc[-1] / equity.iloc[-1])
                                        if equity.iloc[-1] else float("nan"),
                                        mean_cash_share=float((frame["cash"] / frame["equity"]).mean()))


def window_result(strategy, benchmark, window):
    """Net excess in the window and the information ratio times the square root of years."""
    joined = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")],
                       axis=1, join="inner").dropna()
    if len(joined) < 60:
        return dict(status="insufficient", observations=int(len(joined)))
    excess = joined["strategy"] - joined["benchmark"]
    years = len(joined) / capm.TRADING_DAYS
    tracking = float(excess.std(ddof=1) * math.sqrt(capm.TRADING_DAYS))
    net = float((1 + joined["strategy"]).prod() ** (1 / years) - 1
                - ((1 + joined["benchmark"]).prod() ** (1 / years) - 1))
    t = (net / tracking * math.sqrt(years)) if tracking > 0 else float("nan")
    return dict(status="ready", observations=int(len(joined)), years=float(years),
                net_excess=net, tracking_error=tracking, t=t)


def run(start=None, end=None, capital=CAPITAL, dataset=DATASET):
    directory = Path(dataset)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    start = start or manifest.get("start_date") or str(panel.index[0].date())
    end = end or str(panel.index[-1].date())
    settings = dict(initial_capital=capital, seed=42, lookback=1,
                    rebalance_every=HOLD, momentum_top_k=TOP_K, broker=dict(BROKER))
    payload = dict(status="ready", dataset=str(directory), capital=capital,
                   start=str(pd.Timestamp(start).date()), end=str(pd.Timestamp(end).date()),
                   top_k=TOP_K, buffer=BUFFER, hold=HOLD,
                   features=list(manifest["feature_cols"]), windows={})
    curves = {}
    market_codes = [str(code) for code in prices.columns.get_level_values(0).unique()]
    reversal = None
    for label, strategy in (("reversal", ReversalBook(manifest["feature_cols"], market_codes)),
                            ("equal_weight", EqualWeightBook())):
        market = build_market(manifest, panel, prices, start, end, settings["lookback"])
        daily, holdings, orders, assumptions = run_policy(market, strategy, settings)
        frame, equity, returns, summary = curve(daily)
        curves[label] = returns
        if label == "reversal":
            curve_daily = daily
            reversal = strategy
        payload[label] = dict(summary=summary, days=int(len(returns)),
                              assumptions=list(assumptions)[:4])
    # Diagnostics required by doc/selection_diagnostics_20261108.md: where in the size
    # distribution the book sits, and whether its return is a size bet.
    from tools import selection_diagnostics as diag
    frame, _, returns, _ = curve(curve_daily)
    stamps = list(frame.index)
    books = {stamps[min(step, len(stamps) - 1)]: codes for step, codes in reversal.books.items()}
    payload["diagnostics"] = {}
    try:
        manifest_full, prices_full, traded_full, shares_full = __import__(
            "tools.full_market_panel", fromlist=["load"]).load("data/clean/panel_full")
        payload["diagnostics"]["size_percentile"] = diag.cap_percentile(
            books, traded_full * shares_full, list(prices.columns.get_level_values(0)))
    except (OSError, KeyError, ValueError) as error:
        payload["diagnostics"]["size_percentile"] = dict(status=f"skipped: {error}")
    payload["diagnostics"]["attribution"] = diag.factor_attribution(returns)
    for window, (low, high) in WINDOWS.items():
        sub = {key: series.loc[low:high] for key, series in curves.items()}
        payload["windows"][window] = window_result(sub["reversal"], sub["equal_weight"], window)
    ready = [row for row in payload["windows"].values() if row.get("status") == "ready"]
    if len(ready) == 2:
        # V (2019-2024) sits inside full (2017-2024), so the two windows cannot be pooled
        # as if independent. The combined statistic is therefore the full-sample one, and
        # the 2017-2018 complement is reported separately so a reader can see whether the
        # answer comes from the pre-2019 segment.
        complement = {key: series.loc[WINDOWS["full"][0]:WINDOWS["V"][0]]
                      for key, series in curves.items()}
        payload["windows"]["pre_2019"] = window_result(complement["reversal"], complement["equal_weight"], "pre_2019")
        positive = all(row["net_excess"] > 0 for row in ready)
        combined = payload["windows"]["full"]["t"]
        payload["verdict"] = dict(net_positive_both=bool(positive),
                                  combined_t=float(combined),
                                  combined_basis="full window, since V is nested inside it",
                                  pre_2019_t=payload["windows"]["pre_2019"].get("t"),
                                  passed=bool(positive and combined >= 2.0))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run E1 once.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=CAPITAL)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = run(args.start, args.end, args.capital)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "e1_reversal.json", payload)
    print(json.dumps(payload.get("windows"), ensure_ascii=False, indent=2, default=float))
    print(json.dumps(payload.get("verdict"), ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
