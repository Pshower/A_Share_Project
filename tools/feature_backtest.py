"""Select features on one window, trade them on the next, and report both halves.

"Pick effective features" is the most reliable way to manufacture a backtest, because a
selection made on the same window it is judged on will always succeed.  So the split is
hard: features are chosen and signed using rank IC measured **only** on the selection
window, then the book is built and judged **only** on the test window.  The selection
window result is printed beside it as a specification check, not as evidence.

The book is the construction E1 already fixed -- top thirty, sixty-name buffer, twenty-
session holds, three million through the same cost model -- so that the only thing
changing here is which features are wanted and which way round.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import alpha_ic
from tools import capm_backtest as capm
from tools.feature_panel import load_panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "data" / "clean" / "feature_top1800"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "feature_backtest"
SELECT = ("2019-01-02", "2021-12-31")
TEST = ("2022-01-03", "2024-12-31")
HORIZON = 20
TOP_K = 30
BUFFER = 60
CAPITAL = 3_000_000
LOT = 100
COMMISSION, MIN_COMMISSION, STAMP, SLIPPAGE = 0.0003, 5.0, 0.0005, 0.001
RANDOM_DRAWS = 200


def forward(prices, horizon=HORIZON):
    return prices.shift(-horizon) / prices - 1.0


def ic_table(panel, prices, features, window):
    """Mean rank IC and its t statistic per feature inside one window."""
    start, end = window
    labels = forward(prices).loc[start:end]
    dates = labels.index[labels.index.isin(panel.index)]
    rows = {}
    for name in features:
        block = panel.xs(name, axis=1, level=1).reindex(index=dates)
        series = alpha_ic.rank_ic(block.to_numpy(dtype=float), labels.to_numpy(dtype=float), dates)
        stats = alpha_ic.ic_stats(series, HORIZON, threshold=0.0)
        rows[name] = dict(mean_ic=stats.get("mean_ic"), t=stats.get("t_nw_h"),
                          independent=stats.get("independent_dates"))
    return rows


def select_features(table, min_ic=0.01, min_t=2.0):
    """Keep the features whose selection-window IC is large enough and signed."""
    chosen = {}
    for name, row in table.items():
        ic, t = row.get("mean_ic"), row.get("t")
        if ic is None or t is None or not np.isfinite(ic) or not np.isfinite(t):
            continue
        if abs(ic) < min_ic or abs(t) < min_t:
            continue
        chosen[name] = 1.0 if ic > 0 else -1.0
    return chosen


def composite(panel, features, signs, date, codes):
    """Signed rank mean on one date; higher is a stronger buy."""
    score, used = np.zeros(len(codes)), 0
    for name, sign in signs.items():
        try:
            values = panel.xs(name, axis=1, level=1).loc[date].reindex(codes).to_numpy(dtype=float)
        except KeyError:
            continue
        if not np.isfinite(values).any():
            continue
        rank = pd.Series(values).rank(method="average", pct=True).to_numpy()
        score += sign * np.nan_to_num(rank, nan=0.5)
        used += 1
    return None if not used else pd.Series(score / used, index=codes)


def rebalance_dates(index, window, step=HORIZON):
    inside = index[(index >= pd.Timestamp(window[0])) & (index <= pd.Timestamp(window[1]))]
    return list(inside[::step])


def simulate(panel, prices, signs, features, window, capital=CAPITAL):
    """Whole-lot book, equal weight, with the buffer rule and the cost model."""
    codes = list(prices.columns)
    dates = prices.index[(prices.index >= pd.Timestamp(window[0])) & (prices.index <= pd.Timestamp(window[1]))]
    marks = rebalance_dates(prices.index, window)
    target, cash, holdings, equity, cash_ratio = {}, capital, {}, [], []
    for date in dates:
        if date in marks:
            for code, shares in list(holdings.items()):
                price = prices.at[date, code]
                if not np.isfinite(price) or price <= 0:
                    continue          # no quote today: carry the position, do not price it
                notional = shares * price
                cash += notional - max(notional * COMMISSION, MIN_COMMISSION) - notional * STAMP - notional * SLIPPAGE
                del holdings[code]
            score = composite(panel, features, signs, date, codes)
            if score is not None:
                ordered = score.sort_values(ascending=False)
                rank_of = {code: position for position, code in enumerate(ordered.index)}
                keep = [code for code in target if rank_of.get(code, 10 ** 9) < BUFFER]
                keep = sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))[:TOP_K]
                for code in ordered.index:
                    if len(keep) >= TOP_K:
                        break
                    if code not in keep and np.isfinite(prices.at[date, code]):
                        keep.append(code)
                target = keep
                budget = cash / len(keep) if keep else 0.0
                for code in keep:
                    price = float(prices.at[date, code])
                    if not np.isfinite(price) or price <= 0:
                        continue
                    shares = int(budget // (price * LOT)) * LOT
                    while shares and shares * price + max(shares * price * COMMISSION, MIN_COMMISSION) \
                            + shares * price * SLIPPAGE > cash:
                        shares -= LOT
                    if shares <= 0:
                        continue
                    notional = shares * price
                    cash -= notional + max(notional * COMMISSION, MIN_COMMISSION) + notional * SLIPPAGE
                    holdings[code] = shares
        value = cash + sum(shares * prices.at[date, code]
                           for code, shares in holdings.items() if np.isfinite(prices.at[date, code]))
        equity.append(value)
        cash_ratio.append(cash / value if value > 0 else 1.0)
    series = pd.Series(equity, index=dates)
    return series.pct_change().dropna(), float(np.mean(cash_ratio))


def summarise(returns, label):
    years = len(returns) / capm.TRADING_DAYS
    nav = (1 + returns).cumprod()
    annual = float(nav.iloc[-1] ** (1 / years) - 1)
    vol = float(returns.std(ddof=1) * math.sqrt(capm.TRADING_DAYS))
    return dict(label=label, annual=annual, vol=vol,
                drawdown=float((nav / nav.cummax() - 1).min()),
                sharpe=(annual - capm.CASH_RATE) / vol if vol else float("nan"))


def randomised(prices, window, size=TOP_K, draws=RANDOM_DRAWS, seed=7):
    """Random equal-weight books of the same size, no buffer, same costs."""
    rng = np.random.default_rng(seed)
    codes = np.array(list(prices.columns))
    marks = set(rebalance_dates(prices.index, window))
    out = []
    for _ in range(draws):
        cash, holdings, equity = CAPITAL, {}, []
        dates = prices.index[(prices.index >= pd.Timestamp(window[0])) & (prices.index <= pd.Timestamp(window[1]))]
        for date in dates:
            if date in marks:
                for code, shares in list(holdings.items()):
                    price_now = prices.at[date, code]
                    if not np.isfinite(price_now):
                        continue
                    notional = shares * price_now
                    cash += notional - max(notional * COMMISSION, MIN_COMMISSION) - notional * STAMP - notional * SLIPPAGE
                    del holdings[code]
                pick = [code for code in rng.choice(codes, size=size, replace=False)
                        if np.isfinite(prices.at[date, code])]
                budget = cash / len(pick) if pick else 0.0
                for code in pick:
                    price = float(prices.at[date, code])
                    if not np.isfinite(price) or price <= 0:
                        continue
                    shares = int(budget // (price * LOT)) * LOT
                    while shares and shares * price * (1 + COMMISSION + SLIPPAGE) > cash:
                        shares -= LOT
                    if shares <= 0:
                        continue
                    notional = shares * price
                    cash -= notional + max(notional * COMMISSION, MIN_COMMISSION) + notional * SLIPPAGE
                    holdings[code] = shares
            equity.append(cash + sum(shares * prices.at[date, code]
                                     for code, shares in holdings.items() if np.isfinite(prices.at[date, code])))
        series = pd.Series(equity, index=dates).pct_change().dropna()
        years = len(series) / capm.TRADING_DAYS
        out.append(float((1 + series).cumprod().iloc[-1] ** (1 / years) - 1))
    return np.array(out)


def measure(panel_dir=PANEL, min_ic=0.01, min_t=2.0):
    manifest, panel, prices = load_panel(panel_dir)
    features = list(manifest["feature_cols"])
    selection = ic_table(panel, prices, features, SELECT)
    signs = select_features(selection, min_ic, min_t)
    payload = dict(status="ready", panel=str(panel_dir), codes=len(prices.columns),
                   features=len(features), horizon=HORIZON, select=SELECT, test=TEST,
                   selected=signs, selection_ic=selection, windows={})
    if not signs:
        payload["status"] = "no_features_selected"
        return payload
    for label, window in (("selection", SELECT), ("test", TEST)):
        returns, cash_share = simulate(panel, prices, signs, features, window)
        payload["windows"][label] = dict(summary=summarise(returns, label), cash_share=cash_share,
                                         returns=returns)
    null = randomised(prices, TEST)
    selected = payload["windows"]["test"]["summary"]["annual"]
    payload["null"] = dict(draws=int(len(null)), median=float(np.median(null)),
                           p10=float(np.percentile(null, 10)), p90=float(np.percentile(null, 90)),
                           share_beaten=float(np.mean(null < selected)))
    benchmark = payload["windows"]["test"]["returns"]
    excess = benchmark - float(np.median(null)) / capm.TRADING_DAYS
    joined = payload["windows"]["test"]["returns"]
    years = len(joined) / capm.TRADING_DAYS
    gap = selected - payload["null"]["median"]
    tracking = float(joined.std(ddof=1) * math.sqrt(capm.TRADING_DAYS)) if len(joined) else float("nan")
    payload["verdict"] = dict(net_excess_vs_random=gap, years=years,
                              beats_p90=bool(selected > payload["null"]["p90"]),
                              passed=bool(selected > payload["null"]["p90"]))
    for window in payload["windows"].values():
        window.pop("returns", None)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select features on one window, trade on the next.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--min-ic", type=float, default=0.01)
    parser.add_argument("--min-t", type=float, default=2.0)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(args.panel, args.min_ic, args.min_t)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "feature_backtest.json", payload)
    print(json.dumps(dict(selected=payload["selected"], windows=payload.get("windows"),
                          null=payload.get("null"), verdict=payload.get("verdict")),
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
