"""S1: does excluding imminent lockup expiries improve the same book?

The previous seven attempts all built a new portfolio and compared its return against a
benchmark, which is the most expensive way to test a signal: the two series differ
everywhere, so the tracking error is large and the required effect is enormous.  This
test is paired instead.  The same reversal book is run twice on the same dates with the
same costs, and the only difference is that the treated version refuses to buy a name
whose lockup expires inside the holding period, filling the slot from the next rank.

A paired design does not make a weak signal strong.  It makes the variance of the
comparison small, which is a different and much cheaper thing: the difference series is
zero on most days because both books hold the same names.

The rule is fixed at twenty trading sessions -- the holding period itself -- so that
"exclude what unlocks before I can sell" is a sentence about the strategy rather than a
parameter to tune.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import capm_backtest as capm
from tools import feature_backtest as fb
from tools import unlock_ic
from tools.feature_panel import load_panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "data" / "clean" / "feature_top1800"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "supply_overlay"
WINDOW = ("2019-01-02", "2024-12-31")
HORIZON = 20
BAR = 2.5          # raised from 2.0: this is the eighth test on the same sample
BLOCK = 20


def unlock_window(events, index, codes, horizon=HORIZON):
    """For each decision date, the names with an expiry inside the next ``horizon`` sessions."""
    dates = list(index)
    position = {date: number for number, date in enumerate(dates)}
    holding_end = {}
    for date in dates:
        number = position[date]
        holding_end[date] = dates[min(number + horizon, len(dates) - 1)]
    blocked = {date: set() for date in dates}
    for row in events.itertuples(index=False):
        if row.code not in codes:
            continue
        for date, end in holding_end.items():
            if date < row.date <= end:
                blocked[date].add(row.code)
    return blocked


def simulate(panel, prices, signs, features, window, blocked=None):
    """The reversal book, optionally refusing names that unlock inside the holding period."""
    codes = list(prices.columns)
    dates = prices.index[(prices.index >= pd.Timestamp(window[0])) & (prices.index <= pd.Timestamp(window[1]))]
    marks = set(fb.rebalance_dates(prices.index, window))
    cash, holdings, equity, target = fb.CAPITAL, {}, [], []
    refusals = 0
    for date in dates:
        if date in marks:
            for code, shares in list(holdings.items()):
                price = prices.at[date, code]
                if not np.isfinite(price) or price <= 0:
                    continue
                notional = shares * price
                cash += notional - max(notional * fb.COMMISSION, fb.MIN_COMMISSION) - notional * fb.STAMP - notional * fb.SLIPPAGE
                del holdings[code]
            score = fb.composite(panel, features, signs, date, codes)
            if score is not None:
                banned = blocked.get(date, set()) if blocked is not None else set()
                ordered = score.sort_values(ascending=False)
                rank_of = {code: number for number, code in enumerate(ordered.index)}
                keep = [code for code in target if rank_of.get(code, 10 ** 9) < fb.BUFFER]
                keep = sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))[:fb.TOP_K]
                for code in ordered.index:
                    if len(keep) >= fb.TOP_K:
                        break
                    if code in keep:
                        continue
                    if not np.isfinite(prices.at[date, code]):
                        continue
                    if code in banned:
                        refusals += 1
                        continue
                    keep.append(code)
                target = keep
                budget = cash / len(keep) if keep else 0.0
                for code in keep:
                    price = float(prices.at[date, code])
                    if not np.isfinite(price) or price <= 0:
                        continue
                    shares = int(budget // (price * fb.LOT)) * fb.LOT
                    while shares and shares * price * (1 + fb.COMMISSION + fb.SLIPPAGE) > cash:
                        shares -= fb.LOT
                    if shares <= 0:
                        continue
                    notional = shares * price
                    cash -= notional + max(notional * fb.COMMISSION, fb.MIN_COMMISSION) + notional * fb.SLIPPAGE
                    holdings[code] = shares
        equity.append(cash + sum(shares * prices.at[date, code]
                                 for code, shares in holdings.items() if np.isfinite(prices.at[date, code])))
    returns = pd.Series(equity, index=dates).pct_change().dropna()
    return returns, refusals


def paired(control, treated, block=BLOCK):
    """Difference of the two books, with a t computed on non-overlapping blocks."""
    joined = pd.concat([control.rename("control"), treated.rename("treated")], axis=1, join="inner").dropna()
    difference = joined["treated"] - joined["control"]
    years = len(joined) / capm.TRADING_DAYS
    blocks = [difference.iloc[start:start + block].sum()
              for start in range(0, len(difference) - block + 1, block)]
    values = np.array(blocks, dtype=float)
    t = float(values.mean() / (values.std(ddof=1) / math.sqrt(len(values)))) if len(values) > 1 else float("nan")
    return dict(days=int(len(joined)),
                annual_gap=float((1 + joined["treated"]).prod() ** (1 / years)
                                 - (1 + joined["control"]).prod() ** (1 / years)),
                mean_daily=float(difference.mean()),
                blocks=int(len(values)), block_t=t,
                share_positive=float((values > 0).mean()) if len(values) else None)


def measure(panel_dir=PANEL, window=WINDOW):
    manifest, panel, prices = load_panel(panel_dir)
    features = list(manifest["feature_cols"])
    codes = list(prices.columns)
    signs = {name: -1.0 for name in features}          # E1's construction, unchanged
    index = prices.index[(prices.index >= pd.Timestamp(window[0])) & (prices.index <= pd.Timestamp(window[1]))]
    events = unlock_ic.download()
    blocked = unlock_window(events, index, set(codes))
    control, _ = simulate(panel, prices, signs, features, window)
    treated, refusals = simulate(panel, prices, signs, features, window, blocked)
    result = paired(control, treated)
    return dict(status="ready", panel=str(panel_dir), window=dict(start=window[0], end=window[1]),
                events=int(len(events)),
                decision_dates_with_event=int(sum(1 for date in blocked if blocked[date])),
                refusals=int(refusals),
                control=fb.summarise(control, "control"),
                treated=fb.summarise(treated, "treated"),
                paired=result, bar=BAR,
                verdict=dict(passed=bool(result["annual_gap"] > 0 and result["block_t"] >= BAR)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired test of excluding imminent lockups.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(args.panel)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "supply_overlay.json", payload)
    print(json.dumps({k: payload[k] for k in ("control", "treated", "paired", "verdict", "refusals")},
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
