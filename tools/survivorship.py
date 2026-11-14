"""How much of a stock-study number is survivorship, and where does it actually bite.

Three questions, in the order they can be answered:

1. **Can the local panel even see delistings?**  ``data/clean/panel_full`` holds the stocks listed
   when it was built.  Counting how many of its columns stop trading before the end answers this
   directly -- if almost none do, no amount of analysis on it can measure survivorship.
2. **How many names left the market?**  The exchanges publish delisting lists; the count inside the
   study window is the numerator of any bound.
3. **What does that do to the numbers we quote?**  For an equal-weight universe it can be bounded
   from the count, the survivors' multiple and an assumed recovery of the delisted names.  For a
   *difference* measured inside one pool -- which is what decided E1 -- it cancels, because both arms
   share the pool.

The last point is the one worth remembering: the ETF plan's returns never touch a stock pool, so
they are not exposed to this at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "data" / "clean" / "panel_full"
WINDOW = ("2019-01-01", "2024-12-31")
EARLY_STOP_DAYS = 20


def aliveness(close, end=None, early_days=EARLY_STOP_DAYS):
    """Which columns stop printing before the end, and which start after the beginning."""
    end = pd.Timestamp(end) if end else close.index[-1]
    start = close.index[0]
    last = close.apply(lambda series: series.last_valid_index())
    first = close.apply(lambda series: series.first_valid_index())
    return dict(codes=int(close.shape[1]),
                alive_at_end=int((last >= end - pd.Timedelta(days=early_days)).sum()),
                stopped_early=int((last < end - pd.Timedelta(days=early_days)).sum()),
                listed_during_window=int((first > start + pd.Timedelta(days=10)).sum()),
                first=str(start.date()), last=str(end.date()))


def equal_weight(close, start, end, codes=None, cadence="buy_hold"):
    """Equal-weighted index over whatever prints a price, in one of two cadences.

    ``buy_hold`` buys the starting basket and holds it (names that stop printing leave at their last
    price) -- this is what a person could actually have owned.  ``daily`` rebalances every session,
    which adds the rebalance premium on a volatile pool and is *not* comparable to a buy-and-hold
    benchmark; both are reported because conflating them moves the answer by several pp a year.
    """
    window = close.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if codes is not None:
        window = window.reindex(columns=[code for code in codes if code in window.columns])
    if window.empty:
        raise ValueError("empty window")
    if cadence == "daily":
        daily = window.pct_change().mean(axis=1, skipna=True).fillna(0.0)
        return (1.0 + daily).cumprod()
    opening = window.iloc[0]
    usable = [code for code in window.columns if np.isfinite(opening.get(code, np.nan))]
    level = window[usable].div(opening[usable]).mean(axis=1)
    return level.ffill().fillna(1.0)


def annualised(series):
    if len(series) < 2:
        return float("nan")
    years = len(series) / 242.0
    return float(series.iloc[-1] ** (1 / years) - 1)


def bias_bound(n_total, n_delisted, survivor_multiple, delisted_recovery, years):
    """Annualised overstatement of an equal-weight universe that forgot the delisted names.

    ``survivor_multiple`` is how much a surviving name grew over the window (e.g. 3.0 for +200%),
    ``delisted_recovery`` what a delisted name is worth at the end (0.0 = total loss, 0.2 = -80%).
    The forgotten names reweight the mean from ``survivor_multiple`` down to the blended value.
    """
    if n_total <= 0 or years <= 0:
        return float("nan")
    blended = ((n_total - n_delisted) * survivor_multiple + n_delisted * delisted_recovery) / n_total
    if survivor_multiple <= 0 or blended <= 0:
        return float("nan")
    return float(blended ** (1 / years) - survivor_multiple ** (1 / years))


def run(panel=PANEL, window=WINDOW, delisted=213, pools=None, log=print):
    close = pd.read_parquet(Path(panel) / "hfq_close.parquet")
    alive = aliveness(close)
    start, end = window
    result = dict(aliveness=alive, window=dict(start=start, end=end),
                  delisted_in_window=delisted, bounds=[], pools=[], universe=[])
    for cadence in ("buy_hold", "daily"):
        universe = equal_weight(close, start, end, cadence=cadence)
        years = len(universe) / 242.0
        survivors = float(universe.iloc[-1])
        result["universe"].append(dict(cadence=cadence, codes=int(close.shape[1]),
                                       multiple=survivors, annualised=annualised(universe)))
        for recovery in (0.0, 0.2, 0.5):
            bound = bias_bound(alive["codes"], delisted, survivors, recovery, years)
            result["bounds"].append(dict(cadence=cadence, delisted_recovery=recovery, annual_gap=bound))
        log("%-9s 全市场等权 %4d 只  年化 %6.2f%%" % (cadence, alive["codes"], 100 * annualised(universe)))
    for recovery in (0.0, 0.2, 0.5):
        bound = [row for row in result["bounds"] if row["cadence"] == "buy_hold" and row["delisted_recovery"] == recovery][0]
        log("退市按残值 %.0f%% 计：等权宇宙年化高估 %.2fpp" % (100 * recovery, -100 * bound["annual_gap"]))
    for name, codes in (pools or {}).items():
        for cadence in ("buy_hold", "daily"):
            series = equal_weight(close, start, end, codes=codes, cadence=cadence)
            result["pools"].append(dict(name=name, cadence=cadence, codes=len(codes),
                                        annualised=annualised(series)))
            log("%-12s %-9s %4d 只  年化 %6.2f%%" % (name, cadence, len(codes), 100 * annualised(series)))
    return result
    for recovery in (0.0, 0.2, 0.5):
        bound = bias_bound(alive["codes"], delisted, survivors, recovery, years)
        result["bounds"].append(dict(delisted_recovery=recovery, annual_gap=bound))



def main(argv=None):
    parser = argparse.ArgumentParser(description="Survivorship: what the panel can and cannot say.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--delisted", type=int, default=213,
                        help="Delistings inside the window (SH 80 + SZ 133 from the exchange lists).")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "runs" / "survivorship"))
    args = parser.parse_args(argv)
    result = run(args.panel, delisted=args.delisted)
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "survivorship.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
