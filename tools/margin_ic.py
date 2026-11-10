"""Does a rising margin balance predict the next twenty sessions?

Margin financing is retail leverage.  When the balance on a stock grows fast, the buyers who
added it are already positioned and paying interest, so the marginal buyer becomes a
potential seller -- the classic crowding story.  That is the prior this file registers, and
it is written here **before** the archive finished downloading.

**Which quantity.** The twenty-session percentage change in the outstanding financing
balance (``融资余额``), not the balance itself: the level is a function of price and float,
the change is the flow.

**Which direction.** Negative.  A stock whose leverage grew fastest over the last month is
the one expected to do worse over the next one.

**Availability.** The exchanges publish each day's balances after the close, so a book
decided at the close of ``t`` can use the figure for ``t`` only from ``t+1``.  The signal is
shifted one session for that reason, not because the data looked better that way.

Coverage is the first thing to report: only margin-eligible stocks exist in this table
(about 1,000 names in 2019 growing to 3,600 by 2024), so the signal cannot rank names that
are not on the list, whatever its IC is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import alpha_ic
from tools.feature_panel import load_panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "data" / "clean" / "feature_top1800"
CACHE = PROJECT_ROOT / "data" / "online" / "margin" / "margin.parquet"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "margin_ic"
START, END = "2019-01-02", "2024-12-31"
HORIZON = 20
LOOKBACK = 20
BAR = 0.031
T_BAR = 2.5


def balance_matrix(events, index, codes):
    """Outstanding financing balance per stock-day, in yuan."""
    data = events[events["code"].isin(set(codes))]
    return data.pivot_table(index="date", columns="code", values="balance", aggfunc="last") \
        .reindex(index=index, columns=codes)


def growth_matrix(balance, lookback=LOOKBACK):
    """Twenty-session change, shifted one session: published after the close."""
    return balance.pct_change(lookback, fill_method=None).shift(1)


def measure(panel_dir=PANEL, cache=CACHE, bar=BAR):
    manifest, panel, prices = load_panel(panel_dir)
    codes = list(prices.columns)
    index = prices.index[(prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(END))]
    events = pd.read_parquet(cache)
    events["date"] = pd.to_datetime(events["date"])
    balance = balance_matrix(events, index, codes)
    growth = growth_matrix(balance)
    labels = (prices.shift(-HORIZON) / prices - 1.0).reindex(index)
    decision = list(index[::HORIZON])
    series, coverage = {}, []
    for date in decision:
        if date not in labels.index:
            continue
        values = growth.loc[date].to_numpy(dtype=float)
        target = labels.loc[date].to_numpy(dtype=float)
        usable = np.isfinite(values) & np.isfinite(target)
        coverage.append(float(usable.sum() / max(int(np.isfinite(target).sum()), 1)))
        if usable.sum() < 10:
            series[date] = np.nan
            continue
        a = pd.Series(values[usable]).rank().to_numpy()
        b = pd.Series(target[usable]).rank().to_numpy()
        series[date] = float(np.corrcoef(a, b)[0, 1])
    clean = pd.Series(series).dropna()
    stats = alpha_ic.ic_stats(clean, HORIZON, threshold=bar) if len(clean) else {}
    full = dict(dates=int(len(clean)), mean_ic=stats.get("mean_ic"), t=stats.get("t_nw_h"),
                hit=float((clean < 0).mean()) if len(clean) else None,
                coverage=float(np.mean(coverage)) if coverage else None)
    return dict(status="ready", panel=str(panel_dir), codes=len(codes), horizon=HORIZON,
                lookback=LOOKBACK, events=int(len(events)), bar=bar, t_bar=T_BAR, full=full,
                verdict=dict(
                    direction_negative=bool((full.get("mean_ic") or 0) < 0),
                    clears_bar=bool(abs(full.get("mean_ic") or 0) >= bar),
                    clears_t=bool(abs(full.get("t") or 0) >= T_BAR),
                    coverage=full.get("coverage"),
                    passes=bool((full.get("mean_ic") or 0) < 0
                                and abs(full.get("mean_ic") or 0) >= bar
                                and abs(full.get("t") or 0) >= T_BAR)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank IC of the margin-balance change.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(Path(args.panel))
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "margin_ic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                                           encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
