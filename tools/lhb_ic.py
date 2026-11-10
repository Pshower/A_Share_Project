"""Does a 龙虎榜 net-buy day predict the next twenty sessions?

The 龙虎榜 is an **event** disclosure: a stock only appears when it moved, or traded, enough
to trigger a rule.  That makes the coverage a property of the exchange's rulebook rather
than a modelling choice, and it is the first thing to measure -- a source that names 1.4% of
the pool every day cannot rank the pool, whatever its IC is.

Two decisions are fixed before the data is touched:

**Which quantity.** ``净买额 / 龙虎榜成交额`` -- the net institutional-and-broker buying as a
share of that day's top-trader turnover.  The raw 净买额 in yuan is a size bet; the ratio is
the same number per unit of trading, and it is the one that can be compared across stocks.

**Which direction.** Positive, written here.  The story is that the seats which show up on
the list know something the tape has not priced yet, so net buying continues.  It is a
weaker prior than the lockup case (which had a mechanical seller), which is why the bar is
the raised one.

The signal is shifted one session: the list is published after the close, so a book decided
at the close of ``t`` may use it from ``t+1`` at the earliest.  Nothing else is tuned.
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
CACHE = PROJECT_ROOT / "data" / "online" / "lhb" / "lhb.parquet"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "lhb_ic"
START, END = "2019-01-02", "2024-12-31"
HORIZON = 20
BAR = 0.031            # the same break-even IC the other sources are held to
T_BAR = 2.5            # the raised bar for a second-family source


def signal_matrix(events, index, codes):
    """Net buying as a share of top-trader turnover, one row per stock-date, lagged a day."""
    data = events[events.code.isin(set(codes))]
    grouped = data.groupby(["date", "code"]).agg(net=("net_buy", "sum"), amount=("amount", "sum"))
    grouped = grouped[grouped.amount > 0]
    grouped["signal"] = grouped.net / grouped.amount
    frame = grouped["signal"].unstack("code").reindex(index=index, columns=codes)
    return frame.shift(1)                       # published after the close of the event day


def measure(panel_dir=PANEL, cache=CACHE, bar=BAR):
    manifest, panel, prices = load_panel(panel_dir)
    codes = list(prices.columns)
    index = prices.index[(prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(END))]
    events = pd.read_parquet(cache)
    signal = signal_matrix(events, index, codes)
    neutral = signal.fillna(0.0)               # a stock that never appeared is neutral
    labels = (prices.shift(-HORIZON) / prices - 1.0).reindex(index)
    decision = list(index[::HORIZON])
    rows = []
    for date in decision:
        if date not in signal.index or date not in labels.index:
            continue
        rows.append(dict(date=date, signal=signal.loc[date].to_numpy(dtype=float),
                         neutral=neutral.loc[date].to_numpy(dtype=float),
                         target=labels.loc[date].to_numpy(dtype=float)))
    if not rows:
        return dict(status="no_dates")

    def ic(restrict):
        series, coverage = {}, []
        for row in rows:
            values = row["signal"] if restrict else row["neutral"]
            target = row["target"]
            usable = np.isfinite(values) & np.isfinite(target)
            universe = np.isfinite(target)
            coverage.append(float((np.isfinite(row["signal"]) & universe).sum() / universe.sum())
                            if universe.any() else 0.0)
            if usable.sum() < 10:
                series[row["date"]] = np.nan
                continue
            a = pd.Series(values[usable]).rank().to_numpy()
            b = pd.Series(target[usable]).rank().to_numpy()
            series[row["date"]] = float(np.corrcoef(a, b)[0, 1])
        clean = pd.Series(series).dropna()
        stats = alpha_ic.ic_stats(clean, HORIZON, threshold=bar) if len(clean) else {}
        return dict(dates=int(len(clean)), mean_ic=stats.get("mean_ic"), t=stats.get("t_nw_h"),
                    hit=float((clean > 0).mean()) if len(clean) else None,
                    coverage=float(np.mean(coverage)))

    conditioned, full = ic(True), ic(False)
    return dict(status="ready", panel=str(panel_dir), codes=len(codes), horizon=HORIZON,
                events=int(len(events)), bar=bar, t_bar=T_BAR,
                full=full, conditioned=conditioned,
                verdict=dict(
                    direction_positive=bool((conditioned.get("mean_ic") or 0) > 0),
                    clears_bar=bool(abs(conditioned.get("mean_ic") or 0) >= bar),
                    clears_t=bool(abs(conditioned.get("t") or 0) >= T_BAR),
                    coverage=conditioned.get("coverage"),
                    passes=bool((conditioned.get("mean_ic") or 0) > 0
                                and abs(conditioned.get("mean_ic") or 0) >= bar
                                and abs(conditioned.get("t") or 0) >= T_BAR),
                    note="conditioned IC uses only stock-dates that appeared on the list"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank IC of the 龙虎榜 net-buy ratio.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(Path(args.panel))
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "lhb_ic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                                        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
