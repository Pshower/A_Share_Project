"""Does an impending lockup expiry predict the next twenty sessions?

Lockup expiry is the cleanest of the candidate sources: the date is fixed by rule when a
placement or an IPO is priced, so it is known long before it happens and no disclosure
lag can leak.  The factor is the share of float that unlocks inside the next twenty
sessions, and the expected sign is negative because the unlocking holders are the ones
under pressure to sell.

Two measurements are reported, and the second is the one that matters.  Most stock-dates
carry no event at all, so a rank IC computed over the whole cross-section is diluted by
construction; the same statistic restricted to the names that actually unlock says whether
the signal exists where it applies.  A source that only discriminates among five percent
of the pool is a different product from one that ranks everything, and the difference
belongs in the answer rather than in the footnotes.
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
CACHE = PROJECT_ROOT / "data" / "online" / "unlock"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "unlock_ic"
START, END = "2019-01-02", "2024-12-31"
HORIZON = 20
BAR = 0.031          # the repository's break-even IC at this horizon


def download(start="2019-01-01", end="2024-12-31", cache=CACHE):
    """Year by year, cached to CSV so a rerun costs nothing."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        path = cache / f"unlock_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"股票代码": str}))
            continue
        import akshare as ak
        low = max(start, f"{year}-01-01")
        high = min(end, f"{year}-12-31")
        frame = ak.stock_restricted_release_detail_em(start_date=low.replace("-", ""),
                                                      end_date=high.replace("-", ""))
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["date"] = pd.to_datetime(table["解禁时间"])
    table["code"] = table["股票代码"].astype(str).str.zfill(6)
    table["ratio"] = pd.to_numeric(table["占解禁前流通市值比例"], errors="coerce")
    return table[["date", "code", "ratio"]].dropna()


def forward_matrix(events, index, codes, horizon=HORIZON):
    """Share of float unlocking in ``(t, t+horizon]`` for every stock-date."""
    matrix = pd.DataFrame(0.0, index=index, columns=codes)
    for row in events.itertuples(index=False):
        if row.code not in matrix.columns:
            continue
        later = index[(index >= row.date - pd.Timedelta(days=horizon * 2)) & (index < row.date)]
        for date in later:
            matrix.at[date, row.code] += row.ratio
    return matrix


def measure(panel_dir=PANEL, bar=BAR, cache=CACHE):
    manifest, panel, prices = load_panel(panel_dir)
    codes = list(prices.columns)
    index = prices.index[(prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(END))]
    events = download(cache=cache)
    ratio = forward_matrix(events, index, codes)
    labels = (prices.shift(-HORIZON) / prices - 1.0).reindex(index)
    decision = list(index[::HORIZON])

    rows = []
    for date in decision:
        if date not in ratio.index or date not in labels.index:
            continue
        signal = ratio.loc[date].to_numpy(dtype=float)
        target = labels.loc[date].to_numpy(dtype=float)
        rows.append(dict(date=date, signal=signal, target=target))
    if not rows:
        return dict(status="no_dates")

    def ic(restrict):
        series, coverage = {}, []
        for row in rows:
            signal, target = row["signal"], row["target"]
            usable = np.isfinite(signal) & np.isfinite(target)
            if restrict:
                usable &= signal > 0
            coverage.append(float((signal > 0)[np.isfinite(target)].mean()) if np.isfinite(target).any() else 0.0)
            if usable.sum() < 10:
                series[row["date"]] = np.nan
                continue
            a = pd.Series(signal[usable]).rank().to_numpy()
            b = pd.Series(target[usable]).rank().to_numpy()
            series[row["date"]] = float(np.corrcoef(a, b)[0, 1])
        clean = pd.Series(series).dropna()
        stats = alpha_ic.ic_stats(clean, HORIZON, threshold=bar) if len(clean) else {}
        return dict(dates=int(len(clean)), mean_ic=stats.get("mean_ic"), t=stats.get("t_nw_h"),
                    hit=float((clean < 0).mean()) if len(clean) else None,
                    coverage=float(np.mean(coverage)))

    return dict(status="ready", panel=str(panel_dir), codes=len(codes), horizon=HORIZON,
                events=int(len(events)), bar=bar,
                full=ic(False), conditioned=ic(True),
                verdict=dict(
                    direction_negative=bool((ic(False).get("mean_ic") or 0) < 0),
                    clears_bar=bool(abs(ic(True).get("mean_ic") or 0) >= bar),
                    note="conditioned IC is measured only where an unlock falls in the window"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank IC of the lockup-expiry factor.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(args.panel)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "unlock_ic.json", payload)
    print(json.dumps({k: payload.get(k) for k in ("full", "conditioned", "verdict", "events")},
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
