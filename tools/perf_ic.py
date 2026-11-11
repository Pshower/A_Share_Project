"""Does reported profit growth predict the next twenty sessions, at a point in time?

Earnings reports are the clearest example of why this repository ranks sources by PIT quality
before testing them: the number is measured at the period end and published weeks later, and
a backtest that uses the period end is reading the future.

**What the interface actually gives.** ``stock_yjbb_em`` carries a ``最新公告日期`` column, but
it is not the first announcement: for the 2024 Q1 period it puts 5,440 of 5,889 rows in April
**2025**, i.e. the date of the later filing that superseded the row.  Using it would delay
every signal by a year and turn the test into a different question.

**What this file uses instead.** The statutory deadline for each report period -- April 30 for
the annual and Q1 reports, August 31 for the half year, October 31 for Q3.  A report cannot be
published after its deadline, so a signal dated at the deadline can never look ahead; the price
is that the early announcers are treated as if they disclosed last.  That is the conservative
side of the trade-off, and it is the only anchor the interface supports.

**The two decisions, fixed before the data is touched.**  The quantity is ``净利润-同比增长`` --
the year-on-year growth rate of net profit, already a ratio and already signed, so no choice of
encoding is left to be made after the fact.  The direction is **positive**: an accelerating
company keeps drifting up (PEAD).  The criterion is the repository's: rank IC at least 0.031
with |t| at least 2.5.
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
CACHE = PROJECT_ROOT / "data" / "online" / "yjbb" / "yjbb.parquet"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "perf_ic"
START, END = "2019-01-02", "2024-12-31"
HORIZON = 20
BAR, T_BAR = 0.031, 2.5
MAX_AGE = 250
QUARTERS = [f"{year}-{month}" for year in range(2018, 2025) for month in ("03-31", "06-30", "09-30", "12-31")]

# Statutory disclosure deadlines: a report for the period ending on the key cannot be
# published after the value, so the value is the earliest date the number is certainly public.
DEADLINES = {"03-31": ("04-30", 0), "06-30": ("08-31", 0), "09-30": ("10-31", 0), "12-31": ("04-30", 1)}


def deadline_for(period):
    """The date on which a report for ``period`` is certainly public."""
    stamp = pd.Timestamp(period)
    month_day = stamp.strftime("%m-%d")
    target, year_offset = DEADLINES[month_day]
    return pd.Timestamp(year=stamp.year + year_offset, month=int(target[:2]), day=int(target[3:]))


def download(cache=CACHE, log=print):
    import akshare as ak
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(
        columns=["code", "period", "value", "available"])
    done = set(existing["period"]) if len(existing) else set()
    frames = [existing]
    for period in QUARTERS:
        if period in done:
            continue
        frame = ak.stock_yjbb_em(date=period.replace("-", ""))
        if frame is None or not len(frame):
            continue
        data = pd.DataFrame(dict(
            code=frame["股票代码"].astype(str).str.zfill(6),
            period=period,
            value=pd.to_numeric(frame["净利润-同比增长"], errors="coerce"),
            available=deadline_for(period)))
        frames.append(data.dropna(subset=["value"]))
        log("%s: %d rows" % (period, len(data)))
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code", "period"], keep="last")
    merged = merged.sort_values(["code", "available"]).reset_index(drop=True)
    merged.to_parquet(cache, index=False)
    return merged


def asof_matrix(events, index, codes, max_age=MAX_AGE):
    values = pd.DataFrame(np.nan, index=index, columns=codes)
    for code, group in events.groupby("code"):
        if code not in values.columns:
            continue
        group = group.sort_values("available")
        available = group["available"].to_numpy(dtype="datetime64[ns]")
        numbers = group["value"].to_numpy(dtype=float)
        position = np.searchsorted(available, index.to_numpy(dtype="datetime64[ns]"), side="right") - 1
        for row, place in enumerate(position):
            if place < 0:
                continue
            if (index[row] - pd.Timestamp(available[place])).days <= max_age * 1.45:
                values.iat[row, values.columns.get_loc(code)] = numbers[place]
    return values


def measure(panel_dir=PANEL, cache=CACHE, bar=BAR):
    manifest, panel, prices = load_panel(panel_dir)
    codes = list(prices.columns)
    index = prices.index[(prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(END))]
    events = download(cache=cache, log=lambda *_: None)
    decision = list(index[::HORIZON])
    signal = asof_matrix(events, pd.DatetimeIndex(decision), codes)
    labels = (prices.shift(-HORIZON) / prices - 1.0).reindex(index)
    series, coverage = {}, []
    for date in decision:
        if date not in labels.index:
            continue
        values = signal.loc[date].to_numpy(dtype=float)
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
                hit=float((clean > 0).mean()) if len(clean) else None,
                coverage=float(np.mean(coverage)) if coverage else None)
    return dict(status="ready", panel=str(panel_dir), codes=len(codes), horizon=HORIZON,
                events=int(len(events)), bar=bar, t_bar=T_BAR, full=full,
                pit=dict(anchor="statutory deadline", note="最新公告日期 is the superseding filing, not the first"),
                verdict=dict(direction_positive=bool((full.get("mean_ic") or 0) > 0),
                             clears_bar=bool(abs(full.get("mean_ic") or 0) >= bar),
                             clears_t=bool(abs(full.get("t") or 0) >= T_BAR),
                             passes=bool((full.get("mean_ic") or 0) > 0
                                         and abs(full.get("mean_ic") or 0) >= bar
                                         and abs(full.get("t") or 0) >= T_BAR)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank IC of reported profit growth, on a PIT anchor.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(Path(args.panel))
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "perf_ic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                                         encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
