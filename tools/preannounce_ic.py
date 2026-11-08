"""Does a recent earnings preannouncement predict the next twenty sessions?

Two decisions are fixed before the data is touched, because both are places where a
result can be manufactured after the fact:

**Which quantity.** The signal is ``业绩变动幅度``, the percentage change against the same
period a year earlier, taken from the most recent announcement in the last sixty sessions.
The categorical ``预告类型`` is not used: mapping eleven Chinese labels onto an order is a
choice, and every choice like it is a place to tune until the sign looks right.  The
percentage is already ordered and already signed.

**Which direction.** Positive, fixed here.  A company that tells the market earnings are
better than last year should drift up, which is the oldest result in this literature.

Names with no recent announcement are ranked neutral in the full cross-section and
dropped from the conditioned one, so that "issued no forecast" and "issued a good
forecast" cannot be confused with each other.
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
CACHE = PROJECT_ROOT / "data" / "online" / "preannounce"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "preannounce_ic"
START, END = "2019-01-02", "2024-12-31"
HORIZON = 20
RECENT = 60          # a preannouncement counts for sixty sessions after it is filed
BAR = 0.031


def periods(start_year=2018, end_year=2024):
    for year in range(start_year, end_year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            stamp = f"{year}{month:02d}{day:02d}"
            if "20181231" <= stamp <= "20240930":
                yield stamp


def download(cache=CACHE):
    """One call per reporting period, cached; announcement dates decide availability."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for period in periods():
        path = cache / f"yjyg_{period}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"股票代码": str}))
            continue
        import akshare as ak
        frame = ak.stock_yjyg_em(date=period)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["code"] = table["股票代码"].astype(str).str.zfill(6)
    table["filed"] = pd.to_datetime(table["公告日期"], errors="coerce")
    table["change"] = pd.to_numeric(table["业绩变动幅度"], errors="coerce")
    table = table.dropna(subset=["filed", "change"])
    return table[["code", "filed", "change"]].sort_values("filed")


def signal_matrix(events, index, codes, recent=RECENT):
    """Latest filed change within the trailing window, else NaN."""
    matrix = pd.DataFrame(np.nan, index=index, columns=codes)
    lookup = {code: position for position, code in enumerate(codes)}
    for code, group in events.groupby("code"):
        if code not in lookup:
            continue
        stamps = group["filed"].to_numpy()
        values = group["change"].to_numpy()
        for date in index:
            earlier = np.flatnonzero((stamps <= np.datetime64(date)) &
                                     (stamps > np.datetime64(date) - np.timedelta64(recent, "D")))
            if earlier.size:
                matrix.at[date, code] = values[earlier[-1]]
    return matrix


def measure(panel_dir=PANEL, bar=BAR, cache=CACHE):
    manifest, panel, prices = load_panel(panel_dir)
    codes = list(prices.columns)
    index = prices.index[(prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(END))]
    events = download(cache)
    signal = signal_matrix(events, index, codes)
    labels = (prices.shift(-HORIZON) / prices - 1.0).reindex(index)
    dates = list(index[::HORIZON])

    def ic(conditioned):
        series, coverage = {}, []
        for date in dates:
            raw = signal.loc[date].to_numpy(dtype=float)
            target = labels.loc[date].to_numpy(dtype=float)
            usable = np.isfinite(target)
            if conditioned:
                usable &= np.isfinite(raw)
            coverage.append(float((np.isfinite(raw) & np.isfinite(target)).sum() / max(np.isfinite(target).sum(), 1)))
            if usable.sum() < 20:
                series[date] = np.nan
                continue
            values = raw.copy()
            if not conditioned:
                values = np.where(np.isfinite(values), values, np.nanmedian(values) if np.isfinite(values).any() else 0.0)
            a = pd.Series(values[usable]).rank().to_numpy()
            b = pd.Series(target[usable]).rank().to_numpy()
            series[date] = float(np.corrcoef(a, b)[0, 1])
        clean = pd.Series(series).dropna()
        stats = alpha_ic.ic_stats(clean, HORIZON, threshold=bar) if len(clean) else {}
        return dict(dates=int(len(clean)), mean_ic=stats.get("mean_ic"), t=stats.get("t_nw_h"),
                    hit=float((clean > 0).mean()) if len(clean) else None,
                    coverage=float(np.mean(coverage)))

    full, conditioned = ic(False), ic(True)
    return dict(status="ready", panel=str(panel_dir), codes=len(codes), events=int(len(events)),
                horizon=HORIZON, bar=bar, direction="positive (fixed before the test)",
                full=full, conditioned=conditioned,
                verdict=dict(direction_positive=bool((full.get("mean_ic") or 0) > 0),
                             clears_bar=bool(abs(conditioned.get("mean_ic") or 0) >= bar)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank IC of the earnings preannouncement signal.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(args.panel)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "preannounce_ic.json", payload)
    print(json.dumps({k: payload.get(k) for k in ("full", "conditioned", "verdict", "events")},
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
