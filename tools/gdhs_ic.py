"""Does a falling shareholder count predict the next twenty sessions?

The 股东户数 is the one fundamentals-adjacent series in this repository that is **not** a
financial statement: it is a register snapshot, and the interface carries the actual
announcement date next to it, so point-in-time use costs nothing but a sort.

Two decisions are fixed before the data is touched:

**Which quantity.** ``股东户数-增减比例`` against the previous snapshot -- the percentage
change in the number of holders.  The level (how many holders a stock has) is mostly a
function of its price and float; the change is the part that carries information.

**Which direction.** **Negative** -- holders falling means the float is being concentrated,
which the A-share folklore reads as accumulation.  Written here, before the run, because
"holder count" has no mechanical sign the way a lockup expiry does.

Availability is the **announcement date**, never the statistics cut-off: the register is
counted at quarter end and published six to eight weeks later.  A value is used from its
announcement day and expires after ``MAX_AGE`` sessions, so a stock that stopped disclosing
does not carry a stale number forever.
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
CACHE = PROJECT_ROOT / "data" / "online" / "gdhs" / "gdhs.parquet"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "gdhs_ic"
START, END = "2019-01-02", "2024-12-31"
HORIZON = 20
BAR = 0.031
T_BAR = 2.5
MAX_AGE = 250          # four quarters of sessions; older than that is not news
QUARTERS = ["2018-12-31", "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
            "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
            "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
            "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
            "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
            "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]


def download(cache=CACHE, log=print):
    """One call per quarter; the interface returns the whole market at once."""
    import akshare as ak
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        existing = pd.read_parquet(cache)
    else:
        existing = pd.DataFrame(columns=["code", "announced", "change_pct", "cutoff"])
    done = set(existing["cutoff"]) if len(existing) else set()
    frames = [existing]
    for quarter in QUARTERS:
        if quarter in done:
            continue
        frame = ak.stock_zh_a_gdhs(symbol=quarter.replace("-", ""))
        if frame is None or not len(frame):
            continue
        data = pd.DataFrame(dict(
            code=frame["代码"].astype(str).str.zfill(6),
            announced=pd.to_datetime(frame["公告日期"]),
            change_pct=pd.to_numeric(frame["股东户数-增减比例"], errors="coerce"),
            cutoff=quarter))
        frames.append(data.dropna(subset=["announced", "change_pct"]))
        log("%s: %d rows" % (quarter, len(data)))
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code", "cutoff"], keep="last")
    merged = merged.sort_values(["code", "announced"]).reset_index(drop=True)
    merged.to_parquet(cache, index=False)
    return merged


def asof_matrix(events, index, codes, max_age=MAX_AGE):
    """Latest announced change per stock as of each decision date, expiring when stale."""
    values = pd.DataFrame(np.nan, index=index, columns=codes)
    dated = {code: group for code, group in events.groupby("code")}
    for code, group in dated.items():
        if code not in values.columns:
            continue
        group = group.sort_values("announced")
        announced = group["announced"].to_numpy(dtype="datetime64[ns]")
        changes = group["change_pct"].to_numpy(dtype=float)
        position = np.searchsorted(announced, index.to_numpy(dtype="datetime64[ns]"), side="right") - 1
        for row, place in enumerate(position):
            if place < 0:
                continue
            age = (index[row] - pd.Timestamp(announced[place])).days
            if age <= max_age * 1.45:                 # sessions to calendar days
                values.iat[row, values.columns.get_loc(code)] = changes[place]
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
        universe = np.isfinite(target)
        coverage.append(float(usable.sum() / universe.sum()) if universe.any() else 0.0)
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
                events=int(len(events)), bar=bar, t_bar=T_BAR, full=full,
                verdict=dict(
                    direction_negative=bool((full.get("mean_ic") or 0) < 0),
                    clears_bar=bool(abs(full.get("mean_ic") or 0) >= bar),
                    clears_t=bool(abs(full.get("t") or 0) >= T_BAR),
                    coverage=full.get("coverage"),
                    passes=bool((full.get("mean_ic") or 0) < 0
                                and abs(full.get("mean_ic") or 0) >= bar
                                and abs(full.get("t") or 0) >= T_BAR)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank IC of the shareholder-count change.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(Path(args.panel))
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "gdhs_ic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                                         encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
