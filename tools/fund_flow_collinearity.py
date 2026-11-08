"""Is fund flow new information, or twenty-day reversal wearing a different hat?

The question the source list asked was whether the cross-sectional correlation between
fund flow and past return exceeds 0.5.  If it does, the factor is not a new input; it is
the reversal factor rescaled, and testing it would be testing the same thing twice.

Two constructions are compared on the same dates and the same names:

* **flow** -- the mean of ``主力净流入-净占比`` over the trailing twenty sessions, which is
  net main-force inflow as a percentage of turnover, so no float-capitalisation series is
  needed and no separate join can go wrong;
* **momentum** -- the trailing twenty-session return from the same feed's closing prices.

Both are taken from the archive rather than from the price panel, because the feed's window
is the last few months and the price panel stops in 2024.

A high correlation answers the question negatively for the factor.  A low one does not
answer it positively: it only means this particular rescaling is not the same series,
which is a much weaker statement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE = PROJECT_ROOT / "data" / "online" / "fund_flow"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "fund_flow_collinearity"
WINDOW = 20
THRESHOLD = 0.5


def load(store=STORE):
    frame = pd.read_parquet(Path(store) / "flow.parquet")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["code", "date"])


def factor_tables(frame, window=WINDOW):
    """Trailing mean flow and trailing return, per code, aligned on the same dates."""
    flow, momentum = {}, {}
    for code, group in frame.groupby("code"):
        series = group.set_index("date").sort_index()
        if len(series) < window + 2:
            continue
        flow[code] = series["main_pct"].rolling(window).mean()
        momentum[code] = series["close"].pct_change(window)
    return pd.DataFrame(flow), pd.DataFrame(momentum)


def correlation(flow, momentum, min_names=30):
    """Cross-sectional Pearson and rank correlation at each date."""
    rows = []
    shared = flow.index.intersection(momentum.index)
    for date in shared:
        a = flow.loc[date]
        b = momentum.loc[date]
        usable = a.notna() & b.notna()
        if usable.sum() < min_names:
            continue
        x, y = a[usable].to_numpy(dtype=float), b[usable].to_numpy(dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        rows.append(dict(date=date, names=int(usable.sum()),
                         pearson=float(np.corrcoef(x, y)[0, 1]),
                         rank=float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])))
    return pd.DataFrame(rows)


def summarise(table):
    if table.empty:
        return dict(status="empty")
    years = len(table) / 243
    block = {}
    for column in ("pearson", "rank"):
        values = table[column].dropna()
        mean = float(values.mean())
        standard = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
        block[column] = dict(mean=mean, sd=float(values.std(ddof=1)), t=mean / standard if standard else float("nan"),
                             share_above=float((values.abs() > THRESHOLD).mean()))
    return dict(status="ready", dates=int(len(table)), names_median=float(table["names"].median()),
                years=float(years), **block)


def measure(store=STORE):
    frame = load(store)
    flow, momentum = factor_tables(frame)
    table = correlation(flow, momentum)
    summary = summarise(table)
    verdict = None
    if summary.get("status") == "ready":
        mean_rank = summary["rank"]["mean"]
        verdict = dict(mean_rank_correlation=mean_rank,
                       threshold=THRESHOLD,
                       redundant=bool(abs(mean_rank) > THRESHOLD),
                       note="redundant means fund flow is the reversal factor rescaled and "
                            "should not be tested as a separate source")
    return dict(status=summary.get("status"), store=str(store),
                codes=int(frame["code"].nunique()),
                first_date=str(frame["date"].min().date()),
                last_date=str(frame["date"].max().date()),
                window=WINDOW, summary=summary, verdict=verdict,
                dates=[str(d.date()) for d in table["date"]] if not table.empty else [])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fund flow against twenty-day reversal.")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    payload = measure(args.store)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    from src.data.preprocess import write_json
    write_json(target / "collinearity.json", payload)
    print(json.dumps({k: payload.get(k) for k in ("codes", "first_date", "last_date", "summary", "verdict")},
                     ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
