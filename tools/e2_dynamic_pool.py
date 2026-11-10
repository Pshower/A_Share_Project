"""E2: the E1 signal on a **point-in-time** top-1,800 pool.

[The protocol](doc/e2_protocol_20261108.md) allows exactly one thing to change relative to
E1 -- the universe -- and fixes it as "the 1,800 largest by the *previous* session's float
capitalisation", with an explicit ban on choosing the members from a whole-period ranking.
The panel on disk (``feature_top1800``) does the banned thing: it takes the largest names
by *median* capitalisation, so membership quietly uses the future.

This tool builds the pool the protocol actually describes, and nothing else:

* membership is ranked on capitalisation lagged one session, re-derived every session;
* a name's features are set to missing on every session it is outside that day's pool, so
  the environment's own validity rule (all features finite) excludes it -- no custom
  backtest, the same ``run_policy`` and broker E1 used;
* the union of every name that was ever a member is what gets a column, so the pool can
  breathe without rebuilding the panel;
* the twenty features come from the repository's ``DataPreprocessor.add_features``, not
  from a second implementation of the definitions.

Nothing here decides anything.  The verdict belongs to ``tools/e2_run.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import FEATURES, DataPreprocessor, file_hash, write_json
from tools import full_market_panel as fmp
from tools.feature_panel import read_bars

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_FULL = PROJECT_ROOT / "data" / "clean" / "panel_full"
OUTPUT = PROJECT_ROOT / "data" / "clean" / "e2_dynamic1800"
START, END = "2017-01-02", "2024-12-31"
TOP = 1800


def capitalisation(panel_dir=PANEL_FULL, start=START, end=END):
    """Float capitalisation, unadjusted price times free float, in the panel's own dates."""
    _, _, unadjusted, shares = fmp.load(panel_dir)
    return (unadjusted * shares).sort_index().loc[start:end]


def membership(cap, top=TOP):
    """Boolean frame: is the code inside the top ``top`` by the **previous** session's cap?

    Ranking on the lagged value is the whole point of the protocol.  A missing lagged cap
    (not yet listed, or suspended the day before) is not a member: absent evidence is not
    membership.
    """
    lagged = cap.sort_index().ffill().shift(1)
    ranks = lagged.rank(axis=1, ascending=False, method="first")
    return ranks.le(top).fillna(False)


def build(output=OUTPUT, top=TOP, start=START, end=END, log=print):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to rewrite an existing dataset: %s" % output)
    cap = capitalisation(start=start, end=end)
    member = membership(cap, top=top)
    union = [code for code in member.columns if bool(member[code].any())]
    dates = cap.index
    log("pool: %d sessions, top %d, union %d codes" % (len(dates), top, len(union)))
    blocks, closes, opens, volumes, kept, skipped = {}, {}, {}, {}, [], []
    for position, code in enumerate(union):
        bars = read_bars(code)
        if bars is None:
            skipped.append(code)
            continue
        try:
            features = DataPreprocessor.add_features(bars, [])
        except (KeyError, ValueError):
            skipped.append(code)
            continue
        window = features.reindex(dates).loc[start:end]
        if window.dropna(how="all").empty:
            skipped.append(code)
            continue
        inside = pd.Series(member[code].reindex(window.index).fillna(False).to_numpy(dtype=bool),
                           index=window.index)
        blocks[code] = window.where(inside, axis=0)           # outside the pool: no features
        closes[code] = bars["close"].reindex(window.index)
        opens[code] = bars["open"].reindex(window.index)
        volumes[code] = bars["volume"].reindex(window.index)
        kept.append(code)
        if position and position % 500 == 0:
            log("  %d/%d codes" % (position, len(union)))
    if not kept:
        raise ValueError("No code produced features")
    panel = pd.concat(blocks, axis=1)
    panel.columns = pd.MultiIndex.from_tuples([(code, name) for code, name in panel.columns])
    prices = pd.concat({field: pd.DataFrame(frame) for field, frame in
                        (("close", closes), ("open", opens), ("volume", volumes))}, axis=1)
    prices = prices.swaplevel(0, 1, axis=1).sort_index(axis=1)
    prices.columns.names = ["code", "feature"]
    output.mkdir(parents=True)
    panel.to_parquet(output / "panel_data.parquet")
    prices.to_parquet(output / "prices.parquet")
    manifest = dict(version=output.name, schema_version=2, price_basis="hfq_research",
                    feature_cols=list(FEATURES), stock_codes=kept,
                    start_date=str(dates.min().date()), end_date=str(dates.max().date()),
                    train_end=str(dates.min().date()), val_end=str(dates.max().date()),
                    skipped=skipped,
                    pool=dict(top=top, ranked_on="float capitalisation, lagged one session",
                              union=len(union), members_mean=float(member.sum(axis=1).mean()),
                              overlaps_feature_top1800=None),
                    build_config=dict(raw_pattern="h_*/hfq/{code}.csv", source="data/online/history",
                                      panel_full=str(PANEL_FULL.relative_to(PROJECT_ROOT)).replace("\\", "/")),
                    artifacts={name: file_hash(output / name) for name in ("panel_data.parquet", "prices.parquet")})
    write_json(output / "manifest.json", manifest)
    log("built %s: %d codes, %d sessions" % (output.name, len(kept), panel.index.size))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the E2 point-in-time top-1800 panel.")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--top", type=int, default=TOP)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args(argv)
    manifest = build(args.output, args.top, args.start, args.end)
    print(json.dumps({k: manifest[k] for k in ("version", "stock_codes", "pool")}, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
