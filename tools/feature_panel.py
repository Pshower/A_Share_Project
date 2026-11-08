"""Build a feature panel for the names a 3-million account can actually hold.

``research_v3`` carries the twenty features for 298 large-cap survivors only.  Any test
of a cross-sectional signal needs a wider pool, because the spread the signal has to
exploit is what makes it visible at all: the expected extreme z of a top-thirty basket is
about 1.76 out of 298 names and about 2.76 out of 4,000.

Features are computed by the repository's own ``DataPreprocessor.add_features`` from the
raw downloaded bars, so the definitions are the ones the rest of the project uses.  The
panel is written to its own directory; ``research_v3`` is never touched, because
``run_suite`` verifies its artifact hashes.

Nothing here decides which features are worth anything.  Selection is a separate step and
must be done on one window and checked on another, or it is just fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import DataPreprocessor, FEATURES, file_hash, write_json
from tools import capm_backtest as capm
from tools.full_market_panel import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY = PROJECT_ROOT / "data" / "online" / "history"
PANEL = PROJECT_ROOT / "data" / "clean" / "panel_full"
OUTPUT = PROJECT_ROOT / "data" / "clean" / "feature_top"
START, END = "2017-01-03", "2024-12-31"
LOOKBACK = 60          # the longest rolling window any feature needs


def newest_snapshot(code):
    """The most recent downloaded file for one code, ready or not."""
    candidates = sorted(HISTORY.glob(f"h_*/hfq/{code}.csv"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def read_bars(code):
    path = newest_snapshot(code)
    if path is None:
        return None
    frame = pd.read_csv(path, dtype={"code": str})
    if "date" not in frame or len(frame) < LOOKBACK:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    needed = {"open", "close", "high", "low", "volume", "turnover"}
    if not needed <= set(frame.columns):
        return None
    frame["valid_price"] = np.isfinite(frame[["open", "close", "high", "low"]].astype(float)).all(axis=1)
    return frame


def select_codes(count=1800, panel_dir=PANEL):
    """The ``count`` largest names by median float capitalisation."""
    manifest, prices, traded, shares = load(panel_dir)
    capitalisation = (traded * shares).median().sort_values(ascending=False)
    return list(capitalisation.index[:count]), capitalisation


def build(count=1800, output=OUTPUT, panel_dir=PANEL, log=print):
    codes, capitalisation = select_codes(count, panel_dir)
    blocks, closes, kept, skipped = {}, {}, [], []
    for code in codes:
        bars = read_bars(code)
        if bars is None:
            skipped.append(code)
            continue
        try:
            features = DataPreprocessor.add_features(bars, [])
        except (KeyError, ValueError):
            skipped.append(code)
            continue
        window = features.loc[START:END]
        if window.empty or window.dropna(how="all").empty:
            skipped.append(code)
            continue
        blocks[code] = window
        closes[code] = bars["close"].loc[START:END]
        kept.append(code)
    if not kept:
        raise ValueError("No code produced features")
    panel = pd.concat(blocks, axis=1)
    panel.columns = pd.MultiIndex.from_tuples([(code, name) for code, name in panel.columns])
    prices = pd.DataFrame(closes).sort_index()
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(target / "panel_data.parquet")
    prices.to_parquet(target / "prices.parquet")
    manifest = dict(
        version=str(target.name), start_date=START, end_date=END,
        stock_codes=kept, feature_cols=list(FEATURES), price_basis="hfq",
        train_end="2022-08-05", val_end="2023-10-20",
        build_config=dict(raw_dir=str(HISTORY), raw_pattern="h_*/hfq/{code}.csv",
                          source="data/online/history", requested=count),
        skipped=sorted(skipped),
        artifacts={name: file_hash(target / name)
                   for name in ("panel_data.parquet", "prices.parquet")},
    )
    write_json(target / "manifest.json", manifest)
    log("built %d codes (%d skipped), %d features, %d sessions" % (
        len(kept), len(skipped), len(FEATURES), panel.index.size))
    return manifest


def load_panel(directory=OUTPUT):
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(Path(directory) / "panel_data.parquet")
    prices = pd.read_parquet(Path(directory) / "prices.parquet")
    return manifest, panel, prices


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a feature panel for a given universe size.")
    parser.add_argument("--count", type=int, default=1800)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    manifest = build(args.count, args.output)
    print(json.dumps({k: manifest[k] for k in ("version", "stock_codes", "feature_cols")}, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
