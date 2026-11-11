"""Does excluding crowded-leverage names improve the book it is applied to?

[The protocol](doc/crowding_overlay_protocol_20261111.md) fixes the question: not "can this
signal pick stocks" (answered no) but "does the same book do better without the names whose
financing balance grew fastest".  The overlay blanks those names' features for the day, so the
environment's own validity rule drops them and the frozen book rule fills from the next rank --
no second stock-picking implementation and no new buffer logic.

Both arms run the same code on the same dates through the same broker; only the panel differs,
and it differs in memory so nothing half-written is left on disk.  The comparison is the paired
block difference the protocol registered, not a comparison of two separate backtest summaries.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools import e1_reversal, margin_ic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "clean" / "e2_dynamic1800"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "crowding_overlay"
THRESHOLD = 0.90
HOLD = e1_reversal.HOLD


def crowded_mask(events, index, codes, threshold=THRESHOLD):
    """True where the lagged margin growth sits in the top ``1 - threshold`` of the pool."""
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    balance = margin_ic.balance_matrix(events, index, codes)
    growth = margin_ic.growth_matrix(balance)
    return growth.rank(axis=1, pct=True).ge(threshold)


def blank_crowded(panel, mask):
    """Return the panel with every feature of a crowded stock-day missing."""
    codes = list(mask.columns)
    position = {code: place for place, code in enumerate(codes)}
    columns = panel.columns.get_level_values(0)
    picked = np.array([position.get(code, -1) for code in columns])
    blocked = np.zeros(panel.shape, dtype=bool)
    known = picked >= 0
    blocked[:, known] = mask.to_numpy()[:, picked[known]]
    values = panel.to_numpy(dtype="float64", copy=True)
    values[blocked] = np.nan
    return pd.DataFrame(values, index=panel.index, columns=panel.columns)


def daily_returns(panel, prices, manifest):
    """Run the frozen book on one panel and return its daily return series."""
    settings = dict(initial_capital=e1_reversal.CAPITAL, seed=42, lookback=1,
                    rebalance_every=HOLD, momentum_top_k=e1_reversal.TOP_K,
                    broker=dict(e1_reversal.BROKER))
    market = e1_reversal.build_market(manifest, panel, prices,
                                      manifest["start_date"], manifest["end_date"], 1)
    strategy = e1_reversal.ReversalBook(
        manifest["feature_cols"],
        [str(code) for code in prices.columns.get_level_values(0).unique()])
    daily, _, _, _ = e1_reversal.run_policy(market, strategy, settings)
    _, _, returns, _ = e1_reversal.curve(daily)
    return returns


def block_pairs(left, right, horizon=HOLD):
    """Non-overlapping block returns of two daily series, paired by block."""
    joined = pd.concat([left.rename("overlay"), right.rename("base")], axis=1).dropna()
    stamps = list(joined.index)
    rows = []
    for start in range(0, len(stamps) - horizon + 1, horizon):
        window = joined.iloc[start:start + horizon]
        rows.append(dict(start=stamps[start],
                         overlay=float((1 + window["overlay"]).prod() - 1),
                         base=float((1 + window["base"]).prod() - 1)))
    frame = pd.DataFrame(rows)
    frame["diff"] = frame["overlay"] - frame["base"]
    return frame


def paired(frame):
    values = frame["diff"].dropna()
    if len(values) < 6:
        return dict(status="insufficient", blocks=int(len(values)))
    mean = float(values.mean())
    spread = float(values.std(ddof=1))
    t = float(mean / (spread / math.sqrt(len(values)))) if spread > 0 else float("nan")
    return dict(status="ready", blocks=int(len(values)), mean_block=mean,
                annualised=float((1 + mean) ** (242.0 / HOLD) - 1), t=t,
                positive_share=float((values > 0).mean()),
                base_annual=float((1 + frame["base"].mean()) ** (242.0 / HOLD) - 1),
                overlay_annual=float((1 + frame["overlay"].mean()) ** (242.0 / HOLD) - 1))


def run(base=BASE, output=OUTPUT, threshold=THRESHOLD, log=print):
    directory = Path(base)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    codes = [str(code) for code in prices.columns.get_level_values(0).unique()]
    events = pd.read_parquet(margin_ic.CACHE)
    mask = crowded_mask(events, panel.index, codes, threshold).reindex(
        index=panel.index, columns=codes).fillna(False)
    log("crowded: %.1f%% of pool-days are blanked" % (100.0 * float(mask.to_numpy().mean())))
    log("running the base book ...")
    base_returns = daily_returns(panel, prices, manifest)
    log("running the overlay book ...")
    overlay_returns = daily_returns(blank_crowded(panel, mask), prices, manifest)
    frame = block_pairs(overlay_returns, base_returns)
    stats = paired(frame)
    stats["passed"] = bool(stats.get("status") == "ready"
                           and stats["mean_block"] > 0 and stats["t"] >= 2.5)
    return dict(status="ready", base=str(base), threshold=threshold, paired=stats,
                blocks=[dict(start=str(row["start"].date()), overlay=row["overlay"],
                             base=row["base"], diff=row["diff"])
                        for row in frame.to_dict("records")])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired test of the crowding overlay.")
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args(argv)
    payload = run(args.base, args.output, args.threshold)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "crowding_overlay.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(payload["paired"], ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
