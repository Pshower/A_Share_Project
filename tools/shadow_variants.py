"""Parallel shadow books: one frozen rule, four frozen weightings, recorded side by side.

The recorder next door writes the E1 book and nothing else, and its construction may not be touched.
This module answers a different question -- "does the weighting matter?" -- without spending any
sample: every weighting is fixed here, in advance, and each gets its own append-only ledger.  A
variant that needs a weight tuned after the fact is a new pre-registration, not a variant.

A is the control: twenty equal-weighted features, which must reproduce ``e1_rule.composite``
exactly.  If A ever disagrees with the main ledger on a date, the harness is wrong, not the market.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import e1_rule as rule                                        # noqa: E402
from tools import shadow_run                                             # noqa: E402

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT = PROJECT_ROOT / "reports" / "shadow" / "variants"

# The twenty features the E1 protocol froze.  Kept as a literal so a panel that silently drops a
# feature column cannot change a variant's definition.
FEATURES = ("return", "log_return", "range_ratio", "body_ratio", "volume_ratio", "turnover",
            "rsi_14", "macd_ratio", "macd_signal_ratio", "bb_position", "momentum_5",
            "close_ma_ratio_5", "momentum_10", "close_ma_ratio_10", "momentum_20",
            "close_ma_ratio_20", "momentum_60", "close_ma_ratio_60", "volatility_20",
            "volatility_60")

VARIANTS = {
    "A": dict(label="20 特征等权（现行构造）",
              spec=[(name, 1.0 / len(FEATURES)) for name in FEATURES]),
    "B": dict(label="反转两因子（1 日 + 5 日）",
              spec=[("return", 0.5), ("momentum_5", 0.5)]),
    "C": dict(label="反转 + 低波（3 因子）",
              spec=[("return", 0.4), ("momentum_5", 0.4), ("volatility_20", 0.2)]),
    "D": dict(label="随机对照（固定种子，不看特征）", spec=None),
}

TOP_K, BUFFER, HOLD = rule.TOP_K, rule.BUFFER, shadow_run.HOLD


def _percentile(block, name, codes):
    try:
        values = block.xs(name, level=1).reindex(codes).to_numpy(dtype=float)
    except KeyError:
        return None
    if not np.isfinite(values).any():
        return None
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def blend(panel, codes, spec, date):
    """Weighted mean of cross-sectional percentiles; lower is the buy.

    Every feature is used with a positive sign because the direction is built into the feature:
    a high 1-day return or a high volatility is "worse" for a reversal book, so its high percentile
    carries a high score and the ranking puts it at the sell end.
    """
    block = panel.loc[date]
    total, weight = np.zeros(len(codes)), 0.0
    for name, share in spec:
        percentiles = _percentile(block, name, codes)
        if percentiles is None:
            continue
        total += share * np.nan_to_num(percentiles, nan=0.5)
        weight += share
    if not weight:
        return None
    return pd.Series(total / weight, index=codes)


def random_score(codes, date):
    """A deterministic control: no feature information, but the same book mechanics."""
    stamp = str(date)
    values = [zlib.crc32((stamp + str(code)).encode("utf-8")) / 0xFFFFFFFF for code in codes]
    return pd.Series(values, index=codes)


def score_for(letter, panel, codes, date):
    variant = VARIANTS[letter]
    if variant["spec"] is None:
        return random_score(codes, date)
    return blend(panel, codes, variant["spec"], date)


def paths(letter, output=OUTPUT):
    base = Path(output)
    return base / ("%s_shadow.jsonl" % letter), base / ("%s_state.json" % letter)


def load_state(letter, output=OUTPUT):
    _, state_path = paths(letter, output)
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(letter, state, output=OUTPUT):
    """Write through a temp file: a half-written state would wedge the ledger on the next run."""
    _, state_path = paths(letter, output)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_name(state_path.name + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, state_path)


def append(letter, record, output=OUTPUT):
    log_path, _ = paths(letter, output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_one(letter, dataset, force=False, output=OUTPUT, log=print):
    directory = Path(dataset)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    codes = [code for code in manifest["stock_codes"]
             if code in set(panel.columns.get_level_values(0))]
    if len(codes) < 0.9 * len(manifest["stock_codes"]):
        record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), variant=letter,
                      status="bad_data", panel_date=str(panel.index.max().date()),
                      reason="panel only has %d of %d codes" % (len(codes), len(manifest["stock_codes"])))
        append(letter, record, output)
        log("%s bad_data" % letter)
        return record

    latest = panel.index.max()
    state = load_state(letter, output)
    if state.get("last_panel_date") == str(latest.date()) and not force:
        record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), variant=letter,
                      status="stale", panel_date=str(latest.date()),
                      reason="panel has not advanced since the last run; no decision recorded")
        append(letter, record, output)
        log("%s stale" % letter)
        return record

    score = score_for(letter, panel, codes, latest)
    if score is None:
        record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), variant=letter,
                      status="no_score", panel_date=str(latest.date()),
                      reason="no usable feature at the last date")
        append(letter, record, output)
        log("%s no_score" % letter)
        return record

    held = state.get("holdings", [])
    sessions = shadow_run.sessions_between(panel.index, state.get("entry_date"), latest)
    due = state.get("entry_date") is None or (sessions is not None and sessions >= HOLD)
    book = rule.decide(score, held, top_k=TOP_K, buffer=BUFFER) if due \
        else sorted(held, key=lambda code: float(score.get(code, float("inf"))))

    close = prices.xs("close", axis=1, level=1) if "close" in set(prices.columns.get_level_values(1)) \
        else prices.xs(prices.columns.get_level_values(1)[0], axis=1, level=1)
    record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), variant=letter,
                  status="decided", panel_date=str(latest.date()), book=book,
                  score={code: round(float(score[code]), 6) for code in book if code in score},
                  nav=shadow_run.mark(close, book, latest),
                  held_from=state.get("entry_date"),
                  decision="rebalance" if due else "hold", rebalance_due=bool(due),
                  construction=VARIANTS[letter]["label"])
    if held and state.get("last_panel_date"):
        record["interval"] = dict(
            **{"from": state.get("last_panel_date"), "to": str(latest.date()),
               "sessions": shadow_run.sessions_between(panel.index, state.get("last_panel_date"),
                                                       latest),
               "return": shadow_run.basket_return(close, held, state.get("last_panel_date"), latest)})
        record["benchmark"] = shadow_run.benchmarks(close, codes, state.get("last_panel_date"),
                                                    str(latest.date()))
    append(letter, record, output)
    save_state(letter, dict(status="running", last_panel_date=str(latest.date()), holdings=book,
                            entry_date=str(latest.date()) if due else state.get("entry_date"),
                            runs=int(state.get("runs", 0)) + 1), output)
    log("%s %s %s: %d names" % (letter, "rebalance" if due else "hold", latest.date(), len(book)))
    return record


def run(variants="ABCD", dataset=None, force=False, output=OUTPUT, log=print):
    dataset = Path(dataset) if dataset else (shadow_run.latest_panel() or shadow_run.DATASET)
    letters = [letter for letter in variants.upper() if letter in VARIANTS]
    unknown = [letter for letter in variants.upper() if letter not in VARIANTS]
    if unknown:
        raise ValueError("unknown variant(s): %s" % ", ".join(unknown))
    return [run_one(letter, dataset, force=force, output=output, log=log) for letter in letters]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record the parallel weightings for one session.")
    parser.add_argument("--variants", default="ABCD")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    records = run(args.variants, args.dataset, force=args.force, output=Path(args.output))
    print(json.dumps([{k: v for k, v in record.items() if k != "score"} for record in records],
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
