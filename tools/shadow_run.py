"""S2: freeze E1's construction and record what it would decide, one day at a time.

The point is not to win a backtest. Every sample in this repository is spent -- V was
used on the exposure work, R was consumed by the September training, and E1 has just
spent the full window. A forward shadow run is the only evidence source left that costs
calendar time instead of sample.

The construction is copied from ``tools/e1_reversal.py`` and must not drift: twenty price
features all negated, rank equal weight, top thirty, replace only below the sixtieth
rank, twenty-session holds. What this tool adds is a dated record -- the book it would
hold, the score each name had, and the mark-to-market -- appended once per run and never
rewritten. Returns are recorded but not read: nothing here selects on the outcome.

A run that finds the panel has not advanced records that fact and stops. Recording a
decision on stale data would contaminate the only clean sample this project has.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The forward panel, not ``research_v3``: the frozen dataset ends 2024-12-31 and can never
# advance, so reading it only ever reported ``stale``.  ``research_v3`` is still the record
# of what E1 decided in-sample and is left untouched; this is the same twenty features on
# the same pool, captured every trading day from 2026-11-10 onward.  The construction
# constants below are not affected by the switch.
DATASET = PROJECT_ROOT / "data" / "clean" / "forward_pool_20261110"
STATE = PROJECT_ROOT / "reports" / "shadow" / "e1_state.json"
LOG = PROJECT_ROOT / "reports" / "shadow" / "e1_shadow.jsonl"
TOP_K = 30
BUFFER = 60
HOLD = 20


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return dict(status="new", last_panel_date=None, holdings=[], entry_date=None, runs=0)


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append(record):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def composite(panel, codes, features, date):
    """Negated rank-equal-weighted score on ``date``; lower is better."""
    block = panel.loc[date]
    ranks, usable = np.zeros(len(codes)), 0
    for name in features:
        try:
            values = block.xs(name, level=1).reindex(codes).to_numpy(dtype=float)
        except KeyError:
            continue
        if not np.isfinite(values).any():
            continue
        ordered = pd.Series(values).rank(method="average", pct=True).to_numpy()
        ranks += np.nan_to_num(ordered, nan=0.5)
        usable += 1
    if not usable:
        return None
    score = ranks / usable          # mean percentile; the smallest are the reversal buy
    return pd.Series(score, index=codes)


def decide(score, held, top_k=TOP_K, buffer=BUFFER):
    """The frozen rule: keep whatever is still inside the buffer, then fill."""
    order = score.sort_values(kind="stable")
    rank_of = {code: position for position, code in enumerate(order.index)}
    keep = [code for code in held if rank_of.get(code, 10 ** 9) < buffer]
    keep = sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))[:top_k]
    for code in order.index:
        if len(keep) >= top_k:
            break
        if code not in keep:
            keep.append(code)
    return keep


def mark(panel_prices, holdings, date):
    """Equal-weight mark-to-market of the current book at ``date``."""
    prices = panel_prices.loc[:date]
    if prices.empty or not holdings:
        return None
    last = prices.iloc[-1]
    values = [float(last.get(code, np.nan)) for code in holdings]
    if not any(np.isfinite(values)):
        return None
    return dict(date=str(pd.Timestamp(date).date()), holdings=len(holdings),
                priced=int(np.isfinite(values).sum()),
                equal_weight_nav=float(np.nanmean(values)))


def run(force=False, dataset=DATASET, log=print):
    directory = Path(dataset)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    codes = [code for code in manifest["stock_codes"] if code in set(panel.columns.get_level_values(0))]
    features = list(manifest["feature_cols"])
    latest = panel.index.max()
    state = load_state()
    from tools import trading_calendar
    try:
        trading = trading_calendar.is_trading_day()
    except Exception:                                  # noqa: BLE001 - calendar is optional
        trading = True
    if not trading:
        # A holiday and a broken feed look identical from the ledger, so the calendar answers
        # the question before it is asked.  Nothing is appended: a non-trading day is not news.
        record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="nontrading",
                      panel_date=str(latest.date()),
                      reason="not a trading day in the exchange calendar; no decision recorded")
        log("nontrading: %s" % date.today())
        return record
    if state.get("last_panel_date") == str(latest.date()) and not force:
        record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="stale",
                      panel_date=str(latest.date()),
                      reason="panel has not advanced since the last run; no decision recorded")
        append(record)
        log("stale: panel still ends %s" % latest.date())
        return record
    score = composite(panel, codes, features, latest)
    if score is None:
        record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="no_score",
                      panel_date=str(latest.date()), reason="no usable feature at the last date")
        append(record)
        log("no score at %s" % latest.date())
        return record
    book = decide(score, state.get("holdings", []))
    close = prices.xs("close", axis=1, level=1) if "close" in set(prices.columns.get_level_values(1)) else None
    nav = mark(close if close is not None else prices.xs(prices.columns.get_level_values(1)[0], axis=1, level=1),
               book, latest)
    due = state.get("entry_date") is None or (
        (latest - pd.Timestamp(state["entry_date"])).days >= HOLD * 1.4)
    record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="decided",
                  panel_date=str(latest.date()), book=book,
                  score={code: round(float(score[code]), 6) for code in book},
                  nav=nav, held_from=state.get("entry_date"),
                  rebalance_due=bool(due), construction="e1 reversal v1")
    append(record)
    save_state(dict(status="running", last_panel_date=str(latest.date()),
                    holdings=book, entry_date=str(latest.date()) if due else state.get("entry_date"),
                    runs=int(state.get("runs", 0)) + 1))
    log("decided on %s: %d names" % (latest.date(), len(book)))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record one E1 shadow decision.")
    parser.add_argument("--force", action="store_true", help="Record even if the panel is unchanged.")
    parser.add_argument("--dataset", default=str(DATASET))
    args = parser.parse_args(argv)
    record = run(force=args.force, dataset=args.dataset)
    print(json.dumps({k: v for k, v in record.items() if k != "score"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
