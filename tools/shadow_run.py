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

from tools import e1_rule as rule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The forward panel, not ``research_v3``: the frozen dataset ends 2024-12-31 and can never
# advance, so reading it only ever reported ``stale``.  ``research_v3`` is still the record
# of what E1 decided in-sample and is left untouched; this is the same twenty features on
# the same pool, captured every trading day from 2026-11-10 onward.  The construction
# constants below are not affected by the switch.
#
# The default follows the newest panel on disk rather than naming one: the daily job renames
# its dataset every trading day and prunes the old ones, so a hard-coded path goes stale the
# first time it is pruned -- and then the tool reports ``stale`` for a reason that has nothing
# to do with the data.  ``--dataset`` still overrides this.
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
STATE = PROJECT_ROOT / "reports" / "shadow" / "e1_state.json"
LOG = PROJECT_ROOT / "reports" / "shadow" / "e1_shadow.jsonl"
TOP_K = 30
BUFFER = 60
HOLD = 20


def latest_panel(root=CLEAN_DIR):
    """The newest ``forward_pool_<date>`` on disk, or None when there is none."""
    panels = sorted(path for path in Path(root).glob("forward_pool_*") if path.is_dir())
    return panels[-1] if panels else None


DATASET = latest_panel() or (CLEAN_DIR / "forward_pool_20261110")


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
    """Negated rank-equal-weighted score on ``date``; lower is better.

    Delegates to ``tools.e1_rule`` so the recorder and the backtest cannot drift apart.
    """
    return rule.composite(panel, codes, features, date)


def decide(score, held, top_k=TOP_K, buffer=BUFFER):
    """The frozen rule: keep whatever is still inside the buffer, then fill.

    Same delegation as ``composite``: one implementation, two callers.
    """
    return rule.decide(score, held, top_k=top_k, buffer=buffer)


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


def sessions_between(index, start, end):
    """Trading sessions in the panel strictly after ``start``, up to and including ``end``."""
    if not start:
        return None
    return int(((index > pd.Timestamp(start)) & (index <= pd.Timestamp(end))).sum())


def basket_return(close, codes, start, end):
    """Equal-weight return of a basket between two dates, ignoring names without both prices."""
    if close is None or not codes or not start:
        return None
    window = close.loc[:end]
    if window.empty:
        return None
    ends = window.iloc[-1]
    starts = window.loc[:start]
    if starts.empty:
        return None
    begin = starts.iloc[-1]
    ratios = []
    for code in codes:
        first, last = begin.get(code, np.nan), ends.get(code, np.nan)
        if np.isfinite(first) and np.isfinite(last) and first > 0:
            ratios.append(float(last) / float(first) - 1.0)
    return float(np.mean(ratios)) if ratios else None


def index_close(symbol="sh000300", start=None, end=None):
    """Daily close of a benchmark index, or None when the feed is unreachable.

    The recorder must never fail because a benchmark is missing: a book without a comparison is
    still a recorded book, and ``null`` in the log is honest about the gap.
    """
    try:
        import akshare as ak

        frame = ak.stock_zh_index_daily(symbol=symbol)
        frame["date"] = pd.to_datetime(frame["date"])
        series = frame.set_index("date").sort_index()["close"].astype(float)
        if start:
            series = series.loc[series.index >= pd.Timestamp(start)]
        if end:
            series = series.loc[series.index <= pd.Timestamp(end)]
        return series
    except Exception:                                  # noqa: BLE001 - the feed is optional
        return None


def index_return(symbol, start, end):
    if not start or not end:
        return None
    series = index_close(symbol, start, end)
    if series is None or len(series) < 2:
        return None
    return float(series.iloc[-1] / series.iloc[0] - 1.0)


def benchmarks(close, pool_codes, start, end):
    """The two comparisons a shadow book needs: the pool it picks from, and the index."""
    return dict(**{"from": start, "to": end},
                pool_equal=basket_return(close, pool_codes, start, end),
                hs300=index_return("sh000300", start, end))


def stop_record(kind, latest, reason):
    """A guard's record: dated, honest, and shaped like every other row in the log."""
    return dict(recorded_at=datetime.now(timezone.utc).isoformat(), status=kind,
                panel_date=str(latest.date()), reason=reason)


def shared_guards(codes, expected, state, latest, force=False):
    """The guard chain every shadow book runs before it may record a decision.

    Returns ``None`` when the book may proceed, else a dict with the ``kind`` ("bad_data" /
    "nontrading" / "stale"), the ``record`` to return, and whether it should be ``append``ed.
    Two shadow recorders used to carry this chain as parallel copies and the wording had already
    drifted; one copy is the whole point of the merge (2026-11-23), and the parity test pins
    the two books together through it.

    ``bad_data`` -- a truncated panel would freeze a shrunken pool into the book; refuse.
    ``nontrading`` -- nothing is appended: a holiday is not news, from either ledger's view.
    ``stale`` -- the panel has not advanced; re-deciding on it would double-count a session.
    """
    if len(codes) < 0.9 * expected:
        return dict(kind="bad_data", append=True,
                    record=stop_record(
                        "bad_data", latest,
                        "panel only has %d of %d pool codes; a truncated download, not a decision"
                        % (len(codes), expected)))
    from tools import trading_calendar
    try:
        trading = trading_calendar.is_trading_day()
    except Exception:                                  # noqa: BLE001 - calendar is optional
        trading = True
    if not trading:
        return dict(kind="nontrading", append=False,
                    record=stop_record(
                        "nontrading", latest,
                        "not a trading day in the exchange calendar; no decision recorded"))
    if state.get("last_panel_date") == str(latest.date()) and not force:
        return dict(kind="stale", append=True,
                    record=stop_record(
                        "stale", latest,
                        "panel has not advanced since the last run; no decision recorded"))
    return None


def interval_for(index, close, held, state, latest):
    """The accounting tuple every shadow book must book identically, key for key.

    ``test_variant_a_books_its_intervals_exactly_like_the_main_ledger`` (2026-11-20) pins the
    two recorders to this one construction: an interval a variant books differently from the
    book it is the control for would make the 2027 comparison meaningless.
    """
    if held and state.get("last_panel_date"):
        return dict(**{"from": state.get("last_panel_date"), "to": str(latest.date()),
                       "sessions": sessions_between(index, state.get("last_panel_date"), latest),
                       "return": basket_return(close, held,
                                               state.get("last_panel_date"), latest)})
    return None


def chain(records):
    """Turn the log's interval returns into an index, starting at 1.0 before the first decision.

    ``mark`` alone cannot answer "how much has this book earned" because its level jumps whenever
    the composition changes; chaining the interval returns is immune to that.
    """
    level = 1.0
    out = []
    for record in records:
        change = (record.get("interval") or {}).get("return")
        if change is not None:
            level *= 1.0 + float(change)
        out.append(level)
    return out


def run(force=False, dataset=DATASET, log=print):
    directory = Path(dataset)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    codes = [code for code in manifest["stock_codes"] if code in set(panel.columns.get_level_values(0))]
    features = list(manifest["feature_cols"])
    latest = panel.index.max()
    state = load_state()
    # A truncated download would silently shrink the pool and the recorded book would come
    # from a different universe than the one the rule was frozen on -- refused in the guards.
    expected = len(manifest["stock_codes"])
    stop = shared_guards(codes, expected, state, latest, force=force)
    if stop:
        if stop["append"]:
            append(stop["record"])
        log({"bad_data": "bad_data: %d of %d codes" % (len(codes), expected),
             "nontrading": "nontrading: %s" % date.today(),
             "stale": "stale: panel still ends %s" % latest.date()}[stop["kind"]])
        return stop["record"]
    held = state.get("holdings", [])
    sessions = sessions_between(panel.index, state.get("entry_date"), latest)
    # The frozen rule holds a book for twenty sessions; deciding every session is what turned a
    # 15x-a-year turnover into 105x (doc/e1_topk_20261114.md).  So the recorder only re-decides when
    # the entry date is at least HOLD sessions old -- which is what "20 日持有" was always meant to be.
    due = state.get("entry_date") is None or (sessions is not None and sessions >= HOLD)
    # Scored every session, not only on rebalance days: the page shows "today's top K of the book",
    # and a hold day without scores made that slice arbitrary and the score column blank.
    score = composite(panel, codes, features, latest)
    if score is None:
        record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="no_score",
                      panel_date=str(latest.date()), reason="no usable feature at the last date")
        append(record)
        log("no score at %s" % latest.date())
        return record
    if due:
        book = decide(score, held)
    else:
        book = sorted(held, key=lambda code: float(score.get(code, float("inf"))))
    close = prices.xs("close", axis=1, level=1) if "close" in set(prices.columns.get_level_values(1)) else None
    close = close if close is not None else prices.xs(prices.columns.get_level_values(1)[0], axis=1, level=1)
    nav = mark(close, book, latest)
    interval = interval_for(panel.index, close, held, state, latest)
    comparison = benchmarks(close, codes, state.get("last_panel_date"), str(latest.date())) \
        if state.get("last_panel_date") else None
    record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="decided",
                  panel_date=str(latest.date()), book=book,
                  score={code: round(float(score[code]), 6) for code in book if code in score},
                  nav=nav, held_from=state.get("entry_date"),
                  decision="rebalance" if due else "hold",
                  rebalance_due=bool(due), sessions_held=sessions,
                  interval=interval, benchmark=comparison, construction="e1 reversal v1")
    append(record)
    save_state(dict(status="running", last_panel_date=str(latest.date()),
                    holdings=book, entry_date=str(latest.date()) if due else state.get("entry_date"),
                    runs=int(state.get("runs", 0)) + 1))
    log("%s on %s: %d names" % ("rebalance" if due else "hold", latest.date(), len(book)))
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
