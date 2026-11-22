"""Allocation shadow books: two optimised arms and one static reference, one frozen rule each.

``equity_bond_tail`` ended with a pre-registered consequence: the optimised arms (``minvar``,
``cvar``) enter the portfolio-layer shadow candidate list -- calendar time only, no live
changes.  This module is that list made real: three append-only books under
``reports/shadow/alloc/`` that replay the study's frozen rules forward,

* ``fixed6040``  -- the plan's static leg weights, annual rebalance (the paired reference);
* ``minvar``     -- rolling minimum-variance, quarterly re-estimate on the 10% simplex grid;
* ``cvar``       -- rolling CVaR95 minimisation, same grid, same cadence;

all identical to the study in estimation window (504 trading days, returns ending at the
decision close), costs (13bp per side on turnover), codes and repair rules -- the accounting is
``equity_bond_tail.run_arm`` step for step, and ``test_shadow_alloc`` pins the two engines
together bit for bit, the same way the variants books pin to the main E1 ledger.

State is the log's own last decision row, never a side file: the books follow the account
ledger's rule of one truth.  A non-trading day appends nothing; a day whose bar is missing from
the cache is reported ``behind`` and caught up on the next successful refresh; re-running the
same day appends one ``stale`` status row and no decisions.  Weights, cadence, grid and costs
are frozen here -- "the optimiser is losing, adjust it" is a new pre-registration, not a
parameter.

The books were opened at the 2026-11-20 close with the establishment cost booked.  The frozen
first judgment date and its verdict bars live in ``doc/shadow_alloc_20261123.md`` -- before that
date every row is a record, not a result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import equity_bond_tail as ebt                  # noqa: E402

OUTPUT = PROJECT_ROOT / "reports" / "shadow" / "alloc"
ARMS = ("fixed6040", "minvar", "cvar")
OPEN_DATE = "2026-11-20"          # the cached data's edge on opening night; establishment at that close
CAPITAL = 1_000_000.0             # notional the nav index refers to; the books themselves are unit-level
PACE = 0.3                       # seconds between carrier refreshes, daily_ops' own courtesy


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def book_path(arm: str, output=OUTPUT) -> Path:
    return Path(output) / ("%s_alloc.jsonl" % arm)


def read_book(arm: str, output=OUTPUT) -> list:
    path = book_path(arm, output)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(arm: str, record: dict, output=OUTPUT) -> None:
    path = book_path(arm, output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())       # a half-written line would poison the ledger's read side


def state_from(records: list) -> dict | None:
    """The last decision row is the whole state; status rows never carry book state."""
    for record in reversed(records):
        if record.get("kind") == "decision":
            return record
    return None


def frame_returns(frame: pd.DataFrame) -> np.ndarray:
    """Exactly ``run_arm``'s own return table, so the two engines see identical numbers."""
    returns = frame.pct_change().to_numpy()
    returns[0] = 0.0
    return returns


def refresh_carriers(pace: float = PACE, log=print) -> dict:
    """Pull the four legs' prices and published payouts into the local cache, politely."""
    from tools import fund_vehicles
    for code in ebt.CODES:
        fund_vehicles.load_prices(code, refresh=True)
        fund_vehicles.load_distributions(code, refresh=True)
        if pace:
            time.sleep(pace)
    return dict(carriers=len(ebt.CODES))


def establish(arm: str, frame: pd.DataFrame, output=OUTPUT, log=print) -> dict:
    """Open one book at the frozen date: full investment, establishment cost, frozen weights.

    Mirrors ``run_arm`` step 0 key for key: cost 13bp on a 1.0 turnover, weights from the
    frozen rule on the 504-day window ending at the open close, nav on the unit index.
    """
    if state_from(read_book(arm, output)) is not None:
        raise SystemExit("%s is already open; a second establishment would rewrite history" % arm)
    stamp = pd.Timestamp(OPEN_DATE)
    pos = frame.index.get_loc(stamp)
    returns = frame_returns(frame)
    target = ebt.target_for(arm, returns, pos)
    record = dict(
        recorded_at=now_iso(), arm=arm, kind="decision", action="establish",
        price_date=str(stamp.date()), steps=0,
        nav=1.0 * (1.0 - ebt.COST_PER_SIDE),
        weights={code: float(weight) for code, weight in zip(ebt.CODES, target)},
        turnover_total=1.0, cost_total=ebt.COST_PER_SIDE,
        rebalance_every=ebt.REBALANCE[arm], capital=CAPITAL,
        construction="equity_bond_tail frozen rule, forward from %s" % OPEN_DATE)
    append(arm, record, output)
    log("%s establish %s nav %.6f" % (arm, record["price_date"], record["nav"]))
    return record


def advance(arm: str, frame: pd.DataFrame, output=OUTPUT, today: date | None = None,
            log=print) -> dict:
    """Append one decision row per cached session after the book's last row, then report status.

    Catch-up is bar-driven: every session the cache actually has gets a row, whether the run
    happens that evening or three days later.  A trading day whose bar is still missing gets an
    honest ``behind`` status row, never an invented one.  Non-trading days append nothing.
    """
    today = today or date.today()
    records = read_book(arm, output)
    state = state_from(records)
    if state is None:
        record = dict(recorded_at=now_iso(), arm=arm, kind="status", status="unopened",
                      reason="no decision rows; run with --init first")
        log("%s unopened" % arm)
        return record

    returns = frame_returns(frame)
    weights = np.array([float(state["weights"][code]) for code in ebt.CODES])
    nav = float(state["nav"])
    steps = int(state["steps"])
    turnover = float(state["turnover_total"])
    cost = float(state["cost_total"])
    cadence = ebt.REBALANCE[arm]
    pos_last = frame.index.get_loc(pd.Timestamp(state["price_date"]))
    appended, actions = 0, []
    for pos in range(pos_last + 1, len(frame.index)):
        steps += 1
        row = returns[pos]
        growth = 1.0 + float(np.dot(weights, row))
        nav *= growth
        weights = weights * (1.0 + row) / growth
        action, moved = "drift", 0.0
        if steps % cadence == 0:
            target = ebt.target_for(arm, returns, pos)
            moved = float(np.abs(target - weights).sum())
            if moved > 0:
                nav *= 1.0 - ebt.COST_PER_SIDE * moved
            weights = target
            action = "rebalance"
        turnover += moved
        cost += ebt.COST_PER_SIDE * moved
        record = dict(
            recorded_at=now_iso(), arm=arm, kind="decision", action=action,
            price_date=str(frame.index[pos].date()), steps=steps, nav=nav,
            weights={code: float(weight) for code, weight in zip(ebt.CODES, weights)},
            turnover_total=turnover, cost_total=cost,
            rebalance_in=cadence - (steps % cadence))
        append(arm, record, output)
        appended += 1
        actions.append(action)

    if not appended:
        try:
            from tools import trading_calendar
            trading = trading_calendar.is_trading_day(today)
        except Exception:                                  # noqa: BLE001 - calendar is optional
            trading = True
        if not trading:
            log("%s nontrading %s (nothing appended)" % (arm, today))
            return dict(recorded_at=now_iso(), arm=arm, kind="status", status="nontrading")
        if pd.Timestamp(today) > frame.index[-1]:
            record = dict(recorded_at=now_iso(), arm=arm, kind="status", status="behind",
                          last_bar=str(frame.index[-1].date()),
                          reason="today is a trading day but the cache has no bar for it")
            append(arm, record, output)
            log("%s behind (cache ends %s)" % (arm, record["last_bar"]))
            return record
        record = dict(recorded_at=now_iso(), arm=arm, kind="status", status="stale",
                      price_date=str(frame.index[-1].date()),
                      reason="no sessions past the book's last row; nothing to append")
        append(arm, record, output)
        log("%s stale %s" % (arm, record["price_date"]))
        return record
    return dict(recorded_at=now_iso(), arm=arm, kind="decision", action=actions[-1],
                price_date=str(frame.index[-1].date()), appended=appended, steps=steps,
                nav=nav, turnover_total=turnover, cost_total=cost)


def run(arms=ARMS, output=OUTPUT, today: date | None = None, log=print) -> list:
    frame, _, _ = ebt.total_return_legs(include_manual=True)
    return [advance(arm, frame, output=Path(output), today=today, log=log)
            for arm in arms if arm in ARMS]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record the allocation shadow books for one session.")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--init", action="store_true",
                        help="open the books at %s (refuses an already-open book)" % OPEN_DATE)
    parser.add_argument("--no-refresh", action="store_true",
                        help="trust the local cache; the books go 'behind' if it lacks today's bar")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    arms = [arm for arm in args.arms.split(",") if arm]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError("unknown arm(s): %s" % ", ".join(unknown))
    output = Path(args.output)

    if args.init:
        frame, _, _ = ebt.total_return_legs(include_manual=True)
        for arm in arms:
            if state_from(read_book(arm, output)) is not None:
                raise SystemExit("%s is already open; --init refuses a second establishment" % arm)
            establish(arm, frame, output=output)
        return 0

    if not args.no_refresh:
        refresh_carriers()
    records = run(arms=arms, output=output)
    print(json.dumps(records, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
