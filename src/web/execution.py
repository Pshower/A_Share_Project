"""Read-only views of the three execution books and the S2 shadow record.

The workbench is a PPO-era tool and had no view of the plan's account at all: the three books
(live / paper / backtest) and the shadow recorder existed only on the command line.  These
helpers add the missing read path and nothing else -- no writes, no order placement.

Every number here is computed by ``tools.account_ledger`` itself (``curve``, ``replay``), not
re-derived: a dashboard that rounds its own way is a second source of truth, and the whole point
of the ledger is that there is only one.  ``tests/test_web_execution.py`` asserts the page and the
CLI agree on the real books.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools import account_ledger as ledger

BOOKS = ledger.ACCOUNTS
SHADOW = Path("reports") / "shadow" / "e1_shadow.jsonl"
SHADOW_STATE = Path("reports") / "shadow" / "e1_state.json"
NAMES = Path("data") / "stock_list" / "stock_names.csv"

_name_cache = {"key": None, "map": {}}


def names(root):
    """Code to Chinese short name, from the one local snapshot; empty when it is absent.

    The shadow ledger stores codes only -- that is the right unit for the record and the wrong
    unit for a person reading a list of thirty of them.
    """
    path = Path(root) / NAMES
    if not path.exists():
        return {}
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return {}
    if _name_cache["key"] != (str(path), stamp):
        import pandas as pd
        frame = pd.read_csv(path)
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        _name_cache["map"] = dict(zip(frame["code"], frame["name"].astype(str)))
        _name_cache["key"] = (str(path), stamp)
    return _name_cache["map"]


def _read(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _paths(root, name):
    base = Path(root) / "reports" / "account" / name
    return dict(base=base, config=base / "account.json", trades=base / "trades.jsonl",
                snapshots=base / "snapshots.jsonl")


def books(root):
    """One row per book: what it is worth now, and whether it has been opened at all."""
    rows = []
    for name in BOOKS:
        path = _paths(root, name)
        if not path["config"].exists():
            rows.append(dict(name=name, opened=False))
            continue
        cfg = ledger.config(path["config"])
        snapshots = _read(path["snapshots"])
        trades = _read(path["trades"])
        curve = ledger.curve(snapshots=snapshots, cfg=cfg) if snapshots else dict(status="no_snapshots")
        rows.append(dict(name=name, opened=True, capital=cfg["capital"], start_date=cfg["start_date"],
                         fills=len([row for row in trades if row.get("action") != "dividend"]),
                         dividends=len([row for row in trades if row.get("action") == "dividend"]),
                         last=curve.get("last"), nav=curve.get("nav"),
                         total_return=curve.get("total_return"),
                         worst_drawdown=curve.get("worst_drawdown"), status=curve.get("status")))
    return rows


def book(root, name):
    """Curve, holdings and fills for one book, as the CLI reports them."""
    if name not in BOOKS:
        raise ValueError("unknown book %r; choose one of %s" % (name, ", ".join(BOOKS)))
    path = _paths(root, name)
    if not path["config"].exists():
        return dict(name=name, opened=False)
    cfg = ledger.config(path["config"])
    snapshots = _read(path["snapshots"])
    trades = _read(path["trades"])
    live_book = ledger.replay(trades, cfg["capital"])
    curve = ledger.curve(snapshots=snapshots, cfg=cfg) if snapshots else dict(status="no_snapshots")
    last = max(snapshots, key=lambda row: str(row["date"])) if snapshots else None
    prices = (last or {}).get("prices") or {}
    positions = []
    for code, shares in sorted(live_book["positions"].items()):
        price = float(prices.get(code) or 0.0)
        value = price * shares
        positions.append(dict(code=code, shares=shares, price=price, value=value,
                              weight=value / float((last or {}).get("nav") or 1.0)))
    fills = [row for row in trades if row.get("action") != "dividend"]
    fees = sum(float(row.get("fee") or 0.0) for row in fills)
    series = curve.get("series") or {}
    return dict(name=name, opened=True, capital=cfg["capital"], start_date=cfg["start_date"],
                curve=curve, dates=list(series.keys()), navs=list(series.values()),
                cash=live_book["cash"], positions=positions,
                trades=dict(count=len(fills), fees=round(fees, 2),
                            last=[dict(date=row.get("date"), code=row.get("code"),
                                       action=row.get("action"), shares=row.get("shares"),
                                       price=row.get("price"), source=row.get("source"))
                                  for row in fills[-8:]]),
                last_snapshot=dict(date=(last or {}).get("date"),
                                   price_dates=(last or {}).get("price_dates") or {},
                                   provisional=bool((last or {}).get("provisional"))))


def shadow(root, limit=8):
    """The S2 recorder's decisions, the names it picked, and what state it is holding.

    ``days`` is the list a reader actually wants: every session that produced a book, the thirty
    names in it, and what was bought/dropped against the previous decision.  The recorder stores
    codes and scores; names are joined from the local snapshot when it exists.
    """
    rows = _read(Path(root) / SHADOW)
    state_path = Path(root) / SHADOW_STATE
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    lookup = names(root)
    recent = []
    for row in rows[-limit:]:
        recent.append(dict(recorded_at=row.get("recorded_at"), status=row.get("status"),
                           panel_date=row.get("panel_date"), holdings=len(row.get("book") or []),
                           held_from=row.get("held_from"), rebalance_due=row.get("rebalance_due"),
                           decision=row.get("decision") or "legacy",
                           sessions_held=row.get("sessions_held"),
                           reason=row.get("reason"), construction=row.get("construction")))
    days, previous = [], set()
    level, chain = 1.0, []
    for row in rows:
        change = (row.get("interval") or {}).get("return")
        if change is not None:
            level *= 1.0 + float(change)
        chain.append(level)
    index = 0
    for row in rows:
        if row.get("status") != "decided" or not row.get("book"):
            continue
        # A row whose ``held_from`` is empty is a book being opened, not a book being rotated:
        # the first entry in the ledger is the archived 2024-12-31 seed, and diffing the forward
        # run against it would report 26 spurious "buy" decisions.
        fresh = not row.get("held_from")
        if fresh:
            previous = set()
        book = list(row["book"])
        scores = row.get("score") or {}
        ordered = sorted(book, key=lambda code: scores.get(code, float("inf")))
        current = set(book)
        interval = row.get("interval") or {}
        days.append(dict(
            panel_date=str(row.get("panel_date") or ""),
            recorded_at=row.get("recorded_at"),
            held_from=row.get("held_from"),
            decision=row.get("decision") or "legacy",   # rows written before the 20-session cadence
            sessions_held=row.get("sessions_held"),
            interval_from=interval.get("from"),
            interval_return=interval.get("return"),
            chained_nav=round(chain[index], 6),
            rebalance_due=bool(row.get("rebalance_due")),
            names=[dict(code=code, name=lookup.get(code, ""), score=round(float(scores[code]), 4)
                        if code in scores else None, change="in" if previous and code not in previous
                        else ("held" if previous else "new"))
                   for code in ordered],
            entries=[dict(code=code, name=lookup.get(code, ""))
                     for code in (sorted(current - previous) if previous else [])],
            exits=[dict(code=code, name=lookup.get(code, ""))
                   for code in (sorted(previous - current) if previous else [])],
            overlap=len(current & previous) if previous else None,
        ))
        previous = current
        index += 1
    return dict(rows=len(rows), decided=len([row for row in rows if row.get("status") == "decided"]),
                stale=len([row for row in rows if row.get("status") == "stale"]),
                recent=recent, days=days,
                chained_nav=round(level, 6),
                state=dict(holdings=len(state.get("holdings") or []),
                           entry_date=state.get("entry_date"),
                           last_panel_date=state.get("last_panel_date"),
                           runs=state.get("runs")))
