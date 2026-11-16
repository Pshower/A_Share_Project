"""Compare everything the account books recorded against the real bars, and repair what drifted.

The daily job used to price a day off whatever the fund history ended on, so a snapshot's date
said one thing and its prices said another.  ``mark`` now refuses that, but rows already written
stay wrong until someone recomputes them, and a NAV is only worth keeping if it can be rebuilt
from the closes of its own date.

Two things are checked, because a stale quote can enter either:

* **snapshots** -- each row's held shares and cash (which come from the trade log) are kept, the
  prices are replaced with that date's real closes, and the NAV is recomputed.  A wrong price is
  also traced: if the stored close belongs to another session, that session is named.
* **fills** -- a row recorded as ``model_open`` / ``model_close`` must equal that day's open or
  close.  A ``manual`` row carries a price only the person who traded knows, so it is listed as
  unverifiable rather than passed or failed.

Only the real books are audited.  The ``backtest`` book is priced by the simulator's own series,
not by the fund endpoint, so comparing it against Sina would manufacture mismatches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tools import account_ledger as ledger
from tools import plan_orders

AUDITED = ("live", "paper")
TOLERANCE = 0.005          # a half-cent: below this the row is the same price, rounded
# Model fills recorded at the next session's opening print.  Everything else the model writes is
# priced off that day's close -- including ``unified_open``, whose name was a misnomer.
OPEN_PRICED_SOURCES = ("model_open",)


def bars(codes, start=None, end=None):
    """Open and close per ETF, from the same endpoint and prefix rule as ``mark``."""
    import akshare as ak
    frames = {}
    for code in codes:
        frame = ak.fund_etf_hist_sina(symbol=plan_orders.prefix_for(code) + code)
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.set_index("date")[["open", "close"]].astype(float).sort_index()
        frames[code] = frame
    if start is not None or end is not None:
        frames = {code: frame.loc[start:end] for code, frame in frames.items()}
    return frames


def _wide(frames, field):
    return pd.DataFrame({code: frame[field] for code, frame in frames.items()})


def audit_snapshots(snapshots, closes):
    """Every row whose stored price is not that day's close, with where it really came from."""
    findings = []
    for row in sorted(snapshots, key=lambda item: str(item["date"])):
        day = pd.Timestamp(str(row["date"]))
        for code, held in (row.get("positions") or {}).items():
            if not held:
                continue
            stored = (row.get("prices") or {}).get(code)
            if stored is None:
                findings.append(dict(date=str(row["date"]), code=code, stored=None, truth=None,
                                     issue="no price recorded"))
                continue
            if code not in closes.columns or day not in closes.index:
                findings.append(dict(date=str(row["date"]), code=code, stored=float(stored),
                                     truth=None, issue="no close on that date"))
                continue
            truth = float(closes.at[day, code])
            if abs(truth - float(stored)) <= TOLERANCE:
                continue
            origin = origin_of(closes[code], stored, day)
            finding = dict(date=str(row["date"]), code=code, stored=float(stored), truth=truth,
                           issue="priced off another session" if origin
                           else "price does not match the close")
            if origin:
                finding["origin"] = origin
            findings.append(finding)
    return findings


def origin_of(column, stored, day):
    """Which session the stored close actually belongs to, if it is not the row's own."""
    column = column.dropna()
    same = column[(column - float(stored)).abs() <= TOLERANCE]
    same = same[same.index != day]
    return str(same.index[-1].date()) if len(same) else None


def audit_fills(trades, opens, closes):
    """Fills priced by the model must match the bar; manual prices can only be listed."""
    findings = []
    for row in trades:
        source = str(row.get("source", ""))
        if source.startswith("manual") or source in ("backtest", "fund_etf_dividend_sina"):
            if source.startswith("manual"):
                findings.append(dict(date=str(row["date"]), code=row["code"], price=row["price"],
                                     issue="manual fill: price cannot be checked against the bar"))
            continue
        # Which bar a model fill was priced off is a property of the recorder, not of the name:
        # ``model_open`` fills at the next open, while ``unified_open`` (kept for the fills
        # written on 2026-11-13) was recorded at that day's **close** and still ends in "open".
        label = "open" if source in OPEN_PRICED_SOURCES else "close"
        frame = opens if label == "open" else closes
        day, code = pd.Timestamp(str(row["date"])), row["code"]
        if code not in frame.columns or day not in frame.index:
            findings.append(dict(date=str(row["date"]), code=code, price=row["price"],
                                 issue="no bar on that date"))
            continue
        truth = float(frame.at[day, code])
        if abs(truth - float(row["price"])) > TOLERANCE:
            findings.append(dict(date=str(row["date"]), code=code, price=float(row["price"]),
                                 truth=truth, issue="fill price does not match the %s" % label))
    return findings


def repair_snapshots(snapshots, closes):
    """Rebuild each row's NAV from its own date's closes; return the rows and what changed.

    Rows whose prices already match are not rewritten, but they do get the ``price_dates`` stamp
    they were written without -- after verification, "these prices are this day's" is a fact, and
    a book where only the newest rows say it is a book you cannot audit in one pass.
    """
    fixed, changed, stamped = [], [], []
    for row in sorted(snapshots, key=lambda item: str(item["date"])):
        row = json.loads(json.dumps(row))
        day = pd.Timestamp(str(row["date"]))
        prices = dict(row.get("prices") or {})
        before = float(row["nav"])
        touched = False
        for code, held in (row.get("positions") or {}).items():
            if not held or code not in closes.columns or day not in closes.index:
                continue
            truth = float(closes.at[day, code])
            if code not in prices or abs(float(prices[code]) - truth) > TOLERANCE:
                prices[code] = truth
                touched = True
        if touched:
            row["prices"] = prices
            row["price_dates"] = {code: str(day.date()) for code in prices}
            row.pop("provisional", None)
            row["nav"] = ledger.nav(row.get("positions") or {}, float(row["cash"]), prices)
            changed.append(dict(date=str(day.date()), nav_before=before, nav_after=row["nav"],
                                delta=row["nav"] - before,
                                prices={code: round(value, 4) for code, value in prices.items()}))
        elif prices and not row.get("price_dates"):
            row["price_dates"] = {code: str(day.date()) for code in prices}
            stamped.append(str(day.date()))
        fixed.append(row)
    return fixed, changed, stamped


def run(account, repair_it=False, log=print):
    if account not in AUDITED:
        raise ValueError("%s is priced by its own simulator, not by the fund endpoint; "
                         "auditing it against Sina would invent mismatches" % account)
    ledger.use(account)
    snapshots = ledger._read(ledger.SNAPSHOTS)
    trades = ledger._read(ledger.TRADES)
    codes = sorted({code for row in snapshots for code in (row.get("positions") or {})})
    if not snapshots:
        return dict(account=account, status="no_snapshots")
    if not codes:
        # A cash-only book has no price to be stale, but every row should still say so.
        stamped = []
        if repair_it:
            rows = []
            for row in snapshots:
                row = json.loads(json.dumps(row))
                if "price_dates" not in row:
                    row["price_dates"] = {}
                    stamped.append(str(row["date"]))
                rows.append(row)
            Path(ledger.SNAPSHOTS).write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        for row in sorted(rows, key=lambda item: str(item["date"]))),
                encoding="utf-8")
        return dict(account=account, status="cash_only", rows=len(snapshots),
                    trades=len(trades), stamped=stamped)
    frames = bars(codes, start=min(str(row["date"]) for row in snapshots),
                  end=max([str(row["date"]) for row in snapshots]
                          + [str(row["date"]) for row in trades]))
    closes, opens = _wide(frames, "close"), _wide(frames, "open")
    snap_findings = audit_snapshots(snapshots, closes)
    fill_findings = audit_fills([row for row in trades if row.get("action") != "dividend"],
                                opens, closes)
    result = dict(account=account, status="checked", rows=len(snapshots), codes=codes,
                  snapshot_findings=snap_findings, fill_findings=fill_findings,
                  repaired=None, stamped=[])
    if repair_it:
        fixed, changed, stamped = repair_snapshots(snapshots, closes)
        Path(ledger.SNAPSHOTS).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in fixed),
            encoding="utf-8")
        result["repaired"], result["stamped"] = changed, stamped
        log("repaired %d row(s), stamped %d row(s) in %s"
            % (len(changed), len(stamped), ledger.SNAPSHOTS))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-check recorded NAVs and fills against real bars.")
    parser.add_argument("--account", default="paper", choices=list(ledger.ACCOUNTS))
    parser.add_argument("--repair", action="store_true", help="Rewrite the rows that drifted.")
    args = parser.parse_args(argv)
    try:
        row = run(args.account, repair_it=args.repair)
    except ValueError as problem:
        print(json.dumps(dict(account=args.account, status="refused", reason=str(problem)),
                         ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
