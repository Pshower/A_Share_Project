"""Restate a book's fees after a fee-schedule fix, with the arithmetic written down.

The backtest book charged 0.05% stamp duty on every ETF sale.  Exchange-traded funds are exempt, so
291 sell legs carry a fee that the broker never takes: 6,786.88 yuan, 79.6% of everything the book
recorded as fees, 0.68% of capital over six years.

The fix in ``account_ledger`` stops the charge for *future* trades but leaves history untouched --
recorded rows carry their own ``fee``.  This tool rewrites those rows and the snapshots that follow
them, keeps a record of what changed, and refuses to run twice.

The restatement is exact rather than approximate: the tax was a cash outflow on the sale date, so
every snapshot after that date is raised by the tax removed up to it, and nothing else moves --
positions and prices are copied through untouched.

    python -B -m tools.restate_book --book backtest              # report only
    python -B -m tools.restate_book --book backtest --apply      # rewrite, keeping a record
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = PROJECT_ROOT / "reports" / "account"
TAX = 0.0005


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")


def exempt(code):
    """The stamp duty does not apply to fund units, so these codes should never have been charged."""
    return str(code).startswith(("5", "1"))


def plan_restatement(trades):
    """Which rows change, by how much, and the cash it gives back on each date."""
    changes, by_date = [], {}
    for index, row in enumerate(trades):
        if row.get("action") != "sell" or not exempt(row.get("code")):
            continue
        amount = abs(float(row["shares"]) * float(row["price"]))
        commission = max(5.0, amount * 0.0003)          # the broker's schedule: 0.03%, floor 5 yuan
        recorded = float(row.get("fee") or 0.0)
        # Ask which model the recorded fee fits.  A fee equal to the commission carries no tax, and
        # inferring one from the trade amount would "refund" money that was never spent -- the first
        # version of this tool did exactly that, to the tune of 6,786.99 that no book ever paid.
        with_tax = commission + amount * TAX
        if abs(recorded - commission) <= 0.05:
            continue
        if abs(recorded - with_tax) > 0.05:
            continue                                    # some other schedule; do not guess
        tax = round(recorded - commission, 2)
        changes.append(dict(index=index, date=row["date"], code=row["code"], amount=round(amount, 2),
                            old_fee=row.get("fee"), tax=tax,
                            new_fee=round(float(row.get("fee") or 0) - tax, 2)))
        by_date[row["date"]] = by_date.get(row["date"], 0.0) + tax
    return changes, by_date


def restate(book, apply_changes=False, log=print):
    folder = ACCOUNT / book
    marker = folder / "restatement_20261115.json"
    if apply_changes and marker.exists():
        log("%s already restated; returning the record" % book)
        return json.loads(marker.read_text(encoding="utf-8"))
    trades = read_jsonl(folder / "trades.jsonl")
    snapshots = read_jsonl(folder / "snapshots.jsonl")
    if not trades:
        raise SystemExit("no trades in %s" % folder)
    changes, by_date = plan_restatement(trades)
    total = round(sum(item["tax"] for item in changes), 2)
    if not changes:
        log("%s: nothing to restate" % book)
        return dict(book=book, legs=0, removed=0.0, applied=False)

    cash_back = 0.0
    for row in snapshots:
        cash_back += by_date.get(row["date"], 0.0)
        row["nav_before_restatement"] = row.get("nav")
        row["cash_before_restatement"] = row.get("cash")
        row["nav"] = round(float(row["nav"]) + cash_back, 2)
        row["cash"] = round(float(row["cash"]) + cash_back, 2)

    for item in changes:
        trades[item["index"]]["fee"] = item["new_fee"]
        trades[item["index"]]["restated_20261115"] = item["tax"]

    record = dict(book=book, restated_at=datetime.now(timezone.utc).isoformat(),
                  rule="exchange-traded funds are exempt from stamp duty (0.05% on sales)",
                  legs=len(changes), removed=total,
                  capital=round(float(snapshots[0]["nav"] - snapshots[0]["nav_before_restatement"]), 2)
                  if snapshots else 0.0,
                  first_date=changes[0]["date"], last_date=changes[-1]["date"],
                  old_last_nav=snapshots[-1]["nav_before_restatement"] if snapshots else None,
                  new_last_nav=snapshots[-1]["nav"] if snapshots else None,
                  applied=bool(apply_changes))
    log("%s: %d sell legs, %.2f removed (old last nav %s -> %s)" % (
        book, len(changes), total, record["old_last_nav"], record["new_last_nav"]))

    if apply_changes:
        write_jsonl(folder / "trades.jsonl", trades)
        write_jsonl(folder / "snapshots.jsonl", snapshots)
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        account_path = folder / "account.json"
        if account_path.exists():
            account = json.loads(account_path.read_text(encoding="utf-8"))
            account["cash"] = round(float(account.get("cash") or 0) + total, 2)
            account["restated_20261115"] = total
            account_path.write_text(json.dumps(account, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        log("wrote %s" % marker)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Restate a book's fees after a fee-schedule fix.")
    parser.add_argument("--book", default="backtest")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    restate(args.book, apply_changes=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
