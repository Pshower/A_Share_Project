"""Mark a day once its quotes exist, instead of marking it with yesterday's close.

The daily job runs at 20:15 and the Sina fund history is sometimes still on the previous
session then -- on 2026-11-13 all four ETFs were still ending 11-12 at 20:15.  Writing today
against yesterday's closes produces a flat day that never happened, so ``account_ledger.mark``
now refuses it.  The refusal is right, but it leaves the day unmarked, and nothing re-runs the
job later.  This tool closes that gap: it waits for the session to be published and then does
the mark that the 20:15 run could not.

    python -B -m tools.mark_when_ready --account paper --date 2026-11-13 --minutes 45

Exit code 0 means the day is marked; 1 means the quotes never arrived inside the window and
the day is still unmarked (which is the honest state, not an error to hide).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date as date_type

from tools import account_ledger as ledger


def quotes_for(stamp):
    """The closes and their dates, straight from the source ``mark`` uses."""
    return ledger._prices_with_dates(ledger.LEGS)


def wait_and_mark(stamp, account="paper", minutes=45, interval=180, log=print):
    """Poll the quote dates and mark the day the moment all held legs are current."""
    deadline = time.monotonic() + minutes * 60
    ledger.use(account)
    attempts = 0
    while True:
        attempts += 1
        prices, dates = quotes_for(stamp)
        lagging = {code: day for code, day in dates.items() if str(day) != str(stamp)}
        log("attempt %d: quote dates %s" % (attempts, json.dumps(dates, sort_keys=True)))
        if not lagging:
            row = ledger.mark(stamp, prices=prices, price_dates=dates)
            return dict(status="marked", attempts=attempts, account=account,
                        date=str(stamp), nav=row["nav"], cash=row["cash"],
                        positions=row["positions"], prices=row["prices"])
        if time.monotonic() >= deadline:
            return dict(status="still_unpublished", attempts=attempts, account=account,
                        date=str(stamp), lagging=lagging,
                        note="leave the day unmarked rather than price it off another session")
        time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mark a day when its quotes finally exist.")
    parser.add_argument("--date", default=str(date_type.today()))
    parser.add_argument("--account", default="paper", choices=list(ledger.ACCOUNTS))
    parser.add_argument("--minutes", type=float, default=45.0)
    parser.add_argument("--interval", type=float, default=180.0)
    args = parser.parse_args(argv)
    row = wait_and_mark(args.date, account=args.account, minutes=args.minutes,
                        interval=args.interval,
                        log=lambda message: print(message, file=sys.stderr))
    print(json.dumps(row, ensure_ascii=False, indent=2, default=float))
    return 0 if row["status"] == "marked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
