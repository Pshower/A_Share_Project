"""Keep the account that the plan is actually run on, and refuse bad orders before they leave.

The plan tool answers "what should I hold"; this one answers "what do I hold, what is it worth,
and is this order safe to send".  Everything is derived from an append-only trade log plus a
daily mark, so there is exactly one copy of the truth: a stored balance that could disagree
with the log is how account tools go wrong.

Files under ``reports/account``:

``account.json``     starting capital and start date; written once by ``init``
``trades.jsonl``     append-only: one record per fill, with fees and the price it was decided at
``snapshots.jsonl``  append-only: one record per marked day (prices, positions, NAV, peak)

Nothing here talks to a broker.  A trade is recorded when the fill is known; the tool does not
place it, and it says so in every report.
"""
from __future__ import annotations

import argparse
import json
from datetime import date as date_type
from pathlib import Path

import pandas as pd
import numpy as np

from tools import plan_orders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_DIR = PROJECT_ROOT / "reports" / "account"
ACCOUNTS = ("live", "paper", "backtest")
DEFAULT_ACCOUNT = "live"
CONFIG = ACCOUNT_DIR / DEFAULT_ACCOUNT / "account.json"
TRADES = ACCOUNT_DIR / DEFAULT_ACCOUNT / "trades.jsonl"
SNAPSHOTS = ACCOUNT_DIR / DEFAULT_ACCOUNT / "snapshots.jsonl"
ACTIVE = DEFAULT_ACCOUNT


def use(name):
    """Point the module at one account. A paper book and a real book must never share a file.

    Two books that look alike are exactly the failure this separation prevents: a modelled fill
    landing in the real ledger makes the account report a position nobody owns.
    """
    global ACTIVE, CONFIG, TRADES, SNAPSHOTS
    if name not in ACCOUNTS:
        raise ValueError("Unknown account %r; expected one of %s" % (name, ", ".join(ACCOUNTS)))
    ACTIVE = name
    base = ACCOUNT_DIR / name
    CONFIG, TRADES, SNAPSHOTS = base / "account.json", base / "trades.jsonl", base / "snapshots.jsonl"
    return name
LEGS = list(plan_orders.LEGS) + [plan_orders.SAFE]
RISKY = list(plan_orders.LEGS)
LOT = plan_orders.LOT


def _read(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def init(capital, start, path=None):
    """Write the starting capital once; never overwrite an existing account."""
    path = Path(path) if path is not None else CONFIG
    if path.exists():
        raise ValueError("Account already initialised at %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(capital=float(capital), start_date=str(start))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def config(path=None):
    path = Path(path) if path is not None else CONFIG
    if not Path(path).exists():
        raise ValueError("No account at %s; run init first" % path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fee_for(amount, broker=None):
    """Commission floor plus the sell-side stamp duty, as the plan's broker charges them."""
    broker = broker or {"commission": 0.0003, "min_commission": 5.0, "sell_tax": 0.0005}
    return max(broker["min_commission"], amount * broker["commission"])


def replay(trades, capital):
    """Cash and positions implied by the trade log.  The only source of truth.

    A distribution is cash in with no change in shares, and it has to be in this log or the
    account understates itself on every ex-date: the price drops by the distribution and
    nothing replaces it.  The treasury leg pays quarterly, so that is 2% a year of missing cash.
    """
    cash, positions = float(capital), {}
    for row in trades:
        if row["action"] == "dividend":
            cash += float(row["amount"])
            continue
        shares, price = int(row["shares"]), float(row["price"])
        amount = shares * price
        fee = float(row.get("fee", fee_for(amount)))
        if row["action"] == "buy":
            cash -= amount + fee
            positions[row["code"]] = positions.get(row["code"], 0) + shares
        elif row["action"] == "sell":
            cash += amount - fee
            positions[row["code"]] = positions.get(row["code"], 0) - shares
        else:
            raise ValueError("Unknown action %r" % row["action"])
        if positions[row["code"]] == 0:
            del positions[row["code"]]
    return dict(cash=cash, positions=positions)


def shares_held_at(trades, code, stamp):
    """Shares of ``code`` held strictly before ``stamp`` -- what an ex-date rewards."""
    held = 0
    for row in trades:
        if row["code"] != code or row["action"] == "dividend":
            continue
        if str(row["date"]) >= str(stamp):
            continue
        held += int(row["shares"]) if row["action"] == "buy" else -int(row["shares"])
    return held


def nav(positions, cash, prices):
    missing = [code for code in positions if code not in prices]
    if missing:
        raise ValueError("No price for held %s" % ", ".join(sorted(missing)))
    return cash + sum(shares * float(prices[code]) for code, shares in positions.items())


def peak_of(snapshots, capital):
    values = [float(row["nav"]) for row in snapshots] + [float(capital)]
    return max(values)


def state(dates=None, prices=None, cfg=None, trades=None, snapshots=None):
    """Current cash, positions, NAV and peak -- NAV from prices when they are given."""
    cfg = cfg or config()
    trades = _read(TRADES) if trades is None else trades
    snapshots = _read(SNAPSHOTS) if snapshots is None else snapshots
    book = replay(trades, cfg["capital"])
    last = snapshots[-1] if snapshots else None
    stamp = str(dates) if dates is not None else (last["date"] if last else cfg["start_date"])
    if prices is not None:
        value, source = nav(book["positions"], book["cash"], prices), "prices"
    elif last is not None:
        value, source = float(last["nav"]), "snapshot %s" % last["date"]
    else:
        value, source = float(cfg["capital"]), "initial capital"
    peak = peak_of(snapshots, cfg["capital"])
    trailing = [row for row in trades if last is None or str(row["date"]) > str(last["date"])]
    return dict(date=stamp, cash=book["cash"], positions=book["positions"], nav=value,
                nav_source=source, peak=peak, drawdown=value / peak - 1.0,
                unmarked_trades=len(trailing))


def distributions(code, start=None, end=None):
    """Every cash distribution per unit for one ETF, from the Sina fund history.

    The interface publishes the **cumulative** amount per unit, so the per-event figure is
    the difference between consecutive rows; the first row is its own event.
    """
    import akshare as ak
    frame = ak.fund_etf_dividend_sina(symbol=plan_orders.prefix_for(code) + code)
    if frame is None or not len(frame):
        return []
    frame = frame.copy()
    frame["日期"] = pd.to_datetime(frame["日期"])
    frame = frame.sort_values("日期")
    frame["per_unit"] = frame["累计分红"].diff().fillna(frame["累计分红"])
    rows = []
    for _, row in frame.iterrows():
        stamp = row["日期"].date()
        if start is not None and stamp < pd.Timestamp(start).date():
            continue
        if end is not None and stamp > pd.Timestamp(end).date():
            continue
        if float(row["per_unit"]) > 0:
            rows.append((stamp, float(row["per_unit"])))
    return rows


def sync_dividends(start=None, end=None, codes=None, path=None, cfg=None):
    """Append every distribution the log does not have yet, using the shares held then."""
    cfg = cfg or config()
    path = Path(path) if path is not None else TRADES
    codes = list(codes or LEGS)
    start = start or cfg["start_date"]
    end = end or str(date_type.today())
    trades = _read(path)
    known = {(str(row["date"]), row["code"]) for row in trades if row["action"] == "dividend"}
    added = []
    for code in codes:
        for stamp, per_unit in distributions(code, start, end):
            if (str(stamp), code) in known:
                continue
            held = shares_held_at(trades, code, stamp)
            if held <= 0:
                continue
            row = dict(recorded_at=pd.Timestamp.utcnow().isoformat(), date=str(stamp), code=code,
                       action="dividend", shares=held, cash_per_share=per_unit,
                       amount=round(held * per_unit, 2), fee=0.0, source="fund_etf_dividend_sina")
            _append(path, row)
            trades.append(row)
            known.add((str(stamp), code))
            added.append(row)
    return added


def market_price(code, stamp, field="open"):
    """The traded price on one day, from the same Sina history the plan is priced from.

    This is what makes a fill recordable without typing a number: the plan executes at the
    next open, so the next open is the honest stand-in for a fill that has already happened.
    It is still a model price -- it assumes you got the open with no queue and no partial
    fill -- which is why every row carries a ``source``.
    """
    import akshare as ak
    frame = ak.fund_etf_hist_sina(symbol=plan_orders.prefix_for(code) + code)
    if frame is None or not len(frame):
        raise ValueError("No history for %s" % code)
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    row = frame[frame["date"] == pd.Timestamp(stamp).date()]
    if row.empty:
        raise ValueError("No bar for %s on %s" % (code, stamp))
    return float(row[field].iloc[0])


def record_trade(stamp, code, action, shares, price=None, decision_price=None,
                 path=None, cfg=None, field="open", source=None):
    """Validate one fill against the log, then append it.  Never silently adjusts."""
    cfg = cfg or config()
    path = Path(path) if path is not None else TRADES
    if price is None:
        price = market_price(code, stamp, field)
        source = source or ("model_" + field)
    else:
        source = source or "manual"
    shares = int(shares)
    if shares <= 0 or shares % LOT:
        raise ValueError("Shares must be a positive multiple of %d" % LOT)
    if price <= 0:
        raise ValueError("Price must be positive")
    book = replay(_read(path), cfg["capital"])
    amount = shares * float(price)
    fee = fee_for(amount)
    if action == "sell":
        held = book["positions"].get(code, 0)
        if shares > held:
            raise ValueError("Selling %d of %s but only %d held" % (shares, code, held))
        fee += amount * 0.0005
    elif action == "buy":
        if amount + fee > book["cash"]:
            raise ValueError("Order needs %.2f but only %.2f cash" % (amount + fee, book["cash"]))
    else:
        raise ValueError("Unknown action %r" % action)
    row = dict(recorded_at=pd.Timestamp.utcnow().isoformat(), date=str(stamp), code=code,
               action=action, shares=shares, price=float(price), fee=round(fee, 2),
               source=source)
    if decision_price is not None:
        row["decision_price"] = float(decision_price)
    _append(path, row)
    return row


def mark(stamp, prices=None, price_dates=None, allow_stale=False, path=None, cfg=None):
    """Record what the account was worth on a day; prices default to the latest ETF closes.

    A mark is only honest if the quotes belong to the day being marked.  The daily job runs at
    20:15 and the Sina fund history sometimes still ends on the previous session then, which
    used to write "today" against yesterday's close -- a flat day that never happened, with no
    trace of the substitution left in the file.  So the quote date is now part of the row and a
    stale quote is refused unless the caller explicitly asks for a provisional mark.

    Idempotent per date: the daily job can run twice, or a re-mark can follow a late fill, and
    the curve must still have one point per day.  The newest write wins and the file is
    rewritten, which is the one place this tool is not append-only -- a duplicate NAV would
    silently double-count a day in every return computed from it.
    """
    cfg = cfg or config()
    path = Path(path) if path is not None else SNAPSHOTS
    if cfg.get("sealed"):
        raise ValueError("%s is a closed book (sealed export); marking it with a later date "
                         "would add a day the export never proved -- re-run the export instead"
                         % cfg.get("source", "this account"))
    book = replay(_read(TRADES), cfg["capital"])
    if prices is None:
        prices, price_dates = _prices_with_dates(LEGS)
    # Only a leg the book actually holds prices the NAV, so only those quotes can be stale.
    stale = {code: day for code, day in (price_dates or {}).items()
             if code in book["positions"] and str(day) != str(stamp)}
    if stale and not allow_stale:
        detail = ", ".join("%s %s" % (code, day) for code, day in sorted(stale.items()))
        raise ValueError("quotes are not from %s (%s); re-run once the session is published, or "
                         "pass allow_stale to record the day as provisional" % (stamp, detail))
    value = nav(book["positions"], book["cash"], prices)
    row = dict(date=str(stamp), nav=value, cash=book["cash"], positions=dict(book["positions"]),
               prices={code: float(prices[code]) for code in book["positions"]})
    if price_dates:
        row["price_dates"] = {code: str(price_dates[code]) for code in book["positions"]
                              if code in price_dates}
        if stale:
            row["provisional"] = True
    rows = [existing for existing in _read(path) if str(existing["date"]) != str(stamp)]
    rows.append(row)
    rows.sort(key=lambda item: str(item["date"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                            for item in rows), encoding="utf-8")
    return row


def curve(snapshots=None, cfg=None):
    """The running record: cumulative return, peak, drawdown and the worst one so far."""
    cfg = cfg or config()
    snapshots = _read(SNAPSHOTS) if snapshots is None else snapshots
    if not snapshots:
        return dict(status="no_snapshots", note="run mark after the first trading day")
    rows = sorted(snapshots, key=lambda item: str(item["date"]))
    navs = pd.Series({pd.Timestamp(row["date"]): float(row["nav"]) for row in rows}).sort_index()
    capital = float(cfg["capital"])
    # The starting capital is the first peak: a first day that loses money is a drawdown, and
    # measuring only from the recorded NAVs would report it as flat.
    peak_values = np.maximum.accumulate(np.r_[capital, navs.to_numpy(dtype=float)])[1:]
    peak = pd.Series(peak_values, index=navs.index)
    drawdown = navs / peak - 1.0
    years = max(len(navs) - 1, 1) / 242.0
    return dict(status="ready", first=str(navs.index[0].date()), last=str(navs.index[-1].date()),
                days=int(len(navs)), nav=float(navs.iloc[-1]),
                total_return=float(navs.iloc[-1] / capital - 1.0),
                annualised=float((navs.iloc[-1] / capital) ** (1 / years) - 1) if years >= 0.25 else None,
                peak=float(peak.iloc[-1]), drawdown=float(drawdown.iloc[-1]),
                worst_drawdown=float(drawdown.min()),
                days_in_drawdown=int((drawdown < 0).sum()),
                series={str(stamp.date()): round(float(value), 2) for stamp, value in navs.items()}).copy()


def gate(orders, prices, price_dates, state_row, floor=plan_orders.FLOOR):
    """Pre-trade checks.  Returns the list of things that must stop the order going out."""
    problems = []
    stamp = str(state_row["date"])
    # Every leg, the safe one included: 511010 is 40% of the book, and when this loop walked
    # ``plan_orders.LEGS`` (the three risky legs) a stale safe-leg quote passed the gate silently.
    for code in LEGS:
        quoted = price_dates.get(code)
        if quoted and quoted != stamp:
            problems.append("%s quote is from %s, not %s" % (code, quoted, stamp))
    if state_row["nav"] <= floor * state_row["peak"]:
        buys = [row for row in orders if row["delta"] > 0]
        if buys:
            problems.append("NAV %.2f is at or below the %.2f floor; risky legs must not be bought"
                            % (state_row["nav"], floor * state_row["peak"]))
    cash_left = state_row["cash"]
    for row in orders:
        if row["delta"] % LOT:
            problems.append("%s order is not a whole lot" % row["code"])
        if row["delta"] > 0:
            cost = row["delta"] * float(prices[row["code"]])
            cash_left -= cost + fee_for(cost)
    if cash_left < 0:
        problems.append("orders need %.2f more cash" % -cash_left)
    return problems


def slippage(trades):
    """Fill against the price the order was decided at, signed so positive is adverse."""
    rows = []
    for row in trades:
        if "decision_price" not in row:
            continue
        reference = float(row["decision_price"])
        moved = (float(row["price"]) - reference) / reference
        rows.append(moved if row["action"] == "buy" else -moved)
    if not rows:
        return dict(trades=0)
    series = pd.Series(rows)
    return dict(trades=len(series), mean_bps=float(series.mean() * 10000),
                worst_bps=float(series.min() * 10000), share_adverse=float((series > 0).mean()))


def book(stamp, field="open", path=None, cfg=None):
    """Execute the current plan inside the ledger at that day's traded prices.

    Sells are recorded before buys so the proceeds can fund the purchases, and every row is
    tagged with the price field it came from.  This is a *model* execution: it assumes the
    open (or close) was available for the whole size.  Running it on a day you did not trade
    turns the account into a paper account, which is a legitimate thing to want and a
    dishonest thing to discover later -- so the rows say ``model_open``, not ``manual``.
    """
    cfg = cfg or config()
    path = Path(path) if path is not None else TRADES
    trades = _read(path)
    current = replay(trades, cfg["capital"])
    prices = {code: market_price(code, stamp, field) for code in LEGS}
    row = state(prices=prices, cfg=cfg, trades=trades, dates=stamp)
    targets = plan_orders.decide(row["nav"], row["peak"], prices, held=current["positions"])
    ordered = sorted(targets["orders"], key=lambda item: item["delta"] > 0)
    booked = []
    for item in ordered:
        if item["delta"] == 0:
            continue
        booked.append(record_trade(stamp, item["code"], "buy" if item["delta"] > 0 else "sell",
                                   abs(item["delta"]), price=prices[item["code"]],
                                   path=path, cfg=cfg, source="model_" + field))
    return dict(date=str(stamp), field=field, booked=len(booked), weights=targets["weights"],
                prices=prices, trades=booked)


def monthly_report(month, snapshots=None, trades=None, cfg=None):
    """NAV, drawdown and per-leg contribution for one calendar month."""
    cfg = cfg or config()
    snapshots = _read(SNAPSHOTS) if snapshots is None else snapshots
    trades = _read(TRADES) if trades is None else trades
    window = [row for row in snapshots if str(row["date"]).startswith(month)]
    if not window:
        return dict(status="no_snapshots", month=month)
    first, last = window[0], window[-1]
    legs = {}
    for code in LEGS:
        before = float(first.get("positions", {}).get(code, 0)) * float(first.get("prices", {}).get(code, 0))
        after = float(last.get("positions", {}).get(code, 0)) * float(last.get("prices", {}).get(code, 0))
        legs[code] = round(after - before, 2)
    peak = peak_of([row for row in snapshots if str(row["date"]) <= last["date"]], cfg["capital"])
    fills = [row for row in trades if str(row["date"]).startswith(month) and row["action"] != "dividend"]
    paid = [row for row in trades if str(row["date"]).startswith(month) and row["action"] == "dividend"]
    return dict(status="ready", month=month, days=len(window),
                nav_start=float(first["nav"]), nav_end=float(last["nav"]),
                return_pct=float(last["nav"] / first["nav"] - 1.0),
                peak=peak, drawdown=float(last["nav"] / peak - 1.0),
                leg_change=legs, trades=len(fills),
                fees=round(sum(float(row.get("fee", 0.0)) for row in fills), 2),
                dividends=round(sum(float(row.get("amount", 0.0)) for row in paid), 2),
                slippage=slippage(fills))


def _prices_with_dates(codes, stamp=None):
    import akshare as ak
    prices, dates = {}, {}
    for code in codes:
        frame = ak.fund_etf_hist_sina(symbol=plan_orders.prefix_for(code) + code)
        prices[code] = float(frame["close"].iloc[-1])
        dates[code] = str(frame["date"].iloc[-1])
    return prices, dates


def main(argv=None):
    parser = argparse.ArgumentParser(description="Account ledger for the plan.")
    sub = parser.add_subparsers(dest="command", required=True)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT, choices=list(ACCOUNTS),
                        help="Which book: live (real fills), paper (modelled, from today) or backtest (simulated history).")

    sub.add_parser("init", help="Start an account.").add_argument("--capital", type=float, required=True)
    sub.choices["init"].add_argument("--date", default=str(date_type.today()))

    trade = sub.add_parser("trade", help="Record a fill.")
    for name, kwargs in (("--date", dict(default=str(date_type.today()))), ("--code", dict(required=True)),
                         ("--action", dict(required=True, choices=["buy", "sell"])),
                         ("--shares", dict(type=int, required=True)),
                         ("--price", dict(type=float, default=None)),
                         ("--field", dict(default="open", choices=["open", "close"])),
                         ("--decision-price", dict(type=float, default=None))):
        trade.add_argument(name, **kwargs)

    mark_cmd = sub.add_parser("mark", help="Record the day's NAV.")
    mark_cmd.add_argument("--date", default=str(date_type.today()))
    mark_cmd.add_argument("--allow-stale", action="store_true",
                          help="Record a day whose quotes are from an earlier session (provisional).")

    sub.add_parser("status", help="Current holdings, NAV and drawdown.")

    check = sub.add_parser("check", help="Pre-trade gate for the plan's orders.")
    check.add_argument("--date", default=str(date_type.today()))
    check.add_argument("--prices", default=None, help='JSON {"510300": price, ...}; omit to fetch.')

    book_cmd = sub.add_parser("book", help="Execute the plan inside the ledger at that day's prices.")
    book_cmd.add_argument("--date", default=str(date_type.today()))
    book_cmd.add_argument("--field", default="open", choices=["open", "close"])

    sub.add_parser("curve", help="Running return, peak and drawdown.")

    report = sub.add_parser("report", help="Monthly statement.")
    report.add_argument("--month", required=True)

    sync = sub.add_parser("sync-dividends", help="Append distributions the log is missing.")
    sync.add_argument("--start", default=None)
    sync.add_argument("--end", default=None)

    # Accept --account on either side of the subcommand: "init --account paper" reads the way
    # people type it, and refusing that would only teach them to look up the syntax.
    for child in sub.choices.values():
        child.add_argument("--account", default=argparse.SUPPRESS, choices=list(ACCOUNTS))

    args = parser.parse_args(argv)
    use(args.account)
    if args.command == "init":
        print(json.dumps(init(args.capital, args.date), ensure_ascii=False, indent=2))
    elif args.command == "trade":
        print(json.dumps(record_trade(args.date, args.code, args.action, args.shares,
                                      args.price, args.decision_price, field=args.field),
                         ensure_ascii=False, indent=2))
    elif args.command == "mark":
        print(json.dumps(mark(args.date, allow_stale=args.allow_stale),
                         ensure_ascii=False, indent=2, default=float))
    elif args.command == "status":
        row = state()
        print(json.dumps(row, ensure_ascii=False, indent=2, default=float))
    elif args.command == "check":
        cfg = config()
        trades = _read(TRADES)
        ledger_book = replay(trades, cfg["capital"])
        if args.prices:
            prices, dates = json.loads(args.prices), {code: args.date for code in LEGS}
        else:
            prices, dates = _prices_with_dates(LEGS)
        row = state(prices=prices, cfg=cfg, trades=trades, dates=args.date)
        targets = plan_orders.decide(row["nav"], row["peak"], prices, held=ledger_book["positions"])
        problems = gate(targets["orders"], prices, dates, row)
        current = {}
        for code in LEGS:
            value = float(ledger_book["positions"].get(code, 0)) * float(prices[code])
            current[code] = round(value / row["nav"], 6) if row["nav"] else 0.0
        drift = {code: round(current[code] - targets["weights"][code], 6) for code in LEGS}
        print(json.dumps(dict(state={k: row[k] for k in ("date", "nav", "peak", "drawdown", "cash")},
                              current=current, targets=targets["weights"], drift=drift,
                              band_triggered=bool(targets["traded"]),
                              band=plan_orders.BAND, orders=targets["orders"],
                              problems=problems, safe_to_send=not problems),
                         ensure_ascii=False, indent=2, default=float))
    elif args.command == "book":
        print(json.dumps(book(args.date, args.field), ensure_ascii=False, indent=2, default=float))
    elif args.command == "curve":
        print(json.dumps(curve(), ensure_ascii=False, indent=2, default=float))
    elif args.command == "report":
        print(json.dumps(monthly_report(args.month), ensure_ascii=False, indent=2, default=float))
    elif args.command == "sync-dividends":
        added = sync_dividends(args.start, args.end)
        print(json.dumps(dict(added=len(added), records=added), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
