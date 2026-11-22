"""一个入口跑完整个每日流程：闸门 → 刷新行情 → 分红/记账 → 预检 → 落账 → 曲线 → 月报。

原来一天要顺序执行六条命令，任何一条挂掉，后面几条照样会跑——于是可能出现
"live 落了账、paper 没落"这种半截状态，而退出码还是 0。

这里改成两阶段：

  * **事实先写**（分红、当日快照）——它们是客观发生的事，可重复、幂等；
  * **决策后写**（调仓成交）——只有两本账的预检**全部通过**才动手。

任何一步失败，整轮标记为 failed，退出码非零，**并且不再往下走**。
失败点写进 ``reports/runs/daily_ops/<date>.json``，一眼能看出停在哪。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from tools import account_ledger as ledger
from tools import daily_allocation, trading_calendar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "daily_ops"
DEFAULT_ACCOUNTS = ("live", "paper")
# The repository's first invariant: a model fill never enters the real-money book.  ``live`` is
# prepared, checked and advised on every run -- but only ``paper`` is booked automatically.
# Real fills are recorded by hand with ``account_ledger trade --account live``.
DEFAULT_BOOK_ACCOUNTS = ("paper",)
# Where today's ten-year yield sits, and what the 129-start sweep did from starts in that bucket.
# The medians come from splitting ``plan_backtest_daily/monthly.json`` by the yield on each start
# date (see doc/delivery_20261116.md section 4): 22 starts below 2%, 59 between 2% and 3%, 48 above.
REGIMES = ((0.0, 2.0, "低 carry", 0.0067, 0.0124),
           (2.0, 3.0, "中 carry", 0.0460, 0.0544),
           (3.0, 99.0, "高 carry", 0.0431, 0.0454))


def refresh_prices(pace: float = 0.3) -> dict:
    """Pull every carrier's history into the local cache; offline runs fail here, loudly.

    Eighteen back-to-back requests is rude to a free endpoint, so each carrier is followed by a
    short pause.  The cache writes are atomic, so an interrupted refresh leaves the old file.
    """
    from tools import fund_vehicles
    for code in fund_vehicles.CARRIERS:
        fund_vehicles.load_prices(code, refresh=True)
        fund_vehicles.load_distributions(code, refresh=True)
        if pace:
            time.sleep(pace)
    return dict(carriers=len(fund_vehicles.CARRIERS))


def refresh_premium_cache() -> dict:
    """Best-effort refresh of the QDII premium cache that feeds the advice's circuit breaker.

    A failed fetch must never fail the day: the breaker then speaks from the stale cache and
    dates itself, which the advice shows next to the number.
    """
    from tools import etf_premium
    try:
        return etf_premium.refresh_cache()
    except Exception as exc:                                   # noqa: BLE001
        return dict(error=str(exc),
                    note="溢价缓存未刷新；建议将按旧读数及其日期标注")


def refresh_rates() -> dict:
    """Best-effort refresh of the treasury curve the nightly regime line reads.

    Until 2026-11-23 this file was maintained by hand; a stale curve makes every summary quote
    an outdated bucket, so it belongs in the nightly refresh, not in a weekly chore.
    """
    from tools import rates
    try:
        return rates.refresh()
    except Exception as exc:                                   # noqa: BLE001
        return dict(error=str(exc), note="利率曲线未刷新；今晚的 regime 档将按旧读数标注")


def prepare(account: str, day: str) -> dict:
    """Facts: distributions the log is missing, then the day's mark."""
    ledger.use(account)
    added = ledger.sync_dividends()
    row = ledger.mark(day)
    return dict(dividends=len(added), nav=row["nav"], cash=row["cash"])


def check(account: str, day: str) -> dict:
    """Read-only pre-trade gate, the same one the CLI prints."""
    ledger.use(account)
    return ledger.check(day)


def book(account: str, day: str) -> dict:
    """Decision: book the allocation (or just re-mark when the band says hold)."""
    result = daily_allocation.recommendation(book=account)
    if str(result["as_of"]) != str(day):
        raise RuntimeError("报价停在 %s，不是 %s" % (result["as_of"], day))
    # The gate is a policy, not a parameter of the caller: passing the account it is about would
    # make it vacuous.  ``allowed`` stays at its default ("paper"), so the model can never write
    # the real-money book even if ``book_accounts`` is misconfigured.
    log = daily_allocation.apply(result)
    return dict(action=log["action"], trades=len(log["trades"]),
                nav=log["marked_nav"], risky_target=result["risky_target"],
                risky_now=result["risky_now"], filled=log["trades"])


def curve(account: str) -> dict:
    ledger.use(account)
    return ledger.curve()


def is_month_open(day: str) -> bool:
    """True on the first trading session of a month, when the previous month's statement is due."""
    stamp = pd.Timestamp(day)
    dates = trading_calendar.load()
    window = dates[(dates >= stamp.replace(day=1)) & (dates <= stamp)]
    return bool(len(window)) and window.min() == stamp


def monthly(account: str, day: str) -> dict:
    """The previous month's statement."""
    month = (pd.Timestamp(day).replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
    ledger.use(account)
    return ledger.monthly_report(month)


def run(day: str | None = None, accounts=DEFAULT_ACCOUNTS,
        book_accounts=DEFAULT_BOOK_ACCOUNTS, refresh: bool = True,
        dry: bool = False) -> dict:
    """The whole day as one unit.  Returns the run record; ``status`` is ok / closed / failed."""
    day = str(day or pd.Timestamp.now().date())
    book_accounts = tuple(book_accounts)
    report = dict(day=day, accounts=list(accounts), book_accounts=list(book_accounts),
                  dry=bool(dry), steps=[], status="ok", bookings={}, curves={},
                  summary="", error=None)

    def note(step: str, payload=None, error: str | None = None) -> None:
        report["steps"].append(dict(step=step, ok=error is None, detail=payload, error=error))

    try:
        # The gate is inside the try on purpose: if the calendar cannot be read, that is a failed
        # run that must leave a failure record, not an exception that exits 1 with no JSON.
        if not trading_calendar.is_trading_day(day):
            report["status"] = "closed"
            report["summary"] = "%s 休市，未做任何事" % day
            return report

        if refresh and not dry:
            note("refresh", refresh_prices())
            note("refresh_premium", refresh_premium_cache())
            note("refresh_rates", refresh_rates())
        for account in accounts:
            note("prepare:" + account, dict(skipped="dry run") if dry else prepare(account, day))

        # Every book is checked before any book is booked.
        gates = {}
        for account in accounts:
            gates[account] = check(account, day)
            note("check:" + account, dict(safe_to_send=gates[account]["safe_to_send"],
                                          band_triggered=gates[account]["band_triggered"],
                                          nav=gates[account]["state"]["nav"]))
        blocked = [account for account, gate in gates.items() if not gate["safe_to_send"]]
        if blocked:
            # Carry the gate's own reasons: "预检未通过" alone sends the operator hunting.
            detail = "；".join("%s: %s" % (account,
                                          " / ".join((gates[account].get("problems") or [])[:4]))
                              for account in blocked)
            raise RuntimeError("预检未通过（%s）——%s；已停在落账之前"
                               % ("、".join(blocked), detail))

        for account in accounts:
            if dry or account not in book_accounts:
                # A dry run still answers "what would happen"; a non-booking account (live) gets
                # the same read-only advice and is never written by the model.
                advice = daily_allocation.recommendation(book=account)
                report["bookings"][account] = dict(
                    dry=bool(dry), advisory=account not in book_accounts,
                    action=advice["action"], trades=0, nav=advice["nav"],
                    risky_target=advice["risky_target"], risky_now=advice["risky_now"],
                    fills=len([row for row in advice.get("sleeves", []) if row["delta"] != 0]))
                if account not in book_accounts:
                    note("advise:" + account, report["bookings"][account])
                continue
            report["bookings"][account] = book(account, day)
            note("book:" + account, report["bookings"][account])

        for account in accounts:
            report["curves"][account] = curve(account)

        if is_month_open(day):
            for account in accounts:
                note("monthly:" + account, dict(skipped="dry run") if dry else monthly(account, day))

        # Observability last, and in its own guard: a health report that cannot be produced is a
        # note, never a failed booking.
        try:
            note("health", health())
        except Exception as error:                                # noqa: BLE001
            note("health", error="%s: %s" % (type(error).__name__, str(error)[:200]))
    except Exception as error:                                    # noqa: BLE001
        report["status"] = "failed"
        report["error"] = "%s: %s" % (type(error).__name__, str(error)[:400])
        note("abort", error=report["error"])

    if report["status"] == "ok":
        try:
            report["regime"] = regime()
        except Exception:                                         # noqa: BLE001
            # A missing or stale yield cache is worth a blank clause, never a failed day.
            report["regime"] = None
        report["calendar_days_left"] = calendar_left()

    report["summary"] = summary_line(report)
    return report


def regime() -> dict:
    """Where today's ten-year yield sits, and what the sweep earned from similar starts."""
    from tools import asset_panel
    series = asset_panel.curve()["y10"].dropna()
    if series.empty:
        raise ValueError("no ten-year yield history")
    level = float(series.iloc[-1])
    stamp = str(series.index[-1].date())
    for low, high, label, daily, monthly in REGIMES:
        if low <= level < high:
            return dict(y10=level, as_of=stamp, bucket=label,
                        daily_median=daily, monthly_median=monthly)
    raise ValueError("yield %.4f falls outside every bucket" % level)


def calendar_left() -> int | None:
    """Days of exchange calendar remaining after today; None when the cache cannot be read.

    The calendar ends where the feed ends, and ``is_trading_day`` deliberately raises past it.  That
    is the right failure -- but it is total: from the first session after the cache runs out, every
    daily run fails.  Surfacing the countdown turns a January surprise into a November chore.
    """
    try:
        from tools import trading_calendar
        last = trading_calendar.load().max()
        return int((pd.Timestamp(last).normalize() - pd.Timestamp(pd.Timestamp.now().date())).days)
    except Exception:                                             # noqa: BLE001
        return None


def health() -> dict:
    """Read-only health of the raw archive and the forward panel; never fails the day."""
    from tools import data_health
    payload = data_health.report()
    panel = payload.get("panel") or {}
    return dict(panel=panel.get("version"), missing=panel.get("missing_total"),
                jumps=panel.get("jumps"), snapshots=payload["history"]["snapshots"])


def _nav(report, account) -> dict:
    curve = report["curves"].get(account) or {}
    return curve if isinstance(curve, dict) else {}


def summary_line(report: dict) -> str:
    """One line a human can paste into the execution log."""
    if report["status"] == "closed":
        return report["summary"]
    if report["status"] == "failed":
        return " ".join([report["day"], "FAILED", report["error"] or "unknown"])
    blocks = []
    for account in report["accounts"]:
        booked = report["bookings"].get(account, {})
        curve = _nav(report, account)
        bits = [account + ("（仅建议）" if booked.get("advisory") else "")]
        if "nav" in booked:
            bits.append("nav %.2f" % booked["nav"])
        if "drawdown" in curve:
            bits.append("回撤 %.2f%%" % (float(curve["drawdown"]) * 100))
        bits.append(booked.get("action", "?"))
        if booked.get("trades"):
            bits.append("%d 笔成交" % booked["trades"])
        elif booked.get("advisory") and booked.get("fills"):
            bits.append("%d 笔建议（未落账）" % booked["fills"])
        blocks.append(" ".join(bits))
    parts = [report["day"], "，".join(blocks)]
    rate = report.get("regime") or {}
    if rate.get("bucket"):
        parts.append("%s 档：10y %.2f%%（%s），同档起点中位 %.2f%% 日频 / %.2f%% 月频"
                     % (rate["bucket"], rate["y10"], rate["as_of"],
                        rate["daily_median"] * 100, rate["monthly_median"] * 100))
    left = report.get("calendar_days_left")
    if isinstance(left, int) and left < 60:
        parts.append("日历剩 %d 天" % left)
    return "；".join(parts)


def record_path(day: str, dry: bool = False) -> Path:
    """Where one run's record goes.

    A dry run gets its own file: it must not overwrite the evening's real record, which is the
    only evidence of what was actually booked that day.
    """
    return OUTPUT / ("%s%s.json" % (day, "-dry" if dry else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", default=None)
    parser.add_argument("--accounts", default=",".join(DEFAULT_ACCOUNTS))
    parser.add_argument("--book-accounts", default=",".join(DEFAULT_BOOK_ACCOUNTS),
                        help="which books the model may write to; live must be recorded by hand")
    parser.add_argument("--dry", action="store_true", help="读一遍，不写任何文件")
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args(argv)
    accounts = tuple(name.strip() for name in args.accounts.split(",") if name.strip())
    book_accounts = tuple(name.strip() for name in args.book_accounts.split(",") if name.strip())
    report = run(day=args.day, accounts=accounts, book_accounts=book_accounts,
                 refresh=not args.no_refresh, dry=args.dry)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if report["status"] != "closed":
        record_path(report["day"], dry=args.dry).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("SUMMARY:", report["summary"])
    for step in report["steps"]:
        if not step["ok"]:
            print("  FAILED STEP:", step["step"], "-", step["error"])
    return 0 if report["status"] in {"ok", "closed"} else 2


if __name__ == "__main__":
    sys.exit(main())
