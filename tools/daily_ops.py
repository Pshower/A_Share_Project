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
from pathlib import Path

import pandas as pd

from tools import account_ledger as ledger
from tools import daily_allocation, trading_calendar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "daily_ops"
DEFAULT_ACCOUNTS = ("live", "paper")


def refresh_prices() -> dict:
    """Pull every carrier's history into the local cache; offline runs fail here, loudly."""
    from tools import fund_vehicles
    for code in fund_vehicles.CARRIERS:
        fund_vehicles.load_prices(code, refresh=True)
        fund_vehicles.load_distributions(code, refresh=True)
    return dict(carriers=len(fund_vehicles.CARRIERS))


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


def run(day: str | None = None, accounts=DEFAULT_ACCOUNTS, refresh: bool = True,
        dry: bool = False) -> dict:
    """The whole day as one unit.  Returns the run record; ``status`` is ok / closed / failed."""
    day = str(day or pd.Timestamp.now().date())
    report = dict(day=day, accounts=list(accounts), dry=bool(dry), steps=[],
                  status="ok", bookings={}, curves={}, summary="", error=None)

    def note(step: str, payload=None, error: str | None = None) -> None:
        report["steps"].append(dict(step=step, ok=error is None, detail=payload, error=error))

    if not trading_calendar.is_trading_day(day):
        report["status"] = "closed"
        report["summary"] = "%s 休市，未做任何事" % day
        return report

    try:
        if refresh and not dry:
            note("refresh", refresh_prices())
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
            raise RuntimeError("预检未通过：%s；已停在落账之前" % "、".join(blocked))

        for account in accounts:
            if dry:
                # A dry run still answers "what would happen": the recommendation is read-only.
                advice = daily_allocation.recommendation(book=account)
                report["bookings"][account] = dict(
                    dry=True, action=advice["action"], trades=0, nav=advice["nav"],
                    risky_target=advice["risky_target"], risky_now=advice["risky_now"])
                continue
            report["bookings"][account] = book(account, day)
            note("book:" + account, report["bookings"][account])

        for account in accounts:
            report["curves"][account] = curve(account)

        if is_month_open(day):
            for account in accounts:
                note("monthly:" + account, dict(skipped="dry run") if dry else monthly(account, day))
    except Exception as error:                                    # noqa: BLE001
        report["status"] = "failed"
        report["error"] = "%s: %s" % (type(error).__name__, str(error)[:400])
        note("abort", error=report["error"])

    report["summary"] = summary_line(report)
    return report


def _nav(report, account) -> dict:
    curve = report["curves"].get(account) or {}
    return curve if isinstance(curve, dict) else {}


def summary_line(report: dict) -> str:
    """One line a human can paste into the execution log."""
    if report["status"] == "closed":
        return report["summary"]
    parts = [report["day"]]
    if report["status"] == "failed":
        parts.append("FAILED")
        parts.append(report["error"] or "unknown")
        return " ".join(parts)
    for account in report["accounts"]:
        booked = report["bookings"].get(account, {})
        curve = _nav(report, account)
        bits = [account]
        if "nav" in booked:
            bits.append("nav %.2f" % booked["nav"])
        if "drawdown" in curve:
            bits.append("回撤 %.2f%%" % (float(curve["drawdown"]) * 100))
        bits.append(booked.get("action", "?"))
        if booked.get("trades"):
            bits.append("%d 笔成交" % booked["trades"])
        parts.append(" ".join(bits))
    return "；".join([parts[0], "，".join(parts[1:])])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", default=None)
    parser.add_argument("--accounts", default=",".join(DEFAULT_ACCOUNTS))
    parser.add_argument("--dry", action="store_true", help="读一遍，不写任何文件")
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args(argv)
    accounts = tuple(name.strip() for name in args.accounts.split(",") if name.strip())
    report = run(day=args.day, accounts=accounts, refresh=not args.no_refresh, dry=args.dry)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if report["status"] != "closed":
        (OUTPUT / ("%s.json" % report["day"])).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("SUMMARY:", report["summary"])
    for step in report["steps"]:
        if not step["ok"]:
            print("  FAILED STEP:", step["step"], "-", step["error"])
    return 0 if report["status"] in {"ok", "closed"} else 2


if __name__ == "__main__":
    sys.exit(main())
