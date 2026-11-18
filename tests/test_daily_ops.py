"""The daily run is one unit: a failure anywhere must stop it before anything is booked."""
from __future__ import annotations

import pytest

from tools import daily_ops as ops


def green(monkeypatch, advice_action="hold"):
    calls = dict(refresh=0, prepare=[], check=[], book=[], curve=[], monthly=[])
    monkeypatch.setattr(ops.trading_calendar, "is_trading_day", lambda day: True)
    monkeypatch.setattr(ops, "is_month_open", lambda day: False)
    monkeypatch.setattr(ops, "refresh_prices", lambda: calls.__setitem__("refresh", calls["refresh"] + 1)
                        or dict(carriers=9))
    monkeypatch.setattr(ops, "prepare", lambda account, day: calls["prepare"].append(account)
                        or dict(dividends=0, nav=1000.0, cash=10.0))
    monkeypatch.setattr(ops, "check", lambda account, day: calls["check"].append(account)
                        or dict(safe_to_send=True, band_triggered=False, state=dict(nav=1000.0)))
    monkeypatch.setattr(ops, "book", lambda account, day: calls["book"].append(account)
                        or dict(action=advice_action, trades=0, nav=1000.0))
    monkeypatch.setattr(ops, "curve", lambda account: calls["curve"].append(account)
                        or dict(drawdown=-0.01))
    monkeypatch.setattr(ops, "regime", lambda: dict(y10=1.5874, as_of="2026-11-09",
                                                    bucket="低 carry", daily_median=0.0067,
                                                    monthly_median=0.0124))
    monkeypatch.setattr(ops, "calendar_left", lambda: 42)
    monkeypatch.setattr(ops, "health", lambda: dict(panel="pool_test", missing=3, jumps=0,
                                                    snapshots=500))
    return calls


def test_a_closed_day_touches_nothing(monkeypatch):
    monkeypatch.setattr(ops.trading_calendar, "is_trading_day", lambda day: False)
    monkeypatch.setattr(ops, "prepare", lambda *a: pytest.fail("must not prepare on a holiday"))
    report = ops.run(day="2026-11-14")
    assert report["status"] == "closed"
    assert report["steps"] == []
    assert "休市" in report["summary"]


def test_every_account_is_checked_before_any_is_booked(monkeypatch):
    calls = green(monkeypatch)
    seen = []
    original = ops.book

    def spy(account, day):
        seen.append(list(calls["check"]))          # what had been checked when booking started
        return original(account, day)

    monkeypatch.setattr(ops, "book", spy)
    report = ops.run(day="2026-11-13")
    assert report["status"] == "ok"
    assert seen[0] == ["live", "paper"]            # both checked before the first booking
    assert calls["book"] == ["paper"]              # only the paper book is written


def test_the_live_book_is_never_written_by_the_model(monkeypatch):
    """The repository's first invariant: a model fill never enters the real-money book."""
    calls = green(monkeypatch)
    monkeypatch.setattr(ops.daily_allocation, "recommendation",
                        lambda book="paper", cash_safe=False: dict(
                            action="rebalance", nav=1000.0, risky_target=0.6, risky_now=0.0,
                            as_of="2026-11-13", sleeves=[dict(delta=100)]))
    report = ops.run(day="2026-11-13")
    assert "live" not in calls["book"]
    assert report["bookings"]["live"]["advisory"] is True
    assert report["bookings"]["live"]["trades"] == 0
    assert "仅建议" in report["summary"]


def test_one_book_failing_the_gate_stops_before_any_booking(monkeypatch):
    calls = green(monkeypatch)
    monkeypatch.setattr(ops, "check", lambda account, day: dict(
        safe_to_send=(account == "live"), band_triggered=True, state=dict(nav=1000.0)))
    report = ops.run(day="2026-11-13")
    assert report["status"] == "failed"
    assert calls["book"] == []
    assert "预检未通过" in report["error"]


def test_a_failure_while_preparing_aborts_the_run(monkeypatch):
    calls = green(monkeypatch)

    def explode(account, day):
        if account == "paper":
            raise ValueError("摘要文件缺失")
        return dict(dividends=0, nav=1000.0, cash=1.0)

    monkeypatch.setattr(ops, "prepare", explode)
    report = ops.run(day="2026-11-13")
    assert report["status"] == "failed"
    assert calls["book"] == []
    assert any(step["step"] == "abort" for step in report["steps"])


def test_a_dry_run_never_books(monkeypatch):
    calls = green(monkeypatch)
    monkeypatch.setattr(ops.daily_allocation, "recommendation",
                        lambda book="paper", cash_safe=False: dict(
                            action="hold", nav=1000.0, risky_target=0.5, risky_now=0.51,
                            as_of="2026-11-13"))
    report = ops.run(day="2026-11-13", dry=True)
    assert report["status"] == "ok"
    assert calls["book"] == []
    assert calls["prepare"] == []
    assert report["bookings"]["live"]["dry"] is True


def test_the_summary_names_each_account_and_its_action(monkeypatch):
    green(monkeypatch)
    report = ops.run(day="2026-11-13")
    assert "live" in report["summary"] and "paper" in report["summary"]
    assert "hold" in report["summary"]
    assert "FAILED" not in report["summary"]


def test_the_summary_names_the_rate_regime(monkeypatch):
    """A 4.4% median came from 2-3% yields; a 1.6% day must say so on the line itself."""
    green(monkeypatch)
    report = ops.run(day="2026-11-13")
    assert "低 carry" in report["summary"]
    assert "0.67%" in report["summary"] and "1.24%" in report["summary"]


def test_the_summary_counts_down_the_calendar(monkeypatch):
    """When the calendar runs out every run fails; the countdown must appear before it bites."""
    green(monkeypatch)
    report = ops.run(day="2026-11-13")
    assert "日历剩 42 天" in report["summary"]


def test_a_distant_calendar_is_not_mentioned(monkeypatch):
    green(monkeypatch)
    monkeypatch.setattr(ops, "calendar_left", lambda: 300)
    report = ops.run(day="2026-11-13")
    assert "日历剩" not in report["summary"]


def test_a_missing_yield_cache_never_fails_the_day(monkeypatch):
    calls = green(monkeypatch)

    def broken():
        raise FileNotFoundError("yield cache gone")

    monkeypatch.setattr(ops, "regime", broken)
    report = ops.run(day="2026-11-13")
    assert report["status"] == "ok"
    assert report["regime"] is None
    assert "carry" not in report["summary"]
    assert calls["book"] == ["paper"]


def test_the_live_book_cannot_be_booked_even_when_asked(monkeypatch):
    """The gate must be a policy, not a parameter: ``book("live")`` used to pass allowed=("live",)."""
    monkeypatch.setattr(ops.daily_allocation, "recommendation",
                        lambda book="paper", cash_safe=False: dict(
                            book=book, as_of="2026-11-13", action="hold", sleeves=[],
                            nav=1.0, risky_target=0.5, risky_now=0.5, prices={}, price_dates={}))
    with pytest.raises(ValueError, match="不允许写进"):
        ops.book("live", "2026-11-13")


def test_a_blocked_gate_carries_its_reasons(monkeypatch):
    green(monkeypatch)
    monkeypatch.setattr(ops, "check", lambda account, day: dict(
        safe_to_send=False, band_triggered=False, state=dict(nav=1.0),
        problems=["510300 quote is from 2026-11-16, not 2026-11-17"]))
    report = ops.run(day="2026-11-13")
    assert report["status"] == "failed"
    assert "510300" in report["error"], "the gate's own reason must survive into the failure message"


def test_a_broken_health_report_is_a_note_not_a_failure(monkeypatch):
    """Observability must never be the reason a day does not get booked."""
    calls = green(monkeypatch)

    def broken():
        raise OSError("archive unreadable")

    monkeypatch.setattr(ops, "health", broken)
    report = ops.run(day="2026-11-13")
    assert report["status"] == "ok"
    assert calls["book"] == ["paper"]
    step = next(item for item in report["steps"] if item["step"] == "health")
    assert step["ok"] is False and "unreadable" in step["error"]


def test_a_failed_summary_says_so(monkeypatch):
    green(monkeypatch)
    monkeypatch.setattr(ops, "check", lambda account, day: dict(
        safe_to_send=False, band_triggered=False, state=dict(nav=1.0)))
    report = ops.run(day="2026-11-13")
    assert "FAILED" in report["summary"]
