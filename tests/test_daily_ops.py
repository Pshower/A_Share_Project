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
    assert calls["book"] == ["live", "paper"]


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


def test_a_failed_summary_says_so(monkeypatch):
    green(monkeypatch)
    monkeypatch.setattr(ops, "check", lambda account, day: dict(
        safe_to_send=False, band_triggered=False, state=dict(nav=1.0)))
    report = ops.run(day="2026-11-13")
    assert "FAILED" in report["summary"]
