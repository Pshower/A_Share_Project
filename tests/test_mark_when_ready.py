"""Waiting for the session to be published is the only honest way to mark a day late.

The 20:15 job cannot always price the day it is run for, and nothing re-runs it afterwards, so a
day marked with yesterday's close looked flat while an unmarked day looked like a gap in the
curve.  ``mark_when_ready`` polls until the quotes are current and then marks -- these tests pin
that it marks exactly once, at the right date, and gives up instead of inventing a price.
"""
from __future__ import annotations

import json

from tools import mark_when_ready as watcher

PRICES = {"511010": 141.363, "510300": 4.114, "510500": 6.628, "513500": 2.649}


def _isolated(monkeypatch, tmp_path, name="paper"):
    """Point the ledger at a scratch account: this file used to write into the real books.

    The backtest book's 2026-11-13 row -- prices from 11-12, stamped as 11-13, cash from a replay
    that no export had produced -- was written by these tests, not by the daily job.
    """
    base = tmp_path / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "account.json").write_text(json.dumps(dict(capital=1_000_000.0,
                                                       start_date="2026-11-11")), encoding="utf-8")
    monkeypatch.setattr(watcher.ledger, "ACCOUNT_DIR", tmp_path)
    watcher.ledger.use(name)
    return name


def _stub(monkeypatch, sequence):
    """Feed ``quotes_for`` a scripted list of (prices, dates) pairs."""
    calls = []

    def fake(stamp):
        calls.append(stamp)
        return sequence[min(len(calls), len(sequence)) - 1]

    monkeypatch.setattr(watcher, "quotes_for", fake)
    monkeypatch.setattr(watcher.time, "sleep", lambda _seconds: None)
    return calls


def test_it_marks_as_soon_as_the_session_is_published(monkeypatch, tmp_path):
    lagging = (dict(PRICES), {code: "2026-11-12" for code in PRICES})
    current = (dict(PRICES), {code: "2026-11-13" for code in PRICES})
    calls = _stub(monkeypatch, [lagging, lagging, current])
    row = watcher.wait_and_mark("2026-11-13", account=_isolated(monkeypatch, tmp_path), minutes=45, interval=0,
                                log=lambda *_: None)
    assert row["status"] == "marked" and row["attempts"] == 3
    assert len(calls) == 3


def test_it_gives_up_rather_than_price_the_day_off_yesterday(monkeypatch, tmp_path):
    lagging = (dict(PRICES), {code: "2026-11-12" for code in PRICES})
    _stub(monkeypatch, [lagging])
    row = watcher.wait_and_mark("2026-11-13", account=_isolated(monkeypatch, tmp_path), minutes=0, interval=0,
                                log=lambda *_: None)
    assert row["status"] == "still_unpublished"
    assert row["lagging"] == {code: "2026-11-12" for code in PRICES}
    assert "leave the day unmarked" in row["note"]


def test_the_mark_lands_on_the_requested_date(monkeypatch, tmp_path):
    current = (dict(PRICES), {code: "2026-11-13" for code in PRICES})
    _stub(monkeypatch, [current])
    written = {}

    def fake_mark(stamp, **kwargs):
        written.update(stamp=stamp, prices=kwargs.get("prices"), dates=kwargs.get("price_dates"))
        return dict(nav=1.0, cash=1.0, positions={}, prices={})

    monkeypatch.setattr(watcher.ledger, "mark", fake_mark)
    row = watcher.wait_and_mark("2026-11-13", account=_isolated(monkeypatch, tmp_path), minutes=0, interval=0,
                                log=lambda *_: None)
    assert row["status"] == "marked"
    assert written["stamp"] == "2026-11-13" and written["dates"]["510300"] == "2026-11-13"


def test_main_reports_the_unmarked_day_as_a_failure(monkeypatch, capsys):
    lagging = (dict(PRICES), {code: "2026-11-12" for code in PRICES})
    _stub(monkeypatch, [lagging])
    code = watcher.main(["--date", "2026-11-13", "--account", "backtest",
                         "--minutes", "0", "--interval", "0"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "still_unpublished"
