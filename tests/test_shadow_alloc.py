"""The allocation shadow books must be the study's engine, kept honest by a parity pin.

``shadow_alloc`` replays ``equity_bond_tail.run_arm`` forward one session at a time, so the one
failure mode that would quietly corrupt the 2027 judgment is drift between the two engines.  The
tests here pin them together the way the variant books pin to the main E1 ledger: an incremental
append, fed the same frame day by day and reloading its state from its own log between runs,
must reproduce the batch simulation's nav, turnover and cost bit for bit.  Everything else --
establishment cost, idempotence, the trading-day gate, the honest ``behind`` row -- follows the
house conventions the S2 chain already lives by.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tools import equity_bond_tail as ebt
from tools import shadow_alloc as sa


def synthetic_frame(days=620, seed=17):
    """Four legs with very different vols so the optimised arms actually move."""
    generator = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=days)
    vols = {"510300": 0.012, "510500": 0.016, "513500": 0.011, "511010": 0.0004}
    columns = {code: np.cumprod(1.0 + generator.normal(0.0003, vol, size=days))
               for code, vol in vols.items()}
    return pd.DataFrame(columns, index=index)


def open_and_run(arm, frame, start, tmp_path, today=None):
    """Establish at ``start`` then advance day by day, reloading state from the log each run."""
    monkey_open = dict(sa.__dict__)                     # OPEN_DATE is module-frozen; patch it
    sa.OPEN_DATE = str(pd.Timestamp(start).date())
    try:
        sa.establish(arm, frame, output=tmp_path, log=lambda *_: None)
        for horizon in range(frame.index.get_loc(start) + 1, len(frame.index)):
            sa.advance(arm, frame.iloc[:horizon + 1], output=tmp_path, today=today,
                       log=lambda *_: None)
    finally:
        sa.OPEN_DATE = monkey_open["OPEN_DATE"]
    return sa.read_book(arm, tmp_path)


def test_incremental_appends_reproduce_the_study_engine_exactly(tmp_path):
    frame = synthetic_frame(days=900)                 # fixed6040 needs 504 + 252 + runway
    start = frame.index[ebt.ESTIMATE_WINDOW]             # first date owning the full warmup
    for arm in ("fixed6040", "minvar"):
        reference = ebt.run_arm(arm, frame, start)
        rows = [row for row in open_and_run(arm, frame, start, tmp_path)
                if row["kind"] == "decision"]
        assert len(rows) == len(reference["series"])
        assert rows[0]["action"] == "establish"
        assert rows[-1]["nav"] == pytest.approx(float(reference["series"].iloc[-1]), rel=1e-12)
        assert rows[-1]["turnover_total"] == pytest.approx(reference["turnover"])
        assert rows[-1]["cost_total"] == pytest.approx(reference["cost"])
        # every intermediate day agrees too, not just the endpoint
        for row, value in zip(rows, reference["series"]):
            assert row["nav"] == pytest.approx(float(value), rel=1e-12)


def test_establishment_books_the_cost_and_the_frozen_weights(tmp_path):
    frame = synthetic_frame()
    arm = "fixed6040"
    sa.OPEN_DATE = str(frame.index[ebt.ESTIMATE_WINDOW].date())
    try:
        record = sa.establish(arm, frame, output=tmp_path, log=lambda *_: None)
    finally:
        sa.OPEN_DATE = "2026-11-20"
    assert record["steps"] == 0
    assert record["nav"] == pytest.approx(1.0 - ebt.COST_PER_SIDE)
    assert record["turnover_total"] == pytest.approx(1.0)
    assert record["cost_total"] == pytest.approx(ebt.COST_PER_SIDE)
    assert record["rebalance_every"] == ebt.REBALANCE[arm]
    assert record["capital"] == pytest.approx(sa.CAPITAL)
    expected = ebt.static_target(arm)
    assert np.allclose([record["weights"][code] for code in ebt.CODES], expected)


def test_state_is_the_log_not_a_side_file(tmp_path):
    frame = synthetic_frame()
    arm = "minvar"
    rows = [row for row in open_and_run(arm, frame, frame.index[ebt.ESTIMATE_WINDOW], tmp_path)
            if row["kind"] == "decision"]
    assert not (tmp_path / "minvar_state.json").exists()      # no second truth anywhere
    # a status row after the last decision must not become the state
    sa.append(arm, dict(recorded_at="x", arm=arm, kind="status", status="stale"), tmp_path)
    state = sa.state_from(sa.read_book(arm, tmp_path))
    assert state["price_date"] == rows[-1]["price_date"]
    assert state["nav"] == rows[-1]["nav"]


def test_a_repeated_run_appends_status_not_a_second_decision(tmp_path, monkeypatch):
    frame = synthetic_frame()
    arm = "cvar"
    start = frame.index[ebt.ESTIMATE_WINDOW]
    rows = open_and_run(arm, frame, start, tmp_path)
    decisions_before = len([row for row in rows if row["kind"] == "decision"])

    from tools import trading_calendar
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda today=None: True)
    again = sa.advance(arm, frame, output=tmp_path, today=frame.index[-1].date(),
                       log=lambda *_: None)
    book = sa.read_book(arm, tmp_path)
    decisions_after = len([row for row in book if row["kind"] == "decision"])
    assert decisions_after == decisions_before
    assert book[-1]["kind"] == "status" and book[-1]["status"] == "stale"
    assert again["status"] == "stale"


def test_nontrading_days_append_nothing(tmp_path, monkeypatch):
    frame = synthetic_frame()
    arm = "fixed6040"
    start = frame.index[ebt.ESTIMATE_WINDOW]
    open_and_run(arm, frame, start, tmp_path)
    before = sa.read_book(arm, tmp_path)

    from tools import trading_calendar
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda today=None: False)
    sa.advance(arm, frame, output=tmp_path, today=date(2026, 11, 28), log=lambda *_: None)
    after = sa.read_book(arm, tmp_path)
    assert after == before                        # a holiday is not news; nothing is appended


def test_a_missing_bar_gets_an_honest_behind_row(tmp_path, monkeypatch):
    frame = synthetic_frame()
    arm = "fixed6040"
    start = frame.index[ebt.ESTIMATE_WINDOW]
    open_and_run(arm, frame, start, tmp_path)
    saturday = date(2026, 11, 28)                 # frame ends on a bdate before it

    from tools import trading_calendar
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda today=None: True)
    result = sa.advance(arm, frame, output=tmp_path, today=saturday, log=lambda *_: None)
    book = sa.read_book(arm, tmp_path)
    assert result["status"] == "behind"
    assert book[-1]["kind"] == "status" and book[-1]["status"] == "behind"
    decisions = [row for row in book if row["kind"] == "decision"]
    assert decisions[-1]["price_date"] == str(frame.index[-1].date())


def test_an_unopened_book_reports_instead_of_inventing_rows(tmp_path):
    frame = synthetic_frame()
    result = sa.advance("minvar", frame, output=tmp_path, today=None, log=lambda *_: None)
    assert result["status"] == "unopened"
    assert sa.read_book("minvar", tmp_path) == []


def test_catch_up_after_a_gap_appends_one_row_per_missed_session(tmp_path):
    frame = synthetic_frame()
    arm = "minvar"
    start = frame.index[ebt.ESTIMATE_WINDOW]
    sa.OPEN_DATE = str(start.date())
    try:
        sa.establish(arm, frame, output=tmp_path, log=lambda *_: None)
    finally:
        sa.OPEN_DATE = "2026-11-20"
    # the book sits at `start`; the cache then jumps five sessions ahead in one refresh
    horizon = frame.index.get_loc(start) + 5
    result = sa.advance(arm, frame.iloc[:horizon + 1], output=tmp_path, log=lambda *_: None)
    rows = [row for row in sa.read_book(arm, tmp_path) if row["kind"] == "decision"]
    assert result["appended"] == 5
    assert len(rows) == 6                          # establishment + five sessions
    assert [row["steps"] for row in rows] == [0, 1, 2, 3, 4, 5]


def test_establish_refuses_an_already_open_book(tmp_path):
    frame = synthetic_frame()
    sa.OPEN_DATE = str(frame.index[ebt.ESTIMATE_WINDOW].date())
    try:
        sa.establish("fixed6040", frame, output=tmp_path, log=lambda *_: None)
        with pytest.raises(SystemExit):
            sa.establish("fixed6040", frame, output=tmp_path, log=lambda *_: None)
    finally:
        sa.OPEN_DATE = "2026-11-20"
    assert len(sa.read_book("fixed6040", tmp_path)) == 1


def test_records_round_trip_through_json_without_losing_state(tmp_path):
    """Weights and nav must survive the write-read cycle, or parity is fiction by drift."""
    frame = synthetic_frame()
    arm = "minvar"
    rows = open_and_run(arm, frame, frame.index[ebt.ESTIMATE_WINDOW], tmp_path)
    for row in rows:
        reloaded = json.loads(json.dumps(row))
        assert reloaded["nav"] == row["nav"]
        assert reloaded["weights"] == row["weights"]
