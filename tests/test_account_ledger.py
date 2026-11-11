"""The account ledger is only useful if it refuses the things that would lose money.

Each test here corresponds to a way a real account tool goes wrong: a mistyped lot, a sale of
shares that are not held, a peak that quietly resets, a stale quote treated as today's price,
or a risky leg still being bought after the floor was breached.
"""
from __future__ import annotations

import pytest

from tools import account_ledger as ledger
from tools import plan_orders


def write_trades(tmp_path, rows):
    path = tmp_path / "trades.jsonl"
    for row in rows:
        ledger._append(path, row)
    return path


def test_replay_charges_the_commission_floor_and_the_stamp_duty():
    trades = [dict(code="510300", action="buy", shares=100, price=4.0)]
    book = ledger.replay(trades, 10_000.0)
    assert book["cash"] == pytest.approx(10_000.0 - 400.0 - 5.0)      # floor commission
    assert book["positions"] == {"510300": 100}
    sell = ledger.replay(trades + [dict(code="510300", action="sell", shares=100, price=4.1)], 10_000.0)
    assert sell["positions"] == {}


def test_nav_is_cash_plus_positions_and_needs_every_price():
    assert ledger.nav({"510300": 100}, 500.0, {"510300": 4.0}) == pytest.approx(900.0)
    with pytest.raises(ValueError, match="No price"):
        ledger.nav({"510300": 100}, 500.0, {})


def test_the_peak_never_goes_down():
    snapshots = [dict(date="2026-11-10", nav=3_050_000.0), dict(date="2026-11-11", nav=2_900_000.0)]
    assert ledger.peak_of(snapshots, 3_000_000.0) == pytest.approx(3_050_000.0)
    assert ledger.peak_of([], 3_000_000.0) == pytest.approx(3_000_000.0)


def test_init_refuses_to_overwrite_an_account(tmp_path):
    path = tmp_path / "account.json"
    ledger.init(3_000_000, "2026-11-11", path=path)
    with pytest.raises(ValueError, match="already initialised"):
        ledger.init(1_000_000, "2026-11-12", path=path)


def test_a_trade_must_be_whole_lots(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "CONFIG", tmp_path / "account.json")
    ledger.init(3_000_000, "2026-11-11", path=tmp_path / "account.json")
    with pytest.raises(ValueError, match="multiple of 100"):
        ledger.record_trade("2026-11-12", "510300", "buy", 150, 4.0,
                            path=tmp_path / "trades.jsonl", cfg=dict(capital=3_000_000.0))


def test_you_cannot_sell_what_you_do_not_hold(tmp_path):
    config = dict(capital=3_000_000.0)
    path = write_trades(tmp_path, [dict(code="510300", action="buy", shares=100, price=4.0, fee=5.0)])
    with pytest.raises(ValueError, match="only 100 held"):
        ledger.record_trade("2026-11-12", "510300", "sell", 200, 4.1,
                            path=path, cfg=config)


def test_you_cannot_spend_cash_you_do_not_have(tmp_path):
    path = tmp_path / "trades.jsonl"
    with pytest.raises(ValueError, match="only .* cash"):
        ledger.record_trade("2026-11-12", "510300", "buy", 1_000_000, 4.0, path=path,
                            cfg=dict(capital=1_000.0))


def test_the_gate_blocks_a_stale_quote():
    state = dict(date="2026-11-12", nav=3_000_000.0, peak=3_000_000.0, cash=1e6)
    problems = ledger.gate([], {"510300": 4.0}, {"510300": "2026-11-11"}, state)
    assert any("not 2026-11-12" in text for text in problems)


def test_the_gate_blocks_risky_buys_below_the_floor():
    state = dict(date="2026-11-12", nav=2_690_000.0, peak=3_000_000.0, cash=1e6)   # below 0.9 peak
    orders = [dict(code="510300", delta=100, price=4.0)]
    problems = ledger.gate(orders, {"510300": 4.0}, {}, state)
    assert any("floor" in text for text in problems)


def test_the_gate_blocks_an_order_that_does_not_fit_in_cash():
    state = dict(date="2026-11-12", nav=3_000_000.0, peak=3_000_000.0, cash=100.0)
    orders = [dict(code="510300", delta=100, price=4.0)]
    problems = ledger.gate(orders, {"510300": 4.0}, {}, state)
    assert any("more cash" in text for text in problems)


def test_the_gate_passes_a_clean_order():
    state = dict(date="2026-11-12", nav=3_000_000.0, peak=3_000_000.0, cash=1_000_000.0)
    orders = [dict(code="510300", delta=100, price=4.0)]
    assert ledger.gate(orders, {"510300": 4.0}, {"510300": "2026-11-12"}, state) == []


def test_slippage_is_signed_so_positive_is_adverse():
    trades = [dict(code="510300", action="buy", shares=100, price=4.04, decision_price=4.0),
              dict(code="510500", action="buy", shares=100, price=3.98, decision_price=4.0)]
    stats = ledger.slippage(trades)
    assert stats["trades"] == 2
    assert stats["mean_bps"] == pytest.approx(25.0)       # +100bps and -50bps average
    assert stats["share_adverse"] == pytest.approx(0.5)


def test_monthly_report_needs_snapshots():
    assert ledger.monthly_report("2026-11", snapshots=[], trades=[],
                                 cfg=dict(capital=3_000_000.0))["status"] == "no_snapshots"


def test_monthly_report_summarises_a_month():
    snapshots = [dict(date="2026-11-02", nav=3_000_000.0, cash=0.0, prices={"510300": 4.0},
                      positions={"510300": 750_000}),
                 dict(date="2026-11-30", nav=3_030_000.0, cash=0.0, prices={"510300": 4.04},
                      positions={"510300": 750_000})]
    report = ledger.monthly_report("2026-11", snapshots=snapshots, trades=[],
                                   cfg=dict(capital=3_000_000.0))
    assert report["status"] == "ready"
    assert report["return_pct"] == pytest.approx(0.01)
    assert report["leg_change"]["510300"] == pytest.approx(30_000.0)


def test_the_state_flags_trades_that_have_not_been_marked():
    trades = [dict(code="510300", action="buy", shares=100, price=4.0, fee=5.0, date="2026-11-12")]
    snapshots = [dict(date="2026-11-10", nav=3_000_000.0)]
    row = ledger.state(dates="2026-11-12", prices={"510300": 4.0}, trades=trades,
                       snapshots=snapshots, cfg=dict(capital=3_000_000.0, start_date="2026-11-01"))
    assert row["unmarked_trades"] == 1 and row["nav_source"] == "prices"
