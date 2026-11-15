"""The daily advice must be arithmetic on the book, not a second opinion.

The important properties: the band decides whether anything is booked, a book read leaves no
global state behind, and a dry run never touches the ledger.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tools import daily_allocation as daily
from tools import plan_orders


def write_prices(root, scale=1.0, days=30, codes=("510300", "510500", "513500", "511010")):
    base = {"510300": 4.0, "510500": 6.0, "513500": 2.0, "511010": 100.0, "511880": 100.0}
    index = pd.bdate_range("2026-11-02", periods=days)
    (root / "fund").mkdir(parents=True, exist_ok=True)
    for code in codes:
        frame = pd.DataFrame({"date": index.strftime("%Y-%m-%d"),
                              "close": base[code] * scale})
        frame.to_csv(root / "fund" / ("hist_%s.csv" % code), index=False)


def write_book(root, name="paper", drop=1.0):
    base = root / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "account.json").write_text(json.dumps({"capital": 1000000.0,
                                                   "start_date": "2026-11-02"}), encoding="utf-8")
    trades = [
        dict(date="2026-11-02", code="510300", action="buy", shares=37500, price=4.0, fee=5.0),
        dict(date="2026-11-02", code="510500", action="buy", shares=25000, price=6.0, fee=5.0),
        dict(date="2026-11-02", code="513500", action="buy", shares=150000, price=2.0, fee=5.0),
    ]
    (base / "trades.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trades), encoding="utf-8")
    snapshot = dict(date="2026-11-02", nav=1000000.0, cash=400000.0,
                    positions={"510300": 37500, "510500": 25000, "513500": 150000},
                    prices={"510300": 4.0, "510500": 6.0, "513500": 2.0})
    (base / "snapshots.jsonl").write_text(json.dumps(snapshot) + "\n", encoding="utf-8")


def test_cppI_weights_are_sixty_percent_at_the_peak_and_zero_at_the_floor():
    weights, risky = daily.weights_for(1.0, 1.0)
    assert risky == pytest.approx(0.60)
    assert weights["511010"] == pytest.approx(0.40)
    assert weights["510300"] == pytest.approx(0.15)
    floor_weights, floor_risky = daily.weights_for(0.90, 1.0)
    assert floor_risky == pytest.approx(0.0)
    assert floor_weights["511010"] == pytest.approx(1.0)


def test_the_cash_half_of_the_safe_leg_is_a_split_not_a_flag():
    weights, risky = daily.weights_for(1.0, 1.0, cash_safe=True)
    assert weights["511880"] == pytest.approx(0.20)
    assert weights["511010"] == pytest.approx(0.20)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_cached_prices_use_the_last_common_session(tmp_path):
    write_prices(tmp_path)
    frame = pd.read_csv(tmp_path / "fund" / "hist_510500.csv")
    frame.iloc[:-3].to_csv(tmp_path / "fund" / "hist_510500.csv", index=False)
    prices, stamp, dates = daily.cached_prices(["510300", "510500"], tmp_path)
    assert set(dates.values()) == {stamp}
    assert stamp == pd.to_datetime(frame["date"].iloc[-4])     # the shorter history wins


def test_a_book_far_from_its_target_is_told_to_rebalance(tmp_path):
    write_prices(tmp_path / "online", scale=0.9)
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    assert result["action"] == "rebalance"
    assert result["risky_target"] < result["risky_now"] - result["band"]
    sells = [row for row in result["sleeves"] if row["delta"] < 0]
    assert sells, "a cut should sell something"


def test_a_book_inside_the_band_is_told_to_hold(tmp_path):
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    assert result["action"] == "hold"
    assert abs(result["drift"]) < result["band"]


def test_the_asset_table_covers_every_class_and_sums_to_one(tmp_path):
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    names = [row["name"] for row in result["assets"]]
    assert "黄金" in names and "存款" in names and "银行理财" in names
    total = sum(row["weight"] for row in result["assets"])
    assert total == pytest.approx(1.0, abs=0.01)


def test_a_dry_run_records_nothing(tmp_path, monkeypatch):
    write_prices(tmp_path / "online", scale=0.9)
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    before = (tmp_path / "paper" / "trades.jsonl").read_text(encoding="utf-8")
    log = daily.apply(result, dry=True)
    after = (tmp_path / "paper" / "trades.jsonl").read_text(encoding="utf-8")
    assert before == after
    assert log["marked_nav"] is None
    assert len(log["trades"]) == len([row for row in result["sleeves"] if row["delta"] != 0])


def test_the_read_path_leaves_the_ledger_globals_alone(tmp_path):
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path)
    trades_before = daily.ledger.TRADES
    daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    assert daily.ledger.TRADES == trades_before


def test_apply_refuses_a_quote_that_stopped_updating(tmp_path):
    """The scheduled job must fail loudly rather than trade on last week's prices."""
    write_prices(tmp_path / "online", scale=0.9)
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    result["as_of"] = "2026-01-05"
    before = (tmp_path / "paper" / "trades.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="拒绝"):
        daily.apply(result, max_age_days=daily.MAX_AGE_DAYS)
    assert (tmp_path / "paper" / "trades.jsonl").read_text(encoding="utf-8") == before
