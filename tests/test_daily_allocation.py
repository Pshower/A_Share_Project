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


def write_book(root, name="paper", drop=1.0, legs=(37500, 25000, 150000), extra=None):
    base = root / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "account.json").write_text(json.dumps({"capital": 1000000.0,
                                                   "start_date": "2026-11-02"}), encoding="utf-8")
    trades = [
        dict(date="2026-11-02", code="510300", action="buy", shares=legs[0], price=4.0, fee=5.0),
        dict(date="2026-11-02", code="510500", action="buy", shares=legs[1], price=6.0, fee=5.0),
        dict(date="2026-11-02", code="513500", action="buy", shares=legs[2], price=2.0, fee=5.0),
    ]
    positions = {"510300": legs[0], "510500": legs[1], "513500": legs[2]}
    prices = {"510300": 4.0, "510500": 6.0, "513500": 2.0}
    for code, shares in (extra or {}).items():
        trades.append(dict(date="2026-11-02", code=code, action="buy", shares=shares,
                           price=100.0, fee=5.0))
        positions[code] = shares
        prices[code] = 100.0
    (base / "trades.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trades), encoding="utf-8")
    snapshot = dict(date="2026-11-02", nav=1000000.0, cash=400000.0,
                    positions=positions, prices=prices)
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


def test_a_hold_day_never_proposes_a_trade(tmp_path):
    """The band pins the risky sleeve as a whole; it must not restructure the legs inside it.

    A book that sat at 45/18/57 inside its risky bucket used to be handed a "hold" verdict and,
    in the same response, a 90%-of-NAV reshuffle back to 25/25/50.
    """
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path, legs=(56250, 15000, 142500))       # same 60% risky total, skewed legs
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    assert result["action"] == "hold"
    assert [row["delta"] for row in result["sleeves"]] == [0, 0, 0, 0]
    assert sum(row["fee"] for row in result["sleeves"]) == 0
    assert sum(row["weight"] for row in result["assets"]) == pytest.approx(1.0, abs=0.01)


def test_the_asset_table_covers_every_class_and_sums_to_one(tmp_path):
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    names = [row["name"] for row in result["assets"]]
    assert "黄金" in names and "存款" in names and "银行理财" in names
    total = sum(row["weight"] for row in result["assets"])
    assert total == pytest.approx(1.0, abs=0.01)


def test_cash_safe_does_not_lock_the_book_in_either_mode(tmp_path):
    """Holding 511880 used to raise unless cash_safe, and cash_safe dropped it from the plan.

    A hold day must survive both: the ETF is simply left alone.
    """
    write_prices(tmp_path / "online", scale=1.0,
                 codes=("510300", "510500", "513500", "511010", "511880"))
    write_book(tmp_path, extra={"511880": 2000, "511010": 2000})
    for flag in (True, False):
        result = daily.recommendation("paper", cash_safe=flag,
                                      online=tmp_path / "online", base=tmp_path)
        assert result["action"] == "hold"
        assert [row["delta"] for row in result["sleeves"]] == [0] * len(result["sleeves"])
        assert sum(row["weight"] for row in result["assets"]) == pytest.approx(1.0, abs=0.01)


def test_cash_safe_names_the_cash_etf_in_the_asset_table(tmp_path):
    write_prices(tmp_path / "online", scale=1.0,
                 codes=("510300", "510500", "513500", "511010", "511880"))
    write_book(tmp_path, extra={"511880": 2000, "511010": 2000})
    result = daily.recommendation("paper", cash_safe=True,
                                  online=tmp_path / "online", base=tmp_path)
    names = {row["code"]: row for row in result["assets"]}
    assert "511880" in names and names["511880"]["weight"] > 0


def test_an_off_plan_holding_is_sold_on_a_rebalance_day(tmp_path):
    """Switching cash_safe off used to wedge the book: every rebalance raised instead of selling."""
    write_prices(tmp_path / "online", scale=0.9,
                 codes=("510300", "510500", "513500", "511010", "511880"))
    write_book(tmp_path, extra={"511880": 2000, "511010": 2000})
    result = daily.recommendation("paper", cash_safe=False,
                                  online=tmp_path / "online", base=tmp_path)
    assert result["action"] == "rebalance"
    row = next(item for item in result["sleeves"] if item["code"] == "511880")
    assert row["delta"] == -2000, "the off-plan leg is liquidated, not ignored"
    assert any("计划外" in warning for warning in result["warnings"])


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


def test_a_failed_mark_rolls_the_fills_back(tmp_path, monkeypatch):
    """Fills and the day's snapshot are one decision: a failing mark must leave no fills behind."""
    write_prices(tmp_path / "online", scale=0.9)          # far enough from target to rebalance
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    assert result["action"] == "rebalance"
    trades = tmp_path / "paper" / "trades.jsonl"
    before = trades.read_text(encoding="utf-8")
    monkeypatch.setattr(daily.ledger, "use", lambda name: None)
    monkeypatch.setattr(daily.ledger, "TRADES", trades)
    monkeypatch.setattr(daily.ledger, "SNAPSHOTS", tmp_path / "paper" / "snapshots.jsonl")
    monkeypatch.setattr(daily.ledger, "mark",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mark failed")))
    with pytest.raises(RuntimeError, match="mark failed"):
        daily.apply(result)
    assert trades.read_text(encoding="utf-8") == before


def test_the_read_path_leaves_the_ledger_globals_alone(tmp_path):
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path)
    trades_before = daily.ledger.TRADES
    daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    assert daily.ledger.TRADES == trades_before


def test_apply_refuses_to_write_a_model_fill_into_the_real_money_book(tmp_path):
    """A default is not a gate: ``--book=live --apply`` must fail here, not rely on the caller."""
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    claimed = dict(result, book="live")
    with pytest.raises(ValueError, match="not allowed|不允许"):
        daily.apply(claimed, dry=True)


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


# --------------------------------------------------------------------- QDII 熔断

def write_premium(root, premium=0.1268, day="2026-11-18"):
    (root / "fund").mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([dict(date=day, close=2.253, nav=2.0, premium=premium)])
    frame.to_csv(root / "fund" / "premium_513500.csv", index=False)


def test_the_breaker_threshold_is_the_studied_p95():
    """etf_premium_20261114: the post-2024 premium's p95 is 9.52%; the constant may not drift."""
    assert daily.PREMIUM_BREAKER == pytest.approx(0.0952)
    assert daily.QDII_LEG == "513500"


def test_a_tripped_breaker_marks_the_buy_row_but_never_the_fills(tmp_path):
    # An underweight risky sleeve at the peak: the CPPI target is 60%, the book holds 48%, so
    # the plan buys every leg -- 513500 included -- without any price move being needed.
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path, legs=(30000, 20000, 120000))
    write_premium(tmp_path / "online")
    marked = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    row = next(r for r in marked["sleeves"] if r["code"] == "513500")
    assert row["delta"] > 0
    assert row["circuit_breaker"] == "缓建"
    assert marked["breaker"]["tripped"] is True
    assert marked["breaker"]["nav_date"] == "2026-11-18"
    assert any("熔断" in warning for warning in marked["warnings"])

    # The same book without a cache proposes identical fills: the breaker is a label, not a rule.
    bare = tmp_path / "bare"
    write_prices(bare / "online", scale=1.0)
    write_book(bare, legs=(30000, 20000, 120000))
    plain = daily.recommendation("paper", online=bare / "online", base=bare)
    plain_row = next(r for r in plain["sleeves"] if r["code"] == "513500")
    assert plain_row["delta"] == row["delta"]
    assert plain_row["target_shares"] == row["target_shares"]
    assert plain["breaker"] is None


def test_a_calm_premium_leaves_the_rows_unmarked(tmp_path):
    write_prices(tmp_path / "online", scale=1.0)
    write_book(tmp_path, legs=(30000, 20000, 120000))
    write_premium(tmp_path / "online", premium=0.05, day="2026-11-18")
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    row = next(r for r in result["sleeves"] if r["code"] == "513500")
    assert row["delta"] > 0
    assert "circuit_breaker" not in row
    assert result["breaker"]["tripped"] is False
    assert not any("熔断" in warning for warning in result["warnings"])


def test_a_sell_day_trips_the_reading_but_defers_nothing(tmp_path):
    """The breaker refuses to *add*; a day that only sells the leg carries no advice to defer."""
    write_prices(tmp_path / "online", scale=0.9)          # falling prices -> the plan cuts
    write_book(tmp_path)
    write_premium(tmp_path / "online")
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    row = next(r for r in result["sleeves"] if r["code"] == "513500")
    assert row["delta"] < 0
    assert "circuit_breaker" not in row
    assert result["breaker"]["tripped"] is True
    assert not any("熔断" in warning for warning in result["warnings"])


def test_a_missing_premium_cache_is_silent(tmp_path):
    write_prices(tmp_path / "online", scale=1.1)
    write_book(tmp_path)
    result = daily.recommendation("paper", online=tmp_path / "online", base=tmp_path)
    assert result["breaker"] is None
    assert not any("熔断" in warning for warning in result["warnings"])
