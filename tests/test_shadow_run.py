"""The recorder must hold a book for twenty sessions and still be able to report a return.

Two properties matter more than any single row: the cadence (re-deciding every session turned a
15x-a-year turnover into 105x) and the chaining (the mark level jumps whenever the composition
changes, so it cannot answer "how much did the book earn").  Both are pinned here on synthetic
panels, with the state and log pointed at a temporary directory.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tools import shadow_run, trading_calendar


def frame(days, codes, price_of):
    """A (code, feature) close frame plus the panel shape the recorder expects."""
    close = pd.DataFrame({code: [price_of(code, day) for day in days] for code in codes}, index=days)
    prices = pd.concat({"close": close}, axis=1)
    prices.columns = pd.MultiIndex.from_tuples([(code, "close") for code in codes],
                                               names=["code", "feature"])
    features = pd.DataFrame(index=days, columns=pd.MultiIndex.from_product([codes, ["f1"]]),
                            dtype=float)
    for code in codes:
        features[(code, "f1")] = np.arange(len(days), dtype=float)
    return prices, features


def test_sessions_between_counts_only_sessions_after_the_start():
    index = pd.bdate_range("2020-01-01", periods=30)
    assert shadow_run.sessions_between(index, None, index[-1]) is None
    assert shadow_run.sessions_between(index, index[0], index[0]) == 0
    assert shadow_run.sessions_between(index, index[0], index[19]) == 19
    assert shadow_run.sessions_between(index, index[0], index[-1]) == 29


def test_basket_return_is_the_equal_weight_move_of_the_book():
    days = pd.bdate_range("2020-01-01", periods=3)
    close = pd.DataFrame({"a": [10.0, 11.0, 12.0], "b": [20.0, 20.0, 22.0]}, index=days)
    change = shadow_run.basket_return(close, ["a", "b"], days[0], days[2])
    assert change == pytest.approx((0.2 + 0.1) / 2)


def test_basket_return_skips_names_without_both_prices():
    days = pd.bdate_range("2020-01-01", periods=2)
    close = pd.DataFrame({"a": [10.0, 11.0], "b": [np.nan, 20.0]}, index=days)
    assert shadow_run.basket_return(close, ["a", "b"], days[0], days[1]) == pytest.approx(0.10)


def test_basket_return_is_none_without_history():
    days = pd.bdate_range("2020-01-01", periods=1)
    close = pd.DataFrame({"a": [10.0]}, index=days)
    assert shadow_run.basket_return(close, ["a"], None, days[0]) is None


def test_the_chain_compounds_interval_returns_in_order():
    records = [
        dict(interval=None),
        dict(interval=dict(**{"return": 0.10})),
        dict(interval=dict(**{"return": -0.05})),
    ]
    assert shadow_run.chain(records) == pytest.approx([1.0, 1.10, 1.045])


def write_dataset(directory, days, codes, upto):
    prices, features = frame(days[:upto], codes, lambda code, day: 10.0 + days.get_loc(day))
    directory.mkdir(exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(dict(stock_codes=codes, feature_cols=["f1"])), encoding="utf-8")
    features.to_parquet(directory / "panel_data.parquet")
    prices.to_parquet(directory / "prices.parquet")


def test_a_hold_row_does_not_swap_names(tmp_path, monkeypatch):
    """One session after entry the book must be untouched, not re-ranked."""
    days = pd.bdate_range("2020-01-01", periods=8)
    codes = ["000001", "000002", "000003"]
    dataset = tmp_path / "toy"
    monkeypatch.setattr(shadow_run, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(shadow_run, "LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda *a, **k: True)

    write_dataset(dataset, days, codes, upto=6)
    first = shadow_run.run(dataset=dataset, log=lambda *_: None)
    assert first["decision"] == "rebalance" and len(first["book"]) == 3

    write_dataset(dataset, days, codes, upto=7)      # one more session
    second = shadow_run.run(dataset=dataset, log=lambda *_: None)
    assert second["decision"] == "hold"
    assert second["sessions_held"] == 1
    assert second["book"] == first["book"]
    assert second["interval"]["return"] is not None
    assert second["nav"]["equal_weight_nav"] != first["nav"]["equal_weight_nav"]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["holdings"] == first["book"]
    assert state["entry_date"] == first["panel_date"]


def test_twenty_sessions_later_it_rebalances_again(tmp_path, monkeypatch):
    days = pd.bdate_range("2020-01-01", periods=30)
    codes = ["000001", "000002", "000003"]
    dataset = tmp_path / "toy"
    monkeypatch.setattr(shadow_run, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(shadow_run, "LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda *a, **k: True)
    write_dataset(dataset, days, codes, upto=6)
    first = shadow_run.run(dataset=dataset, log=lambda *_: None)
    write_dataset(dataset, days, codes, upto=26)      # twenty sessions after entry
    later = shadow_run.run(dataset=dataset, log=lambda *_: None)
    assert later["sessions_held"] == 20
    assert later["decision"] == "rebalance"
    assert later["held_from"] == first["panel_date"]       # the book it is leaving
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["entry_date"] == later["panel_date"]


def test_benchmarks_report_the_pool_and_the_index(monkeypatch):
    days = pd.bdate_range("2020-01-01", periods=3)
    close = pd.DataFrame({"a": [10.0, 11.0, 12.0], "b": [20.0, 20.0, 22.0]}, index=days)
    monkeypatch.setattr(shadow_run, "index_close",
                        lambda symbol, start=None, end=None: pd.Series([100.0, 101.0, 102.0], index=days))
    row = shadow_run.benchmarks(close, ["a", "b"], days[0], days[2])
    assert row["pool_equal"] == pytest.approx((0.2 + 0.1) / 2)
    assert row["hs300"] == pytest.approx(0.02)


def test_a_missing_index_leaves_the_comparison_null_rather_than_failing(monkeypatch):
    days = pd.bdate_range("2020-01-01", periods=3)
    close = pd.DataFrame({"a": [10.0, 11.0, 12.0]}, index=days)
    monkeypatch.setattr(shadow_run, "index_close", lambda *a, **k: None)
    row = shadow_run.benchmarks(close, ["a"], days[0], days[2])
    assert row["hs300"] is None
    assert row["pool_equal"] == pytest.approx(0.2)


def test_index_return_needs_two_sessions(monkeypatch):
    days = pd.bdate_range("2020-01-01", periods=2)
    monkeypatch.setattr(shadow_run, "index_close", lambda *a, **k: pd.Series([100.0], index=days[:1]))
    assert shadow_run.index_return("sh000300", days[0], days[1]) is None


def test_a_truncated_pool_is_refused_rather_than_scored(tmp_path, monkeypatch):
    """A download that lost half the codes must not silently record a different universe."""
    days = pd.bdate_range("2020-01-01", periods=30)
    codes = ["00000%d" % index for index in range(1, 6)]
    dataset = tmp_path / "toy"
    monkeypatch.setattr(shadow_run, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(shadow_run, "LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda *a, **k: True)
    write_dataset(dataset, days, codes[:2], upto=6)      # only 2 of the 5 declared codes
    (dataset / "manifest.json").write_text(
        json.dumps(dict(stock_codes=codes, feature_cols=["f1"])), encoding="utf-8")
    record = shadow_run.run(dataset=dataset, log=lambda *_: None)
    assert record["status"] == "bad_data"
    assert "truncated" in record["reason"]
