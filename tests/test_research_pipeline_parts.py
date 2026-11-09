"""Tests for the three places where a defect would poison a conclusion rather than a number.

Most of this repository's tools produce a figure that a document quotes.  Three of them
produce something worse when they go wrong: a feature panel that every later backtest is
built on, a point-in-time join whose failure is invisible because the output still looks like
a number, and the E1 book whose buffer rule defines what "the signal" even means.

The point-in-time tests are the important ones.  A fundamental figure that leaks into a date
before it was filed does not raise, does not look wrong, and inflates whatever is measured
with it.  So availability is asserted directly: a filing dated in April must be invisible in
March, and a value must be carried forward unchanged between filings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import cppi_crisis, e1_reversal, feature_panel, fundamental_ic


def observation(count=40, columns=6, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, 1, columns))
    return {"stock_features": features, "valid_mask": np.ones(count, dtype=bool)}


class TestReversalBook:
    def test_the_composite_is_a_mean_percentile(self):
        book = e1_reversal.ReversalBook([f"f{i}" for i in range(6)])
        book.interval = 1
        obs = observation()
        composite, selected = book.scores(obs)
        assert len(selected) == 40
        # a rank average over six features lands in the unit interval
        assert composite.min() >= 0.0
        assert composite.max() <= 1.0

    def test_the_smallest_composite_is_bought_first(self):
        book = e1_reversal.ReversalBook([f"f{i}" for i in range(6)], top_k=5, buffer=60)
        book.interval = 1
        obs = observation()
        composite, selected = book.scores(obs)
        weights = book.act(obs, 0)
        bought = list(np.flatnonzero(weights))
        cheapest = list(selected[np.argsort(composite, kind="stable")][:5])
        assert sorted(bought) == sorted(cheapest)

    def test_the_book_holds_exactly_top_k_names(self):
        book = e1_reversal.ReversalBook([f"f{i}" for i in range(6)], top_k=5, buffer=60)
        book.interval = 1
        obs = observation()
        weights = book.act(obs, 0)
        assert int((weights > 0).sum()) == 5
        assert weights[weights > 0] == pytest.approx(0.2)

    def test_holding_only_updates_on_the_rebalance_session(self):
        book = e1_reversal.ReversalBook([f"f{i}" for i in range(6)])
        obs = observation()
        assert book.act(obs, 1) is None
        assert book.act(obs, 19) is None
        assert book.act(obs, 20) is not None

    def test_a_held_name_inside_the_buffer_is_kept_even_if_out_of_the_top(self):
        # five names with a buffer of five: whoever was bought stays while still ranked
        # within the buffer, which is the whole point of the rule
        book = e1_reversal.ReversalBook([f"f{i}" for i in range(6)], top_k=2, buffer=5)
        book.interval = 1
        first = observation(seed=1)
        book.act(first, 0)
        held = set(np.flatnonzero(book.act(first, 0) > 0).tolist())
        second = observation(seed=2)
        weights = book.act(second, 0)
        kept = held & set(np.flatnonzero(weights > 0).tolist())
        assert kept, "a name still inside the buffer should survive one rebalance"

    def test_names_outside_the_valid_mask_are_never_bought(self):
        book = e1_reversal.ReversalBook([f"f{i}" for i in range(6)], top_k=5, buffer=60)
        book.interval = 1
        obs = observation()
        obs["valid_mask"][:] = False
        obs["valid_mask"][:3] = True
        weights = book.act(obs, 0)
        assert int((weights > 0).sum()) <= 3


class TestPointInTime:
    def test_a_filing_is_invisible_before_it_is_filed(self):
        series = pd.DataFrame(
            {"eps": [1.0, 2.0], "bvps": [10.0, 11.0], "roe": [0.1, 0.2], "gross": [0.3, 0.4]},
            index=pd.to_datetime(["2020-04-30", "2020-08-31"]))
        row, stamp = fundamental_ic.as_of(series, pd.Timestamp("2020-03-31"))
        assert row is None and stamp is None

    def test_the_latest_filing_on_or_before_the_date_is_used(self):
        series = pd.DataFrame(
            {"eps": [1.0, 2.0], "bvps": [10.0, 11.0], "roe": [0.1, 0.2], "gross": [0.3, 0.4]},
            index=pd.to_datetime(["2020-04-30", "2020-08-31"]))
        row, stamp = fundamental_ic.as_of(series, pd.Timestamp("2020-09-15"))
        assert stamp == pd.Timestamp("2020-08-31")
        assert row["bvps"] == pytest.approx(11.0)

    def test_the_panel_carries_a_value_forward_and_never_backward(self):
        index = pd.to_datetime(["2020-03-31", "2020-05-15", "2020-09-30"])
        series = pd.DataFrame(
            {"eps": [1.0, 2.0], "bvps": [10.0, 11.0], "roe": [0.1, 0.2], "gross": [0.3, 0.4]},
            index=pd.to_datetime(["2020-04-30", "2020-08-31"]))
        panel = fundamental_ic.panel_at({"000001": series}, index, ["000001"], "bvps")
        assert np.isnan(panel.at[pd.Timestamp("2020-03-31"), "000001"])
        assert panel.at[pd.Timestamp("2020-05-15"), "000001"] == pytest.approx(10.0)
        assert panel.at[pd.Timestamp("2020-09-30"), "000001"] == pytest.approx(11.0)

    def test_a_duplicate_filing_date_keeps_one_row_not_two(self):
        frame = pd.DataFrame({
            "code": ["000001", "000001"],
            "filed": pd.to_datetime(["2020-04-30", "2020-04-30"]),
            "eps": [1.0, 1.5], "bvps": [10.0, 10.5], "roe": [0.1, 0.11], "gross": [0.3, 0.31]})
        by_code = fundamental_ic.trailing_eps(frame)
        series = by_code["000001"]
        assert len(series) == 1
        assert series["bvps"].iloc[0] == pytest.approx(10.5)


class TestFeaturePanel:
    def test_the_panel_is_keyed_by_code_then_feature(self, tmp_path, monkeypatch):
        index = pd.date_range("2020-01-01", periods=80, freq="B")
        bars = pd.DataFrame({
            "open": np.linspace(10, 11, 80), "close": np.linspace(10, 11, 80),
            "high": np.linspace(10.2, 11.2, 80), "low": np.linspace(9.8, 10.8, 80),
            "volume": np.full(80, 1e6), "amount": np.full(80, 1e7),
            "turnover": np.full(80, 0.01), "valid_price": True}, index=index)
        monkeypatch.setattr(feature_panel, "read_bars", lambda code: bars.copy())
        monkeypatch.setattr(feature_panel, "select_codes",
                            lambda count=0, panel_dir=None: (["000001", "000002"], None))
        manifest = feature_panel.build(count=2, output=tmp_path / "panel")
        panel = pd.read_parquet(tmp_path / "panel" / "panel_data.parquet")
        features = set(panel.columns.get_level_values(1))
        assert set(manifest["feature_cols"]) <= features
        assert set(panel.columns.get_level_values(0)) == {"000001", "000002"}

    def test_a_code_without_enough_history_is_skipped(self, tmp_path, monkeypatch):
        def partial(code):
            if code == "000002":
                return None
            index = pd.date_range("2020-01-01", periods=80, freq="B")
            return pd.DataFrame({
                "open": np.linspace(10, 11, 80), "close": np.linspace(10, 11, 80),
                "high": np.linspace(10.2, 11.2, 80), "low": np.linspace(9.8, 10.8, 80),
                "volume": np.full(80, 1e6), "amount": np.full(80, 1e7),
                "turnover": np.full(80, 0.01), "valid_price": True}, index=index)

        monkeypatch.setattr(feature_panel, "read_bars", partial)
        monkeypatch.setattr(feature_panel, "select_codes",
                            lambda count=0, panel_dir=None: (["000001", "000002"], None))
        manifest = feature_panel.build(count=2, output=tmp_path / "panel")
        assert manifest["stock_codes"] == ["000001"]
        assert manifest["skipped"] == ["000002"]


class TestEpisodeBlock:
    def test_returns_are_measured_from_the_entry_not_the_start_of_time(self):
        index = pd.date_range("2020-01-01", periods=40, freq="B")
        nav = pd.Series(np.linspace(1.0, 1.2, 40), index=index)
        state = pd.DataFrame({"nav": nav, "peak": nav.cummax(), "exposure": 0.5,
                              "r_risky": 0.0, "cushion": nav - 0.9 * nav.cummax()}, index=index)
        risky = pd.Series(0.0, index=index)
        block = cppi_crisis.episode_block(state, risky, index[10], index[19])
        step = 0.2 / 39
        expected = (1.0 + 19 * step) / (1.0 + 10 * step) - 1
        assert block["strategy_return"] == pytest.approx(expected, rel=1e-6)

    def test_a_window_with_no_days_returns_nothing(self):
        index = pd.date_range("2020-01-01", periods=10, freq="B")
        state = pd.DataFrame({"nav": np.ones(10), "peak": np.ones(10), "exposure": 0.0,
                              "r_risky": 0.0, "cushion": np.zeros(10)}, index=index)
        assert cppi_crisis.episode_block(state, pd.Series(0.0, index=index),
                                         pd.Timestamp("2030-01-01"), pd.Timestamp("2030-02-01")) is None
