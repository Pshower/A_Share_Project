"""Tests for the allocation studies that produced the menu, the cash ladder and the tail.

Five tools remain whose output a document quotes: the tracking-error fit behind the fifty-
thirty-twenty weights, the rebalancing and cash study, the window analysis behind "five
percent is a property of the window", the tail sensitivity used to argue that the micro-cap
premium is not trustworthy, and the dividend sleeve ladder.

Each is tested on a construction whose answer is arithmetic rather than historical, so a
future edit that changes a published figure fails here instead of quietly rewriting a
conclusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import allocation_robustness, allocation_study, dividend_sleeve
from tools import return_context, tail_exposure


def frame_from(columns, periods=300, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    return pd.DataFrame({name: rng.normal(0.0003, 0.01, periods) for name in columns}, index=index)


class TestWeightGrid:
    def test_every_point_is_a_valid_convolution(self):
        points = list(allocation_study.grid(0.1))
        assert len(points) == 66          # C(10+2, 2) three-way splits on a ten-step grid
        for weights in points:
            assert sum(weights.values()) == pytest.approx(1.0)
            assert all(value >= 0 for value in weights.values())

    def test_the_resolution_controls_the_count(self):
        assert len(list(allocation_study.grid(0.5))) == 6
        assert len(list(allocation_study.grid(0.25))) == 15


class TestAllocationBlend:
    def test_a_single_asset_reproduces_its_own_returns(self):
        frame = frame_from(["a"])
        series = allocation_study.blend(frame, {"a": 1.0}, rebalance=None)
        assert np.allclose(series.to_numpy(), frame["a"].to_numpy())

    def test_drift_happens_between_resets(self):
        # one asset rises, the other is flat; before any rebalance the weights must drift
        index = pd.date_range("2020-01-01", periods=100, freq="B")
        frame = pd.DataFrame({"a": np.full(100, 0.01), "b": np.zeros(100)}, index=index)
        series = allocation_study.blend(frame, {"a": 0.5, "b": 0.5}, rebalance=None, cost=0.0)
        # a fixed 50/50 would earn 0.5 percent a day; drift makes it earn slightly more
        assert series.iloc[50] > 0.005

    def test_a_reset_charges_only_the_turnover_moved(self):
        # a reset is free when nothing has drifted, and costly once it has: two flat assets
        # never diverge, so the test has to give one of them a return
        index = pd.date_range("2020-01-01", periods=60, freq="B")
        flat = pd.DataFrame({"a": np.zeros(60), "b": np.zeros(60)}, index=index)
        idle = allocation_study.blend(flat, {"a": 0.5, "b": 0.5}, rebalance=20, cost=0.01)
        assert idle.sum() == pytest.approx(0.0, abs=1e-12)
        moving = pd.DataFrame({"a": np.full(60, 0.02), "b": np.zeros(60)}, index=index)
        free = allocation_study.blend(moving, {"a": 0.5, "b": 0.5}, rebalance=20, cost=0.0)
        paid = allocation_study.blend(moving, {"a": 0.5, "b": 0.5}, rebalance=20, cost=0.01)
        assert paid.sum() < free.sum()

    def test_tracking_error_of_identical_series_is_zero(self):
        frame = frame_from(["a"])
        series = allocation_study.blend(frame, {"a": 1.0}, rebalance=None, cost=0.0)
        detail = allocation_study.tracking(series.rename("x"), series.rename("y"))
        assert detail["tracking_error"] == pytest.approx(0.0, abs=1e-12)
        assert detail["correlation"] == pytest.approx(1.0)
        assert detail["annual_gap"] == pytest.approx(0.0, abs=1e-9)


class TestRebalancingRules:
    def test_never_rebalancing_trades_nothing(self):
        frame = frame_from(["a", "b"])
        series, turnover = allocation_robustness.blend(frame, {"a": 0.5, "b": 0.5})
        assert turnover == pytest.approx(0.0)
        assert len(series) == len(frame)

    def test_a_periodic_rule_trades_on_schedule(self):
        frame = frame_from(["a", "b"], periods=240)
        series, turnover = allocation_robustness.blend(
            frame, {"a": 0.5, "b": 0.5}, period=60, cost=0.0)
        assert turnover > 0
        # one reset per sixty sessions over four years is a handful of round trips
        assert turnover < 1.0

    def test_a_band_only_triggers_when_drift_crosses_it(self):
        index = pd.date_range("2020-01-01", periods=400, freq="B")
        calm = pd.DataFrame({"a": np.full(400, 0.0001), "b": np.full(400, 0.0001)}, index=index)
        _, calm_turnover = allocation_robustness.blend(calm, {"a": 0.5, "b": 0.5}, band=0.05)
        # a sustained trend, not an oscillation: alternating up and down returns cancel the
        # drift and would leave the band untriggered, which is what the first draft of this
        # test assumed wrongly
        trend = pd.DataFrame({"a": np.full(400, 0.01), "b": np.zeros(400)}, index=index)
        _, trend_turnover = allocation_robustness.blend(trend, {"a": 0.5, "b": 0.5}, band=0.05)
        assert calm_turnover == pytest.approx(0.0)
        assert trend_turnover > 0.0

    def test_the_cash_ladder_reduces_risk_monotonically(self):
        frame = frame_from(["000300", "000905", "000852"], periods=500)
        rows = allocation_robustness.cash_ladder(frame)
        vols = [row["vol"] for row in rows]
        drawdowns = [abs(row["drawdown"]) for row in rows]
        assert vols == sorted(vols, reverse=True)
        assert drawdowns == sorted(drawdowns, reverse=True)

    def test_full_exposure_is_the_first_rung(self):
        frame = frame_from(["000300", "000905", "000852"], periods=400)
        rows = allocation_robustness.cash_ladder(frame)
        assert rows[0]["exposure"] == pytest.approx(1.0)

    def test_a_lower_volatility_target_holds_less_equity(self):
        frame = frame_from(["000300", "000905", "000852"], periods=800)
        loose = allocation_robustness.volatility_target(frame, 0.30)
        tight = allocation_robustness.volatility_target(frame, 0.05)
        assert tight["mean_exposure"] < loose["mean_exposure"]
        assert 0.0 <= tight["worst_exposure"] <= 1.0
        assert 0.0 <= loose["mean_exposure"] <= 1.0


class TestWindowStatistics:
    def test_a_known_growth_rate_is_recovered(self):
        # a series that doubles over 243 sessions is one hundred percent a year
        nav = pd.Series(np.exp(np.linspace(0, np.log(2), 243)),
                        index=pd.date_range("2020-01-01", periods=243, freq="B"))
        block = return_context.window_stats(nav)
        assert block["annual"] == pytest.approx(1.0, rel=1e-3)

    def test_a_flat_series_has_no_return_and_no_drawdown(self):
        nav = pd.Series(np.ones(300), index=pd.date_range("2020-01-01", periods=300, freq="B"))
        block = return_context.window_stats(nav)
        assert block["annual"] == pytest.approx(0.0, abs=1e-12)
        assert block["drawdown"] == pytest.approx(0.0, abs=1e-12)

    def test_the_drawdown_is_taken_from_the_peak(self):
        values = np.concatenate([np.linspace(1, 2, 150), np.linspace(2, 1, 150)])
        nav = pd.Series(values, index=pd.date_range("2020-01-01", periods=300, freq="B"))
        block = return_context.window_stats(nav)
        assert block["drawdown"] == pytest.approx(-0.5, abs=1e-6)

    def test_rolling_windows_report_the_share_that_lost_money(self):
        nav = pd.Series(np.linspace(1.0, 0.8, 2000),
                        index=pd.date_range("2010-01-01", periods=2000, freq="B"))
        roll = return_context.rolling(nav, length=243)
        assert roll["share_negative"] == pytest.approx(1.0)
        assert roll["median"] < 0

    def test_a_series_shorter_than_the_window_returns_nothing(self):
        nav = pd.Series(np.ones(100), index=pd.date_range("2020-01-01", periods=100, freq="B"))
        assert return_context.rolling(nav, length=1215) is None


class TestTailSensitivity:
    def test_the_blend_weights_the_observed_and_the_assumed(self):
        tail = dict(mean_members=2000.0, equal_annual=0.15, cap_annual=0.13)
        gap = dict(missing=500, missing_share=0.2, missing_capitalisation=0.02)
        result = tail_exposure.sensitivity(tail, gap, scenarios=(-0.60,))
        # five hundred missing names against two thousand present ones is a fifth of the book
        assert result["missing_count_share"] == pytest.approx(0.2)
        assert result["rows"][0]["equal_tail"] == pytest.approx(0.8 * 0.15 + 0.2 * -0.60)
        assert result["rows"][0]["cap_tail"] == pytest.approx(0.98 * 0.13 + 0.02 * -0.60)

    def test_a_harmless_assumption_leaves_the_tail_unchanged(self):
        tail = dict(mean_members=2000.0, equal_annual=0.15, cap_annual=0.13)
        gap = dict(missing=500, missing_share=0.2, missing_capitalisation=0.02)
        result = tail_exposure.sensitivity(tail, gap, scenarios=(0.15,))
        assert result["rows"][0]["equal_tail"] == pytest.approx(0.15)

    def test_the_capitalisation_share_always_beats_the_count_share(self):
        tail = dict(mean_members=2000.0, equal_annual=0.15, cap_annual=0.13)
        gap = dict(missing=500, missing_share=0.2, missing_capitalisation=0.02)
        result = tail_exposure.sensitivity(tail, gap, scenarios=(-0.6,))
        assert result["missing_cap_share"] < result["missing_count_share"]


class TestDividendSleeve:
    def test_an_annual_rate_is_spread_over_the_sessions(self):
        index = pd.date_range("2020-01-01", periods=243, freq="B")
        daily = dividend_sleeve.daily_from_annual(0.0243, index)
        assert len(daily) == 243
        assert daily.sum() == pytest.approx(0.0243, rel=1e-9)

    def test_the_sleeve_weights_sum_to_one_at_every_rung(self):
        for share in (0.0, 0.2, 0.4):
            total = sum(weight * (1 - share) for weight in dividend_sleeve.BASE.values()) + share
            assert total == pytest.approx(1.0)
