"""Invariants for the allocation and drawdown machinery built in the 2026-11 series.

The research tools were written to answer questions, not to be called by other code, so they
had no tests and their outputs were checked by reading the numbers.  That is fine for a
one-off and wrong for anything a later reader might rerun: a silent change in the drawdown
calculation or the floor rule would leave every document quoting a figure the code no longer
produces.

These tests pin the arithmetic that the published numbers depend on, using synthetic series
where the correct answer is known by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import cppi_gap_risk, cppi_strategy, global_frontier
from tools.drawdown_target import drawdown_curve


def frame_from(ashare, sp500, bond):
    index = pd.date_range("2020-01-01", periods=len(ashare), freq="B")
    return pd.DataFrame({"ashare": ashare, "sp500": sp500, "bond": bond}, index=index)


class TestDrawdown:
    def test_a_fall_from_the_opening_value_counts_in_full(self):
        # the opening value is a peak; without it the first loss is invisible
        assert drawdown_curve(pd.Series([-0.10, -0.10])) == pytest.approx(-0.19, abs=1e-12)

    def test_drawdown_is_measured_from_the_peak_not_the_start(self):
        # up 50 percent then down 33.3 percent ends flat, so the trough drawdown is the fall
        returns = pd.Series([0.5, -1 / 3])
        assert drawdown_curve(returns) == pytest.approx(-1 / 3, abs=1e-9)

    def test_a_monotone_rise_has_no_drawdown(self):
        assert drawdown_curve(pd.Series([0.01, 0.02, 0.03])) == pytest.approx(0.0)


class TestCppi:
    def test_exposure_stays_inside_its_bounds(self):
        rng = np.random.default_rng(0)
        frame = frame_from(rng.normal(0, 0.02, 300), rng.normal(0, 0.02, 300),
                           rng.normal(0, 0.001, 300))
        state = cppi_gap_risk.walk(frame, 6)
        assert state["exposure"].min() >= 0.0
        assert state["exposure"].max() <= 1.0

    def test_the_floor_slows_a_one_way_fall_but_does_not_abolish_it(self):
        # 400 sessions of minus one percent is a ninety-eight percent market fall, far beyond
        # anything observed; the band makes exposure lag the rule, so the floor is exceeded
        # rather than held.  What the test pins is that the strategy still loses an order of
        # magnitude less than the market, not that it is immune.
        fall = np.full(400, -0.01)
        frame = frame_from(fall, fall, np.zeros(400))
        result = cppi_strategy.run(frame, 6)
        assert result["drawdown"] > -0.25
        market_total = float(np.prod(1 + fall) - 1)
        assert result["drawdown"] > market_total + 0.5

    def test_a_flat_market_costs_only_the_turnover(self):
        # with a flat market every trade is pure cost, so the account drifts slightly down
        frame = frame_from(np.zeros(250), np.zeros(250), np.zeros(250))
        result = cppi_strategy.run(frame, 6)
        assert -0.002 < result["annual"] < 0.0
        assert result["drawdown"] == pytest.approx(0.0, abs=1e-12)

    def test_a_higher_multiplier_takes_more_risk(self):
        rng = np.random.default_rng(1)
        frame = frame_from(rng.normal(0.001, 0.015, 500), rng.normal(0.001, 0.015, 500),
                           np.full(500, 0.0001))
        low = cppi_strategy.run(frame, 4)
        high = cppi_strategy.run(frame, 8)
        assert high["vol"] >= low["vol"]

    def test_gap_headroom_is_positive_while_above_the_floor(self):
        rng = np.random.default_rng(2)
        frame = frame_from(rng.normal(0, 0.01, 200), rng.normal(0, 0.01, 200),
                           np.full(200, 0.0001))
        state = cppi_gap_risk.walk(frame, 6)
        exposed = state[state["exposure"] > 0.01]
        assert (exposed["headroom"] > 0).all()


class TestCurrency:
    def test_conversion_compounds_price_and_fixing(self):
        prices = pd.Series([100.0, 110.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
        rates = pd.Series([7.0, 7.07], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
        returns = global_frontier.yuan_returns(prices, rates)
        assert returns.iloc[0] == pytest.approx(1.10 * 1.01 - 1, abs=1e-9)

    def test_without_a_fixing_the_local_return_passes_through(self):
        prices = pd.Series([100.0, 110.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
        assert global_frontier.yuan_returns(prices).iloc[0] == pytest.approx(0.10)


class TestTradingCalendar:
    def test_a_known_holiday_is_not_a_trading_day(self):
        from tools import trading_calendar
        calendar = pd.to_datetime(pd.Series(["2026-10-09", "2026-10-12"]))
        assert not trading_calendar.is_trading_day("2026-10-01", calendar)
        assert trading_calendar.is_trading_day("2026-10-12", calendar)

    def test_weekends_are_not_in_the_calendar(self):
        from tools import trading_calendar
        calendar = pd.to_datetime(pd.Series(["2026-11-06", "2026-11-09"]))
        assert not trading_calendar.is_trading_day("2026-11-07", calendar)


class TestInventory:
    def test_closed_lines_are_classified_and_active_ones_are_not(self):
        from tools.repository_inventory import line_of
        assert line_of("timing_counterfactual") in {"按时点干预（已结束）", "反事实学习（已结束）", "其它分析"}
        assert line_of("capm_backtest") == "回测与归因"
        assert line_of("unlock_ic") == "特征与信号"
