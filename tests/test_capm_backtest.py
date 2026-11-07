"""Tests for the CAPM backtest machinery."""

import numpy as np
import pandas as pd
import pytest

from tools import capm_backtest as capm


def _panel():
    index = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0, 11.0, 11.0, 12.1], "B": [20.0, 20.0, 22.0, 22.0]}, index=index)
    traded = prices.copy()
    shares = pd.DataFrame({"A": [100.0] * 4, "B": [300.0] * 4}, index=index)
    return prices, traded, shares


def test_market_portfolio_weights_by_capitalisation_lagged_one_day():
    prices, traded, shares = _panel()
    market, weights, capitalisation = capm.market_portfolio(prices, traded, shares)
    assert capitalisation.iloc[0].tolist() == pytest.approx([1000.0, 6000.0])
    assert weights.iloc[1].tolist() == pytest.approx([1000 / 7000, 6000 / 7000])
    # Day 1 returns are +10% on A and 0% on B, so the market earns the lagged weights.
    assert market.iloc[1] == pytest.approx((1000 * 0.10 + 6000 * 0.0) / 7000)
    assert market.iloc[0] != market.iloc[0]  # the first row has no lagged weight


def test_market_portfolio_drops_names_without_a_return_and_renormalises():
    prices, traded, shares = _panel()
    prices = prices.copy()
    prices.iloc[1, 1] = np.nan
    market, weights, _ = capm.market_portfolio(prices, traded, shares)
    assert weights.iloc[1, 1] != weights.iloc[1, 1]  # dropped
    assert weights.iloc[1, 0] == pytest.approx(1.0)


def test_rolling_beta_recovers_a_planted_loading():
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2019-01-01", periods=400)
    market = pd.Series(rng.normal(0.0004, 0.01, len(index)), index=index)
    rf = capm.CASH_RATE / capm.TRADING_DAYS
    returns = pd.DataFrame({"A": 1.4 * (market - rf) + rf + rng.normal(0, 0.0002, len(index)),
                            "B": 0.6 * (market - rf) + rf + rng.normal(0, 0.0002, len(index))},
                           index=index)
    betas = capm.rolling_beta(returns, market)
    assert betas["A"].iloc[-1] == pytest.approx(1.4, abs=0.05)
    assert betas["B"].iloc[-1] == pytest.approx(0.6, abs=0.05)
    assert betas["A"].iloc[: capm.MIN_OBS - 1].isna().all()


def test_forward_returns_cover_exactly_step_sessions_after_the_decision():
    index = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.Series([0.0, 0.10, 0.10, 0.0, 0.0], index=index)
    forward = capm.forward_returns(returns.to_frame("A"), step=2)["A"]
    # Two sessions after t=0 are t+1 and t+2: 10% then 10%.
    assert forward.iloc[0] == pytest.approx(0.21)
    # From t=1 the window is t+2 and t+3: 10% then 0%.
    assert forward.iloc[1] == pytest.approx(0.10)
    assert forward.iloc[2] == pytest.approx(0.0)
    assert forward.iloc[3] != forward.iloc[3]  # no room for a two-session window


def test_rebalance_dates_are_spaced_and_inside_the_sample():
    index = pd.bdate_range("2019-01-01", periods=200)
    dates = capm.rebalance_dates(index, "2019-02-01", "2019-08-01", step=20)
    assert len(dates) > 3
    assert all((dates >= pd.Timestamp("2019-02-01")) & (dates <= pd.Timestamp("2019-08-01")))
    assert (dates[1:] - dates[:-1]).days.min() >= 25


def test_performance_annualises_and_reports_drawdown():
    index = pd.bdate_range("2020-01-01", periods=243)
    series = pd.Series(np.linspace(0.0005, 0.0005, 243), index=index)
    result = capm.performance(series)
    assert result["annual_return"] == pytest.approx((1.0005 ** 243) - 1, rel=1e-6)
    assert result["max_drawdown"] == pytest.approx(0.0)
    assert result["observations"] == 243


@pytest.mark.parametrize("gap_t,slope_t,expected", [
    (-3.0, 5.0, True),
    (0.5, 2.5, False),
    (0.5, 1.2, None),
    (3.0, 5.0, False),
])
def test_verdict_ladder_order(gap_t, slope_t, expected):
    summary = dict(status="ready", slope_gap_t=gap_t, gamma1_t=slope_t, alpha_spread=0.001)
    result = capm.verdict(summary, dict(annual_return=0.1, sharpe_excess=0.5),
                          dict(annual_return=0.2, sharpe_excess=1.0))
    assert result["capm_rejected"] is expected
    assert result["market_vs_equal"] == pytest.approx(-0.1)
    assert result["low_vol_anomaly"] is False


def test_summarize_is_empty_without_periods():
    assert capm.summarize([], 10) == dict(status="empty")

def _synthetic(periods=60, names=100, slope=1.0, seed=17):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-02", periods=periods, freq="20B")
    codes = [f"S{i:03d}" for i in range(names)]
    beta = pd.DataFrame(rng.uniform(0.5, 1.6, (periods, names)), index=index, columns=codes)
    market = pd.Series(rng.normal(0.008, 0.05, periods), index=index)
    noise = rng.normal(0.0, 0.002, (periods, names))
    forward = beta.mul(market, axis=0) * slope + noise
    return beta, forward, market, index


def test_positive_control_does_not_reject_capm_when_returns_are_beta_times_the_market():
    beta, forward, market, index = _synthetic()
    rows = capm.cross_section(beta, forward, market, list(index), 0.0)
    summary = capm.summarize(rows, beta.shape[1])
    # The planted slope is exactly the market's own return, which is CAPM.
    assert summary["gamma1_mean"] == pytest.approx(market.mean(), rel=0.25)
    assert abs(summary["slope_gap_t"]) < 3.0
    result = capm.verdict(summary, dict(annual_return=0.1, sharpe_excess=0.5),
                          dict(annual_return=0.1, sharpe_excess=0.5))
    assert result["capm_rejected"] is not True


def test_negative_control_rejects_capm_when_the_slope_is_planted_at_zero():
    beta, forward, market, index = _synthetic(slope=0.0)
    rows = capm.cross_section(beta, forward, market, list(index), 0.0)
    summary = capm.summarize(rows, beta.shape[1])
    # Returns carry no beta loading at all, so the slope must sit near zero while the
    # market premium does not: that is the rejection case of the protocol.
    assert abs(summary["gamma1_mean"]) < 0.01
    assert summary["slope_gap_t"] < -2.0
    result = capm.verdict(summary, dict(annual_return=0.1, sharpe_excess=0.5),
                          dict(annual_return=0.1, sharpe_excess=0.5))
    assert result["capm_rejected"] is True

def test_equal_weight_rebalances_on_schedule_and_drifts_between():
    index = pd.bdate_range("2020-01-01", periods=6)
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0, 0.0, 0.0],
                            "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=index)
    series, turnover = capm.equal_weight_portfolio(returns, step=3)
    assert series.iloc[1] == pytest.approx(0.05)  # half the book earned 10%
    # The rebalance happens on the third session, where A has drifted above 1/2.
    assert turnover.iloc[2] > 0
    assert turnover.iloc[0] == 0.0 and turnover.iloc[1] == 0.0
    assert turnover.sum() == pytest.approx(turnover.iloc[2])


def test_net_of_costs_subtracts_the_round_trip_times_turnover():
    index = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.Series([0.01, 0.01, 0.01], index=index)
    turnover = pd.Series([0.0, 0.5, 0.0], index=index)
    net = capm.net_of_costs(returns, turnover)
    assert net.iloc[1] == pytest.approx(0.01 - 0.5 * 0.0031)
    assert net.iloc[0] == pytest.approx(0.01)

def test_largest_takes_the_top_k_by_median_capitalisation_independent_of_column_order():
    index = pd.bdate_range("2020-01-01", periods=3)
    capitalisation = pd.DataFrame({"small": [10.0] * 3, "big": [900.0] * 3, "mid": [100.0] * 3},
                                  index=index)
    assert capm.largest(capitalisation, 2) == ["big", "mid"]
    assert capm.largest(capitalisation, 3) == ["big", "mid", "small"]


def test_sweep_returns_one_row_per_width_that_can_be_measured(tmp_path):
    # A panel that cannot be loaded must not raise: the sweep simply yields nothing.
    assert capm.sweep(tmp_path, values=(10,)) in ([],)
