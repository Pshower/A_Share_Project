"""Tests for the CAPM decomposition of stored strategies."""

import numpy as np
import pandas as pd
import pytest

from tools import baseline_capm as capm


def test_ols_recovers_an_exact_line():
    x = np.linspace(-0.02, 0.02, 40)
    y = 0.001 + 0.8 * x
    result = capm.ols(y, x)
    assert result["alpha"] == pytest.approx(0.001, abs=1e-12)
    assert result["beta"] == pytest.approx(0.8, abs=1e-12)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_ols_reports_a_positive_slope_t_statistic_for_an_exact_fit():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.01, 250)
    y = 1.3 * x + rng.normal(0.0, 0.001, 250)
    result = capm.ols(y, x)
    assert result["beta"] == pytest.approx(1.3, abs=0.05)
    assert result["beta_t"] > 20
    assert abs(result["residual_lag1"]) < 0.3


def test_ols_rejects_mismatched_or_short_input():
    with pytest.raises(ValueError):
        capm.ols([0.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        capm.ols([0.0, 1.0], [0.0, 1.0])


def test_regress_strategy_recovers_a_planted_alpha_and_beta():
    days = 260
    rng = np.random.default_rng(11)
    market = pd.Series(rng.normal(0.0002, 0.01, days),
                       index=pd.bdate_range("2023-01-02", periods=days))
    rf = capm.CASH_RATE / capm.TRADING_DAYS
    planted_alpha, planted_beta = 0.0004, 0.7
    strategy = planted_alpha + rf + planted_beta * (market - rf)
    result = capm.regress_strategy(strategy, market)
    assert result["beta"] == pytest.approx(planted_beta, abs=1e-9)
    assert result["annual_alpha"] == pytest.approx(planted_alpha * capm.TRADING_DAYS, abs=1e-9)
    assert result["observations"] == days


def test_regress_strategy_needs_enough_overlap():
    index = pd.bdate_range("2023-01-02", periods=10)
    with pytest.raises(ValueError):
        capm.regress_strategy(pd.Series(0.001, index=index), pd.Series(0.001, index=index))


def test_finite_turns_non_finite_numbers_into_null():
    cleaned = capm.finite(dict(a=float("nan"), b=[float("inf"), 1.5], c="x", d=2))
    assert cleaned["a"] is None
    assert cleaned["b"] == [None, 1.5]
    assert cleaned["c"] == "x" and cleaned["d"] == 2


def test_market_returns_are_centred_on_the_requested_window():
    series = capm.market_returns("2023-01-03", "2023-03-31")
    assert len(series) > 20
    assert series.index[0] > pd.Timestamp("2023-01-03")
    assert series.index[-1] <= pd.Timestamp("2023-03-31")
    assert np.isfinite(series.to_numpy()).all()
