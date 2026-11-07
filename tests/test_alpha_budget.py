"""Unit checks for the alpha-budget arithmetic."""

import math

import numpy as np
import pandas as pd
import pytest

from tools import alpha_budget as budget
from tools import exposure_stage2 as stage2


def test_norm_ppf_inverts_norm_cdf():
    for share in (0.01, 0.25, 0.5, 0.9, 0.99):
        assert abs(budget._norm_cdf(budget._norm_ppf(share)) - share) < 1e-9


def test_norm_ppf_rejects_impossible_shares():
    with pytest.raises(ValueError):
        budget._norm_ppf(0.0)
    with pytest.raises(ValueError):
        budget._norm_ppf(1.0)


def test_top_mean_z_matches_the_half_sample_case():
    # E[z | z > 0] = 2 * pdf(0) = sqrt(2 / pi)
    assert abs(budget.top_mean_z(0.5) - math.sqrt(2.0 / math.pi)) < 1e-9


def test_top_mean_z_is_zero_for_the_whole_cross_section():
    assert budget.top_mean_z(1.0) == 0.0


def test_top_mean_z_falls_as_the_book_gets_wider():
    values = [budget.top_mean_z(share) for share in (0.05, 0.10, 0.20, 0.50)]
    assert values == sorted(values, reverse=True)
    assert all(value > 0 for value in values)


def test_alpha_from_ic_scales_linearly_and_rejects_bad_years():
    one = budget.alpha_from_ic(0.02, 0.10, 0.45)
    two = budget.alpha_from_ic(0.04, 0.10, 0.45)
    assert abs(two - 2 * one) < 1e-15
    with pytest.raises(ValueError):
        budget.alpha_from_ic(0.02, 0.10, 0.45, years=0.0)


def test_ic_from_alpha_inverts_alpha_from_ic():
    alpha = budget.alpha_from_ic(0.03, 0.10, 0.45)
    assert abs(budget.ic_from_alpha(alpha, 0.10, 0.45) - 0.03) < 1e-12


def test_required_ic_shrinks_with_a_shorter_holding_period():
    short = budget.required_ic_at_horizon(0.014, 0.10, 0.4758, 20)
    long = budget.required_ic_at_horizon(0.014, 0.10, 0.4758, 243)
    assert short < long
    assert abs(short / long - math.sqrt(20 / 243)) < 1e-12


def test_required_ic_agrees_with_the_annual_convention():
    target = 0.014
    annual = budget.required_ic_at_horizon(target, 0.10, 0.4758, budget.TRADING_DAYS)
    assert abs(annual - budget.ic_from_alpha(target, 0.10, 0.4758)) < 1e-12


def test_blend_with_full_weight_reproduces_the_index():
    nav = pd.Series([1.0, 1.1, 0.9, 1.3], index=pd.date_range("2020-01-01", periods=4))
    assert np.allclose(budget.blend(nav, 1.0).to_numpy(), (nav / nav.iloc[0]).to_numpy())


def test_blend_with_zero_weight_is_cash_only():
    nav = pd.Series([1.0, 1.1, 0.9, 1.3], index=pd.date_range("2020-01-01", periods=4))
    path = budget.blend(nav, 0.0, cash_rate=0.05)
    assert np.allclose(path.to_numpy(), np.power(1.05, np.arange(4) / budget.TRADING_DAYS))


def test_blend_rejects_weights_outside_the_unit_interval():
    nav = pd.Series([1.0, 1.1])
    for weight in (-0.1, 1.1):
        with pytest.raises(ValueError):
            budget.blend(nav, weight)


def test_constant_weight_rebalancing_moves_log_returns_linearly():
    rng = np.random.default_rng(7)
    nav = pd.Series(np.cumprod(1.0 + rng.normal(0.0004, 0.012, size=800)))
    log_growth = np.log(nav.to_numpy() / nav.iloc[0])
    for weight in (0.2, 0.5, 0.8):
        blended = budget.blend(nav, weight, cash_rate=0.0, rebalance=True)
        assert np.allclose(np.log(blended.to_numpy()), weight * log_growth, atol=1e-12)
        # hence log volatility, and the log-return Sharpe, scale with the same factor
        log_returns = np.diff(np.log(blended.to_numpy()))
        reference = np.diff(log_growth)
        assert abs(np.std(log_returns, ddof=1) / np.std(reference, ddof=1) - weight) < 1e-10


def test_buy_and_hold_and_rebalanced_blends_are_different_objects():
    # A single down-tick: buy-and-hold keeps half the loss, constant weight compounds it.
    nav = pd.Series([1.0, 0.5], index=pd.date_range("2020-01-01", periods=2))
    hold = budget.ladder(nav, weights=(0.5,), cash_rate=0.0, rebalance=False)["rows"][0]
    rebal = budget.ladder(nav, weights=(0.5,), cash_rate=0.0, rebalance=True)["rows"][0]
    assert hold["max_drawdown"] == pytest.approx(-0.25)
    assert rebal["max_drawdown"] == pytest.approx(-(1.0 - math.sqrt(0.5)))


def test_ladder_requires_no_alpha_at_full_exposure():
    nav = pd.Series(np.cumprod(1.0 + np.full(600, 0.001)))
    block = budget.ladder(nav, weights=(0.5, 1.0), cash_rate=0.0, rebalance=True)
    assert block["rows"][-1]["required_alpha"] == pytest.approx(0.0, abs=1e-12)
    assert block["rows"][0]["required_alpha"] > 0


def test_ladder_leaves_a_cash_only_row_without_a_ratio():
    nav = pd.Series([1.0, 1.1, 0.9, 1.3], index=pd.date_range("2020-01-01", periods=4))
    block = budget.ladder(nav, weights=(0.0, 1.0), cash_rate=0.02, rebalance=True)
    cash_row = block["rows"][0]
    assert cash_row["return_over_vol"] is None
    assert cash_row["sharpe_excess"] is None


def test_tracking_error_falls_with_the_square_root_of_the_book_size():
    te_ten = budget.tracking_error(0.4758, 10, 0.2455)
    te_forty = budget.tracking_error(0.4758, 40, 0.2455)
    assert abs(te_ten / te_forty - 2.0) < 1e-12


def test_tracking_error_is_zero_when_the_cross_section_has_no_spread():
    assert budget.tracking_error(0.10, 10, 0.20) == 0.0


def test_tracking_error_rejects_an_empty_book():
    with pytest.raises(ValueError):
        budget.tracking_error(0.4758, 0, 0.2455)


def test_cross_sectional_dispersion_matches_a_constant_spread_panel():
    index = pd.date_range("2020-01-01", periods=60)
    prices = pd.DataFrame({f"c{column}": np.linspace(10.0, 20.0, 60) * (1.0 + 0.01 * column)
                           for column in range(5)}, index=index)
    dispersion = budget.cross_sectional_dispersion(prices, horizon=20)
    assert dispersion > 0
    forward = prices.shift(-20) / prices - 1.0
    assert abs(dispersion - forward.std(axis=1, ddof=1).mean()) < 1e-12


def test_ic_power_scales_with_the_root_of_the_sample():
    short = budget.ic_power(100, 0.07)
    long = budget.ic_power(400, 0.07)
    assert abs(short["standard_error"] / long["standard_error"] - 2.0) < 1e-12
    assert abs(short["detectable_2sigma"] - 2 * short["standard_error"]) < 1e-15
    with pytest.raises(ValueError):
        budget.ic_power(0, 0.07)


def test_book_power_needs_a_decade_to_separate_small_alpha():
    result = budget.book_power(0.014, 0.0744, years=1.48)
    assert result["t_stat"] < 1.0
    longer = budget.book_power(0.014, 0.0744, years=10.0)
    assert longer["t_stat"] > result["t_stat"]


def test_book_power_refuses_a_degenerate_tracking_error():
    assert budget.book_power(0.014, 0.0, years=2.0)["t_stat"] is None


def test_transmission_reports_a_decreasing_tracking_error():
    payload = budget.transmission(sigma_cs=0.4758, market_vol=0.2455)
    errors = [row["tracking_error"] for row in payload["rows"]]
    assert errors == sorted(errors, reverse=True)
    assert payload["alpha_table"][0]["book_30"] < payload["alpha_table"][-1]["book_30"]


def _payload(nav, sigma_cs=0.4758, market_vol=0.20):
    return dict(code="TEST", start="2010-01-01", end="2020-12-31", cash_rate=0.02,
                bars=int(len(nav)), days=dict(reserve=359, protected=82),
                policies={"buyhold": budget.ladder(nav, budget.WEIGHTS, 0.02, False),
                          "rebalanced": budget.ladder(nav, budget.WEIGHTS, 0.02, True)},
                transmission=budget.transmission(sigma_cs=sigma_cs, market_vol=market_vol))


def test_verdict_reports_that_rebalancing_holds_sharpe():
    rng = np.random.default_rng(3)
    nav = pd.Series(np.cumprod(1.0 + rng.normal(0.0004, 0.013, size=2500)))
    result = budget.verdict(_payload(nav))
    # Exact in log space; the residual here is the geometric-vs-arithmetic convention.
    assert result["sharpe_span_rebalanced"] < 0.05
    assert result["rebalanced_70"]["required_alpha"] is not None
    assert result["best_calmar_exposure"] is not None and result["best_calmar_exposure"] > 0
    assert result["book_power_reserve"]["t_stat"] < 1.0
    assert set(result["required_ic_table"]) == {"0.7", "0.9"}


def test_verdict_ic_table_grows_with_the_holding_period():
    rng = np.random.default_rng(5)
    nav = pd.Series(np.cumprod(1.0 + rng.normal(0.0003, 0.011, size=2500)))
    rows = budget.verdict(_payload(nav))["required_ic_table"]["0.7"]
    values = [row["book_30"] for row in rows]
    assert values == sorted(values)
    assert [row["horizon"] for row in rows] == list(budget.HORIZONS)


def test_summarize_is_the_stage_two_function():
    assert budget.summarize is stage2.summarize
