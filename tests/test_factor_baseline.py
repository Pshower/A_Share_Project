"""Synthetic causality, purging, ranking and cap checks for the factor baseline."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

factor = import_module("tools.factor_baseline")
FactorBaseline = factor.FactorBaseline
fit_factor = factor.fit_factor

CODES = ["600000", "600001", "600002"]
FEATURE_COLS = ["alpha", "beta"]


def panels(days=60, codes=CODES, features=FEATURE_COLS):
    dates = pd.bdate_range("2023-01-02", periods=days)
    rng = np.random.default_rng(7)
    values = rng.normal(size=(len(dates), len(codes), len(features)))
    panel = pd.DataFrame(values.reshape(len(dates), -1), index=dates,
                         columns=pd.MultiIndex.from_product([codes, features]))
    opens = rng.uniform(8, 12, size=(len(dates), len(codes)))
    closes = opens * (1 + rng.normal(0, .02, size=(len(dates), len(codes))))
    prices = pd.concat({"open": pd.DataFrame(opens, index=dates, columns=codes),
                        "close": pd.DataFrame(closes, index=dates, columns=codes)}, axis=1)
    prices = prices.swaplevel(0, 1, axis=1).sort_index(axis=1)
    prices.columns.names = ["code", "feature"]
    return panel, prices


def model_for(codes, features=FEATURE_COLS, weights=None):
    """A model whose score equals the first feature, so ranking is observable."""
    coef = [1.0] + [0.0] * (len(features) - 1) if weights is None else list(weights)
    return dict(feature_mean=[0.0] * len(features), feature_std=[1.0] * len(features),
                coef=coef, intercept=0.0, zero_variance_features=[False] * len(features),
                feature_cols=list(features), codes=list(codes))


def observation(first_feature, *, valid=None, buy=None, extra=None):
    n = len(first_feature)
    features = np.zeros((n, 1, len(FEATURE_COLS)))
    features[:, -1, 0] = first_feature
    if extra is not None:
        features[:, -1, 1] = extra
    return dict(stock_features=features,
                valid_mask=np.ones(n) if valid is None else np.asarray(valid),
                buy_mask=np.ones(n) if buy is None else np.asarray(buy))


# --- fitting -----------------------------------------------------------------

def test_fit_purges_labels_ending_after_the_training_boundary():
    panel, prices = panels()
    train_end = panel.index[30]
    model = fit_factor(panel, prices, CODES, FEATURE_COLS, train_end, horizon=5)
    assert model["fit"]["max_label_endpoint"] <= str(train_end.date())
    assert model["fit"]["label_end_purge_dates"] > 0
    assert model["fit"]["rows"] > 0
    assert model["horizon"] == 5
    assert model["feature_cols"] == FEATURE_COLS


def test_future_price_perturbation_does_not_change_the_training_fit():
    panel, prices = panels()
    train_end = panel.index[30]
    shifted = prices.copy()
    shifted.loc[shifted.index > train_end] *= 50
    base = fit_factor(panel, prices, CODES, FEATURE_COLS, train_end, horizon=5)
    moved = fit_factor(panel, shifted, CODES, FEATURE_COLS, train_end, horizon=5)
    np.testing.assert_allclose(base["coef"], moved["coef"])
    assert base["intercept"] == pytest.approx(moved["intercept"])


def test_fit_needs_at_least_one_purged_label_and_stays_inside_the_boundary():
    panel, prices = panels(days=6)
    # With a 5-day horizon and a 6-day history no label endpoint fits before train_end.
    with pytest.raises(ValueError, match="No valid purged examples"):
        fit_factor(panel, prices, CODES, FEATURE_COLS, panel.index[0], horizon=5)
    # A full-history boundary leaves exactly one label per stock, the fittable minimum.
    model = fit_factor(panel, prices, CODES, FEATURE_COLS, panel.index[-1], horizon=5)
    assert model["fit"]["rows"] == len(CODES)
    assert model["fit"]["max_label_endpoint"] <= str(panel.index[-1].date())


def test_fit_rejects_unknown_codes_and_bad_horizon():
    panel, prices = panels()
    with pytest.raises(ValueError, match="Panel lacks requested codes"):
        fit_factor(panel, prices, ["999999"], FEATURE_COLS, panel.index[30])
    with pytest.raises(ValueError, match="horizon must be positive"):
        fit_factor(panel, prices, CODES, FEATURE_COLS, panel.index[30], horizon=0)
    with pytest.raises(ValueError, match="alpha must be positive and finite"):
        fit_factor(panel, prices, CODES, FEATURE_COLS, panel.index[30], alpha=0.0)


# --- decision behaviour ------------------------------------------------------

def test_rebalances_only_on_registered_interval():
    strategy = FactorBaseline(model_for(CODES), FEATURE_COLS, CODES, rebalance_every=3)
    obs = observation([3.0, 2.0, 1.0])
    assert strategy.act(obs, 0) is not None
    assert strategy.act(obs, 1) is None
    assert strategy.act(obs, 2) is None
    assert strategy.act(obs, 3) is not None


def test_selects_descending_scores_and_breaks_ties_by_code():
    strategy = FactorBaseline(model_for(CODES), FEATURE_COLS, CODES, max_weight=1.0,
                              rebalance_every=1, top_k=2)
    weights = strategy.act(observation([1.0, 5.0, 5.0]), 0)
    # 600001 and 600002 tie on score; ascending code wins the second slot.
    assert weights[0] == 0
    assert weights[1] == pytest.approx(.5)
    assert weights[2] == pytest.approx(.5)


def test_respects_max_weight_cap():
    strategy = FactorBaseline(model_for(CODES), FEATURE_COLS, CODES, max_weight=.2,
                              rebalance_every=1, top_k=1)
    weights = strategy.act(observation([1.0, 9.0, 2.0]), 0)
    np.testing.assert_allclose(weights, [0, .2, 0])


def test_only_eligible_names_are_scored_and_selected():
    strategy = FactorBaseline(model_for(CODES), FEATURE_COLS, CODES, max_weight=1.0,
                              rebalance_every=1, top_k=3)
    weights = strategy.act(observation([1.0, 9.0, 5.0], valid=[True, False, True],
                                       buy=[True, True, False]), 0)
    np.testing.assert_allclose(weights, [1.0, 0, 0])


def test_nothing_eligible_returns_zero_weights_like_other_baselines():
    strategy = FactorBaseline(model_for(CODES), FEATURE_COLS, CODES, rebalance_every=1)
    weights = strategy.act(observation([1.0, 2.0, 3.0], valid=[False, False, False]), 0)
    np.testing.assert_allclose(weights, np.zeros(3))
    assert strategy.decisions[-1]["chosen"] == 0


def test_rejects_feature_order_pool_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="Feature order differs"):
        FactorBaseline(model_for(CODES), ["beta", "alpha"], CODES)
    with pytest.raises(ValueError, match="stock count differs"):
        FactorBaseline(model_for(CODES), FEATURE_COLS, CODES).act(observation([1.0, 2.0]), 0)
    bad = observation([1.0, 2.0, 3.0])
    bad["stock_features"][0, -1, 0] = np.nan
    with pytest.raises(ValueError, match="finite features"):
        FactorBaseline(model_for(CODES), FEATURE_COLS, CODES, rebalance_every=1).act(bad, 0)
    with pytest.raises(ValueError, match="positive"):
        FactorBaseline(model_for(CODES), FEATURE_COLS, CODES, rebalance_every=0)


def test_decisions_are_recorded_without_scoring_ineligible_names():
    strategy = FactorBaseline(model_for(CODES), FEATURE_COLS, CODES, rebalance_every=1, top_k=2)
    strategy.act(observation([1.0, 9.0, 5.0], valid=[True, False, True]), 0)
    record = strategy.decisions[-1]
    assert record["eligible"] == 2 and record["chosen"] == 2
    assert record["score_min"] == pytest.approx(1.0)
    assert record["score_max"] == pytest.approx(5.0)


def test_fitted_model_round_trips_through_the_strategy():
    panel, prices = panels()
    model = fit_factor(panel, prices, CODES, FEATURE_COLS, panel.index[30], horizon=5)
    strategy = FactorBaseline(model, FEATURE_COLS, CODES, max_weight=1.0, rebalance_every=1)
    weights = strategy.act(observation([0.5, 0.2, 0.9]), 0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert strategy.feature_cols == FEATURE_COLS
    assert int((weights > 0).sum()) == 3
