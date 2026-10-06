"""Synthetic signal-label causality, purging, constant-score and ridge checks."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import FEATURES, file_hash, read_config
from src.training.exploration_data import derive_fold
from src.training.exploration_signal import (fit_ridge, forward_labels, per_date_metrics, predict_ridge,
                                              run_signal_diagnostics, split_examples)
from tests.test_exploration_data import original_dataset


def simple_panels():
    dates = pd.bdate_range("2023-03-23", "2023-04-12")
    panel = pd.DataFrame(np.arange(len(dates) * 2).reshape(-1, 2), index=dates,
                         columns=pd.MultiIndex.from_product([["600000"], ["a", "b"]]))
    prices = pd.DataFrame(dict(open=np.arange(len(dates)) + 10, close=np.arange(len(dates)) + 11), index=dates)
    prices.columns = pd.MultiIndex.from_product([["600000"], prices.columns])
    return panel, prices


def test_one_day_label_is_next_open_to_same_close_and_five_day_endpoint():
    _, prices = simple_panels()
    one, end = forward_labels(prices, ["600000"], 1)
    assert one.iloc[0, 0] == pytest.approx(12 / 11 - 1)
    assert end.iloc[0] == prices.index[1]
    five, end = forward_labels(prices, ["600000"], 5)
    assert five.iloc[0, 0] == pytest.approx(16 / 11 - 1)
    assert end.iloc[0] == prices.index[5]
    assert five.iloc[-5:].isna().all().all()


def test_labels_do_not_jump_missing_intermediate_quotes():
    _, prices = simple_panels()
    prices.loc[prices.index[3], ("600000", "close")] = np.nan
    labels, _ = forward_labels(prices, ["600000"], 5)
    assert np.isnan(labels.iloc[0, 0])
    assert np.isnan(labels.iloc[2, 0])
    assert np.isfinite(labels.iloc[3, 0])


def test_split_purges_training_and_validation_endpoints_and_invalid_features():
    panel, prices = simple_panels()
    panel.loc[panel.index[0], ("600000", "a")] = np.nan
    labels, endpoint = forward_labels(prices, ["600000"], 3)
    _, _, train, audit = split_examples(panel, labels, endpoint, ["600000"], ["a", "b"], None, "2023-03-31")
    assert train.label_endpoint.max() <= pd.Timestamp("2023-03-31")
    assert panel.index[0] not in set(train.date)
    assert audit["label_end_purge_dates"] == 3
    _, _, validation, _ = split_examples(panel, labels, endpoint, ["600000"], ["a", "b"], "2023-04-01", "2023-04-11")
    assert validation.date.min() >= pd.Timestamp("2023-04-01")
    assert validation.label_endpoint.max() <= pd.Timestamp("2023-04-11")
    assert set(train.date).isdisjoint(validation.date)


def test_future_price_perturbations_do_not_change_purged_training_labels():
    panel, prices = simple_panels()
    alternative = prices.copy()
    alternative.loc[alternative.index > "2023-03-31"] *= 100
    first, end = forward_labels(prices, ["600000"], 3)
    second, alternative_end = forward_labels(alternative, ["600000"], 3)
    _, y_a, keys_a, _ = split_examples(panel, first, end, ["600000"], ["a", "b"], None, "2023-03-31")
    _, y_b, keys_b, _ = split_examples(panel, second, alternative_end, ["600000"], ["a", "b"], None, "2023-03-31")
    np.testing.assert_array_equal(y_a, y_b)
    pd.testing.assert_frame_equal(keys_a, keys_b)


def test_ridge_fixed_alpha_centering_constant_features_and_affine_invariance():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 4))
    x[:, -1] = 1
    y = .4 * x[:, 0] - .2 * x[:, 1] + 7
    a = fit_ridge(x, y)
    b = fit_ridge(x * [2, 3, 4, 5] + [40, -20, 5, 2], y)
    assert a["alpha"] == 1
    assert a["zero_variance_features"][-1]
    assert a["intercept"] == pytest.approx(y.mean())
    np.testing.assert_allclose(predict_ridge(a, x), predict_ridge(b, x * [2, 3, 4, 5] + [40, -20, 5, 2]), atol=1e-12)
    assert np.corrcoef(predict_ridge(a, x), y)[0, 1] > .999


def test_zero_score_has_undefined_ic_and_no_arbitrary_tie_spread():
    frame = pd.DataFrame(dict(date=[pd.Timestamp("2023-04-03")] * 50,
                              code=[f"{i:06d}" for i in range(50)], label=np.arange(50), zero=0.0,
                              ridge=np.arange(50)))
    zero = per_date_metrics(frame, "zero")[0]
    ridge = per_date_metrics(frame, "ridge")[0]
    assert zero["pearson_ic"] is None and zero["spearman_ic"] is None
    assert zero["top20_bottom20_gross_spread"] is None
    assert zero["zero_variance_score"]
    assert ridge["pearson_ic"] == pytest.approx(1)
    assert ridge["spearman_ic"] == pytest.approx(1)
    assert ridge["top20_bottom20_gross_spread"] == pytest.approx(30)


def test_full_synthetic_diagnostic_artifacts_are_purged_and_reproducible(tmp_path):
    _, _, _, research = original_dataset(tmp_path / "original")
    paths = derive_fold(tmp_path / "original", research, tmp_path / "fold", "2023")
    result = run_signal_diagnostics([paths], tmp_path / "signals")
    assert len(result["cases"]) == 3
    assert result["ridge_alpha"] == 1
    predictions = pd.read_parquet(tmp_path / "signals" / "predictions.parquet")
    assert predictions.date.min() >= pd.Timestamp("2023-04-01")
    assert predictions.label_endpoint.max() <= pd.Timestamp("2023-06-30")
    assert result["test_evaluated"] is False
    for case in result["cases"]:
        assert case["train"]["max_label_endpoint"] <= "2023-03-31"
        model = read_config(case["ridge_model"])
        assert model["feature_cols"] == FEATURES
        assert file_hash(case["ridge_model"]) == case["ridge_model_sha256"]
    for artifact, digest in result["artifacts"].items():
        assert file_hash(tmp_path / "signals" / artifact) == digest
    with pytest.raises(FileExistsError):
        run_signal_diagnostics([paths], tmp_path / "signals")
