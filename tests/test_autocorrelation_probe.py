"""Tests for the autocorrelation probe: estimators, the T+1 label, and the null hypothesis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import autocorrelation_probe as probe


def test_pooled_autocorrelation_recovers_a_known_ar1():
    rng = np.random.default_rng(3)
    size, columns, phi = 800, 40, -0.1
    values = np.empty((size, columns))
    values[0] = rng.standard_normal(columns)
    for step in range(1, size):
        values[step] = phi * values[step - 1] + rng.standard_normal(columns)
    stats = probe.per_date_products(probe.demean(values), lags=5)
    rhos = probe.pooled_autocorrelations(stats, size)
    assert rhos[1] == pytest.approx(phi, abs=0.05)


def test_pooled_autocorrelation_is_near_zero_for_white_noise():
    rng = np.random.default_rng(4)
    values = rng.standard_normal((600, 40))
    stats = probe.per_date_products(probe.demean(values), lags=5)
    rhos = probe.pooled_autocorrelations(stats, 600)
    assert abs(rhos[1]) < 0.02


def test_variance_ratio_matches_its_definition():
    rhos = np.zeros(probe.AUTOCORR_LAGS + 1)
    rhos[1] = 0.1
    rhos[2] = 0.05
    # VR(5) = 1 + 2 * ((1 - 1/5) * 0.1 + (1 - 2/5) * 0.05)
    assert probe.variance_ratio(rhos, 5) == pytest.approx(1 + 2 * (0.8 * 0.1 + 0.6 * 0.05))
    assert probe.variance_ratio(np.zeros(probe.AUTOCORR_LAGS + 1), 20) == pytest.approx(1.0)


def test_summarise_tests_a_variance_ratio_against_one_not_zero():
    samples = np.full(200, 1.06) + np.random.default_rng(5).standard_normal(200) * 0.03
    ratio = probe.summarise(1.06, samples, "vr_5", null=1.0)
    correlation = probe.summarise(0.03, samples, "rho_1", null=0.0)
    assert ratio["t"] == pytest.approx((1.06 - 1.0) / samples.std(ddof=1))
    assert correlation["t"] == pytest.approx(0.03 / samples.std(ddof=1))
    # The mistake this guards against: 1.06 / se would be about twenty.
    assert abs(ratio["t"]) < 5


def test_open_to_open_label_buys_the_next_open_and_sells_a_later_one():
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    opens = {"000001": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}
    prices = pd.DataFrame({(code, "open"): series for code, series in opens.items()},
                          index=dates)
    prices[("000001", "close")] = opens["000001"]
    labels, endpoint = probe.open_to_open_labels(prices, ["000001"], 1)
    # Decide at t, buy at open[t+1], sell at open[t+2].
    assert labels[("000001")].iloc[0] == pytest.approx(12.0 / 11.0 - 1)
    assert labels[("000001")].iloc[1] == pytest.approx(13.0 / 12.0 - 1)
    assert endpoint.iloc[0] == dates[2]
    # The last two decisions cannot settle inside the frame.
    assert labels.iloc[-2:].isna().all().all()


def test_open_to_open_label_requires_a_complete_path():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    prices = pd.DataFrame({("000001", "open"): [10.0, 11.0, np.nan, 13.0],
                           ("000001", "close"): [10.0, 11.0, 12.0, 13.0]}, index=dates)
    labels, _ = probe.open_to_open_labels(prices, ["000001"], 1)
    assert labels.iloc[0].isna().all()
    assert labels.iloc[1].isna().all()


def test_volatility_standardisation_clips_and_looks_backwards_only():
    rng = np.random.default_rng(6)
    values = pd.DataFrame(rng.standard_normal((80, 3)) * 0.01)
    values.iloc[70] = 0.5  # an implausible jump relative to its trailing scale
    scaled = probe.volatility_standardise(values, window=20)
    array = scaled.to_numpy()
    assert np.nanmax(array) <= probe.CLIP + 1e-12
    assert np.nanmin(array) >= -probe.CLIP - 1e-12
    # The first rows have no trailing scale yet, so they must stay missing rather than leak.
    assert scaled.iloc[:5].isna().all().all()


def test_the_tradable_horizon_is_flagged():
    assert probe.MIN_EXECUTABLE_HORIZON == 2
