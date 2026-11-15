"""The audit's arithmetic and its guardrails: no scipy, honest t, honest sample size."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import feature_audit as audit


def test_spearman_works_without_scipy():
    left = pd.Series([1, 2, 3, 4, 5, 6])
    right = pd.Series([2, 1, 4, 3, 6, 5])
    assert audit.spearman(left, right) == pytest.approx(14.5 / 17.5, abs=1e-9)


def test_a_rank_monotone_transform_does_not_change_spearman():
    left = pd.Series([1, 2, 3, 4, 5, 6.0])
    right = pd.Series(np.exp(left))
    assert audit.spearman(left, right) == pytest.approx(1.0)


def test_hac_finds_a_real_relation_but_not_noise():
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(size=400))
    signal = 0.4 * x + pd.Series(rng.normal(size=400))
    noise = pd.Series(rng.normal(size=400))
    assert abs(audit.hac_slope_t(x, signal, 1)) > 3
    assert abs(audit.hac_slope_t(x, noise, 1)) < 3


def test_hac_penalises_an_overlapping_target():
    """A persistent regressor plus an overlapping target must not count as independent evidence."""
    rng = np.random.default_rng(11)
    x = pd.Series(np.cumsum(rng.normal(size=600)))          # random walk: highly autocorrelated
    y = 0.05 * x + pd.Series(rng.normal(size=600))
    independent = abs(audit.hac_slope_t(x, y, 0))
    overlapping = abs(audit.hac_slope_t(x, y, 12))
    assert overlapping < independent


def test_effective_windows_shrink_with_the_horizon():
    assert audit.effective_n(240, 1) == 240
    assert audit.effective_n(240, 12) == 20
    assert audit.effective_n(240, 36) == 7


def test_a_short_series_gets_no_t_at_all():
    assert np.isnan(audit.hac_slope_t(pd.Series([1.0, 2.0]), pd.Series([1.0, 3.0]), 0))


def test_the_cross_sectional_ic_finds_a_planted_signal():
    dates = pd.bdate_range("2020-01-01", periods=60)
    codes = ["s%02d" % index for index in range(40)]
    rng = np.random.default_rng(3)
    rows = []
    for date in dates:
        factor = rng.normal(size=len(codes))
        noise = rng.normal(size=len(codes))
        for code, factor_value, noise_value in zip(codes, factor, noise):
            rows.append(dict(date=date, code=code, feature=factor_value,
                             close=100.0 * np.exp(noise_value * 0.01)))
    panel = pd.DataFrame(rows)
    # Plant the signal: tomorrow's close moves with the feature.
    panel = panel.sort_values(["code", "date"])
    panel["ret"] = 0.002 * panel.groupby("code")["feature"].shift(1) + 0.001 * panel["feature"] * 0
    panel["close"] = panel.groupby("code")["ret"].transform(lambda s: 100.0 * (1 + s.fillna(0)).cumprod())
    frame = panel.pivot(index="date", columns="code", values="close")
    shifted = frame.shift(-1) / frame - 1.0
    rows_out = audit.cross_sectional_ic(panel, "feature", horizons=(1,), min_names=30)
    assert rows_out and rows_out[0]["mean_ic"] > 0
