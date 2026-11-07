"""Unit checks for the cross-sectional IC leaderboard."""

import numpy as np
import pandas as pd
import pytest

from tools import signal_leaderboard as board


def synthetic(dates=40, codes=40):
    """Three features: perfectly ordered, perfectly reversed, and constant."""
    index = pd.bdate_range("2021-01-04", periods=dates)
    names = [f"c{position:02d}" for position in range(codes)]
    features = ["aligned", "reversed", "constant"]
    values = np.zeros((dates, codes, len(features)))
    for position in range(codes):
        values[:, position, 0] = position
        values[:, position, 1] = -position
    columns = pd.MultiIndex.from_product([names, features], names=["code", "feature"])
    panel = pd.DataFrame(values.reshape(dates, codes * len(features)), index=index, columns=columns)
    labels = pd.DataFrame(np.tile(np.arange(codes, dtype=float), (dates, 1)), index=index, columns=names)
    return panel, labels


def test_daily_ic_recovers_perfect_and_inverted_ordering():
    panel, labels = synthetic()
    daily = board.daily_ic(panel, labels)
    assert list(daily.columns) == ["aligned", "reversed", "constant"]
    assert daily["aligned"].dropna().to_numpy() == pytest.approx(np.ones(len(daily)))
    assert daily["reversed"].dropna().to_numpy() == pytest.approx(-np.ones(len(daily)))
    # A feature that never varies carries no ordering, so it must not report an IC.
    assert daily["constant"].isna().all()


def test_daily_ic_skips_dates_with_too_few_names():
    panel, labels = synthetic()
    thin = labels.copy()
    thin.iloc[:, 10:] = np.nan
    daily = board.daily_ic(panel, thin, min_names=30)
    assert daily.empty


def test_summarize_reports_mean_stability_and_hit_rate():
    daily = pd.DataFrame({"a": [0.2, 0.2, 0.2, -0.2], "b": [-0.1, -0.3, np.nan, -0.1]})
    table = board.summarize(daily)
    assert list(table.index) == ["a", "b"]  # ranked by mean IC, best first
    assert table.loc["a", "mean_ic"] == pytest.approx(0.1)
    assert table.loc["a", "hit_rate"] == pytest.approx(0.75)
    assert table.loc["b", "hit_rate"] == pytest.approx(0.0)
    assert table.loc["b", "dates"] == 3
    # A steady signal has positive stability; an unstable one does not.
    assert table.loc["a", "stability"] > 0 and table.loc["b", "stability"] < 0


def test_windows_split_on_the_manifest_boundaries():
    manifest = dict(start_date="2017-01-01", train_end="2023-03-31", val_end="2023-06-30")
    windows = board.windows(manifest)
    assert windows["train"] == (pd.Timestamp("2017-01-01"), pd.Timestamp("2023-03-31"))
    # The validation window starts strictly after train_end, so no label can straddle it.
    assert windows["val"] == (pd.Timestamp("2023-04-01"), pd.Timestamp("2023-06-30"))
