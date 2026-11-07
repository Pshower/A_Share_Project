"""Unit checks for the new-data-block signal screen."""

import numpy as np
import pandas as pd

from tools import signal_expansion as expansion
from tools import signal_leaderboard as leaderboard


def _events(rows):
    return pd.DataFrame(rows, columns=["code", "available", "value"]).astype(
        {"code": object, "available": "datetime64[ns]", "value": float})


def test_next_trading_day_moves_strictly_forward_and_stops_at_the_edge():
    index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    stamps = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-31", None]))
    taken = expansion.next_trading_day(index, stamps)
    assert taken.iloc[0] == pd.Timestamp("2024-01-03")   # never the announcement day itself
    assert taken.iloc[1] == pd.Timestamp("2024-01-04")
    assert pd.isna(taken.iloc[2])                        # past the last stored trading row
    assert pd.isna(taken.iloc[3])


def test_latest_by_code_carries_the_most_recent_announcement_forward():
    index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
    events = _events([("000001", "2024-01-02", 10.0), ("000001", "2024-01-04", 20.0)])
    panel = expansion.latest_by_code(index, ["000001"], events, tolerance_days=400)
    assert panel["000001"].tolist() == [10.0, 10.0, 20.0, 20.0]


def test_latest_by_code_goes_stale_past_the_tolerance():
    index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-06-03"]))
    events = _events([("000001", "2024-01-02", 10.0)])
    panel = expansion.latest_by_code(index, ["000001"], events, tolerance_days=30)
    assert panel["000001"].iloc[0] == 10.0
    assert pd.isna(panel["000001"].iloc[1])


def test_decay_reaches_zero_and_leaves_the_pre_event_rows_empty():
    index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                                             "2024-01-05", "2024-01-08"]))
    events = _events([("000001", "2024-01-03", 1.0)])
    panel = expansion.decayed_by_code(index, ["000001"], events, decay=2)
    values = panel["000001"].tolist()
    assert pd.isna(values[0])
    assert values[1:] == [1.0, 0.5, 0.0, 0.0]


def test_decay_keeps_the_latest_event_when_two_land_on_one_row():
    index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    events = _events([("000001", "2024-01-02", 1.0), ("000001", "2024-01-02", -1.0)])
    panel = expansion.decayed_by_code(index, ["000001"], events, decay=10)
    assert panel["000001"].iloc[0] == -1.0


def test_daily_rank_ic_matches_the_shared_feature_screen():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=20)
    codes = [f"00000{index}" for index in range(1, 41)]
    values = rng.normal(size=(len(dates), len(codes)))
    target = values * 0.5 + rng.normal(size=values.shape) * 0.2

    mine = expansion.daily_rank_ic(values, target, dates)
    panel = pd.concat({code: pd.DataFrame({"probe": values[:, position]}, index=dates)
                       for position, code in enumerate(codes)}, axis=1)
    labels = pd.DataFrame(target, index=dates, columns=codes)
    theirs = leaderboard.daily_ic(panel, labels)["probe"]

    assert mine.notna().all() and theirs.notna().all()
    assert np.allclose(mine.to_numpy(), theirs.to_numpy(), atol=1e-12)


def test_daily_rank_ic_skips_rows_with_too_few_names():
    dates = pd.bdate_range("2024-01-01", periods=3)
    values = np.ones((3, 10))
    target = np.ones((3, 10))
    series = expansion.daily_rank_ic(values, target, dates, min_names=30)
    assert series.isna().all()


def test_markdown_renders_a_dash_for_a_signal_that_never_scored():
    payload = dict(horizon=20, version="v1", codes=300, rows=100, fundamentals="f",
                   coverage={"log_mcap": 0.9}, nonzero={}, earnings_holder_spearman=None,
                   windows={"train": dict(start="2017-01-03", end="2020-01-01", rows=10, table={
                       "log_mcap": dict(mean_ic=-0.0448, ic_std=0.21, stability=-0.21,
                                        hit_rate=0.419, dates=1665),
                       "earnings_score": None})})
    text = expansion._markdown(payload)
    assert "| log_mcap | -0.0448 |" in text
    assert "| earnings_score | — | — | — | — | 0 |" in text
    assert "## train 2017-01-03 ~ 2020-01-01" in text
