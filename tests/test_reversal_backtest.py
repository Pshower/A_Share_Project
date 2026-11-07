"""Tests for the cost-aware reversal backtest: direction, turnover control, cost accounting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import reversal_backtest as rb


def observation(scores, held=None, eligible=None):
    """A minimal observation whose last-column feature carries the scores."""
    count = len(scores)
    eligible = np.ones(count, dtype=bool) if eligible is None else np.asarray(eligible, bool)
    features = np.asarray(scores, dtype=float).reshape(count, 1, 1)
    portfolio = np.zeros(count + 1, dtype=np.float32)
    if held is not None:
        portfolio[:count] = np.asarray(held, dtype=float) / max(1.0, np.sum(held))
    return dict(stock_features=features, valid_mask=eligible.astype(np.int8),
                buy_mask=eligible.astype(np.int8),
                sell_mask=eligible.astype(np.int8), portfolio=portfolio)


def test_the_ranked_loser_is_bought():
    strategy = rb.ReversalStrategy(["signal"], ["signal"], top_n=2, max_weight=0.6)
    obs = observation([0.9, 0.1, 0.5, 0.3])
    weights = strategy.act(obs)
    # Lowest feature value is the biggest loser and must score first.
    assert weights[1] > 0 and weights[3] > 0
    assert weights[0] == 0 and weights[2] == 0
    assert weights.sum() == pytest.approx(1.0)


def test_ineligible_names_are_never_bought():
    strategy = rb.ReversalStrategy(["signal"], ["signal"], top_n=2, max_weight=0.6)
    obs = observation([0.0, 0.1, 0.9, 0.5], eligible=[False, True, True, True])
    weights = strategy.act(obs)
    assert weights[0] == 0


def test_the_buffer_keeps_a_held_name_that_slipped_out_of_the_top_band():
    strategy = rb.ReversalStrategy(["signal"], ["signal"], top_n=2, max_weight=0.6, buffer_n=4)
    # Index 3 is the held name and now ranks fourth, outside the top two but inside the buffer.
    obs = observation([0.1, 0.2, 0.3, 0.9], held=[0, 0, 0, 1])
    weights = strategy.act(obs)
    assert weights[3] > 0
    assert weights[1] == 0
    assert weights.sum() == pytest.approx(1.0)


def test_without_a_buffer_the_book_is_rebuilt_from_the_top():
    strategy = rb.ReversalStrategy(["signal"], ["signal"], top_n=2, max_weight=0.6)
    obs = observation([0.1, 0.2, 0.3, 0.9], held=[0, 0, 0, 1])
    weights = strategy.act(obs)
    # The held name lost its slot, so rebuilding drops it for the better-ranked candidate.
    assert weights[3] == 0
    assert weights[1] > 0


def test_a_basket_without_a_known_feature_is_rejected():
    with pytest.raises(ValueError):
        rb.ReversalStrategy(["signal"], ["not_a_feature"], top_n=2, max_weight=0.5)


def test_describe_separates_gross_from_net():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    daily = pd.DataFrame(dict(date=dates, equity=[100.0, 101.0, 100.5, 102.0],
                              cash=0.0, daily_return=0.0, fee=[0.0, 1.0, 0.0, 0.0],
                              slippage_cost=[0.0, 2.0, 0.0, 0.0], turnover=[0.0, 0.1, 0.0, 0.0],
                              holding_count=[0, 3, 3, 3]))
    reference = pd.DataFrame(dict(date=dates, equity=[100.0, 100.2, 100.1, 100.4],
                                  cash=0.0, daily_return=0.0, fee=0.0, slippage_cost=0.0,
                                  turnover=0.0, holding_count=[0, 3, 3, 3]))
    row = rb.describe(daily, reference)
    assert row["cumulative_return"] == pytest.approx(0.02)
    # Gross adds back both charges against the opening equity of 100.
    assert row["cost_ratio"] == pytest.approx(0.03)
    assert row["gross_return"] == pytest.approx(0.05)
    assert row["excess_return"] == pytest.approx(0.02 - (100.4 / 100 - 1))
    assert row["annual_turnover"] == pytest.approx(row["mean_daily_turnover"] * 252)
