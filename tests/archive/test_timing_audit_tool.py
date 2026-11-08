"""Audit-tool arithmetic tests use plain arrays, not markets, models or envs."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.archive.audit_timing_study import book, expected_vector, expected_vector_batch
from src.data.preprocess import FEATURES


def account_arrays():
    value = dict(
        decision_date=np.array(["2023-03-31"]), execution_date=np.array(["2023-04-03"]),
        decision_marks=np.array([[10.]]), opening_prices=np.array([[10.]]),
        opening_valuation_prices=np.array([[10.]]), decision_positions=np.array([[0.]]),
        decision_cash=np.array([1e6]), decision_receivables=np.array([0.]), decision_equity=np.array([1e6]),
        opening_positions=np.array([[0.]]), opening_cash=np.array([1e6]), opening_receivables=np.array([0.]),
        opening_equity=np.array([1e6]), opening_available_positions=np.array([[0.]]),
        after_positions=np.array([[1000.]]), after_available_positions=np.array([[0.]]),
        after_cash=np.array([989985.]), after_receivables=np.array([0.]), after_equity=np.array([1000185.]),
        execution_filled_shares=np.array([[1000.]]), execution_filled_notional=np.array([[10010.]]),
        execution_fee=np.array([[5.]]), execution_slippage_cost=np.array([[10.]]),
        action_exit_mask=np.array([[False]]), obs_portfolio=np.array([[0., 1.]], np.float32),
        obs_valid_mask=np.array([[1]], np.int8),
    )
    market = SimpleNamespace(
        indices=lambda dates: np.array([0 if str(date) == "2023-03-31" else 1 for date in dates]),
        mark=np.array([[10.], [10.2]]), open=np.array([[10.], [10.]]),
        features=np.zeros((2, 1, len(FEATURES))), valid=np.ones((2, 1), bool),
    )
    contract = dict(broker=dict(commission=.0003, min_commission=5., sell_tax=.0005,
                               slippage=.001, lot_size=100),
                    trade=dict(max_commission_ratio=.001, max_trade_weight=.05))
    return value, market, contract


def test_pure_book_reconciles_integer_fill_cost_and_t1_without_an_environment():
    data, market, contract = account_arrays()
    index, stats = book(data, market, contract)
    np.testing.assert_array_equal(index, [0])
    assert stats == dict(steps=1, fills=1, full_exits=0, fees=5., slippage=10.)


@pytest.mark.parametrize("field", ["after_cash", "after_equity", "after_available_positions",
                                  "execution_fee", "execution_slippage_cost", "after_positions"])
def test_pure_book_rejects_independent_ledger_errors(field):
    data, market, contract = account_arrays()
    data[field] += 1.
    with pytest.raises(ValueError):
        book(data, market, contract)


def test_pure_vector_is_fixed_phase_current_candidates_and_account_only():
    n = 3
    features = np.arange(n * 20, dtype=np.float32).reshape(n, 1, 20)
    obs = dict(stock_features=features, valid_mask=np.ones(n, np.int8),
        portfolio=np.array([.1, .2, .3, .4], np.float32), receivable_ratio=np.zeros(1, np.float32),
        legal_directions=np.tile([1, 1, 1, 1], (n, 1)), share_state=np.ones((n, 2)),
        execution_features=np.zeros((n, 3), np.float32), allocation_context=np.zeros(7, np.float32),
        top_n=np.array([2], np.float32))
    vector = expected_vector(obs, deepcopy(obs), 7, 1e6, 1e6, ["000001", "000002", "000003"])
    assert vector.shape == (129,) and vector.dtype == np.float32
    np.testing.assert_array_equal(vector[:20], np.eye(20)[7])
    np.testing.assert_array_equal(vector[80:100], features[[2, 1], -1].mean(0))
    assert vector[-1] == 0
    data = {"obs_" + name: np.stack([value, value]) for name, value in obs.items()
            if name != "stock_features"}
    data.update(decision_receivables=np.zeros(2), decision_equity=np.array([1e6, 2e6]),
                action_phase=np.array([7, 8]))
    market = SimpleNamespace(features=np.stack([features[:, 0], features[:, 0]]))
    batch = expected_vector_batch(data, deepcopy(data), market, np.array([0, 1]), ["000001", "000002", "000003"])
    np.testing.assert_array_equal(batch[0], vector)
    np.testing.assert_array_equal(batch[1], expected_vector(obs, deepcopy(obs), 8, 2e6, 2e6, ["000001", "000002", "000003"]))
