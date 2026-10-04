"""Optional tracing observes the same accounting path as ordinary backtests."""

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.run import run_policy
from src.envs.broker import BrokerConfig
from src.envs.market import MarketDataProvider


def market():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    features = pd.DataFrame({("000001", "signal"): np.arange(5) / 10,
                             ("600000", "signal"): np.arange(5) / 20}, index=dates)
    prices = pd.DataFrame({(code, field): values for code, initial in (("000001", 10), ("600000", 20))
                           for field, values in (("open", [initial] * 5),
                                                 ("close", np.arange(5) + initial),
                                                 ("volume", [10000] * 5))}, index=dates)
    return MarketDataProvider(features, ["signal"], price_path=prices, price_basis="unadjusted")


def settings():
    return dict(initial_capital=10000, seed=17, initial_portfolio="cash",
                broker=asdict(BrokerConfig(min_commission=5)))


class BuyThenHold:
    def act(self, observation, step):
        return np.array([.5, .2]) if step == 0 else None


def test_trace_preserves_all_backtest_tables_and_records_decision_state():
    plain = run_policy(market(), BuyThenHold(), settings())
    traces = []

    def sink(date, observation, action, info):
        traces.append(dict(date=date, execution_date=info["date"],
                           features=observation["stock_features"].copy(),
                           action=None if action is None else action.copy(),
                           positions=info["positions"].copy(), orders=list(info["orders"])))

    traced = run_policy(market(), BuyThenHold(), settings(), trace_sink=sink)
    for expected, actual in zip(plain[:3], traced[:3]):
        pd.testing.assert_frame_equal(expected, actual)
    assert plain[3] == traced[3]
    assert len(traces) == 4
    assert [row["date"] for row in traces] == list(traced[0].date.iloc[:-1])
    assert [row["execution_date"] for row in traces] == list(traced[0].date.iloc[1:])
    for i, row in enumerate(traces):
        np.testing.assert_allclose(row["features"][0, 0, 0], i / 10)
        np.testing.assert_array_equal(row["positions"], traced[1].iloc[i + 1].to_numpy())
    assert traces[0]["orders"]
    assert all(row["action"] is None and not row["orders"] for row in traces[1:])


def test_trace_failure_is_not_silently_reported_as_a_complete_backtest():
    def failed_trace(*args):
        raise OSError("trace storage failed")

    with pytest.raises(OSError, match="trace storage failed"):
        run_policy(market(), BuyThenHold(), settings(), trace_sink=failed_trace)
