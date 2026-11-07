"""Fully invested action mapping: no cash latent, weights normalised to one under a cap.

The point of the mode is that "return minus the equal-weight pool" becomes a pure selection
result.  These tests pin the two properties that make that true and the one failure mode that
makes it false: the weights must sum to one, no name may exceed the cap, and when the eligible
names times the cap cannot reach one the shortfall must be reported rather than silently
absorbed as cash.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.action_mapping import (FULL_INVEST_VERSION, MAPPING_VERSION, LatentActionWrapper,
                                       cap_to_one, map_action, mapping_version)
from src.agents.ppo_policy import SharedStockPolicy


def observation(n, valid=None, buy=None):
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    buy = valid.copy() if buy is None else np.asarray(buy, dtype=bool)
    return dict(valid_mask=valid.astype(np.int8), buy_mask=buy.astype(np.int8))


def test_mapping_versions_distinguish_the_modes():
    assert mapping_version() == MAPPING_VERSION
    assert mapping_version(full_invest=True) == FULL_INVEST_VERSION
    assert mapping_version(full_invest=True) != mapping_version()


def test_cap_to_one_normalises_and_respects_the_cap():
    weights = cap_to_one(np.array([0.8, 0.15, 0.05]), 0.5)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.5 + 1e-12
    # The leading name is capped and the budget goes to the others, in order.
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] > weights[2]


def test_cap_to_one_reports_a_shortfall_when_the_cap_cannot_reach_one():
    weights = cap_to_one(np.array([0.5, 0.3, 0.2]), 0.1)
    assert weights.sum() == pytest.approx(0.3, abs=1e-12)
    assert weights.max() <= 0.1 + 1e-12


def test_cap_to_one_handles_an_empty_or_zero_vector():
    assert cap_to_one(np.zeros(4), 0.5).sum() == 0.0
    assert cap_to_one(np.array([]), 0.5).sum() == 0.0


def test_full_invest_mapping_takes_n_latents_not_n_plus_one():
    with pytest.raises(ValueError):
        map_action(np.zeros(4), observation(3), 0.5, full_invest=True)
    map_action(np.zeros(3), observation(3), 0.5, full_invest=True)


def test_full_invest_weights_sum_to_one_and_never_leave_cash():
    mapped = map_action(np.array([2.0, 1.0, 0.0, -1.0]), observation(4), 0.5, full_invest=True)
    assert mapped["weights"].astype(np.float64).sum() == pytest.approx(1.0, abs=1e-6)
    assert mapped["cash_weight"] == pytest.approx(0.0, abs=1e-6)
    assert mapped["feasible"] is True
    assert mapped["full_invest"] is True


def test_full_invest_ignores_names_the_agent_cannot_buy():
    mapped = map_action(np.array([5.0, 5.0, 0.0, 0.0]), observation(4, valid=[1, 1, 1, 1], buy=[1, 0, 1, 1]),
                        0.5, full_invest=True)
    assert mapped["weights"][1] == 0.0
    assert mapped["weights"].astype(np.float64).sum() == pytest.approx(1.0, abs=1e-6)


def test_full_invest_reports_infeasibility_instead_of_hiding_it():
    # Three eligible names at a 10% cap can never reach one.
    mapped = map_action(np.array([1.0, 0.0, 0.0]), observation(3), 0.1, full_invest=True)
    assert mapped["feasible"] is False
    assert mapped["cash_weight"] == pytest.approx(0.7, abs=1e-6)
    assert mapped["weights"].astype(np.float64).sum() == pytest.approx(0.3, abs=1e-6)


def test_cash_mode_still_keeps_its_cash_dimension():
    mapped = map_action(np.array([0.0, 0.0, 0.0]), observation(2), 0.5)
    assert mapped["full_invest"] is False
    # A uniform softmax over two names and cash leaves a third in cash.
    assert mapped["cash_weight"] > 0.0


def test_wrapper_action_space_matches_the_mode():
    import gymnasium as gym

    class FakeMarket:
        n_stocks = 7

    class FakeBroker:
        class config:
            max_weight = 0.5

    class FakeEnv(gym.Env):
        metadata = {}
        action_space = None
        observation_space = None
        reward_range = (-float("inf"), float("inf"))
        market = FakeMarket()
        broker = FakeBroker()

        def reset(self, **kwargs):
            return None, {}

        def step(self, action):
            return None, 0.0, False, False, {}

    assert LatentActionWrapper(FakeEnv(), 8.0).action_space.shape == (8,)
    assert LatentActionWrapper(FakeEnv(), 8.0, full_invest=True).action_space.shape == (7,)


def test_capital_override_is_validated():
    from src.training.env_factory import validate_config
    base = dict(research_config="configs/research.json", stock_codes=None, lookback=1, seed=42,
                device="cpu", torch_threads=1, hidden_sizes=[8, 8], logit_bound=8.0,
                validation_rollouts=10, output_root="reports/runs/tmp",
                ppo=dict(n_steps=8, batch_size=4, learning_rate=1e-3, n_epochs=1, gamma=0.99,
                         gae_lambda=0.95, clip_range=0.2, target_kl=0.03, ent_coef=0.0,
                         vf_coef=0.5, max_grad_norm=0.5))
    assert validate_config(dict(base, initial_capital=3e6))["initial_capital"] == 3e6
    # Whole-lot indivisibility binds before the strategy does, so a too-small account is a
    # configuration error rather than a modelling choice.
    for bad in (0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            validate_config(dict(base, initial_capital=bad))
    with pytest.raises(ValueError):
        validate_config(dict(base, action_full_invest="yes"))


def test_the_policy_emits_one_latent_per_stock_when_fully_invested():
    from gymnasium import spaces

    n, lookback, features = 5, 1, 3
    space = spaces.Dict({
        "stock_features": spaces.Box(-np.inf, np.inf, (n, lookback, features), np.float32),
        "portfolio": spaces.Box(0.0, 1.0, (n + 1,), np.float32),
        "receivable_ratio": spaces.Box(0.0, 1.0, (1,), np.float32),
        "valid_mask": spaces.MultiBinary(n),
        "buy_mask": spaces.MultiBinary(n),
        "sell_mask": spaces.MultiBinary(n),
    })
    schedule = lambda _: 3e-4
    policy = SharedStockPolicy(space, spaces.Box(-8.0, 8.0, (n,), np.float32), schedule,
                               full_invest=True)
    assert policy.network.full_invest is True
    assert not hasattr(policy.network, "cash_head")
    assert policy.action_dist.action_dim == n
    plain = SharedStockPolicy(space, spaces.Box(-8.0, 8.0, (n + 1,), np.float32), schedule)
    assert hasattr(plain.network, "cash_head")
    assert plain.action_dist.action_dim == n + 1
