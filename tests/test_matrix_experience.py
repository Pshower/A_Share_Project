"""Experience audit checks use synthetic data without PPO updates or downloads."""

from collections import Counter
import json

import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.action_mapping import LatentActionWrapper
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from src.training.matrix_experience import EpisodeAudit
from tests.test_trading_env import market


def synthetic_env(**kwargs):
    return LatentActionWrapper(
        TradingEnv(market(), initial_capital=100000, broker_config=BrokerConfig(max_weight=0.6), **kwargs),
        8.0,
    )


def assert_same(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_same(first, second)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


@pytest.mark.parametrize("recent", [0, 4])
def test_audit_preserves_observations_rewards_endings_and_random_sequence(recent):
    options = dict(initial_portfolio="mixed", max_steps=2, random_start=True, recent_episode_every=recent)
    plain = synthetic_env(**options)
    audited = EpisodeAudit(synthetic_env(**options))
    try:
        for episode in range(8):
            seed = 42 if episode == 0 else None
            assert_same(plain.reset(seed=seed), audited.reset(seed=seed))
            audited.stats()
            for action in [[8, -8, -8], [-8, 8, -8]]:
                assert_same(plain.step(np.array(action)), audited.step(np.array(action)))
            assert_same(plain.unwrapped.np_random.bit_generator.state,
                        audited.unwrapped.np_random.bit_generator.state)
        stats = audited.stats()
        assert stats["total_steps"] == 16
        assert stats["started_episodes"] == stats["completed_episodes"] == 8
        assert stats["mode_episodes"] == dict.fromkeys(["cash", "concentrated", "equal", "random"], 2)
        assert stats["mode_steps"] == dict.fromkeys(stats["mode_episodes"], 4)
        assert json.loads(json.dumps(audited.episodes)) == audited.episodes
    finally:
        plain.close()
        audited.close()


def test_initial_accounts_capture_actual_holdings_and_remain_immutable():
    env = EpisodeAudit(synthetic_env(initial_portfolio="equal"))
    try:
        _, info = env.reset(seed=42)
        first = json.loads(json.dumps(env.episodes[0]))
        assert first["initial_cash"] == info["cash"]
        assert first["initial_positions"] == {"000001": 4000.0, "600000": 4000.0}
        assert first["holding_count"] == 2
        assert len(first["account_sha256"]) == len(first["initial_state_sha256"]) == 64
        env.step(np.array([-8, -8, 8]))
        env.reset(seed=42)
        repeated = env.episodes[1]
        assert repeated["account_sha256"] == first["account_sha256"]
        assert repeated["initial_state_sha256"] == first["initial_state_sha256"]
        assert env.episodes[0]["initial_positions"] == first["initial_positions"]
        assert env.episodes[0]["initial_cash"] == first["initial_cash"]
        assert env.episodes[0]["rewards"] == [env.episodes[0]["reward_sum"]]
        assert env.stats()["started_episodes"] == 1
    finally:
        env.close()


def test_empty_resets_are_excluded_from_all_experience_counts():
    env = EpisodeAudit(synthetic_env(initial_portfolio="mixed"))
    try:
        assert env.stats() == dict(total_steps=0, started_episodes=0, completed_episodes=0,
                                   unique_initial_accounts=0, unique_initial_states=0,
                                   unique_decision_dates=0, decision_date_counts={},
                                   mode_steps={}, mode_episodes={})
        for seed in [42, 123, 2026]:
            env.reset(seed=seed)
        assert len(env.episodes) == 3
        assert env.stats()["unique_initial_accounts"] == env.stats()["started_episodes"] == 0
        env.step(np.zeros(3))
        stats = env.stats()
        assert stats["started_episodes"] == stats["unique_initial_accounts"] == 1
        assert stats["completed_episodes"] == 0
        assert stats["mode_episodes"] == {env.episodes[-1]["mode"]: 1}
    finally:
        env.close()


def test_vecenv_auto_reset_does_not_count_an_unexperienced_account():
    audit = EpisodeAudit(synthetic_env(initial_portfolio="mixed", max_steps=2))
    env = DummyVecEnv([lambda: Monitor(audit)])
    try:
        env.seed(42)
        env.reset()
        for _ in range(4):
            env.step(np.zeros((1, 3), dtype=np.float32))
        assert len(audit.episodes) == 3
        assert audit.episodes[-1]["steps"] == 0
        stats = audit.stats()
        assert stats["total_steps"] == 4
        assert stats["started_episodes"] == stats["completed_episodes"] == 2
        assert sum(stats["mode_episodes"].values()) == 2
        assert stats["unique_initial_accounts"] == len({row["account_sha256"] for row in audit.episodes[:2]})
        assert stats["decision_date_counts"] == {"2024-01-02": 2, "2024-01-03": 2}
        assert all(row["truncated"] and not row["terminated"] for row in audit.episodes[:2])
    finally:
        env.close()


def test_hashes_distinguish_dates_from_accounts_and_count_decision_dates():
    env = EpisodeAudit(synthetic_env(initial_portfolio="cash", max_steps=2, random_start=True))
    try:
        for seed in range(10):
            _, info = env.reset(seed=seed)
            start_date = info["date"].date().isoformat()
            _, reward, _, _, info = env.step(np.zeros(3))
            row = env.episodes[-1]
            assert row["decision_dates"] == [start_date]
            assert row["end_date"] == info["date"].date().isoformat()
            assert row["end_date"] != start_date
            assert row["rewards"] == [float(reward)]
            assert row["initial_positions"] == {}
        stats = env.stats()
        counts = Counter(row["start_date"] for row in env.episodes)
        assert stats["unique_initial_accounts"] == 1
        assert stats["unique_initial_states"] == stats["unique_decision_dates"] == len(counts) > 1
        assert stats["decision_date_counts"] == dict(counts)
        assert stats["total_steps"] == stats["started_episodes"] == 10
        assert stats["completed_episodes"] == 0
    finally:
        env.close()


def test_failed_step_does_not_change_experience_counts():
    env = EpisodeAudit(synthetic_env(max_steps=1))
    try:
        env.reset(seed=42)
        env.step(np.zeros(3))
        before = env.stats()
        with pytest.raises(RuntimeError, match="(?i)reset"):
            env.step(np.zeros(3))
        assert env.stats() == before
        assert env.episodes[0]["steps"] == 1
    finally:
        env.close()


def test_natural_termination_records_all_executed_rewards():
    env = EpisodeAudit(synthetic_env())
    try:
        env.reset(seed=42)
        rewards = []
        for _ in range(4):
            _, reward, terminated, truncated, _ = env.step(np.zeros(3))
            rewards.append(float(reward))
        assert terminated and not truncated
        row = env.episodes[0]
        assert row["completed"] and row["terminated"] and not row["truncated"]
        assert row["steps"] == 4
        assert row["rewards"] == rewards
        assert row["reward_sum"] == sum(rewards)
        assert row["end_date"] == "2024-01-08"
        assert env.stats()["completed_episodes"] == 1
    finally:
        env.close()


def test_account_identity_uses_actual_holdings_instead_of_mode_name():
    equal = EpisodeAudit(synthetic_env(initial_portfolio="equal"))
    concentrated = EpisodeAudit(synthetic_env(initial_portfolio="concentrated"))
    try:
        equal.reset(seed=42)
        concentrated.reset(seed=42)
        left, right = equal.episodes[0], concentrated.episodes[0]
        assert left["mode"] != right["mode"]
        assert left["initial_positions"] == right["initial_positions"]
        assert left["initial_cash"] == right["initial_cash"]
        assert left["account_sha256"] == right["account_sha256"]
        assert left["initial_state_sha256"] == right["initial_state_sha256"]
    finally:
        equal.close()
        concentrated.close()
