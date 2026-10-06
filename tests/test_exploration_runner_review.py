"""Independent synthetic runner review; all learning and gradients are forbidden."""

from copy import deepcopy
import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.ppo_agent import PPOAgent
from src.data.preprocess import read_config, write_json
from src.envs.initial_account import MODES
from src.training import research_exploration as runner
from src.training.exploration_env import ExplorationData
from src.training.freeze import json_digest
from src.training.matrix_experience import EpisodeAudit
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def isolated_config(local_data, architecture="baseline", lookback=1):
    config = trade_config(local_data)
    config.update(lookback=lookback, training_episode_steps=8, recent_episode_every=4,
        torch_threads=1, actor_output_gain=.1, initial_portfolio="mixed",
        exploration=dict(architecture=architecture, common_lookback=5,
                         sampling="recent", reward_basis="simple_net", size_annealing=False))
    return config


def fake_records(include_pilots=True):
    metrics = dict(cumulative_return=.02, max_drawdown=-.02,
                   total_fees=100., total_slippage_cost=50.)
    account = dict(ppo=metrics, initialized=dict(metrics, cumulative_return=0.))
    records = []
    for cell in runner.FORMAL:
        for seed in runner.SEEDS:
            for budget, steps in runner.BUDGETS.items():
                records.append(dict(cell=cell, seed=seed, budget=budget, steps=steps,
                    pilot=False, parameters=126343,
                    metrics={mode: deepcopy(account) for mode in MODES}))
    if include_pilots:
        for cell in runner.PILOTS:
            records.append(dict(cell=cell, seed=42, budget="pilot", steps=2048,
                pilot=True, parameters=126343,
                metrics={mode: deepcopy(account) for mode in MODES}))
    return records


def fake_rules():
    return {mode: dict(momentum_daily=dict(cumulative_return=.01)) for mode in MODES}


@pytest.mark.parametrize("corruption", ["missing_pilots", "duplicate_pilot", "pilot_steps", "formal_steps"])
def test_summary_cannot_claim_unexecuted_or_misbudgeted_training(corruption):
    records = fake_records(include_pilots=corruption != "missing_pilots")
    if corruption == "duplicate_pilot":
        records.append(deepcopy(records[-1]))
    elif corruption == "pilot_steps":
        records[-1]["steps"] = 4096
    elif corruption == "formal_steps":
        records[0]["steps"] = 1024
    with pytest.raises(ValueError):
        runner.summarize(records, fake_rules())


def test_empty_trial_cache_is_not_accepted_as_a_completed_experiment(tmp_path, monkeypatch):
    protocol, config = dict(base_config={}), dict(synthetic=True)
    monkeypatch.setattr(runner, "config_for", lambda *args: config)
    monkeypatch.setattr(runner, "_check_sources", lambda *args: None)
    directory = tmp_path / "runs" / "baseline_l1_42"
    directory.mkdir(parents=True)
    write_json(directory / "completed.json", dict(config_sha256=json_digest(config),
        protocol_sha256=json_digest(protocol), files={}, records=[]))
    with pytest.raises(ValueError):
        runner.run_trial(tmp_path, protocol, "baseline_l1", 42)


@pytest.mark.parametrize("architecture,lookback", [("baseline", 1), ("baseline", 5), ("split", 1)])
def test_saved_untrained_exploration_model_and_native_evaluation_are_consistent(
        local_data, tmp_path, architecture, lookback):
    config = isolated_config(local_data, architecture, lookback)
    data = ExplorationData(config)
    env = data.vector_env()
    try:
        model = runner.make_exploration_model(env, config)
        agent = PPOAgent(model, data.contract)
        obs = {name: value[0] for name, value in env.reset().items()}
        before = {name: value.clone() for name, value in model.policy.state_dict().items()}
        destination = tmp_path / f"saved_{architecture}_{lookback}"
        agent.save(destination, config, extra=dict(synthetic_untrained=True))
        loaded = runner.load_agent(destination, data)
        assert loaded.model.num_timesteps == 0
        assert not loaded.model.policy.optimizer.state
        with torch.no_grad():
            original = agent.inspect_action(obs)["latent"]
            restored = loaded.inspect_action(obs)["latent"]
        np.testing.assert_array_equal(original, restored)
        for mode in MODES:
            data.settings.update(seed=runner.EVAL_SEED, initial_portfolio=mode, trade_top_n=2)
            folder = tmp_path / architecture / str(lookback) / mode
            result = runner.evaluate_case(loaded, data, folder, "synthetic_untrained")
            assert result["initial_mode"] == mode and result["test_evaluated"] is False
            assert result["mode_statistics"]["economic_violations"] == 0
            chunks = result["trace"]["chunks"]
            assert sum(chunk["steps"] for chunk in chunks) == result["metrics"]["trading_steps"]
            with np.load(folder / "trace" / chunks[-1]["path"], allow_pickle=False) as arrays:
                assert arrays["done"][-1]
                assert not arrays["done"][:-1].any()
                assert np.isfinite(arrays["reward"]).all()
                assert np.isfinite(arrays["log_prob"]).all()
                assert np.isfinite(arrays["value"]).all()
                np.testing.assert_array_equal(arrays["initial_mode"],
                    np.full(len(arrays["initial_mode"]), mode))
            saved_result = read_config(folder / "result.json")
            assert saved_result["metrics"] == result["metrics"]
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert model.num_timesteps == loaded.model.num_timesteps == 0
        assert not model.policy.optimizer.state
    finally:
        env.close()


def test_snapshot_schema_roundtrip_with_explicit_synthetic_optimizer_state(
        local_data, tmp_path, monkeypatch):
    """Fabricated counters verify schema only; never evidence of actual training."""
    config = isolated_config(local_data, "split")
    data = ExplorationData(config)
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    try:
        model = runner.make_exploration_model(env, config)
        model.set_logger(configure(str(tmp_path / "logs"), ["csv"]))
        initial = {name: value.clone() for name, value in model.policy.state_dict().items()}
        # Inherited production snapshots query CUDA counters even for CPU tests.
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: 0)
        output = tmp_path / "synthetic_snapshot"
        output.mkdir()
        callback = runner.ExplorationCallback(data, output, experience, initial,
            dict(synthetic=8), tmp_path / "status.json", "synthetic_only")
        try:
            model._setup_learn(total_timesteps=8, callback=callback)
            assert model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=8)
            assert model.num_timesteps == experience.stats()["total_steps"] == 8
            assert not model.policy.optimizer.state
            for parameter in model.policy.parameters():
                model.policy.optimizer.state[parameter] = dict(step=torch.tensor(1.),
                    exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
            callback.save_fixed_snapshot()
            audit = read_config(output / "synthetic_audit.json")
            assert audit["actual_timesteps"] == 8
            assert audit["optimizer_steps_min"] == audit["optimizer_steps_max"] == 1
            assert audit["changed_tensors"] == 0
            assert audit["tensor_delta_l2"] and all(value == 0 for value in audit["tensor_delta_l2"].values())
            assert audit["save_load_parameters_equal"] and audit["save_load_latent_close"]
            assert audit["training_trace_prefix"]["total_steps"] == 8
            loaded = runner.load_agent(output / "synthetic_model", data)
            assert loaded.model.num_timesteps == 8
            assert loaded.model.policy.architecture == "split"
            assert set(audit["tensor_delta_l2"]) == set(initial)
            for name, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
            write_json(output / "synthetic_test_only.json", dict(
                no_training=True, collected_synthetic_steps=8, fabricated_optimizer_counter=1,
                purpose="Schema/save-load test, not actual trained-model acceptance"))
        finally:
            callback.trace.close()
    finally:
        env.close()


def test_log_reward_callback_reports_simple_return_without_losing_learning_reward(
        local_data, tmp_path):
    config = isolated_config(local_data)
    config["initial_portfolio"] = "concentrated"
    config["exploration"]["reward_basis"] = "log_net"
    data = ExplorationData(config)
    # Synthetic unit-test quotes only; no on-disk history or scaler is changed.
    rates = np.exp(np.linspace(0., .2, len(data.prices)))
    for code in data.codes:
        for field in ("open", "close"):
            data.prices[(code, field)] *= rates
    data.settings["initial_capital"] = data.contract["initial_capital"] = 1e6
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    output = tmp_path / "log_rewards"
    output.mkdir()
    callback = None
    try:
        model = runner.make_exploration_model(env, config)
        model.set_logger(configure(str(output / "logs"), ["csv"]))
        initial = {name: value.clone() for name, value in model.policy.state_dict().items()}
        callback = runner.ExplorationCallback(data, output, experience, initial,
            dict(synthetic=8), tmp_path / "status.json", "synthetic_log_only")
        model._setup_learn(total_timesteps=8, callback=callback)
        assert model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=8)
        manifest = callback.trace.close()
        with np.load(output / "training_trace" / manifest["chunks"][0]["path"],
                     allow_pickle=False) as arrays:
            raw = arrays["after_equity"] / arrays["decision_equity"] - 1
            learning = arrays["reward"]
            np.testing.assert_allclose(learning, np.log1p(raw), rtol=1e-5, atol=1e-9)
            assert np.max(np.abs(raw - learning)) > 1e-10
            assert callback.portfolio_totals["raw_return"] == pytest.approx(raw.sum(), rel=1e-6)
            assert callback.portfolio_totals["learning_reward"] == pytest.approx(learning.sum(), rel=1e-6)
            assert callback.portfolio_counts["raw_return"] == callback.portfolio_counts["learning_reward"] == 8
        assert not model.policy.optimizer.state
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    finally:
        if callback is not None:
            callback.trace.close()
        env.close()
