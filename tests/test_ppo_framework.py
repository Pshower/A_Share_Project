"""PPO integration tests never optimize a model or evaluate the held-out test split."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.action_mapping import LatentActionWrapper, map_action, MAPPING_VERSION
from src.agents.ppo_agent import PPOAgent, make_model
from src.backtest.run import run_policy, run_strategy
from src.backtest.strategies import Baseline
from src.data.preprocess import ROOT, DataPreprocessor, read_config, write_json
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from src.training.env_factory import ResearchData, validate_config
from src.training.evaluate import evaluate_agent, selection_score, main as evaluate_main
from src.training.train import main, ValidationCallback
from tests.test_research_pipeline import fixture_config
from tests.test_trading_env import market, panels


@pytest.fixture(autouse=True)
def no_parameter_updates(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Training/gradient updates are prohibited in offline framework checks")
    monkeypatch.setattr(PPO, "learn", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


def config():
    value = read_config(ROOT / "configs/ppo.json")
    value["hidden_sizes"] = [16, 8]
    value["ppo"].update(n_steps=8, batch_size=4)
    return value


def synthetic_env(codes=None, bound=8.0, max_steps=None):
    features, prices = panels()
    if codes is not None:
        features = features.loc[:, pd.IndexSlice[codes, :]]
        prices = prices.loc[:, pd.IndexSlice[codes, :]]
    env = TradingEnv(market(features, prices), initial_capital=10000,
                     broker_config=BrokerConfig(max_weight=0.5), max_steps=max_steps)
    return LatentActionWrapper(env, bound)


def contract(env, cfg):
    base = env.unwrapped
    return dict(stock_codes=base.market.stock_codes, feature_cols=base.market.feature_cols,
                lookback=1, broker=asdict(base.broker.config), initial_capital=10000,
                mapping_version=MAPPING_VERSION, logit_bound=cfg["logit_bound"],
                hidden_sizes=cfg["hidden_sizes"], price_basis="unadjusted",
                artifacts={"scaler_params.json": "same-scaler"})


@pytest.fixture
def local_data(tmp_path):
    research = fixture_config(tmp_path)
    DataPreprocessor(**research["dataset"]).process_all_stocks_and_save_scaler()
    research_path = tmp_path / "research.json"
    write_json(research_path, research)
    cfg = config()
    cfg.update(research_config=str(research_path), output_root=str(tmp_path / "ppo"))
    return ResearchData(cfg)


def test_action_mapping_caps_cash_masks_and_extremes():
    env = synthetic_env()
    obs, _ = env.reset()
    for raw in [[10000, -10000, 0], [0, 0, 0], [-10000, -10000, 10000]]:
        mapped = map_action(raw, obs, 0.05)
        assert np.isfinite(mapped["weights"]).all()
        assert (mapped["weights"] >= 0).all() and (mapped["weights"] <= 0.05).all()
        assert mapped["cash_weight"] + mapped["weights"].astype(float).sum() == pytest.approx(1)
    obs["valid_mask"][:] = 0
    assert map_action([0, 0, 0], obs, 0.05)["cash_weight"] == 1
    obs["valid_mask"][:] = 1
    obs["buy_mask"][0] = 0
    assert map_action([8, 0, -8], obs, 0.05)["weights"][0] == 0
    for bad in [[0, 0], [0, np.nan, 0], [0, np.inf, 0]]:
        with pytest.raises(ValueError):
            map_action(bad, obs, 0.05)


def test_wrapper_requires_reset_and_uses_same_inference_mapping():
    env = synthetic_env()
    with pytest.raises(RuntimeError):
        env.step([0, 0, 0])
    obs, _ = env.reset()
    expected = map_action([9, 1, -2], obs, 0.5)
    _, _, _, _, info = env.step([9, 1, -2])
    np.testing.assert_array_equal(info["action_diagnostics"]["weights"], expected["weights"])
    assert info["action_diagnostics"]["cash_weight"] == expected["cash_weight"]


def test_shared_policy_permutation_and_value_invariance():
    cfg = config()
    env = synthetic_env()
    model = make_model(env, cfg)
    obs, _ = env.reset()
    obs["portfolio"][:] = [0.2, 0.3, 0.5]
    permuted = {k: v.copy() for k, v in obs.items()}
    for key in ["stock_features", "valid_mask", "buy_mask", "sell_mask"]:
        permuted[key] = obs[key][[1, 0]]
    permuted["portfolio"] = obs["portfolio"][[1, 0, 2]]
    with torch.no_grad():
        a, value, _ = model.policy(model.policy.obs_to_tensor(obs)[0], deterministic=True)
        b, other_value, _ = model.policy(model.policy.obs_to_tensor(permuted)[0], deterministic=True)
    torch.testing.assert_close(a[:, [1, 0, 2]], b)
    torch.testing.assert_close(value, other_value)
    agent = PPOAgent(model, contract(env, cfg))
    np.testing.assert_allclose(agent.act(obs)[[1, 0]], agent.act(permuted), atol=1e-7)


def test_artifact_roundtrip_contract_hash_and_explicit_pool_transfer(tmp_path):
    cfg = config()
    small = synthetic_env(["000001"])
    agent = PPOAgent(make_model(small, cfg), contract(small, cfg))
    obs, _ = small.reset()
    agent.save(tmp_path / "saved", cfg)
    loaded = PPOAgent.load(tmp_path / "saved", agent.contract)
    np.testing.assert_array_equal(agent.act(obs), loaded.act(obs))
    with pytest.raises(FileExistsError):
        agent.save(tmp_path / "saved", cfg)
    big = synthetic_env()
    target = contract(big, cfg)
    with pytest.raises(ValueError, match="contract mismatch"):
        PPOAgent.load(tmp_path / "saved", target)
    transferred = loaded.transfer_to(big, cfg, target)
    assert sum(p.numel() for p in agent.model.policy.parameters()) == sum(p.numel() for p in transferred.model.policy.parameters())
    for key, value in agent.model.policy.state_dict().items():
        torch.testing.assert_close(value, transferred.model.policy.state_dict()[key])
    assert transferred.act(big.reset()[0]).shape == (2,)
    assert not transferred.model.policy.optimizer.state
    target["artifacts"]["scaler_params.json"] = "refitted"
    with pytest.raises(ValueError, match="scaler"):
        loaded.transfer_to(big, cfg, target)
    path = tmp_path / "saved/model.zip"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        PPOAgent.load(tmp_path / "saved", agent.contract)


class ContinueCallback(BaseCallback):
    def _on_step(self):
        return True


def test_sb3_rollout_stores_latent_log_probabilities_without_updates():
    cfg = config()
    cfg["logit_bound"] = 0.05
    env = DummyVecEnv([lambda: Monitor(synthetic_env(bound=0.05))])
    model = make_model(env, cfg)
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    # Synthetic rollout only: initialize SB3 bookkeeping, never call learn/train/backward.
    _, callback = model._setup_learn(8, ContinueCallback())
    callback.on_training_start({}, {})
    assert model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=8)
    assert (np.abs(model.rollout_buffer.actions) > 0.05).any()
    with torch.no_grad():
        for batch in model.rollout_buffer.get(batch_size=4):
            values, log_probs, entropy = model.policy.evaluate_actions(batch.observations, batch.actions)
            torch.testing.assert_close(log_probs, batch.old_log_prob)
            assert torch.isfinite(values).all() and torch.isfinite(entropy).all()
    for key, value in model.policy.state_dict().items():
        assert torch.equal(before[key], value)
    assert not model.policy.optimizer.state
    env.close()


def test_vecenv_terminal_observation_and_truncation():
    env = DummyVecEnv([lambda: Monitor(synthetic_env(max_steps=1))])
    first = env.reset()
    obs, reward, done, infos = env.step(np.zeros((1, 3), dtype=np.float32))
    assert done[0] and infos[0]["TimeLimit.truncated"]
    assert "terminal_observation" in infos[0]
    np.testing.assert_array_equal(obs["portfolio"], first["portfolio"])
    assert infos[0]["terminal_observation"]["portfolio"][-1] < 1
    env.close()
    env = DummyVecEnv([lambda: Monitor(synthetic_env())])
    env.reset()
    for _ in range(4):
        _, _, done, infos = env.step(np.zeros((1, 3), dtype=np.float32))
    assert done[0] and not infos[0]["TimeLimit.truncated"]
    env.close()


def test_sb3_bootstraps_truncation_with_terminal_value_only():
    cfg = config()
    env = DummyVecEnv([lambda: Monitor(synthetic_env(max_steps=1))])
    model = make_model(env, cfg)
    raw_rewards = []

    class CaptureRewards(BaseCallback):
        def _on_step(self):
            raw_rewards.append(float(self.locals["rewards"][0]))
            return True

    _, callback = model._setup_learn(8, CaptureRewards())
    model.policy.predict_values = lambda obs: torch.full((obs["portfolio"].shape[0], 1), 2.0)
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=8)
    np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0],
                               np.asarray(raw_rewards) + cfg["ppo"]["gamma"] * 2, atol=1e-6)
    assert not model.policy.optimizer.state
    env.close()


def test_targets_execute_buy_then_sell_and_frozen_holding_is_retained():
    env = synthetic_env()
    env.reset()
    _, _, _, _, bought = env.step([8, -8, -8])
    assert bought["positions"][0] > 0
    assert any(o["side"] == "buy" and o["filled_shares"] > 0 for o in bought["orders"])
    _, _, _, _, sold = env.step([-8, -8, 8])
    assert sold["positions"][0] == 0
    assert any(o["side"] == "sell" and o["filled_shares"] < 0 for o in sold["orders"])

    features, prices = panels()
    prices.loc[prices.index[2], ("000001", "suspended")] = 1
    frozen = LatentActionWrapper(TradingEnv(market(features, prices), initial_capital=10000,
                                           broker_config=BrokerConfig(max_weight=0.5)))
    frozen.reset()
    _, _, _, _, bought = frozen.step([8, -8, -8])
    _, _, _, _, blocked = frozen.step([-8, -8, 8])
    assert blocked["positions"][0] == bought["positions"][0]
    assert any(o["status"] != "filled" for o in blocked["orders"])


def test_data_factory_boundaries_subset_future_isolation(local_data):
    data = local_data
    train = data.market("train")
    assert train.panel.index.max() <= pd.Timestamp(data.manifest["train_end"])
    assert train.observation()[1].any()
    before = train.observation()[0].copy()
    data.features.loc[data.features.index > pd.Timestamp(data.manifest["train_end"])] = 12345
    np.testing.assert_array_equal(data.market("train").observation()[0], before)
    validation = data.market("val")
    assert validation.dates[0] == pd.Timestamp(data.manifest["train_end"])
    assert validation.panel.index.max() <= pd.Timestamp(data.manifest["val_end"])
    with pytest.raises(ValueError, match="test"):
        data.market("test")
    cfg = deepcopy(data.config)
    cfg["stock_codes"] = list(reversed(data.codes))
    assert ResearchData(cfg).market("train").stock_codes == cfg["stock_codes"]
    cfg["stock_codes"] = [data.codes[0]]
    assert ResearchData(cfg).market("train").n_stocks == 1


def test_data_factory_rejects_modified_artifact(local_data):
    directory = local_data.research["dataset"]["clean_dir"]
    from pathlib import Path
    path = Path(directory) / "scaler_params.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        ResearchData(local_data.config)


def test_data_factory_rejects_manifest_boundary_drift(local_data):
    from pathlib import Path
    path = Path(local_data.research["dataset"]["clean_dir"]) / "manifest.json"
    manifest = read_config(path)
    manifest["train_end"] = manifest["val_end"]
    write_json(path, manifest)
    with pytest.raises(ValueError, match="boundary mismatch"):
        ResearchData(local_data.config)


def test_check_cli_never_trains_and_evaluator_rejects_initialized_model(local_data, tmp_path):
    cfg_path = tmp_path / "ppo.json"
    write_json(cfg_path, local_data.config)
    output = main(["--config", str(cfg_path), "--run-id", "check"])
    report = read_config(output / "check.json")
    assert report["parameters_unchanged"] and report["save_load_equal"]
    assert report["trained"] is False and report["test_evaluated"] is False
    assert not list(output.glob("validation_*"))
    with pytest.raises(ValueError, match="Untrained"):
        evaluate_main(["--config", str(cfg_path), "--model", str(output / "initialized_model"),
                       "--output", str(tmp_path / "evaluation")])


@pytest.mark.parametrize("args", [["--train"], ["--train", "--total-timesteps", "0"],
                                    ["--total-timesteps", "8"], ["--train", "--check-only"]])
def test_cli_rejects_ambiguous_training_modes(args):
    with pytest.raises(SystemExit):
        main(args)


def test_common_evaluator_and_baseline_compatibility(local_data, tmp_path):
    data = local_data
    baseline = Baseline("equal_weight", data.manifest["feature_cols"],
                        max_weight=data.settings["broker"]["max_weight"],
                        rebalance_every=data.settings["rebalance_every"], top_k=data.settings["momentum_top_k"])
    original = run_strategy(data.market("val"), "equal_weight", data.settings)
    shared = run_policy(data.market("val"), baseline, data.settings)
    for a, b in zip(original[:3], shared[:3]):
        pd.testing.assert_frame_equal(a, b)
    env = data.vector_env()
    agent = PPOAgent(make_model(env, data.config), data.contract)
    before = {k: v.clone() for k, v in agent.model.policy.state_dict().items()}
    metrics = evaluate_agent(agent, data, tmp_path / "synthetic_connectivity", compare_baselines=True)
    assert metrics["trading_steps"] == 30
    scores = read_config(tmp_path / "synthetic_connectivity/metrics.json")
    assert set(scores) == {"ppo", "cash", "buy_hold", "equal_weight", "momentum", "equal_weight_daily", "momentum_daily"}
    for key, value in agent.model.policy.state_dict().items():
        assert torch.equal(before[key], value)
    env.close()


def test_validation_selection_and_schedule(local_data, tmp_path, monkeypatch):
    assert selection_score(dict(sharpe=None, cumulative_return=0)) is None
    assert selection_score(dict(sharpe=1, cumulative_return=0.2)) == (1, 0.2)
    callback = ValidationCallback(local_data, tmp_path)
    model = make_model(local_data.vector_env(), local_data.config)
    callback.init_callback(model)
    calls = []
    monkeypatch.setattr(callback, "evaluate", lambda: calls.append(model.num_timesteps))
    callback._on_rollout_start()
    assert not calls
    model.num_timesteps = callback.frequency
    callback._on_rollout_start()
    assert calls == [callback.frequency]
    model.get_env().close()


def test_invalid_config_and_nonfinite_observation():
    cfg = config()
    cfg["ppo"]["batch_size"] = 3
    with pytest.raises(ValueError, match="divide"):
        validate_config(cfg)
    cfg = config()
    env = synthetic_env()
    agent = PPOAgent(make_model(env, cfg), contract(env, cfg))
    obs, _ = env.reset()
    obs["stock_features"][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        agent.act(obs)
