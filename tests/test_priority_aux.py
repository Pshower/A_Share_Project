"""Synthetic label boundaries and PPO updates; no financial training or network."""

from copy import deepcopy
import json
import zipfile

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from src.data.preprocess import file_hash, read_config, write_json
from src.agents.priority_policy import PriorityTradePolicy
from src.training.priority_aux import (
    AUX_SEED_SALT, POOL_ARRAYS, RankAuxPPO, TrainRankPool, build_rank_pool,
)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def frames(length=24, n=4):
    dates = pd.bdate_range("2023-01-02", periods=length)
    codes, columns = tuple(f"{i:06d}" for i in range(n)), ("return", "momentum", "volatility")
    values = np.arange(length * n * 3).reshape(length, n * 3).astype(float) / 100
    panel = pd.DataFrame(values, index=dates, columns=pd.MultiIndex.from_product([codes, columns]))
    prices = pd.DataFrame(index=dates, columns=pd.MultiIndex.from_product([codes, ("open", "close")]), dtype=float)
    for i, code in enumerate(codes):
        prices[(code, "open")] = 10 + np.arange(length) * (.1 + i * .02)
        prices[(code, "close")] = prices[(code, "open")] * (1 + i * .002)
    return panel, prices, codes, columns


def pool():
    panel, prices, codes, columns = frames()
    return build_rank_pool(panel, prices, codes, columns, panel.index[19])


def test_training_boundary_is_physical_and_rank_endpoints_are_purged():
    labels = pool()
    assert len(labels.dates) == 20
    assert labels.date_indices.min() == 4
    assert labels.date_indices.max() == 14
    assert labels.size == 44
    assert labels.target_endpoints.max() == labels.dates[19]
    assert labels.manifest()["max_target_endpoint"] == "2023-01-27"
    assert labels.manifest()["last_decision_rows_purged"] == 5
    assert labels.manifest()["validation_labels_used"] is False
    assert labels.manifest()["account_inputs"] is False
    for date in np.unique(labels.date_indices):
        mask = labels.date_indices == date
        assert labels.targets[mask].min() == -1
        assert labels.targets[mask].max() == 1
        np.testing.assert_allclose(labels.targets[mask].mean(), 0, atol=1e-7)


def test_labels_start_at_next_open_not_decision_close_and_rank_on_same_day():
    panel, prices, codes, columns = frames()
    actual = build_rank_pool(panel, prices, codes, columns, panel.index[19])
    for row, (date, stock) in enumerate(zip(actual.date_indices, actual.stock_indices)):
        expected = prices[(codes[stock], "close")].iloc[date + 5] / prices[(codes[stock], "open")].iloc[date + 1] - 1
        assert actual.raw_returns[row] == pytest.approx(expected, abs=1e-15)


def test_post_training_features_and_prices_cannot_change_pool_or_input_identity():
    panel, prices, codes, columns = frames()
    original = build_rank_pool(panel, prices, codes, columns, panel.index[19])
    panel.iloc[20:] = np.nan
    prices.iloc[20:] = np.inf
    changed = build_rank_pool(panel, prices, codes, columns, panel.index[19])
    assert changed.identity_sha256 == original.identity_sha256
    for name in POOL_ARRAYS:
        np.testing.assert_array_equal(getattr(changed, name), getattr(original, name))


@pytest.mark.parametrize("field,value", [("open", np.nan), ("close", np.nan), ("open", 0), ("close", -1)])
def test_every_intervening_future_quote_is_required(field, value):
    panel, prices, codes, columns = frames()
    prices.loc[panel.index[8], (codes[0], field)] = value
    actual = build_rank_pool(panel, prices, codes, columns, panel.index[19])
    covered = actual.date_indices[actual.stock_indices == 0]
    assert not np.isin(covered, np.arange(4, 8)).any()
    assert 8 in covered or field == "close"
    assert 9 in covered


def test_common_feature_window_excludes_missing_history_and_masks_current_bad_close():
    panel, prices, codes, columns = frames()
    panel.loc[panel.index[6], (codes[0], columns[1])] = np.nan
    prices.loc[panel.index[13], (codes[1], "close")] = np.nan
    actual = build_rank_pool(panel, prices, codes, columns, panel.index[19])
    first = actual.date_indices[actual.stock_indices == 0]
    assert not np.isin(first, np.arange(6, 11)).any()
    second = actual.date_indices[actual.stock_indices == 1]
    assert 13 not in second
    assert 11 in first


def test_rank_ties_are_average_and_not_stock_order_tie_breaks():
    panel, prices, codes, columns = frames()
    for code in codes:
        prices[(code, "open")] = prices[(codes[0], "open")]
        prices[(code, "close")] = prices[(codes[0], "close")]
    actual = build_rank_pool(panel, prices, codes, columns, panel.index[19])
    assert np.count_nonzero(actual.targets) == 0


def test_single_label_stock_dates_are_not_used_for_rank_training():
    panel, prices, codes, columns = frames(n=1)
    with pytest.raises(ValueError, match="No complete"):
        build_rank_pool(panel, prices, codes, columns, panel.index[19])


def test_pool_arrays_are_immutable_and_sampling_does_not_touch_global_rng():
    actual = pool()
    for name in POOL_ARRAYS:
        with pytest.raises(ValueError):
            getattr(actual, name).flat[0] = 0
    np.random.seed(9)
    before = deepcopy(np.random.get_state())
    indices, inputs, targets = actual.sample(np.random.default_rng(42), 1024)
    after = np.random.get_state()
    assert len(indices) == len(inputs) == len(targets) == 1024
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    other = actual.sample(np.random.default_rng(42), 1024)
    for a, b in zip((indices, inputs, targets), other):
        np.testing.assert_array_equal(a, b)


def test_pool_roundtrip_checks_both_file_and_logical_hashes(tmp_path):
    original = pool()
    manifest = original.save(tmp_path / "pool")
    assert set(manifest["artifacts"]) == {name + ".npy" for name in POOL_ARRAYS}
    path = tmp_path / "pool" / "manifest.json"
    restored = TrainRankPool.load(path.parent, expected_manifest_sha256=file_hash(path))
    assert restored.identity_sha256 == original.identity_sha256
    with pytest.raises(FileExistsError):
        original.save(path.parent)
    with pytest.raises(ValueError, match="manifest SHA256"):
        TrainRankPool.load(path.parent, expected_manifest_sha256="0" * 64)
    manifest["examples"] += 1
    write_json(path, manifest)
    with pytest.raises(ValueError, match="logical identity"):
        TrainRankPool.load(path.parent)


def test_corrupted_pool_array_is_rejected(tmp_path):
    original = pool()
    original.save(tmp_path)
    features = tmp_path / "features.npy"
    features.write_bytes(features.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="artifact SHA256"):
        TrainRankPool.load(tmp_path)


@pytest.mark.parametrize("kind", ["duplicate_date", "unordered_date", "missing_price_date", "duplicate_column",
                                  "missing_open", "empty_codes", "duplicate_codes", "horizon_zero",
                                  "lookback_zero", "too_short"])
def test_invalid_label_input_schemas_are_rejected(kind):
    panel, prices, codes, columns = frames()
    kwargs = {}
    if kind == "duplicate_date":
        panel.index = panel.index.where(np.arange(len(panel)) != 1, panel.index[0])
        prices.index = panel.index
    elif kind == "unordered_date":
        panel, prices = panel.iloc[::-1], prices.iloc[::-1]
    elif kind == "missing_price_date":
        prices = prices.iloc[:-1]
    elif kind == "duplicate_column":
        panel = pd.concat([panel, panel.iloc[:, :1]], axis=1)
    elif kind == "missing_open":
        prices = prices.drop(columns=[(codes[0], "open")])
    elif kind == "empty_codes":
        codes = ()
    elif kind == "duplicate_codes":
        codes = codes + (codes[0],)
    elif kind == "horizon_zero":
        kwargs["horizon"] = 0
    elif kind == "lookback_zero":
        kwargs["common_lookback"] = 0
    elif kind == "too_short":
        panel, prices = panel.iloc[:6], prices.iloc[:6]
    with pytest.raises(ValueError):
        build_rank_pool(panel, prices, codes, columns, "2023-01-27", **kwargs)


def test_input_hashes_cannot_claim_different_physical_training_data():
    panel, prices, codes, columns = frames()
    with pytest.raises(ValueError, match="contradict"):
        build_rank_pool(panel, prices, codes, columns, panel.index[19],
                        input_hashes={"truncated_features": "0" * 64})


@pytest.mark.parametrize("boundary", ["NaT", "2026-07-01", "2026-10-11", "2023-01-27T00:00:00Z"])
def test_invalid_or_protected_training_boundaries_are_rejected(boundary):
    panel, prices, codes, columns = frames()
    with pytest.raises(ValueError):
        build_rank_pool(panel, prices, codes, columns, boundary)


class TinyEnv(gym.Env):
    """Small deterministic control task, unrelated to financial market data."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(-2., 2., (3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.index = 0

    def obs(self):
        return np.array([np.sin(self.index), np.cos(self.index), self.index / 5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self.obs(), {}

    def step(self, action):
        self.index += 1
        reward = float(np.sin(self.index) - .1 * np.square(action).sum())
        return self.obs(), reward, self.index == 4, False, {}


class AuxActorPolicy(ActorCriticPolicy):
    """Expose the Actor's first layer to the stock-only auxiliary test task."""

    def _build(self, schedule):
        super()._build(schedule)
        self.network = nn.Module()
        self.network.market_encoder = nn.Sequential(self.mlp_extractor.policy_net[0],
                                                    self.mlp_extractor.policy_net[1])
        self.network.signal_head = nn.Linear(8, 1)
        nn.init.zeros_(self.network.signal_head.weight)
        nn.init.zeros_(self.network.signal_head.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=schedule(1), **self.optimizer_kwargs)


class TinyPriorityEnv(TinyEnv):
    """Fixed legal synthetic requests: no broker, historical data or financial PnL."""

    def __init__(self):
        from tests.test_exploration_policy import observation, spaces
        tensor = observation(n=4, features=3)
        self.observation_space, _ = spaces(tensor)
        self.values = {key: value[0].numpy().copy() for key, value in tensor.items()}
        self.action_space = gym.spaces.Box(-8., 8., (12,), dtype=np.float32)
        self.index = 0

    def obs(self):
        return {name: value.copy() for name, value in self.values.items()}


def priority_model(algorithm, variant, **kwargs):
    return algorithm(PriorityTradePolicy, TinyPriorityEnv(), seed=13, n_steps=8, batch_size=4,
        n_epochs=2, learning_rate=3e-5, device="cpu", policy_kwargs=dict(hidden_sizes=(16, 8),
        actor_variant=variant), verbose=0, **kwargs)


def model(algorithm=RankAuxPPO, **kwargs):
    return algorithm(AuxActorPolicy, TinyEnv(), seed=13, n_steps=8, batch_size=4, n_epochs=2,
                     learning_rate=1e-3, device="cpu", policy_kwargs=dict(net_arch=dict(pi=[8], vf=[8])),
                     verbose=0, **kwargs)


def assert_same_optimizer(first, second):
    left, right = first.policy.optimizer.state_dict(), second.policy.optimizer.state_dict()
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for key in left["state"]:
        for name in left["state"][key]:
            torch.testing.assert_close(left["state"][key][name], right["state"][key][name], rtol=0, atol=0)


@pytest.mark.parametrize("clip_range_vf", [None, .2])
@pytest.mark.parametrize("entropy", [0., .01])
def test_zero_auxiliary_coefficient_exactly_replays_upstream_ppo(clip_range_vf, entropy):
    reference = model(PPO, clip_range_vf=clip_range_vf, ent_coef=entropy)
    reference.learn(total_timesteps=16)
    actual = model(clip_range_vf=clip_range_vf, ent_coef=entropy, aux_coef=0., aux_pool=pool(), aux_seed=900)
    actual.learn(total_timesteps=16)
    for key, value in reference.policy.state_dict().items():
        torch.testing.assert_close(actual.policy.state_dict()[key], value, rtol=0, atol=0)
    assert_same_optimizer(reference, actual)
    assert actual.actual_optimizer_updates == 8
    assert actual.aux_updates == actual.aux_sample_exposures == 0
    assert actual._n_updates == reference._n_updates
    assert len(actual.policy.optimizer.state) == len(reference.policy.optimizer.state)
    assert not actual.last_auxiliary_update
    reference.env.close()
    actual.env.close()


def test_auxiliary_update_trains_signal_head_without_an_additional_optimizer():
    actual = model(aux_pool=pool(), aux_coef=.05, aux_seed=77, aux_batch_size=16)
    actual.learn(total_timesteps=16)
    audit = actual.auxiliary_audit()
    assert actual.num_timesteps == 16
    assert actual.actual_optimizer_updates == actual.aux_updates == 8
    assert actual.aux_sample_exposures == 128
    assert audit["auxiliary_rl_steps"] == audit["extra_optimizer_steps"] == 0
    assert audit["seed_salt"] == AUX_SEED_SALT
    assert audit["last_auxiliary_update"]["raw_auxiliary_gradient_norm"] > 0
    assert audit["last_auxiliary_update"]["scaled_auxiliary_gradient_norm"] > 0
    assert torch.count_nonzero(actual.policy.network.signal_head.weight) > 0
    assert int(actual.policy.optimizer.state[actual.policy.network.signal_head.weight]["step"]) == 8
    assert np.isfinite(audit["last_optimization_audit"]["preclip_gradient_norm_mean"])
    actual.env.close()


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_real_priority_policy_zero_aux_replays_upstream_three_n_ppo_exactly(variant):
    reference = priority_model(PPO, variant)
    reference.learn(total_timesteps=16)
    actual = priority_model(RankAuxPPO, variant, aux_coef=0., aux_pool=pool(), aux_seed=110)
    actual.learn(total_timesteps=16)
    assert actual.rollout_buffer.actions.shape == (8, 12)
    for name, value in reference.policy.state_dict().items():
        torch.testing.assert_close(actual.policy.state_dict()[name], value, rtol=0, atol=0)
    assert_same_optimizer(reference, actual)
    assert actual.actual_optimizer_updates == 8
    assert actual.aux_updates == 0
    reference.env.close()
    actual.env.close()


@pytest.mark.parametrize("variant", ["coupled", "decoupled"])
def test_real_priority_policy_rank_method_has_exact_joint_update_counts(variant):
    actual = priority_model(RankAuxPPO, variant, aux_coef=.05, aux_pool=pool(), aux_seed=91, aux_batch_size=32)
    before = deepcopy(actual.policy.state_dict())
    actual.learn(total_timesteps=16)
    assert actual.actual_optimizer_updates == actual.aux_updates == 8
    assert actual.aux_sample_exposures == 256
    assert actual.rollout_buffer.actions.shape == (8, 12)
    assert torch.count_nonzero(actual.policy.network.signal_head.weight) > 0
    assert not torch.equal(before["network.market_encoder.0.weight"], actual.policy.network.market_encoder[0].weight)
    assert not torch.equal(before["network.value_head.0.weight"], actual.policy.network.value_head[0].weight)
    actual.env.close()


def test_auxiliary_gradients_reach_only_market_encoder_and_signal_head():
    actual = model(aux_pool=pool(), aux_coef=.05)
    with torch.no_grad():
        actual.policy.network.signal_head.weight.fill_(.1)
    loss, _ = actual._auxiliary_loss()
    gradients = torch.autograd.grad(loss, tuple(actual.policy.parameters()), allow_unused=True)
    eligible = {id(p) for p in actual.policy.network.market_encoder.parameters()}
    eligible |= {id(p) for p in actual.policy.network.signal_head.parameters()}
    for parameter, gradient in zip(actual.policy.parameters(), gradients):
        assert (gradient is not None) == (id(parameter) in eligible)
    assert any(float(gradient.norm()) > 0 for parameter, gradient in zip(actual.policy.parameters(), gradients)
               if id(parameter) in {id(p) for p in actual.policy.network.market_encoder.parameters()})
    actual.env.close()


def test_early_kl_rejection_performs_no_auxiliary_sampling_or_optimizer_step(monkeypatch):
    actual = model(aux_pool=pool(), aux_coef=.05, target_kl=.001)
    evaluate = actual.policy.evaluate_actions
    before_rng, before = deepcopy(actual.aux_rng_state), deepcopy(actual.policy.state_dict())

    def shifted(*args, **kwargs):
        values, log_prob, entropy = evaluate(*args, **kwargs)
        return values, log_prob + 2, entropy

    monkeypatch.setattr(actual.policy, "evaluate_actions", shifted)
    actual.learn(total_timesteps=8)
    assert actual.aux_rng_state == before_rng
    assert actual.actual_optimizer_updates == actual.aux_updates == actual.aux_sample_exposures == 0
    assert actual.last_optimization_audit["early_kl_stopped"] is True
    assert actual.last_optimization_audit["applied_mean_approx_kl"] is None
    for name, value in before.items():
        torch.testing.assert_close(actual.policy.state_dict()[name], value, rtol=0, atol=0)
    actual.env.close()


def test_auxiliary_training_without_labels_is_rejected_before_update():
    actual = model(aux_coef=.05)
    with pytest.raises(ValueError, match="label pool"):
        actual.train()
    assert not actual.policy.optimizer.state
    actual.env.close()


def test_auxiliary_learn_without_labels_is_rejected_before_any_environment_step():
    actual = model(aux_coef=.05)
    with pytest.raises(ValueError, match="label pool"):
        actual.learn(total_timesteps=16)
    assert actual.num_timesteps == 0
    assert not actual.policy.optimizer.state
    actual.env.close()


def test_loading_for_inference_does_not_require_or_serialize_label_pool(tmp_path):
    actual = model(aux_pool=pool(), aux_coef=.05, aux_seed=37, aux_batch_size=16)
    actual.learn(total_timesteps=16)
    actual.save(tmp_path / "model")
    with zipfile.ZipFile(tmp_path / "model.zip") as archive:
        data = json.loads(archive.read("data"))
        assert "aux_pool" not in data and "_aux_rng" not in data
    restored = RankAuxPPO.load(tmp_path / "model", device="cpu")
    assert restored.aux_pool is None
    assert restored.auxiliary_audit() == actual.auxiliary_audit()
    obs = TinyEnv().reset()[0]
    np.testing.assert_array_equal(actual.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
    with pytest.raises(ValueError, match="label pool"):
        restored.train()
    restored.attach_aux_pool(pool())
    expected, observed = actual.aux_pool.sample(actual._aux_rng, 16), restored.aux_pool.sample(restored._aux_rng, 16)
    for first, second in zip(expected, observed):
        np.testing.assert_array_equal(first, second)
    actual.env.close()


def test_different_pool_cannot_be_attached_to_saved_label_identity():
    actual = model(aux_pool=pool(), aux_coef=.05)
    panel, prices, codes, columns = frames()
    panel.iloc[10, 0] += .01
    other = build_rank_pool(panel, prices, codes, columns, panel.index[19])
    with pytest.raises(ValueError, match="saved identity"):
        actual.attach_aux_pool(other)
    actual.env.close()


@pytest.mark.parametrize("kwargs", [dict(aux_coef=-1), dict(aux_coef=np.nan), dict(aux_coef=np.inf),
    dict(aux_coef=True), dict(aux_batch_size=0), dict(aux_batch_size=True), dict(aux_seed=-1),
    dict(aux_seed=True), dict(aux_huber_delta=0), dict(aux_huber_delta=np.nan), dict(aux_pool="bad")])
def test_invalid_auxiliary_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        model(**kwargs)
