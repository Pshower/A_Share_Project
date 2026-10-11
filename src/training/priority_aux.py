"""Train-only rank supervision and jointly optimized, research-only PPO."""

# Adapted PPO train-loop portions are covered by the upstream MIT License:
# Copyright (c) 2019 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from src.data.preprocess import file_hash, read_config, write_json
from .exploration_data import PROTECTED_START
from .exploration_signal import forward_labels


RANK_POOL_VERSION = "train_only_open_to_close_rank_v1"
AUX_SEED_SALT = 0xA071
POOL_ARRAYS = ("features", "targets", "raw_returns", "date_indices", "stock_indices",
               "dates", "target_endpoints")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _array_digest(array):
    value = np.ascontiguousarray(array)
    result = hashlib.sha256()
    result.update(str(value.dtype).encode("ascii"))
    result.update(json.dumps(list(value.shape)).encode("ascii"))
    result.update(value.tobytes())
    return result.hexdigest()


def _frame_digest(frame):
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(json.dumps([list(column) if isinstance(column, tuple) else column
                             for column in frame.columns], ensure_ascii=True).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainRankPool:
    """Immutable stock-date examples stored separately from policy observations."""

    features: np.ndarray
    targets: np.ndarray
    raw_returns: np.ndarray
    date_indices: np.ndarray
    stock_indices: np.ndarray
    dates: np.ndarray
    target_endpoints: np.ndarray
    codes: tuple
    feature_cols: tuple
    train_end: str
    horizon: int
    common_lookback: int
    input_hashes: dict

    def __post_init__(self):
        float_arrays = {"features": np.float32, "targets": np.float32, "raw_returns": np.float64}
        integer_arrays = {"date_indices": np.int64, "stock_indices": np.int64}
        for name in POOL_ARRAYS:
            original = np.asarray(getattr(self, name))
            if original.dtype == object:
                raise ValueError("Rank pool arrays cannot have object dtype")
            if name in float_arrays and (not np.issubdtype(original.dtype, np.number)
                                        or np.issubdtype(original.dtype, np.complexfloating)):
                raise ValueError("Rank features and targets must be real numeric arrays")
            dtype = float_arrays.get(name, integer_arrays.get(name, "datetime64[ns]"))
            if name in integer_arrays and (not np.issubdtype(original.dtype, np.integer)
                                          or np.issubdtype(original.dtype, np.bool_)):
                raise ValueError("Rank pool indices must be integer arrays")
            value = np.array(original, dtype=dtype, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        codes, columns = tuple(self.codes), tuple(self.feature_cols)
        hashes = deepcopy(self.input_hashes)
        if (not codes or len(codes) != len(set(codes)) or not columns or len(columns) != len(set(columns))
                or any(not isinstance(item, str) or not item for item in codes + columns)
                or type(self.horizon) is not int or self.horizon < 1
                or type(self.common_lookback) is not int or self.common_lookback < 1
                or not isinstance(hashes, dict) or not hashes):
            raise ValueError("Invalid rank pool schema")
        for key, value in hashes.items():
            if (not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError("Rank pool input hashes must be SHA256 hex digests")
        if self.targets.ndim != 1:
            raise ValueError("Rank pool targets must be a one-dimensional array")
        rows = len(self.targets)
        if (self.features.shape != (rows, len(columns)) or not rows
                or any(getattr(self, name).shape != (rows,) for name in
                       ("targets", "raw_returns", "date_indices", "stock_indices", "target_endpoints"))
                or self.dates.ndim != 1 or len(self.dates) < self.horizon + 1
                or np.isnat(self.dates).any() or np.isnat(self.target_endpoints).any()
                or not (self.dates[1:] > self.dates[:-1]).all()
                or any(not np.isfinite(getattr(self, name)).all() for name in float_arrays)
                or (np.abs(self.targets) > 1).any()
                or (self.date_indices < self.common_lookback - 1).any()
                or (self.date_indices + self.horizon >= len(self.dates)).any()
                or (self.stock_indices < 0).any() or (self.stock_indices >= len(codes)).any()):
            raise ValueError("Invalid rank pool examples or boundaries")
        timestamp = pd.Timestamp(self.train_end)
        if pd.isna(timestamp) or timestamp.tz is not None or timestamp >= PROTECTED_START:
            raise ValueError("Rank-pool boundary cannot expose the protected period")
        boundary = np.datetime64(timestamp.normalize(), "ns")
        if (self.dates[-1] > boundary or (self.target_endpoints > boundary).any()
                or not np.array_equal(self.target_endpoints, self.dates[self.date_indices + self.horizon])):
            raise ValueError("Rank pool labels must end within the training boundary")
        keys = np.column_stack([self.date_indices, self.stock_indices])
        if len(np.unique(keys, axis=0)) != rows:
            raise ValueError("Duplicate stock-date examples in rank pool")
        for date in np.unique(self.date_indices):
            selected = self.date_indices == date
            if selected.sum() < 2:
                raise ValueError("Rank supervision needs at least two stocks per date")
            values = pd.Series(self.raw_returns[selected]).rank(method="average").to_numpy()
            expected = 2 * (values - 1) / (len(values) - 1) - 1
            if not np.allclose(self.targets[selected], expected, rtol=0, atol=6e-8):
                raise ValueError("Rank targets differ from their original cross-sectional labels")
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "feature_cols", columns)
        object.__setattr__(self, "input_hashes", hashes)
        object.__setattr__(self, "train_end", str(pd.Timestamp(self.train_end).date()))

    @property
    def size(self):
        return len(self.targets)

    @property
    def identity_sha256(self):
        return _digest(self.manifest())

    def manifest(self):
        return dict(schema_version=1, version=RANK_POOL_VERSION, examples=self.size,
            decision_dates=int(len(np.unique(self.date_indices))), stock_count=len(self.codes),
            stocks_with_examples=int(len(np.unique(self.stock_indices))), stock_codes=list(self.codes),
            feature_cols=list(self.feature_cols), lookback=1, common_lookback=self.common_lookback,
            horizon=self.horizon, train_end=self.train_end,
            min_decision_date=str(pd.Timestamp(self.dates[self.date_indices.min()]).date()),
            max_decision_date=str(pd.Timestamp(self.dates[self.date_indices.max()]).date()),
            max_target_endpoint=str(pd.Timestamp(self.target_endpoints.max()).date()),
            last_decision_rows_purged=self.horizon, target_range=[-1, 1], rank_ties="average",
            label_definition="close[t+h] / open[t+1] - 1; complete positive future open/close path",
            input_hashes=deepcopy(self.input_hashes),
            array_hashes={name: _array_digest(getattr(self, name)) for name in POOL_ARRAYS},
            account_inputs=False, validation_labels_used=False, test_evaluated=False)

    def save(self, directory):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Refusing to overwrite rank-pool artifacts")
        directory.mkdir(parents=True, exist_ok=True)
        for name in POOL_ARRAYS:
            np.save(directory / (name + ".npy"), getattr(self, name), allow_pickle=False)
        manifest = self.manifest()
        manifest["identity_sha256"] = self.identity_sha256
        manifest["artifacts"] = {name + ".npy": file_hash(directory / (name + ".npy"))
                                 for name in POOL_ARRAYS}
        write_json(directory / "manifest.json", manifest)
        return manifest

    @classmethod
    def load(cls, directory, expected_manifest_sha256=None):
        directory = Path(directory)
        if (expected_manifest_sha256 is not None
                and file_hash(directory / "manifest.json") != expected_manifest_sha256):
            raise ValueError("Rank-pool manifest SHA256 changed")
        manifest = read_config(directory / "manifest.json")
        expected = {name + ".npy" for name in POOL_ARRAYS}
        if set(manifest.get("artifacts", {})) != expected:
            raise ValueError("Rank-pool artifact schema changed")
        arrays = {}
        for name in POOL_ARRAYS:
            path = directory / (name + ".npy")
            if file_hash(path) != manifest["artifacts"][path.name]:
                raise ValueError("Rank-pool artifact SHA256 changed")
            arrays[name] = np.load(path, allow_pickle=False)
        pool = cls(**arrays, codes=tuple(manifest["stock_codes"]),
            feature_cols=tuple(manifest["feature_cols"]), train_end=manifest["train_end"],
            horizon=manifest["horizon"], common_lookback=manifest["common_lookback"],
            input_hashes=manifest["input_hashes"])
        identity = pool.manifest()
        saved = {key: value for key, value in manifest.items() if key not in ("artifacts", "identity_sha256")}
        if identity != saved or pool.identity_sha256 != manifest.get("identity_sha256"):
            raise ValueError("Rank-pool logical identity changed")
        return pool

    def sample(self, rng, batch_size):
        if not isinstance(rng, np.random.Generator) or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Rank-pool sampling requires an independent RNG and positive batch")
        indices = rng.integers(self.size, size=batch_size)
        return indices, self.features[indices], self.targets[indices]


def build_rank_pool(features, prices, codes, feature_cols, train_end, *, common_lookback=5,
                    horizon=5, input_hashes=None):
    """Build labels only after physically removing all post-training rows."""
    codes, feature_cols = tuple(codes), tuple(feature_cols)
    if (not codes or len(codes) != len(set(codes)) or not feature_cols
            or len(feature_cols) != len(set(feature_cols))
            or type(horizon) is not int or horizon < 1
            or type(common_lookback) is not int or common_lookback < 1
            or not isinstance(features, pd.DataFrame) or not isinstance(prices, pd.DataFrame)
            or not isinstance(features.index, pd.DatetimeIndex) or features.index.tz is not None
            or features.index.has_duplicates or not features.index.is_monotonic_increasing
            or not features.index.equals(prices.index) or features.columns.has_duplicates
            or prices.columns.has_duplicates):
        raise ValueError("Invalid feature/price schema for train-only labels")
    boundary = pd.Timestamp(train_end)
    if pd.isna(boundary) or boundary.tz is not None or boundary >= PROTECTED_START:
        raise ValueError("Training boundary must be finite, timezone-naive and before the protected period")
    feature_columns = pd.MultiIndex.from_product([codes, feature_cols])
    price_columns = pd.MultiIndex.from_product([codes, ("open", "close")])
    if not set(feature_columns) <= set(features.columns) or not set(price_columns) <= set(prices.columns):
        raise ValueError("Rank-pool stocks/features/open-close prices are missing")
    # Truncation precedes hashing, support computation and all forward labels.
    panel = features.loc[features.index <= boundary, feature_columns].copy()
    quotes = prices.loc[prices.index <= boundary, price_columns].copy()
    values = panel.to_numpy(dtype=np.float64).reshape(len(panel), len(codes), len(feature_cols))
    if len(panel) <= horizon + common_lookback:
        raise ValueError("Insufficient training rows for rank labels")
    finite = np.isfinite(values).all(axis=2)
    support = pd.DataFrame(finite).rolling(common_lookback).sum().eq(common_lookback).to_numpy()
    current_close = quotes.loc[:, pd.IndexSlice[:, "close"]].to_numpy(dtype=np.float64)
    support &= np.isfinite(current_close) & (current_close > 0)
    labels, endpoints = forward_labels(quotes, codes, horizon)
    x, targets, returns, date_indices, stock_indices = [], [], [], [], []
    for index in range(len(panel) - horizon):
        raw = labels.iloc[index].to_numpy(dtype=np.float64)
        eligible = np.flatnonzero(support[index] & np.isfinite(raw))
        if len(eligible) < 2:
            continue
        ranked = pd.Series(raw[eligible]).rank(method="average").to_numpy()
        x.append(values[index, eligible])
        targets.append(2 * (ranked - 1) / (len(eligible) - 1) - 1)
        returns.append(raw[eligible])
        date_indices.append(np.full(len(eligible), index, dtype=np.int64))
        stock_indices.append(eligible)
    if not x:
        raise ValueError("No complete, past-boundary rank examples")
    hashes = dict(input_hashes or {})
    reserved = {"truncated_features": _frame_digest(panel), "truncated_open_close": _frame_digest(quotes)}
    if any(name in hashes and hashes[name] != value for name, value in reserved.items()):
        raise ValueError("Caller hashes contradict physical training inputs")
    hashes.update(reserved)
    positions = np.concatenate(date_indices)
    return TrainRankPool(features=np.concatenate(x), targets=np.concatenate(targets),
        raw_returns=np.concatenate(returns), date_indices=positions,
        stock_indices=np.concatenate(stock_indices), dates=panel.index.to_numpy(),
        target_endpoints=endpoints.iloc[positions].to_numpy(), codes=codes, feature_cols=feature_cols,
        train_end=str(boundary.date()), horizon=horizon, common_lookback=common_lookback,
        input_hashes=hashes)


class RankAuxPPO(PPO):
    """PPO with a stock-only auxiliary objective in the same clipped Adam update.

    The train loop is adapted from Stable-Baselines3 2.7.1 PPO (MIT license),
    https://github.com/DLR-RM/stable-baselines3/blob/v2.7.1/stable_baselines3/ppo/ppo.py.
    Its surrogate, GAE targets, joint action likelihood, KL guard and clipping
    are unchanged. Only the auxiliary term and explicit update audits are new.
    """

    def __init__(self, *args, aux_pool=None, aux_coef=0., aux_batch_size=1024, aux_seed=0,
                 aux_huber_delta=.1, **kwargs):
        if (isinstance(aux_coef, bool) or not np.isfinite(aux_coef) or aux_coef < 0
                or type(aux_batch_size) is not int or aux_batch_size < 1
                or type(aux_seed) is not int or aux_seed < 0
                or isinstance(aux_huber_delta, bool) or not np.isfinite(aux_huber_delta)
                or aux_huber_delta <= 0 or (aux_pool is not None and not isinstance(aux_pool, TrainRankPool))):
            raise ValueError("Invalid auxiliary PPO settings")
        self.aux_coef, self.aux_batch_size, self.aux_seed = float(aux_coef), aux_batch_size, aux_seed
        self.aux_huber_delta = float(aux_huber_delta)
        self.aux_pool = aux_pool
        self.aux_pool_identity = None if aux_pool is None else aux_pool.identity_sha256
        self._aux_rng = np.random.default_rng(aux_seed ^ AUX_SEED_SALT)
        self.aux_rng_state = deepcopy(self._aux_rng.bit_generator.state)
        self.aux_updates = self.aux_sample_exposures = self.actual_optimizer_updates = 0
        self.last_auxiliary_update = {}
        self.last_optimization_audit = {}
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["aux_pool", "_aux_rng"]

    def attach_aux_pool(self, pool):
        if not isinstance(pool, TrainRankPool):
            raise ValueError("Auxiliary training requires a verified TrainRankPool")
        if self.aux_pool_identity is not None and self.aux_pool_identity != pool.identity_sha256:
            raise ValueError("Attached training-label pool differs from the saved identity")
        self.aux_pool, self.aux_pool_identity = pool, pool.identity_sha256
        self._aux_rng = np.random.default_rng(self.aux_seed ^ AUX_SEED_SALT)
        self._aux_rng.bit_generator.state = deepcopy(self.aux_rng_state)

    def auxiliary_audit(self):
        return dict(aux_coef=self.aux_coef, batch_size=self.aux_batch_size, huber_delta=self.aux_huber_delta,
            seed=self.aux_seed, seed_salt=AUX_SEED_SALT, pool_identity_sha256=self.aux_pool_identity,
            actual_joint_optimizer_updates=self.actual_optimizer_updates, actual_aux_updates=self.aux_updates,
            supervised_stock_date_exposures=self.aux_sample_exposures,
            independent_rng_state=deepcopy(self.aux_rng_state), rng_sha256=_digest(self.aux_rng_state),
            last_auxiliary_update=deepcopy(self.last_auxiliary_update),
            last_optimization_audit=deepcopy(self.last_optimization_audit),
            extra_optimizer_steps=0, supervision_in_observations=False, auxiliary_rl_steps=0)

    def _auxiliary_loss(self):
        indices, features, targets = self.aux_pool.sample(self._aux_rng, self.aux_batch_size)
        self.aux_rng_state = deepcopy(self._aux_rng.bit_generator.state)
        network = self.policy.network
        prediction = network.signal_head(network.market_encoder(
            th.as_tensor(features, dtype=th.float32, device=self.device))).flatten()
        target = th.as_tensor(targets, dtype=th.float32, device=self.device)
        loss = F.huber_loss(prediction, target, delta=self.aux_huber_delta)
        parameters = tuple(network.market_encoder.parameters()) + tuple(network.signal_head.parameters())
        gradients = th.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        squared = sum((gradient.detach().square().sum() for gradient in gradients if gradient is not None),
                      th.zeros((), device=self.device))
        norm = float(th.sqrt(squared).cpu())
        audit = dict(loss=float(loss.detach().cpu()), raw_auxiliary_gradient_norm=norm,
            scaled_auxiliary_gradient_norm=self.aux_coef * norm, examples=len(indices),
            unique_examples_in_batch=int(len(np.unique(indices))),
            sampled_indices_sha256=_array_digest(indices),
            prediction_mean=float(prediction.detach().mean().cpu()),
            target_mean=float(target.detach().mean().cpu()))
        return loss, audit

    def _validate_auxiliary_training(self):
        if self.aux_coef:
            if self.aux_pool is None:
                raise ValueError("Auxiliary PPO cannot train without its train-only label pool")
            if self.aux_pool.identity_sha256 != self.aux_pool_identity:
                raise ValueError("Auxiliary label pool identity changed")
            network = getattr(self.policy, "network", None)
            if network is None or not hasattr(network, "market_encoder") or not hasattr(network, "signal_head"):
                raise ValueError("Auxiliary PPO requires stock-only market_encoder and signal_head")

    def learn(self, *args, **kwargs):
        self._validate_auxiliary_training()
        return super().learn(*args, **kwargs)

    def train(self):
        self._validate_auxiliary_training()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        auxiliary_losses, auxiliary_norms, total_norms, applied_kls = [], [], [], []
        continue_training = True
        updates_before = self.actual_optimizer_updates
        attempted_minibatches = 0
        stopped_kl = None
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                attempted_minibatches += 1
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    stopped_kl = float(approx_kl_div)
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                auxiliary = None
                if self.aux_coef:
                    aux_loss, auxiliary = self._auxiliary_loss()
                    loss = loss + self.aux_coef * aux_loss
                self.policy.optimizer.zero_grad()
                loss.backward()
                total_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.actual_optimizer_updates += 1
                applied_kls.append(float(approx_kl_div))
                total_norms.append(float(total_norm.detach().cpu()))
                if auxiliary is not None:
                    self.aux_updates += 1
                    self.aux_sample_exposures += auxiliary["examples"]
                    auxiliary.update(joint_optimizer_update=self.actual_optimizer_updates,
                                     joint_preclip_gradient_norm=total_norms[-1])
                    self.last_auxiliary_update = auxiliary
                    auxiliary_losses.append(auxiliary["loss"])
                    auxiliary_norms.append(auxiliary["raw_auxiliary_gradient_norm"])
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.last_optimization_audit = dict(timesteps=self.num_timesteps,
            applied_joint_updates=self.actual_optimizer_updates - updates_before,
            attempted_minibatches=attempted_minibatches, early_kl_stopped=not continue_training,
            rejected_approx_kl=stopped_kl, applied_mean_approx_kl=float(np.mean(applied_kls)) if applied_kls else None,
            preclip_gradient_norm_mean=float(np.mean(total_norms)) if total_norms else None)
        self.logger.record("train/actual_optimizer_updates", self.actual_optimizer_updates)
        self.logger.record("train/applied_mean_approx_kl", float(np.mean(applied_kls)) if applied_kls else 0.)
        self.logger.record("train/joint_preclip_gradient_norm", float(np.mean(total_norms)) if total_norms else 0.)
        self.logger.record("train/auxiliary_loss", float(np.mean(auxiliary_losses)) if auxiliary_losses else 0.)
        self.logger.record("train/auxiliary_gradient_norm", float(np.mean(auxiliary_norms)) if auxiliary_norms else 0.)
        self.logger.record("train/scaled_auxiliary_gradient_norm",
                           self.aux_coef * float(np.mean(auxiliary_norms)) if auxiliary_norms else 0.)
        self.logger.record("train/auxiliary_updates", self.aux_updates)
        self.logger.record("train/supervised_stock_date_exposures", self.aux_sample_exposures)
