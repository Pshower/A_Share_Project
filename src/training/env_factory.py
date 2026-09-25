"""Verified local artifacts and date-isolated PPO environments; never downloads."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.action_mapping import LatentActionWrapper, MAPPING_VERSION
from src.data.preprocess import file_hash, read_config, resolve_path
from src.envs.broker import BrokerConfig
from src.envs.market import MarketDataProvider
from src.envs.trading import TradingEnv


def validate_config(config):
    for field in ["lookback", "torch_threads", "validation_rollouts"]:
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(config["logit_bound"]) or config["logit_bound"] <= 0:
        raise ValueError("logit_bound must be positive and finite")
    ppo = config["ppo"]
    for field in ["n_steps", "batch_size", "n_epochs"]:
        if type(ppo[field]) is not int or ppo[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if ppo["n_steps"] < 2 or ppo["batch_size"] < 2 or ppo["n_steps"] % ppo["batch_size"]:
        raise ValueError("batch_size must divide n_steps; both must be at least two")
    for field in ["learning_rate", "clip_range", "target_kl", "max_grad_norm"]:
        if not np.isfinite(ppo[field]) or ppo[field] <= 0:
            raise ValueError(f"{field} must be positive and finite")
    for field in ["gamma", "gae_lambda"]:
        if not 0 <= ppo[field] <= 1:
            raise ValueError(f"{field} must be in [0, 1]")
    for field in ["ent_coef", "vf_coef"]:
        if not np.isfinite(ppo[field]) or ppo[field] < 0:
            raise ValueError(f"{field} must be nonnegative and finite")
    return config


class ResearchData:
    def __init__(self, config):
        self.config = deepcopy(validate_config(config))
        self.research = read_config(resolve_path(config["research_config"]))
        dataset = self.research["dataset"]
        directory = resolve_path(dataset["clean_dir"])
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest = self.manifest
        if manifest.get("schema_version") != 2:
            raise ValueError("Unsupported dataset schema")
        for key in ["train_end", "val_end"]:
            if manifest[key] != dataset[key]:
                raise ValueError(f"Dataset boundary mismatch: {key}")
        for key in ["version", "start_date", "end_date", "train_end", "val_end", "volume_multiplier", "raw_pattern"]:
            expected = dataset.get(key, "{code}_data_*_hfq.csv" if key == "raw_pattern" else None)
            if manifest["build_config"][key] != expected:
                raise ValueError(f"Dataset config mismatch: {key}")
        train_codes = dataset.get("train_codes")
        if train_codes is None:
            train_codes = [s.strip() for s in resolve_path(dataset["stock_list_path"]).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        if manifest["build_config"]["train_codes"] != train_codes:
            raise ValueError("Dataset training stock pool mismatch")
        hashes = {}
        for name in ["panel_data.parquet", "prices.parquet", "scaler_params.json"]:
            hashes[name] = file_hash(directory / name)
            if hashes[name] != manifest["artifacts"][name]:
                raise ValueError(f"Dataset artifact was modified: {name}")
        scaler = json.loads((directory / "scaler_params.json").read_text(encoding="utf-8"))
        if (scaler.get("schema_version") != 2 or scaler["feature_cols"] != manifest["feature_cols"]
                or scaler["train_end"] != manifest["train_end"] or scaler["train_codes"] != train_codes):
            raise ValueError("Scaler feature order or training boundary mismatch")
        self.features = pd.read_parquet(directory / "panel_data.parquet")
        self.prices = pd.read_parquet(directory / "prices.parquet")
        codes = config.get("stock_codes")
        codes = self.features.columns.get_level_values(0).unique().tolist() if codes is None else codes
        if (not isinstance(codes, list) or not codes or len(set(codes)) != len(codes)
                or any(not isinstance(c, str) or c not in manifest["stock_codes"] for c in codes)):
            raise ValueError("stock_codes must be a nonempty unique list from the dataset")
        self.codes = codes
        expected_cols = pd.MultiIndex.from_product([manifest["stock_codes"], manifest["feature_cols"]])
        if self.features.columns.has_duplicates or set(self.features.columns) != set(expected_cols):
            raise ValueError("Feature panel fields do not match manifest")
        # Parquet/pivot order can differ from the manifest's feature whitelist.
        ordered = pd.MultiIndex.from_product([self.features.columns.get_level_values(0).unique(), manifest["feature_cols"]])
        self.features = self.features.reindex(columns=ordered)
        if not self.features.index.equals(self.prices.index):
            raise ValueError("Feature and price dates differ")
        self.settings = deepcopy(self.research["backtest"])
        self.settings.update(seed=config["seed"], lookback=config["lookback"])
        self.contract = dict(schema_version=1, dataset_version=manifest["version"],
                             manifest_sha256=file_hash(directory / "manifest.json"), artifacts=hashes,
                             feature_cols=manifest["feature_cols"], stock_codes=list(codes),
                             lookback=config["lookback"], price_basis=manifest["price_basis"],
                             broker=asdict(BrokerConfig(**self.settings["broker"])),
                             initial_capital=self.settings["initial_capital"],
                             mapping_version=MAPPING_VERSION, logit_bound=config["logit_bound"],
                             hidden_sizes=config["hidden_sizes"])

    def market(self, split, *, allow_test=False):
        if split not in {"train", "val", "test"} or (split == "test" and not allow_test):
            raise ValueError("Only train and val are available; test evaluation is disabled")
        end = pd.Timestamp(self.manifest["build_config"]["end_date"] if split == "test"
                           else self.manifest["train_end" if split == "train" else "val_end"])
        # Slice before provider construction: its internal arrays cannot expose later rows.
        features = self.features.loc[:end, pd.IndexSlice[self.codes, :]].copy()
        prices = self.prices.loc[:end, pd.IndexSlice[self.codes, :]].copy()
        if split in {"val", "test"}:
            boundary = "train_end" if split == "val" else "val_end"
            start = features.index[features.index <= pd.Timestamp(self.manifest[boundary])][-1]
        else:
            n = len(self.codes)
            complete = np.isfinite(features.to_numpy().reshape(len(features), n, -1)).all(axis=2)
            windows = pd.DataFrame(complete).rolling(self.config["lookback"]).sum().eq(self.config["lookback"])
            valid_days = np.flatnonzero(windows.any(axis=1).to_numpy())
            if not len(valid_days) or valid_days[0] >= len(features) - 1:
                raise ValueError("No train episode after feature warmup")
            start = features.index[valid_days[0]]
        return MarketDataProvider(features, self.manifest["feature_cols"], price_path=prices,
                                  price_basis=self.manifest["price_basis"], start_date=start,
                                  end_date=end, lookback=self.config["lookback"])

    def env(self, split):
        return LatentActionWrapper(TradingEnv(self.market(split), self.settings["initial_capital"],
                                   BrokerConfig(**self.settings["broker"])), self.config["logit_bound"])

    def vector_env(self, split="train"):
        env = DummyVecEnv([lambda: Monitor(self.env(split))])
        env.seed(self.config["seed"])
        return env
