"""Verified local artifacts and date-isolated PPO environments; never downloads."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.action_mapping import LatentActionWrapper, MAPPING_VERSION, mapping_version
from src.agents.delta_mapping import DeltaActionWrapper, DELTA_MAPPING_VERSION, EXECUTION_FIELDS, ALLOCATION_FIELDS
from src.agents.trade_mapping import (TradeActionWrapper, TRADE_MAPPING_VERSION, TRADE_ALLOCATION_FIELDS,
                                      SHARE_FIELDS, LOT_BOUND_FIELDS, DIRECTION_NAMES)
from src.data.preprocess import file_hash, read_config, resolve_path
from src.envs.broker import BrokerConfig
from src.envs.market import MarketDataProvider
from src.envs.trading import TradingEnv


def validate_config(config):
    mode = config.get("action_mode", "target")
    if mode not in {"target", "stock_delta", "stock_trade"}:
        raise ValueError("Unknown action_mode")
    if mode == "stock_delta":
        if config.get("prior_feature_index") is not None or config.get("action_top_k") is not None:
            raise ValueError("Delta actor cannot use a legacy prior or static action_top_k")
        for key, default in [("max_delta_weight", .05), ("actor_output_gain", .01)]:
            if not np.isfinite(config.get(key, default)) or not 0 < config.get(key, default) <= 1:
                raise ValueError("Invalid delta actor parameter: " + key)
        choices = config.get("top_n_choices", [])
        if not isinstance(choices, list) or any(type(k) is not int for k in choices) or len(set(choices)) != len(choices):
            raise ValueError("top_n_choices must be a unique integer list")
        if any(type(k) is not int or not 0 <= k <= 500 for k in [config.get("top_n", 20), *choices]):
            raise ValueError("top_n must be in [0, 500]")
    elif mode == "stock_trade":
        if config.get("prior_feature_index") is not None or config.get("action_top_k") is not None:
            raise ValueError("Trade actor cannot use a legacy prior or static action_top_k")
        for key, default in [("max_trade_weight", .05), ("max_commission_ratio", .001), ("actor_output_gain", .01)]:
            value = config.get(key, default)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Invalid trade actor parameter: " + key)
        choices = config.get("top_n_choices", [])
        if (not isinstance(choices, list) or any(type(k) is not int for k in choices)
                or len(set(choices)) != len(choices)
                or any(type(k) is not int or not 0 <= k <= 500 for k in [config.get("top_n", 20), *choices])):
            raise ValueError("Invalid unique trade top_n conditions")
        if config["ppo"].get("use_sde", False):
            raise ValueError("Trade actor does not support SDE")
    top_k = config.get("action_top_k")
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 500):
        raise ValueError("action_top_k must be in [1, 500]")
    if not np.isfinite(config.get("initial_log_std", 0.0)) or not -5 <= config.get("initial_log_std", 0.0) <= 1:
        raise ValueError("initial_log_std must be in [-5, 1]")
    for key, default in [("prior_strength", 4.0), ("residual_scale", 0.25)]:
        if not np.isfinite(config.get(key, default)) or not 0 < config.get(key, default) <= 8:
            raise ValueError("Invalid prior parameter: " + key)
    recent = config.get("recent_episode_every", 0)
    if type(recent) is not int or recent < 0 or (recent and not config.get("training_episode_steps")):
        raise ValueError("Recent episode sampling requires a bounded training episode")
    from src.envs.initial_account import MODES
    if config.get("initial_portfolio", "cash") not in (*MODES, "mixed"):
        raise ValueError("Unknown initial portfolio mode")
    episode = config.get("training_episode_steps")
    if episode is not None and (type(episode) is not int or episode < 2):
        raise ValueError("training_episode_steps must be at least two")
    if config.get("learning_rate_schedule", "constant") not in {"constant", "linear"}:
        raise ValueError("Unknown learning-rate schedule")
    factor = config.get("learning_rate_end_factor", 0.1)
    if not np.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError("learning_rate_end_factor must be in [0, 1]")
    selection = config.get("selection_policy", {})
    if selection:
        if type(selection.get("min_steps")) is not int or selection["min_steps"] < 2:
            raise ValueError("Selection min_steps must be at least two")
        for field, maximum in [("max_drawdown", 1), ("max_turnover", 10)]:
            value = selection.get(field)
            if value is not None and (not np.isfinite(value) or not 0 < value <= maximum):
                raise ValueError("Invalid selection constraint: " + field)
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
        from src.data.preprocess import FEATURES
        from src.data.feature_definitions import validate_definitions, DEFINITION_VERSION
        definitions = validate_definitions(dataset.get("extra_features", []), FEATURES)
        if (manifest.get("extra_features", []) != definitions or scaler.get("extra_features", []) != definitions
                or manifest["feature_cols"] != FEATURES + [d["name"] for d in definitions]
                or (definitions and (manifest.get("feature_definition_version") != DEFINITION_VERSION
                                     or scaler.get("feature_definition_version") != DEFINITION_VERSION))):
            raise ValueError("Feature definitions do not match dataset/scaler contract")
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
        if (config.get("action_mode") == "stock_trade"
                and config.get("max_commission_ratio", .001) < BrokerConfig(**self.settings["broker"]).commission):
            raise ValueError("max_commission_ratio must be at least broker commission")
        self.training_coverage = None
        if config.get("training_date_policy") == "intersection":
            from src.data.coverage import training_coverage
            self.training_coverage = training_coverage(self.features, self.prices, self.codes,
                                                       config["lookback"], manifest["train_end"])
        self.contract = dict(schema_version=1, dataset_version=manifest["version"],
                             manifest_sha256=file_hash(directory / "manifest.json"), artifacts=hashes,
                             feature_cols=manifest["feature_cols"], stock_codes=list(codes),
                             lookback=config["lookback"], price_basis=manifest["price_basis"],
                             broker=asdict(BrokerConfig(**self.settings["broker"])),
                             initial_capital=self.settings["initial_capital"],
                             mapping_version=mapping_version(top_k=config.get("action_top_k")), logit_bound=config["logit_bound"],
                             hidden_sizes=config["hidden_sizes"])
        if self.training_coverage is not None:
            self.contract["training_date_policy"] = "intersection"
            self.contract["train_interval"] = {k: self.training_coverage[k] for k in ["train_start", "train_end"]}
        if config.get("initial_portfolio", "cash") != "cash":
            self.contract["initial_portfolio"] = config["initial_portfolio"]
        if config.get("training_episode_steps") is not None:
            self.contract["training_episode_steps"] = config["training_episode_steps"]
        if config.get("recent_episode_every"):
            self.contract["recent_episode_every"] = config["recent_episode_every"]
        if config.get("action_top_k") is not None:
            self.contract["action_top_k"] = config["action_top_k"]
        if config.get("action_mode") == "stock_delta":
            self.contract.update(action_mode="stock_delta", mapping_version=DELTA_MAPPING_VERSION,
                                 delta=dict(max_delta_weight=config.get("max_delta_weight", .05),
                                            actor_output_gain=config.get("actor_output_gain", .01),
                                            top_n=config.get("top_n", 20), top_n_choices=config.get("top_n_choices", []),
                                            execution_fields=list(EXECUTION_FIELDS), allocation_fields=list(ALLOCATION_FIELDS),
                                            top_n_semantics="grandfather_existing_actual_exit_releases_slot",
                                            cash_semantics="residual_after_stock_trades_and_receivables"))
        elif config.get("action_mode") == "stock_trade":
            self.contract.update(action_mode="stock_trade", mapping_version=TRADE_MAPPING_VERSION,
                trade=dict(max_trade_weight=config.get("max_trade_weight", .05),
                    max_commission_ratio=config.get("max_commission_ratio", .001),
                    actor_output_gain=config.get("actor_output_gain", .01), top_n=config.get("top_n", 20),
                    top_n_choices=config.get("top_n_choices", []), execution_fields=list(EXECUTION_FIELDS),
                    allocation_fields=list(TRADE_ALLOCATION_FIELDS), share_fields=list(SHARE_FIELDS),
                    lot_bound_fields=list(LOT_BOUND_FIELDS), direction_names=list(DIRECTION_NAMES),
                    action_layout="N_direction_codes_then_N_conditional_size_latents",
                    quantity_units="integer_lots_buy_reduce_full_book_shares_exit",
                    share_state_units="shares_divided_by_lot_size_including_odd_fraction",
                    minimum_trade_value="max(one_lot_value,min_commission/max_commission_ratio)",
                    conditional_size="BUY_always_REDUCE_varying_bounds_HOLD_EXIT_zero_atom",
                    reduce_semantics="whole_lots_preserve_at_least_one_share",
                    exit_semantics="known_full_available_actual_full_exit_only_waiver_and_slot_release",
                    top_n_semantics="grandfather_existing_actual_exit_releases_slot",
                    cash_semantics="residual_after_stock_trades_and_receivables"))
        if config.get("prior_feature_index") is not None:
            idx = config["prior_feature_index"]
            if type(idx) is not int or not 0 <= idx < len(manifest["feature_cols"]):
                raise ValueError("Invalid prior feature index")
            self.contract["prior"] = dict(feature_index=idx, strength=config.get("prior_strength", 4.0),
                                           residual_scale=config.get("residual_scale", 0.25))
        if "data_source" in self.manifest:
            if self.research.get("online_source", {}).get("provider") != self.manifest["data_source"]:
                raise ValueError("Dataset source differs from research configuration")
            self.contract["data_source"] = self.manifest["data_source"]

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
        elif self.training_coverage is not None:
            if not self.training_coverage["eligible"]:
                raise ValueError(self.training_coverage["reason"])
            start = pd.Timestamp(self.training_coverage["train_start"])
            end = pd.Timestamp(self.training_coverage["train_end"])
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
        env = TradingEnv(self.market(split), self.settings["initial_capital"],
                                   BrokerConfig(**self.settings["broker"]),
                                   initial_portfolio=self.config.get("initial_portfolio", "cash") if split == "train" else "cash",
                                   max_steps=self.config.get("training_episode_steps") if split == "train" else None,
                                   random_start=split == "train" and self.config.get("training_episode_steps") is not None,
                                   recent_episode_every=self.config.get("recent_episode_every", 0) if split == "train" else 0)
        if self.config.get("action_mode") == "stock_delta":
            return DeltaActionWrapper(env, self.config["logit_bound"], self.config.get("max_delta_weight", .05),
                                      self.settings.get("delta_top_n", self.config.get("top_n", 20)),
                                      self.config.get("top_n_choices") if split == "train" else None)
        if self.config.get("action_mode") == "stock_trade":
            return TradeActionWrapper(env, self.config["logit_bound"], self.config.get("max_trade_weight", .05),
                                      self.config.get("max_commission_ratio", .001),
                                      self.settings.get("trade_top_n", self.config.get("top_n", 20)),
                                      self.config.get("top_n_choices") if split == "train" else None)
        return LatentActionWrapper(env, self.config["logit_bound"], self.config.get("action_top_k"))

    def vector_env(self, split="train"):
        env = DummyVecEnv([lambda: Monitor(self.env(split))])
        env.seed(self.config["seed"])
        return env
