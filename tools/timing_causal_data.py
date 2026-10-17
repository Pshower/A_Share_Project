"""Inner-fitted, immutable-scaler data views for the new paired/PPO study.

No download, label collection, model fit or environment advancement occurs on
construction. Only the newly requested directory is written by save_artifacts.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.trade_mapping import TradeActionWrapper
from src.data.preprocess import ROOT, file_hash, read_config, resolve_path
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv
from src.training.exploration_data import inverse_affine_features, fit_fold_scaler
from src.training.exploration_env import ExplorationMarket
from src.training.research_timing import TimingData
from src.training.timing_env import TimingEnv, build_timing_contract
from tools.counterfactual_exact_schedule import ExactPhaseSampler


DERIVATION_VERSION = "inner_affine_fixed_through_outer_q2_v1"
TEACHER_EPISODE_STEPS = 80
HORIZON = 20
EMBARGO_SESSIONS = 20
ALLOWED_FOLDS = ("2023", "2024", "2025", "2026")
PROTECTED_START = pd.Timestamp("2026-07-01")


def json_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frame_value_sha256(frame):
    """Bind dates, ordered coordinates, dtypes and exact numeric storage values."""
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
            or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or not isinstance(frame.columns, pd.MultiIndex) or frame.columns.has_duplicates):
        raise ValueError("Value identity needs ordered datetime rows and unique panel coordinates")
    digest = hashlib.sha256()
    header = dict(index_name=frame.index.name, index_timezone=str(frame.index.tz),
                  columns=[list(pair) for pair in frame.columns],
                  column_names=list(frame.columns.names), dtypes=[str(d) for d in frame.dtypes])
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.asarray(frame.index.asi8, dtype="<i8").tobytes())
    dtypes = frame.dtypes
    if len(dtypes) and all(dtype == dtypes.iloc[0] for dtype in dtypes):
        array = frame.to_numpy(copy=False)
        if array.dtype.hasobject:
            raise ValueError("Object panel values are unsupported")
        digest.update(np.ascontiguousarray(array).tobytes())
    else:
        for column in frame.columns:
            array = frame[column].to_numpy(copy=False)
            if array.dtype.hasobject:
                raise ValueError("Object panel values are unsupported")
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def causal_boundaries(dates, fold):
    fold = str(fold)
    if fold not in ALLOWED_FOLDS:
        raise ValueError("Only the four registered historical folds are supported")
    dates = pd.DatetimeIndex(dates)
    if (dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or dates.tz is not None):
        raise ValueError("Causal boundaries require unique ordered naive dates")
    inner_requested = pd.Timestamp(f"{int(fold) - 2}-12-31")
    outer_requested = pd.Timestamp(f"{fold}-03-31")
    validation_requested = pd.Timestamp(f"{fold}-06-30")
    if validation_requested >= PROTECTED_START:
        raise ValueError("Outer validation cannot enter the protected period")
    inner_rows = dates[dates <= inner_requested]
    outer_rows = dates[dates <= outer_requested]
    validation_rows = dates[dates <= validation_requested]
    if len(inner_rows) < HORIZON + TEACHER_EPISODE_STEPS + 3 or not len(outer_rows):
        raise ValueError("Insufficient inner training history")
    inner_end, outer_end = inner_rows[-1], outer_rows[-1]
    after_inner = dates[(dates > inner_end) & (dates <= outer_end)]
    if len(after_inner) < EMBARGO_SESSIONS + TEACHER_EPISODE_STEPS + 3:
        raise ValueError("Inner holdout needs a full embargo and three feasible 80-step starts")
    embargo = after_inner[:EMBARGO_SESSIONS]
    holdout_start = after_inner[EMBARGO_SESSIONS]
    if not len(validation_rows) or validation_rows[-1] <= outer_end:
        raise ValueError("An outer Q2 evaluation interval is required")
    return dict(fold=fold, horizon=HORIZON, embargo_sessions=EMBARGO_SESSIONS,
        inner_train_requested_end=str(inner_requested.date()), inner_train_end=str(inner_end.date()),
        inner_train_last_h20_decision=str(inner_rows[-HORIZON - 1].date()),
        embargo_dates=[str(date.date()) for date in embargo],
        inner_holdout_start=str(holdout_start.date()), inner_holdout_end=str(outer_end.date()),
        inner_holdout_last_h20_decision=str(outer_rows[-HORIZON - 1].date()),
        outer_train_requested_end=str(outer_requested.date()), outer_train_end=str(outer_end.date()),
        outer_validation_start=str(outer_end.date()),
        outer_validation_first_reward_date=str(dates[dates > outer_end][0].date()),
        outer_validation_end=str(validation_rows[-1].date()),
        teacher_episode_steps=TEACHER_EPISODE_STEPS,
        scaler_policy="fit inner train once; unchanged for inner holdout, outer PPO and Q2")


def derive_inner_transform(features, prices, source_scaler, feature_cols, codes, fold):
    """Pure affine recovery/refit; all statistics use inner-train valid rows only."""
    for key in ("clip", "clip_range", "clip_value"):
        if source_scaler.get(key) is not None:
            raise ValueError("A clipped source transform is not invertible")
    if (source_scaler.get("schema_version") != 2
            or source_scaler.get("feature_cols") != list(feature_cols)):
        raise ValueError("Source affine scaler schema/order differs")
    expected = pd.MultiIndex.from_product([codes, feature_cols], names=features.columns.names)
    if (features.columns.has_duplicates or set(features.columns) != set(expected)
            or not features.index.equals(prices.index)):
        raise ValueError("Feature/price rows or feature coordinates differ")
    end = pd.Timestamp(f"{fold}-06-30")
    features = features.loc[:end].reindex(columns=expected).copy()
    prices = prices.loc[features.index].copy()
    boundaries = causal_boundaries(features.index, fold)
    for name in feature_cols:
        stat = source_scaler["stats"][name]
        if not np.isfinite([stat["mean"], stat["std"]]).all() or stat["std"] <= 0:
            raise ValueError("Source affine statistics must be finite with positive scale")
    raw = inverse_affine_features(features, source_scaler, feature_cols)
    stats, rows = fit_fold_scaler(raw, prices, feature_cols, codes, boundaries["inner_train_end"])
    transformed = raw.copy()
    for name, stat in stats.items():
        columns = transformed.columns.get_level_values(1) == name
        transformed.loc[:, columns] = (raw.loc[:, columns] - stat["mean"]) / stat["std"]
    if not np.array_equal(np.isfinite(features.to_numpy()), np.isfinite(transformed.to_numpy())):
        raise ValueError("Inner affine transform changed feature validity")
    scaler = dict(schema_version=2, feature_cols=list(feature_cols), stats=stats,
                  train_codes=list(codes), train_start=str(features.index.min().date()),
                  train_end=boundaries["inner_train_end"], fit_rows=rows, ddof=0,
                  derivation_version=DERIVATION_VERSION, clipping=None)
    return transformed, prices, scaler, boundaries


def verify_m20_ranking(source, transformed, prices, codes, feature_cols, common_lookback=5):
    """Fail rather than silently change the fixed rule through float32 tie rounding."""
    columns = pd.MultiIndex.from_product([codes, feature_cols], names=transformed.columns.names)
    source = source.loc[transformed.index].reindex(columns=columns)
    shape = (len(source), len(codes), len(feature_cols))
    original, derived = source.to_numpy().reshape(shape), transformed.to_numpy().reshape(shape)
    finite = np.isfinite(original).all(axis=2)
    valid = pd.DataFrame(finite).rolling(common_lookback).sum().eq(common_lookback).to_numpy()
    closing = prices.loc[source.index, [(code, "close") for code in codes]].ffill().to_numpy()
    valid &= np.isfinite(closing)
    feature_index = list(feature_cols).index("momentum_20")
    with np.errstate(over="ignore"):
        old_scores = original[:, :, feature_index].astype(np.float32)
        new_scores = derived[:, :, feature_index].astype(np.float32)
    checked = 0
    for index, date in enumerate(source.index):
        eligible = np.flatnonzero(valid[index])
        if not len(eligible):
            continue
        if (not np.isfinite(old_scores[index, eligible]).all()
                or not np.isfinite(new_scores[index, eligible]).all()):
            raise ValueError("M20 ranking inputs overflow float32")
        old_order = sorted(eligible, key=lambda i: (-float(old_scores[index, i]), codes[i]))
        new_order = sorted(eligible, key=lambda i: (-float(new_scores[index, i]), codes[i]))
        if old_order != new_order:
            raise ValueError("Inner scaling changed the fixed M20 float32 ranking on " + str(date.date()))
        checked += 1
    return dict(all_valid_stock_orders_equal=True, checked_dates=checked,
                score_precision="float32; stock-code ascending exact ties")


class CausalTimingData(TimingData):
    """Unchanged execution engines with an explicitly inner-fitted feature view."""

    def __init__(self, config):
        fold = str(config.get("timing", {}).get("fold"))
        if (fold not in ALLOWED_FOLDS or type(config.get("training_episode_steps")) is not int
                or config["training_episode_steps"] != 240):
            raise ValueError("Causal study supports four frozen folds and retains 240-step PPO episodes")
        super().__init__(config)
        if (self.manifest["train_end"] != fold + "-03-31"
                or self.manifest["val_end"] != fold + "-06-30"
                or self.contract.get("data_source") != "sina"):
            raise ValueError("Source fold boundaries or Sina provenance differ")
        self._source_directory = resolve_path(self.research["dataset"]["clean_dir"]).resolve()
        source_scaler = read_config(self._source_directory / "scaler_params.json")
        features, prices, scaler, boundaries = derive_inner_transform(
            self.features, self.prices, source_scaler, self.manifest["feature_cols"], self.codes, fold)
        ranking = verify_m20_ranking(self.features, features, prices, self.codes,
                                     self.manifest["feature_cols"], self.common_lookback)
        self.features, self.prices = features, prices
        self._inner_scaler, self._boundaries = scaler, boundaries
        identity = dict(version=DERIVATION_VERSION, fold=fold,
            source_manifest_sha256=self.contract["manifest_sha256"],
            source_artifacts=deepcopy(self.contract["artifacts"]),
            source_scaler_file_sha256=file_hash(self._source_directory / "scaler_params.json"),
            scaler_value_sha256=json_digest(scaler),
            derived_feature_value_sha256=frame_value_sha256(features),
            derived_prices_value_sha256=frame_value_sha256(prices),
            boundaries=deepcopy(boundaries), price_basis=self.manifest["price_basis"],
            common_lookback=self.common_lookback,
            m20_ranking=ranking,
            recovery="inverse unclipped source affine; floating point roundoff remains",
            scaler_refitted_at_outer_train_end=False, execution_mechanism_changed=False)
        self._causal_identity = identity
        stock = deepcopy(self.stock_contract)
        stock["source_artifacts"] = deepcopy(stock["artifacts"])
        stock["artifacts"] = dict(panel_data_value_sha256=identity["derived_feature_value_sha256"],
            prices_value_sha256=identity["derived_prices_value_sha256"],
            scaler_params_value_sha256=identity["scaler_value_sha256"])
        stock["causal_data"] = deepcopy(identity)
        self.stock_contract = stock
        self.contract = build_timing_contract(stock, config["timing"]["reward_basis"], calendar_margin=2.)
        self.contract["timing_research"] = deepcopy(config["timing"])

    @property
    def identity(self):
        return deepcopy(self._causal_identity)

    @property
    def boundaries(self):
        return deepcopy(self._boundaries)

    @property
    def scaler(self):
        return deepcopy(self._inner_scaler)

    @property
    def derived_feature_sha256(self):
        return self._causal_identity["derived_feature_value_sha256"]

    def _market_view(self, split):
        boundary = self._boundaries
        ends = dict(train=boundary["outer_train_end"], val=boundary["outer_validation_end"],
                    inner_train=boundary["inner_train_end"], inner_holdout=boundary["inner_holdout_end"])
        if split not in ends:
            raise ValueError("Only registered inner/outer train and holdout/val views are available")
        end = pd.Timestamp(ends[split])
        columns = pd.MultiIndex.from_product([self.codes, self.manifest["feature_cols"]],
                                            names=self.features.columns.names)
        features = self.features.loc[:end].reindex(columns=columns).copy()
        prices = self.prices.loc[features.index, pd.IndexSlice[self.codes, :]].copy()
        if split == "val":
            start = pd.Timestamp(boundary["outer_validation_start"])
        elif split == "inner_holdout":
            start = pd.Timestamp(boundary["inner_holdout_start"])
        else:
            finite = np.isfinite(features.to_numpy().reshape(len(features), len(self.codes), -1)).all(axis=2)
            eligible = pd.DataFrame(finite).rolling(self.common_lookback).sum().eq(self.common_lookback).any(axis=1)
            indices = np.flatnonzero(eligible.to_numpy())
            if not len(indices):
                raise ValueError("No common-window feature support")
            start = features.index[indices[0]]
        return ExplorationMarket(features, self.manifest["feature_cols"], price_path=prices,
            price_basis=self.manifest["price_basis"], start_date=start, end_date=end,
            lookback=self.config["lookback"], common_lookback=self.common_lookback)

    def market(self, split, *, allow_test=False):
        if allow_test or split not in {"train", "val"}:
            raise ValueError("Outer factory opens only train/val; reserved periods remain closed")
        return self._market_view(split)

    def teacher_env(self, split):
        if split not in {"inner_train", "inner_holdout"}:
            raise ValueError("Teacher source opens only the two inner chronological views")
        market = self._market_view(split)
        market.sampler = None
        base = TradingEnv(market, self.settings["initial_capital"], BrokerConfig(**self.settings["broker"]),
            initial_portfolio=self.config.get("initial_portfolio", "mixed"),
            max_steps=TEACHER_EPISODE_STEPS, random_start=False, recent_episode_every=0)
        stock = TradeActionWrapper(base, self.config["logit_bound"], self.config["max_trade_weight"],
                                   self.config["max_commission_ratio"], top_n=20, top_n_choices=None)
        contract = deepcopy(self.contract)
        contract["training_episode_steps"] = TEACHER_EPISODE_STEPS
        contract["teacher_source"] = dict(split=split, episode_steps=TEACHER_EPISODE_STEPS,
                                         sampler="tools.counterfactual_exact_schedule.ExactPhaseSampler")
        env = TimingEnv(stock, contract, training=True, reward_basis=self.config["timing"]["reward_basis"],
                        episode_steps=TEACHER_EPISODE_STEPS)
        env.sampler = ExactPhaseSampler(market.dates, TEACHER_EPISODE_STEPS)
        return env

    def export_payloads(self):
        self._check_value_identity()
        return dict(features=self.features.copy(deep=True), prices=self.prices.copy(deep=True),
                    scaler=self.scaler, identity=self.identity, boundaries=self.boundaries)

    def _check_value_identity(self):
        if (frame_value_sha256(self.features) != self.derived_feature_sha256
                or frame_value_sha256(self.prices) != self._causal_identity["derived_prices_value_sha256"]
                or json_digest(self._inner_scaler) != self._causal_identity["scaler_value_sha256"]):
            raise ValueError("Derived data or the fixed inner scaler changed after construction")

    def save_artifacts(self, directory):
        """Explicit new-directory export; never called implicitly by the factory."""
        self._check_value_identity()
        directory = Path(directory).resolve()
        root = ROOT.resolve()
        frozen = [self._source_directory,
                  root / "reports/runs/ppo_timing_phase_state_repair_20261015",
                  root / "reports/runs/ppo_counterfactual_pilot_20261016"]
        if (not directory.is_relative_to(root) or directory.exists()
                or any(directory.is_relative_to(path) or path.is_relative_to(directory) for path in frozen)):
            raise ValueError("Causal artifacts require a new workspace directory outside immutable inputs")
        directory.mkdir(parents=True, exist_ok=False)
        self.features.to_parquet(directory / "panel_data.parquet")
        self.prices.to_parquet(directory / "prices.parquet")
        for name, value in (("scaler_params.json", self.scaler), ("identity.json", self.identity),
                            ("boundaries.json", self.boundaries)):
            (directory / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
        return {path.name: file_hash(path) for path in sorted(directory.iterdir())}
