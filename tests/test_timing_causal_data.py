"""Synthetic causal transformations and construction spies; no real env steps."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import FEATURES
from tools import timing_causal_data as causal
from tools.counterfactual_exact_schedule import ExactPhaseSampler


def synthetic(fold="2023"):
    dates = pd.bdate_range("2017-01-02", str(int(fold) + 1) + "-06-30", name="date")
    codes = ["000001", "600000"]
    columns = pd.MultiIndex.from_product([codes, FEATURES], names=["code", "feature"])
    t = np.arange(len(dates), dtype=float)[:, None, None]
    stock = np.arange(2, dtype=float)[None, :, None]
    feature = np.arange(20, dtype=float)[None, None, :]
    raw = np.sin(t / (13 + feature)) * .03 + stock * .01 + feature * .001 + t * .000001
    stats = {name: dict(mean=float(-.3 + i * .01), std=float(.5 + i * .02)) for i, name in enumerate(FEATURES)}
    mean = np.array([stats[name]["mean"] for name in FEATURES])
    scale = np.array([stats[name]["std"] for name in FEATURES])
    normalized = (raw - mean) / scale
    normalized[:5] = np.nan
    normalized[100, 1, 0] = np.nan
    features = pd.DataFrame(normalized.reshape(len(dates), -1), index=dates, columns=columns)
    price_columns = pd.MultiIndex.from_product([codes, ["open", "close", "volume"]], names=["code", "feature"])
    price_values = np.tile([10., 10.1, 1_000_000., 20., 20.2, 2_000_000.], (len(dates), 1))
    prices = pd.DataFrame(price_values, index=dates, columns=price_columns)
    scaler = dict(schema_version=2, feature_cols=list(FEATURES), stats=stats)
    return features, prices, scaler, codes


def derive(fold="2023"):
    features, prices, scaler, codes = synthetic(fold)
    return causal.derive_inner_transform(features, prices, scaler, FEATURES, codes, fold)


@pytest.mark.parametrize("fold,expected", [("2023", "2021-12-31"), ("2024", "2022-12-30"),
                                          ("2025", "2023-12-29"), ("2026", "2024-12-31")])
def test_inner_cutoff_and_exact_twenty_session_embargo(fold, expected):
    features, prices, scaler, boundaries = derive(fold)
    dates = features.index
    assert boundaries["inner_train_end"] == expected
    cut = dates.get_loc(pd.Timestamp(expected))
    embargo = dates[cut + 1:cut + 21]
    assert boundaries["embargo_dates"] == [str(date.date()) for date in embargo]
    assert boundaries["inner_holdout_start"] == str(dates[cut + 21].date())
    assert boundaries["inner_train_last_h20_decision"] == str(dates[cut - 20].date())
    assert dates.max() <= pd.Timestamp(fold + "-06-30")
    assert scaler["train_end"] == expected
    assert scaler["ddof"] == 0
    assert boundaries["scaler_policy"].startswith("fit inner train once")
    assert len(dates[(dates >= boundaries["inner_holdout_start"]) &
                     (dates <= boundaries["inner_holdout_end"])]) > 250


def test_inner_scaler_cannot_see_holdout_or_outer_q2_values():
    features, prices, source_scaler, codes = synthetic()
    original = causal.derive_inner_transform(features, prices, source_scaler, FEATURES, codes, "2023")
    changed = features.copy()
    changed.loc[changed.index > "2021-12-31"] += 10_000
    second = causal.derive_inner_transform(changed, prices, source_scaler, FEATURES, codes, "2023")
    assert original[2] == second[2]
    pd.testing.assert_frame_equal(original[0].loc[:"2021-12-31"], second[0].loc[:"2021-12-31"])
    assert causal.frame_value_sha256(original[0]) != causal.frame_value_sha256(second[0])


def test_2026_transform_excludes_july_and_later_before_inverse_fit_or_identity():
    features, prices, source_scaler, codes = synthetic("2026")
    first = causal.derive_inner_transform(features, prices, source_scaler, FEATURES, codes, "2026")
    changed = features.copy()
    changed.loc[changed.index >= "2026-07-01"] = 1e100
    changed_prices = prices.copy()
    changed_prices.loc[changed_prices.index >= "2026-07-01"] = 1e100
    second = causal.derive_inner_transform(changed, changed_prices, source_scaler, FEATURES, codes, "2026")
    assert first[0].index.max() == pd.Timestamp("2026-06-30")
    assert first[1].index.max() == pd.Timestamp("2026-06-30")
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[2:] == second[2:]
    assert causal.frame_value_sha256(first[0]) == causal.frame_value_sha256(second[0])


def test_2026_provider_internal_arrays_never_include_july_or_later():
    features, prices, source_scaler, codes = synthetic("2026")
    transformed, derived_prices, _, boundaries = causal.derive_inner_transform(
        features, prices, source_scaler, FEATURES, codes, "2026")
    data = object.__new__(causal.CausalTimingData)
    data.features, data.prices, data.codes = transformed, derived_prices, codes
    data._boundaries = boundaries
    data.manifest = dict(feature_cols=list(FEATURES), price_basis="hfq_research")
    data.config, data.common_lookback = dict(lookback=1), 5
    for split, end in (("train", "2026-03-31"), ("val", "2026-06-30")):
        market = data.market(split)
        assert market.panel.index.max() == pd.Timestamp(end)
        assert market.dates.max() == pd.Timestamp(end)
        assert len(market._features) == len(market._valuation) == len(market.panel)
        assert all(len(value) == len(market.panel) for value in market._fields.values())
        assert market.panel.index.max() < causal.PROTECTED_START


def test_scaler_fits_only_complete_rows_with_positive_current_close():
    features, prices, source_scaler, codes = synthetic()
    prices.loc[features.index[150], (codes[0], "close")] = np.nan
    transformed, _, scaler, boundaries = causal.derive_inner_transform(features, prices, source_scaler, FEATURES, codes, "2023")
    raw = causal.inverse_affine_features(features.loc[:boundaries["inner_train_end"]], source_scaler, FEATURES)
    rows = []
    for code in codes:
        values = raw[code].reindex(columns=FEATURES).to_numpy()
        close = prices.loc[raw.index, (code, "close")].to_numpy()
        rows.append(values[np.isfinite(values).all(axis=1) & np.isfinite(close) & (close > 0)])
    rows = np.concatenate(rows)
    assert scaler["fit_rows"] == len(rows)
    for i, feature in enumerate(FEATURES):
        assert scaler["stats"][feature]["mean"] == pytest.approx(rows[:, i].mean(), abs=1e-15)
        assert scaler["stats"][feature]["std"] == pytest.approx(rows[:, i].std(ddof=0), abs=1e-15)
    assert np.array_equal(np.isfinite(features.loc[transformed.index].to_numpy()), np.isfinite(transformed.to_numpy()))


def test_transform_does_not_mutate_inputs_or_fill_missing_features():
    features, prices, scaler, codes = synthetic()
    before_f, before_p, before_s = features.copy(), prices.copy(), deepcopy(scaler)
    transformed, derived_prices, _, _ = causal.derive_inner_transform(features, prices, scaler, FEATURES, codes, "2023")
    pd.testing.assert_frame_equal(features, before_f)
    pd.testing.assert_frame_equal(prices, before_p)
    assert scaler == before_s
    assert transformed.iloc[0].isna().all()
    pd.testing.assert_frame_equal(derived_prices, prices.loc[transformed.index])


def test_m20_rank_guard_uses_common_masks_and_actual_float32_scores():
    features, prices, source_scaler, codes = synthetic()
    transformed, derived_prices, _, _ = causal.derive_inner_transform(features, prices, source_scaler, FEATURES, codes, "2023")
    result = causal.verify_m20_ranking(features, transformed, derived_prices, codes, FEATURES)
    assert result["all_valid_stock_orders_equal"] is True
    assert result["checked_dates"] == len(transformed) - 9
    changed = transformed.copy()
    changed.loc[changed.index[200], (codes[0], "momentum_20")] = 1_000
    with pytest.raises(ValueError, match="ranking"):
        causal.verify_m20_ranking(features, changed, derived_prices, codes, FEATURES)


@pytest.mark.parametrize("key,value", [("clip", 5), ("clip_range", [-4, 4]), ("clip_value", 1)])
def test_clipped_source_is_rejected(key, value):
    features, prices, scaler, codes = synthetic()
    scaler[key] = value
    with pytest.raises(ValueError, match="clipped"):
        causal.derive_inner_transform(features, prices, scaler, FEATURES, codes, "2023")


@pytest.mark.parametrize("field,value", [("mean", np.nan), ("mean", np.inf), ("std", 0), ("std", -1)])
def test_invalid_source_statistics_are_rejected(field, value):
    features, prices, scaler, codes = synthetic()
    scaler["stats"][FEATURES[0]][field] = value
    with pytest.raises(ValueError, match="statistics"):
        causal.derive_inner_transform(features, prices, scaler, FEATURES, codes, "2023")


def test_value_identity_binds_values_order_dates_and_exact_mixed_integer_storage():
    index = pd.date_range("2020-01-01", periods=2, name="date")
    cols = pd.MultiIndex.from_tuples([("000001", "x"), ("000001", "y")], names=["code", "feature"])
    frame = pd.DataFrame({cols[0]: np.array([2**53, 2**53 + 1], dtype=np.uint64), cols[1]: [1., 2.]}, index=index)
    frame.columns = cols
    sha = causal.frame_value_sha256(frame)
    assert sha == causal.frame_value_sha256(frame.copy())
    changed = frame.copy()
    changed.iloc[1, 0] = np.uint64(2**53 + 2)
    assert sha != causal.frame_value_sha256(changed)
    assert sha != causal.frame_value_sha256(frame.iloc[:, ::-1])
    changed = frame.copy()
    changed.index = changed.index + pd.Timedelta(days=1)
    assert sha != causal.frame_value_sha256(changed)


@pytest.fixture
def data(tmp_path, monkeypatch):
    features, prices, scaler, codes = synthetic()
    source = tmp_path / "source"
    source.mkdir()
    (source / "scaler_params.json").write_text(json.dumps(scaler), encoding="utf-8")
    config = dict(timing=dict(fold="2023", reward_basis="relative_log", calendar_margin=2., period=20),
                  training_episode_steps=240, lookback=1, logit_bound=8., max_trade_weight=.05,
                  max_commission_ratio=.001, initial_portfolio="mixed", seed=42)

    def fake_init(self, received):
        self.config = deepcopy(received)
        self.research = dict(dataset=dict(clean_dir=str(source)))
        self.manifest = dict(train_end="2023-03-31", val_end="2023-06-30", feature_cols=list(FEATURES), price_basis="hfq_research")
        self.codes, self.features, self.prices = codes, features.copy(), prices.copy()
        self.common_lookback = 5
        self.settings = dict(initial_capital=1_000_000, broker={})
        self.contract = dict(feature_cols=list(FEATURES), stock_codes=codes, price_basis="hfq_research",
            manifest_sha256="a" * 64, artifacts=dict(panel_data="b" * 64, prices="c" * 64),
            data_source="sina", training_episode_steps=240, broker=dict(lot_size=100), logit_bound=8.)
        self.stock_contract = deepcopy(self.contract)

    monkeypatch.setattr(causal.TimingData, "__init__", fake_init)
    monkeypatch.setattr(causal, "ROOT", tmp_path)
    return causal.CausalTimingData(config)


def test_factory_uses_one_frozen_inner_transform_for_all_four_views(data):
    before = deepcopy(data.scaler)
    for split, expected_end in (("inner_train", "2021-12-31"), ("inner_holdout", "2023-03-31"),
                                ("train", "2023-03-31"), ("val", "2023-06-30")):
        market = data._market_view(split)
        assert market.panel.index.max() <= pd.Timestamp(expected_end)
        assert len(market._features) == len(market.panel)
        assert market.dates.max() <= pd.Timestamp(expected_end)
        assert market.price_basis == "hfq_research"
        assert data.scaler == before
    assert data.contract["causal_data"]["scaler_refitted_at_outer_train_end"] is False
    assert data.contract["source_artifacts"] == dict(panel_data="b" * 64, prices="c" * 64)
    assert data.contract["artifacts"]["panel_data_value_sha256"] == data.derived_feature_sha256


def test_validation_keeps_past_rows_and_prior_volume_without_exposing_future(data):
    market = data.market("val")
    assert market.panel.index.min() < market.dates.min()
    assert str(market.dates.min().date()) == "2023-03-31"
    j = market._indices[0]
    assert j >= 5
    np.testing.assert_array_equal(market.execution(0)["volume"], market._fields["volume"][j - 1])
    hold = data._market_view("inner_holdout")
    assert str(hold.dates.min().date()) == data.boundaries["inner_holdout_start"]
    assert hold.panel.index.min() < hold.dates.min()
    assert hold.panel.index.max() <= pd.Timestamp(data.boundaries["inner_holdout_end"])


@pytest.mark.parametrize("split,allow", [("test", False), ("inner_holdout", False), ("train", True), ("val", True)])
def test_outer_factory_rejects_unregistered_views(data, split, allow):
    with pytest.raises(ValueError):
        data.market(split, allow_test=allow)


def test_teacher_construction_uses_80_steps_and_exact_sampler_only(data, monkeypatch):
    calls = []

    def base(market, capital, broker, **kwargs):
        calls.append((market, capital, broker, kwargs))
        return SimpleNamespace(market=market, kwargs=kwargs)

    def stock(env, *args, **kwargs):
        return SimpleNamespace(env=env)

    def timing(env, contract, **kwargs):
        return SimpleNamespace(env=env, contract=contract, kwargs=kwargs, sampler="original")

    monkeypatch.setattr(causal, "TradingEnv", base)
    monkeypatch.setattr(causal, "TradeActionWrapper", stock)
    monkeypatch.setattr(causal, "TimingEnv", timing)
    for split in ("inner_train", "inner_holdout"):
        env = data.teacher_env(split)
        assert isinstance(env.sampler, ExactPhaseSampler)
        assert env.kwargs["training"] is True
        assert env.kwargs["episode_steps"] == 80
        assert env.contract["teacher_source"]["split"] == split
        assert env.contract["training_episode_steps"] == 80
    assert all(item[3]["max_steps"] == 80 and item[3]["random_start"] is False for item in calls)
    assert data.config["training_episode_steps"] == data.contract["training_episode_steps"] == 240
    with pytest.raises(ValueError):
        data.teacher_env("val")


def test_export_properties_are_copies_and_save_refuses_overwrite_or_frozen_parent(data, tmp_path):
    first = data.export_payloads()
    first["scaler"]["stats"][FEATURES[0]]["mean"] = 100
    first["identity"]["boundaries"]["horizon"] = 1
    first["features"].iloc[100, 0] = 999
    assert data.scaler["stats"][FEATURES[0]]["mean"] != 100
    assert data.identity["boundaries"]["horizon"] == 20
    assert data.features.iloc[100, 0] != 999
    output = tmp_path / "new-study" / "data"
    hashes = data.save_artifacts(output)
    assert set(hashes) == {"panel_data.parquet", "prices.parquet", "scaler_params.json", "identity.json", "boundaries.json"}
    assert json.loads((output / "scaler_params.json").read_text())["train_end"] == "2021-12-31"
    assert causal.frame_value_sha256(pd.read_parquet(output / "panel_data.parquet")) == data.derived_feature_sha256
    for destination in (output, tmp_path / "source" / "child", tmp_path.parent / "outside",
                        tmp_path / "reports/runs/ppo_timing_phase_state_repair_20261015/new"):
        with pytest.raises(ValueError):
            data.save_artifacts(destination)


def test_export_refuses_a_mutated_derived_frame(data, tmp_path):
    data.features.iloc[100, 0] += .1
    with pytest.raises(ValueError, match="changed"):
        data.export_payloads()
    with pytest.raises(ValueError, match="changed"):
        data.save_artifacts(tmp_path / "must-not-write")
    assert not (tmp_path / "must-not-write").exists()


@pytest.mark.parametrize("fold,steps", [("2027", 240), ("2028", 240), ("2023", 80), ("2023", 240.0)])
def test_constructor_rejects_unregistered_fold_or_changed_ppo_length(fold, steps):
    with pytest.raises(ValueError):
        causal.CausalTimingData(dict(timing=dict(fold=fold), training_episode_steps=steps))
