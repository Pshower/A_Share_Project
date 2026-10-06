"""Synthetic verification of fold isolation, affine recovery and loader schema."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import FEATURES, file_hash, read_config, write_json
from src.training.exploration_data import derive_fold, fit_fold_scaler


def original_dataset(directory, future_shift=0.0, heldout_shift=0.0, old_mean=50.0, old_std=7.0):
    directory.mkdir()
    dates = pd.bdate_range("2022-01-03", "2026-07-08")
    codes = ["600000", "000001"]
    columns = pd.MultiIndex.from_product([codes, FEATURES], names=["code", "feature"])
    raw = pd.DataFrame(np.random.default_rng(42).normal(size=(len(dates), len(columns))), index=dates, columns=columns)
    raw.loc[dates > "2023-03-31"] += future_shift
    heldout_columns = raw.columns.get_level_values(0) == "000001"
    raw.loc[:, heldout_columns] += heldout_shift
    raw.iloc[1, 0] = np.nan
    panel = (raw - old_mean) / old_std
    prices = pd.DataFrame(index=dates, columns=pd.MultiIndex.from_product([codes, ["open", "close", "volume"]], names=["code", "feature"]), dtype=float)
    for offset, code in enumerate(codes):
        prices[(code, "open")] = 10 + offset + np.arange(len(dates)) * .01
        prices[(code, "close")] = prices[(code, "open")] * (1 + .002 * np.sin(np.arange(len(dates)) + offset))
        prices[(code, "volume")] = 1000000
    panel.to_parquet(directory / "panel_data.parquet")
    prices.to_parquet(directory / "prices.parquet")
    scaler = dict(schema_version=2, feature_cols=FEATURES, stats={f: dict(mean=old_mean, std=old_std) for f in FEATURES},
                  train_start="2022-01-03", train_end="2026-03-31", train_codes=codes, ddof=0)
    write_json(directory / "scaler_params.json", scaler)
    build = dict(version="synthetic", start_date="2022-01-03", end_date="2026-07-08", train_end="2026-03-31",
                 val_end="2026-06-30", volume_multiplier=100, raw_pattern="{code}.csv", train_codes=codes)
    manifest = dict(schema_version=2, version="synthetic", feature_cols=FEATURES, stock_codes=codes,
                    build_config=build, price_basis="hfq_research", train_end="2026-03-31", val_end="2026-06-30",
                    start_date="2022-01-03", end_date="2026-07-08", limitations=[],
                    artifacts={name: file_hash(directory / name) for name in ("panel_data.parquet", "prices.parquet", "scaler_params.json")})
    write_json(directory / "manifest.json", manifest)
    config = dict(dataset=dict(build, clean_dir=str(directory), stock_list_path="unused.txt"),
                  backtest=dict(initial_capital=1000000, broker={}, start_date=None, end_date=None))
    return raw, panel, prices, config


def test_fold_isolates_future_scaler_rows_and_physically_excludes_protected_period(tmp_path):
    _, _, _, config = original_dataset(tmp_path / "original")
    _, _, _, perturbed_config = original_dataset(tmp_path / "perturbed", future_shift=12345)
    original_hashes = {p.name: file_hash(p) for p in (tmp_path / "original").iterdir()}
    first = derive_fold(tmp_path / "original", config, tmp_path / "first", "2023")
    second = derive_fold(tmp_path / "perturbed", perturbed_config, tmp_path / "second", "2023")
    stats = read_config(first["scaler"])
    np.testing.assert_allclose([s["mean"] for s in stats["stats"].values()],
                               [s["mean"] for s in read_config(second["scaler"])["stats"].values()])
    assert pd.read_parquet(first["panel"]).index.max() < pd.Timestamp("2026-07-01")
    assert read_config(first["manifest"])["stock_codes"] == ["600000", "000001"]
    assert {p.name: file_hash(p) for p in (tmp_path / "original").iterdir()} == original_hashes
    assert read_config(first["research_config"])["dataset"]["train_end"] == "2023-03-31"
    assert stats["fit_rows"] == 2 * len(pd.bdate_range("2022-01-03", "2023-03-31")) - 1


def test_old_scaler_affine_parameters_do_not_change_refitted_features(tmp_path):
    _, _, _, config_a = original_dataset(tmp_path / "a", old_mean=50, old_std=7)
    _, _, _, config_b = original_dataset(tmp_path / "b", old_mean=-23, old_std=3)
    a = derive_fold(tmp_path / "a", config_a, tmp_path / "fold_a", "2023")
    b = derive_fold(tmp_path / "b", config_b, tmp_path / "fold_b", "2023")
    np.testing.assert_allclose(pd.read_parquet(a["panel"]), pd.read_parquet(b["panel"]), equal_nan=True, atol=1e-12)


def test_heldout_stocks_never_fit_scaler(tmp_path):
    _, _, _, config_a = original_dataset(tmp_path / "a")
    _, _, _, config_b = original_dataset(tmp_path / "b", heldout_shift=999)
    a = derive_fold(tmp_path / "a", config_a, tmp_path / "fold_a", "2023", training_codes=["600000"])
    b = derive_fold(tmp_path / "b", config_b, tmp_path / "fold_b", "2023", training_codes=["600000"])
    assert read_config(a["scaler"])["stats"] == read_config(b["scaler"])["stats"]
    assert read_config(a["scaler"])["train_codes"] == ["600000"]
    assert set(pd.read_parquet(a["panel"]).columns.get_level_values(0)) == {"600000", "000001"}


def test_fold_rejects_modified_artifacts_unknown_fold_bad_codes_and_overwrite(tmp_path):
    _, _, _, config = original_dataset(tmp_path / "original")
    with pytest.raises(ValueError, match="Registered"):
        derive_fold(tmp_path / "original", config, tmp_path / "x", "2026")
    with pytest.raises(ValueError, match="training_codes"):
        derive_fold(tmp_path / "original", config, tmp_path / "x", "2023", ["600000", "600000"])
    derive_fold(tmp_path / "original", config, tmp_path / "x", "2023")
    with pytest.raises(FileExistsError):
        derive_fold(tmp_path / "original", config, tmp_path / "x", "2023")
    scaler = read_config(tmp_path / "original" / "scaler_params.json")
    scaler["stats"][FEATURES[0]]["mean"] += 1
    write_json(tmp_path / "original" / "scaler_params.json", scaler)
    with pytest.raises(ValueError, match="hash"):
        derive_fold(tmp_path / "original", config, tmp_path / "y", "2023")


def test_no_valid_training_rows_or_invalid_scales_fail_closed(tmp_path):
    raw, panel, prices, _ = original_dataset(tmp_path / "original")
    raw[:] = np.nan
    with pytest.raises(ValueError, match="No complete"):
        fit_fold_scaler(raw, prices, FEATURES, ["600000"], "2023-03-31")


def test_derived_schema_is_accepted_by_research_data(tmp_path):
    _, _, _, research = original_dataset(tmp_path / "original")
    paths = derive_fold(tmp_path / "original", research, tmp_path / "fold", "2023")
    from src.training.env_factory import ResearchData
    config = dict(research_config=paths["research_config"], stock_codes=["600000", "000001"],
                  lookback=5, torch_threads=1, validation_rollouts=1, seed=42, logit_bound=8,
                  hidden_sizes=[8, 4], ppo=dict(n_steps=8, batch_size=4, n_epochs=1, learning_rate=1e-5,
                      clip_range=.2, target_kl=.03, max_grad_norm=.5, gamma=.99, gae_lambda=.95, ent_coef=0, vf_coef=.5))
    loaded = ResearchData(config)
    assert loaded.manifest["train_end"] == "2023-03-31"
    assert loaded.codes == ["600000", "000001"]
    assert loaded.features.index.max() < pd.Timestamp("2026-07-01")
