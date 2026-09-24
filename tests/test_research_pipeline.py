import json

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import performance
from src.backtest.run import run_suite
from src.backtest.strategies import Baseline
from src.data.preprocess import DataPreprocessor, FEATURES
from src.envs.trading import TradingEnv
from tests.test_trading_env import market


def raw_source(code, dates):
    close = 10 + np.arange(len(dates)) * 0.02 + np.sin(np.arange(len(dates)) / 5) * 0.1
    return pd.DataFrame({"date": dates, "code": code, "open": close, "close": close,
                         "high": close + 0.1, "low": close - 0.1, "volume": 100000,
                         "amount": close * 100000, "turnover": 1.0})


def fixture_config(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=140)
    for code in ["000001", "600000"]:
        raw_source(code, dates).to_csv(raw / f"{code}_data_fixture_hfq.csv", index=False)
    raw_source("000003", pd.bdate_range(dates[-1] + pd.Timedelta(days=10), periods=5)).to_csv(
        raw / "000003_data_fixture_hfq.csv", index=False)
    stock_list = tmp_path / "stocks.txt"
    stock_list.write_text("000001\n600000\n000003\n", encoding="utf-8-sig")
    return dict(dataset=dict(version="fixture", raw_dir=str(raw), clean_dir=str(tmp_path / "clean"),
                             stock_list_path=str(stock_list), start_date=str(dates[0].date()),
                             train_end=str(dates[89].date()), val_end=str(dates[119].date()),
                             end_date=str(dates[-1].date()), train_codes=["000001"], volume_multiplier=1),
                backtest=dict(split="val", initial_capital=100000, seed=42, lookback=1, rebalance_every=5,
                              momentum_top_k=1, broker=dict(commission=0.001, lot_size=1, max_weight=1),
                              benchmark_path=None, output_root=str(tmp_path / "runs")))


def test_training_scaler_ignores_future_and_heldout_stocks(tmp_path):
    config = fixture_config(tmp_path)
    prep = DataPreprocessor(**config["dataset"])
    before = prep.process_all_stocks_and_save_scaler()
    scaler = json.loads((prep.clean_dir / "scaler_params.json").read_text())
    path = prep.raw_dir / "000001_data_fixture_hfq.csv"
    raw = pd.read_csv(path, dtype={"code": str})
    raw.loc[90:, ["open", "close", "high", "low"]] *= 10
    raw.to_csv(path, index=False)
    path = prep.raw_dir / "600000_data_fixture_hfq.csv"
    raw = pd.read_csv(path, dtype={"code": str})
    raw.loc[:, ["open", "close", "high", "low"]] *= 2
    raw.to_csv(path, index=False)
    config["dataset"]["clean_dir"] = str(tmp_path / "clean2")
    other = DataPreprocessor(**config["dataset"])
    after = other.process_all_stocks_and_save_scaler()
    assert scaler == json.loads((other.clean_dir / "scaler_params.json").read_text())
    select = (before.code == "000001") & (before.date <= prep.train_end)
    pd.testing.assert_frame_equal(before.loc[select], after.loc[select])
    assert not {"open", "close", "code", "股票代码"}.intersection(scaler["feature_cols"])
    prices = pd.read_parquet(prep.clean_dir / "prices.parquet")
    assert prices[("000001", "open")].iloc[0] == 10
    valid = pd.read_parquet(prep.clean_dir / "valid_mask.parquet")
    assert not valid.iloc[:60].any().any()
    assert valid.iloc[-1].all()
    audit = pd.read_csv(prep.clean_dir / "source_audit.csv", dtype={"code": str})
    assert audit.loc[audit.code == "000003", "status"].iloc[0] == "no_rows_in_requested_dates"
    with pytest.raises(FileExistsError):
        prep.process_all_stocks_and_save_scaler()


def test_duplicates_and_missing_dates_are_not_silently_filled(tmp_path):
    config = fixture_config(tmp_path)
    prep = DataPreprocessor(**config["dataset"])
    raw, _ = prep._read_stock_data("000001")
    with pytest.raises(ValueError, match="duplicate"):
        prep.clean_single_stock(pd.concat([raw, raw.iloc[:1]]))
    raw = raw.drop(index=70)
    raw.to_csv(prep.raw_dir / "000001_data_fixture_hfq.csv", index=False)
    prep.process_all_stocks_and_save_scaler()
    panel = pd.read_parquet(prep.clean_dir / "panel_data.parquet")
    prices = pd.read_parquet(prep.clean_dir / "prices.parquet")
    assert np.isnan(prices[("000001", "open")].iloc[70])
    assert panel[("000001", "return")].iloc[70:72].isna().all()
    with pytest.raises(ValueError, match="retain missing"):
        prep.align_to_panel(pd.DataFrame(), fill_method="bfill")


def test_no_order_advance_really_holds_share_counts():
    env = TradingEnv(market(), initial_capital=10000)
    env.reset()
    env.step([0.5, 0.5])
    before = env.portfolio.positions.copy()
    _, _, _, _, info = env.advance()
    np.testing.assert_array_equal(info["positions"], before)
    assert info["orders"] == []


def test_missing_close_cannot_erase_known_opening_quote(tmp_path):
    config = fixture_config(tmp_path)
    prep = DataPreprocessor(**config["dataset"])
    raw, _ = prep._read_stock_data("000001")
    opening = raw.loc[95, "open"]
    raw.loc[95, "close"] = np.nan
    clean = prep.clean_single_stock(raw)
    assert clean.loc[95, "open"] == opening
    assert not clean.loc[95, "valid_price"]
    features = prep.add_features(clean.set_index("date"))
    assert features.iloc[95].isna().all()


def test_baseline_selection_and_hold_schedule():
    obs = dict(valid_mask=np.array([1, 1, 0], dtype=np.int8), buy_mask=np.ones(3, dtype=np.int8),
               stock_features=np.array([[[0.1]], [[0.3]], [[100]]]))
    strategy = Baseline("momentum", ["momentum_20"], top_k=1, rebalance_every=5)
    np.testing.assert_array_equal(strategy.act(obs, 0), [0, 1, 0])
    assert strategy.act(obs, 1) is None
    hold = Baseline("buy_hold", [])
    np.testing.assert_array_equal(hold.act(obs, 0), [0.5, 0.5, 0])
    assert hold.act(obs, 5) is None


def test_metrics_include_initial_drawdown_and_zero_volatility():
    daily = pd.DataFrame(dict(equity=[100, 90, 99], fee=[0, 1, 2], slippage_cost=[0, 0, 0], turnover=[0, 1, 2]))
    metrics = performance(daily)
    assert metrics["cumulative_return"] == pytest.approx(-0.01)
    assert metrics["max_drawdown"] == pytest.approx(-0.1)
    assert metrics["total_fees"] == 3
    assert metrics["gross_turnover"] == 3
    daily.equity = 100
    assert performance(daily)["sharpe"] is None
    daily.equity = [100, 90, 0]
    assert performance(daily)["max_drawdown"] == -1
    assert performance(daily)["cumulative_return"] == -1


def test_end_to_end_suite_is_reproducible_and_validation_only(tmp_path):
    config = fixture_config(tmp_path)
    first, metrics = run_suite(config, "first")
    second, repeated = run_suite(config, "second")
    assert metrics == repeated
    assert metrics["cash"]["cumulative_return"] == 0
    for name in metrics:
        pd.testing.assert_frame_equal(pd.read_csv(first / name / "daily.csv"), pd.read_csv(second / name / "daily.csv"))
    assert (first / "curves.png").stat().st_size > 1000
    daily = pd.read_csv(first / "cash/daily.csv", parse_dates=["date"])
    assert daily.date.max() == pd.Timestamp(config["dataset"]["val_end"])
    config["backtest"]["split"] = "test"
    with pytest.raises(ValueError, match="only runs validation"):
        run_suite(config, "test")


def test_existing_local_index_is_used_without_network(tmp_path):
    config = fixture_config(tmp_path)
    dates = pd.bdate_range(config["dataset"]["train_end"], config["dataset"]["val_end"])
    index_path = tmp_path / "index.csv"
    pd.DataFrame({"date": dates, "close": np.linspace(100, 110, len(dates))}).to_csv(index_path, index=False)
    config["backtest"]["benchmark_path"] = str(index_path)
    output, metrics = run_suite(config, "with_index")
    assert metrics["index_benchmark"]["cumulative_return"] == pytest.approx(0.1)
    assert metrics["index_benchmark"]["mean_cash_ratio"] == 0
    assert (output / "index_benchmark.csv").exists()


def test_transform_reuses_saved_scaler_and_is_invertible(tmp_path):
    config = fixture_config(tmp_path)
    prep = DataPreprocessor(**config["dataset"])
    combined = prep.process_all_stocks_and_save_scaler()
    raw, _ = prep._read_stock_data("000001")
    transformed = prep.transform_new_data(raw, prep.clean_dir / "scaler_params.json", code="000001")
    expected = combined.loc[combined.code == "000001", ["date", *FEATURES, "code"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(transformed, expected)
    scaler = json.loads((prep.clean_dir / "scaler_params.json").read_text())
    restored = transformed[FEATURES].copy()
    for col in FEATURES:
        restored[col] = restored[col] * scaler["stats"][col]["std"] + scaler["stats"][col]["mean"]
    original = prep.add_features(prep.clean_single_stock(raw).set_index("date")).reset_index(drop=True)
    np.testing.assert_allclose(restored, original, atol=1e-12, equal_nan=True)
