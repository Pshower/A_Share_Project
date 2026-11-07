"""Unit checks for the pre-open advisory, using in-memory panels only."""

import json

import numpy as np
import pandas as pd
import pytest

from tools import daily_advice as advice


def panel():
    """Four names: steady up, steady down, flat, and one that stops trading."""
    dates = pd.bdate_range("2024-01-01", periods=80)
    up = np.linspace(10.0, 10.0 * 1.01 ** 79, 80)
    down = np.linspace(10.0, 10.0 * 0.995 ** 79, 80)
    flat = np.full(80, 10.0)
    closes = pd.DataFrame({"000001": up, "600000": down, "000002": flat}, index=dates)
    closes["300750"] = np.nan
    closes.iloc[:60, closes.columns.get_loc("300750")] = 10.0
    volumes = pd.DataFrame(1000.0, index=dates, columns=closes.columns)
    return dict(source="panel", label="synthetic", codes=list(closes.columns), closes=closes,
                volumes=volumes, data=None, price_basis="hfq")


def test_momentum_scores_are_trailing_returns_over_the_stated_window():
    closes = panel()["closes"]
    scores = advice.momentum_scores(closes)
    as_of = closes.index[40]
    expected = closes.loc[as_of, "000001"] / closes.loc[closes.index[20], "000001"] - 1
    assert scores.loc[as_of, "000001"] == pytest.approx(expected)
    assert np.isnan(scores.iloc[5, 0])


def test_trend_frame_reports_returns_averages_and_drawdown_without_future_bars():
    closes = panel()["closes"]
    as_of = closes.index[79]
    trend = advice.trend_frame(closes, as_of, panel()["volumes"])
    assert trend.loc["000001", "r20"] > 0 and trend.loc["600000", "r20"] < 0
    assert trend.loc["000001", "ma20_dev"] > 0 and trend.loc["600000", "ma20_dev"] < 0
    # A name that stopped trading keeps its last stored bar and is not treated as rising.
    assert np.isnan(trend.loc["300750", "r20"])
    assert trend.loc["000001", "dd60"] == pytest.approx(0.0, abs=1e-12)
    assert trend.loc["600000", "dd60"] < 0


def test_momentum_engine_picks_the_strongest_tradeable_names_and_caps_their_weight():
    closes = panel()["closes"]
    as_of = closes.index[79]
    weights, scores, tradeable = advice.select_weights("momentum20", closes, as_of, top_k=2, max_weight=1.0)
    assert weights["000001"] > 0 and weights["000002"] > 0
    assert weights["600000"] == 0
    assert weights["300750"] == 0 and not tradeable["300750"]
    assert weights["000001"] + weights["000002"] == pytest.approx(1.0)


def test_equal_weight_covers_every_tradeable_name_and_cash_stays_flat():
    closes = panel()["closes"]
    as_of = closes.index[79]
    equal, _, _ = advice.select_weights("equal_weight", closes, as_of, max_weight=1.0)
    assert set(equal[equal > 0].index) == {"000001", "000002", "600000"}
    assert equal.sum() == pytest.approx(1.0)
    cash, _, _ = advice.select_weights("cash", closes, as_of)
    assert cash.sum() == 0


def test_select_weights_rejects_an_engine_it_cannot_simulate():
    with pytest.raises(ValueError, match="not a rule engine"):
        advice.select_weights("ppo", panel()["closes"], panel()["closes"].index[-1])


def test_classify_covers_buy_add_hold_trim_and_exit():
    assert advice.classify(0.05, 0.0) == "BUY"
    assert advice.classify(0.10, 0.05) == "ADD"
    assert advice.classify(0.05, 0.05) == "HOLD"
    assert advice.classify(0.02, 0.05) == "TRIM"
    assert advice.classify(0.0, 0.05) == "SELL"


def test_next_open_uses_the_first_strictly_later_trading_day():
    calendar = pd.Series(pd.bdate_range("2024-03-01", periods=5))
    assert advice.next_open(pd.Timestamp("2024-03-01"), calendar) == "2024-03-04"
    assert advice.next_open(pd.Timestamp("2024-03-07"), calendar) is None


def test_build_advice_ranks_names_and_reports_actions(monkeypatch):
    monkeypatch.setattr(advice, "_names", lambda root, codes: {"000001": "示例"})
    monkeypatch.setattr(advice, "_calendar_dates", lambda root: None)
    stage = panel()
    as_of = stage["closes"].index[79]
    result = advice.build_advice(stage, as_of, "momentum20", top_k=1, max_weight=1.0)
    summary = result["summary"]
    assert summary["as_of"] == str(as_of.date()) and summary["next_open"] is None
    assert summary["validated"] is False and summary["actionable"] is False
    assert summary["counts"]["BUY"] == 1
    assert summary["counts"]["HOLD"] == 2
    assert summary["counts"]["UNAVAILABLE"] == 1
    best = result["rows"][0]
    assert best["code"] == "000001" and best["action"] == "BUY" and best["name"] == "示例"
    assert best["momentum_rank"] == 1 and "动量排名 #1" in best["reason"]
    unavailable = next(row for row in result["rows"] if row["code"] == "300750")
    assert unavailable["action"] == "UNAVAILABLE" and unavailable["tradeable"] is False


def test_build_advice_turns_a_held_name_into_a_sell_when_it_leaves_the_target(monkeypatch):
    monkeypatch.setattr(advice, "_names", lambda root, codes: {})
    monkeypatch.setattr(advice, "_calendar_dates", lambda root: None)
    stage = panel()
    as_of = stage["closes"].index[79]
    result = advice.build_advice(stage, as_of, "momentum20", top_k=1, max_weight=1.0,
                                 holdings={"600000": 1.0})
    actions = {row["code"]: row["action"] for row in result["rows"]}
    assert actions["600000"] == "SELL" and actions["000001"] == "BUY"


def test_build_advice_rejects_holdings_outside_the_pool_and_unknown_dates(monkeypatch):
    monkeypatch.setattr(advice, "_names", lambda root, codes: {})
    monkeypatch.setattr(advice, "_calendar_dates", lambda root: None)
    stage = panel()
    with pytest.raises(ValueError, match="outside the advisory pool"):
        advice.build_advice(stage, stage["closes"].index[79], "cash", holdings={"999999": 0.1})
    with pytest.raises(ValueError, match="not a stored trading date"):
        advice.build_advice(stage, pd.Timestamp("2030-01-01"), "cash")


def test_markdown_reports_the_gate_state_and_never_claims_execution(monkeypatch):
    monkeypatch.setattr(advice, "_names", lambda root, codes: {})
    monkeypatch.setattr(advice, "_calendar_dates", lambda root: None)
    stage = panel()
    as_of = stage["closes"].index[79]
    unvalidated = advice.build_advice(stage, as_of, "momentum20", top_k=1, max_weight=1.0)
    text = advice.render_markdown(unvalidated)
    assert "未通过或缺失" in text and "不构成已验证策略" in text
    evidence = dict(folds={"2023": dict(cumulative_return=0.05, max_drawdown=-0.1, sharpe=1.0)},
                    gate={"rule": "两折累计收益均为正", "pass": True}, path="x.json")
    validated = advice.build_advice(stage, as_of, "momentum20", top_k=1, max_weight=1.0,
                                    evidence=evidence)
    text = advice.render_markdown(validated)
    assert "已通过" in text and "不构成已验证策略" not in text
    assert "不是保证成交" not in text  # the Chinese limitation replaces the English one


def backtest_frames():
    """The smallest daily/holdings/orders triple run_strategy can hand back."""
    dates = pd.bdate_range("2024-04-01", periods=5)
    daily = pd.DataFrame(dict(
        date=dates, equity=[1e6, 1.01e6, 1.02e6, 1.015e6, 1.03e6], cash=[0.0] * 5,
        daily_return=[0.0, 0.01, 0.0099, -0.0049, 0.0148], fee=[0.0, 10.0, 0.0, 0.0, 5.0],
        slippage_cost=[0.0] * 5, turnover=[0.0, 0.5, 0.0, 0.0, 0.2], holding_count=[2] * 5,
        unfilled_orders=[0] * 5))
    daily["net_value"] = daily.equity / daily.equity.iloc[0]
    daily["drawdown"] = daily.equity / daily.equity.cummax() - 1
    holdings = pd.DataFrame({"000001": [500.0] * 5, "600000": [300.0] * 5}, index=dates)
    orders = pd.DataFrame([dict(date=dates[1], code="000001", side="buy", filled_shares=100.0, status="filled")])
    return daily, holdings, orders


def metrics_table(sharpe=1.0):
    return {name: dict(cumulative_return=0.03, annual_volatility=0.12, sharpe=sharpe, sortino=1.2,
                       max_drawdown=-0.02, total_fees=15.0, mean_cash_ratio=0.1)
            for name in ("momentum", "equal_weight")}


def test_write_report_emits_the_layout_the_workbench_reads(tmp_path):
    daily, holdings, orders = backtest_frames()
    directory = advice.write_report(tmp_path / "fold_2023",
                                     {"momentum": (daily, holdings, orders),
                                      "equal_weight": (daily, holdings, orders)},
                                     metrics_table(), title="动量与等权",
                                     notes=["- 门槛：两折均为正 → 通过"],
                                     metadata=dict(split="val", fold="2023"))
    saved = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert list(saved) == ["momentum", "equal_weight"]
    for name in saved:
        frame = pd.read_csv(directory / name / "daily.csv")
        assert list(frame["net_value"]) == pytest.approx(list(daily["net_value"]))
        assert (directory / name / "orders.csv").exists()
        assert pd.read_parquet(directory / name / "positions.parquet").shape == holdings.shape
    assert json.loads((directory / "experiment.json").read_text(encoding="utf-8"))["fold"] == "2023"
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "两折均为正" in report and "| momentum |" in report


def test_write_report_stays_idempotent_and_marks_an_undefined_sharpe(tmp_path):
    daily, holdings, orders = backtest_frames()
    arms = {"momentum": (daily, holdings, orders), "equal_weight": (daily, holdings, orders)}
    target = tmp_path / "fold_2023"
    advice.write_report(target, arms, metrics_table(), title="第一版")
    advice.write_report(target, arms, metrics_table(sharpe=None), title="第二版")
    report = (target / "report.md").read_text(encoding="utf-8")
    assert "第二版" in report and "第一版" not in report
    assert "| — |" in report  # an undefined Sharpe is a dash, never a fabricated number


def test_lot_plan_buys_whole_lots_and_flags_targets_below_one_lot():
    prices = pd.Series({"000001": 30.0, "600000": 1000.0, "000002": np.nan})
    weights = pd.Series({"000001": 0.05, "600000": 0.0034, "000002": 0.02, "300750": 0.0})
    plan = advice.lot_plan(weights, prices, 1_000_000)
    # 5% of 1M is 50,000; one lot of a 30 yuan stock is 3,000; floor gives 16 lots.
    assert plan["000001"]["lots"] == 16 and plan["000001"]["status"] == "ok"
    assert plan["000001"]["executable_value"] == pytest.approx(48_000)
    assert plan["000001"]["executable_weight"] == pytest.approx(0.048)
    # 3,400 yuan cannot buy one 100,000 yuan lot: the position cannot be opened at all.
    assert plan["600000"]["lots"] == 0 and plan["600000"]["status"] == "below_one_lot"
    assert plan["000002"]["status"] == "no_price" and plan["000002"]["lots"] == 0
    assert plan["300750"]["status"] == "idle"


def test_lot_plan_requires_a_positive_capital():
    with pytest.raises(ValueError, match="positive number"):
        advice.lot_plan(pd.Series({"000001": 0.05}), pd.Series({"000001": 30.0}), 0)


def observation(names=12, eligible=5):
    valid = np.zeros(names, dtype=bool)
    valid[:eligible] = True
    return dict(valid_mask=valid, buy_mask=valid.copy())


def test_random_control_is_seeded_deterministic_and_holds_exactly_top_k():
    strategy = advice.RandomTopK(3, top_k=4, max_weight=1.0, interval=20)
    first = strategy.act(observation(), 0)
    assert int((first > 0).sum()) == 4
    assert first.sum() == pytest.approx(1.0)
    # The same seed and step reproduce the same draw; the schedule is respected.
    assert strategy.act(observation(), 0) == pytest.approx(list(first))
    assert strategy.act(observation(), 7) is None
    other = advice.RandomTopK(4, top_k=4, max_weight=1.0, interval=20).act(observation(), 0)
    assert list(other) != list(first)


def test_random_control_respects_the_weight_cap_and_never_leaves_the_eligible_pool():
    strategy = advice.RandomTopK(1, top_k=10, max_weight=0.05, interval=20)
    weights = strategy.act(observation(names=12, eligible=5), 0)
    assert int((weights > 0).sum()) == 5 and weights.max() == pytest.approx(0.05)
    assert not weights[5:].any()
