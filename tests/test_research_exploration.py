"""Fixed configuration, seed aggregation, identity guards and native evaluation."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.preprocess import read_config
from src.training.exploration_env import EconomicRule
from src.training.research_exploration import (FORMAL, PILOTS, BUDGETS, SEEDS, config_for, summarize,
                                              ExplorationCallback, evaluate_case, load_agent)
from tests.test_ppo_framework import no_parameter_updates


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def base_config():
    return dict(ppo=dict(use_sde=False), prior_feature_index=14, action_top_k=20,
                max_delta_weight=.1, training_date_policy="intersection")


@pytest.mark.parametrize("cell", list(FORMAL) + list(PILOTS))
def test_registered_config_changes_only_declared_axes(tmp_path, cell):
    original = base_config()
    unchanged = deepcopy(original)
    config = config_for(original, tmp_path, cell, 42)
    assert original == unchanged
    assert config["actor_output_gain"] == .1 and config["initial_log_std"] == -.7
    assert config["exploration"]["common_lookback"] == 5
    assert config["prior_feature_index"] is None and "action_top_k" not in config
    assert "training_date_policy" not in config
    assert config["ppo"]["learning_rate"] == 1e-5
    assert config["ppo"]["gamma"] == (.995 if cell == "gamma_0995" else .99)
    assert config["exploration"]["reward_basis"] == ("log_net" if cell == "log_net" else "simple_net")
    assert config["exploration"]["size_annealing"] == (cell == "size_anneal")
    assert config["exploration"]["sampling"] == ("quarter_balanced" if cell == "balanced_l1" else "recent")


def records(gain=.02, drawdown=-.02):
    rows = []
    for cell in FORMAL:
        for budget, steps in BUDGETS.items():
            for seed in SEEDS:
                roi = gain + (seed == 123) * .01
                row = dict(cell=cell, budget=budget, seed=seed, steps=steps, pilot=False, parameters=126343,
                    metrics={mode: dict(ppo=dict(cumulative_return=roi, max_drawdown=drawdown,
                        total_fees=100., total_slippage_cost=50.),
                        initialized=dict(cumulative_return=0., max_drawdown=0., total_fees=0., total_slippage_cost=0.))
                        for mode in ("cash", "equal", "concentrated", "random")})
                rows.append(row)
    for cell in PILOTS:
        rows.append(dict(cell=cell, seed=42, budget="pilot", steps=2048, pilot=True, parameters=126343,
                         metrics=deepcopy(rows[0]["metrics"])))
    return rows


def rules():
    return {mode: dict(momentum_daily=dict(cumulative_return=.01, max_drawdown=-.01,
        total_fees=10., total_slippage_cost=5.)) for mode in ("cash", "equal", "concentrated", "random")}


def test_summary_counts_seeds_not_accounts_and_preserves_actual_budget():
    report = summarize(records(), rules())
    cell = report["cells"]["baseline_l1_short"]
    assert cell["metrics"]["cumulative_return"]["count"] == 3
    assert cell["metrics"]["cumulative_return"]["mean"] == pytest.approx(.02 + .01 / 3)
    assert cell["positive_learning_seeds"] == 3
    assert report["actual_training_steps"] == 104448
    assert report["qualified_candidate"] == dict(cell="balanced_l1", budget="short", representative_seed=42)
    assert report["paired_to_baseline"]["lookback5_short"]["mean"] == 0
    assert report["actionable"] is False and report["test_evaluated"] is False


@pytest.mark.parametrize("bad", ["dd", "profit", "incomplete"])
def test_risk_profit_and_complete_seed_gates(bad):
    rows = records()
    if bad == "incomplete":
        with pytest.raises(ValueError):
            summarize(rows[:-1], rules())
        return
    for row in rows:
        for account in row["metrics"].values():
            if bad == "dd":
                account["ppo"]["max_drawdown"] = -.101
            else:
                account["ppo"]["cumulative_return"] = .005
    assert summarize(rows, rules())["qualified_candidate"] is None


@pytest.mark.parametrize("issue", ["no_pilots", "duplicate", "wrong_steps", "wrong_seed"])
def test_incomplete_or_invalid_pilots_never_claim_completed_budget(issue):
    from src.training.research_exploration import validate_records
    rows = records()
    if issue == "no_pilots":
        rows = rows[:-3]
    elif issue == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif issue == "wrong_steps":
        rows[-1]["steps"] = 8192
    else:
        rows[-1]["seed"] = 123
    with pytest.raises(ValueError):
        validate_records(rows)


def test_one_bad_account_is_not_hidden_by_mean_drawdown():
    rows = records()
    for row in rows:
        if row["seed"] == 42:
            row["metrics"]["cash"]["ppo"]["max_drawdown"] = -.11
    assert summarize(rows, rules())["qualified_candidate"] is None


def test_annealing_is_at_rollout_start_not_during_sampling(monkeypatch, tmp_path):
    from src.training.research_exploration import TradeCallback
    monkeypatch.setattr(TradeCallback, "_on_rollout_start", lambda self: None)
    callback = object.__new__(ExplorationCallback)
    callback.annealing, callback.schedule, callback.budgets, callback.output = True, [], dict(pilot=2048), tmp_path
    parameter = torch.nn.Parameter(torch.tensor([-.7]))
    callback.model = SimpleNamespace(num_timesteps=0, n_steps=512,
                                    policy=SimpleNamespace(network=SimpleNamespace(log_std=parameter)))
    for step, expected in [(0, -.7), (512, -1.133333333333), (1024, -1.56666666667), (1536, -2.)]:
        callback.model.num_timesteps = step
        callback._on_rollout_start()
        assert parameter.item() == pytest.approx(expected)
    saved = read_config(tmp_path / "size_schedule.json")
    assert len(saved["events"]) == 4
    assert saved["optimizer_still_updates_std"] and saved["mutations_only_before_rollout"]


def test_rule_native_evaluation_saves_book_and_done(tmp_path):
    from tests.test_exploration_env import market
    from src.agents.delta_mapping import EXECUTION_FIELDS
    from src.agents.trade_mapping import TRADE_ALLOCATION_FIELDS
    settings = dict(initial_capital=1e6, broker=dict(commission=.0003, sell_tax=.0005, slippage=.001,
        min_commission=5., lot_size=100, max_weight=.05, volume_fraction=.01), seed=20260927,
        initial_portfolio="equal", trade_top_n=1)
    contract = dict(action_mode="stock_trade", stock_codes=["A", "B"], feature_cols=["momentum_20"],
        broker=settings["broker"], logit_bound=8., artifacts={},
        trade=dict(max_trade_weight=.05, max_commission_ratio=.001, top_n=1))
    data = SimpleNamespace(contract=contract, settings=settings, codes=["A", "B"], config=dict(max_commission_ratio=.001),
                           market=lambda split: market())
    result = evaluate_case(EconomicRule(contract), data, tmp_path / "hold", "rule")
    assert result["metrics"]["total_fees"] == 0
    assert result["mode_statistics"]["economic_violations"] == 0
    arrays = np.load(tmp_path / "hold" / "trace" / result["trace"]["chunks"][0]["path"])
    assert arrays["done"][-1]
    assert not arrays["done"][:-1].any()
    assert "reward" in arrays
