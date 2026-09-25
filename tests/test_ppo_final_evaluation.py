"""Freeze/test guards and diagnostic regressions; no parameter optimization."""

from copy import deepcopy

import numpy as np
import pytest
from stable_baselines3.common.logger import configure

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.preprocess import read_config, write_json
from src.training.evaluate import evaluate_agent
from src.training.freeze import freeze_selection, validate_frozen
from src.training.train import ValidationCallback
from tests.test_ppo_framework import local_data, no_parameter_updates

pytestmark = pytest.mark.usefixtures("no_parameter_updates")


def test_explicit_test_dates_and_default_guard(local_data):
    data = local_data
    with pytest.raises(ValueError, match="disabled"):
        data.market("test")
    market = data.market("test", allow_test=True)
    assert str(market.dates[0].date()) == data.manifest["val_end"]
    assert str(market.dates[-1].date()) == data.research["dataset"]["end_date"]
    assert market.n_dates == 21


def test_frozen_selection_binds_model_config_and_test_report(local_data, tmp_path):
    data = local_data
    model = make_model(data.vector_env(), data.config)
    model.num_timesteps = 8  # Synthetic checkpoint metadata, not an actual training run.
    agent = PPOAgent(model, data.contract)
    folder = tmp_path / "experiment" / "best_8"
    agent.save(folder, data.config)
    write_json(folder.parent / "selection.json", dict(best_path="best_8", split="val"))
    frozen = freeze_selection(folder, data, tmp_path / "frozen.json")
    loaded = PPOAgent.load(folder, data.contract)
    validate_frozen(loaded, data, frozen)
    for field in ["model_sha256", "policy_sha256", "config_sha256", "research_config_sha256"]:
        wrong = deepcopy(frozen)
        wrong[field] = "mismatch"
        with pytest.raises(ValueError, match="mismatch"):
            validate_frozen(loaded, data, wrong)
    wrong = deepcopy(frozen)
    wrong["source_hashes"] = {}
    with pytest.raises(ValueError, match="mismatch"):
        validate_frozen(loaded, data, wrong)
    with pytest.raises(ValueError, match="frozen"):
        evaluate_agent(loaded, data, tmp_path / "rejected", split="test")
    metrics = evaluate_agent(loaded, data, tmp_path / "test", split="test", frozen=frozen)
    assert metrics["trading_steps"] == 20
    assert read_config(tmp_path / "test/experiment.json")["split"] == "test"
    assert read_config(tmp_path / "frozen.json") == frozen
    model.get_env().close()


def test_freeze_rejects_nonselected_checkpoint(local_data, tmp_path):
    path = tmp_path / "not_best"
    path.mkdir()
    write_json(tmp_path / "selection.json", dict(best_path="best_8", split="val"))
    with pytest.raises(ValueError, match="validation-selected"):
        freeze_selection(path, local_data, tmp_path / "frozen.json")


def test_zero_fills_with_missing_price_do_not_poison_turnover(local_data, tmp_path):
    model = make_model(local_data.vector_env(), local_data.config)
    model.set_logger(configure(str(tmp_path / "logs"), []))
    callback = ValidationCallback(local_data, tmp_path)
    callback.init_callback(model)
    callback.locals = dict(infos=[dict(action_diagnostics=dict(cash_weight=1, saturation_fraction=0),
                                      reward=0, cash=100000, equity=100000,
                                      orders=[dict(filled_shares=0, price=np.nan, fee=0,
                                                   status="rejected", reasons=["missing_open"])])], dones=[False])
    assert callback._on_step()
    assert model.logger.name_to_value["portfolio/gross_turnover"] == 0
    model.get_env().close()
