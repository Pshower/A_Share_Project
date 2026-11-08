"""Mixed-action diagnostics must not imply an independent 2N-Gaussian policy."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import ValidationCallback


class Logger:
    def __init__(self):
        self.name_to_value = {"train/entropy_loss": -30.0}

    def record(self, key, value):
        self.name_to_value[key] = value

    record_mean = record


def callback(tmp_path, mode):
    codes = ["000001", "600000", "600001"]
    data = SimpleNamespace(codes=codes, config=dict(action_mode=mode, validation_rollouts=10000,
                                                   ppo=dict(n_steps=512)),
                           settings=dict(initial_capital=10000, broker=dict(max_weight=.05)))
    result = ValidationCallback(data, tmp_path)
    result.model = SimpleNamespace(num_timesteps=512, _n_updates=1, logger=Logger(), device=torch.device("cpu"),
                                   action_space=SimpleNamespace(shape=(6 if mode == "stock_trade" else 4,)),
                                   policy=SimpleNamespace(parameters=lambda: [], optimizer=SimpleNamespace(state={})),
                                   _last_obs={"valid_mask": np.ones((1, 3))})
    return result


@pytest.mark.parametrize("mode,metric,value", [("stock_trade", "entropy_per_stock", 10),
                                                ("target_weights", "entropy_per_dimension", 7.5)])
def test_entropy_label_matches_distribution_not_packed_storage(tmp_path, mode, metric, value):
    result = callback(tmp_path, mode)
    result.record_update()
    row = result.update_records[0]
    assert row["train/" + metric] == value
    other = "entropy_per_dimension" if mode == "stock_trade" else "entropy_per_stock"
    assert "train/" + other not in row


def test_share_intents_do_not_generate_target_weight_entropy_or_legacy_l1(tmp_path):
    result = callback(tmp_path, "stock_trade")
    result.locals = dict(dones=[False], infos=[dict(
        initial_account={"mode": "cash"}, reward=.01, equity=10000, cash=4000, orders=[],
        action_diagnostics=dict(cash_weight=0, saturation_fraction=0, weights=np.array([1.1, 1.1, 0]),
                                eligible=np.ones(3, dtype=bool), top_n_excess=2, frozen_weight=.1,
                                requested_notional_l1=.6))])
    assert result._on_step()
    result.record_update()
    row = result.update_records[0]
    assert row["portfolio/requested_cash_reference"] == 0
    assert row["portfolio/actual_cash"] == .4
    assert row["portfolio/requested_notional_l1"] == .6
    assert "portfolio/target_cash" not in row
    assert "portfolio/target_weight_entropy" not in row
    assert "portfolio/target_change_l1" not in row
    assert result.previous_target == {}
