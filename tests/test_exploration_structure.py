"""Synthetic structure checks; no real-history diagnostics or optimizer steps."""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from src.data.preprocess import read_config, write_json, file_hash
from src.training import exploration_structure as structure
from src.training.exploration_env import ExplorationEnv
from tests.test_exploration_policy import observation, policy
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


@pytest.mark.parametrize("architecture,lookback,parameters", [
    ("baseline", 1, 126343), ("baseline", 5, 146823), ("split", 1, 125063)])
def test_fixed_forward_probes_assert_parameter_counts_and_account_semantics(architecture, lookback, parameters):
    tensor = observation(n=4, lookback=lookback)
    model = policy(tensor, architecture)
    obs = {key: value[0].numpy().copy() for key, value in tensor.items()}
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        result, arrays = structure._checks(model, obs)
    assert model.parameter_count == parameters
    assert result["finite_outputs"] and result["permutation_equivariant"]
    assert max(result["permutation_max_errors"]) < 1e-6
    assert arrays["logits"].shape == (4, 4)
    assert arrays["size_mean"].shape == (4,)
    if architecture == "split":
        assert result["split_market_account_invariant"] is True
        assert result["account_perturbation_market_max_delta"] == 0.
    else:
        assert result["split_market_account_invariant"] is None
        assert result["account_perturbation_market_max_delta"] > 0.
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_real_orchestration_of_synthetic_inputs_writes_all18_no_trade_cases(
        local_data, tmp_path, monkeypatch):
    original_config = trade_config(local_data)
    root, old_root = tmp_path / "study", tmp_path / "old"
    old_root.mkdir()
    write_json(old_root / "protocol.json", dict(base_config=original_config))
    real_config_for = structure.config_for

    def synthetic_config(base, path, cell, seed):
        config = real_config_for(base, path, cell, seed)
        config.update(research_config=original_config["research_config"], training_episode_steps=8)
        return config

    monkeypatch.setattr(structure, "config_for", synthetic_config)
    monkeypatch.setattr(structure, "source_hashes", lambda: {"synthetic": "identity"})
    monkeypatch.setattr(structure, "INDICES", (-5, -4, -3))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ExplorationEnv, "step", lambda *args: pytest.fail("Structure probes must never step"))
    before = (torch.random.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())
    threads = torch.get_num_threads()
    report = structure.run_structure_diagnostic(root, old_root)
    assert report["case_count"] == 18
    assert report["training_steps"] == report["training_updates"] == report["environment_steps"] == 0
    assert report["test_evaluated"] is report["validation_evaluated"] is False
    assert report["device"] == "cpu"
    assert torch.equal(before[0], torch.random.get_rng_state())
    state = np.random.get_state()
    assert before[1][0] == state[0] and before[1][2:] == state[2:]
    np.testing.assert_array_equal(before[1][1], state[1])
    assert before[2] == random.getstate()
    assert threads == torch.get_num_threads()
    assert {row["cell"] for row in report["cases"]} == set(structure.CELLS)
    assert {row["train_index"] for row in report["cases"]} == set(structure.INDICES)
    assert {row["mode"] for row in report["cases"]} == {"cash", "random"}
    by_key = {(row["cell"], row["train_index"], row["mode"]): row for row in report["cases"]}
    path = root / "structure" / "observations.npz"
    assert file_hash(path) == report["artifacts"]["observations.npz"]
    with np.load(path, allow_pickle=False) as arrays:
        for negative in structure.INDICES:
            for mode in structure.MODES:
                base = f"baseline_l1_index{negative}_{mode}__"
                five = f"lookback5_index{negative}_{mode}__"
                split = f"split_l1_index{negative}_{mode}__"
                assert by_key["baseline_l1", negative, mode]["date"] == by_key["lookback5", negative, mode]["date"]
                np.testing.assert_array_equal(arrays[base + "stock_features"][:, -1],
                                               arrays[five + "stock_features"][:, -1])
                for name in ("valid_mask", "buy_mask", "sell_mask", "portfolio", "share_state",
                             "trade_lot_bounds", "legal_directions"):
                    np.testing.assert_array_equal(arrays[base + name], arrays[five + name])
                    np.testing.assert_array_equal(arrays[base + name], arrays[split + name])
        assert all(value.dtype != object for value in arrays.values())
    assert read_config(root / "structure" / "summary.json") == report
    with pytest.raises(FileExistsError, match="overwrite"):
        structure.run_structure_diagnostic(root, old_root)


def test_cli_uses_registered_output_defaults_without_running(monkeypatch):
    called = []
    monkeypatch.setattr(structure, "run_structure_diagnostic", lambda *args: called.append(args))
    structure.main([])
    assert called == [("reports/runs/ppo_exploration_20261006", "reports/runs/ppo_trade_actor_20261005")]
