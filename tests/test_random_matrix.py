"""Random-initialization matrix controls use mocks and temporary local artifacts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from src.data.preprocess import file_hash, read_config, write_json
from src.envs.initial_account import MODES
from src.training import research_matrix


def control_data(widths=(8, 4), seed=42, prior=False):
    contract = dict(hidden_sizes=list(widths), stock_codes=["000001", "600000"],
                    feature_cols=["signal"], lookback=1, data_source="sina")
    if prior:
        contract["prior"] = dict(feature_index=0, strength=4.0, residual_scale=0.25)
    return SimpleNamespace(
        config=dict(seed=seed),
        contract=contract,
        settings=dict(broker=dict(max_weight=0.05), initial_capital=1000000, seed=seed),
    )


def initial_model(root, run_id, data, *, steps=0, stock_zero=False, cash_zero=False):
    directory = root / "runs" / run_id / "initial_model"
    directory.mkdir(parents=True)
    payload = f"synthetic initial policy {run_id} {data.config['seed']} {data.contract['hidden_sizes']}"
    (directory / "model.zip").write_bytes(payload.encode("ascii"))
    write_json(directory / "metadata.json", dict(
        contract=data.contract, config=data.config, num_timesteps=steps,
        model_sha256=file_hash(directory / "model.zip"),
        test_stock_zero=stock_zero, test_cash_zero=cash_zero,
    ))
    return directory


@pytest.fixture
def mock_controls(monkeypatch):
    calls = dict(load=[], evaluate=[])

    def fake_load(cls, directory, expected_contract, device):
        metadata = read_config(directory / "metadata.json")
        assert metadata["contract"] == expected_contract
        calls["load"].append(directory)

        def head(zero):
            value = 0.0 if zero else 1.0
            return [SimpleNamespace(weight=torch.tensor([[value]]), bias=torch.tensor([value]))]

        network = SimpleNamespace(stock_head=head(metadata["test_stock_zero"]),
                                  cash_head=head(metadata["test_cash_zero"]))
        return SimpleNamespace(model=SimpleNamespace(num_timesteps=metadata["num_timesteps"],
                               policy=SimpleNamespace(network=network)))

    def fake_evaluate(agent, data, output, compare_baselines=False):
        assert compare_baselines
        assert data.settings["seed"] == 20260927
        mode = data.settings["initial_portfolio"]
        calls["evaluate"].append(output)
        output.mkdir(parents=True)
        value = (0.17 if data.contract.get("prior") is not None
                 else data.config["seed"] / 1000 + data.contract["hidden_sizes"][0] / 100)
        metric = dict(cumulative_return=value + MODES.index(mode) / 100)
        write_json(output / "metrics.json", dict(ppo=metric, cash=dict(cumulative_return=0)))
        return metric

    monkeypatch.setattr(research_matrix.PPOAgent, "load", classmethod(fake_load))
    monkeypatch.setattr(research_matrix, "evaluate_agent", fake_evaluate)
    monkeypatch.setattr(research_matrix.torch.cuda, "empty_cache", lambda: None)
    return calls


def test_random_config_removes_continuing_prior_without_changing_other_training_options(tmp_path):
    base = dict(seed=123, hidden_sizes=[256, 128], device="cpu", torch_threads=2,
                output_root="old", prior_feature_index=14, prior_strength=4.0, residual_scale=0.25,
                initial_log_std=-3.0, action_top_k=20, training_episode_steps=256,
                ppo=dict(n_steps=512, batch_size=128, learning_rate=1e-5, n_epochs=5))
    original = deepcopy(base)
    prior = research_matrix.config_for(base, tmp_path, "large", [2048, 1024], 42)
    explicit_prior = research_matrix.config_for(base, tmp_path, "large", [2048, 1024], 42,
                                               initialization_mode="prior")
    random = research_matrix.config_for(base, tmp_path, "large", [2048, 1024], 42,
                                       initialization_mode="random")

    assert prior == explicit_prior
    assert prior["prior_feature_index"] == 14
    assert random["prior_feature_index"] is None
    assert {key: value for key, value in random.items() if key != "prior_feature_index"} == {
        key: value for key, value in prior.items() if key != "prior_feature_index"
    }
    assert random["hidden_sizes"] == [2048, 1024] and random["seed"] == 42
    assert random["initial_log_std"] == -3.0 and random["action_top_k"] == 20
    assert base == original


def test_random_controls_are_per_width_and_seed_and_preserve_the_legacy_prior_cache(tmp_path, mock_controls):
    prior = control_data(prior=True)
    initial = initial_model(tmp_path, "prior_42", prior, stock_zero=True, cash_zero=True)
    old_metrics = research_matrix.evaluate_controls(tmp_path, prior, initial)
    old_directory = tmp_path / "controls"
    old_record = read_config(old_directory / "controls.json")
    old_contract = deepcopy(prior.contract)
    old_contract.pop("hidden_sizes")
    assert old_record["identity"] == dict(data=old_contract, seed=20260927, modes=list(MODES),
                                          broker=prior.settings["broker"], initial_capital=1000000)
    old_files = {path: path.read_bytes() for path in old_directory.rglob("*") if path.is_file()}

    alternate_prior = control_data((16, 8), 123, prior=True)
    alternate_initial = initial_model(tmp_path, "prior_123", alternate_prior, stock_zero=True, cash_zero=True)
    assert research_matrix.evaluate_controls(tmp_path, alternate_prior, alternate_initial) == old_metrics
    assert len(mock_controls["load"]) == 1

    identities, results = [], []
    for size, widths in [("small", (8, 4)), ("large", (16, 8))]:
        for seed in [42, 123]:
            data = control_data(widths, seed)
            run_id = f"{size}_{seed}"
            initial = initial_model(tmp_path, run_id, data)
            result = research_matrix.evaluate_controls(tmp_path, data, initial,
                                                       initialization_mode="random", run_id=run_id)
            control_path = old_directory / run_id / "controls.json"
            record = read_config(control_path)
            identity = record["identity"]
            assert identity["initialization_mode"] == "random"
            assert identity["training_seed"] == seed
            assert identity["seed"] == 20260927
            assert identity["data"]["hidden_sizes"] == list(widths)
            assert identity["initial_model_sha256"] == file_hash(initial / "model.zip")
            assert set(record["files"]) >= {f"{mode}/metrics.json" for mode in MODES}
            count = len(mock_controls["evaluate"])
            assert research_matrix.evaluate_controls(tmp_path, data, initial,
                    initialization_mode="random", run_id=run_id) == result
            assert len(mock_controls["evaluate"]) == count
            identities.append(identity)
            results.append(result["cash"]["ppo"]["cumulative_return"])

    assert len({(identity["training_seed"], tuple(identity["data"]["hidden_sizes"]),
                 identity["initial_model_sha256"]) for identity in identities}) == 4
    assert len(set(results)) == 4
    assert len(mock_controls["load"]) == 5
    assert len(mock_controls["evaluate"]) == 5 * len(MODES)
    assert {path: path.read_bytes() for path in old_files} == old_files


@pytest.mark.parametrize("changed", ["seed", "width", "model"])
def test_random_cache_rejects_changed_initialization_identity(tmp_path, mock_controls, changed):
    data = control_data()
    initial = initial_model(tmp_path, "small_42", data)
    research_matrix.evaluate_controls(tmp_path, data, initial, initialization_mode="random", run_id="small_42")
    if changed == "seed":
        data.config["seed"] = 123
    elif changed == "width":
        data.contract["hidden_sizes"] = [16, 8]
    else:
        (initial / "model.zip").write_bytes(b"different synthetic initialization")
        metadata = read_config(initial / "metadata.json")
        metadata["model_sha256"] = file_hash(initial / "model.zip")
        write_json(initial / "metadata.json", metadata)
    with pytest.raises(ValueError):
        research_matrix.evaluate_controls(tmp_path, data, initial, initialization_mode="random", run_id="small_42")
    assert len(mock_controls["load"]) == 1
    assert len(mock_controls["evaluate"]) == len(MODES)


@pytest.mark.parametrize("damage", ["empty_manifest", "missing_entry", "changed_metric"])
def test_random_cache_requires_all_control_artifacts_to_be_intact(tmp_path, mock_controls, damage):
    data = control_data()
    initial = initial_model(tmp_path, "small_42", data)
    research_matrix.evaluate_controls(tmp_path, data, initial, initialization_mode="random", run_id="small_42")
    directory = tmp_path / "controls" / "small_42"
    marker = directory / "controls.json"
    record = read_config(marker)
    if damage == "empty_manifest":
        record["files"] = {}
        write_json(marker, record)
    elif damage == "missing_entry":
        del record["files"]["cash/metrics.json"]
        write_json(marker, record)
    else:
        write_json(directory / "cash" / "metrics.json", dict(ppo=dict(cumulative_return=999)))
    with pytest.raises(ValueError):
        research_matrix.evaluate_controls(tmp_path, data, initial, initialization_mode="random", run_id="small_42")
    assert len(mock_controls["load"]) == 1
    assert len(mock_controls["evaluate"]) == len(MODES)


@pytest.mark.parametrize("invalid", ["prior", "trained", "zero_stock", "zero_cash", "both_zero"])
def test_random_controls_require_an_untrained_no_prior_random_actor(tmp_path, mock_controls, invalid):
    data = control_data(prior=invalid == "prior")
    initial = initial_model(tmp_path, "small_42", data, steps=1 if invalid == "trained" else 0,
                            stock_zero=invalid in {"zero_stock", "both_zero"},
                            cash_zero=invalid in {"zero_cash", "both_zero"})
    with pytest.raises(ValueError):
        research_matrix.evaluate_controls(tmp_path, data, initial, initialization_mode="random", run_id="small_42")
    assert not mock_controls["evaluate"]
    assert not (tmp_path / "controls" / "small_42" / "controls.json").exists()
