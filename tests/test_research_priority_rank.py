"""Synthetic priority-study controls; no learning or protected performance."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.agents.priority_agent import PriorityAgent
from src.backtest.run import run_policy
from src.data.preprocess import file_hash, read_config, write_json
from src.envs.initial_account import MODES
from src.training import research_priority_rank as study
from src.training.freeze import json_digest
from src.training.priority_aux import build_rank_pool
from src.training.matrix_experience import EpisodeAudit
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_trade_policy import trade_config


pytestmark = pytest.mark.usefixtures("no_parameter_updates")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def reference(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    write_json(root / "protocol.json", dict(folds={fold: dict(directory=str(tmp_path / fold)) for fold in study.FOLDS}))
    return root


@pytest.mark.parametrize("fold", study.FOLDS)
@pytest.mark.parametrize("actor", study.ACTORS)
@pytest.mark.parametrize("method", study.METHODS)
def test_config_registers_exact_axes_with_no_other_reward_changes(reference, tmp_path, fold, actor, method):
    base = dict(ppo={})
    before = deepcopy(base)
    config = study.config_for(base, reference, tmp_path / "run", fold, actor, method)
    assert base == before
    assert config["action_mode"] == "stock_priority"
    assert config["priority_rank"]["actor_variant"] == actor
    assert config["priority_rank"]["training_method"] == method
    assert config["priority_rank"]["auxiliary_coefficient"] == (.05 if method == "rank_aux" else 0.)
    assert config["priority_rank"]["market_pooling"] == "pure_mean"
    assert config["ppo"]["learning_rate"] == 3e-5
    assert config["ppo"]["n_steps"] == 512 and config["ppo"]["batch_size"] == 128
    assert config["ppo"]["gamma"] == .99 and config["ppo"]["n_epochs"] == 5
    assert config["initial_portfolio"] == "mixed" and config["training_episode_steps"] == 256
    assert config["top_n_choices"] == [10, 20, 40] and config["prior_feature_index"] is None
    assert config["exploration"]["reward_basis"] == "simple_net"


@pytest.mark.parametrize("fold,actor,method,profile", [
    ("2025", "coupled", "ppo", "standard"), ("2023", "other", "ppo", "standard"),
    ("2023", "coupled", "teacher", "standard"), ("2023", "coupled", "ppo", "adaptive")])
def test_invalid_condition_rejected_before_file_reads(tmp_path, fold, actor, method, profile):
    with pytest.raises(ValueError, match="Unregistered"):
        study.config_for({}, tmp_path / "missing", tmp_path, fold, actor, method, profile)


def priority_config(local_data, actor="coupled", method="ppo"):
    config = trade_config(local_data)
    config.update(action_mode="stock_priority", torch_threads=1, training_episode_steps=8,
        recent_episode_every=4, actor_output_gain=.1, initial_log_std=-.7, initial_portfolio="mixed",
        exploration=dict(architecture="split", common_lookback=5, sampling="recent", reward_basis="simple_net", size_annealing=False),
        priority_rank=dict(fold="2023", actor_variant=actor, training_method=method, market_pooling="pure_mean",
            profile="standard", auxiliary_coefficient=.05 if method == "rank_aux" else 0.,
            auxiliary_batch_size=1024, auxiliary_horizon=5, auxiliary_huber_delta=.1, pool_identity={}))
    config["ppo"]["learning_rate"] = 3e-5
    config["hidden_sizes"] = [256, 128]
    config["device"] = "cpu"
    return config


@pytest.mark.parametrize("actor", study.ACTORS)
@pytest.mark.parametrize("method", study.METHODS)
def test_priority_data_keeps_original_support_but_registers_3n(local_data, actor, method):
    config = priority_config(local_data, actor, method)
    data = study.PriorityData(config)
    borrowed = deepcopy(config)
    borrowed["action_mode"] = "stock_trade"
    original = study.ExplorationData(borrowed)
    pd.testing.assert_frame_equal(data.features, original.features)
    pd.testing.assert_frame_equal(data.prices, original.prices)
    assert data.contract["artifacts"] == original.contract["artifacts"]
    assert data.contract["action_mode"] == "stock_priority"
    assert data.contract["priority_rank"] == config["priority_rank"]
    env, reference_env = data.env("train"), original.env("train")
    try:
        obs, _ = env.reset(seed=42)
        old, _ = reference_env.reset(seed=42)
        assert env.action_space.shape == (3 * len(data.codes),)
        for name in obs:
            np.testing.assert_array_equal(obs[name], old[name])
        assert env.observation_space.contains(obs)
    finally:
        env.close()
        reference_env.close()


@pytest.mark.parametrize("field,value", [("market_pooling", "fused"), ("auxiliary_coefficient", .1),
    ("auxiliary_horizon", 1), ("auxiliary_batch_size", 128), ("auxiliary_huber_delta", 1.)])
def test_data_rejects_unregistered_auxiliary_or_pooling_changes(local_data, field, value):
    config = priority_config(local_data)
    config["priority_rank"][field] = value
    with pytest.raises(ValueError, match="registered"):
        study.PriorityData(config)


def test_methods_share_initial_policy_bytes_but_not_canonical_actor_contract(local_data):
    values = []
    for actor in study.ACTORS:
        for method in study.METHODS:
            data = study.PriorityData(priority_config(local_data, actor, method))
            env = data.env("train")
            try:
                model = study.make_model(env, data.config, None)
                assert sum(p.numel() for p in model.policy.parameters()) == study.PARAMETERS
                values.append((actor, method, {k: v.clone() for k, v in model.policy.state_dict().items()}, data.contract))
            finally:
                env.close()
    for _, _, state, _ in values:
        for name, tensor in values[0][2].items():
            torch.testing.assert_close(state[name], tensor, rtol=0, atol=0)
    assert study._canonical_contract(values[0][3]) == study._canonical_contract(values[1][3])
    assert study._canonical_contract(values[0][3]) != study._canonical_contract(values[2][3])


@pytest.mark.parametrize("actor", study.ACTORS)
def test_execution_bridge_matches_new_wrapper_orders_and_account_book(local_data, actor):
    data = study.PriorityData(priority_config(local_data, actor))
    data.settings.update(seed=study.EVAL_SEED, initial_portfolio="equal", trade_top_n=20)
    env = data.env("val")
    model = study.make_model(env, data.config, None)
    agent = PriorityAgent(model, data.contract)
    try:
        bridge_daily, positions, orders, _ = run_policy(data.market("val"), study._ExecutionAdapter(agent), data.settings)
        obs, initial = env.reset(seed=study.EVAL_SEED)
        direct_equity = [initial["equity"]]
        direct_positions = [initial["positions"]]
        i = 0
        while True:
            action = agent.act(obs, i)
            obs, _, terminated, truncated, info = env.step_order(action)
            direct_equity.append(info["equity"])
            direct_positions.append(info["positions"])
            i += 1
            if terminated or truncated:
                break
        np.testing.assert_array_equal(bridge_daily.equity.to_numpy(), direct_equity)
        np.testing.assert_array_equal(positions.to_numpy(), direct_positions)
        assert len(orders) == len(env.unwrapped.broker.orders)
        assert agent.contract["action_mode"] == "stock_priority"
    finally:
        env.close()


@pytest.mark.parametrize("kind,interval", [("hold", 1), ("momentum_20", 1), ("momentum_20", 20)])
def test_lifted_rules_keep_exact_original_economic_actions(local_data, kind, interval):
    data = study.PriorityData(priority_config(local_data))
    env = data.env("train")
    try:
        obs, _ = env.reset(seed=42)
        lifted = study.PriorityEconomicRule(data.contract, kind, interval).act(obs, 0)
        original = study.EconomicRule(data.contract, kind, interval).act(obs, 0)
        for name in ("shares", "priority", "freeze_mask", "exit_mask"):
            np.testing.assert_array_equal(getattr(lifted, name), getattr(original, name))
        assert lifted.diagnostics["raw_packed_action"].shape == (3 * len(data.codes),)
    finally:
        env.close()


def fake_records(profile="standard"):
    values = []
    for fold in study.FOLDS:
        for actor in study.ACTORS:
            for method in study.METHODS:
                for seed in study.SEEDS:
                    for budget, steps in study.PROFILES[profile].items():
                        initial = .01 + (actor == "decoupled") * .001 + (seed == 123) * .002
                        gain = .02 + (actor == "decoupled") * .003 + (method == "rank_aux") * .004
                        gain += (actor == "decoupled" and method == "rank_aux") * .002
                        gain += (fold == "2024") * .001
                        metrics = dict(cumulative_return=initial + gain, max_drawdown=-.04, total_fees=100., total_slippage_cost=50.)
                        values.append(dict(fold=fold, actor=actor, method=method, seed=seed, budget=budget, steps=steps,
                            metrics={mode: dict(ppo=deepcopy(metrics), initialized=dict(metrics, cumulative_return=initial)) for mode in MODES}))
    return values


def fake_rules():
    return {fold: {mode: dict(momentum_20day=dict(cumulative_return=.005)) for mode in MODES} for fold in study.FOLDS}


@pytest.mark.parametrize("profile,steps,snapshots,checkpoint_sum", [
    ("standard", 393216, 48, 589824), ("extended", 786432, 72, 1376256)])
def test_summary_counts_only_long_trajectory_and_pairs_three_seeds(profile, steps, snapshots, checkpoint_sum):
    result = study.summarize(fake_records(profile), fake_rules(), study.PROFILES[profile])
    assert result["actual_training_steps"] == steps and result["checkpoint_steps_sum"] == checkpoint_sum
    assert result["snapshots"] == snapshots and result["trained_runs"] == 24
    assert result["cells"]["coupled_ppo_short"]["metrics"]["cumulative_return"]["count"] == 3
    assert result["qualified_candidate"] == dict(actor="decoupled", method="rank_aux", budget="short",
                                                 representative_seed=42, representative_fold="2024")
    assert not result["actionable"] and not result["test_evaluated"]


def test_factorial_init_learning_and_interaction_reconcile():
    result = study.summarize(fake_records(), fake_rules(), study.PROFILES["standard"])
    effect = result["factorial_effects"]["across_fold"]["long"]
    assert effect["net_roi"]["actor_decoupled_minus_coupled"]["mean"] == pytest.approx(.005)
    assert effect["net_roi"]["method_rank_aux_minus_ppo"]["mean"] == pytest.approx(.005)
    assert effect["net_roi"]["interaction"]["mean"] == pytest.approx(.002)
    assert effect["initialization"]["actor_decoupled_minus_coupled"]["mean"] == pytest.approx(.001)
    assert effect["initialization"]["method_rank_aux_minus_ppo"]["mean"] == 0.
    assert effect["learning_gain"]["actor_decoupled_minus_coupled"]["mean"] == pytest.approx(.004)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "steps", "fold", "seed", "actor", "method", "budget"])
def test_partial_or_changed_results_cannot_claim_complete(failure):
    records = fake_records()
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records[-1] = deepcopy(records[-2])
    else:
        field, value = {"steps": ("steps", 4096), "fold": ("fold", "2025"), "seed": ("seed", 7),
                       "actor": ("actor", "other"), "method": ("method", "teacher"), "budget": ("budget", "extra")}[failure]
        records[-1][field] = value
    with pytest.raises(ValueError):
        study.validate_records(records, study.PROFILES["standard"])


def test_gate_requires_worst_account_and_each_fold_not_pooled_average():
    records = fake_records()
    for row in records:
        if row["fold"] == "2023":
            row["metrics"]["cash"]["ppo"]["max_drawdown"] = -.101
    assert study.summarize(records, fake_rules(), study.PROFILES["standard"])["qualified_candidate"] is None


def test_methods_cannot_silently_reuse_unequal_initial_function_controls():
    records = fake_records()
    records[-1]["metrics"]["cash"]["initialized"]["cumulative_return"] += .01
    with pytest.raises(ValueError, match="identical initialized"):
        study.summarize(records, fake_rules(), study.PROFILES["standard"])


def test_summary_is_invariant_to_record_order():
    records = fake_records()
    assert study.summarize(records, fake_rules(), study.PROFILES["standard"]) == study.summarize(list(reversed(records)), fake_rules(), study.PROFILES["standard"])


@pytest.mark.parametrize("failure", ["account", "nan", "boolean", "missing_metric", "profile"])
def test_summary_rejects_incomplete_or_nonfinite_economic_inputs(failure):
    records = fake_records()
    budgets = deepcopy(study.PROFILES["standard"])
    if failure == "account":
        records[-1]["metrics"].pop("cash")
    elif failure == "nan":
        records[-1]["metrics"]["cash"]["ppo"]["cumulative_return"] = np.nan
    elif failure == "boolean":
        records[-1]["metrics"]["cash"]["ppo"]["max_drawdown"] = False
    elif failure == "missing_metric":
        records[-1]["metrics"]["cash"]["ppo"].pop("total_fees")
    else:
        budgets["long"] = 32768
    with pytest.raises(ValueError):
        study.summarize(records, fake_rules(), budgets)


def worker_fixture(tmp_path):
    protocol = dict(synthetic=True)
    write_json(tmp_path / "protocol.json", protocol)
    completed = []
    for actor in study.ACTORS:
        for method in study.METHODS:
            for seed in study.SEEDS:
                run_id = f"2023_{actor}_{method}_{seed}"
                marker = tmp_path / "runs" / run_id / "completed.json"
                marker.parent.mkdir(parents=True)
                write_json(marker, dict(synthetic=run_id))
                completed.append(dict(run_id=run_id, completed_sha256=file_hash(marker)))
    receipt = dict(status="complete", fold="2023", protocol_sha256=file_hash(tmp_path / "protocol.json"),
                   completed_runs=completed, no_global_summary_writes=True)
    write_json(tmp_path / "worker_2023.json", receipt)
    return protocol, receipt


def test_report_barrier_requires_all_twelve_signed_worker_trials(tmp_path):
    protocol, receipt = worker_fixture(tmp_path)
    assert study.verify_worker_receipt(tmp_path, protocol, "2023") == receipt


@pytest.mark.parametrize("bad", ["running", "duplicate", "missing", "protocol", "marker", "shared_summary"])
def test_worker_receipt_rejects_partial_or_changed_completion(tmp_path, bad):
    protocol, receipt = worker_fixture(tmp_path)
    if bad == "running":
        receipt["status"] = "running"
    elif bad == "duplicate":
        receipt["completed_runs"][-1] = deepcopy(receipt["completed_runs"][-2])
    elif bad == "missing":
        receipt["completed_runs"].pop()
    elif bad == "protocol":
        receipt["protocol_sha256"] = "0" * 64
    elif bad == "shared_summary":
        receipt["no_global_summary_writes"] = False
    else:
        receipt["completed_runs"][-1]["completed_sha256"] = "0" * 64
    write_json(tmp_path / "worker_2023.json", receipt)
    with pytest.raises(ValueError):
        study.verify_worker_receipt(tmp_path, protocol, "2023")


def test_missing_cache_returns_none_and_empty_signed_cache_is_not_success(tmp_path):
    assert study.cached_trial(tmp_path, {}, {}) is None
    write_json(tmp_path / "completed.json", dict(config_sha256=json_digest({}), protocol_sha256=json_digest({}), records=[], files={}))
    with pytest.raises(ValueError, match="nonempty"):
        study.cached_trial(tmp_path, {}, {})


@pytest.mark.parametrize("args", [["--stage", "train"], ["--stage", "report", "--fold", "2023"],
                                  ["--stage", "prepare", "--fold", "2024"]])
def test_cli_rejects_ambiguous_worker_ownership_before_prepare(args):
    with pytest.raises(SystemExit):
        study.main(args)


def test_report_stage_does_not_implicitly_create_registration(tmp_path):
    with pytest.raises(ValueError, match="Only prepare"):
        study.main(["--stage", "report", "--output", str(tmp_path / "missing")])


def test_evaluate_saves_original_3n_and_real_ledger_not_targets(local_data, tmp_path):
    data = study.PriorityData(priority_config(local_data, "decoupled"))
    data.settings.update(seed=study.EVAL_SEED, initial_portfolio="equal", trade_top_n=20)
    env = data.env("train")
    try:
        agent = PriorityAgent(study.make_model(env, data.config, None), data.contract)
        result = study.evaluate_case(agent, data, tmp_path / "case", "synthetic")
        assert result["trace"]["schema_version"] == 2
        assert result["contract"] == data.contract
        assert result["initial_mode"] == "equal" and result["top_n"] == 20
        assert result["mode_statistics"]["economic_violations"] == 0
        assert not result["test_evaluated"] and not result["actionable"]
        chunk = result["trace"]["chunks"][0]
        with np.load(tmp_path / "case" / "trace" / chunk["path"], allow_pickle=False) as arrays:
            assert arrays["raw_packed_action"].shape[1] == 3 * len(data.codes)
            assert np.isfinite(arrays["log_prob"]).all()
    finally:
        env.close()


@pytest.mark.parametrize("actor", study.ACTORS)
def test_callback_keeps_original_rollout_and_cpu_safe_fixed_snapshot(local_data, tmp_path, monkeypatch, actor):
    data = study.PriorityData(priority_config(local_data, actor))
    experience = EpisodeAudit(data.env("train"))
    vector = DummyVecEnv([lambda: Monitor(experience)])
    vector.seed(42)
    output = tmp_path / "callback"
    output.mkdir()
    callback = None
    try:
        model = study.make_model(vector, data.config, None)
        model.set_logger(configure(str(output / "logs"), ["csv"]))
        initial = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
        callback = study.PriorityCallback(data, output, experience, initial, dict(synthetic=8),
                                          output / "status.json", "synthetic_only")
        model._setup_learn(8, callback)
        assert model.collect_rollouts(vector, callback, model.rollout_buffer, 8)
        assert model.num_timesteps == experience.stats()["total_steps"] == 8
        assert not model.policy.optimizer.state
        # Fabricated state counters exercise persistence, never an optimizer update.
        for parameter in model.policy.parameters():
            model.policy.optimizer.state[parameter] = dict(step=torch.tensor(1.),
                exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
        def forbidden(*args):
            raise AssertionError("CPU snapshot must not query a CUDA device")
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", forbidden)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", forbidden)
        callback.save_fixed_snapshot()
        audit = read_config(output / "synthetic_audit.json")
        assert audit["cuda_peak_allocated_bytes"] is None and audit["cuda_peak_reserved_bytes"] is None
        assert audit["auxiliary_audit"]["actual_joint_optimizer_updates"] == 0
        assert audit["auxiliary_audit"]["actual_aux_updates"] == 0
        assert audit["changed_tensors"] == 0 and audit["all_parameters_finite"]
        trace = callback.trace.close()
        study.verify_priority_trace(output / "training_trace", trace, actor_variant=actor, steps=8)
        with np.load(output / "training_trace" / trace["chunks"][0]["path"], allow_pickle=False) as values:
            np.testing.assert_array_equal(values["raw_packed_action"], model.rollout_buffer.actions[:, 0])
            np.testing.assert_array_equal(values["log_prob"], model.rollout_buffer.log_probs[:, 0])
        for name, tensor in model.policy.state_dict().items():
            torch.testing.assert_close(tensor, initial[name], rtol=0, atol=0)
    finally:
        if callback is not None:
            callback.trace.close()
        vector.close()


@pytest.mark.parametrize("bad", ["codes", "features", "train_end", "horizon", "common_lookback", "input_hashes"])
def test_pool_validation_rejects_same_width_wrong_training_schema(local_data, bad):
    from types import SimpleNamespace
    data = study.PriorityData(priority_config(local_data))
    hashes = dict(panel="0" * 64)
    columns = pd.MultiIndex.from_product([data.codes, data.manifest["feature_cols"]])
    prices = pd.MultiIndex.from_product([data.codes, ("open", "close")])
    pool_hashes = dict(hashes,
        truncated_features=study._frame_digest(data.features.loc[:data.manifest["train_end"], columns]),
        truncated_open_close=study._frame_digest(data.prices.loc[:data.manifest["train_end"], prices]))
    values = dict(codes=tuple(data.codes), feature_cols=tuple(data.manifest["feature_cols"]),
                  train_end=data.manifest["train_end"], horizon=5, common_lookback=5, input_hashes=pool_hashes)
    if bad in {"codes", "features"}:
        key = "feature_cols" if bad == "features" else "codes"
        values[key] = tuple(reversed(values[key]))
    elif bad == "train_end":
        values[bad] = "2024-01-01"
    elif bad == "input_hashes":
        values[bad] = dict(panel="1" * 64)
    else:
        values[bad] = 1
    with pytest.raises(ValueError, match="label order"):
        study.validate_pool(SimpleNamespace(**values), data, hashes)


def test_real_builder_pool_extra_truncated_hashes_match_runner(local_data):
    data = study.PriorityData(priority_config(local_data))
    hashes = dict(panel="0" * 64)
    pool = build_rank_pool(data.features, data.prices, data.codes, data.manifest["feature_cols"],
        data.manifest["train_end"], input_hashes=hashes)
    assert study.validate_pool(pool, data, hashes) is pool
    assert set(pool.input_hashes) == {"panel", "truncated_features", "truncated_open_close"}


def case_fixture():
    contract = dict(stock_codes=["A", "B"], trade=dict(actor_variant="coupled"),
                    priority_rank=dict(training_method="ppo", auxiliary_coefficient=0., actor_variant="coupled"))
    result = dict(contract=deepcopy(contract), initial_mode="cash", top_n=20, test_evaluated=False,
        trace=dict(identity=dict(contract_sha256=json_digest(contract), codes=["A", "B"], actor_variant="coupled",
                                 seed=study.EVAL_SEED, initial_mode="cash", top_n=20)))
    return result, contract


def test_initial_control_case_allows_only_canonical_method_difference():
    result, contract = case_fixture()
    contract["priority_rank"].update(training_method="rank_aux", auxiliary_coefficient=.05)
    assert study.validate_case_identity(result, contract, "cash", initialized=True) is result
    with pytest.raises(ValueError, match="different contract"):
        study.validate_case_identity(result, contract, "cash")


@pytest.mark.parametrize("bad", ["contract", "digest", "codes", "actor", "seed", "mode", "top_n", "test"])
def test_case_identity_rejects_cross_pool_contract_or_account_curves(bad):
    result, contract = case_fixture()
    identity = result["trace"]["identity"]
    if bad == "contract":
        result["contract"]["stock_codes"] = ["B", "A"]
    elif bad == "digest":
        identity["contract_sha256"] = "0" * 64
    elif bad == "codes":
        identity["codes"] = ["B", "A"]
    elif bad == "actor":
        identity["actor_variant"] = "decoupled"
    elif bad == "seed":
        identity["seed"] = 42
    elif bad == "mode":
        identity["initial_mode"] = "equal"
    elif bad == "top_n":
        result["top_n"] = 40
    else:
        result["test_evaluated"] = True
    with pytest.raises(ValueError, match="different contract"):
        study.validate_case_identity(result, contract, "cash")
