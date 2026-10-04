"""Pure trade-study registration, arithmetic, original traces and pilot routing."""

from copy import deepcopy
import gzip
import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.training import research_trade as study
from tests.test_ppo_framework import local_data


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    from stable_baselines3 import PPO
    monkeypatch.setattr(PPO, "learn", lambda *args, **kwargs: pytest.fail("Unit tests cannot train"))
    monkeypatch.setattr(torch.Tensor, "backward", lambda *args, **kwargs: pytest.fail("Unit tests cannot backpropagate"))
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *args, **kwargs: pytest.fail("Unit tests cannot update parameters"))


def _base():
    return dict(action_top_k=20, prior_feature_index=14, seed=1, hidden_sizes=[64, 32],
        ppo=dict(n_steps=512, batch_size=128, n_epochs=5, learning_rate=1e-5, gamma=.99,
                 gae_lambda=.95, clip_range=.2, target_kl=.03, ent_coef=0., vf_coef=.5, max_grad_norm=.5))


def _pilot(candidate=None, **changes):
    result = dict(candidate or study.PILOT_CANDIDATES[0], status="success", actual_steps=2048,
        all_parameters_finite=True, all_actions_probabilities_finite=True, optimizer_steps=80,
        tail_mean_kl=.02, deterministic_cash_buy_dates=1, probe_economic_violations=0,
        mode_statistics=dict(filled_modes=dict(buy=10, reduce=3, exit=0), voluntary_hold=8,
                             cash_training_buy_fills=2, economic_violations=0),
        validation_evaluated=False, test_evaluated=False)
    result.update(changes)
    return result


def test_registered_trade_config_preserves_base_and_fixes_exploration_and_rewards(tmp_path):
    base = _base()
    original = deepcopy(base)
    config = study.config_for(base, tmp_path, 123, study.PILOT_CANDIDATES[-1])
    assert base == original
    assert config["action_mode"] == "stock_trade" and "action_top_k" not in config
    assert config["prior_feature_index"] is None and config["hidden_sizes"] == [256, 128]
    assert config["seed"] == 123 and config["ppo"]["learning_rate"] == 3e-5
    assert config["actor_output_gain"] == .1 and config["initial_log_std"] == -.7
    assert config["max_trade_weight"] == .05 and config["max_commission_ratio"] == .001
    assert config["top_n"] == 20 and config["top_n_choices"] == [10, 20, 40]
    assert config["ppo"]["ent_coef"] == 0 and config["ppo"]["vf_coef"] == .5


@pytest.mark.parametrize("field,value", [("n_steps", 256), ("target_kl", .01), ("ent_coef", .1), ("n_epochs", 4)])
def test_config_rejects_unregistered_ppo_variants(tmp_path, field, value):
    base = _base()
    base["ppo"][field] = value
    with pytest.raises(ValueError, match="fixed PPO"):
        study.config_for(base, tmp_path)


@pytest.mark.parametrize("changes", [dict(status="error"), dict(actual_steps=512),
    dict(all_parameters_finite=False), dict(all_actions_probabilities_finite=False), dict(optimizer_steps=63),
    dict(tail_mean_kl=.04500001), dict(tail_mean_kl=None), dict(tail_mean_kl=float("nan")),
    dict(tail_mean_kl=-.01), dict(deterministic_cash_buy_dates=0), dict(probe_economic_violations=1),
    dict(validation_evaluated=True), dict(test_evaluated=True)])
def test_pilot_gate_fails_closed(changes):
    assert not study.pilot_eligible(_pilot(**changes))
    assert study.choose_pilot([_pilot(**changes)]) is None


@pytest.mark.parametrize("field,value", [("cash_training_buy_fills", 0), ("voluntary_hold", 0), ("economic_violations", 1)])
def test_pilot_gate_requires_actual_cash_buy_voluntary_hold_and_economic_fills(field, value):
    report = _pilot()
    report["mode_statistics"][field] = value
    assert not study.pilot_eligible(report)
    report = _pilot()
    report["mode_statistics"]["filled_modes"].update(reduce=0, exit=0)
    assert not study.pilot_eligible(report)


def test_selector_prefers_updates_then_distinct_modes_not_volume_or_profit():
    reports = [_pilot(candidate, validation_return=100 - index) for index, candidate in enumerate(study.PILOT_CANDIDATES)]
    assert study.choose_pilot(reports) == reports[0]
    reports[-1]["optimizer_steps"] = 81
    assert study.choose_pilot(reports) == reports[-1]
    reports[-1]["optimizer_steps"] = 80
    reports[1]["mode_statistics"]["filled_modes"]["buy"] = 999999
    assert study.choose_pilot(reports) == reports[0]
    reports[-1]["mode_statistics"]["filled_modes"]["exit"] = 1
    assert study.choose_pilot(reports) == reports[-1]


def test_selector_healthy_kl_band_is_not_minimum_kl():
    tiny = _pilot(study.PILOT_CANDIDATES[0], tail_mean_kl=1e-8)
    healthy = _pilot(study.PILOT_CANDIDATES[-1], tail_mean_kl=.02)
    assert study.choose_pilot([tiny, healthy]) == healthy
    healthy["tail_mean_kl"] = .04
    assert study.choose_pilot([tiny, healthy]) == healthy
    assert study.kl_band_distance(.001) == 0 == study.kl_band_distance(.03)
    assert study.kl_band_distance(.0005) == pytest.approx(math.log(2))
    assert study.kl_band_distance(.06) == pytest.approx(math.log(2))
    assert study.pilot_eligible(_pilot(optimizer_steps=64, tail_mean_kl=.045))


def test_four_train_only_pilots_do_not_automatically_evaluate_or_train_formal(tmp_path, monkeypatch):
    calls = []
    def fake(config, root, candidate, protocol):
        calls.append(config)
        folder = root / "pilots" / candidate["name"]
        folder.mkdir(parents=True)
        study.write_json(folder / "completed.json", dict(mock=True))
        return _pilot(candidate)
    monkeypatch.setattr(study, "run_pilot", fake)
    monkeypatch.setattr(study, "evaluate_agent", lambda *args, **kwargs: pytest.fail("Pilot evaluated validation"))
    monkeypatch.setattr(study, "run_formal", lambda *args: pytest.fail("Pilot auto-dispatched formal training"))
    result = study.run_pilots(dict(base_config=_base()), tmp_path)
    assert len(calls) == 4 and result["actual_steps"] == 8192 and result["probe_steps"] == 48
    assert {(config["ppo"]["learning_rate"], config["actor_output_gain"]) for config in calls} == {
        (lr, gain) for lr in (1e-5, 3e-5) for gain in (.01, .1)}
    assert all(config["seed"] == 42 and config["initial_log_std"] == -.7 for config in calls)
    assert result["validation_evaluated"] is False and result["test_evaluated"] is False


def test_no_eligible_pilot_archives_and_disables_formal(tmp_path, monkeypatch):
    def fake(config, root, candidate, protocol):
        folder = root / "pilots" / candidate["name"]
        folder.mkdir(parents=True)
        study.write_json(folder / "completed.json", dict(mock=True))
        return _pilot(candidate, deterministic_cash_buy_dates=0)
    monkeypatch.setattr(study, "run_pilot", fake)
    with pytest.raises(RuntimeError, match="formal training is disabled"):
        study.run_pilots(dict(base_config=_base()), tmp_path)
    assert study.read_config(tmp_path / "status.json")["status"] == "pilot_failed_closed"


def _diagnostics(directions=(0, 1, 2, 3)):
    n = len(directions)
    return dict(directions=np.asarray(directions), size_latent=np.zeros(n), requested_lots=np.zeros(n),
        selected_lots=np.zeros(n), proposed_shares=np.zeros(n), shares=np.zeros(n), priority=np.zeros(n),
        freeze_mask=np.zeros(n, bool), exit_mask=np.asarray(directions) == 3,
        lot_bounds=np.zeros((n, 4), np.int64), legal_directions=np.ones((n, 4), np.int8), size_active=np.asarray(directions) == 1)


def _order(quantity=100, direction=1, after=100, price=50.):
    return dict(code="000001", filled_shares=quantity, requested_shares=quantity,
        trade_direction=direction, position_after=after, price=price, fee=5., reasons=[], status="filled", opening_equity=1e6)


def test_actual_modes_full_exit_exception_and_voluntary_hold_are_not_requests():
    stats = study.ModeStatistics()
    diagnostics = _diagnostics()
    diagnostics["legal_directions"][0] = [1, 0, 0, 0]
    info = dict(initial_account=dict(mode="cash"), orders=[_order(), _order(-100, 2, 100), _order(-1, 3, 0)])
    stats.observe(diagnostics, info)
    result = stats.snapshot()
    assert result["filled_modes"] == dict(buy=1, reduce=1, exit=1)
    assert result["forced_hold"] == 1 and result["voluntary_hold"] == 0
    assert result["cash_training_buy_fills"] == 1 and result["economic_violations"] == 0
    assert result["actual_full_exit_small_or_odd_exceptions"] == 1
    stats.observe(_diagnostics((0,)), dict(orders=[]))
    assert stats.snapshot()["voluntary_hold"] == 1
    stats.observe(_diagnostics((3,)), dict(orders=[_order(-100, 3, 100)]))
    assert stats.snapshot()["filled_modes"] == dict(buy=1, reduce=2, exit=1)


@pytest.mark.parametrize("order", [_order(1), _order(100, price=49.99), _order(-1, 3, 100),
    _order(-100, 2, 0), _order(100, 2, 100), _order(1100, price=50.)])
def test_fill_validator_rejects_micro_lots_partial_exit_and_opening_budget(order):
    stats = study.ModeStatistics()
    stats.observe(_diagnostics((0,)), dict(orders=[order]))
    assert stats.snapshot()["economic_violations"] == 1


def _trace_case(n=4):
    observation = dict(portfolio=np.append(np.zeros(n), 1.).astype(np.float32),
        valid_mask=np.ones(n, np.float32), buy_mask=np.ones(n, np.float32), sell_mask=np.ones(n, np.float32),
        execution_features=np.zeros((n, 3), np.float32), allocation_context=np.zeros(7, np.float32),
        share_state=np.zeros((n, 2), np.float64), trade_lot_bounds=np.zeros((n, 4), np.int64),
        legal_directions=np.ones((n, 4), np.int8), top_n=np.array([20.], np.float32),
        stock_features=np.ones((n, 1, 20), np.float32))
    diagnostics = _diagnostics(tuple([0] * n))
    book = dict(date=pd.Timestamp("2026-03-02"), codes=[f"{i:06d}" for i in range(n)],
        positions=np.zeros(n), available_positions=np.zeros(n), cash=1e6, receivables=0., equity=1e6, marks=np.ones(n))
    info = dict(date=pd.Timestamp("2026-03-03"), decision_snapshot=book, opening_snapshot=deepcopy(book),
        action_diagnostics=diagnostics, orders=[], cash=1e6, equity=1e6, initial_account=dict(mode="cash"), top_n=20)
    return observation, np.zeros(2 * n, np.float32), info


def test_original_trace_preserves_dtypes_shape_all_steps_and_short_prefix(tmp_path):
    trace = study.TradeTrace(tmp_path / "trace", dict(mode="sampled_training"), chunk_steps=2)
    obs, raw, info = _trace_case()
    for _ in range(2):
        trace.append("2026-03-02", obs, raw, info, log_prob=-2., value=.1, reward=.01, done=False)
    prefix = trace.manifest()
    study.verify_trace_prefix(tmp_path / "trace", prefix)
    for _ in range(2):
        trace.append("2026-03-02", obs, raw, info, log_prob=-2., value=.1, reward=.01, done=False)
    complete = trace.close()
    assert complete["total_steps"] == 4 and len(complete["chunks"]) == 2
    assert complete["chunks"][:1] == prefix["chunks"]
    study.verify_trace_prefix(tmp_path / "trace", prefix)
    study.verify_trace_prefix(tmp_path / "trace", complete)
    with np.load(tmp_path / "trace" / prefix["chunks"][0]["path"], allow_pickle=False) as arrays:
        assert arrays["obs_share_state"].shape == (2, 4, 2) and arrays["obs_share_state"].dtype == np.float64
        assert arrays["obs_trade_lot_bounds"].dtype == np.int64 and arrays["obs_allocation_context"].shape == (2, 7)
        assert arrays["raw_packed_action"].shape == (2, 8) and arrays["action_legal_directions"].shape == (2, 4, 4)
        assert "obs_stock_features" not in arrays.files and arrays["decision_positions"].shape == (2, 4)
        np.testing.assert_array_equal(arrays["step"], [1, 2])


def test_trace_sparse_json_keeps_rejected_reasons_and_normalizes_missing_price(tmp_path):
    trace = study.TradeTrace(tmp_path / "trace", {}, chunk_steps=1)
    obs, raw, info = _trace_case()
    info["orders"] = [dict(code="000000", filled_shares=0, price=float("nan"), reasons=["minimum_trade_notional"], status="rejected")]
    trace("2026-03-02", obs, raw, info)
    trace.close()
    with gzip.open(tmp_path / "trace" / "orders.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 1 and rows[0]["orders"][0]["price"] is None
    assert rows[0]["orders"][0]["reasons"] == ["minimum_trade_notional"]
    assert trace.statistics.snapshot()["rejection_reasons"] == dict(minimum_trade_notional=1)
    with np.load(tmp_path / "trace" / trace.chunks[0]["path"], allow_pickle=False) as arrays:
        assert json.loads(arrays["execution_reasons_json"][0, 0]) == ["minimum_trade_notional"]
        np.testing.assert_array_equal(arrays["execution_filled_shares"], np.zeros((1, 4)))


@pytest.mark.parametrize("mutation", ["book", "nonfinite", "illegal", "micro"])
def test_original_trace_fails_closed_on_unverifiable_books_or_invalid_fills(tmp_path, mutation):
    trace = study.TradeTrace(tmp_path / "trace", {})
    obs, raw, info = _trace_case()
    if mutation == "book":
        info.pop("decision_snapshot")
    elif mutation == "nonfinite":
        raw[0] = float("nan")
    elif mutation == "illegal":
        info["action_diagnostics"]["legal_directions"][0, 0] = 0
    else:
        info["orders"] = [_order(1)]
    with pytest.raises((ValueError, FloatingPointError)):
        trace("2026-03-02", obs, raw, info)


def _records():
    rows = []
    for budget, steps in study.BUDGETS.items():
        for index, seed in enumerate(study.SEEDS):
            value = .18 + index * .02 + (.06 if budget == "long" else 0.)
            scenarios = {}
            for account, offset in zip(study.MODES, [-.03, -.01, .01, .03]):
                def metric(value):
                    return dict(cumulative_return=value + offset, max_drawdown=-.04, total_fees=100.,
                                total_slippage_cost=50., mean_cash_ratio=.5)
                scenarios[account] = dict(ppo=metric(value), initialized=metric(.08),
                    baselines=dict(momentum=metric(.25), momentum_daily=metric(.17), cash=metric(0.)))
            rows.append(dict(size="small", budget=budget, seed=seed, steps=steps, hidden_sizes=[256, 128],
                parameters=126343, metrics=scenarios, training_run_id=f"small_{seed}", prefix_shared=True,
                model_path=f"fixture/{budget}/{seed}", model_sha256=f"fixture-{budget}-{seed}",
                auxiliary_metrics={str(k): {name: item["ppo"] for name, item in scenarios.items()} for k in (10, 40)},
                test_evaluated=False))
    references = {}
    for mode, adjustment in [("prior", -.02), ("random", -.05), ("delta", -.10)]:
        previous = deepcopy(rows)
        for row in previous:
            if row["budget"] == "long":
                row["steps"] = 65536
            for scenario in row["metrics"].values():
                scenario["ppo"]["cumulative_return"] += adjustment
        references[mode] = dict(records=previous)
    return rows, references


def test_new_budget_arithmetic_same_seed_8192_pairing_and_long_is_not_old_paired():
    rows, references = _records()
    summary = study.summarize_study(rows, references)
    short, long = summary["cells"]["short"], summary["cells"]["long"]
    assert short["metrics"]["cumulative_return"]["mean"] == pytest.approx(.20)
    assert short["metrics"]["cumulative_return"]["std"] == pytest.approx(.02)
    assert short["return_gaps"]["initialized"]["mean"] == pytest.approx(.12)
    assert short["same_budget_legacy_return_gaps"]["prior"]["mean"] == pytest.approx(.02)
    assert long["same_budget_legacy_return_gaps"] == {}
    assert long["descriptive_legacy_return_gaps"]["prior_65536"]["mean"] == pytest.approx(.02)
    assert short["per_account"]["cash"]["cumulative_return"]["mean"] == pytest.approx(.17)
    assert summary["paired_long_minus_short"]["mean"] == pytest.approx(.06)
    assert summary["actual_formal_steps"] == 98304 and summary["checkpoint_steps_sum"] == 122880
    assert summary["qualified_candidate"]["budget"] == "long"
    assert summary["qualified_candidate"]["representative_seed"] == 42
    assert summary["qualified_candidate"]["model_sha256"] == "fixture-long-42"
    assert summary["actionable"] is False and summary["test_evaluated"] is False


@pytest.mark.parametrize("change,message", [("missing", "three seeds"), ("duplicate", "three seeds"),
    ("budget", "budget"), ("capacity", "capacity"), ("test", "split"), ("baseline", "baseline values"),
    ("cost", "total-cost")])
def test_summary_rejects_incomplete_and_incomparable_evidence(change, message):
    rows, references = _records()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif change == "budget":
        rows[0]["steps"] += 512
    elif change == "capacity":
        rows[0]["parameters"] += 1
    elif change == "test":
        rows[0]["test_evaluated"] = True
    elif change == "cost":
        for scenario in rows[0]["metrics"].values():
            scenario["ppo"].pop("total_fees")
    else:
        rows[0]["metrics"]["cash"]["baselines"]["momentum"]["cumulative_return"] += .01
    with pytest.raises(ValueError, match=message):
        study.summarize_study(rows, references)


def test_candidate_gate_is_daily_momentum_not_old_prior_or_auxiliary_lucky_k():
    rows, references = _records()
    for row in rows:
        for scenario in row["metrics"].values():
            scenario["baselines"]["momentum_daily"]["cumulative_return"] += 1.
    for reference in references.values():
        for row in reference["records"]:
            for scenario in row["metrics"].values():
                scenario["baselines"]["momentum_daily"]["cumulative_return"] += 1.
    summary = study.summarize_study(rows, references)
    assert summary["qualified_candidate"] is None
    assert all(not cell["gates"]["mean_return_above_daily_momentum"] for cell in summary["cells"].values())
    assert summary["auxiliary_used_for_selection"] is False


def test_candidate_cost_tie_and_fixed_seed42_never_closest_mean():
    rows, references = _records()
    for row in rows:
        if row["budget"] == "long":
            for scenario in row["metrics"].values():
                scenario["ppo"]["cumulative_return"] -= .06
                scenario["ppo"]["total_fees"] -= 1
    summary = study.summarize_study(rows, references)
    assert summary["qualified_candidate"]["budget"] == "long"
    assert summary["qualified_candidate"]["representative_seed"] == 42


@pytest.mark.parametrize("relation", ["same", "ancestor", "descendant"])
def test_output_path_cannot_overwrite_or_contain_reference(tmp_path, relation, monkeypatch):
    root = tmp_path / "study"
    reference = root if relation == "same" else tmp_path if relation == "ancestor" else root / "old"
    monkeypatch.setattr(study, "ResearchData", lambda *args: pytest.fail("Path protection must precede data/model work"))
    with pytest.raises(ValueError, match="paths must not contain"):
        study.prepare_study(_base(), root, dict(prior=reference, random=tmp_path / "r", delta=tmp_path / "d"))


def test_direct_sigma_uses_network_parameter_not_joint_entropy_or_2n(tmp_path, monkeypatch):
    data = SimpleNamespace(codes=["000001"], config=dict(validation_rollouts=10000, ppo=dict(n_steps=512)),
        contract=dict(artifacts={}), settings=dict(broker=dict(lot_size=100, min_commission=5., commission=.0003)))
    callback = study.TradeCallback(data, tmp_path, None, {}, dict(pilot=2048), tmp_path / "status.json", "pilot")
    logger = MagicMock()
    logger.name_to_value = {"train/entropy_loss": 999999.}
    callback.model = SimpleNamespace(policy=SimpleNamespace(network=SimpleNamespace(log_std=torch.tensor([-.7]))), logger=logger)
    monkeypatch.setattr(study.MatrixCallback, "record_update", lambda self: None)
    callback.record_update()
    sigma = next(call.args[1] for call in logger.record.call_args_list if call.args[0] == "train/conditional_size_sigma")
    assert sigma == pytest.approx(math.exp(-.7))


def test_trace_prefix_hash_rejects_tampering(tmp_path):
    trace = study.TradeTrace(tmp_path / "trace", {}, chunk_steps=1)
    trace("2026-03-02", *_trace_case())
    manifest = trace.close()
    path = tmp_path / "trace" / manifest["chunks"][0]["path"]
    path.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="prefix changed"):
        study.verify_trace_prefix(tmp_path / "trace", manifest)


def test_fatal_account_invariant_stops_pilots_without_using_other_configuration(tmp_path, monkeypatch):
    calls = []
    def fake(config, root, candidate, protocol):
        calls.append(candidate["name"])
        folder = root / "pilots" / candidate["name"]
        folder.mkdir(parents=True)
        study.write_json(folder / "completed.json", dict(mock=True))
        return _pilot(candidate, status="error", fatal_invariant_failure=True)
    monkeypatch.setattr(study, "run_pilot", fake)
    with pytest.raises(RuntimeError, match="formal training is disabled"):
        study.run_pilots(dict(base_config=_base()), tmp_path)
    assert len(calls) == 1
    result = study.read_config(tmp_path / "pilot_selection.json")
    assert result["selected"] is None and len(result["reports"]) == 1


def test_one_missing_account_cost_and_nonshared_prefix_are_rejected():
    rows, references = _records()
    rows[0]["metrics"]["cash"]["ppo"].pop("total_fees")
    with pytest.raises(ValueError, match="total-cost"):
        study.summarize_study(rows, references)
    rows, references = _records()
    rows[0]["prefix_shared"] = False
    with pytest.raises(ValueError, match="training prefix"):
        study.summarize_study(rows, references)


def _local_trade(local_data):
    config = deepcopy(local_data.config)
    config.update(action_mode="stock_trade", prior_feature_index=None, device="cpu", top_n=2,
                  top_n_choices=[1, 2], max_trade_weight=.05, max_commission_ratio=.001,
                  actor_output_gain=.01, initial_log_std=-.7)
    config.pop("action_top_k", None)
    return study.ResearchData(config)


def test_cpu_native_evaluation_trace_has_identical_accounting_and_no_parameter_updates(local_data, tmp_path):
    from src.backtest.run import run_policy
    data = _local_trade(local_data)
    env = data.env("train")
    model = study.make_model(env, data.config)
    agent = study.PPOAgent(model, data.contract)
    before = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
    data.settings.update(seed=study.PROBE_SEED, initial_portfolio="random", trade_top_n=2)
    plain = run_policy(data.market("val"), agent, data.settings)
    trace = study.TradeTrace(tmp_path / "native", study._trace_identity(data, "deterministic_development"), chunk_steps=7)
    traced = run_policy(data.market("val"), agent, data.settings, trace_sink=trace)
    manifest = trace.close()
    for expected, actual in zip(plain[:3], traced[:3]):
        pd.testing.assert_frame_equal(expected, actual)
    assert plain[3] == traced[3] and manifest["total_steps"] == len(traced[0]) - 1
    assert manifest["mode_statistics"]["economic_violations"] == 0
    study.verify_trace_prefix(tmp_path / "native", manifest)
    for chunk in manifest["chunks"]:
        with np.load(tmp_path / "native" / chunk["path"], allow_pickle=False) as arrays:
            assert arrays["obs_allocation_context"].shape[1] == 7
            assert arrays["obs_share_state"].dtype == np.float64
            assert arrays["decision_positions"].shape[1] == len(data.codes)
    for name, tensor in model.policy.state_dict().items():
        assert torch.equal(before[name], tensor)
    assert not model.policy.optimizer.state and model.num_timesteps == 0
    env.close()


def test_cpu_probe_two_date_provider_resets_account_at_correct_training_decision(local_data):
    data = _local_trade(local_data)
    market = data.market("train")
    index = market.n_dates - 2
    for mode in ("cash", "random"):
        env = study._probe_env(data, market, index, mode)
        try:
            observation, initial = env.reset(seed=study.PROBE_SEED)
            assert initial["date"] == market.dates[index]
            assert env.unwrapped.market.n_dates == 2
            assert env.unwrapped.market.panel.index.max() == market.dates[index + 1]
            assert observation["top_n"][0] == 20
            model = study.make_model(env, data.config)
            agent = study.PPOAgent(model, data.contract)
            before = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
            action = agent.act(observation)
            _, _, terminated, truncated, info = env.step_order(action)
            assert terminated and not truncated
            assert info["date"] <= pd.Timestamp(data.manifest["train_end"])
            stats = study.ModeStatistics(lot_size=data.contract["broker"]["lot_size"], min_commission=data.contract["broker"]["min_commission"])
            stats.observe(action.diagnostics, info)
            assert stats.violations == 0
            for name, tensor in model.policy.state_dict().items():
                assert torch.equal(before[name], tensor)
        finally:
            env.close()


def test_callback_records_original_sb3_rollout_actions_log_probs_and_exact_books_without_updates(local_data, tmp_path):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.training.matrix_experience import EpisodeAudit
    data = _local_trade(local_data)
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    model = study.make_model(env, data.config)
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.policy.state_dict().items()}
    callback = study.TradeCallback(data, tmp_path, experience, initial, dict(pilot=2048), tmp_path / "status.json", "fixture")
    _, initialized_callback = model._setup_learn(8, callback)
    initialized_callback.on_training_start({}, {})
    try:
        assert model.collect_rollouts(env, initialized_callback, model.rollout_buffer, n_rollout_steps=8)
        callback.record_update()
        assert np.isfinite(model.logger.name_to_value["train/post_update_categorical_entropy"])
        assert np.isfinite(model.logger.name_to_value["train/post_update_conditional_size_entropy"])
        assert model.logger.name_to_value["train/conditional_size_sigma"] == pytest.approx(math.exp(-.7))
        manifest = callback.trace.close()
        assert manifest["total_steps"] == 8 and experience.stats()["total_steps"] == 8
        assert manifest["mode_statistics"]["economic_violations"] == 0
        with np.load(tmp_path / "training_trace" / manifest["chunks"][0]["path"], allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["raw_packed_action"], model.rollout_buffer.actions[:, 0])
            np.testing.assert_allclose(arrays["log_prob"], model.rollout_buffer.log_probs[:, 0], atol=1e-6)
            assert arrays["decision_positions"].dtype == np.float64
            assert arrays["obs_trade_lot_bounds"].dtype == np.int64
            assert arrays["action_directions"].shape == (8, len(data.codes))
        for name, tensor in model.policy.state_dict().items():
            assert torch.equal(initial[name], tensor.detach().cpu())
        assert not model.policy.optimizer.state and not callback.saved
    finally:
        env.close()


def test_native_evaluation_trace_adapter_persists_actual_state_and_keeps_model_untrained(local_data, tmp_path):
    data = _local_trade(local_data)
    data.settings.update(seed=study.PROBE_SEED, initial_portfolio="cash", trade_top_n=2)
    env = data.env("train")
    agent = study.PPOAgent(study.make_model(env, data.config), data.contract)
    before = {name: tensor.detach().clone() for name, tensor in agent.model.policy.state_dict().items()}
    try:
        metrics, manifest = study._evaluate_trace(agent, data, tmp_path / "case")
        assert "ppo" in metrics and manifest["identity"]["top_n"] == 2
        assert manifest["total_steps"] == metrics["ppo"]["trading_steps"]
        assert (tmp_path / "case" / "ppo" / "orders.csv").exists()
        study.verify_trace_prefix(tmp_path / "case_trace", manifest)
        for name, tensor in agent.model.policy.state_dict().items():
            assert torch.equal(before[name], tensor)
        assert agent.model.num_timesteps == 0 and not agent.model.policy.optimizer.state
    finally:
        env.close()


def test_pilot_cache_cannot_silently_drop_native_training_or_probe_registration(tmp_path):
    candidate = study.PILOT_CANDIDATES[0]
    directory = tmp_path / candidate["name"]
    directory.mkdir()
    config = study.config_for(_base(), tmp_path, candidate=candidate)
    protocol = {"registered": True}
    report = _pilot(candidate)
    study.write_json(directory / "report.json", report)
    study.write_json(directory / "completed.json", dict(config_sha256=study.json_digest(config),
        protocol_sha256=study.json_digest(protocol), report=report, files=study._registered_files(directory)))
    with pytest.raises(ValueError, match="required registered training/probe"):
        study._load_pilot(directory, config, protocol)


def test_initial_control_cache_binds_seed_model_hash_four_accounts_and_native_trace_indexes(tmp_path, monkeypatch):
    initial = tmp_path / "initial_model"
    initial.mkdir()
    (initial / "model.zip").write_bytes(b"untrained fixture")
    data = SimpleNamespace(contract={"action_mode": "stock_trade"}, config={"seed": 123})
    directory = tmp_path / "controls" / "small_123"
    metrics = {mode: dict(ppo=dict(cumulative_return=0.)) for mode in study.MODES}
    for mode in study.MODES:
        (directory / mode / "ppo").mkdir(parents=True)
        study.write_json(directory / mode / "metrics.json", metrics[mode])
        for name in ("daily.csv", "orders.csv", "positions.parquet"):
            (directory / mode / "ppo" / name).write_bytes(b"sealed fixture")
        (directory / (mode + "_trace")).mkdir()
        study.write_json(directory / (mode + "_trace") / "index.json", {"registered": True})
    identity = dict(contract=data.contract, training_seed=123, evaluation_seed=study.PROBE_SEED,
        evaluation_modes=list(study.MODES), top_n=20, initial_model_sha256=study.file_hash(initial / "model.zip"))
    files = study._registered_files(directory)
    study.write_json(directory / "controls.json", dict(identity=identity, metrics=metrics, files=files))
    monkeypatch.setattr(study.PPOAgent, "load", lambda *args: pytest.fail("Valid cache must not reload or reevaluate"))
    assert study.evaluate_initial(data, initial, tmp_path, "small_123") == metrics
    files.pop("cash_trace/index.json")
    study.write_json(directory / "controls.json", dict(identity=identity, metrics=metrics, files=files))
    with pytest.raises(ValueError, match="registered account metrics/traces"):
        study.evaluate_initial(data, initial, tmp_path, "small_123")


def test_prepare_cli_does_not_require_cuda_or_automatically_train(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    study.write_json(config, _base())
    prepared = {"test_evaluated": False}
    calls = []
    def fake(base, root, references):
        calls.append((root, references))
        return prepared
    monkeypatch.setattr(study, "prepare_study", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Prepare must not initialize CUDA"))
    monkeypatch.setattr(study, "run_pilots", lambda *args: pytest.fail("Prepare cannot run pilots"))
    monkeypatch.setattr(study, "run_formal", lambda *args: pytest.fail("Prepare cannot run formal"))
    result = study.main(["--stage", "prepare", "--config", str(config), "--output", str(tmp_path / "new"),
        "--prior-root", str(tmp_path / "prior"), "--random-root", str(tmp_path / "random"), "--delta-root", str(tmp_path / "delta")])
    assert result is prepared and set(calls[0][1]) == {"prior", "random", "delta"}
