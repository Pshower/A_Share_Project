"""Synthetic glue checks only: no real provider, labels, PPO or downloads."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools import research_counterfactual_learning as runner
from tools.counterfactual_exact_schedule import ExactPhaseSampler


def metrics(roi=.02, drawdown=-.04):
    return {mode: dict(cumulative_return=roi, max_drawdown=drawdown,
                       total_fees=100., total_slippage_cost=120.) for mode in runner.MODES}


def set_gains(item, gap, ppo_gain=.002):
    benchmark_return = item["benchmark"][runner.MODES[0]]["cumulative_return"]
    for mode in runner.MODES:
        item["trained"][mode]["cumulative_return"] = benchmark_return + gap
    item["initialized"] = deepcopy(item["benchmark"])
    item["auxiliary"] = deepcopy(item["benchmark"])
    if item["method"] != "ppo_only":
        for mode in runner.MODES:
            item["auxiliary"][mode]["cumulative_return"] = benchmark_return + gap - ppo_gain
    item["baseline_gap"] = item["initial_gain"] = gap
    item["ppo_gain"] = gap if item["method"] == "ppo_only" else ppo_gain


def row(fold, method, steps, seed=42, gap=.006, ppo_gain=.002):
    item = dict(fold=fold, phase="screen" if fold in runner.SCREEN else "confirm",
                method=method, steps=steps, seed=seed, actionable=False,
                trained=metrics(), benchmark=metrics(.014, -.03),
                changes={mode: dict(greedy_changed=True, economic_ledger_changed=True) for mode in runner.MODES},
                changes_vs_aux={mode: dict(greedy_changed=True, economic_ledger_changed=True) for mode in runner.MODES})
    set_gains(item, gap, ppo_gain)
    return item


def screen_rows(gap=.006):
    return [row(fold, method, steps, gap=gap) for fold in runner.SCREEN
            for method in runner.METHODS for steps in runner.SNAPSHOTS]


def confirmation_rows(budget=8192):
    return [row(fold, method, budget, seed) for fold in runner.CONFIRM
            for method in runner.METHODS for seed in runner.SEEDS]


def gate():
    return dict(across_fold_mean_baseline_gap_min=.005, max_all_account_drawdown=.10,
                max_drawdown_excess_over_baseline=.02, each_fold_positive_gap_seeds=2)


def selection(budget=8192, method="value_warmup"):
    return dict(admitted=True, selected=dict(method=method, steps=budget))


class FakeMarket:
    def __init__(self):
        self.dates = pd.bdate_range("2017-01-02", periods=420)
        self.current_index = 0

    @property
    def current_date(self):
        return self.dates[self.current_index]


class FakeEnv:
    def __init__(self, fail_step=None):
        self.env = SimpleNamespace(env=SimpleNamespace(market=FakeMarket(), _steps=0, max_steps=80))
        self.benchmark = SimpleNamespace(env=SimpleNamespace(market=FakeMarket(), _steps=0, max_steps=80))
        self.sampler = ExactPhaseSampler(self.env.env.market.dates, 80)
        self.phase, self.episode = 0, -1
        self._done, self.closed = True, False
        self.reset_count = 0
        self.actions = []
        self.episode_actions = []
        self.fail_step = fail_step
        self.contract = dict(stock_codes=["000001"], broker=dict(lot_size=100))

    def reset(self, seed=None):
        if seed is not None:
            self.sampler.seed(seed)
            self.episode = -1
        schedule = self.sampler.next()
        self.episode += 1
        self.reset_count += 1
        self.phase = schedule["phase"]
        self._done = False
        self.episode_actions = []
        for base in (self.env.env, self.benchmark.env):
            base._steps = 0
            base.market.current_index = schedule["start_index"]
        return self.observe(), dict(timing=dict(schedule=deepcopy(schedule)))

    def observe(self):
        observation = np.zeros(129, dtype=np.float32)
        observation[self.phase] = 1.
        observation[20:24] = [self.episode / 240, self.env.env._steps / 80,
                              self.env.env.market.current_index / 420, .5]
        return observation

    def step(self, action):
        assert action in (0, 1)
        self.actions.append((self.episode, self.env.env._steps, self.phase, action))
        self.episode_actions.append(action)
        self.env.env._steps += 1
        self.env.env.market.current_index += 1
        if self.fail_step is not None and len(self.actions) == self.fail_step:
            raise RuntimeError("Synthetic agent advanced; shadow did not")
        self.benchmark.env._steps += 1
        self.benchmark.env.market.current_index += 1
        self.phase = (self.phase + 1) % 20
        return self.observe(), 0., False, False, {}

    def close(self):
        self.closed = True


def fake_fingerprint(env):
    return runner.digest(dict(episode=env.episode, phase=env.phase,
        agent_steps=env.env.env._steps, shadow_steps=env.benchmark.env._steps,
        agent_index=env.env.env.market.current_index,
        shadow_index=env.benchmark.env.market.current_index,
        actions=env.episode_actions, source_behavior=env.counterfactual_source_behavior,
        source_rng=env.counterfactual_source_rng.bit_generator.state))


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(runner.paired, "snapshot_state_fingerprint", fake_fingerprint)
    return FakeEnv


def make_plan(source):
    env = source()
    counts = runner.teacher_counts()
    states = [record for record, _ in runner.collection(env, "2023", "inner_train", counts)]
    return dict(fold="2023", segment="inner_train", train_end="2021-12-31", source_start_min="2017-01-02",
                states=states, data_identity=dict(boundaries=dict(
                    inner_train_end="2021-12-31", inner_holdout_start="2022-02-01",
                    inner_holdout_end="2023-03-31")))


def test_registration_has_exact_physical_budgets_and_no_offline_cancellation():
    spec = runner.registration()
    assert spec["teacher_episodes_per_segment"] == 240
    assert spec["teacher_states_per_segment"] == 480
    assert spec["teacher_offsets"] == [0, 60]
    assert spec["teacher_passes"] == 2
    per_segment = runner.expected_teacher_counts()
    assert per_segment["collection_agent_steps"] == per_segment["collection_benchmark_steps"] == 2 * 240 * 60
    assert per_segment["branch_agent_steps"] == per_segment["branch_benchmark_steps"] == 480 * 2 * 20
    books = sum(per_segment[key] for key in ("collection_agent_steps", "collection_benchmark_steps",
                                            "branch_agent_steps", "branch_benchmark_steps"))
    assert spec["teacher_book_steps_screen"] == 2 * 2 * books == 384000
    assert spec["ppo_screen_steps"] == spec["ppo_shadow_screen_steps"] == 4 * 32768 == 131072
    assert spec["snapshots"] == [8192, 32768]
    assert spec["teacher_source"] == "calendar_initialized_bernoulli_v1"
    assert spec["source_action_salt"] == 173701
    assert spec["offline_gate_cancels_ppo"] is False
    assert spec["test_evaluated"] is spec["actionable"] is False


def test_collection_replays_exact_analytic_bernoulli_source_and_two_pass_costs(source):
    left, right = source(), source()
    first_counts, second_counts = runner.teacher_counts(), runner.teacher_counts()
    first = [record for record, _ in runner.collection(left, "2023", "inner_train", first_counts)]
    second = [record for record, _ in runner.collection(right, "2023", "inner_train", second_counts)]
    assert runner.canonical(first) == runner.canonical(second)
    assert left.actions == right.actions
    assert len(first) == 480 and len(left.actions) == 14400
    assert first_counts == second_counts
    assert first_counts["collection_agent_steps"] == first_counts["collection_benchmark_steps"] == 14400
    assert first_counts["collection_macro_attempts"] == 14400
    assert sum(first_counts[key] for key in ("branch_agent_steps", "branch_benchmark_steps", "completed_pairs")) == 0
    different_from_calendar = 0
    for episode in range(240):
        rng = np.random.default_rng(np.random.SeedSequence([42, 173701, episode]))
        actions = left.actions[episode * 60:(episode + 1) * 60]
        for actual_episode, offset, phase, action in actions:
            probability = 1 / (1 + np.exp(-2. if phase == 0 else 2.))
            assert actual_episode == episode
            assert action == int(rng.random() < probability)
            different_from_calendar += action != int(phase == 0)
        assert [item[1] for item in actions] == list(range(60))
    assert different_from_calendar > 1000
    runner.validate_plan(make_plan(source))


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate_cell", "wrong_fold", "wrong_segment", "wrong_episode", "wrong_offset",
    "wrong_bin", "short_input", "nan_input", "endpoint_after_cutoff", "backward_endpoint",
    "unknown_fold", "unknown_segment",
])
def test_validate_plan_rejects_incomplete_or_off_domain_records(source, mutation):
    plan = make_plan(source)
    record = plan["states"][0]
    if mutation == "missing":
        plan["states"].pop()
    elif mutation == "duplicate_cell":
        for index in (0, 1):
            for key in ("initial_mode", "date_stratum", "phase", "phase_bin"):
                plan["states"][index][key] = plan["states"][index + 2][key]
    elif mutation == "wrong_fold":
        record["fold"] = "2024"
    elif mutation == "wrong_segment":
        record["segment"] = "inner_holdout"
    elif mutation == "wrong_episode":
        record["episode"] = 999
    elif mutation == "wrong_offset":
        record["offset"] = 59
    elif mutation == "wrong_bin":
        record["phase_bin"] = (record["phase_bin"] + 1) % 4
    elif mutation in ("short_input", "nan_input"):
        if mutation == "short_input":
            record["observation"].pop()
        else:
            record["observation"][-1] = float("nan")
        record["input_sha256"] = "0" * 64
    elif mutation == "endpoint_after_cutoff":
        record["endpoint"] = "2022-01-03"
    elif mutation == "backward_endpoint":
        record["endpoint"] = "2010-01-01"
    elif mutation == "unknown_fold":
        plan["fold"] = "2030"
    elif mutation == "unknown_segment":
        plan["segment"] = "test"
    with pytest.raises((ValueError, AssertionError, TypeError)):
        runner.validate_plan(plan)


def test_collection_accounts_asymmetric_failure_without_claiming_rollback(source):
    env = source(fail_step=1)
    counts = runner.teacher_counts()
    iterator = runner.collection(env, "2023", "inner_train", counts)
    next(iterator)
    with pytest.raises(RuntimeError, match="shadow did not"):
        next(iterator)
    assert counts["collection_agent_steps"] == counts["collection_macro_attempts"] == 1
    assert counts["collection_benchmark_steps"] == 0
    assert env.reset_count == 1


@pytest.fixture
def teacher_workspace(tmp_path, monkeypatch, source):
    created = []

    class Data:
        identity = dict(name="synthetic-data", boundaries=dict(
            inner_train_end="2021-12-31", inner_holdout_start="2022-02-01", inner_holdout_end="2023-03-31"))
        boundaries = identity["boundaries"]

        def teacher_env(self, segment):
            assert segment == "inner_train"
            env = source()
            created.append(env)
            return env

    root = tmp_path / "study"
    root.mkdir()
    directory = root / "teacher/2023/inner_train"
    calls = []

    def label(state, *, horizon, train_end, gamma):
        assert horizon == 20 and train_end == "2021-12-31" and gamma == .995
        seal = runner.read_json(directory / "sampling_plan_seal.json")
        plan = runner.read_json(directory / "sampling_plan.json")
        assert seal["states"] == len(plan["states"]) == 480
        assert seal["labels_started"] is False
        assert all(runner.file_hash(directory / name) == sha for name, sha in seal["files"].items())
        with np.load(directory / "inputs.npz", allow_pickle=False) as archive:
            assert archive["observations"].shape == (480, 129)
        calls.append((state.episode, state.env.env._steps))
        return dict(labels=dict(discounted_relative_delta=state.phase * .0001),
            counters=dict(branch_agent_steps=40, branch_benchmark_steps=40, macro_attempts=40))

    monkeypatch.setattr(runner.paired, "paired_rule_label", label)
    monkeypatch.setattr(runner, "validate_pair", lambda label, *args: label["labels"]["discounted_relative_delta"])
    return SimpleNamespace(root=root, directory=directory, data=Data(), created=created, calls=calls,
                           protocol=dict(study="synthetic-frozen"), label=label)


def test_teacher_whole_segment_freezes_all_inputs_before_labels_and_reuses_only_seal(teacher_workspace):
    workspace = teacher_workspace
    result = runner.teacher_segment(workspace.root, workspace.protocol, workspace.data, "2023", "inner_train")
    assert result == workspace.directory
    assert len(workspace.created) == 2 and len(workspace.calls) == 480
    assert all(env.closed and env.reset_count == 240 for env in workspace.created)
    assert len(list((result / "labels").glob("*.json.gz"))) == 480
    assert runner.read_json(result / "counters.json") == runner.expected_teacher_counts()
    X, y = runner.teacher_arrays(result, workspace.protocol)
    assert X.shape == (480, 129) and X.dtype == np.float32
    assert y.shape == (480,) and np.isfinite(y).all()
    assert runner.closed(result, workspace.protocol)["counts"]["completed_pairs"] == 480
    assert runner.teacher_segment(workspace.root, workspace.protocol, workspace.data, "2023", "inner_train") == result
    assert len(workspace.created) == 2 and len(workspace.calls) == 480


def test_teacher_partial_failure_keeps_actual_asymmetric_costs_and_never_replaces(teacher_workspace, monkeypatch):
    workspace = teacher_workspace

    def failure(state, **kwargs):
        if len(workspace.calls) == 1:
            raise runner.paired.CounterfactualLabelError("synthetic partial branch", counters=dict(
                branch_agent_steps=1, branch_benchmark_steps=0, macro_attempts=1), evidence=dict(partial=True))
        return workspace.label(state, **kwargs)

    monkeypatch.setattr(runner.paired, "paired_rule_label", failure)
    with pytest.raises(runner.paired.CounterfactualLabelError):
        runner.teacher_segment(workspace.root, workspace.protocol, workspace.data, "2023", "inner_train")
    failure_record = runner.read_json(workspace.directory / "failure.json")
    assert failure_record["no_replacement"] is True
    counts = failure_record["counts"]
    assert counts["collection_agent_steps"] == counts["collection_benchmark_steps"] == 14460
    assert counts["collection_macro_attempts"] == 14460
    assert counts["branch_agent_steps"] == counts["branch_macro_attempts"] == 41
    assert counts["branch_benchmark_steps"] == 40
    assert counts["completed_pairs"] == 1
    assert not (workspace.directory / "completed.json").exists()
    assert len(list((workspace.directory / "labels").glob("*.json.gz"))) == 1
    assert len(workspace.created) == 2 and all(env.closed for env in workspace.created)
    with pytest.raises(ValueError, match="partial"):
        runner.teacher_segment(workspace.root, workspace.protocol, workspace.data, "2023", "inner_train")
    assert len(workspace.created) == 2 and len(workspace.calls) == 1


def test_teacher_arrays_rejects_nonfinite_inputs_even_in_a_synthetic_seal(tmp_path):
    protocol = dict(study="synthetic")
    X = np.zeros((480, 129), dtype=np.float32)
    X[4, 22] = np.nan
    np.savez_compressed(tmp_path / "inputs.npz", observations=X)
    np.savez_compressed(tmp_path / "targets.npz", delta=np.zeros(480))
    runner.seal(tmp_path, protocol)
    with pytest.raises(ValueError):
        runner.teacher_arrays(tmp_path, protocol)


def test_screen_filters_learning_eligibility_before_ranking_and_has_no_fallback():
    rows = screen_rows()
    for item in rows:
        if item["method"] == "value_warmup" and item["steps"] == 32768:
            set_gains(item, .20, ppo_gain=-.001)
        elif item["method"] == "value_warmup" and item["steps"] == 8192:
            set_gains(item, .01)
    selected = runner.select_screen(rows)
    assert selected["selected"]["method"] == "value_warmup"
    assert selected["selected"]["steps"] == 8192
    assert selected["confirmation_rl_steps_authorized"] == 12 * 8192
    assert len(selected["candidates"]) == 4
    assert not next(c for c in selected["candidates"] if c["method"] == "value_warmup" and c["steps"] == 32768)["eligible"]
    for item in rows:
        set_gains(item, 0., ppo_gain=0.)
    failed = runner.select_screen(rows)
    assert failed["admitted"] is False and failed["selected"] is None
    assert failed["confirmation_rl_steps_authorized"] == 0
    assert failed["no_confirmation_fallback"] is True


def test_screen_exact_one_basis_point_uses_only_registered_roundoff_tolerance():
    rows = screen_rows(gap=.0001)
    for item in rows:
        for mode in runner.MODES:
            item["benchmark"][mode]["cumulative_return"] = .02
        set_gains(item, .0001, ppo_gain=.00005)
    assert runner.select_screen(rows)["admitted"] is True
    for item in rows:
        set_gains(item, .000099, ppo_gain=.00005)
    assert runner.select_screen(rows)["admitted"] is False


@pytest.mark.parametrize("field", ["baseline_gap", "initial_gain", "ppo_gain"])
def test_screen_recomputes_stored_learning_gains_from_native_metrics(field):
    rows = screen_rows()
    rows[0][field] += .01
    with pytest.raises(ValueError, match="gap"):
        runner.select_screen(rows)


def test_screen_requires_real_greedy_and_economic_changes_not_only_metrics():
    rows = screen_rows()
    for item in rows:
        for mode in runner.MODES:
            item["changes_vs_aux"][mode]["economic_ledger_changed"] = False
    assert runner.select_screen(rows)["admitted"] is False
    rows[0]["changes"][runner.MODES[0]]["greedy_changed"] = 1
    with pytest.raises(ValueError, match="behavior"):
        runner.select_screen(rows)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "off_domain"])
def test_screen_requires_every_development_snapshot_once(mutation):
    rows = screen_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    else:
        rows[0]["seed"] = 123
    with pytest.raises(ValueError):
        runner.select_screen(rows)


@pytest.mark.parametrize("budget", [8192, 32768])
def test_confirmation_requires_both_methods_every_fold_and_every_seed(budget):
    rows = confirmation_rows(budget)
    result = runner.confirmation_summary(rows, selection(budget), gate())
    assert result["qualified"] is True
    assert set(result["per_fold"]) == set(runner.CONFIRM)
    assert all(info["positive_gap_seeds"] == info["positive_ppo_seeds"] == 3 for info in result["per_fold"].values())
    missing = [item for item in rows if not (item["fold"] == "2025" and item["method"] == "ppo_only"
                                            and item["seed"] == 123)]
    with pytest.raises(ValueError):
        runner.confirmation_summary(missing, selection(budget), gate())


@pytest.mark.parametrize("mutation", ["duplicate", "extra_seed", "extra_fold", "extra_method", "unauthorized_clock"])
def test_confirmation_rejects_duplicates_and_off_domain_rows(mutation):
    rows = confirmation_rows()
    extra = deepcopy(rows[0])
    if mutation == "extra_seed":
        extra["seed"] = 7
    elif mutation == "extra_fold":
        extra["fold"] = "2024"
    elif mutation == "extra_method":
        extra["method"] = "posthoc"
    elif mutation == "unauthorized_clock":
        extra["steps"] = 16384
    rows.append(extra)
    with pytest.raises(ValueError):
        runner.confirmation_summary(rows, selection(), gate())


def test_confirmation_risk_boundaries_and_positive_ppo_requirement():
    rows = confirmation_rows()
    for item in rows:
        set_gains(item, .005)
        for mode in runner.MODES:
            item["trained"][mode]["max_drawdown"] = -.10
            item["benchmark"][mode]["max_drawdown"] = -.08
        item["initialized"] = deepcopy(item["benchmark"])
        if item["method"] == "ppo_only":
            item["auxiliary"] = deepcopy(item["benchmark"])
    exact = runner.confirmation_summary(rows, selection(), gate())
    assert exact["qualified"] is True
    for gap, ppo_gain in ((.0049, .002), (0., .002), (.005, 0.)):
        changed = deepcopy(rows)
        for item in changed:
            if item["method"] == "value_warmup":
                set_gains(item, gap, ppo_gain)
        assert runner.confirmation_summary(changed, selection(), gate())["qualified"] is False
    changed = deepcopy(rows)
    changed[3]["trained"][runner.MODES[0]]["max_drawdown"] = -.10001
    assert runner.confirmation_summary(changed, selection(), gate())["qualified"] is False
    changed = deepcopy(rows)
    for item in changed:
        if item["method"] == "value_warmup":
            for mode in runner.MODES:
                item["benchmark"][mode]["max_drawdown"] = -.07999
            item["initialized"] = deepcopy(item["benchmark"])
    assert runner.confirmation_summary(changed, selection(), gate())["qualified"] is False


@pytest.mark.parametrize("kind", ["baseline", "ppo"])
def test_confirmation_cannot_qualify_from_one_lucky_seed(kind):
    rows = confirmation_rows()
    for item in rows:
        if item["method"] == "value_warmup" and item["fold"] == "2025":
            lucky = item["seed"] == runner.SEEDS[0]
            if kind == "baseline":
                set_gains(item, .03 if lucky else -.001, ppo_gain=.002)
            else:
                set_gains(item, .01, ppo_gain=.02 if lucky else -.001)
    result = runner.confirmation_summary(rows, selection(), gate())
    assert result["mean_baseline_gap"] >= .005
    assert result["per_fold"]["2025"]["mean_gap"] > 0
    assert result["per_fold"]["2025"]["mean_ppo_gain"] > 0
    assert result["qualified"] is False


@pytest.fixture
def registered_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "study")
    for name in runner.INVENTORY:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen inventory: " + name, encoding="utf-8")
    reference = tmp_path / "reference_config.json"
    runner.write_json(reference, dict(timing=dict(fold="2023"), ppo={}, research_config="source"))
    monkeypatch.setattr(runner.parent, "reference_config", lambda fold: reference)
    old = dict(folds={fold: dict(research_config="source-" + fold) for fold in (*runner.SCREEN, *runner.CONFIRM)},
               gate=gate(), baseline_reference_root="synthetic-baseline")
    pilot = dict(source_hashes={"src/synthetic.py": "f" * 64})
    monkeypatch.setattr(runner, "upstream", lambda: (deepcopy(old), deepcopy(pilot)))
    monkeypatch.setattr(runner, "git_metadata", lambda: dict(commit="synthetic", dirty=False))
    constructed = []

    class Data:
        def __init__(self, config):
            constructed.append(deepcopy(config))
            self.identity = dict(fold=config["timing"]["fold"], scaler="synthetic-inner")

    monkeypatch.setattr(runner, "CausalTimingData", Data)
    return SimpleNamespace(root=runner.OUTPUT, constructed=constructed)


def test_prepare_freezes_every_required_file_and_rechecks_existing_registration(registered_workspace):
    workspace = registered_workspace
    protocol = runner.prepare(workspace.root)
    assert len(workspace.constructed) == 2
    assert set(protocol["inventory_sha256"]) == set(runner.INVENTORY)
    assert set(protocol["data"]) == set(runner.SCREEN)
    assert runner.prepare(workspace.root) == protocol
    assert len(workspace.constructed) == 2
    runner.check_frozen(protocol, workspace.root)


def test_prepare_missing_required_file_is_not_silently_skipped(registered_workspace, monkeypatch):
    workspace = registered_workspace
    (runner.ROOT / runner.INVENTORY[-1]).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        runner.prepare(workspace.root)
    assert not workspace.root.exists()
    assert workspace.constructed == []


def test_frozen_guard_rejects_changed_inventory_registration_and_partial_output(registered_workspace):
    workspace = registered_workspace
    protocol = runner.prepare(workspace.root)
    changed = deepcopy(protocol)
    changed["ppo_screen_steps"] += 1
    with pytest.raises(ValueError):
        runner.check_frozen(changed, workspace.root)
    changed = deepcopy(protocol)
    changed["inventory_sha256"].pop(next(iter(changed["inventory_sha256"])))
    with pytest.raises(ValueError):
        runner.check_frozen(changed, workspace.root)
    (runner.ROOT / runner.INVENTORY[0]).write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.prepare(workspace.root)


def test_prepare_refuses_a_partial_output_directory(registered_workspace):
    registered_workspace.root.mkdir()
    with pytest.raises(ValueError, match="partial"):
        runner.prepare(registered_workspace.root)


def test_report_cannot_publish_without_complete_development(registered_workspace, monkeypatch):
    workspace = registered_workspace
    protocol = runner.prepare(workspace.root)
    monkeypatch.setattr(runner, "load_rows", lambda *args: screen_rows()[:-1])
    with pytest.raises(ValueError):
        runner.report(workspace.root, protocol)
    assert not (workspace.root / "summary.json").exists()
    assert not (workspace.root / "selection.json").exists()


def test_failed_screen_report_never_opens_confirmation_or_claims_qualified(registered_workspace, monkeypatch):
    workspace = registered_workspace
    protocol = runner.prepare(workspace.root)
    phases = []

    def rows(root, protocol, phase):
        phases.append(phase)
        assert phase == "screen"
        return screen_rows(gap=0.)

    monkeypatch.setattr(runner, "load_rows", rows)
    result = runner.report(workspace.root, protocol)
    assert phases == ["screen"]
    assert result["status"] == "screen_failed"
    assert result["qualified"] is False and result["candidate"] is None
    assert result["protected_period_evaluated"] is result["actionable"] is False


def test_real_sb3_callback_checkpoint_roundtrip_on_synthetic_four_stock_market(tmp_path, monkeypatch):
    """Exercise real trace, timeout credit and save/load glue before long runs."""
    from dataclasses import asdict
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.agents.timing_policy import CalendarTimingPolicy
    from src.agents.trade_mapping import TradeActionWrapper
    from src.data.preprocess import FEATURES
    from src.envs.broker import BrokerConfig
    from src.envs.trading import TradingEnv
    from src.training.exploration_env import ExplorationMarket
    from src.training.timing_env import TimingEnv
    from tests.test_timing_counterfactual import make_env

    source = make_env(days=80, episode_steps=12)
    source_market = source.env.env.market
    codes = source_market.stock_codes[:4]
    features = source_market.panel.loc[:, pd.IndexSlice[codes, :]].copy()
    quotes = pd.DataFrame(index=features.index,
        columns=pd.MultiIndex.from_product([codes, ("open", "close", "volume")]), dtype=float)
    for index, code in enumerate(codes):
        for field in ("open", "close", "volume"):
            quotes[code, field] = source_market._fields[field][:, index]
    broker = BrokerConfig()
    market = ExplorationMarket(features, FEATURES, price_path=quotes, price_basis="hfq_research",
        start_date=features.index[4], common_lookback=5)
    stock = TradeActionWrapper(TradingEnv(market, broker_config=broker, max_steps=12,
        initial_portfolio="mixed"), top_n=20)
    contract = dict(stock_codes=codes, feature_cols=list(FEATURES), logit_bound=8.,
        broker=asdict(broker), price_basis="hfq_research", artifacts=dict(synthetic="four_stock_generated_values"))
    timing = TimingEnv(stock, contract, training=True, reward_basis="relative_log", episode_steps=12)
    source.close()
    config = dict(timing=dict(reward_basis="relative_log", phase="screen"), max_commission_ratio=.001,
        ppo=dict(gamma=.995, gae_lambda=1.))
    data = SimpleNamespace(contract=deepcopy(timing.contract), config=config, codes=codes,
                           settings=dict(broker=asdict(broker)))
    experience = runner.original.TimingExperience(timing)
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(42)
    torch.set_num_threads(1)
    model = PPO(CalendarTimingPolicy, env, seed=42, device="cpu", verbose=0,
        policy_kwargs=dict(calendar_margin=2., net_arch={"pi": [], "vf": [64, 64]}),
        n_steps=16, batch_size=8, n_epochs=1, learning_rate=3e-4,
        gamma=.995, gae_lambda=1., target_kl=None)
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.policy.state_dict().items()}
    output = tmp_path / "synthetic_real_callback"
    output.mkdir()
    callback = runner.LearningCallback(data, output, experience, initial, 32, "synthetic32", protocol={})
    # Match this bounded test's real trace chunks to its two 16-step rollouts.
    callback.trace.chunk_steps = callback.trace.benchmark_trace.chunk_steps = 16
    monkeypatch.setattr(runner, "SNAPSHOTS", (32,))
    model.set_logger(configure(str(output / "logs"), []))
    try:
        model.learn(total_timesteps=32, callback=callback)
        before = runner.training_fingerprint(model, experience)
        callback.capture(32)
        assert runner.training_fingerprint(model, experience) == before
        assert model.num_timesteps == 32 and model._n_updates == 2
        assert experience.stats()["total_steps"] == experience.stats()["benchmark_steps"] == 32
        audit = callback.snapshots[32]
        assert audit["checkpoint_training_state_unchanged"] is True
        assert len(audit["rollout_buffers"]) == len(audit["credit_estimator_buffers"]) == 2
        assert len(audit["training_trace_prefix"]["chunks"]) == 2
        counts = audit["optimizer_steps_by_parameter"]
        assert len(counts) == 8 and set(counts.values()) == {4}
        assert all(value > 0 for name, value in audit["tensor_delta_l2"].items()
                   if name.startswith(("action_net.", "mlp_extractor.value_net.", "value_net.")))
        assert (output / "snapshots/32/model/model.zip").exists()
        timeout_count = 0
        for rollout, credit in zip(audit["rollout_buffers"], audit["credit_estimator_buffers"]):
            with np.load(output / rollout["path"], allow_pickle=False) as buffer, np.load(output / credit["path"], allow_pickle=False) as side:
                timeout_count += int(side["truncated"].sum())
                advantages = np.zeros_like(buffer["advantages"])
                last_gae = np.zeros(1, dtype=np.float32)
                for step in reversed(range(16)):
                    next_active = 1. - (side["last_dones"] if step == 15 else buffer["episode_starts"][step + 1])
                    next_value = side["last_values"] if step == 15 else buffer["values"][step + 1]
                    delta = buffer["rewards"][step] + .995 * next_value * next_active - buffer["values"][step]
                    last_gae = delta + .995 * next_active * last_gae
                    advantages[step] = last_gae
                np.testing.assert_allclose(advantages, buffer["advantages"], rtol=0, atol=2e-6)
                np.testing.assert_allclose(advantages + buffer["values"], buffer["returns"], rtol=0, atol=2e-6)
        assert timeout_count >= 2
    finally:
        callback.trace.close()
        env.close()
