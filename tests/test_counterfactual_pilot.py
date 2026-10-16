"""Driver-only synthetic bookkeeping: no market provider, model or fitting."""

from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools import research_timing_counterfactual_pilot as pilot


class FakeMarket:
    def __init__(self):
        self.dates = pd.bdate_range("2017-01-02", periods=160)
        self.panel = pd.DataFrame(index=self.dates)
        self.prices = pd.DataFrame(index=self.dates)
        self.current_index = 0

    @property
    def current_date(self):
        return self.dates[self.current_index]


class FakeEnv:
    def __init__(self):
        self.training, self.reward_basis = True, "relative_log"
        self.env = SimpleNamespace(env=SimpleNamespace(market=FakeMarket(), _steps=0, max_steps=80))
        self.benchmark = SimpleNamespace(env=SimpleNamespace(market=FakeMarket(), _steps=0, max_steps=80))
        self.episode, self.phase, self._done = -1, 0, True
        self.total_steps = 0
        self.closed = False

    def reset(self, seed=None):
        if seed is not None:
            assert seed == 42
            self.episode = -1
        self.episode += 1
        modes = ("cash", "equal", "concentrated", "random")
        dates = ("early", "middle", "latest")
        mode, date, phase = list(product(modes, dates, range(4)))[self.episode]
        self.phase = phase * 5 + self.episode % 5
        self._done = False
        for base in (self.env.env, self.benchmark.env):
            base._steps = 0
            base.market.current_index = self.episode % 20
        return self.observe(), {"timing": {"schedule": dict(initial_mode=mode, date_stratum=date, phase_bin=phase)}}

    def observe(self):
        return np.full(129, self.episode + self.env.env._steps / 100, dtype=np.float32)

    def step(self, action):
        assert action == int(self.phase == 0)
        for base in (self.env.env, self.benchmark.env):
            base._steps += 1
            base.market.current_index += 1
        self.total_steps += 1
        self.phase = (self.phase + 1) % 20
        return self.observe(), 0., False, False, {}

    def close(self):
        self.closed = True


def fake_fingerprint(env):
    return pilot.digest(dict(episode=env.episode, phase=env.phase, step=env.env.env._steps,
                             index=env.env.env.market.current_index, total_steps=env.total_steps))


def fake_label(env, *, horizon, train_end, gamma):
    assert horizon == 20 and gamma == .995
    delta = (0., -.0002, .0002)[env.episode % 3]
    branches = {}
    for first, key in enumerate(("hold", "rebalance")):
        steps = []
        for j in range(20):
            phase = (env.phase + j) % 20
            steps.append(dict(index=j, phase=phase, macro_action=first if j == 0 else int(phase == 0),
                terminated=False, truncated=j == 19 and env.env.env._steps == 60,
                relative_log_reward=delta if first and j == 0 else 0., benchmark_ledger={}))
        branches[key] = dict(initial_agent_nav=1e6, initial_benchmark_nav=1e6,
            final_agent_nav=float(1e6 * np.exp(delta if first else 0.)), final_benchmark_nav=1e6,
            actual_agent_steps=20, actual_benchmark_steps=20,
            endpoint=str(env.env.env.market.dates[env.env.env.market.current_index + 20].date()), steps=steps)
    proof = {key: True for key in ("source_unchanged", "global_rng_unchanged", "benchmark_paths_equal",
        "clone_initial_state_equal", "clone_storage_independent", "current_close_observation_verified")}
    proof.update(source_before_sha256=fake_fingerprint(env), source_after_sha256=fake_fingerprint(env), actionable=False,
                 price_basis="hfq_research", research_only=True)
    return dict(schema_version=1, label_version=pilot.paired.LABEL_VERSION,
        input=dict(observation=env.observe().tolist(), feature_names=list(pilot.paired.FEATURE_NAMES),
        decision_date=str(env.env.env.market.current_date.date()), phase=env.phase), provenance=proof,
        labels=dict(horizon=20, gamma=.995, train_end=train_end, discounted_relative_delta=delta,
            undiscounted_relative_delta=delta, terminal_log_nav_ratio=delta,
            per_step_relative_reward_difference=[delta] + [0.] * 19, branches=branches),
        counters=dict(branch_agent_steps=40, branch_benchmark_steps=40, macro_attempts=40,
                      collection_steps=0, rl_steps=0, nn_forwards=0, optimizer_updates=0))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "OUTPUT", tmp_path / "pilot")
    monkeypatch.setattr(pilot, "PARENT", tmp_path / "parent")
    for name in pilot.INVENTORY:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic inventory " + name, encoding="utf-8")
    monkeypatch.setattr(pilot, "__file__", str(tmp_path / pilot.INVENTORY[0]))
    research = tmp_path / "research.json"
    research.write_text("{}", encoding="utf-8")
    signed = {}
    for fold in pilot.FOLDS:
        path = pilot.reference_config(fold)
        path.parent.mkdir(parents=True)
        pilot.write_json(path, dict(seed=42, timing=dict(fold=fold, reward_basis="relative_log"),
            training_episode_steps=240, output_root="parent", research_config=str(research)))
        signed[path.relative_to(tmp_path).as_posix()] = pilot.file_hash(path)
    monkeypatch.setattr(pilot, "parent_reference", lambda: ({"signed_files_sha256": signed}, {"src/synthetic.py": "a" * 64}))
    created, splits = [], []

    class FakeData:
        def __init__(self, config):
            self.config = deepcopy(config)
            self.manifest = dict(train_end=config["timing"]["fold"] + "-03-31")
            self.contract = dict(stock_codes=[str(i).zfill(6) for i in range(300)], feature_cols=list(range(20)),
                price_basis="hfq_research", training_episode_steps=config["training_episode_steps"],
                manifest_sha256="a" * 64, artifacts={"panel": "b" * 64})

        def env(self, split):
            assert split == "train"
            splits.append(split)
            env = FakeEnv()
            created.append(env)
            return env

    monkeypatch.setattr(pilot, "make_data", FakeData)
    monkeypatch.setattr(pilot.paired, "snapshot_state_fingerprint", fake_fingerprint)
    monkeypatch.setattr(pilot.paired, "paired_rule_label", fake_label)
    return SimpleNamespace(root=pilot.OUTPUT, created=created, splits=splits, data_class=FakeData)


def test_prepare_registers_explicit_80_override_and_exact_six_files(workspace):
    protocol = pilot.prepare(workspace.root)
    assert len(protocol["inventory_sha256"]) == 6
    assert protocol["collection_episode_steps"] == 80
    assert all(v["config"]["training_episode_steps"] == 80 for v in protocol["folds"].values())
    assert workspace.created == []
    assert not (workspace.root / "folds").exists()


def test_full_two_pass_and_report_are_diagnostic_not_training(workspace):
    protocol = pilot.prepare(workspace.root)
    for fold in pilot.FOLDS:
        rows = pilot.run_fold(workspace.root, protocol, fold)
        assert len(rows) == 96
        assert {row["discounted_relative_delta"] for row in rows} == {0., -.0002, .0002}
        directory = workspace.root / "folds" / fold
        counts = pilot.read_json(directory / "counters.json")
        assert counts == pilot.expected_counts()
        assert counts["collection_agent_steps"] == 5760
        assert counts["branch_agent_steps"] == 3840
        assert counts["rl_steps"] == counts["optimizer_updates"] == counts["nn_forwards"] == 0
        assert len(list((directory / "labels").glob("*.json.gz"))) == 96
        assert (directory / "sampling_plan.json").exists()
        with np.load(directory / "inputs.npz", allow_pickle=False) as z:
            assert z["observations"].shape == (96, 129)
        before = len(workspace.created)
        assert pilot.cached_fold(workspace.root, protocol, fold) == rows
        assert len(workspace.created) == before
    summary = pilot.write_report(workspace.root, protocol)
    assert summary["agent_steps"] == summary["benchmark_steps"] == 19200
    assert summary["book_steps"] == 38400
    assert summary["states"] == 192 and summary["model_fits"] == 0
    assert summary["qualified_candidate"] is None
    assert workspace.splits == ["train"] * 4
    assert all(env.total_steps == 2880 and env.closed for env in workspace.created)
    assert not list(workspace.root.rglob("*.zip")) and not list(workspace.root.rglob("*.pkl"))


def test_report_never_collects_missing_fold(workspace, monkeypatch):
    protocol = pilot.prepare(workspace.root)
    monkeypatch.setattr(pilot, "new_environment", lambda *a: pytest.fail("Report collected a new state"))
    with pytest.raises(ValueError, match="missing fold"):
        pilot.write_report(workspace.root, protocol)


def test_complete_plan_and_input_sha_are_sealed_before_first_label(workspace, monkeypatch):
    protocol = pilot.prepare(workspace.root)
    calls = []

    def checked(env, **kwargs):
        directory = workspace.root / "folds/2023"
        seal = pilot.read_json(directory / "sampling_plan_seal.json")
        assert seal["sampling_plan_sha256"] == pilot.file_hash(directory / "sampling_plan.json")
        assert seal["inputs_sha256"] == pilot.file_hash(directory / "inputs.npz")
        assert seal["states"] == len(pilot.read_json(directory / "sampling_plan.json")["states"]) == 96
        assert seal["labels_started"] is False
        calls.append(env.episode)
        return fake_label(env, **kwargs)

    monkeypatch.setattr(pilot.paired, "paired_rule_label", checked)
    pilot.run_fold(workspace.root, protocol, "2023")
    assert len(calls) == 96


def test_unexpected_nested_completed_file_is_not_exempt_from_leaf_binding(tmp_path):
    pilot.write_json(tmp_path / "completed.json", {})
    nested = tmp_path / "nested"
    nested.mkdir()
    pilot.write_json(nested / "completed.json", {})
    assert set(pilot.leaf_files(tmp_path)) == {"nested/completed.json"}


@pytest.mark.parametrize("field,value", [("horizon", True), ("gamma", True), ("collection_episode_steps", 240),
    ("collection_passes", 1), ("states_per_fold", 95), ("total_book_steps", 26880)])
def test_design_drift_is_rejected_before_provider(workspace, monkeypatch, field, value):
    protocol = pilot.prepare(workspace.root)
    protocol[field] = value
    monkeypatch.setattr(pilot, "make_data", lambda *a: pytest.fail("Created provider before frozen guard"))
    with pytest.raises(ValueError, match="design or budget"):
        pilot.run_fold(workspace.root, protocol, "2023")


def test_inventory_tamper_is_rejected_before_provider(workspace, monkeypatch):
    protocol = pilot.prepare(workspace.root)
    (pilot.ROOT / pilot.INVENTORY[2]).write_text("changed", encoding="utf-8")
    monkeypatch.setattr(pilot, "make_data", lambda *a: pytest.fail("Created provider after tool tamper"))
    with pytest.raises(ValueError, match="Frozen pilot"):
        pilot.run_fold(workspace.root, protocol, "2023")


def test_reference_config_tamper_cannot_change_data(workspace):
    protocol = pilot.prepare(workspace.root)
    path = pilot.reference_config("2023")
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Signed parent config"):
        pilot.check_frozen(protocol, workspace.root)


def test_partial_label_failure_keeps_asymmetric_paid_steps_and_refuses_resume(workspace, monkeypatch):
    protocol = pilot.prepare(workspace.root)

    def failed(*args, **kwargs):
        count = dict(branch_agent_steps=1, branch_benchmark_steps=0, macro_attempts=1)
        raise pilot.paired.CounterfactualLabelError("synthetic partial branch", counters=count,
                                                  evidence={"partial_branch_effects_are_not_rolled_back": True})

    monkeypatch.setattr(pilot.paired, "paired_rule_label", failed)
    with pytest.raises(pilot.paired.CounterfactualLabelError):
        pilot.run_fold(workspace.root, protocol, "2023")
    directory = workspace.root / "folds/2023"
    failure = pilot.read_json(directory / "failure.json")
    assert failure["counters"]["collection_agent_steps"] == 2880
    assert failure["counters"]["branch_agent_steps"] == 1
    assert failure["counters"]["branch_benchmark_steps"] == 0
    assert not (directory / "completed.json").exists()
    with pytest.raises(ValueError, match="partial"):
        pilot.run_fold(workspace.root, protocol, "2023")


def test_replay_full_state_mismatch_stops_before_any_label(workspace, monkeypatch):
    protocol = pilot.prepare(workspace.root)
    factory = pilot.new_environment

    def changed(protocol, fold):
        env = factory(protocol, fold)
        if len(workspace.created) == 2:
            env.total_steps = 1
        return env

    monkeypatch.setattr(pilot, "new_environment", changed)
    monkeypatch.setattr(pilot.paired, "paired_rule_label", lambda *a, **k: pytest.fail("Label before replay equality"))
    with pytest.raises(ValueError, match="Second-pass full state"):
        pilot.run_fold(workspace.root, protocol, "2023")
    assert pilot.read_json(workspace.root / "folds/2023/failure.json")["completed_pairs"] == 0


def test_provider_future_rows_rejected_without_steps(workspace, monkeypatch):
    protocol = pilot.prepare(workspace.root)
    original = workspace.data_class.env

    def future(self, split):
        env = original(self, split)
        env.env.env.market.panel = pd.DataFrame(index=pd.to_datetime(["2026-07-01"]))
        return env

    monkeypatch.setattr(workspace.data_class, "env", future)
    with pytest.raises(ValueError, match="physically cropped"):
        pilot.run_fold(workspace.root, protocol, "2023")
    assert workspace.created[0].total_steps == 0


def test_plan_requires_all_strata_and_training_endpoint(workspace):
    protocol = pilot.prepare(workspace.root)
    env = pilot.new_environment(protocol, "2023")
    states = [r for r, _ in pilot.collection_states(env, "2023", pilot.counters())]
    plan = dict(fold="2023", seed=42, states=states)
    pilot.validate_plan(plan, "2023")
    bad = deepcopy(plan)
    bad["states"][-1]["endpoint"] = "2023-04-03"
    with pytest.raises(ValueError, match="endpoint"):
        pilot.validate_plan(bad, "2023")
    bad = deepcopy(plan)
    bad["states"][0]["initial_mode"] = "unknown"
    with pytest.raises(ValueError):
        pilot.validate_plan(bad, "2023")


@pytest.mark.parametrize("phase,delta,expected", [(0, .001, 0.), (0, -.001, .001), (3, .001, .001), (3, -.001, 0.)])
def test_oracle_uses_original_calendar_action_not_positive_delta_alone(phase, delta, expected):
    record = {k: 0 for k in ("state_id", "episode", "offset", "decision_date", "endpoint", "initial_mode",
                             "date_stratum", "phase_bin", "input_sha256", "full_state_sha256")}
    record["phase"] = phase
    label = {"labels": {"discounted_relative_delta": delta, "terminal_log_nav_ratio": delta}}
    assert pilot.index_row(record, label, "hash")["oracle_calendar_advantage"] == expected


@pytest.mark.parametrize("args", [["--stage", "run"], ["--stage", "report", "--fold", "2023"],
    ["--stage", "prepare", "--fold", "2024"], ["--stage", "run", "--fold", "2025"]])
def test_cli_rejects_unregistered_stage_fold_combinations(args):
    with pytest.raises(SystemExit):
        pilot.main(args)


def test_output_root_must_be_exact_fresh_registered_directory(workspace):
    with pytest.raises(ValueError, match="registered fresh"):
        pilot.prepare(pilot.PARENT / "child")
    workspace.root.mkdir()
    with pytest.raises(ValueError, match="partial output"):
        pilot.prepare(workspace.root)
