"""Registered reward x GAE trace study using immutable timing execution modules."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.timing_agent import TimingAgent
from src.agents.timing_policy import CalendarTimingPolicy
from src.backtest.run import git_metadata
from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.envs.initial_account import MODES
from src.training import research_timing as original
from src.training.exploration_env import ExplorationData
from src.training.freeze import source_hashes, json_digest
from src.training.research_exploration import config_for as exploration_config
from src.training.research_matrix import preserve_training_rng, diagnostic_summary
from src.training.research_delta import _release
from src.training.timing_env import FEATURE_NAMES, ACTION_NAMES, build_timing_contract
from tools import research_timing_margin as shared


REWARDS = ("log_net", "relative_log")
LAMBDAS = (.95, 1.)
CELLS = tuple((reward, value) for value in LAMBDAS for reward in REWARDS)
BUDGETS = {"screen": 8192, "confirm": 16384}
DOCUMENT = ROOT / "doc" / "ppo_timing_credit_protocol_20261015.md"
REFERENCE = ROOT / "reports" / "runs" / "ppo_timing_margin_20261014"
REFERENCE_AUDIT_SHA256 = "1e8b7363582b452f6f1b42a79049cdd94f067b21a244abf8b8734e4a390453df"
HELPER_FILE = "tools/research_timing_margin.py"
TEST_FILES = ("tests/test_research_timing_credit.py", "tests/test_timing_credit_review.py")


def cell_label(reward, gae_lambda):
    if reward not in REWARDS or isinstance(gae_lambda, (bool, np.bool_)) or gae_lambda not in LAMBDAS:
        raise ValueError("Unregistered reward or GAE lambda")
    return ("log" if reward == "log_net" else "relative") + ("_l095" if gae_lambda == .95 else "_l100")


def run_id(fold, reward, gae_lambda, phase, seed):
    return f'{phase}_{fold}_{cell_label(reward, gae_lambda)}_{seed}'


def reference_protocol(directory):
    directory = Path(directory).resolve()
    if REFERENCE_AUDIT_SHA256 is None or file_hash(directory / "final_audit.json") != REFERENCE_AUDIT_SHA256:
        raise ValueError("The completed margin audit must be pinned before registration")
    audit, protocol = read_config(directory / "final_audit.json"), read_config(directory / "protocol.json")
    if not directory.is_relative_to(ROOT.resolve()):
        raise ValueError("Reference must remain in its registered workspace")
    key = (directory / "protocol.json").relative_to(ROOT.resolve()).as_posix()
    if audit.get("status") != "passed" or audit.get("signed_files_sha256", {}).get(key) != file_hash(directory / "protocol.json"):
        raise ValueError("Margin audit does not sign its actual registration")
    shared.check_frozen(protocol)
    return protocol


def check_frozen(protocol):
    for value in protocol.get("gae_lambdas", []):
        cell_label("log_net", value)
    fixed = dict(study="timing_credit", rewards=list(REWARDS), gae_lambdas=list(LAMBDAS), budgets={"screen": 8192, "confirm": 16384},
        parameters=12805, feature_names=list(FEATURE_NAMES), planned_screen_trials=8, planned_screen_steps=65536,
        maximum_confirmation_trials=6, maximum_confirmation_steps=98304, maximum_actual_training_steps=163840,
        screen_minimum_each_fold_gap=.0001, screen_requires_greedy_and_economic_change=True,
        tie_break="lambda .95 then log_net")
    if BUDGETS != fixed["budgets"] or any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError("Credit grid, budget or admission rule changed")
    if (source_hashes() != protocol["source_hashes"] or len(protocol["source_hashes"]) != 96
            or file_hash(Path(__file__)) != protocol["tool_sha256"] or file_hash(ROOT / HELPER_FILE) != protocol["helper_sha256"]
            or file_hash(DOCUMENT) != protocol["document_sha256"] or set(protocol.get("test_hashes", {})) != set(TEST_FILES)
            or protocol["reference_final_audit_sha256"] != REFERENCE_AUDIT_SHA256):
        raise ValueError("Credit runner, helper, tests, document or original source changed")
    for name, digest in protocol["test_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Frozen credit test changed")
    previous = reference_protocol(protocol["reference_root"])
    if (file_hash(Path(protocol["reference_root"]) / "protocol.json") != protocol["reference_protocol_sha256"]
            or any(protocol[key] != previous[key] for key in ("folds", "base_config", "gate"))
            or protocol["reproduction_reference_root"] != previous["reference_root"]
            or file_hash(Path(protocol["reproduction_reference_root"]) / "protocol.json") != protocol["reproduction_protocol_sha256"]):
        raise ValueError("Credit study changed original data, shared settings or reference lineage")


def prepare(root, reference_root=REFERENCE):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    previous = reference_protocol(reference_root)
    reproduction = Path(previous["reference_root"]).resolve()
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in (reference_root, reproduction)):
        raise ValueError("Credit output cannot overlap either immutable predecessor")
    if (root / "protocol.json").exists():
        saved = read_config(root / "protocol.json")
        if saved["reference_root"] != str(reference_root):
            raise ValueError("Registration cannot replace its predecessor")
        check_frozen(saved)
        return saved
    if root.exists():
        raise ValueError("Incomplete unregistered output requires review")
    protocol = dict(schema_version=1, study="timing_credit", rewards=list(REWARDS), gae_lambdas=list(LAMBDAS), budgets=deepcopy(BUDGETS),
        folds=deepcopy(previous["folds"]), base_config=deepcopy(previous["base_config"]), gate=deepcopy(previous["gate"]),
        reference_root=str(reference_root), reference_protocol_sha256=file_hash(reference_root / "protocol.json"),
        reference_final_audit_sha256=REFERENCE_AUDIT_SHA256, reproduction_reference_root=str(reproduction),
        reproduction_protocol_sha256=file_hash(reproduction / "protocol.json"), source_hashes=source_hashes(),
        tool_sha256=file_hash(Path(__file__)), helper_sha256=file_hash(ROOT / HELPER_FILE), document_sha256=file_hash(DOCUMENT),
        test_hashes={name: file_hash(ROOT / name) for name in TEST_FILES}, git=git_metadata(), parameters=12805,
        feature_names=list(FEATURE_NAMES), planned_screen_trials=8, planned_screen_steps=65536,
        maximum_confirmation_trials=6, maximum_confirmation_steps=98304, maximum_actual_training_steps=163840,
        screen_minimum_each_fold_gap=.0001, screen_requires_greedy_and_economic_change=True,
        tie_break="lambda .95 then log_net", test_evaluated=False, actionable=False)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", protocol)
    check_frozen(protocol)
    return protocol


def config_for(protocol, root, fold, reward, gae_lambda, phase, seed):
    label = cell_label(reward, gae_lambda)
    if (phase not in BUDGETS or fold not in (original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS)
            or seed not in original.SEEDS or phase == "screen" and seed != 42):
        raise ValueError("Unregistered credit condition")
    config = exploration_config(protocol["base_config"], root, "split_l1", seed)
    config.update(research_config=protocol["folds"][fold]["research_config"], action_mode="timing_control",
        device="cpu", torch_threads=1, hidden_sizes=[64, 64], top_n=20, top_n_choices=[],
        training_episode_steps=240, recent_episode_every=0, output_root=str(Path(root) / "runs"),
        timing=dict(fold=fold, phase=phase, reward_basis=reward, calendar_margin=2., period=20),
        credit_study=dict(study="timing_credit", reward_basis=reward, gae_lambda=gae_lambda, label=label,
            protocol_sha256=json_digest(protocol), tool_sha256=protocol["tool_sha256"], helper_sha256=protocol["helper_sha256"],
            reference_audit_sha256=protocol["reference_final_audit_sha256"]))
    config["exploration"]["reward_basis"] = "log_net"
    config["ppo"].update(learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=5, gamma=.995, gae_lambda=gae_lambda,
        clip_range=.2, target_kl=.03, ent_coef=0., vf_coef=.5, max_grad_norm=.5)
    return config


class CreditData(ExplorationData):
    def __init__(self, config):
        spec, identity = config["timing"], config["credit_study"]
        cell_label(spec["reward_basis"], identity["gae_lambda"])
        cell_label(spec["reward_basis"], config["ppo"]["gae_lambda"])
        if (config["action_mode"] != "timing_control" or spec["calendar_margin"] != 2. or spec["period"] != 20
                or config["ppo"]["gae_lambda"] != identity["gae_lambda"] or config["ppo"]["gamma"] != .995
                or identity["reward_basis"] != spec["reward_basis"] or config["top_n"] != 20 or config["top_n_choices"]):
            raise ValueError("Invalid registered reward/credit contract")
        borrowed = deepcopy(config)
        borrowed["action_mode"] = "stock_trade"
        super().__init__(borrowed)
        self.stock_contract, self.config = deepcopy(self.contract), deepcopy(config)
        self.contract = build_timing_contract(self.stock_contract, spec["reward_basis"], 2.)
        self.contract["timing_research"] = deepcopy(spec)
        self.contract["credit_study"] = deepcopy(identity)

    def env(self, split):
        return original.TimingData.env(self, split)


def make_model(env, config):
    torch.set_num_threads(1)
    return PPO(CalendarTimingPolicy, env, seed=config["seed"], device="cpu", verbose=0,
        policy_kwargs=dict(calendar_margin=2., net_arch={"pi": [], "vf": [64, 64]}), **config["ppo"])


class CreditCallback(original.TimingCallback):
    """Retain the generic native trace and add actual estimator-scale diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator_statistics = {}
        self.estimator_buffers, self.boundary_rows = [], []
        self.pending_timeout = None

    def _on_rollout_start(self):
        super()._on_rollout_start()
        self.boundary_rows = []
        self.pending_timeout = None

    def _finish_pending_timeout(self):
        if self.pending_timeout is None:
            return
        value = self.locals.get("terminal_value")
        if value is None:
            raise ValueError("SB3 actual timeout bootstrap value was not exposed by callback locals")
        value = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError("SB3 actual timeout bootstrap must be one finite value")
        self.boundary_rows[self.pending_timeout]["terminal_value"] = np.float32(value.reshape(-1)[0])
        self.pending_timeout = None

    def _on_step(self):
        self._finish_pending_timeout()
        if len(self.locals["infos"]) != 1:
            raise ValueError("Credit bootstrap evidence is registered for one environment per worker")
        info = self.locals["infos"][0]
        done = bool(np.asarray(self.locals["dones"])[0])
        timeout = bool(info.get("TimeLimit.truncated", False))
        if timeout and (not done or info.get("terminal_observation") is None):
            raise ValueError("Timeout evidence lacks its actual terminal observation")
        self.boundary_rows.append(dict(raw_reward=np.float32(np.asarray(self.locals["rewards"])[0]),
            native_learning_reward=float(info["timing"]["learning_reward"]), done=done, truncated=timeout,
            terminal_value=np.float32(0.)))
        if timeout:
            self.pending_timeout = len(self.boundary_rows) - 1
        return super()._on_step()

    def _on_rollout_end(self):
        super()._on_rollout_end()
        self._finish_pending_timeout()
        buffer = self.model.rollout_buffer
        cell_label(self.data.config["timing"]["reward_basis"], buffer.gae_lambda)
        cell_label(self.data.config["timing"]["reward_basis"], self.data.config["ppo"]["gae_lambda"])
        if buffer.gamma != .995 or buffer.gae_lambda != self.data.config["ppo"]["gae_lambda"]:
            raise ValueError("Actual rollout buffer uses a different credit estimator")
        self.estimator_statistics = {"credit/gamma": float(buffer.gamma), "credit/gae_lambda": float(buffer.gae_lambda),
            "credit/returns_std": float(np.std(buffer.returns)), "credit/advantages_std": float(np.std(buffer.advantages)),
            "credit/buffer_rewards_std": float(np.std(buffer.rewards))}
        if len(self.boundary_rows) != self.model.n_steps:
            raise ValueError("Credit boundary evidence does not cover the actual complete rollout")
        tail = self.locals["values"]
        tail = tail.detach().cpu().numpy() if torch.is_tensor(tail) else np.asarray(tail)
        arrays = dict(gamma=np.asarray(buffer.gamma), gae_lambda=np.asarray(buffer.gae_lambda),
            last_values=np.asarray(tail, dtype=np.float32).reshape(1),
            last_dones=np.asarray(self.locals["dones"], dtype=bool).reshape(1),
            dones=np.asarray([row["done"] for row in self.boundary_rows], dtype=bool).reshape(-1, 1),
            truncated=np.asarray([row["truncated"] for row in self.boundary_rows], dtype=bool).reshape(-1, 1),
            terminal_values=np.asarray([row["terminal_value"] for row in self.boundary_rows], dtype=np.float32).reshape(-1, 1),
            raw_rewards=np.asarray([row["raw_reward"] for row in self.boundary_rows], dtype=np.float32).reshape(-1, 1),
            native_learning_rewards=np.asarray([row["native_learning_reward"] for row in self.boundary_rows], dtype=np.float64).reshape(-1, 1))
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Actual bootstrap sidecar contains nonfinite evidence")
        directory = self.output / "credit_estimator_buffers"
        directory.mkdir(exist_ok=True)
        path = directory / f'rollout_{len(self.estimator_buffers) + 1:03d}.npz'
        np.savez_compressed(path, **arrays)
        self.estimator_buffers.append(dict(path=path.relative_to(self.output).as_posix(), sha256=file_hash(path),
            first_step=self.model.num_timesteps - self.model.n_steps + 1, last_step=self.model.num_timesteps, steps=self.model.n_steps,
            value_source="Existing SB3 collect_rollouts locals only; no new predictions or inferred terminal values"))

    def _record_update(self):
        for key, value in self.estimator_statistics.items():
            self.model.logger.record(key, value)
        super()._record_update()


def validate_records(records, phase, cell=None):
    expected = {(fold, reward, value, 42) for fold in original.SCREEN_FOLDS for reward, value in CELLS} if phase == "screen" else {
        (fold, cell[0], cell[1], seed) for fold in original.CONFIRM_FOLDS for seed in original.SEEDS} if cell in CELLS else set()
    actual = [(row["fold"], row["reward_basis"], row["gae_lambda"], row["seed"]) for row in records]
    if phase not in BUDGETS or not expected or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("All registered credit cells and fixed final budgets are required exactly once")
    for row in records:
        cell_label(row["reward_basis"], row["gae_lambda"])
        if row["phase"] != phase or row["steps"] != BUDGETS[phase] or set(row["metrics"]) != set(MODES) or set(row["changes"]) != set(MODES):
            raise ValueError("Credit snapshot budget or account set changed")
        for mode in MODES:
            values = row["metrics"][mode]
            if values["initialized"] != values["benchmark"] or any(type(row["changes"][mode].get(key)) is not bool for key in ("greedy_changed", "economic_ledger_changed")):
                raise ValueError("Credit initialized control or behavior evidence changed")
            for name in ("trained", "initialized", "benchmark"):
                if any(isinstance(values[name][key], bool) or not isinstance(values[name][key], (int, float)) or not np.isfinite(values[name][key])
                       for key in ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost")):
                    raise ValueError("Credit economic metrics must be finite")
    return records


def select_screen(records):
    validate_records(records, "screen")
    candidates = {}
    for reward, value in CELLS:
        rows = [row for row in records if (row["reward_basis"], row["gae_lambda"]) == (reward, value)]
        gaps = {row["fold"]: original._account_mean(row)["cumulative_return"] - original._account_mean(row, "benchmark")["cumulative_return"] for row in rows}
        candidates[cell_label(reward, value)] = dict(reward_basis=reward, gae_lambda=value, fold_gaps=gaps,
            minimum_fold_gap=min(gaps.values()), mean_gap=float(np.mean(list(gaps.values()))),
            worst_account_drawdown=max(abs(row["metrics"][mode]["trained"]["max_drawdown"]) for row in rows for mode in MODES),
            mean_cost=float(np.mean([original._account_mean(row)["total_fees"] + original._account_mean(row)["total_slippage_cost"] for row in rows])),
            fold_behavior_changes={row["fold"]: any(flags["greedy_changed"] and flags["economic_ledger_changed"] for flags in row["changes"].values()) for row in rows})
    selected = min(candidates.values(), key=lambda row: (-row["minimum_fold_gap"], -row["mean_gap"], row["worst_account_drawdown"],
        row["mean_cost"], row["gae_lambda"], 0 if row["reward_basis"] == "log_net" else 1))
    admitted = selected["minimum_fold_gap"] >= .0001 and any(selected["fold_behavior_changes"].values())
    return dict(selected_reward=selected["reward_basis"], selected_lambda=selected["gae_lambda"], admitted=bool(admitted),
        status="screen_passed" if admitted else "screen_failed", candidates=candidates, actual_screen_trials=8,
        actual_screen_steps=sum(row["steps"] for row in records), confirmation_steps_authorized=98304 if admitted else 0, no_fallback=True)


def verify_model(directory, meta, config, protocol, contract, steps):
    cell_label(config["timing"]["reward_basis"], config["ppo"]["gae_lambda"])
    for declared in (meta["contract"]["credit_study"], meta["config"]["credit_study"], meta["provenance"].get("credit_study", {})):
        cell_label(declared.get("reward_basis"), declared.get("gae_lambda"))
    record = dict(policy_class="src.agents.timing_policy.CalendarTimingPolicy", observation_shape=[129], observation_dtype="float32",
        feature_names=list(FEATURE_NAMES), action_names=list(ACTION_NAMES), net_arch={"pi": [], "vf": [64, 64]}, calendar_margin=2., phase_period=20, parameter_count=12805)
    if (meta["contract"] != contract or meta["config"] != config or meta["source_hashes"] != protocol["source_hashes"]
            or meta["provenance"].get("credit_study") != config["credit_study"] or meta["policy_record"] != record
            or meta["num_timesteps"] != steps or meta["actual_device"] != "cpu" or meta["feature_names"] != list(FEATURE_NAMES)
            or meta["algorithm_class"] != "stable_baselines3.ppo.ppo.PPO" or file_hash(Path(directory) / "model.zip") != meta["model_sha256"]):
        raise ValueError("Credit model does not bind its estimator, actual reward, runner and data")
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        saved = json.loads(archive.read("data"))
    cell_label(config["timing"]["reward_basis"], saved.get("gae_lambda"))
    archived_identity = saved.get("timing_registered_contract", {}).get("credit_study", {})
    cell_label(archived_identity.get("reward_basis"), archived_identity.get("gae_lambda"))
    if (saved.get("timing_registered_contract") != contract or saved.get("timing_policy_record") != record
            or saved.get("timing_registered_algorithm") != meta["algorithm_class"] or saved.get("num_timesteps") != steps
            or saved.get("gae_lambda") != config["ppo"]["gae_lambda"] or saved.get("gamma") != .995):
        raise ValueError("Credit PPO archive differs from its declared estimator and contract")


def baseline_metrics(protocol, fold):
    directory = protocol["reproduction_reference_root"]
    return original.rule_controls(directory, read_config(Path(directory) / "protocol.json"), fold, create=False)


def verify_estimator_buffers(directory, audit, config):
    cell_label(config["timing"]["reward_basis"], config["ppo"]["gae_lambda"])
    records = audit.get("credit_estimator_buffers", [])
    if len(records) != len(audit["rollout_buffers"]) or len(records) != len(audit["training_trace_prefix"]["chunks"]):
        raise ValueError("Actual bootstrap evidence must cover every original PPO rollout")
    required = {"gamma", "gae_lambda", "last_values", "last_dones", "dones", "truncated", "terminal_values", "raw_rewards", "native_learning_rewards"}
    for record, rollout, chunk in zip(records, audit["rollout_buffers"], audit["training_trace_prefix"]["chunks"]):
        path = (Path(directory) / record["path"]).resolve()
        if (not path.is_relative_to(Path(directory).resolve()) or file_hash(path) != record["sha256"]
                or any(record[key] != rollout[key] or record[key] != chunk[key] for key in ("first_step", "last_step", "steps"))):
            raise ValueError("Credit bootstrap sidecar identity or clock changed")
        with np.load(path, allow_pickle=False) as side, np.load(Path(directory) / rollout["path"], allow_pickle=False) as buffer, np.load(Path(directory) / "training_trace" / chunk["path"], allow_pickle=False) as native:
            if set(side.files) != required or any(not np.isfinite(side[key]).all() for key in side.files):
                raise ValueError("Actual bootstrap sidecar is incomplete or nonfinite")
            cell_label(config["timing"]["reward_basis"], side["gae_lambda"].item())
            if side["gamma"].shape != () or side["gae_lambda"].shape != () or float(side["gamma"]) != .995 or float(side["gae_lambda"]) != config["ppo"]["gae_lambda"]:
                raise ValueError("Actual bootstrap sidecar uses a different estimator")
            count = record["steps"]
            if (side["last_values"].shape != (1,) or side["last_dones"].shape != (1,)
                    or any(side[key].shape != (count, 1) for key in required - {"gamma", "gae_lambda", "last_values", "last_dones"})
                    or side["dones"].dtype != bool or side["truncated"].dtype != bool or side["last_dones"].dtype != bool):
                raise ValueError("Actual bootstrap sidecar layout changed")
            timeouts = side["truncated"]
            if (np.any(timeouts & ~side["dones"]) or np.any(side["terminal_values"][~timeouts] != 0)
                    or not np.array_equal(side["last_dones"], side["dones"][-1])
                    or not np.array_equal(side["dones"].reshape(-1), native["done"])
                    or not np.array_equal(buffer["episode_starts"][1:], side["dones"][:-1])
                    or not np.array_equal(side["native_learning_rewards"].reshape(-1), native["action_learning_reward"])
                    or not np.array_equal(side["raw_rewards"], side["native_learning_rewards"].astype(np.float32))):
                raise ValueError("Actual termination or raw reward evidence differs from the native rollout")
            corrected = side["raw_rewards"].copy()
            corrected[timeouts] += np.float32(.995) * side["terminal_values"][timeouts]
            if not np.allclose(corrected, buffer["rewards"], rtol=0., atol=1e-6):
                raise ValueError("PPO rewards do not match existing actual timeout bootstrap values")
    return records


def verify_worker_receipt(root, protocol, phase, fold, cell=None):
    if phase not in BUDGETS or fold not in (original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS) or phase == "confirm" and cell not in CELLS:
        raise ValueError("Unregistered credit worker")
    cells, seeds = (CELLS, (42,)) if phase == "screen" else ((cell,), original.SEEDS)
    expected = {run_id(fold, reward, value, phase, seed) for reward, value in cells for seed in seeds}
    saved = read_config(Path(root) / f'worker_{phase}_{fold}.json')
    if (saved["status"] != "complete" or saved["phase"] != phase or saved["fold"] != fold
            or saved["protocol_sha256"] != file_hash(Path(root) / "protocol.json")
            or saved["selection_sha256"] != (file_hash(Path(root) / "selection.json") if phase == "confirm" else None)
            or saved.get("no_global_summary_writes") is not True or len(saved["completed_runs"]) != len(expected)
            or {item["run_id"] for item in saved["completed_runs"]} != expected):
        raise ValueError("Every exclusive credit worker must seal all registered trials")
    for item in saved["completed_runs"]:
        if file_hash(Path(root) / "runs" / item["run_id"] / "completed.json") != item["completed_sha256"]:
            raise ValueError("Credit worker marker changed")
    return saved


def load_stage(root, protocol, phase, cell=None):
    rows = []
    for fold in original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS:
        verify_worker_receipt(root, protocol, phase, fold, cell)
        for reward, value in CELLS if phase == "screen" else (cell,):
            for seed in (42,) if phase == "screen" else original.SEEDS:
                config = config_for(protocol, root, fold, reward, value, phase, seed)
                row = cached_trial(Path(root) / "runs" / run_id(fold, reward, value, phase, seed), config, protocol,
                    file_hash(Path(root) / "selection.json") if phase == "confirm" else None)
                if row is None:
                    raise ValueError("Aggregation cannot create or borrow missing credit trials")
                rows.append(row)
    return validate_records(rows, phase, cell)


def seal_selection(root, protocol, rows):
    selection = dict(protocol_sha256=json_digest(protocol), **select_screen(rows), completed_sha256={
        run_id(row["fold"], row["reward_basis"], row["gae_lambda"], "screen", row["seed"]): file_hash(Path(root) / "runs" /
        run_id(row["fold"], row["reward_basis"], row["gae_lambda"], "screen", row["seed"]) / "completed.json") for row in rows})
    path = Path(root) / "selection.json"
    if path.exists() and read_config(path) != selection:
        raise ValueError("Frozen credit selection cannot change")
    if not path.exists():
        write_json(path, selection)
    return selection


def require_confirmation(root, protocol, cell, selection_sha256):
    cell_label(*cell)
    path = Path(root) / "selection.json"
    if not path.exists() or selection_sha256 != file_hash(path):
        raise ValueError("Confirmation requires the immutable eight-trial selection")
    selected = read_config(path)
    if selected["protocol_sha256"] != json_digest(protocol) or not selected["admitted"] or (selected["selected_reward"], selected["selected_lambda"]) != cell:
        raise ValueError("Failed or different credit selection cannot start confirmation")
    screen = load_stage(root, protocol, "screen")
    recomputed = select_screen(screen)
    expected = {run_id(row["fold"], row["reward_basis"], row["gae_lambda"], "screen", 42) for row in screen}
    if set(selected["completed_sha256"]) != expected or any(selected.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Credit admission changed or lacks complete original evidence")
    for identifier, digest in selected["completed_sha256"].items():
        if file_hash(Path(root) / "runs" / identifier / "completed.json") != digest:
            raise ValueError("Selected credit screen anchor changed")
    return selected


def cached_trial(directory, config, protocol, selection_sha256=None):
    directory = Path(directory)
    if not (directory / "completed.json").exists():
        return None
    saved = read_config(directory / "completed.json")
    if (saved["protocol_sha256"] != json_digest(protocol) or saved["config_sha256"] != json_digest(config)
            or saved["tool_sha256"] != protocol["tool_sha256"] or saved["selection_sha256"] != selection_sha256):
        raise ValueError("Credit cache cannot reuse a different estimator, runner or old sample")
    original.verify_leaf_files(directory, saved["files"])
    if read_config(directory / "config.json") != config:
        raise ValueError("Native credit configuration changed")
    row, data, spec = saved["record"], CreditData(config), config["timing"]
    value = config["ppo"]["gae_lambda"]
    if (row["fold"], row["reward_basis"], row["gae_lambda"], row["phase"], row["seed"], row["steps"]) != (spec["fold"], spec["reward_basis"], value, spec["phase"], config["seed"], BUDGETS[spec["phase"]]):
        raise ValueError("Credit checkpoint identity changed")
    for name, steps in (("initial_model", 0), ("final_model", row["steps"])):
        verify_model(directory / name, read_config(directory / name / "metadata.json"), config, protocol, data.contract, steps)
    audit = read_config(directory / "final_audit.json")
    if (row["audit"] != audit or audit["actual_timesteps"] != row["steps"] or audit["experience"]["total_steps"] != row["steps"]
            or audit["experience"]["benchmark_steps"] != row["steps"] or audit["optimizer_steps_min"] <= 0
            or audit["optimizer_steps_max"] != audit["actual_optimizer_updates"] or audit["parameters"] != 12805
            or audit["source_hashes"] != protocol["source_hashes"] or audit["tool_sha256"] != protocol["tool_sha256"]
            or audit.get("actual_model_gamma") != .995 or audit.get("actual_buffer_gamma") != .995
            or audit.get("actual_model_gae_lambda") != value or audit.get("actual_buffer_gae_lambda") != value
            or audit["model_sha256"] != file_hash(directory / "final_model" / "model.zip")
            or any(audit.get(key) is not True for key in ("all_parameters_finite", "save_load_parameters_equal", "save_load_macro_action_equal"))):
        raise ValueError("Credit actual optimizer and experience evidence changed")
    original.verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=row["steps"], expected_contract_sha256=json_digest(data.contract))
    original.verify_rollouts(directory, audit["rollout_buffers"], row["steps"], audit["training_trace_prefix"])
    verify_estimator_buffers(directory, audit, config)
    benchmark = baseline_metrics(protocol, spec["fold"])
    for mode in MODES:
        for name in ("initialized", "trained"):
            folder = directory / name / mode
            result = read_config(folder / "result.json")
            original._verify_result(folder, result, data.contract, mode, expected_dates=original.validation_dates(data))
            if (row["metrics"][mode][name] != result["metrics"] or row["metrics"][mode]["benchmark"] != result["benchmark_metrics"]
                    or result["benchmark_metrics"] != benchmark[mode] or row["evaluation_steps"][name][mode] != result["evaluation_steps"]):
                raise ValueError("Credit source economic metrics or full calendar changed")
            changes = shared.case_changes(folder, result)
            if name == "trained" and changes != row["changes"][mode] or name == "initialized" and any(changes.values()):
                raise ValueError("Credit behavior evidence differs from native execution")
    if spec["phase"] == "confirm":
        require_confirmation(directory.parent.parent, protocol, (spec["reward_basis"], value), selection_sha256)
    if value == .95 and spec["phase"] == "screen":
        previous = Path(protocol["reproduction_reference_root"]) / "runs" / f'{spec["phase"]}_{spec["fold"]}_{spec["reward_basis"]}_{config["seed"]}'
        if read_config(directory / "reproduction.json") != compare_control(directory, previous, audit):
            raise ValueError("Credit control reproduction evidence changed")
    return row


def run_trial(root, protocol, fold, reward, gae_lambda, phase, seed, selection_sha256=None):
    check_frozen(protocol)
    if phase == "confirm":
        require_confirmation(root, protocol, (reward, gae_lambda), selection_sha256)
    config = config_for(protocol, root, fold, reward, gae_lambda, phase, seed)
    identifier = run_id(fold, reward, gae_lambda, phase, seed)
    directory = Path(root) / "runs" / identifier
    cached = cached_trial(directory, config, protocol, selection_sha256)
    if cached is not None:
        return cached
    if directory.exists():
        raise ValueError("Incomplete credit trial requires explicit review")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    data, benchmark = CreditData(config), baseline_metrics(protocol, fold)
    if json_digest(data.stock_contract) != protocol["folds"][fold]["stock_contract_sha256"]:
        raise ValueError("Credit factor changed original economic execution or data")
    experience = original.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    model = make_model(env, config)
    initial = {key: tensor.detach().cpu().clone() for key, tensor in model.policy.state_dict().items()}
    agent = TimingAgent(model, data.contract, dict(credit_study=config["credit_study"]))
    agent.save(directory / "initial_model", config)
    for other_reward, other_value in CELLS:
        other = Path(root) / "runs" / run_id(fold, other_reward, other_value, phase, seed) / "initial_model"
        if other.exists() and other != directory / "initial_model":
            tensors = shared.policy_tensors(other)
            if set(tensors) != set(initial) or any(not torch.equal(tensors[key], initial[key]) for key in initial):
                raise ValueError("Credit factorial cells do not have identical initial Actor and Critic tensors")
    evaluation_steps = {"initialized": {}, "trained": {}}
    with preserve_training_rng(model.device):
        for mode in MODES:
            result = original.evaluate_case(agent, data, directory / "initialized" / mode, mode, "deterministic_initialized")
            if result["metrics"] != benchmark[mode] or any(shared.case_changes(directory / "initialized" / mode, result).values()):
                raise ValueError("Credit initialization differs from unchanged twenty-day momentum")
            evaluation_steps["initialized"][mode] = result["evaluation_steps"]
    callback = CreditCallback(data, directory, experience, initial, BUDGETS[phase], identifier)
    model.set_logger(configure(str(directory / "logs"), ["csv"]))
    try:
        model.learn(total_timesteps=BUDGETS[phase], callback=callback)
        model.logger.dump(step=model.num_timesteps)
        counts = original._optimizer_counts(model)
        if model.num_timesteps != BUDGETS[phase] or not counts or min(counts.values()) <= 0 or not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise ValueError("Credit checkpoint lacks actual finite final-budget optimizer updates")
        agent.save(directory / "final_model", config, dict(credit_study=config["credit_study"], fixed_final_budget=model.num_timesteps))
        with preserve_training_rng(model.device):
            restored = TimingAgent.load(directory / "final_model", data.contract, "cpu")
            if any(not torch.equal(value.detach().cpu(), restored.model.policy.state_dict()[key]) for key, value in model.policy.state_dict().items()):
                raise ValueError("Credit tensors changed during save/load")
            first, second = agent.inspect_action(model._last_obs[0]), restored.inspect_action(model._last_obs[0])
            np.testing.assert_allclose(first["action_probabilities"], second["action_probabilities"], rtol=1e-6, atol=1e-7)
            if first["macro_action"] != second["macro_action"]:
                raise ValueError("Credit deterministic action changed during save/load")
            del restored
        audit = dict(actual_timesteps=model.num_timesteps, actual_optimizer_updates=max(counts.values()), optimizer_steps_min=min(counts.values()),
            optimizer_steps_max=max(counts.values()), optimizer_steps_by_parameter=counts, parameters=sum(p.numel() for p in model.policy.parameters()),
            model_sha256=file_hash(directory / "final_model" / "model.zip"),
            tensor_delta_l2={key: float(torch.linalg.vector_norm(value.detach().cpu() - initial[key])) for key, value in model.policy.state_dict().items()},
            experience=experience.stats(), training_schedule=experience.env.schedule_audit(), training_trace_prefix=callback.trace.manifest(),
            rollout_buffers=callback.rollouts, mode_statistics=callback.trace.statistics.snapshot(), training_diagnostics=diagnostic_summary(callback.update_records),
            credit_estimator_buffers=callback.estimator_buffers,
            elapsed_seconds=perf_counter() - callback.started, source_hashes=protocol["source_hashes"], tool_sha256=protocol["tool_sha256"],
            actual_model_gamma=float(model.gamma), actual_buffer_gamma=float(model.rollout_buffer.gamma),
            actual_model_gae_lambda=float(model.gae_lambda), actual_buffer_gae_lambda=float(model.rollout_buffer.gae_lambda),
            save_load_parameters_equal=True, save_load_macro_action_equal=True, all_parameters_finite=True, test_evaluated=False)
        original.verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=model.num_timesteps, expected_contract_sha256=json_digest(data.contract))
        original.verify_rollouts(directory, audit["rollout_buffers"], model.num_timesteps, audit["training_trace_prefix"])
        verify_estimator_buffers(directory, audit, config)
        write_json(directory / "final_audit.json", audit)
        write_json(directory / "episodes.json", [item for item in experience.episodes if item["steps"]])
    finally:
        callback.trace.close()
        env.close()
    del model, agent, callback, env, experience, initial
    _release()
    agent = TimingAgent.load(directory / "final_model", data.contract, "cpu")
    metrics, changes = {}, {}
    for mode in MODES:
        result = original.evaluate_case(agent, data, directory / "trained" / mode, mode, "trained")
        if result["benchmark_metrics"] != benchmark[mode]:
            raise ValueError("Credit training changed the frozen reference portfolio")
        metrics[mode] = dict(trained=result["metrics"], initialized=benchmark[mode], benchmark=benchmark[mode])
        changes[mode] = shared.case_changes(directory / "trained" / mode, result)
        evaluation_steps["trained"][mode] = result["evaluation_steps"]
    row = dict(fold=fold, reward_basis=reward, gae_lambda=gae_lambda, method=reward, phase=phase, seed=seed, steps=BUDGETS[phase],
        metrics=metrics, changes=changes, audit=audit, evaluation_steps=evaluation_steps, model_path=str(directory / "final_model"), test_evaluated=False, actionable=False)
    if gae_lambda == .95 and phase == "screen":
        previous = Path(protocol["reproduction_reference_root"]) / "runs" / f'{phase}_{fold}_{reward}_{seed}'
        write_json(directory / "reproduction.json", compare_control(directory, previous, audit))
    check_frozen(protocol)
    write_json(directory / "completed.json", dict(protocol_sha256=json_digest(protocol), config_sha256=json_digest(config), tool_sha256=protocol["tool_sha256"],
        selection_sha256=selection_sha256, record=row, files=original._leaf_files(directory)))
    return row


def compare_control(directory, previous, audit):
    proof = shared.compare_reproduction(directory, previous, audit)
    old = read_config(Path(previous) / "final_audit.json")
    comparisons = (
        (Path(directory) / "training_trace" / "benchmark", audit["training_trace_prefix"]["benchmark_trace"]["chunks"],
         Path(previous) / "training_trace" / "benchmark", old["training_trace_prefix"]["benchmark_trace"]["chunks"]),
        (Path(directory), audit["rollout_buffers"], Path(previous), old["rollout_buffers"]),
    )
    for first_root, first_files, second_root, second_files in comparisons:
        if len(first_files) != len(second_files):
            raise ValueError("Credit control benchmark or PPO buffer count differs")
        for first, second in zip(first_files, second_files):
            if any(first[key] != second[key] for key in ("first_step", "last_step", "steps")):
                raise ValueError("Credit control benchmark or PPO buffer clocks differ")
            with np.load(first_root / first["path"], allow_pickle=False) as one, np.load(second_root / second["path"], allow_pickle=False) as two:
                if set(one.files) != set(two.files) or any(not np.array_equal(one[key], two[key], equal_nan=True) if one[key].dtype.kind in "fc" else not np.array_equal(one[key], two[key]) for key in one.files):
                    raise ValueError("Credit control benchmark or original PPO arrays do not reproduce")
    proof.update(benchmark_native_arrays_equal=True, rollout_buffers_equal=True)
    return proof


def run_worker(root, protocol, phase, fold, cell=None):
    if phase not in BUDGETS or fold not in (original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS) or phase == "confirm" and cell not in CELLS:
        raise ValueError("Unregistered exclusive credit worker")
    selection = file_hash(Path(root) / "selection.json") if phase == "confirm" else None
    if phase == "confirm":
        require_confirmation(root, protocol, cell, selection)
    completed = []
    for reward, value in CELLS if phase == "screen" else (cell,):
        for seed in (42,) if phase == "screen" else original.SEEDS:
            run_trial(root, protocol, fold, reward, value, phase, seed, selection)
            identifier = run_id(fold, reward, value, phase, seed)
            completed.append(dict(run_id=identifier, completed_sha256=file_hash(Path(root) / "runs" / identifier / "completed.json")))
            write_json(Path(root) / f'worker_{phase}_{fold}.json', dict(status="running", phase=phase, fold=fold,
                protocol_sha256=file_hash(Path(root) / "protocol.json"), selection_sha256=selection, completed_runs=completed, no_global_summary_writes=True))
    write_json(Path(root) / f'worker_{phase}_{fold}.json', dict(status="complete", phase=phase, fold=fold,
        protocol_sha256=file_hash(Path(root) / "protocol.json"), selection_sha256=selection, completed_runs=completed, no_global_summary_writes=True))
    return completed


def write_report(root, protocol, screen, selected):
    check_frozen(protocol)
    root = Path(root)
    if not (root / "selection.json").exists():
        raise ValueError("Report requires an already frozen credit selection")
    actual = load_stage(root, protocol, "screen")
    if actual != screen or seal_selection(root, protocol, actual) != selected:
        raise ValueError("Report cannot accept unsourced records or fabricated admission")
    cell = (selected["selected_reward"], selected["selected_lambda"])
    confirmation = load_stage(root, protocol, "confirm", cell) if selected["admitted"] else []
    summary = original.summarize_confirmation(confirmation, cell[0]) if confirmation else dict(qualified=False, qualified_candidate=None)
    summary.update(screen=selected, actual_screen_steps=65536, actual_confirmation_steps=sum(row["steps"] for row in confirmation),
        actual_training_steps=sum(row["steps"] for row in screen + confirmation), final_snapshots=len(screen) + len(confirmation),
        actual_optimizer_updates=sum(row["audit"]["actual_optimizer_updates"] for row in screen + confirmation),
        independent_training_benchmark_steps=sum(row["audit"]["experience"]["benchmark_steps"] for row in screen + confirmation),
        evaluation_agent_steps=sum(sum(values.values()) for row in screen + confirmation for values in row["evaluation_steps"].values()),
        evaluation_parallel_benchmark_steps=sum(sum(values.values()) for row in screen + confirmation for values in row["evaluation_steps"].values()),
        new_rule_control_steps=0, additional_reproduction_environment_steps=0, status="complete" if selected["admitted"] else "screen_failed",
        test_evaluated=False, actionable=False)
    write_json(root / "summary.json", screen + confirmation)
    write_json(root / "study_summary.json", summary)
    write_json(root / "status.json", {key: summary[key] for key in ("status", "actual_training_steps", "final_snapshots", "qualified", "test_evaluated", "actionable")})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "confirm", "report"), required=True)
    parser.add_argument("--output", default="reports/runs/ppo_timing_credit_20261015")
    parser.add_argument("--reference-root", default=str(REFERENCE))
    parser.add_argument("--fold", choices=(*original.SCREEN_FOLDS, *original.CONFIRM_FOLDS))
    args = parser.parse_args(argv)
    if args.fold is not None and (args.stage not in BUDGETS or args.fold not in (original.SCREEN_FOLDS if args.stage == "screen" else original.CONFIRM_FOLDS)):
        parser.error("--fold selects one exclusive screen or confirmation worker")
    root = Path(args.output).resolve()
    if args.stage != "prepare" and not (root / "protocol.json").exists():
        raise ValueError("Only prepare may register a new credit study")
    protocol = prepare(root, args.reference_root)
    check_frozen(protocol)
    if args.stage == "prepare":
        return protocol
    if args.stage == "screen" and args.fold is not None:
        return run_worker(root, protocol, "screen", args.fold)
    screen = load_stage(root, protocol, "screen")
    selected = seal_selection(root, protocol, screen)
    if args.stage == "screen":
        write_json(root / "status.json", dict(status=selected["status"], actual_training_steps=65536, test_evaluated=False, actionable=False))
        return selected
    if args.stage == "confirm":
        if not selected["admitted"]:
            raise ValueError("Screen failed; this batch authorizes zero confirmation")
        cell = (selected["selected_reward"], selected["selected_lambda"])
        return run_worker(root, protocol, "confirm", args.fold, cell) if args.fold is not None else original.summarize_confirmation(load_stage(root, protocol, "confirm", cell), cell[0])
    return write_report(root, protocol, screen, selected)


if __name__ == "__main__":
    main()
