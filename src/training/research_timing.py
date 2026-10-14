"""Frozen screen/confirmation study of a two-action momentum timing controller.

No downloads or protected-period evaluations occur here. Each trial saves only
its registered final budget; screen returns cannot select a shorter checkpoint.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import zipfile

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.timing_agent import TimingAgent
from src.agents.timing_agent import _feature_schema
from src.agents.timing_policy import CalendarTimingPolicy
from src.agents.trade_mapping import TradeActionWrapper
from src.backtest.metrics import performance
from src.backtest.run import git_metadata, run_policy
from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.envs.broker import BrokerConfig
from src.envs.initial_account import MODES
from src.envs.trading import TradingEnv
from src.runtime import events
from src.runtime.device import resolve_device
from .exploration_env import ExplorationData, EconomicRule
from .freeze import json_digest, source_hashes
from .matrix_report import _metric_mean, _metric_stats, _stats
from .research_delta import _registered_files, _verify_files, _release
from .research_exploration import config_for as exploration_config
from .research_matrix import preserve_training_rng, diagnostic_summary
from .research_trade import _trace_identity
from .timing_env import TimingEnv, FEATURE_NAMES, ACTION_NAMES, build_timing_contract
from .timing_trace import TimingTrace, verify_timing_trace


METHODS = ("log_net", "relative_log")
SCREEN_FOLDS = ("2023", "2024")
CONFIRM_FOLDS = ("2025", "2026")
SEEDS = (42, 123, 2026)
BUDGETS = {"screen": 8192, "confirm": 16384}
PARAMETERS = 12805
NEW_SOURCE_FILES = ("src/agents/timing_policy.py", "src/agents/timing_agent.py", "src/training/timing_env.py",
                    "src/training/research_timing.py", "src/training/timing_trace.py")
EVAL_SEED = 20260927
DOCUMENT = ROOT / "doc" / "ppo_timing_protocol_20261014.md"
REFERENCE_AUDIT_SHA256 = "89bc5f465eada6a845812b9f538886fc510dfcaecd392e54100b3f1f93379ce2"
DEFAULT_REFERENCE = ROOT / "reports" / "runs" / "ppo_priority_rank_20261011_16k"
DEFAULT_FOLDS = ROOT / "reports" / "runs" / "ppo_exploration_20261006"
DEFAULT_2026 = ROOT / "reports" / "runs" / "sina_hs300_study_20260927" / "research.json"


def config_for(base, root, fold_spec, fold, method, phase, seed):
    if (method not in METHODS or phase not in BUDGETS or seed not in SEEDS
            or fold not in (SCREEN_FOLDS if phase == "screen" else CONFIRM_FOLDS)
            or phase == "screen" and seed != 42):
        raise ValueError("Unregistered timing condition")
    config = exploration_config(base, root, "split_l1", seed)
    config.update(research_config=fold_spec["research_config"], action_mode="timing_control",
        device="cpu", torch_threads=1, hidden_sizes=[64, 64], top_n=20, top_n_choices=[],
        training_episode_steps=240, recent_episode_every=0,
        timing=dict(fold=fold, phase=phase, reward_basis=method, calendar_margin=2., period=20),
        output_root=str(Path(root) / "runs"))
    config["exploration"]["reward_basis"] = "log_net"
    config["ppo"].update(learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=5,
        gamma=.995, gae_lambda=.95, clip_range=.2, target_kl=.03,
        ent_coef=0., vf_coef=.5, max_grad_norm=.5)
    return config


def trial_config(protocol, root, fold, method, phase, seed):
    return config_for(protocol["base_config"], root, protocol["folds"][fold], fold, method, phase, seed)


class TimingData(ExplorationData):
    """Physically cropped old market support with a new categorical contract."""

    def __init__(self, config):
        borrowed = deepcopy(config)
        borrowed["action_mode"] = "stock_trade"
        super().__init__(borrowed)
        spec = config["timing"]
        if (config["action_mode"] != "timing_control" or spec["reward_basis"] not in METHODS
                or spec["calendar_margin"] != 2. or spec["period"] != 20
                or config["top_n"] != 20 or config["top_n_choices"]):
            raise ValueError("Invalid timing prediction contract")
        self.config = deepcopy(config)
        self.stock_contract = deepcopy(self.contract)
        self.contract = build_timing_contract(self.stock_contract, spec["reward_basis"], calendar_margin=2.)
        self.contract["timing_research"] = deepcopy(spec)

    def env(self, split):
        training = split == "train"
        market = self.market(split)
        # TimingEnv alone owns independent date/mode/phase/account RNGs.
        market.sampler = None
        base = TradingEnv(market, self.settings["initial_capital"], BrokerConfig(**self.settings["broker"]),
            initial_portfolio=self.config.get("initial_portfolio", "mixed") if training else self.settings.get("initial_portfolio", "cash"),
            max_steps=self.config["training_episode_steps"] if training else None,
            random_start=False, recent_episode_every=0)
        stock = TradeActionWrapper(base, self.config["logit_bound"], self.config["max_trade_weight"],
            self.config["max_commission_ratio"], top_n=20, top_n_choices=None)
        return TimingEnv(stock, self.contract, training=training, reward_basis=self.config["timing"]["reward_basis"],
                         episode_steps=self.config["training_episode_steps"])


def make_model(env, config):
    torch.set_num_threads(config["torch_threads"])
    return PPO(CalendarTimingPolicy, env, seed=config["seed"], device=resolve_device(config["device"]), verbose=0,
        policy_kwargs=dict(calendar_margin=2., net_arch={"pi": [], "vf": [64, 64]}), **config["ppo"])


def _leaf_files(directory):
    files = _registered_files(Path(directory))
    return {name: digest for name, digest in files.items() if name != "completed.json"}


def verify_leaf_files(directory, files):
    directory = Path(directory)
    _verify_files(directory, files)
    if set(_leaf_files(directory)) != set(files):
        raise ValueError("Registered leaf set changed, including an unregistered extra file")


def _trusted_reference(reference_root):
    reference_root = Path(reference_root).resolve()
    if file_hash(reference_root / "final_audit.json") != REFERENCE_AUDIT_SHA256:
        raise ValueError("Trusted Oct11 final audit changed")
    audit = read_config(reference_root / "final_audit.json")
    reference = read_config(reference_root / "protocol.json")
    if audit.get("status") != "passed" or file_hash(reference_root / "protocol.json") != audit.get("signed_files_sha256", {}).get("protocol.json"):
        raise ValueError("Trusted reference protocol changed")
    for name, digest in reference["source_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("An original source changed: " + name)
    current = source_hashes()
    allowed = {str(Path(name)) for name in NEW_SOURCE_FILES}
    if len(reference["source_hashes"]) != 91 or set(current) != set(reference["source_hashes"]) | allowed:
        raise ValueError("Original source inventory or the five explicitly new timing modules changed")
    return reference


def check_frozen(protocol):
    if source_hashes() != protocol["source_hashes"] or file_hash(DOCUMENT) != protocol["document_sha256"]:
        raise ValueError("Timing source or preregistration changed")
    _trusted_reference(protocol["reference_root"])
    if file_hash(Path(protocol["reference_root"]) / "protocol.json") != protocol["reference_protocol_sha256"]:
        raise ValueError("Trusted reference identity changed")
    for fold, item in protocol["folds"].items():
        directory = Path(item["directory"])
        if file_hash(directory / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Timing dataset manifest changed: " + fold)
        _verify_files(directory, item["files"])
        if file_hash(Path(item["research_config"])) != item["research_sha256"]:
            raise ValueError("Timing research config changed: " + fold)


def prepare(root, reference_root=DEFAULT_REFERENCE, folds_root=DEFAULT_FOLDS, research_2026=DEFAULT_2026):
    root, reference_root, folds_root = Path(root).resolve(), Path(reference_root).resolve(), Path(folds_root).resolve()
    research_2026 = Path(research_2026).resolve()
    for parent in (reference_root, folds_root, research_2026.parent):
        if root.is_relative_to(parent) or parent.is_relative_to(root):
            raise ValueError("Timing output must not overlap any immutable reference")
    reference = _trusted_reference(reference_root)
    path = root / "protocol.json"
    if path.exists():
        protocol = read_config(path)
        if (protocol["reference_root"] != str(reference_root) or protocol["folds_root"] != str(folds_root)
                or protocol["research_2026"] != str(research_2026)):
            raise ValueError("Registration cannot replace its data references")
        check_frozen(protocol)
        return protocol
    if root.exists():
        raise ValueError("Unregistered output directory requires review")
    old = read_config(folds_root / "protocol.json")
    folds = {}
    for fold in (*SCREEN_FOLDS, *CONFIRM_FOLDS):
        if fold == "2026":
            research_path = research_2026
            research = read_config(research_path)
            directory = Path(research["dataset"]["clean_dir"])
            directory = (ROOT / directory).resolve() if not directory.is_absolute() else directory.resolve()
        else:
            registered = old["folds"][fold]
            directory = Path(registered["directory"]).resolve()
            if file_hash(directory / "manifest.json") != registered["manifest_sha256"]:
                raise ValueError("Old historical manifest changed")
            _verify_files(directory, registered["artifacts"])
            research_path = directory / "research.json"
        manifest = read_config(directory / "manifest.json")
        if (manifest["train_end"] != fold + "-03-31" or manifest["val_end"] != fold + "-06-30"
                or fold != "2026" and (not manifest.get("protected_period_excluded") or manifest["end_date"] > "2026-06-30")):
            raise ValueError("Timing chronological boundaries do not match registration")
        files = {name: file_hash(directory / name) for name in ("panel_data.parquet", "prices.parquet", "scaler_params.json")}
        item = dict(directory=str(directory), manifest_sha256=file_hash(directory / "manifest.json"), files=files,
            research_config=str(research_path), research_sha256=file_hash(research_path),
            train_end=manifest["train_end"], val_end=manifest["val_end"], provider_physically_cropped=True)
        data = TimingData(config_for(reference["base_config"], root, item, fold, "log_net",
                                   "screen" if fold in SCREEN_FOLDS else "confirm", 42))
        if data.contract.get("data_source") != "sina" or len(data.codes) != 300 or len(data.manifest["feature_cols"]) != 20:
            raise ValueError("Timing study requires the unchanged Sina 300-stock 20-feature data")
        item["stock_contract_sha256"] = json_digest(data.stock_contract)
        folds[fold] = item
    protocol = dict(schema_version=1, study="timing_controller", methods=list(METHODS), seeds=list(SEEDS),
        screen_folds=list(SCREEN_FOLDS), confirm_folds=list(CONFIRM_FOLDS), budgets=BUDGETS,
        folds=folds, base_config=reference["base_config"], reference_root=str(reference_root),
        folds_root=str(folds_root), research_2026=str(research_2026),
        reference_protocol_sha256=file_hash(reference_root / "protocol.json"),
        reference_final_audit_sha256=REFERENCE_AUDIT_SHA256, original_source_count=len(reference["source_hashes"]),
        source_hashes=source_hashes(), document_sha256=file_hash(DOCUMENT), git=git_metadata(),
        feature_names=list(FEATURE_NAMES), actions=["HOLD", "REBALANCE_M20"], policy={"actor": "linear", "critic": [64, 64], "calendar_margin": 2.},
        parameters=PARAMETERS,
        gamma=.995, learning_rate=3e-4, device="cpu", evaluation_seed=EVAL_SEED, top_n=20,
        episode_steps=240, screen_trials=4, confirmation_trials=6, planned_trials=10,
        planned_screen_steps=32768, planned_confirmation_steps=98304, planned_actual_training_steps=131072,
        planned_final_snapshots=10, checkpoint_selection="registered final budget only",
        planned_initialized_paths=40, planned_trained_paths=40, planned_rule_paths=16,
        planned_parallel_benchmark_paths=96, planned_legacy_baseline_identity_paths=16,
        selection_rule="maximize minimum fold mean baseline gap, then mean gap; minimize worst DD, fees+slippage; method lexical tie",
        gate=dict(each_fold_mean_baseline_gap_positive=True, each_fold_positive_gap_seeds=2,
                  across_fold_mean_baseline_gap_min=.005, max_all_account_drawdown=.10,
                  max_drawdown_excess_over_baseline=.02, own_initialization_gain_positive=True),
        protected_period_start="2026-07-01", test_evaluated=False, actionable=False)
    root.mkdir(parents=True, exist_ok=False)
    write_json(path, protocol)
    write_json(root / "status.json", dict(status="prepared", test_evaluated=False, actionable=False))
    check_frozen(protocol)
    return protocol


def _audit_ledger(previous, info, market):
    orders = info["orders"]
    fees = sum(order["fee"] for order in orders)
    slippage = sum(order["slippage_cost"] for order in orders)
    net = sum(order["filled_shares"] * order["price"] for order in orders if order["filled_shares"])
    gross = sum(abs(order["filled_shares"]) * order["price"] for order in orders if order["filled_shares"])
    positions = previous["positions"].copy() + info["event_shares"]
    index = {code: i for i, code in enumerate(market.stock_codes)}
    for order in orders:
        positions[index[order["code"]]] += order["filled_shares"]
    np.testing.assert_allclose(info["positions"], positions, rtol=0, atol=1e-8)
    np.testing.assert_allclose(info["cash"], previous["cash"] + info["event_cash"] - net - fees, rtol=0, atol=1e-6)
    np.testing.assert_allclose(info["total_fees"], previous["total_fees"] + fees, rtol=0, atol=1e-6)
    close = market.prices.iloc[market.current_index].to_numpy()
    held = info["positions"] > 0
    np.testing.assert_allclose(info["equity"], info["cash"] + info["receivables"] + np.dot(info["positions"][held], close[held]), rtol=0, atol=1e-6)
    return dict(date=info["date"], equity=info["equity"], cash=info["cash"], daily_return=info["raw_net_return"] if "raw_net_return" in info else info["reward"],
        fee=fees, slippage_cost=slippage, turnover=gross / previous["equity"], holding_count=int(held.sum()),
        unfilled_orders=sum(order["status"] != "filled" for order in orders))


def _initial_daily(info):
    return dict(date=info["date"], equity=info["equity"], cash=info["cash"], daily_return=0., fee=0.,
                slippage_cost=0., turnover=0., holding_count=int((info["positions"] > 0).sum()), unfilled_orders=0)


def _daily_frame(rows):
    frame = pd.DataFrame(rows)
    frame["net_value"] = frame.equity / frame.equity.iloc[0]
    frame["drawdown"] = frame.equity / frame.equity.cummax() - 1
    return frame


def evaluate_case(agent, data, folder, mode, label):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
    env = data.env("val")
    identity = _trace_identity(data, label, seed=EVAL_SEED, initial_mode=mode, top_n=20,
        timing_observation_width=len(FEATURE_NAMES), reward_basis=data.config["timing"]["reward_basis"],
        has_neural_macro_likelihood=agent is not None)
    trace = TimingTrace(folder / "trace", identity)
    observation, initial = env.reset(seed=EVAL_SEED, options={"initial_mode": mode})
    previous = initial
    benchmark_previous = initial["timing"]["benchmark_ledger"]
    daily, benchmark_daily = [_initial_daily(initial)], [_initial_daily(benchmark_previous)]
    positions, macro_counts = [initial["positions"].copy()], Counter()
    try:
        while True:
            events.check_cancel()
            vector = observation.copy()
            stock_obs = deepcopy(env.stock_observation)
            if agent is None:
                inspect = None
                macro = int(np.argmax(vector[:20]) == 0)
            else:
                inspect = agent.inspect_action(vector)
                macro = inspect["macro_action"]
            observation, reward, terminated, truncated, info = env.step(macro)
            if not env.observation_space.contains(observation):
                raise ValueError("Invalid timing controller observation")
            daily.append(_audit_ledger(previous, info, env.env.unwrapped.market))
            benchmark = info["timing"]["benchmark_ledger"]
            benchmark_daily.append(_audit_ledger(benchmark_previous, benchmark, env.benchmark.unwrapped.market))
            trace.append(previous["date"], stock_obs, env.last_order, info, timing_observation=vector, macro_action=macro,
                log_prob=inspect["raw_log_prob"] if inspect else None, value=inspect["value"] if inspect else None,
                action_probabilities=inspect["action_probabilities"] if inspect else None, reward=reward, done=terminated or truncated)
            macro_counts.update([str(macro)])
            positions.append(info["positions"].copy())
            previous, benchmark_previous = info, benchmark
            if terminated or truncated:
                break
    finally:
        trace.close()
        env.close()
    daily, benchmark_daily = _daily_frame(daily), _daily_frame(benchmark_daily)
    daily.to_csv(folder / "daily.csv", index=False)
    benchmark_daily.to_csv(folder / "benchmark_daily.csv", index=False)
    pd.DataFrame(positions, index=daily.date, columns=data.codes).to_parquet(folder / "positions.parquet")
    pd.DataFrame(env.env.unwrapped.broker.orders).to_csv(folder / "orders.csv", index=False)
    pd.DataFrame(env.benchmark.unwrapped.broker.orders).to_csv(folder / "benchmark_orders.csv", index=False)
    verify_timing_trace(folder / "trace", trace.manifest(), steps=len(daily) - 1, expected_contract_sha256=json_digest(data.contract))
    result = dict(metrics=performance(daily), benchmark_metrics=performance(benchmark_daily), trace=trace.manifest(),
        mode_statistics=trace.statistics.snapshot(), macro_counts=dict(macro_counts), contract=data.contract, initial_mode=mode,
        top_n=20, evaluation_steps=len(daily) - 1, benchmark_steps=len(benchmark_daily) - 1,
        first_execution=str(pd.Timestamp(daily.date.iloc[1]).date()), last_execution=str(pd.Timestamp(daily.date.iloc[-1]).date()),
        assumptions=info["assumptions"], test_evaluated=False, actionable=False)
    if result["last_execution"] >= "2026-07-01":
        raise ValueError("Protected period was evaluated")
    verify_evaluation_dates(folder, result, validation_dates(data))
    write_json(folder / "result.json", result)
    return result


def rule_controls(root, protocol, fold, *, create=True):
    directory = Path(root) / "rules" / fold
    marker = directory / "completed.json"
    if marker.exists():
        saved = read_config(marker)
        if saved["protocol_sha256"] != json_digest(protocol) or saved.get("exact_legacy_ledger") is not True:
            raise ValueError("Baseline controls belong to another registration")
        verify_leaf_files(directory, saved["files"])
        phase = "screen" if fold in SCREEN_FOLDS else "confirm"
        data = TimingData(trial_config(protocol, root, fold, "log_net", phase, 42))
        for mode in MODES:
            result = read_config(directory / mode / "result.json")
            _verify_result(directory / mode, result, data.contract, mode, expected_dates=validation_dates(data))
            if result["metrics"] != saved["metrics"][mode] or result["metrics"] != result["benchmark_metrics"]:
                raise ValueError("Frozen baseline metrics differ from their native evidence")
        return saved["metrics"]
    if not create or directory.exists():
        raise ValueError("Incomplete baseline controls require review")
    phase = "screen" if fold in SCREEN_FOLDS else "confirm"
    data = TimingData(trial_config(protocol, root, fold, "log_net", phase, 42))
    metrics, evaluation_steps = {}, {}
    for mode in MODES:
        result = evaluate_case(None, data, directory / mode, mode, "calendar_rule")
        np.testing.assert_allclose(result["metrics"]["cumulative_return"], result["benchmark_metrics"]["cumulative_return"], atol=0., rtol=0.)
        # A separate unchanged old-engine path checks cloning/reset/phase semantics.
        data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        old_daily, old_positions, old_orders, _ = run_policy(data.market("val"), EconomicRule(data.stock_contract, "momentum_20", 20), data.settings)
        saved_daily = pd.read_csv(directory / mode / "daily.csv", float_precision="round_trip")
        for field in ("equity", "cash", "daily_return", "fee", "slippage_cost", "turnover", "holding_count", "unfilled_orders"):
            np.testing.assert_array_equal(saved_daily[field], old_daily[field])
        np.testing.assert_allclose(pd.read_parquet(directory / mode / "positions.parquet"), old_positions, rtol=0., atol=0.)
        old_daily.to_csv(directory / mode / "legacy_daily.csv", index=False)
        old_orders.to_csv(directory / mode / "legacy_orders.csv", index=False)
        metrics[mode] = result["metrics"]
        evaluation_steps[mode] = result["evaluation_steps"]
    write_json(marker, dict(protocol_sha256=json_digest(protocol), metrics=metrics, evaluation_steps=evaluation_steps,
        agent_steps=sum(evaluation_steps.values()), parallel_benchmark_steps=sum(evaluation_steps.values()),
        legacy_identity_steps=sum(evaluation_steps.values()), exact_legacy_ledger=True, files=_leaf_files(directory)))
    return metrics


class TimingExperience(gym.Wrapper):
    """Audit Box-input episodes without assuming a legacy Dict observation."""

    def __init__(self, env):
        super().__init__(env)
        self.episodes, self.dates, self.modes, self.strata, self.phases = [], Counter(), Counter(), Counter(), Counter()

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        timing = info["timing"]
        account = dict(positions=np.asarray(info["positions"]).tolist(), cash=float(info["cash"]), receivables=float(info.get("receivables", 0.)))
        self.active = dict(episode=len(self.episodes) + 1, start_date=str(info["date"]), mode=info["initial_account"]["mode"],
            phase=int(np.argmax(observation[:20])), date_stratum=timing["schedule"]["date_stratum"],
            phase_bin=timing["schedule"]["phase_bin"],
            account_sha256=json_digest(account), initial_account=account, steps=0, completed=False, rewards=[], benchmark_returns=[], raw_returns=[])
        self.episodes.append(self.active)
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        _, reward, terminated, truncated, info = result
        timing = info["timing"]
        self.active["steps"] += 1
        self.active["rewards"].append(float(reward))
        self.active["benchmark_returns"].append(float(timing["benchmark_net_return"]))
        self.active["raw_returns"].append(float(timing["raw_net_return"]))
        self.active.update(completed=bool(terminated or truncated), terminated=bool(terminated), truncated=bool(truncated), end_date=str(info["date"]))
        self.dates.update([str(info["decision_snapshot"]["date"])])
        self.modes.update([self.active["mode"]])
        self.strata.update([str(self.active["date_stratum"])])
        self.phases.update([str(timing["phase"])])
        return result

    def stats(self):
        experienced = [episode for episode in self.episodes if episode["steps"]]
        return dict(total_steps=sum(self.dates.values()), benchmark_steps=sum(self.dates.values()),
            started_episodes=len(experienced), completed_episodes=sum(e["completed"] for e in experienced),
            partial_episode_steps=sum(e["steps"] for e in experienced if not e["completed"]),
            unique_initial_accounts=len({e["account_sha256"] for e in experienced}), unique_decision_dates=len(self.dates),
            mode_steps=dict(self.modes), date_stratum_steps=dict(self.strata), phase_steps=dict(self.phases),
            mode_date_phase_episodes=dict(Counter(str((e["mode"], e["date_stratum"], e["phase_bin"])) for e in experienced)),
            decision_date_counts=dict(sorted(self.dates.items())))


def _optimizer_counts(model):
    counts = {name: int(model.policy.optimizer.state[p]["step"].item()) for name, p in model.policy.named_parameters()
              if p in model.policy.optimizer.state and "step" in model.policy.optimizer.state[p]}
    return counts


class TimingCallback(BaseCallback):
    """Discrete samples and actual Adam evidence, with no .network assumptions."""

    def __init__(self, data, output, experience, initial, budget, run_id):
        super().__init__()
        self.data, self.output, self.experience, self.initial = data, Path(output), experience, initial
        self.budget, self.run_id = budget, run_id
        self.started, self.update_records, self.last_optimizer = perf_counter(), [], 0
        self.rollouts = []
        self.trace = TimingTrace(self.output / "training_trace", _trace_identity(data, "sampled_training", run_id=run_id,
                    timing_observation_width=len(FEATURE_NAMES), reward_basis=data.config["timing"]["reward_basis"],
                    has_neural_macro_likelihood=True))

    def _on_step(self):
        events.check_cancel()
        for index, info in enumerate(self.locals["infos"]):
            def item(name):
                value = self.locals[name][index]
                return value.detach().cpu().numpy() if torch.is_tensor(value) else value
            timing = info["timing"]
            diagnostics = info["action_diagnostics"]
            from src.agents.trade_mapping import TradeOrder
            order = TradeOrder(diagnostics["shares"], diagnostics["freeze_mask"], diagnostics["exit_mask"],
                               diagnostics["priority"], diagnostics["top_n"], diagnostics)
            self.trace.append(info["decision_snapshot"]["date"], info["timing_stock_observation"], order, info,
                timing_observation=self.model._last_obs[index], macro_action=item("actions"), log_prob=item("log_probs"),
                value=item("values"), reward=timing["learning_reward"], done=item("dones"))
        return True

    def _record_update(self):
        counts = _optimizer_counts(self.model)
        updates = max(counts.values(), default=0)
        if updates == self.last_optimizer:
            return
        row = {key: float(value) for key, value in self.model.logger.name_to_value.items()
               if isinstance(value, (int, float, np.number)) and np.isfinite(value)}
        row.update(timesteps=self.model.num_timesteps, actual_optimizer_steps=updates, applied_optimizer_steps=updates - self.last_optimizer,
                   elapsed_seconds=perf_counter() - self.started)
        self.update_records.append(row)
        self.last_optimizer = updates
        pd.DataFrame(self.update_records).to_csv(self.output / "training_updates.csv", index=False)
        write_json(self.output / "status.json", dict(status="training", run_id=self.run_id, steps=self.model.num_timesteps,
            budget=self.budget, diagnostics=row, test_evaluated=False))
        print(f'PROGRESS {self.run_id} {self.model.num_timesteps}/{self.budget} Adam={updates}', flush=True)

    def _on_rollout_start(self):
        self._record_update()

    def _on_rollout_end(self):
        # Buffer rewards include SB3's TimeLimit bootstrap, unlike realized net
        # rewards in the native trace. Keep both with explicitly distinct names.
        buffer = self.model.rollout_buffer
        directory = self.output / "rollout_buffers"
        directory.mkdir(exist_ok=True)
        path = directory / f'rollout_{len(self.rollouts) + 1:03d}.npz'
        arrays = {name: np.asarray(getattr(buffer, name)).copy() for name in
                  ("observations", "actions", "rewards", "values", "log_probs", "returns", "advantages", "episode_starts")}
        if any(not np.isfinite(array).all() or array.dtype == object for array in arrays.values()):
            raise FloatingPointError("Timing PPO rollout contains nonfinite or nonnative evidence")
        np.savez_compressed(path, **arrays)
        self.rollouts.append(dict(path=path.relative_to(self.output).as_posix(), sha256=file_hash(path),
            first_step=self.model.num_timesteps - self.model.n_steps + 1, last_step=self.model.num_timesteps,
            steps=self.model.n_steps, likelihood="categorical macro old log probability",
            reward_semantics="PPO buffer reward including standard TimeLimit terminal-value bootstrap"))

    def _on_training_end(self):
        self._record_update()
        self.trace.close()


def _policy_bytes(directory):
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        return archive.read("policy.pth")


def validation_dates(data):
    """Actual full validation calendar, derived only from the registered panel."""
    dates = pd.DatetimeIndex(data.features.index)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Registered timing panel dates must be unique and ordered")
    before = dates[dates <= pd.Timestamp(data.manifest["train_end"])]
    if not len(before):
        raise ValueError("Validation has no final training-close initializer")
    expected = dates[(dates >= before[-1]) & (dates <= pd.Timestamp(data.manifest["val_end"]))]
    if len(expected) < 2 or expected[-1] >= pd.Timestamp("2026-07-01"):
        raise ValueError("Invalid full timing validation calendar")
    return expected


def verify_evaluation_dates(directory, result, expected_dates):
    """Reject shortened or shifted Q2 curves without loading or replaying a model."""
    directory = Path(directory)
    expected = pd.DatetimeIndex(expected_dates)
    steps = len(expected) - 1
    if (steps < 1 or result["evaluation_steps"] != steps or result["benchmark_steps"] != steps
            or result["first_execution"] != str(expected[1].date())
            or result["last_execution"] != str(expected[-1].date())):
        raise ValueError("Timing result does not cover the complete registered validation calendar")
    for name in ("daily.csv", "benchmark_daily.csv"):
        actual = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(directory / name, usecols=["date"])["date"]))
        if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
            raise ValueError("Timing daily dates omit or shift a registered validation row")
    positions = pd.read_parquet(directory / "positions.parquet")
    if not np.array_equal(pd.DatetimeIndex(positions.index).to_numpy(), expected.to_numpy()):
        raise ValueError("Timing holdings dates omit or shift a registered validation row")
    decision, execution = [], []
    for chunk in result["trace"]["chunks"]:
        with np.load(directory / "trace" / chunk["path"], allow_pickle=False) as native:
            decision.extend(native["decision_date"].tolist())
            execution.extend(native["execution_date"].tolist())
    if (not np.array_equal(pd.to_datetime(decision).to_numpy(), expected[:-1].to_numpy())
            or not np.array_equal(pd.to_datetime(execution).to_numpy(), expected[1:].to_numpy())):
        raise ValueError("Native timing decisions or executions do not cover the complete validation calendar")
    return expected


def _verify_result(directory, result, contract, mode, *, expected_dates=None):
    identity = result["trace"]["identity"]
    if (result["contract"] != contract or result["initial_mode"] != mode or result["top_n"] != 20
            or result["test_evaluated"] is not False or result["last_execution"] >= "2026-07-01"
            or result["evaluation_steps"] != result["benchmark_steps"]
            or identity.get("seed") != EVAL_SEED or identity.get("initial_mode") != mode
            or identity.get("codes") != contract["stock_codes"] or identity.get("top_n") != 20):
        raise ValueError("Timing evaluation identity or protected boundary differs")
    verify_timing_trace(Path(directory) / "trace", result["trace"], steps=result["evaluation_steps"], expected_contract_sha256=json_digest(contract))
    if expected_dates is not None:
        verify_evaluation_dates(directory, result, expected_dates)
    for key in ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost"):
        for name in ("metrics", "benchmark_metrics"):
            if not np.isfinite(result[name][key]):
                raise ValueError("Timing result contains nonfinite economic evidence")


def cached_trial(directory, config, protocol, selection_sha256=None):
    directory = Path(directory)
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    saved = read_config(marker)
    if (saved["protocol_sha256"] != json_digest(protocol) or saved["config_sha256"] != json_digest(config)
            or saved["selection_sha256"] != selection_sha256):
        raise ValueError("Timing cache belongs to a different registration or selection")
    verify_leaf_files(directory, saved["files"])
    if read_config(directory / "config.json") != config:
        raise ValueError("Cached timing configuration changed")
    row, audit = saved["record"], read_config(directory / "final_audit.json")
    spec = config["timing"]
    expected = (spec["fold"], spec["reward_basis"], spec["phase"], config["seed"], BUDGETS[spec["phase"]])
    if (row["fold"], row["method"], row["phase"], row["seed"], row["steps"]) != expected:
        raise ValueError("Cached timing condition changed")
    metadata = read_config(directory / "final_model" / "metadata.json")
    initial = read_config(directory / "initial_model" / "metadata.json")
    data = TimingData(config)
    expected_contract = data.contract
    expected_dates = validation_dates(data)
    _validate_snapshot_metadata(directory / "initial_model", initial, config, protocol, expected_contract, 0)
    _validate_snapshot_metadata(directory / "final_model", metadata, config, protocol, expected_contract, row["steps"])
    if (metadata["num_timesteps"] != row["steps"] or initial["num_timesteps"] != 0
            or metadata["config"] != config or initial["config"] != config
            or metadata["contract"] != initial["contract"]
            or metadata["source_hashes"] != protocol["source_hashes"] or initial["source_hashes"] != protocol["source_hashes"]
            or metadata.get("algorithm_class") != "stable_baselines3.ppo.ppo.PPO"
            or initial.get("algorithm_class") != "stable_baselines3.ppo.ppo.PPO"
            or metadata["contract"].get("timing_research") != spec
            or metadata["contract"]["timing"]["reward_basis"] != spec["reward_basis"]
            or metadata.get("feature_names") != protocol["feature_names"]
            or initial.get("feature_names") != protocol["feature_names"]
            or metadata["policy_record"] != initial["policy_record"]
            or file_hash(directory / "initial_model" / "model.zip") != initial["model_sha256"]
            or file_hash(directory / "final_model" / "model.zip") != row["model_sha256"]
            or metadata["model_sha256"] != row["model_sha256"] or audit["model_sha256"] != row["model_sha256"]):
        raise ValueError("Cached timing model identity changed")
    if (audit["actual_timesteps"] != row["steps"] or audit["experience"]["total_steps"] != row["steps"]
            or audit["experience"]["benchmark_steps"] != row["steps"]
            or audit["parameters"] != PARAMETERS or protocol["parameters"] != PARAMETERS
            or audit["optimizer_steps_min"] <= 0 or audit["optimizer_steps_max"] != audit["actual_optimizer_updates"]):
        raise ValueError("Cached timing budget or actual optimizer evidence changed")
    if row["audit"] != audit:
        raise ValueError("Cached record audit changed")
    verify_rollouts(directory, audit["rollout_buffers"], row["steps"], audit["training_trace_prefix"])
    verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=row["steps"], expected_contract_sha256=json_digest(metadata["contract"]))
    if spec["phase"] == "confirm":
        validate_confirmation_selection(directory.parent.parent, protocol, spec["reward_basis"], selection_sha256)
    for mode in MODES:
        for name in ("trained", "initialized"):
            result = read_config(directory / name / mode / "result.json")
            _verify_result(directory / name / mode, result, metadata["contract"], mode, expected_dates=expected_dates)
            if row["metrics"][mode][name] != result["metrics"]:
                raise ValueError("Cached timing metrics changed")
            field = "trained_evaluation_steps" if name == "trained" else "initialized_evaluation_steps"
            if row[field][mode] != result["evaluation_steps"]:
                raise ValueError("Cached timing evaluation step counts changed")
            if row["metrics"][mode]["benchmark"] != result["benchmark_metrics"]:
                raise ValueError("Per-trial unchanged benchmark changed")
        if row["metrics"][mode]["initialized"] != row["metrics"][mode]["benchmark"]:
            raise ValueError("Calendar initialization no longer matches the fixed benchmark")
    return row


def _validate_snapshot_metadata(directory, metadata, config, protocol, contract, steps):
    record = dict(policy_class="src.agents.timing_policy.CalendarTimingPolicy", observation_shape=[len(FEATURE_NAMES)],
        observation_dtype="float32", feature_names=list(FEATURE_NAMES), action_names=list(ACTION_NAMES),
        net_arch={"pi": [], "vf": [64, 64]}, calendar_margin=2., phase_period=20, parameter_count=PARAMETERS)
    if (metadata.get("contract") != contract or metadata.get("config") != config
            or metadata.get("algorithm_class") != "stable_baselines3.ppo.ppo.PPO"
            or metadata.get("num_timesteps") != steps or metadata.get("actual_device") != "cpu"
            or config.get("device") != "cpu" or metadata.get("feature_names") != list(FEATURE_NAMES)
            or metadata.get("policy_record") != record or metadata.get("source_hashes") != protocol["source_hashes"]
            or file_hash(Path(directory) / "model.zip") != metadata.get("model_sha256")):
        raise ValueError("Cached timing model identity or registered CPU policy schema changed")
    _feature_schema(contract)
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        data = json.loads(archive.read("data"))
    if (data.get("timing_registered_algorithm") != "stable_baselines3.ppo.ppo.PPO"
            or data.get("timing_registered_contract") != contract or data.get("timing_feature_names") != list(FEATURE_NAMES)
            or data.get("timing_policy_record") != record or data.get("num_timesteps") != steps):
        raise ValueError("Timing archive registration differs from the strict snapshot metadata")


def validate_confirmation_selection(root, protocol, method, selection_sha256):
    root = Path(root)
    path = root / "selection.json"
    if not path.exists() or selection_sha256 is None or file_hash(path) != selection_sha256:
        raise ValueError("Confirmation requires the immutable completed-screen selection identity")
    selection = read_config(path)
    expected = {f'{fold}_{candidate}' for fold in SCREEN_FOLDS for candidate in METHODS}
    anchors = selection.get("screen_completed_sha256", {})
    if (selection.get("protocol_sha256") != json_digest(protocol) or selection.get("selected_method") != method
            or set(anchors) != expected):
        raise ValueError("Confirmation method differs from the frozen screen-selected method")
    rows = []
    for fold in SCREEN_FOLDS:
        verify_worker_receipt(root, protocol, "screen", fold)
        for candidate in METHODS:
            marker = root / "runs" / f'screen_{fold}_{candidate}_42' / "completed.json"
            if file_hash(marker) != anchors[f'{fold}_{candidate}']:
                raise ValueError("Screen completion used by the frozen selection changed")
            saved = read_config(marker)
            if saved.get("protocol_sha256") != json_digest(protocol):
                raise ValueError("Screen marker belongs to another registration")
            rows.append(saved["record"])
    recomputed = select_screen(rows)
    if any(selection.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Frozen screen selection no longer follows the registered worst-fold rule")
    return selection


def run_trial(root, protocol, fold, method, phase, seed, selection_sha256=None):
    root = Path(root)
    check_frozen(protocol)
    if phase == "confirm":
        validate_confirmation_selection(root, protocol, method, selection_sha256)
    config = trial_config(protocol, root, fold, method, phase, seed)
    run_id = f'{phase}_{fold}_{method}_{seed}'
    directory = root / "runs" / run_id
    cached = cached_trial(directory, config, protocol, selection_sha256)
    if cached is not None:
        return cached
    if directory.exists():
        raise ValueError("Incomplete timing trial requires explicit review, not silent continuation")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    data = TimingData(config)
    rules = rule_controls(root, protocol, fold)
    experience = TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    model = make_model(env, config)
    if sum(p.numel() for p in model.policy.parameters()) != protocol["parameters"]:
        raise ValueError("Timing policy parameter count differs from the registration")
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.policy.state_dict().items()}
    TimingAgent(model, data.contract).save(directory / "initial_model", config)
    if phase == "screen":
        for other in METHODS:
            other_path = root / "runs" / f'screen_{fold}_{other}_{seed}' / "initial_model"
            if other_path.exists() and _policy_bytes(other_path) != _policy_bytes(directory / "initial_model"):
                raise ValueError("Reward-screen methods must start with identical actor and critic tensors")
    initialized, initialized_steps = {}, {}
    with preserve_training_rng(model.device):
        agent = TimingAgent.load(directory / "initial_model", data.contract, "cpu")
        for mode in MODES:
            result = evaluate_case(agent, data, directory / "initialized" / mode, mode, "deterministic_initialized")
            if result["metrics"] != rules[mode] or result["metrics"] != result["benchmark_metrics"]:
                raise ValueError("Untrained deterministic calendar is not the exact unchanged baseline")
            initialized[mode] = result["metrics"]
            initialized_steps[mode] = result["evaluation_steps"]
        del agent
    callback = TimingCallback(data, directory, experience, initial, BUDGETS[phase], run_id)
    model.set_logger(configure(str(directory / "logs"), ["csv"]))
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    try:
        model.learn(total_timesteps=BUDGETS[phase], callback=callback)
        model.logger.dump(step=model.num_timesteps)
        if model.num_timesteps != BUDGETS[phase]:
            raise AssertionError("Timing trial did not finish exactly its registered final budget")
        counts = _optimizer_counts(model)
        if not counts or min(counts.values()) <= 0 or not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise AssertionError("Timing snapshot lacks actual finite Adam updates")
        TimingAgent(model, data.contract).save(directory / "final_model", config, dict(phase=phase, fixed_final_budget=BUDGETS[phase]))
        with preserve_training_rng(model.device):
            restored = TimingAgent.load(directory / "final_model", data.contract, "cpu")
            for name, value in model.policy.state_dict().items():
                if not torch.equal(value.detach().cpu(), restored.model.policy.state_dict()[name]):
                    raise AssertionError("Timing parameters changed during save/load")
            vector = model._last_obs[0]
            before, after = TimingAgent(model, data.contract).inspect_action(vector), restored.inspect_action(vector)
            np.testing.assert_allclose(before["action_probabilities"], after["action_probabilities"], rtol=1e-6, atol=1e-7)
            if before["macro_action"] != after["macro_action"]:
                raise AssertionError("Timing deterministic action changed during loading")
            del restored
        cuda = model.device.type == "cuda"
        audit = dict(actual_timesteps=model.num_timesteps, actual_optimizer_updates=max(counts.values()),
            optimizer_steps_min=min(counts.values()), optimizer_steps_max=max(counts.values()), optimizer_steps_by_parameter=counts,
            parameters=sum(p.numel() for p in model.policy.parameters()),
            tensor_delta_l2={name: float(torch.linalg.vector_norm(value.detach().cpu() - initial[name])) for name, value in model.policy.state_dict().items()},
            experience=experience.stats(), training_trace_prefix=callback.trace.manifest(), mode_statistics=callback.trace.statistics.snapshot(),
            rollout_buffers=callback.rollouts, training_schedule=experience.env.schedule_audit(),
            training_diagnostics=diagnostic_summary(callback.update_records), elapsed_seconds=perf_counter() - callback.started,
            all_parameters_finite=True, save_load_parameters_equal=True, save_load_macro_action_equal=True,
            model_sha256=file_hash(directory / "final_model" / "model.zip"),
            initial_policy_pth_sha256=hashlib.sha256(_policy_bytes(directory / "initial_model")).hexdigest(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device) if cuda else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device) if cuda else None,
            source_hashes=protocol["source_hashes"], git=protocol["git"], test_evaluated=False)
        if audit["experience"]["total_steps"] != BUDGETS[phase] or audit["experience"]["benchmark_steps"] != BUDGETS[phase]:
            raise AssertionError("Timing actual experience or dual-book steps differ from the final budget")
        verify_rollouts(directory, audit["rollout_buffers"], BUDGETS[phase], audit["training_trace_prefix"])
        write_json(directory / "final_audit.json", audit)
        write_json(directory / "episodes.json", [e for e in experience.episodes if e["steps"]])
    finally:
        callback.trace.close()
        env.close()
    del model, callback, initial, experience, env
    _release()
    agent = TimingAgent.load(directory / "final_model", data.contract, "cpu")
    metrics, evaluation_steps = {}, {}
    for mode in MODES:
        result = evaluate_case(agent, data, directory / "trained" / mode, mode, "trained")
        if result["benchmark_metrics"] != rules[mode]:
            raise ValueError("Trained path changed the frozen calendar benchmark")
        metrics[mode] = dict(trained=result["metrics"], initialized=initialized[mode], benchmark=rules[mode])
        evaluation_steps[mode] = result["evaluation_steps"]
    del agent
    _release()
    row = dict(fold=fold, method=method, phase=phase, seed=seed, steps=BUDGETS[phase], metrics=metrics, audit=audit,
        initialized_evaluation_steps=initialized_steps, trained_evaluation_steps=evaluation_steps,
        parallel_benchmark_evaluation_steps=sum(initialized_steps.values()) + sum(evaluation_steps.values()),
        model_sha256=audit["model_sha256"], model_path=str((directory / "final_model").relative_to(ROOT)),
        test_evaluated=False, actionable=False)
    check_frozen(protocol)
    write_json(directory / "status.json", dict(status="complete", test_evaluated=False, actionable=False))
    write_json(directory / "completed.json", dict(protocol_sha256=json_digest(protocol), config_sha256=json_digest(config),
        selection_sha256=selection_sha256, record=row, files=_leaf_files(directory)))
    return row


def verify_rollouts(directory, rollouts, steps, trace=None):
    previous = 0
    if trace is not None and len(trace["chunks"]) != len(rollouts):
        raise ValueError("PPO rollouts and native timing chunks differ")
    for index, record in enumerate(rollouts):
        path = (Path(directory) / record["path"]).resolve()
        if (not path.is_relative_to(Path(directory).resolve()) or file_hash(path) != record["sha256"]
                or record["first_step"] != previous + 1 or record["last_step"] - previous != record["steps"]):
            raise ValueError("Categorical PPO rollout prefix changed")
        with np.load(path, allow_pickle=False) as data:
            size = record["steps"]
            if data["observations"].shape != (size, 1, len(FEATURE_NAMES)) or data["actions"].shape != (size, 1, 1):
                raise ValueError("Categorical PPO rollout uses the wrong input or macro-action layout")
            if not np.isin(data["actions"], [0, 1]).all() or any(not np.isfinite(data[name]).all() for name in data.files):
                raise ValueError("Categorical PPO rollout contains invalid original samples")
            if not np.allclose(data["returns"], data["advantages"] + data["values"], atol=1e-6):
                raise ValueError("PPO return targets no longer reconcile with advantages and old values")
            if trace is not None:
                chunk = trace["chunks"][index]
                if any(record[key] != chunk[key] for key in ("first_step", "last_step", "steps")):
                    raise ValueError("PPO rollouts and native timing prefix clocks differ")
                native_path = Path(directory) / "training_trace" / chunk["path"]
                if file_hash(native_path) != chunk["sha256"]:
                    raise ValueError("Native original macro samples changed")
                with np.load(native_path, allow_pickle=False) as native:
                    pairs = (("actions", "action_macro_action"), ("values", "action_macro_value"), ("log_probs", "action_macro_log_prob"))
                    if (not np.all(native["action_has_policy_probabilities"])
                            or not np.array_equal(data["observations"][:, 0], native["action_timing_observation"])
                            or any(not np.array_equal(data[buffer_name].reshape(-1), native[trace_name]) for buffer_name, trace_name in pairs)):
                        raise ValueError("Native sampled macro inputs/action/logP/value are not the actual PPO rollout")
        previous = record["last_step"]
    if previous != steps:
        raise ValueError("Categorical PPO rollout buffers do not cover the complete final budget")
    return rollouts


def _account_mean(row, strategy="trained"):
    return _metric_mean([row["metrics"][mode][strategy] for mode in MODES])


def validate_records(records, phase, method=None):
    if phase == "screen":
        expected = {(fold, candidate, 42) for fold in SCREEN_FOLDS for candidate in METHODS}
    elif phase == "confirm" and method in METHODS:
        expected = {(fold, method, seed) for fold in CONFIRM_FOLDS for seed in SEEDS}
    else:
        raise ValueError("Unregistered timing result phase")
    actual = [(row["fold"], row["method"], row["seed"]) for row in records]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Timing stage requires every registered final-budget trial exactly once")
    for row in records:
        if row["phase"] != phase or row["steps"] != BUDGETS[phase] or set(row["metrics"]) != set(MODES):
            raise ValueError("Timing record has the wrong final budget or account set")
        for mode in MODES:
            for name in ("trained", "initialized", "benchmark"):
                metric = row["metrics"][mode][name]
                for key in ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost"):
                    item = metric[key]
                    if isinstance(item, bool) or not isinstance(item, (int, float)) or not np.isfinite(item):
                        raise ValueError("Timing result lacks finite economic metrics")
            if row["metrics"][mode]["initialized"] != row["metrics"][mode]["benchmark"]:
                raise ValueError("Own deterministic initialization is not the unchanged benchmark")
    return dict(trials=len(records), final_snapshots=len(records), actual_training_steps=sum(row["steps"] for row in records))


def select_screen(records):
    counts = validate_records(records, "screen")
    candidates = {}
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        gaps = {_row["fold"]: _account_mean(_row)["cumulative_return"] - _account_mean(_row, "benchmark")["cumulative_return"] for _row in rows}
        candidates[method] = dict(minimum_fold_gap=min(gaps.values()), mean_gap=float(np.mean(list(gaps.values()))),
            fold_gaps=gaps, worst_account_drawdown=max(abs(row["metrics"][mode]["trained"]["max_drawdown"]) for row in rows for mode in MODES),
            mean_cost=float(np.mean([_account_mean(row)["total_fees"] + _account_mean(row)["total_slippage_cost"] for row in rows])))
    selected = min(METHODS, key=lambda name: (-candidates[name]["minimum_fold_gap"], -candidates[name]["mean_gap"],
                   candidates[name]["worst_account_drawdown"], candidates[name]["mean_cost"], name))
    return dict(**counts, selected_method=selected, candidates=candidates, selection_uses_only="2023/2024 fixed-final screen", test_evaluated=False)


def summarize_confirmation(records, selected_method):
    counts = validate_records(records, "confirm", selected_method)
    folds = {}
    for fold in CONFIRM_FOLDS:
        rows = sorted([row for row in records if row["fold"] == fold], key=lambda row: row["seed"])
        means = [_account_mean(row) for row in rows]
        gaps = [mean["cumulative_return"] - _account_mean(row, "benchmark")["cumulative_return"] for row, mean in zip(rows, means)]
        gains = [mean["cumulative_return"] - _account_mean(row, "initialized")["cumulative_return"] for row, mean in zip(rows, means)]
        worst = max(abs(row["metrics"][mode]["trained"]["max_drawdown"]) for row in rows for mode in MODES)
        excess = max(abs(row["metrics"][mode]["trained"]["max_drawdown"]) - abs(row["metrics"][mode]["benchmark"]["max_drawdown"]) for row in rows for mode in MODES)
        folds[fold] = dict(metrics=_metric_stats(means), baseline_gap=_stats(gaps), own_initialization_gain=_stats(gains),
            positive_gap_seeds=sum(gap > 0 for gap in gaps), worst_account_drawdown=worst, worst_account_drawdown_excess=excess,
            qualified=bool(np.mean(gaps) > 0 and sum(gap > 0 for gap in gaps) >= 2 and worst <= .10 and excess <= .02 and np.mean(gains) > 0),
            per_seed={str(row["seed"]): dict(metrics=mean, gap=gap, own_initialization_gain=gain) for row, mean, gap, gain in zip(rows, means, gaps, gains)})
    paired = {str(seed): float(np.mean([folds[fold]["per_seed"][str(seed)]["gap"] for fold in CONFIRM_FOLDS])) for seed in SEEDS}
    overall_gap = _stats(list(paired.values()))
    qualified = all(item["qualified"] for item in folds.values()) and overall_gap["mean"] >= .005
    return dict(**counts, selected_method=selected_method, folds=folds, paired_across_fold_gap=overall_gap, per_seed_gap=paired,
        qualified=bool(qualified), qualified_candidate=dict(method=selected_method, representative_fold="2026", representative_seed=42) if qualified else None,
        aggregation="Four-account mean within seed/fold, mean over two folds within seed, sample SD over three seeds",
        test_evaluated=False, actionable=False)


def _load_stage(root, protocol, phase, method=None):
    records = []
    selection_sha = file_hash(Path(root) / "selection.json") if phase == "confirm" else None
    folds, methods, seeds = (SCREEN_FOLDS, METHODS, (42,)) if phase == "screen" else (CONFIRM_FOLDS, (method,), SEEDS)
    for fold in folds:
        verify_worker_receipt(root, protocol, phase, fold, method)
        for candidate in methods:
            for seed in seeds:
                config = trial_config(protocol, root, fold, candidate, phase, seed)
                row = cached_trial(Path(root) / "runs" / f'{phase}_{fold}_{candidate}_{seed}', config, protocol, selection_sha)
                if row is None:
                    raise ValueError("Stage is incomplete; reporting cannot create missing trials")
                records.append(row)
    validate_records(records, phase, method)
    return records


def verify_worker_receipt(root, protocol, phase, fold, selected_method=None):
    root = Path(root)
    if (phase not in BUDGETS or fold not in (SCREEN_FOLDS if phase == "screen" else CONFIRM_FOLDS)
            or phase == "confirm" and selected_method not in METHODS):
        raise ValueError("Unregistered timing worker receipt")
    methods, seeds = (METHODS, (42,)) if phase == "screen" else ((selected_method,), SEEDS)
    expected = {f'{phase}_{fold}_{method}_{seed}' for method in methods for seed in seeds}
    path = root / f'worker_{phase}_{fold}.json'
    if not path.exists():
        raise ValueError("Missing exclusive worker receipt; aggregation cannot start training")
    receipt = read_config(path)
    completed = receipt["completed_runs"]
    if (receipt["status"] != "complete" or receipt["phase"] != phase or receipt["fold"] != fold
            or receipt["protocol_sha256"] != file_hash(root / "protocol.json")
            or receipt["selection_sha256"] != (file_hash(root / "selection.json") if phase == "confirm" else None)
            or receipt.get("no_global_summary_writes") is not True or len(completed) != len(expected)
            or {item["run_id"] for item in completed} != expected):
        raise ValueError("Exclusive fold workers must seal all registered final-budget trials before aggregation")
    for item in completed:
        if file_hash(root / "runs" / item["run_id"] / "completed.json") != item["completed_sha256"]:
            raise ValueError("Timing worker completion identity changed")
    return receipt


def run_worker(root, protocol, phase, fold, selected_method=None):
    folds = SCREEN_FOLDS if phase == "screen" else CONFIRM_FOLDS
    if phase not in BUDGETS or fold not in folds or phase == "confirm" and selected_method not in METHODS:
        raise ValueError("Unregistered exclusive timing worker")
    root = Path(root)
    methods, seeds = (METHODS, (42,)) if phase == "screen" else ((selected_method,), SEEDS)
    selection_sha = file_hash(root / "selection.json") if phase == "confirm" else None
    if phase == "confirm":
        validate_confirmation_selection(root, protocol, selected_method, selection_sha)
    completed = []
    for method in methods:
        for seed in seeds:
            run_trial(root, protocol, fold, method, phase, seed, selection_sha)
            run_id = f'{phase}_{fold}_{method}_{seed}'
            completed.append(dict(run_id=run_id, completed_sha256=file_hash(root / "runs" / run_id / "completed.json")))
            write_json(root / f'worker_{phase}_{fold}.json', dict(status="running", phase=phase, fold=fold,
                protocol_sha256=file_hash(root / "protocol.json"), selection_sha256=selection_sha,
                completed_runs=completed, no_global_summary_writes=True))
    write_json(root / f'worker_{phase}_{fold}.json', dict(status="complete", phase=phase, fold=fold,
        protocol_sha256=file_hash(root / "protocol.json"), selection_sha256=selection_sha,
        completed_runs=completed, no_global_summary_writes=True))
    return completed


def seal_selection(root, protocol, records):
    selection = dict(protocol_sha256=json_digest(protocol), **select_screen(records),
        screen_completed_sha256={f'{r["fold"]}_{r["method"]}': file_hash(Path(root) / "runs" / f'screen_{r["fold"]}_{r["method"]}_42' / "completed.json") for r in records})
    path = Path(root) / "selection.json"
    if path.exists() and read_config(path) != selection:
        raise ValueError("The frozen screen selection cannot change")
    if not path.exists():
        write_json(path, selection)
    return selection


def write_report(root, protocol, screen, confirmation):
    root = Path(root)
    selection = seal_selection(root, protocol, screen)
    summary = summarize_confirmation(confirmation, selection["selected_method"])
    summary.update(screen=selection, total_actual_training_steps=sum(row["steps"] for row in screen + confirmation),
                   total_final_snapshots=len(screen) + len(confirmation),
                   actual_optimizer_updates=sum(row["audit"]["actual_optimizer_updates"] for row in screen + confirmation),
                   total_training_benchmark_steps=sum(row["audit"]["experience"]["benchmark_steps"] for row in screen + confirmation))
    controls = [read_config(root / "rules" / fold / "completed.json") for fold in (*SCREEN_FOLDS, *CONFIRM_FOLDS)]
    summary["evaluation_counts"] = dict(initialized_paths=40, trained_paths=40, rule_paths=16, parallel_benchmark_paths=96, legacy_identity_paths=16,
        initialized_agent_steps=sum(sum(row["initialized_evaluation_steps"].values()) for row in screen + confirmation),
        trained_agent_steps=sum(sum(row["trained_evaluation_steps"].values()) for row in screen + confirmation),
        rule_agent_steps=sum(row["agent_steps"] for row in controls), legacy_identity_steps=sum(row["legacy_identity_steps"] for row in controls),
        parallel_benchmark_steps=sum(row["parallel_benchmark_evaluation_steps"] for row in screen + confirmation) + sum(row["parallel_benchmark_steps"] for row in controls))
    write_json(root / "summary.json", screen + confirmation)
    write_json(root / "study_summary.json", summary)
    pd.DataFrame([dict(phase=row["phase"], fold=row["fold"], method=row["method"], seed=row["seed"], steps=row["steps"], mode=mode,
                      **values["trained"]) for row in screen + confirmation for mode, values in row["metrics"].items()]).to_csv(root / "metrics.csv", index=False)
    lines = ["# Timing-controller research", "", "Already-exposed historical development evidence; no protected test or automatic deployment.", "",
             f'Screen-selected reward: {selection["selected_method"]}.', "", "| Confirmation fold | Mean net ROI % | Paired baseline gap pp | Worst DD % | Qualified |",
             "| --- | ---: | ---: | ---: | --- |"]
    for fold, row in summary["folds"].items():
        lines.append(f'| {fold} | {row["metrics"]["cumulative_return"]["mean"]*100:.4f} | {row["baseline_gap"]["mean"]*100:.4f} | {row["worst_account_drawdown"]*100:.4f} | {row["qualified"]} |')
    lines.extend(["", f'Overall qualified: {summary["qualified"]}.', "", "Baseline stock ranking/quantities are inherited; only hold/rebalance timing is learned.",
                  "RL samples count once. The parallel benchmark book is separately counted, not extra RL experience.",
                  "Relative log reward with gamma=.995 is a discounted learning proxy, not an exact terminal-return difference."])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "status.json", dict(status="complete", total_actual_training_steps=summary["total_actual_training_steps"],
                                         final_snapshots=summary["total_final_snapshots"], qualified=summary["qualified"], test_evaluated=False, actionable=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "confirm", "report"), required=True)
    parser.add_argument("--output", default="reports/runs/ppo_timing_20261014")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--folds-root", default=str(DEFAULT_FOLDS))
    parser.add_argument("--research-2026", default=str(DEFAULT_2026))
    parser.add_argument("--fold", choices=(*SCREEN_FOLDS, *CONFIRM_FOLDS))
    args = parser.parse_args(argv)
    root = Path(args.output).resolve()
    if args.fold is not None and (args.stage not in BUDGETS or args.fold not in (SCREEN_FOLDS if args.stage == "screen" else CONFIRM_FOLDS)):
        parser.error("--fold selects one exclusive screen/confirmation worker only")
    if args.stage != "prepare" and not (root / "protocol.json").exists():
        raise ValueError("Only prepare may create a study registration")
    protocol = prepare(root, args.reference_root, args.folds_root, args.research_2026)
    check_frozen(protocol)
    if args.stage == "prepare":
        return protocol
    if args.stage == "screen":
        if args.fold is not None:
            return run_worker(root, protocol, "screen", args.fold)
        selection = seal_selection(root, protocol, _load_stage(root, protocol, "screen"))
        write_json(root / "status.json", dict(status="screen_complete", selected_method=selection["selected_method"],
                   actual_training_steps=selection["actual_training_steps"], test_evaluated=False, actionable=False))
        return selection
    screen = _load_stage(root, protocol, "screen")
    selection = seal_selection(root, protocol, screen)
    if args.stage == "confirm":
        if args.fold is not None:
            return run_worker(root, protocol, "confirm", args.fold, selection["selected_method"])
        summary = summarize_confirmation(_load_stage(root, protocol, "confirm", selection["selected_method"]), selection["selected_method"])
        write_json(root / "confirmation_summary.json", summary)
        write_json(root / "status.json", dict(status="confirmation_complete", actual_training_steps=summary["actual_training_steps"],
                   qualified=summary["qualified"], test_evaluated=False, actionable=False))
        return summary
    return write_report(root, protocol, screen, _load_stage(root, protocol, "confirm", selection["selected_method"]))


if __name__ == "__main__":
    main()
