"""Tools-only registered calendar-margin experiment; the existing 96 src files stay frozen."""

import argparse
from copy import deepcopy
import io
import zipfile
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
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


MARGINS = (.5, 1., 2.)
LABELS = {.5: "m05", 1.: "m10", 2.: "m20"}
DOCUMENT = ROOT / "doc" / "ppo_timing_margin_protocol_20261014.md"
REFERENCE = ROOT / "reports" / "runs" / "ppo_timing_20261014"
REFERENCE_AUDIT_SHA256 = "8c3e663f7cb3a74b686921058fa3d9e6408711f37e4518ffc9088992529d835e"
TEST_FILES = ("tests/test_research_timing_margin.py", "tests/test_timing_margin_review.py")


def reference_protocol(directory):
    directory = Path(directory).resolve()
    if REFERENCE_AUDIT_SHA256 is None or file_hash(directory / "final_audit.json") != REFERENCE_AUDIT_SHA256:
        raise ValueError("The completed predecessor audit must be pinned before preparing this study")
    audit, protocol = read_config(directory / "final_audit.json"), read_config(directory / "protocol.json")
    if not directory.is_relative_to(ROOT.resolve()):
        raise ValueError("Pinned predecessor must be inside its registered workspace")
    protocol_key = (directory / "protocol.json").relative_to(ROOT.resolve()).as_posix()
    if (audit.get("status") != "passed" or audit.get("signed_files_sha256", {}).get(protocol_key) != file_hash(directory / "protocol.json")
            or source_hashes() != protocol["source_hashes"]):
        raise ValueError("Predecessor protocol or immutable source inventory changed")
    original.check_frozen(protocol)
    return protocol


def check_frozen(protocol):
    fixed = dict(study="timing_margin", margins=list(MARGINS), reward_basis="log_net",
        screen_folds=list(original.SCREEN_FOLDS), confirm_folds=list(original.CONFIRM_FOLDS), seeds=list(original.SEEDS),
        budgets={"screen": 8192, "confirm": 16384}, feature_names=list(FEATURE_NAMES), parameters=12805, planned_screen_trials=6,
        planned_screen_steps=49152, maximum_confirmation_trials=6, maximum_confirmation_steps=98304,
        maximum_actual_training_steps=147456, screen_minimum_each_fold_gap=.0001,
        screen_requires_greedy_and_economic_change=True, tie_break="larger margin")
    if original.BUDGETS != fixed["budgets"] or any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError("The registered margin grid, budget or admission rule changed")
    if (source_hashes() != protocol["source_hashes"] or file_hash(Path(__file__)) != protocol["tool_sha256"]
            or file_hash(DOCUMENT) != protocol["document_sha256"]
            or set(protocol.get("test_hashes", {})) != set(TEST_FILES)
            or protocol.get("reference_final_audit_sha256") != REFERENCE_AUDIT_SHA256):
        raise ValueError("Frozen source, margin runner or human protocol changed")
    for name, digest in protocol["test_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Registered margin admission tests changed")
    previous = reference_protocol(protocol["reference_root"])
    if any(protocol.get(key) != previous[key] for key in ("folds", "base_config", "gate")):
        raise ValueError("Margin study changed its original data, shared configuration or confirmation gate")
    if file_hash(Path(protocol["reference_root"]) / "protocol.json") != protocol["reference_protocol_sha256"]:
        raise ValueError("Predecessor registration changed")


def prepare(root, reference_root=REFERENCE):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    if root.is_relative_to(reference_root) or reference_root.is_relative_to(root):
        raise ValueError("New output must not overlap a predecessor study")
    previous = reference_protocol(reference_root)
    path = root / "protocol.json"
    if path.exists():
        protocol = read_config(path)
        if protocol["reference_root"] != str(reference_root):
            raise ValueError("An existing margin study cannot change references")
        check_frozen(protocol)
        return protocol
    if root.exists():
        raise ValueError("Incomplete unregistered output requires review")
    protocol = dict(schema_version=1, study="timing_margin", margins=list(MARGINS), reward_basis="log_net",
        screen_folds=list(original.SCREEN_FOLDS), confirm_folds=list(original.CONFIRM_FOLDS), seeds=list(original.SEEDS),
        budgets=deepcopy(original.BUDGETS), folds=previous["folds"], base_config=previous["base_config"],
        reference_root=str(reference_root), reference_protocol_sha256=file_hash(reference_root / "protocol.json"),
        reference_final_audit_sha256=REFERENCE_AUDIT_SHA256, source_hashes=source_hashes(),
        tool_sha256=file_hash(Path(__file__)), document_sha256=file_hash(DOCUMENT),
        test_hashes={name: file_hash(ROOT / name) for name in TEST_FILES}, git=git_metadata(),
        feature_names=list(FEATURE_NAMES), parameters=original.PARAMETERS, evaluation_seed=original.EVAL_SEED,
        planned_screen_trials=6, planned_screen_steps=49152, maximum_confirmation_trials=6,
        maximum_confirmation_steps=98304, maximum_actual_training_steps=147456,
        screen_minimum_each_fold_gap=.0001, screen_requires_greedy_and_economic_change=True,
        tie_break="larger margin", gate=previous["gate"], test_evaluated=False, actionable=False)
    root.mkdir(parents=True, exist_ok=False)
    write_json(path, protocol)
    write_json(root / "status.json", dict(status="prepared", test_evaluated=False, actionable=False))
    check_frozen(protocol)
    return protocol


def config_for(protocol, root, fold, margin, phase, seed):
    if (isinstance(margin, bool) or margin not in MARGINS or phase not in original.BUDGETS or seed not in original.SEEDS
            or fold not in (original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS)
            or phase == "screen" and seed != 42):
        raise ValueError("Unregistered margin condition")
    config = exploration_config(protocol["base_config"], root, "split_l1", seed)
    config.update(research_config=protocol["folds"][fold]["research_config"], action_mode="timing_control",
        device="cpu", torch_threads=1, hidden_sizes=[64, 64], top_n=20, top_n_choices=[],
        training_episode_steps=240, recent_episode_every=0, output_root=str(Path(root) / "runs"),
        timing=dict(fold=fold, phase=phase, reward_basis="log_net", calendar_margin=float(margin), period=20),
        margin_study=dict(study="timing_margin", margin=float(margin), label=LABELS[margin],
            protocol_sha256=json_digest(protocol), tool_sha256=protocol["tool_sha256"],
            reference_audit_sha256=protocol["reference_final_audit_sha256"]))
    config["exploration"]["reward_basis"] = "log_net"
    config["ppo"].update(learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=5,
        gamma=.995, gae_lambda=.95, clip_range=.2, target_kl=.03, ent_coef=0., vf_coef=.5, max_grad_norm=.5)
    return config


class MarginData(ExplorationData):
    def __init__(self, config):
        spec = config["timing"]
        if (config["action_mode"] != "timing_control" or isinstance(spec["calendar_margin"], bool) or spec["calendar_margin"] not in MARGINS
                or spec["reward_basis"] != "log_net" or spec["period"] != 20
                or config["margin_study"]["margin"] != spec["calendar_margin"] or config["top_n"] != 20 or config["top_n_choices"]):
            raise ValueError("Invalid registered margin contract")
        borrowed = deepcopy(config)
        borrowed["action_mode"] = "stock_trade"
        super().__init__(borrowed)
        self.stock_contract = deepcopy(self.contract)
        self.config = deepcopy(config)
        self.contract = build_timing_contract(self.stock_contract, "log_net", spec["calendar_margin"])
        self.contract["timing_research"] = deepcopy(spec)
        self.contract["margin_study"] = deepcopy(config["margin_study"])

    def env(self, split):
        # This factory is margin-agnostic; no global substitution or hard-2 runner is used.
        return original.TimingData.env(self, split)


def make_model(env, config):
    torch.set_num_threads(1)
    return PPO(CalendarTimingPolicy, env, seed=config["seed"], device="cpu", verbose=0,
        policy_kwargs=dict(calendar_margin=config["timing"]["calendar_margin"], net_arch={"pi": [], "vf": [64, 64]}), **config["ppo"])


def baseline_metrics(protocol, fold):
    previous = read_config(Path(protocol["reference_root"]) / "protocol.json")
    return original.rule_controls(protocol["reference_root"], previous, fold, create=False)


def case_changes(directory, result):
    macro, economic = False, False
    for chunk, reference in zip(result["trace"]["chunks"], result["trace"]["benchmark_trace"]["chunks"]):
        with np.load(Path(directory) / "trace" / chunk["path"], allow_pickle=False) as agent, np.load(Path(directory) / "trace" / "benchmark" / reference["path"], allow_pickle=False) as benchmark:
            macro |= bool(np.any(agent["action_macro_action"] != (agent["action_phase"] == 0)))
            economic |= bool(np.any(np.abs(agent["after_positions"] - benchmark["after_positions"]) > 1e-8)
                or np.any(np.abs(agent["after_cash"] - benchmark["after_cash"]) > 1e-6)
                or np.any(np.abs(agent["after_equity"] - benchmark["after_equity"]) > 1e-6))
    return dict(greedy_changed=macro, economic_ledger_changed=economic)


def validate_records(records, phase, margin=None):
    expected = {(fold, value, 42) for fold in original.SCREEN_FOLDS for value in MARGINS} if phase == "screen" else {
        (fold, margin, seed) for fold in original.CONFIRM_FOLDS for seed in original.SEEDS}
    actual = [(row["fold"], row["margin"], row["seed"]) for row in records]
    if phase not in original.BUDGETS or phase == "confirm" and margin not in MARGINS or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Every registered final-budget margin trial is required exactly once")
    for row in records:
        if isinstance(row["margin"], bool) or row["phase"] != phase or row["steps"] != original.BUDGETS[phase] or set(row["metrics"]) != set(MODES) or set(row["changes"]) != set(MODES):
            raise ValueError("Margin checkpoint budget or accounts changed")
        for mode in MODES:
            value = row["metrics"][mode]
            if any(type(row["changes"][mode].get(name)) is not bool for name in ("greedy_changed", "economic_ledger_changed")):
                raise ValueError("Admission behavior flags must be actual Boolean evidence")
            if value["initialized"] != value["benchmark"]:
                raise ValueError("Calendar deterministic initialization no longer equals the unchanged baseline")
            for name in ("trained", "initialized", "benchmark"):
                if any(not isinstance(value[name][key], (int, float)) or isinstance(value[name][key], bool) or not np.isfinite(value[name][key])
                       for key in ("cumulative_return", "max_drawdown", "total_fees", "total_slippage_cost")):
                    raise ValueError("Nonfinite or missing economic results")
    return records


def select_screen(records):
    validate_records(records, "screen")
    candidates = {}
    for margin in MARGINS:
        rows = [row for row in records if row["margin"] == margin]
        gaps = {row["fold"]: original._account_mean(row)["cumulative_return"] - original._account_mean(row, "benchmark")["cumulative_return"] for row in rows}
        changed = {row["fold"]: any(value["greedy_changed"] and value["economic_ledger_changed"] for value in row["changes"].values()) for row in rows}
        candidates[LABELS[margin]] = dict(margin=margin, fold_gaps=gaps, minimum_fold_gap=min(gaps.values()), mean_gap=float(np.mean(list(gaps.values()))),
            worst_account_drawdown=max(abs(row["metrics"][mode]["trained"]["max_drawdown"]) for row in rows for mode in MODES),
            mean_cost=float(np.mean([original._account_mean(row)["total_fees"] + original._account_mean(row)["total_slippage_cost"] for row in rows])),
            fold_behavior_changes=changed)
    chosen = min(candidates.values(), key=lambda row: (-row["minimum_fold_gap"], -row["mean_gap"], row["worst_account_drawdown"], row["mean_cost"], -row["margin"]))
    admitted = chosen["minimum_fold_gap"] >= .0001 and any(chosen["fold_behavior_changes"].values())
    return dict(selected_margin=chosen["margin"], admitted=bool(admitted), status="screen_passed" if admitted else "screen_failed",
        candidates=candidates, actual_screen_trials=6, actual_screen_steps=sum(row["steps"] for row in records),
        confirmation_steps_authorized=98304 if admitted else 0, no_fallback=True)


def run_id(fold, margin, phase, seed):
    return f'{phase}_{fold}_{LABELS[margin]}_{seed}'


def verify_worker_receipt(root, protocol, phase, fold, margin=None):
    values, seeds = (MARGINS, (42,)) if phase == "screen" else ((margin,), original.SEEDS)
    expected = {run_id(fold, value, phase, seed) for value in values for seed in seeds}
    receipt = read_config(Path(root) / f'worker_{phase}_{fold}.json')
    if (receipt["status"] != "complete" or receipt["phase"] != phase or receipt["fold"] != fold
            or receipt["protocol_sha256"] != file_hash(Path(root) / "protocol.json")
            or receipt["selection_sha256"] != (file_hash(Path(root) / "selection.json") if phase == "confirm" else None)
            or receipt.get("no_global_summary_writes") is not True or len(receipt["completed_runs"]) != len(expected)
            or {item["run_id"] for item in receipt["completed_runs"]} != expected):
        raise ValueError("Margin workers must complete every exclusive condition before aggregation")
    for item in receipt["completed_runs"]:
        if file_hash(Path(root) / "runs" / item["run_id"] / "completed.json") != item["completed_sha256"]:
            raise ValueError("Margin worker marker changed")
    return receipt


def load_stage(root, protocol, phase, margin=None):
    records = []
    for fold in original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS:
        verify_worker_receipt(root, protocol, phase, fold, margin)
        for value in MARGINS if phase == "screen" else (margin,):
            for seed in (42,) if phase == "screen" else original.SEEDS:
                config = config_for(protocol, root, fold, value, phase, seed)
                record = cached_trial(Path(root) / "runs" / run_id(fold, value, phase, seed), config, protocol,
                    file_hash(Path(root) / "selection.json") if phase == "confirm" else None)
                if record is None:
                    raise ValueError("Aggregation cannot create or borrow missing trials")
                records.append(record)
    return validate_records(records, phase, margin)


def seal_selection(root, protocol, records):
    selection = dict(protocol_sha256=json_digest(protocol), **select_screen(records), completed_sha256={
        run_id(row["fold"], row["margin"], "screen", row["seed"]): file_hash(Path(root) / "runs" / run_id(row["fold"], row["margin"], "screen", row["seed"]) / "completed.json") for row in records})
    path = Path(root) / "selection.json"
    if path.exists() and read_config(path) != selection:
        raise ValueError("Frozen margin selection cannot change")
    if not path.exists():
        write_json(path, selection)
    return selection


def require_confirmation(root, protocol, margin, selection_sha256):
    path = Path(root) / "selection.json"
    if not path.exists() or selection_sha256 != file_hash(path):
        raise ValueError("Confirmation needs the frozen six-trial selection")
    selected = read_config(path)
    if selected["protocol_sha256"] != json_digest(protocol) or not selected["admitted"] or selected["selected_margin"] != margin:
        raise ValueError("Failed or different margin selection cannot start confirmation")
    expected = {run_id(fold, value, "screen", 42) for fold in original.SCREEN_FOLDS for value in MARGINS}
    if set(selected["completed_sha256"]) != expected:
        raise ValueError("Confirmation requires all six exact screen anchors")
    for fold in original.SCREEN_FOLDS:
        verify_worker_receipt(root, protocol, "screen", fold)
    for identifier, digest in selected["completed_sha256"].items():
        if file_hash(Path(root) / "runs" / identifier / "completed.json") != digest:
            raise ValueError("Screen selection anchors changed")
    records = []
    for fold in original.SCREEN_FOLDS:
        for value in MARGINS:
            config = config_for(protocol, root, fold, value, "screen", 42)
            record = cached_trial(Path(root) / "runs" / run_id(fold, value, "screen", 42), config, protocol)
            if record is None:
                raise ValueError("Confirmation cannot start from incomplete screen evidence")
            records.append(record)
    recomputed = select_screen(records)
    if any(selected.get(key) != value for key, value in recomputed.items()):
        raise ValueError("The admission rule or selected margin changed")
    return selected


def verify_model(directory, metadata, config, protocol, contract, steps):
    record = dict(policy_class="src.agents.timing_policy.CalendarTimingPolicy", observation_shape=[129], observation_dtype="float32",
        feature_names=list(FEATURE_NAMES), action_names=list(ACTION_NAMES), net_arch={"pi": [], "vf": [64, 64]},
        calendar_margin=config["timing"]["calendar_margin"], phase_period=20, parameter_count=12805)
    if (metadata["contract"] != contract or metadata["config"] != config or metadata["source_hashes"] != protocol["source_hashes"]
            or metadata["provenance"].get("margin_study") != config["margin_study"] or metadata["policy_record"] != record
            or metadata["feature_names"] != list(FEATURE_NAMES) or metadata["algorithm_class"] != "stable_baselines3.ppo.ppo.PPO"
            or metadata["num_timesteps"] != steps or metadata["actual_device"] != "cpu"
            or file_hash(Path(directory) / "model.zip") != metadata["model_sha256"]):
        raise ValueError("Saved model does not bind the actual margin, runner, algorithm and data contract")
    import json
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        saved = json.loads(archive.read("data"))
    if (saved.get("timing_registered_contract") != contract or saved.get("timing_policy_record") != record
            or saved.get("timing_registered_algorithm") != metadata["algorithm_class"] or saved.get("num_timesteps") != steps):
        raise ValueError("Saved archive differs from margin metadata")


def cached_trial(directory, config, protocol, selection_sha256=None):
    directory = Path(directory)
    if not (directory / "completed.json").exists():
        return None
    completed = read_config(directory / "completed.json")
    if (completed["protocol_sha256"] != json_digest(protocol) or completed["config_sha256"] != json_digest(config)
            or completed["tool_sha256"] != protocol["tool_sha256"] or completed["selection_sha256"] != selection_sha256):
        raise ValueError("Margin cache cannot reuse another condition, runner or predecessor sample")
    original.verify_leaf_files(directory, completed["files"])
    if read_config(directory / "config.json") != config:
        raise ValueError("Native margin configuration changed")
    row, data = completed["record"], MarginData(config)
    benchmark = baseline_metrics(protocol, config["timing"]["fold"])
    spec = config["timing"]
    if (row["fold"], row["margin"], row["phase"], row["seed"], row["steps"]) != (spec["fold"], spec["calendar_margin"], spec["phase"], config["seed"], original.BUDGETS[spec["phase"]]):
        raise ValueError("Margin checkpoint identity changed")
    for name, steps in (("initial_model", 0), ("final_model", row["steps"])):
        verify_model(directory / name, read_config(directory / name / "metadata.json"), config, protocol, data.contract, steps)
    audit = read_config(directory / "final_audit.json")
    if (row["audit"] != audit or audit["actual_timesteps"] != row["steps"] or audit["experience"]["total_steps"] != row["steps"]
            or audit["experience"]["benchmark_steps"] != row["steps"] or audit["optimizer_steps_min"] <= 0
            or audit["actual_optimizer_updates"] != audit["optimizer_steps_max"] or audit["parameters"] != 12805
            or audit.get("source_hashes") != protocol["source_hashes"] or audit.get("tool_sha256") != protocol["tool_sha256"]
            or any(audit.get(flag) is not True for flag in ("save_load_parameters_equal", "save_load_macro_action_equal", "all_parameters_finite"))
            or audit["model_sha256"] != file_hash(directory / "final_model" / "model.zip")):
        raise ValueError("Margin training evidence changed")
    original.verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=row["steps"], expected_contract_sha256=json_digest(data.contract))
    original.verify_rollouts(directory, audit["rollout_buffers"], row["steps"], audit["training_trace_prefix"])
    for mode in MODES:
        for name in ("initialized", "trained"):
            folder = directory / name / mode
            result = read_config(folder / "result.json")
            original._verify_result(folder, result, data.contract, mode, expected_dates=original.validation_dates(data))
            if result["metrics"] != row["metrics"][mode][name] or result["benchmark_metrics"] != row["metrics"][mode]["benchmark"]:
                raise ValueError("Margin economic metrics changed")
            if row["metrics"][mode]["benchmark"] != benchmark[mode]:
                raise ValueError("Margin benchmark differs from the immutable predecessor economic control")
            if row["evaluation_steps"][name][mode] != result["evaluation_steps"]:
                raise ValueError("Margin evaluation step counters changed")
            if name == "trained" and case_changes(folder, result) != row["changes"][mode]:
                raise ValueError("Margin admission behavior flags differ from the original native accounts")
            if name == "initialized" and any(case_changes(folder, result).values()):
                raise ValueError("Margin initialized control differs from the unchanged economic ledger")
    if spec["phase"] == "confirm":
        require_confirmation(directory.parent.parent, protocol, spec["calendar_margin"], selection_sha256)
    if spec["calendar_margin"] == 2.:
        previous = Path(protocol["reference_root"]) / "runs" / f'{spec["phase"]}_{spec["fold"]}_log_net_{config["seed"]}'
        if read_config(directory / "reproduction.json") != compare_reproduction(directory, previous, audit):
            raise ValueError("Cached margin2 reproduction evidence changed")
    return row


def run_trial(root, protocol, fold, margin, phase, seed, selection_sha256=None):
    check_frozen(protocol)
    if phase == "confirm":
        require_confirmation(root, protocol, margin, selection_sha256)
    config = config_for(protocol, root, fold, margin, phase, seed)
    identifier = run_id(fold, margin, phase, seed)
    directory = Path(root) / "runs" / identifier
    existing = cached_trial(directory, config, protocol, selection_sha256)
    if existing is not None:
        return existing
    if directory.exists():
        raise ValueError("Incomplete margin trial needs explicit review")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    data, benchmark = MarginData(config), baseline_metrics(protocol, fold)
    if json_digest(data.stock_contract) != protocol["folds"][fold]["stock_contract_sha256"]:
        raise ValueError("Margin changed the original data or economic execution contract")
    experience = original.TimingExperience(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    model = make_model(env, config)
    initial = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}
    agent = TimingAgent(model, data.contract, dict(margin_study=config["margin_study"]))
    agent.save(directory / "initial_model", config)
    verify_initial_tensors(root, directory, initial, fold, margin, phase, seed)
    evaluation_steps = {"initialized": {}, "trained": {}}
    with preserve_training_rng(model.device):
        for mode in MODES:
            result = original.evaluate_case(agent, data, directory / "initialized" / mode, mode, "deterministic_initialized")
            if result["metrics"] != benchmark[mode] or any(case_changes(directory / "initialized" / mode, result).values()):
                raise ValueError("Margin deterministic initialization changed the unchanged calendar baseline")
            evaluation_steps["initialized"][mode] = result["evaluation_steps"]
    callback = original.TimingCallback(data, directory, experience, initial, original.BUDGETS[phase], identifier)
    model.set_logger(configure(str(directory / "logs"), ["csv"]))
    try:
        model.learn(total_timesteps=original.BUDGETS[phase], callback=callback)
        model.logger.dump(step=model.num_timesteps)
        counts = original._optimizer_counts(model)
        if model.num_timesteps != original.BUDGETS[phase] or not counts or min(counts.values()) <= 0 or not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise ValueError("Margin final budget lacks actual finite optimizer updates")
        agent.save(directory / "final_model", config, dict(margin_study=config["margin_study"], fixed_final_budget=model.num_timesteps))
        with preserve_training_rng(model.device):
            restored = TimingAgent.load(directory / "final_model", data.contract, "cpu")
            for key, value in model.policy.state_dict().items():
                if not torch.equal(value.detach().cpu(), restored.model.policy.state_dict()[key]):
                    raise ValueError("Margin model tensors changed during save/load")
            before, after = agent.inspect_action(model._last_obs[0]), restored.inspect_action(model._last_obs[0])
            np.testing.assert_allclose(before["action_probabilities"], after["action_probabilities"], rtol=1e-6, atol=1e-7)
            if before["macro_action"] != after["macro_action"]:
                raise ValueError("Margin deterministic action changed during save/load")
            del restored
        audit = dict(actual_timesteps=model.num_timesteps, actual_optimizer_updates=max(counts.values()), optimizer_steps_min=min(counts.values()),
            optimizer_steps_max=max(counts.values()), optimizer_steps_by_parameter=counts, parameters=sum(p.numel() for p in model.policy.parameters()),
            model_sha256=file_hash(directory / "final_model" / "model.zip"),
            tensor_delta_l2={key: float(torch.linalg.vector_norm(value.detach().cpu() - initial[key])) for key, value in model.policy.state_dict().items()},
            experience=experience.stats(), training_schedule=experience.env.schedule_audit(), training_trace_prefix=callback.trace.manifest(),
            rollout_buffers=callback.rollouts, mode_statistics=callback.trace.statistics.snapshot(), training_diagnostics=diagnostic_summary(callback.update_records),
            elapsed_seconds=perf_counter() - callback.started, source_hashes=protocol["source_hashes"], tool_sha256=protocol["tool_sha256"], test_evaluated=False)
        audit.update(save_load_parameters_equal=True, save_load_macro_action_equal=True, all_parameters_finite=True)
        original.verify_timing_trace(directory / "training_trace", audit["training_trace_prefix"], steps=model.num_timesteps, expected_contract_sha256=json_digest(data.contract))
        original.verify_rollouts(directory, audit["rollout_buffers"], model.num_timesteps, audit["training_trace_prefix"])
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
            raise ValueError("Margin changed the frozen twenty-day benchmark")
        metrics[mode] = dict(trained=result["metrics"], initialized=benchmark[mode], benchmark=benchmark[mode])
        changes[mode] = case_changes(directory / "trained" / mode, result)
        evaluation_steps["trained"][mode] = result["evaluation_steps"]
    row = dict(fold=fold, margin=margin, method="log_net", phase=phase, seed=seed, steps=original.BUDGETS[phase],
        metrics=metrics, changes=changes, audit=audit, evaluation_steps=evaluation_steps,
        model_path=str(directory / "final_model"), test_evaluated=False, actionable=False)
    if margin == 2.:
        old_directory = Path(protocol["reference_root"]) / "runs" / f'{phase}_{fold}_log_net_{seed}'
        reproduction = compare_reproduction(directory, old_directory, audit)
        write_json(directory / "reproduction.json", reproduction)
    check_frozen(protocol)
    write_json(directory / "completed.json", dict(protocol_sha256=json_digest(protocol), config_sha256=json_digest(config), tool_sha256=protocol["tool_sha256"],
        selection_sha256=selection_sha256, record=row, files=original._leaf_files(directory)))
    return row


def policy_tensors(directory):
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        return torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)


def verify_initial_tensors(root, directory, current, fold, margin, phase, seed):
    expected = torch.zeros_like(current["action_net.weight"])
    expected[0, 1:20], expected[1, 0] = margin, margin
    if not torch.equal(current["action_net.weight"], expected) or torch.count_nonzero(current["action_net.bias"]):
        raise ValueError("Initial Actor is not exactly the registered trainable calendar")
    compared = []
    for other in MARGINS:
        path = Path(root) / "runs" / run_id(fold, other, phase, seed) / "initial_model"
        if path.exists() and path != Path(directory) / "initial_model":
            tensors = policy_tensors(path)
            if set(tensors) != set(current) or any(not torch.equal(current[key], tensors[key]) for key in current if key != "action_net.weight"):
                raise ValueError("Margin cells do not share identical Critic and noncalendar initial tensors")
            compared.append(LABELS[other])
    write_json(Path(directory) / "initial_equivalence.json", dict(margin=margin, compared_margins=compared,
        actor_calendar_exact=True, critic_and_noncalendar_tensors_equal=True, no_extra_forward=True))


def compare_reproduction(directory, previous, audit):
    previous_marker = read_config(Path(previous) / "completed.json")
    original.verify_leaf_files(previous, previous_marker["files"])
    old_audit = read_config(Path(previous) / "final_audit.json")
    if audit["actual_timesteps"] != old_audit["actual_timesteps"]:
        raise ValueError("Margin2 reproduction must use the same actual training budget")
    equal = {}
    for name in ("initial_model", "final_model"):
        first, second = policy_tensors(Path(directory) / name), policy_tensors(Path(previous) / name)
        equal[name] = set(first) == set(second) and all(torch.equal(first[key], second[key]) for key in first)
    old_chunks, new_chunks = old_audit["training_trace_prefix"]["chunks"], audit["training_trace_prefix"]["chunks"]
    if not all(equal.values()) or len(old_chunks) != len(new_chunks):
        raise ValueError("Margin2 policy tensors do not reproduce the matched predecessor")
    for new, old in zip(new_chunks, old_chunks):
        with np.load(Path(directory) / "training_trace" / new["path"], allow_pickle=False) as first, np.load(Path(previous) / "training_trace" / old["path"], allow_pickle=False) as second:
            if set(first.files) != set(second.files) or any(not np.array_equal(first[key], second[key], equal_nan=True) if first[key].dtype.kind in "fc" else not np.array_equal(first[key], second[key]) for key in first.files):
                raise ValueError("Margin2 native training trajectory does not reproduce the matched predecessor")
    return dict(predecessor_completed_sha256=file_hash(Path(previous) / "completed.json"), policy_tensors_equal=equal,
        training_native_arrays_equal=True, reused_as_sample=False, no_extra_forward=True, no_extra_environment_steps=True)


def run_worker(root, protocol, phase, fold, margin=None):
    if phase not in original.BUDGETS or fold not in (original.SCREEN_FOLDS if phase == "screen" else original.CONFIRM_FOLDS):
        raise ValueError("Unregistered margin worker")
    selection = file_hash(Path(root) / "selection.json") if phase == "confirm" else None
    if phase == "confirm":
        require_confirmation(root, protocol, margin, selection)
    completed = []
    for value in MARGINS if phase == "screen" else (margin,):
        for seed in (42,) if phase == "screen" else original.SEEDS:
            run_trial(root, protocol, fold, value, phase, seed, selection)
            identifier = run_id(fold, value, phase, seed)
            completed.append(dict(run_id=identifier, completed_sha256=file_hash(Path(root) / "runs" / identifier / "completed.json")))
            write_json(Path(root) / f'worker_{phase}_{fold}.json', dict(status="running", phase=phase, fold=fold,
                protocol_sha256=file_hash(Path(root) / "protocol.json"), selection_sha256=selection, completed_runs=completed, no_global_summary_writes=True))
    write_json(Path(root) / f'worker_{phase}_{fold}.json', dict(status="complete", phase=phase, fold=fold,
        protocol_sha256=file_hash(Path(root) / "protocol.json"), selection_sha256=selection, completed_runs=completed, no_global_summary_writes=True))
    return completed


def write_report(root, protocol, screen, selected):
    root = Path(root)
    check_frozen(protocol)
    if not (root / "selection.json").exists():
        raise ValueError("Reporting requires an already sealed selection")
    actual_screen = load_stage(root, protocol, "screen")
    actual_selected = seal_selection(root, protocol, actual_screen)
    if screen != actual_screen or selected != actual_selected:
        raise ValueError("Reporting cannot accept unsourced rows or a fabricated failed-screen selection")
    confirmation = load_stage(root, protocol, "confirm", selected["selected_margin"]) if selected["admitted"] else []
    summary = original.summarize_confirmation(confirmation, "log_net") if confirmation else dict(qualified=False, qualified_candidate=None)
    summary.update(screen=selected, actual_screen_steps=49152, actual_confirmation_steps=sum(row["steps"] for row in confirmation),
        actual_training_steps=sum(row["steps"] for row in screen + confirmation), final_snapshots=len(screen) + len(confirmation),
        actual_optimizer_updates=sum(row["audit"]["actual_optimizer_updates"] for row in screen + confirmation),
        independent_training_benchmark_steps=sum(row["audit"]["experience"]["benchmark_steps"] for row in screen + confirmation),
        evaluation_agent_steps=sum(sum(values.values()) for row in screen + confirmation for values in row["evaluation_steps"].values()),
        evaluation_parallel_benchmark_steps=sum(sum(values.values()) for row in screen + confirmation for values in row["evaluation_steps"].values()),
        evaluation_initialized_paths=4 * (len(screen) + len(confirmation)), evaluation_trained_paths=4 * (len(screen) + len(confirmation)),
        new_rule_control_steps=0, reused_rule_reference=True, additional_reproduction_environment_steps=0,
        status="complete" if selected["admitted"] else "screen_failed", actionable=False, test_evaluated=False)
    write_json(root / "summary.json", screen + confirmation)
    write_json(root / "study_summary.json", summary)
    write_json(root / "status.json", {key: summary[key] for key in ("status", "actual_training_steps", "final_snapshots", "qualified", "test_evaluated", "actionable")})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "confirm", "report"), required=True)
    parser.add_argument("--output", default="reports/runs/ppo_timing_margin_20261014")
    parser.add_argument("--reference-root", default=str(REFERENCE))
    parser.add_argument("--fold", choices=(*original.SCREEN_FOLDS, *original.CONFIRM_FOLDS))
    args = parser.parse_args(argv)
    if args.fold is not None and (args.stage not in original.BUDGETS or args.fold not in (original.SCREEN_FOLDS if args.stage == "screen" else original.CONFIRM_FOLDS)):
        parser.error("--fold selects exactly one exclusive screen or confirmation worker")
    root = Path(args.output).resolve()
    if args.stage != "prepare" and not (root / "protocol.json").exists():
        raise ValueError("Only prepare may register a new study")
    protocol = prepare(root, args.reference_root)
    check_frozen(protocol)
    if args.stage == "prepare":
        return protocol
    if args.stage == "screen" and args.fold is not None:
        return run_worker(root, protocol, "screen", args.fold)
    screen = load_stage(root, protocol, "screen")
    selected = seal_selection(root, protocol, screen)
    if args.stage == "screen":
        write_json(root / "status.json", dict(status=selected["status"], selected_margin=selected["selected_margin"], actual_training_steps=49152, test_evaluated=False, actionable=False))
        return selected
    if args.stage == "confirm":
        if not selected["admitted"]:
            raise ValueError("Screen failed; this batch authorizes zero confirmation trials")
        if args.fold is not None:
            return run_worker(root, protocol, "confirm", args.fold, selected["selected_margin"])
        return original.summarize_confirmation(load_stage(root, protocol, "confirm", selected["selected_margin"]), "log_net")
    return write_report(root, protocol, screen, selected)


if __name__ == "__main__":
    main()
