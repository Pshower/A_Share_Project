"""Preregistered pure-market pooling study with identical initial policy functions."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.market_summary_policy import MarketSummaryPolicy
from src.agents.ppo_agent import PPOAgent
from src.backtest.run import git_metadata
from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.envs.initial_account import MODES
from src.runtime import events
from src.runtime.device import resolve_device
from .exploration_env import ExplorationData, EconomicRule
from .freeze import json_digest, source_hashes
from .matrix_experience import EpisodeAudit
from .matrix_report import _metric_mean, _metric_stats, _stats
from .research_delta import _registered_files, _verify_files, _release
from .research_exploration import config_for as old_config, evaluate_case, ExplorationCallback
from .research_trade import verify_trace_prefix


SEEDS = (42, 123, 2026)
FOLDS = ("2023", "2024")
ARMS = ("mean", "attention", "dispersion")
BUDGETS = dict(short=4096, long=8192)
EVAL_SEED = 20260927
PARAMETERS = 141575
DOCUMENT = ROOT / "doc" / "ppo_market_summary_protocol_20261007.md"
REFERENCE_AUDIT_SHA256 = "4f9485bd43cec7afe60f423aa1b088e7b3c8b436226ea80deb739cbe36935b7a"


def config_for(base, reference_root, root, fold, arm, seed):
    if fold not in FOLDS or arm not in ARMS or seed not in SEEDS:
        raise ValueError("Unregistered market-summary condition")
    config = old_config(base, root, "split_l1", seed)
    reference = read_config(Path(reference_root) / "protocol.json")
    config["research_config"] = str(Path(reference["folds"][fold]["directory"]) / "research.json")
    config["market_summary"] = dict(pooling=arm, fold=fold, critic_features="shared",
        zero_query_initialization=True, zero_variance_projection_initialization=True)
    return config


class SummaryData(ExplorationData):
    def __init__(self, config):
        super().__init__(config)
        spec = config["market_summary"]
        if (spec["pooling"] not in ARMS or spec["fold"] not in FOLDS
                or spec["critic_features"] != "shared"
                or spec["zero_query_initialization"] is not True
                or spec["zero_variance_projection_initialization"] is not True):
            raise ValueError("Invalid market-summary prediction contract")
        if self.architecture != "split":
            raise ValueError("Pure-market summary requires the split architecture")
        self.contract["market_summary"] = deepcopy(spec)


def make_model(env, config):
    torch.set_num_threads(config["torch_threads"])
    return PPO(MarketSummaryPolicy, env, seed=config["seed"], device=resolve_device(config["device"]), verbose=0,
        policy_kwargs=dict(architecture="split", hidden_sizes=tuple(config["hidden_sizes"]),
            initial_log_std=config["initial_log_std"], actor_output_gain=config["actor_output_gain"],
            pooling=config["market_summary"]["pooling"]), **config["ppo"])


def load_agent(directory, data):
    agent = PPOAgent.load(directory, data.contract, "cpu")
    policy = agent.model.policy
    if (not isinstance(policy, MarketSummaryPolicy) or policy.architecture != "split"
            or policy.pooling != data.contract["market_summary"]["pooling"]
            or policy.network.pooling != policy.pooling):
        raise ValueError("Saved market-summary policy flags do not match its contract")
    return agent


def parameter_participation(model):
    named = dict(model.policy.named_parameters())
    pooling = model.policy.pooling
    inactive = [name for name in named if
        (name.endswith("summary_query") and pooling != "attention") or
        (name.endswith("variance_projection.weight") and pooling != "dispersion")]
    states = model.policy.optimizer.state
    return dict(total_parameters=sum(p.numel() for p in named.values()),
        expected_functionally_active_parameters=sum(p.numel() for name, p in named.items() if name not in inactive),
        summary_query_l2=float(model.policy.network.summary_query.detach().norm().cpu()),
        variance_projection_l2=float(model.policy.network.variance_projection.weight.detach().norm().cpu()),
        inactive_names=sorted(inactive),
        adam_state_tensor_count=sum(p in states and bool(states[p]) for p in named.values()),
        adam_state_parameters=sum(p.numel() for p in named.values() if p in states and bool(states[p])),
        inactive_adam_state_names=sorted(name for name in inactive if named[name] in states and bool(states[named[name]])),
        inactive_nonzero_names=sorted(name for name in inactive if torch.count_nonzero(named[name]).item()))


class SummaryCallback(ExplorationCallback):
    def save_fixed_snapshot(self):
        before = set(self.saved)
        super().save_fixed_snapshot()
        for budget in self.saved - before:
            path = self.output / (budget + "_audit.json")
            audit = read_config(path)
            audit["parameter_participation"] = parameter_participation(self.model)
            write_json(path, audit)


def check_frozen(protocol):
    if source_hashes() != protocol["source_hashes"] or file_hash(DOCUMENT) != protocol["document_sha256"]:
        raise ValueError("Frozen market-summary implementation or preregistration changed")
    reference = Path(protocol["reference_root"])
    if (file_hash(reference / "final_audit.json") != protocol["reference_final_audit_sha256"]
            or file_hash(reference / "protocol.json") != protocol["reference_protocol_sha256"]):
        raise ValueError("Trusted previous-study anchors changed")
    for item in protocol["folds"].values():
        directory = Path(item["directory"])
        if file_hash(directory / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Historical fold manifest changed")
        _verify_files(directory, item["files"])


def prepare(root, reference_root):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    if root.is_relative_to(reference_root) or reference_root.is_relative_to(root):
        raise ValueError("Study output and sealed reference must not overlap")
    if file_hash(reference_root / "final_audit.json") != REFERENCE_AUDIT_SHA256:
        raise ValueError("The sealed reference audit was changed")
    audit = read_config(reference_root / "final_audit.json")
    if audit["status"] != "passed" or file_hash(reference_root / "protocol.json") != audit["anchors"]["protocol.json"]:
        raise ValueError("The reference registration differs from its sealed audit")
    reference = read_config(reference_root / "protocol.json")
    for name, digest in reference["source_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("An original research source was changed: " + name)
    folds = {}
    for fold in FOLDS:
        old = reference["folds"][fold]
        directory = Path(old["directory"])
        if file_hash(directory / "manifest.json") != old["manifest_sha256"]:
            raise ValueError("Historical manifest differs from sealed reference")
        _verify_files(directory, old["files"])
        manifest = read_config(directory / "manifest.json")
        if (not manifest.get("protected_period_excluded") or manifest["end_date"] > "2026-06-30"
                or manifest["train_end"] != f"{fold}-03-31"):
            raise ValueError("Historical fold can expose the protected period")
        folds[fold] = dict(directory=str(directory), manifest_sha256=file_hash(directory / "manifest.json"),
            files={name: file_hash(directory / name) for name in
                   ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json")})
    protocol = dict(schema_version=1, study="pure_market_summary", arms=list(ARMS), folds=folds,
        seeds=list(SEEDS), budgets=BUDGETS, base_config=reference["base_config"], reference_root=str(reference_root),
        reference_protocol_sha256=file_hash(reference_root / "protocol.json"),
        reference_final_audit_sha256=REFERENCE_AUDIT_SHA256,
        source_hashes=source_hashes(), original_source_count=len(reference["source_hashes"]),
        document_sha256=file_hash(DOCUMENT), git=git_metadata(), parameters=PARAMETERS,
        planned_actual_steps=147456, planned_trials=18, planned_snapshots=36,
        planned_trained_evaluation_paths=144, planned_initialized_paths=24, planned_rule_paths=24,
        gradient_diagnostics=False, test_evaluated=False, actionable=False,
        evaluation_seed=EVAL_SEED, main_top_n=20, auxiliary_evaluations=False,
        gate=dict(max_all_account_drawdown=.1, each_fold_mean_gain_positive=True,
            each_fold_positive_learning_seeds=2, each_fold_beats_same_economic_20day_rule=True,
            new_arm_each_fold_beats_mean=True, new_arm_each_fold_positive_paired_seeds=2),
        representative_seed=42, representative_fold="2024")
    path = root / "protocol.json"
    if path.exists():
        saved = read_config(path)
        protocol["git"] = saved["git"]
        if saved != protocol:
            raise ValueError("Market-summary registration identity changed")
        check_frozen(saved)
        return saved
    data = SummaryData(config_for(protocol["base_config"], reference_root, root, "2023", "mean", 42))
    env = data.env("train")
    try:
        observation, _ = env.reset(seed=42)
        if not env.observation_space.contains(observation) or env.unwrapped.market.stock_codes != data.codes:
            raise ValueError("Market-summary environment order/schema mismatch")
    finally:
        env.close()
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, protocol)
    return protocol


def rule_controls(root, protocol, fold):
    directory = Path(root) / "rules" / fold
    marker = directory / "completed.json"
    if marker.exists():
        saved = read_config(marker)
        if saved["protocol_sha256"] != json_digest(protocol):
            raise ValueError("Economic controls belong to a different study")
        _verify_files(directory, saved["files"])
        for mode in MODES:
            for name, metrics in saved["metrics"][mode].items():
                if read_config(directory / mode / name / "result.json")["metrics"] != metrics:
                    raise ValueError("Economic control metrics changed")
        return saved["metrics"]
    if directory.exists():
        raise ValueError("Incomplete economic controls require review")
    data = SummaryData(config_for(protocol["base_config"], protocol["reference_root"], root, fold, "mean", 42))
    metrics = {}
    for mode in MODES:
        data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        metrics[mode] = {}
        for name, kind, interval in (("hold", "hold", 1), ("momentum_daily", "momentum_20", 1),
                                     ("momentum_20day", "momentum_20", 20)):
            result = evaluate_case(EconomicRule(data.contract, kind, interval), data, directory / mode / name,
                                   "market_summary_economic_control")
            metrics[mode][name] = result["metrics"]
    write_json(marker, dict(protocol_sha256=json_digest(protocol), metrics=metrics, files=_registered_files(directory)))
    return metrics


def _policy_bytes(directory):
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        return archive.read("policy.pth")


def neutral_initialization(policy):
    network = policy.network
    if (torch.count_nonzero(network.summary_query).item()
            or torch.count_nonzero(network.variance_projection.weight).item()):
        raise ValueError("Initial pooling query and variance projection must both be exactly zero")
    return _neutral_pooling_proof()


def _neutral_pooling_proof():
    return dict(query_exact_zero=True, variance_projection_exact_zero=True,
        equivalence="Zero query gives uniform masked attention; zero residual gives the same valid-market mean.")


def verify_initial_pair(root, fold, arm, seed, directory, policy):
    proof = neutral_initialization(policy)
    current = _policy_bytes(directory / "initial_model")
    matches = []
    for other in ARMS:
        if other == arm:
            continue
        path = Path(root) / "runs" / f"{fold}_{other}_{seed}" / "initial_model"
        if path.exists():
            if _policy_bytes(path) != current:
                raise AssertionError("Pooling arms must have byte-identical initial policy tensors")
            matches.append(other)
    return dict(policy_pth_sha256=hashlib.sha256(current).hexdigest(), matched_existing_arms=matches,
                neutral_pooling=proof, scope="Same parameters and initial Actor/Critic functions across pooling arms")


def _canonical_contract(contract):
    canonical = deepcopy(contract)
    canonical["market_summary"].pop("pooling")
    return canonical


def initial_controls(root, protocol, data, initial_path, fold, seed):
    directory = Path(root) / "initial_controls" / f"{fold}_{seed}"
    agent = load_agent(initial_path, data)
    try:
        if agent.model.num_timesteps != 0:
            raise ValueError("Initialized control policy is already trained")
        proof = neutral_initialization(agent.model.policy)
        identity = dict(protocol_sha256=json_digest(protocol), prediction_contract=_canonical_contract(data.contract),
            initial_policy_pth_sha256=hashlib.sha256(_policy_bytes(initial_path)).hexdigest(),
            evaluation_seed=EVAL_SEED, modes=list(MODES), top_n=20, neutral_pooling=proof)
        marker = directory / "completed.json"
        if marker.exists():
            saved = read_config(marker)
            if saved["identity"] != identity:
                raise ValueError("Initial control is not the same function, theta or canonical contract")
            _verify_files(directory, saved["files"])
            for mode in MODES:
                if read_config(directory / mode / "result.json")["metrics"] != saved["metrics"][mode]:
                    raise ValueError("Initialized control metrics changed")
            return saved["metrics"], directory
        if directory.exists():
            raise ValueError("Incomplete initialized controls require review")
        metrics = {}
        for mode in MODES:
            data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
            metrics[mode] = evaluate_case(agent, data, directory / mode, "market_summary_initialized")["metrics"]
        write_json(marker, dict(identity=identity, metrics=metrics, files=_registered_files(directory)))
        return metrics, directory
    finally:
        del agent
        _release()


def cached_trial(directory, config, protocol):
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    saved = read_config(marker)
    if saved["config_sha256"] != json_digest(config) or saved["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Trial cache configuration changed")
    _verify_files(directory, saved["files"])
    expected = {(name, steps) for name, steps in BUDGETS.items()}
    if {(row["budget"], row["steps"]) for row in saved["records"]} != expected or len(saved["records"]) != 2:
        raise ValueError("Cached trial lacks exactly the two registered snapshots")
    if read_config(directory / "config.json") != config:
        raise ValueError("Cached native configuration changed")
    initial_metadata = read_config(directory / "initial_model" / "metadata.json")
    initial_contract = initial_metadata["contract"]
    if (initial_metadata["config"] != config or initial_metadata["num_timesteps"] != 0
            or initial_contract["market_summary"] != config["market_summary"]
            or initial_metadata["source_hashes"] != protocol["source_hashes"]
            or file_hash(directory / "initial_model" / "model.zip") != initial_metadata["model_sha256"]):
        raise ValueError("Cached initial model config or prediction contract changed")
    for row in saved["records"]:
        spec = config["market_summary"]
        if row["fold"] != spec["fold"] or row["arm"] != spec["pooling"] or row["seed"] != config["seed"]:
            raise ValueError("Cached row identifies a different summary condition")
        audit = read_config(directory / (row["budget"] + "_audit.json"))
        metadata = read_config(directory / (row["budget"] + "_model") / "metadata.json")
        if (audit["actual_timesteps"] != row["steps"] or metadata["num_timesteps"] != row["steps"]
                or audit["model_sha256"] != row["model_sha256"] or metadata["model_sha256"] != row["model_sha256"]
                or metadata["config"] != config or metadata["contract"] != initial_contract
                or metadata["source_hashes"] != protocol["source_hashes"]
                or file_hash(directory / (row["budget"] + "_model") / "model.zip") != row["model_sha256"]):
            raise ValueError("Cached row differs from its saved model snapshot")
        for name, field in (("parameters", "parameters"), ("optimizer_steps_min", "optimizer_steps_min"),
                ("optimizer_steps_max", "optimizer_steps_max"), ("experience", "experience"),
                ("training_diagnostics", "training_diagnostics"), ("training_mode_statistics", "mode_statistics"),
                ("parameter_participation", "parameter_participation"), ("tensor_delta_l2", "tensor_delta_l2"),
                ("training_trace_prefix", "training_trace_prefix")):
            if row[name] != audit[field]:
                raise ValueError("Cached training evidence changed: " + name)
        if row["training_trace_prefix"]["total_steps"] != row["steps"]:
            raise ValueError("Cached training trace is not the complete registered snapshot prefix")
        verify_trace_prefix(directory / "training_trace", row["training_trace_prefix"])
        control_directory = ROOT / row["initial_controls_path"]
        wanted = directory.parent.parent / "initial_controls" / f"{row['fold']}_{row['seed']}"
        if control_directory.resolve() != wanted.resolve():
            raise ValueError("Initialized controls came from a different fold or seed")
        controls = read_config(control_directory / "completed.json")
        _verify_files(control_directory, controls["files"])
        expected_identity = dict(protocol_sha256=json_digest(protocol), prediction_contract=_canonical_contract(initial_contract),
            initial_policy_pth_sha256=hashlib.sha256(_policy_bytes(directory / "initial_model")).hexdigest(),
            evaluation_seed=EVAL_SEED, modes=list(MODES), top_n=20, neutral_pooling=_neutral_pooling_proof())
        if controls["identity"] != expected_identity:
            raise ValueError("Cached initialized-control identity changed")
        for mode in MODES:
            result = read_config(directory / row["budget"] / mode / "result.json")
            initialized = read_config(control_directory / mode / "result.json")
            if (row["metrics"][mode]["ppo"] != result["metrics"]
                    or row["metrics"][mode]["initialized"] != initialized["metrics"]
                    or controls["metrics"][mode] != initialized["metrics"]
                    or row["metrics"][mode]["trace"] != result["trace"]
                    or row["metrics"][mode]["mode_statistics"] != result["mode_statistics"]):
                raise ValueError("Cached metrics differ from their registered evaluations")
    return saved["records"]


def run_trial(root, protocol, fold, arm, seed):
    root = Path(root)
    config = config_for(protocol["base_config"], protocol["reference_root"], root, fold, arm, seed)
    run_id = f"{fold}_{arm}_{seed}"
    directory = root / "runs" / run_id
    cached = cached_trial(directory, config, protocol)
    if cached is not None:
        return cached
    if directory.exists():
        raise ValueError("Incomplete market-summary trial requires explicit review")
    check_frozen(protocol)
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    (directory / "events").mkdir()
    events.configure(directory / "events")
    data = SummaryData(config)
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    callback = None
    try:
        model = make_model(env, config)
        initial = {name: tensor.detach().cpu().clone() for name, tensor in model.policy.state_dict().items()}
        if sum(p.numel() for p in model.policy.parameters()) != protocol["parameters"]:
            raise ValueError("Market-summary parameter count changed")
        PPOAgent(model, data.contract).save(directory / "initial_model", config)
        write_json(directory / "initial_identity.json", verify_initial_pair(root, fold, arm, seed, directory, model.policy))
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        callback = SummaryCallback(data, directory, experience, initial, BUDGETS, root / "status.json", run_id)
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        model.learn(total_timesteps=max(BUDGETS.values()), callback=callback)
        model.logger.dump(step=model.num_timesteps)
        if callback.saved != set(BUDGETS) or callback.trace.statistics.violations:
            raise AssertionError("Missing fixed snapshots or economic execution violations")
    finally:
        env.close()
        if callback is not None:
            callback.trace.close()
        events.configure(None)
    del model, callback, initial, env, experience
    _release()
    initial_metrics, controls = initial_controls(root, protocol, data, directory / "initial_model", fold, seed)
    rows = []
    for budget, steps in BUDGETS.items():
        agent = load_agent(directory / (budget + "_model"), data)
        metrics = {}
        for mode in MODES:
            data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
            write_json(root / "status.json", dict(status="evaluating", fold=fold, arm=arm, seed=seed,
                budget=budget, account=mode, test_evaluated=False))
            result = evaluate_case(agent, data, directory / budget / mode, "market_summary_trained")
            metrics[mode] = dict(ppo=result["metrics"], initialized=initial_metrics[mode],
                                mode_statistics=result["mode_statistics"], trace=result["trace"])
        audit = read_config(directory / (budget + "_audit.json"))
        rows.append(dict(fold=fold, arm=arm, seed=seed, budget=budget, steps=steps,
            parameters=audit["parameters"], model_path=str((directory / (budget + "_model")).relative_to(ROOT)),
            model_sha256=audit["model_sha256"], experience=audit["experience"],
            optimizer_steps_min=audit["optimizer_steps_min"], optimizer_steps_max=audit["optimizer_steps_max"],
            training_diagnostics=audit["training_diagnostics"], training_mode_statistics=audit["mode_statistics"],
            training_trace_prefix=audit["training_trace_prefix"], tensor_delta_l2=audit["tensor_delta_l2"],
            parameter_participation=audit["parameter_participation"], initial_controls_path=str(controls.relative_to(ROOT)),
            metrics=metrics, test_evaluated=False))
        del agent
        _release()
    check_frozen(protocol)
    write_json(directory / "completed.json", dict(config_sha256=json_digest(config),
        protocol_sha256=json_digest(protocol), records=rows, files=_registered_files(directory)))
    return rows


def validate_records(records):
    expected = {(fold, arm, seed, budget, steps) for fold in FOLDS for arm in ARMS for seed in SEEDS
                for budget, steps in BUDGETS.items()}
    actual = [(r["fold"], r["arm"], r["seed"], r["budget"], r["steps"]) for r in records]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Market-summary results require all 18 trials and 36 snapshots exactly once")
    runs = {}
    for row in records:
        key = (row["fold"], row["arm"], row["seed"])
        runs[key] = max(row["steps"], runs.get(key, 0))
    return dict(actual_training_steps=sum(runs.values()), trained_runs=len(runs), snapshots=len(records),
                checkpoint_steps_sum=sum(r["steps"] for r in records))


def summarize(records, rules):
    counts = validate_records(records)
    fold_arms, arms = {}, {}
    for fold in FOLDS:
        fold_arms[fold] = {}
        for arm in ARMS:
            for budget, steps in BUDGETS.items():
                rows = sorted([r for r in records if r["fold"] == fold and r["arm"] == arm and r["budget"] == budget],
                              key=lambda row: row["seed"])
                means = [_metric_mean([r["metrics"][mode]["ppo"] for mode in MODES]) for r in rows]
                initial = [_metric_mean([r["metrics"][mode]["initialized"] for mode in MODES])["cumulative_return"] for r in rows]
                gains = [mean["cumulative_return"] - start for mean, start in zip(means, initial)]
                worst = max(abs(r["metrics"][mode]["ppo"]["max_drawdown"]) for r in rows for mode in MODES)
                benchmark = _metric_mean([rules[fold][mode]["momentum_20day"] for mode in MODES])["cumulative_return"]
                metric = _metric_stats(means)
                fold_arms[fold][f"{arm}_{budget}"] = dict(arm=arm, budget=budget, steps=steps, metrics=metric,
                    own_initialization_gain=_stats(gains), positive_learning_seeds=sum(g > 0 for g in gains),
                    worst_account_drawdown=worst, twenty_day_gap=metric["cumulative_return"]["mean"] - benchmark,
                    qualified=bool(worst <= .1 and np.mean(gains) > 0 and sum(g > 0 for g in gains) >= 2
                                   and metric["cumulative_return"]["mean"] > benchmark),
                    per_seed={str(r["seed"]): dict(metrics=m, initialization=start, gain=gain)
                              for r, m, start, gain in zip(rows, means, initial, gains)})
        for arm in ARMS:
            for budget in BUDGETS:
                item = fold_arms[fold][f"{arm}_{budget}"]
                baseline = fold_arms[fold][f"mean_{budget}"]
                if any(item["per_seed"][str(seed)]["initialization"] != baseline["per_seed"][str(seed)]["initialization"]
                       for seed in SEEDS):
                    raise ValueError("Pooling arms must share identical initialized control returns")
                differences = [item["per_seed"][str(seed)]["metrics"]["cumulative_return"]
                    - baseline["per_seed"][str(seed)]["metrics"]["cumulative_return"] for seed in SEEDS]
                item["paired_roi_vs_mean"] = dict(_stats(differences), by_seed=dict(zip(map(str, SEEDS), differences)))
                item["positive_paired_seeds"] = sum(value > 0 for value in differences)
                item["improves_mean"] = bool(arm != "mean" and np.mean(differences) > 0
                                             and item["positive_paired_seeds"] >= 2)
    for arm in ARMS:
        for budget, steps in BUDGETS.items():
            key = f"{arm}_{budget}"
            per_seed = {str(seed): _metric_mean([fold_arms[fold][key]["per_seed"][str(seed)]["metrics"] for fold in FOLDS])
                        for seed in SEEDS}
            paired = [np.mean([fold_arms[fold][key]["paired_roi_vs_mean"]["by_seed"][str(seed)] for fold in FOLDS]) for seed in SEEDS]
            arms[key] = dict(arm=arm, budget=budget, steps=steps, metrics=_metric_stats(list(per_seed.values())),
                own_initialization_gain=_stats([np.mean([fold_arms[fold][key]["per_seed"][str(seed)]["gain"]
                                                         for fold in FOLDS]) for seed in SEEDS]),
                paired_roi_vs_mean=dict(_stats(paired), by_seed=dict(zip(map(str, SEEDS), paired))),
                per_seed=per_seed, qualified=all(fold_arms[fold][key]["qualified"] for fold in FOLDS),
                improves_mean=all(fold_arms[fold][key]["improves_mean"] for fold in FOLDS),
                worst_account_drawdown=max(fold_arms[fold][key]["worst_account_drawdown"] for fold in FOLDS))
    eligible = sorted([item for item in arms.values() if item["qualified"] and (item["arm"] == "mean" or item["improves_mean"])],
        key=lambda item: (-item["metrics"]["cumulative_return"]["mean"], item["worst_account_drawdown"],
            item["metrics"]["total_fees"]["mean"] + item["metrics"]["total_slippage_cost"]["mean"], item["steps"], item["arm"]))
    selected = dict(arm=eligible[0]["arm"], budget=eligible[0]["budget"], representative_seed=42,
                    representative_fold="2024") if eligible else None
    return dict(**counts, fold_arms=fold_arms, arms=arms, qualified_candidate=selected,
        actionable=False, test_evaluated=False,
        aggregation="Four account mean within seed/fold; two-fold mean within seed; sample SD across three paired seeds.",
        limitations=["Previously exposed historical folds; not new out-of-sample performance.",
            "Historical fixed 2026 membership and HFQ-only execution remain.",
            "Equal total parameters and initial functions do not imply equal active parameters or Adam states.",
            "Shared Critic gradients also train the active summary module; this is not an Actor-only intervention."])


def write_report(root, records, rules):
    root = Path(root)
    summary = summarize(records, rules)
    write_json(root / "study_summary.json", summary)
    table = [dict(fold=row["fold"], arm=row["arm"], budget=row["budget"], seed=row["seed"],
        account=mode, steps=row["steps"], **metrics["ppo"]) for row in records for mode, metrics in row["metrics"].items()]
    pd.DataFrame(table).to_csv(root / "metrics.csv", index=False)
    lines = ["# Pure-market summary research", "", "Previously exposed historical folds; not independent forward testing.", "",
        "| Fold | Arm | Steps | ROI % +/- seed SD | Own-init gain pp | Paired vs mean pp | Worst DD % | Qualified | Improves mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for fold, items in summary["fold_arms"].items():
        for item in items.values():
            roi = item["metrics"]["cumulative_return"]
            lines.append(f"| {fold} | {item['arm']} | {item['steps']} | {roi['mean']*100:.4f} +/- {roi['std']*100:.4f} | "
                f"{item['own_initialization_gain']['mean']*100:.4f} | {item['paired_roi_vs_mean']['mean']*100:.4f} | "
                f"{item['worst_account_drawdown']*100:.4f} | {item['qualified']} | {item['improves_mean']} |")
    lines += ["", f"Registered research candidate: {summary['qualified_candidate']}", "",
        "No automatic model promotion. Same initial functions; active parameter and optimizer-state counts differ.",
        "[Machine summary](study_summary.json) | [All accounts](metrics.csv) | [Raw records](summary.json)"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "report"), required=True)
    parser.add_argument("--output", default="reports/runs/ppo_market_summary_20261007")
    parser.add_argument("--reference-root", default="reports/runs/ppo_attribution_20261007")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    protocol = prepare(root, args.reference_root)
    check_frozen(protocol)
    if args.stage == "prepare":
        return
    records, rules = [], {}
    for fold in FOLDS:
        if args.stage == "report" and not (root / "rules" / fold / "completed.json").exists():
            raise ValueError("Report cannot execute missing economic controls")
        rules[fold] = rule_controls(root, protocol, fold)
        for arm in ARMS:
            for seed in SEEDS:
                if args.stage == "report" and not (root / "runs" / f"{fold}_{arm}_{seed}" / "completed.json").exists():
                    raise ValueError("Report cannot train missing trials")
                records.extend(run_trial(root, protocol, fold, arm, seed))
                if args.stage == "train":
                    write_json(root / "summary.json", records)
    if args.stage == "report" and read_config(root / "summary.json") != records:
        raise ValueError("Global records differ from sealed individual trials")
    write_report(root, records, rules)
    check_frozen(protocol)
    write_json(root / "status.json", dict(status="complete", **validate_records(records),
        gradient_diagnostic_batches=0, test_evaluated=False, actionable=False))


if __name__ == "__main__":
    main()
