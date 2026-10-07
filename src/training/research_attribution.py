"""Same-parameter factorial study of market routing and Critic feature gradients."""

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

from src.agents.attribution_policy import AttributionTradePolicy
from src.agents.ppo_agent import PPOAgent
from src.backtest.run import git_metadata
from src.data.preprocess import ROOT, read_config, file_hash, write_json
from src.envs.initial_account import MODES
from src.runtime import events
from src.runtime.device import resolve_device
from .attribution_gradients import gradient_diagnostics
from .exploration_env import ExplorationData, EconomicRule
from .freeze import source_hashes, json_digest
from .matrix_experience import EpisodeAudit
from .matrix_report import _metric_mean, _metric_stats, _stats
from .research_delta import _registered_files, _verify_files, _release
from .research_exploration import config_for as old_config, evaluate_case, ExplorationCallback
from .research_trade import verify_trace_prefix


SEEDS = (42, 123, 2026)
FOLDS = ("2023", "2024")
BUDGETS = dict(short=2048, long=4096)
CELLS = {f"{route}_{gradient}": dict(market_context=route, critic_features=gradient)
         for route in ("pure", "fused") for gradient in ("shared", "detached")}
EVAL_SEED = 20260927
DOCUMENT = ROOT / "doc" / "ppo_attribution_protocol_20261007.md"
REFERENCE_AUDIT_SHA256 = "7a82ca745d5bf52f20e2537bd6692fda49595cd141b2b62fbddd2bbe2e53d234"


def config_for(base, reference_root, root, fold, cell, seed):
    if fold not in FOLDS or cell not in CELLS or seed not in SEEDS:
        raise ValueError("Unregistered attribution condition")
    config = old_config(base, root, "split_l1", seed)
    config["research_config"] = str(Path(reference_root) / "folds" / fold / "research.json")
    config["attribution"] = dict(CELLS[cell], fold=fold, gradient_batch_size=128,
                                  gradient_steps=[2048, 4096], gradient_phases=["pre_update", "post_update"])
    return config


class AttributionData(ExplorationData):
    def __init__(self, config):
        super().__init__(config)
        spec = config["attribution"]
        if spec["market_context"] not in {"pure", "fused"} or spec["critic_features"] not in {"shared", "detached"}:
            raise ValueError("Invalid attribution route or feature-gradient mode")
        if self.architecture != "split":
            raise ValueError("Attribution requires the same split parameter structure")
        self.contract["attribution"] = deepcopy(spec)


def make_model(env, config):
    torch.set_num_threads(config["torch_threads"])
    spec = config["attribution"]
    return PPO(AttributionTradePolicy, env, seed=config["seed"], device=resolve_device(config["device"]), verbose=0,
        policy_kwargs=dict(architecture="split", hidden_sizes=tuple(config["hidden_sizes"]),
            initial_log_std=config["initial_log_std"], actor_output_gain=config["actor_output_gain"],
            market_context=spec["market_context"], critic_features=spec["critic_features"]), **config["ppo"])


def load_agent(directory, data):
    agent = PPOAgent.load(directory, data.contract, "cpu")
    policy = agent.model.policy
    spec = data.contract["attribution"]
    if (not isinstance(policy, AttributionTradePolicy)
            or policy.market_context != spec["market_context"] or policy.critic_features != spec["critic_features"]):
        raise ValueError("Saved attribution routing/gradient flags do not match the contract")
    return agent


class AttributionCallback(ExplorationCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_records, self.gradient_batch, self.gradient_step = [], None, None

    def _diagnose(self, phase):
        record = gradient_diagnostics(self.model, self.gradient_batch,
                                      vf_coef=self.model.vf_coef, clip_range=.2)
        record.update(phase=phase, timesteps=self.gradient_step, ppo_epochs_completed=self.model._n_updates,
                      training_only=True, additional_environment_steps=0, additional_optimizer_steps=0,
                      batch_path=str(self.gradient_file.relative_to(self.output)), batch_sha256=file_hash(self.gradient_file))
        self.gradient_records.append(record)
        write_json(self.output / "gradient_diagnostics.json", dict(records=self.gradient_records,
            fixed_first_batch_size=128, targets_reused_between_pre_post=True,
            diagnostics_are_not_training_updates=True))

    def _on_rollout_end(self):
        super()._on_rollout_end()
        if self.model.num_timesteps not in self.budgets.values():
            return
        buffer = self.model.rollout_buffer
        self.gradient_batch = dict(observations={name: values[:128, 0].copy()
            for name, values in buffer.observations.items()}, actions=buffer.actions[:128, 0].copy(),
            old_log_prob=buffer.log_probs[:128, 0].copy(), advantages=buffer.advantages[:128, 0].copy(),
            returns=buffer.returns[:128, 0].copy())
        self.gradient_step = self.model.num_timesteps
        folder = self.output / "gradient_batches"
        folder.mkdir(exist_ok=True)
        self.gradient_file = folder / f"batch_{self.gradient_step:06d}.npz"
        if self.gradient_file.exists():
            raise ValueError("Registered gradient batch cannot be overwritten")
        arrays = {"obs_" + name: values for name, values in self.gradient_batch["observations"].items()}
        arrays.update({name: values for name, values in self.gradient_batch.items() if name != "observations"})
        np.savez_compressed(self.gradient_file, **arrays)
        self._diagnose("pre_update")

    def record_update(self):
        super().record_update()
        if self.gradient_batch is not None:
            self._diagnose("post_update")
            self.gradient_batch, self.gradient_step = None, None


def check_frozen(protocol):
    if source_hashes() != protocol["source_hashes"] or file_hash(DOCUMENT) != protocol["document_sha256"]:
        raise ValueError("Frozen attribution implementation or preregistration changed")
    for item in protocol["folds"].values():
        directory = Path(item["directory"])
        if file_hash(directory / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Historical fold manifest changed")
        _verify_files(directory, item["files"])


def prepare(root, reference_root):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    if file_hash(reference_root / "final_audit.json") != REFERENCE_AUDIT_SHA256:
        raise ValueError("The sealed reference audit was changed")
    audit = read_config(reference_root / "final_audit.json")
    if audit["status"] != "passed" or file_hash(reference_root / "protocol.json") != audit["anchors"]["protocol.json"]:
        raise ValueError("The reference registration differs from its sealed audit anchor")
    reference = read_config(reference_root / "protocol.json")
    for name, digest in reference["source_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("An original research source was changed: " + name)
    folds = {}
    for fold in FOLDS:
        directory = Path(reference["folds"][fold]["directory"])
        old_identity = reference["folds"][fold]
        if file_hash(directory / "manifest.json") != old_identity["manifest_sha256"]:
            raise ValueError("A historical fold manifest differs from the sealed reference")
        _verify_files(directory, old_identity["artifacts"])
        manifest = read_config(directory / "manifest.json")
        if manifest["end_date"] > "2026-06-30" or manifest["train_end"] != f"{fold}-03-31":
            raise ValueError("Fold is outside the registered research boundary")
        folds[fold] = dict(directory=str(directory), manifest_sha256=file_hash(directory / "manifest.json"),
            files={name: file_hash(directory / name) for name in
                   ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json")})
    protocol = dict(schema_version=1, study="market_account_attribution", cells=CELLS, folds=folds,
        seeds=list(SEEDS), budgets=BUDGETS, base_config=reference["base_config"], reference_root=str(reference_root),
        reference_protocol_sha256=file_hash(reference_root / "protocol.json"),
        reference_final_audit_sha256=REFERENCE_AUDIT_SHA256,
        source_hashes=source_hashes(), original_source_count=len(reference["source_hashes"]),
        document_sha256=file_hash(DOCUMENT), git=git_metadata(), parameters=125063,
        planned_actual_steps=98304, planned_trials=24, planned_snapshots=48,
        gradient_batch_size=128, gradient_diagnostics_per_trial=4, test_evaluated=False, actionable=False,
        evaluation_seed=EVAL_SEED, main_top_n=20, auxiliary_evaluations=False,
        gate=dict(max_all_account_drawdown=.1, each_fold_mean_gain_positive=True,
            each_fold_positive_learning_seeds=2, each_fold_beats_same_economic_20day_rule=True),
        representative_seed=42, representative_fold="2024")
    path = root / "protocol.json"
    if path.exists():
        saved = read_config(path)
        protocol["git"] = saved["git"]
        if saved != protocol:
            raise ValueError("Attribution registration identity changed")
        check_frozen(saved)
        return saved
    data = AttributionData(config_for(protocol["base_config"], reference_root, root, "2023", "pure_shared", 42))
    env = data.env("train")
    try:
        observation, _ = env.reset(seed=42)
        if not env.observation_space.contains(observation) or env.unwrapped.market.stock_codes != data.codes:
            raise ValueError("Attribution environment ordering or observation schema mismatch")
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
            raise ValueError("Economic controls belong to a different registration")
        _verify_files(directory, saved["files"])
        for mode in MODES:
            for name, metrics in saved["metrics"][mode].items():
                if read_config(directory / mode / name / "result.json")["metrics"] != metrics:
                    raise ValueError("Rule marker metrics differ from registered evaluation")
        return saved["metrics"]
    if directory.exists():
        raise ValueError("Incomplete rule controls require review")
    data = AttributionData(config_for(protocol["base_config"], protocol["reference_root"], root, fold, "pure_shared", 42))
    metrics = {}
    for mode in MODES:
        data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        metrics[mode] = {}
        for name, kind, interval in (("hold", "hold", 1), ("momentum_daily", "momentum_20", 1),
                                     ("momentum_20day", "momentum_20", 20)):
            result = evaluate_case(EconomicRule(data.contract, kind, interval), data, directory / mode / name,
                                   "attribution_economic_control")
            metrics[mode][name] = result["metrics"]
    write_json(marker, dict(protocol_sha256=json_digest(protocol), metrics=metrics, files=_registered_files(directory)))
    return metrics


def _policy_bytes(directory):
    with zipfile.ZipFile(Path(directory) / "model.zip") as archive:
        return archive.read("policy.pth")


def verify_initial_pair(root, fold, cell, seed, directory):
    current = _policy_bytes(directory / "initial_model")
    matches = []
    for other in CELLS:
        if other == cell:
            continue
        path = Path(root) / "runs" / f"{fold}_{other}_{seed}" / "initial_model"
        if path.exists():
            if _policy_bytes(path) != current:
                raise AssertionError("Factorial cells must have byte-identical initial policy tensors")
            matches.append(other)
    return dict(policy_pth_sha256=hashlib.sha256(current).hexdigest(), matched_existing_cells=matches,
                scope="Same parameters, not necessarily same initial function across market routes")


def initial_controls(root, protocol, data, initial_path, fold, cell, seed):
    route = CELLS[cell]["market_context"]
    directory = Path(root) / "initial_controls" / f"{fold}_{route}_{seed}"
    canonical = deepcopy(data.contract)
    canonical["attribution"].pop("critic_features")
    identity = dict(protocol_sha256=json_digest(protocol), prediction_contract=canonical,
        initial_policy_pth_sha256=hashlib.sha256(_policy_bytes(initial_path)).hexdigest(),
        evaluation_seed=EVAL_SEED, modes=list(MODES), top_n=20,
        reuse_basis="Critic detach preserves all initial forward values at identical theta")
    marker = directory / "completed.json"
    if marker.exists():
        saved = read_config(marker)
        if saved["identity"] != identity:
            raise ValueError("Initial controls cannot be reused for a different function or initial theta")
        _verify_files(directory, saved["files"])
        for mode in MODES:
            if read_config(directory / mode / "result.json")["metrics"] != saved["metrics"][mode]:
                raise ValueError("Initialized control marker disagrees with its actual evaluation")
        return saved["metrics"], directory
    if directory.exists():
        raise ValueError("Incomplete initialized controls require review")
    agent = load_agent(initial_path, data)
    if agent.model.num_timesteps != 0:
        raise ValueError("Initial control policy is already trained")
    metrics = {}
    for mode in MODES:
        data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        metrics[mode] = evaluate_case(agent, data, directory / mode, "attribution_initialized")["metrics"]
    write_json(marker, dict(identity=identity, metrics=metrics, files=_registered_files(directory)))
    del agent
    _release()
    return metrics, directory


def validate_gradient_manifest(directory, records):
    directory = Path(directory).resolve()
    expected = {(steps, phase) for steps in BUDGETS.values() for phase in ("pre_update", "post_update")}
    keys = [(row["timesteps"], row["phase"]) for row in records]
    if len(keys) != 4 or len(set(keys)) != 4 or set(keys) != expected:
        raise ValueError("Each trial requires exactly one pre/post diagnostic for each fixed snapshot")
    for steps in BUDGETS.values():
        pair = [row for row in records if row["timesteps"] == steps]
        paths, hashes = set(), set()
        for row in pair:
            path = (directory / row["batch_path"]).resolve()
            wanted = directory / "gradient_batches" / f"batch_{steps:06d}.npz"
            if path != wanted or file_hash(path) != row["batch_sha256"] or row["batch_size"] != 128:
                raise ValueError("Gradient batch identity, shape or path changed")
            with np.load(path, allow_pickle=False) as batch:
                required = {"actions", "old_log_prob", "advantages", "returns"}
                if not required <= set(batch.files) or not any(name.startswith("obs_") for name in batch.files):
                    raise ValueError("Gradient batch lacks original observations or PPO targets")
                if any(value.ndim < 1 or value.shape[0] != 128 or value.dtype.kind not in "biuf"
                       or not np.isfinite(value).all() for value in batch.values()):
                    raise ValueError("Gradient inputs must have exactly the registered 128 samples")
            paths.add(str(path))
            hashes.add(row["batch_sha256"])
        if len(paths) != 1 or len(hashes) != 1:
            raise ValueError("Pre/post gradient diagnostics did not reuse the same original batch")


def cached_trial(directory, config, protocol):
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    saved = read_config(marker)
    if saved["config_sha256"] != json_digest(config) or saved["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Trial cache configuration changed")
    _verify_files(directory, saved["files"])
    gradients = read_config(directory / "gradient_diagnostics.json")["records"]
    validate_gradient_manifest(directory, gradients)
    expected = {(name, steps) for name, steps in BUDGETS.items()}
    if {(row["budget"], row["steps"]) for row in saved["records"]} != expected or len(saved["records"]) != 2:
        raise ValueError("Cached trial lacks the complete two fixed checkpoints")
    for row in saved["records"]:
        spec = config["attribution"]
        expected_cell = f"{spec['market_context']}_{spec['critic_features']}"
        if row["fold"] != spec["fold"] or row["cell"] != expected_cell or row["seed"] != config["seed"]:
            raise ValueError("Cached record identifies a different factorial condition")
        audit = read_config(directory / (row["budget"] + "_audit.json"))
        metadata = read_config(directory / (row["budget"] + "_model") / "metadata.json")
        if (audit["actual_timesteps"] != row["steps"] or metadata["num_timesteps"] != row["steps"]
                or audit["model_sha256"] != row["model_sha256"] or metadata["model_sha256"] != row["model_sha256"]):
            raise ValueError("Cached record differs from its actual snapshot")
        for name in ("parameters", "optimizer_steps_min", "optimizer_steps_max", "experience", "training_diagnostics"):
            if row[name] != audit[name]:
                raise ValueError("Cached training evidence changed: " + name)
        verify_trace_prefix(directory / "training_trace", row["training_trace_prefix"])
        control_directory = ROOT / row["initial_controls_path"]
        expected_control = directory.parent.parent / "initial_controls" / f"{row['fold']}_{spec['market_context']}_{row['seed']}"
        if control_directory.resolve() != expected_control.resolve():
            raise ValueError("Initialized controls were linked from a different condition")
        control_marker = read_config(control_directory / "completed.json")
        _verify_files(control_directory, control_marker["files"])
        canonical = deepcopy(read_config(directory / "initial_model" / "metadata.json")["contract"])
        canonical["attribution"].pop("critic_features")
        if (control_marker["identity"]["protocol_sha256"] != json_digest(protocol)
                or control_marker["identity"]["initial_policy_pth_sha256"]
                != hashlib.sha256(_policy_bytes(directory / "initial_model")).hexdigest()
                or control_marker["identity"]["prediction_contract"] != canonical):
            raise ValueError("Cached initialized control identity changed")
        if row["gradient_records"] != [item for item in gradients if item["timesteps"] == row["steps"]]:
            raise ValueError("Cached gradient evidence differs from the native diagnostic")
        for mode in MODES:
            result = read_config(directory / row["budget"] / mode / "result.json")
            initialized = read_config(control_directory / mode / "result.json")
            if (row["metrics"][mode]["ppo"] != result["metrics"]
                    or row["metrics"][mode]["initialized"] != initialized["metrics"]
                    or row["metrics"][mode]["trace"] != result["trace"]
                    or row["metrics"][mode]["mode_statistics"] != result["mode_statistics"]):
                raise ValueError("Cached result differs from original evaluation")
    return saved["records"]


def run_trial(root, protocol, fold, cell, seed):
    root = Path(root)
    config = config_for(protocol["base_config"], protocol["reference_root"], root, fold, cell, seed)
    run_id = f"{fold}_{cell}_{seed}"
    directory = root / "runs" / run_id
    cached = cached_trial(directory, config, protocol)
    if cached is not None:
        return cached
    if directory.exists():
        raise ValueError("Incomplete attribution trial requires explicit review")
    check_frozen(protocol)
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    (directory / "events").mkdir()
    events.configure(directory / "events")
    data = AttributionData(config)
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    callback = None
    try:
        model = make_model(env, config)
        initial = {name: tensor.detach().cpu().clone() for name, tensor in model.policy.state_dict().items()}
        if sum(p.numel() for p in model.policy.parameters()) != protocol["parameters"]:
            raise ValueError("Factorial parameter count changed")
        PPOAgent(model, data.contract).save(directory / "initial_model", config)
        write_json(directory / "initial_identity.json", verify_initial_pair(root, fold, cell, seed, directory))
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        callback = AttributionCallback(data, directory, experience, initial, BUDGETS, root / "status.json", run_id)
        torch.cuda.reset_peak_memory_stats(model.device)
        model.learn(total_timesteps=4096, callback=callback)
        model.logger.dump(step=model.num_timesteps)
        if callback.saved != set(BUDGETS) or len(callback.gradient_records) != 4:
            raise AssertionError("Missing attribution snapshots or registered gradient diagnostics")
        validate_gradient_manifest(directory, callback.gradient_records)
        if callback.trace.statistics.violations:
            raise AssertionError("Economic execution invariants failed")
    finally:
        env.close()
        if callback is not None:
            callback.trace.close()
        events.configure(None)
    del model, callback, initial, env, experience
    _release()
    initial_metrics, control_directory = initial_controls(root, protocol, data, directory / "initial_model",
                                                         fold, cell, seed)
    rows = []
    for budget, steps in BUDGETS.items():
        agent = load_agent(directory / (budget + "_model"), data)
        metrics = {}
        for mode in MODES:
            data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
            write_json(root / "status.json", dict(status="evaluating", fold=fold, cell=cell, seed=seed,
                budget=budget, account=mode, test_evaluated=False))
            result = evaluate_case(agent, data, directory / budget / mode, "attribution_trained")
            metrics[mode] = dict(ppo=result["metrics"], initialized=initial_metrics[mode],
                                mode_statistics=result["mode_statistics"], trace=result["trace"])
        audit = read_config(directory / (budget + "_audit.json"))
        rows.append(dict(fold=fold, cell=cell, seed=seed, budget=budget, steps=steps,
            parameters=audit["parameters"], model_path=str((directory / (budget + "_model")).relative_to(ROOT)),
            model_sha256=audit["model_sha256"], experience=audit["experience"],
            optimizer_steps_min=audit["optimizer_steps_min"], optimizer_steps_max=audit["optimizer_steps_max"],
            training_diagnostics=audit["training_diagnostics"], training_mode_statistics=audit["mode_statistics"],
            training_trace_prefix=audit["training_trace_prefix"], tensor_delta_l2=audit["tensor_delta_l2"],
            initial_controls_path=str(control_directory.relative_to(ROOT)),
            gradient_records=[r for r in read_config(directory / "gradient_diagnostics.json")["records"]
                              if r["timesteps"] == steps], metrics=metrics, test_evaluated=False))
        del agent
        _release()
    check_frozen(protocol)
    write_json(directory / "completed.json", dict(config_sha256=json_digest(config),
        protocol_sha256=json_digest(protocol), records=rows, files=_registered_files(directory)))
    return rows


def validate_records(records):
    expected = {(fold, cell, seed, budget, steps) for fold in FOLDS for cell in CELLS for seed in SEEDS
                for budget, steps in BUDGETS.items()}
    actual = [(r["fold"], r["cell"], r["seed"], r["budget"], r["steps"]) for r in records]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Attribution results require all 24 trials and 48 snapshots exactly once")
    runs = {}
    for row in records:
        key = (row["fold"], row["cell"], row["seed"])
        runs[key] = max(row["steps"], runs.get(key, 0))
    return dict(actual_training_steps=sum(runs.values()), trained_runs=len(runs), snapshots=len(records),
                checkpoint_steps_sum=sum(r["steps"] for r in records))


def factorial_effects(fold_cells, folds):
    effects = {}
    for budget in BUDGETS:
        effects[budget] = {}
        paired = {}
        for label in ("net_roi", "initialization", "learning_gain"):
            estimates = dict(fused_minus_pure=[], detached_minus_shared=[], interaction=[],
                fused_minus_pure_shared=[], fused_minus_pure_detached=[],
                detached_minus_shared_pure=[], detached_minus_shared_fused=[])
            for seed in SEEDS:
                results = []
                for fold in folds:
                    values = []
                    for cell in ("pure_shared", "pure_detached", "fused_shared", "fused_detached"):
                        row = fold_cells[fold][f"{cell}_{budget}"]["per_seed"][str(seed)]
                        values.append(row["metrics"]["cumulative_return"] if label == "net_roi" else
                                      row["initialization"] if label == "initialization" else row["gain"])
                    ps, pd_, fs, fd = values
                    results.append(((fs + fd - ps - pd_) / 2, (pd_ + fd - ps - fs) / 2,
                        (fd - fs) - (pd_ - ps), fs - ps, fd - pd_, pd_ - ps, fd - fs))
                for index, name in enumerate(estimates):
                    estimates[name].append(float(np.mean([result[index] for result in results])))
            paired[label] = estimates
            effects[budget][label] = {name: dict(_stats(values), by_seed=dict(zip(map(str, SEEDS), values)))
                                     for name, values in estimates.items()}
        errors = [abs(net - initial - learned) for name in paired["net_roi"] for net, initial, learned in
                  zip(paired["net_roi"][name], paired["initialization"][name], paired["learning_gain"][name])]
        if max(errors) > 1e-12:
            raise AssertionError("Factorial ROI decomposition does not match initial plus learning effects")
        effects[budget]["arithmetic_decomposition_max_error"] = max(errors)
    return effects


def summarize(records, rules):
    counts = validate_records(records)
    fold_cells, cells = {}, {}
    for fold in FOLDS:
        fold_cells[fold] = {}
        for cell in CELLS:
            for budget in BUDGETS:
                rows = sorted([r for r in records if r["fold"] == fold and r["cell"] == cell and r["budget"] == budget],
                              key=lambda row: row["seed"])
                means = [_metric_mean([r["metrics"][mode]["ppo"] for mode in MODES]) for r in rows]
                initial = [_metric_mean([r["metrics"][mode]["initialized"] for mode in MODES])["cumulative_return"] for r in rows]
                gains = [mean["cumulative_return"] - start for mean, start in zip(means, initial)]
                worst = max(abs(r["metrics"][mode]["ppo"]["max_drawdown"]) for r in rows for mode in MODES)
                benchmark = _metric_mean([rules[fold][mode]["momentum_20day"] for mode in MODES])["cumulative_return"]
                metric = _metric_stats(means)
                fold_cells[fold][f"{cell}_{budget}"] = dict(cell=cell, budget=budget, steps=BUDGETS[budget],
                    metrics=metric, own_initialization_gain=_stats(gains), positive_learning_seeds=sum(g > 0 for g in gains),
                    worst_account_drawdown=worst, twenty_day_gap=metric["cumulative_return"]["mean"] - benchmark,
                    qualified=bool(worst <= .10 and np.mean(gains) > 0 and sum(g > 0 for g in gains) >= 2
                                   and metric["cumulative_return"]["mean"] > benchmark),
                    per_seed={str(r["seed"]): dict(metrics=m, initialization=start, gain=gain)
                              for r, m, start, gain in zip(rows, means, initial, gains)})
    for cell in CELLS:
        for budget, steps in BUDGETS.items():
            key = f"{cell}_{budget}"
            per_seed = {str(seed): _metric_mean([fold_cells[fold][key]["per_seed"][str(seed)]["metrics"] for fold in FOLDS])
                        for seed in SEEDS}
            cells[key] = dict(cell=cell, budget=budget, steps=steps, metrics=_metric_stats(list(per_seed.values())),
                own_initialization_gain=_stats([np.mean([fold_cells[fold][key]["per_seed"][str(seed)]["gain"]
                                                         for fold in FOLDS]) for seed in SEEDS]),
                per_seed=per_seed, qualified=all(fold_cells[fold][key]["qualified"] for fold in FOLDS),
                worst_account_drawdown=max(fold_cells[fold][key]["worst_account_drawdown"] for fold in FOLDS))
    eligible = sorted([cell for cell in cells.values() if cell["qualified"]],
        key=lambda item: (-item["metrics"]["cumulative_return"]["mean"], item["worst_account_drawdown"],
            item["metrics"]["total_fees"]["mean"] + item["metrics"]["total_slippage_cost"]["mean"],
            item["steps"], item["cell"]))
    selected = dict(cell=eligible[0]["cell"], budget=eligible[0]["budget"], representative_seed=42,
                    representative_fold="2024") if eligible else None
    return dict(**counts, fold_cells=fold_cells, cells=cells, factorial_effects=factorial_effects(fold_cells, FOLDS),
        factorial_effects_by_fold={fold: factorial_effects(fold_cells, [fold]) for fold in FOLDS},
        qualified_candidate=selected, actionable=False, test_evaluated=False,
        aggregation="Four account mean within seed/fold; two-fold mean within seed; sample SD across three seeds, not six independent seeds.",
        limitations=["Historical fixed 2026 membership and HFQ-only execution remain.",
            "Same theta does not mean same initial policy function across market routes.",
            "Initial behavior and learning gains are separately reported; raw ROI is not pure learning attribution.",
            "Detached Critic features are not an independent Critic encoder or policy-gradient correction."])


def write_report(root, records, rules):
    root = Path(root)
    summary = summarize(records, rules)
    write_json(root / "study_summary.json", summary)
    table = [dict(fold=row["fold"], cell=row["cell"], budget=row["budget"], seed=row["seed"],
        account=mode, steps=row["steps"], **metrics["ppo"]) for row in records for mode, metrics in row["metrics"].items()]
    pd.DataFrame(table).to_csv(root / "metrics.csv", index=False)
    lines = ["# 市场/账户分离归因研究", "", "固定参数结构；两历史折，不是独立前瞻测试。", "",
        "| Fold | Cell | Steps | ROI % +/- seed SD | Own-init gain pp | Worst DD % | 20-day gap pp | Qualified |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for fold, fold_cells in summary["fold_cells"].items():
        for cell in fold_cells.values():
            roi = cell["metrics"]["cumulative_return"]
            lines.append(f"| {fold} | {cell['cell']} | {cell['steps']} | {roi['mean']*100:.4f} +/- {roi['std']*100:.4f} | "
                f"{cell['own_initialization_gain']['mean']*100:.4f} | {cell['worst_account_drawdown']*100:.4f} | "
                f"{cell['twenty_day_gap']*100:.4f} | {cell['qualified']} |")
    lines += ["", f"Registered research candidate: {summary['qualified_candidate']}", "",
        "各fold均须通过初始化增量/风险/20日规则门槛；候选也不升级默认。",
        "梯度诊断是固定训练batch的autograd计算，不计新增优化量；初始化函数route差异不被相同参数掩盖。",
        "[Machine summary](study_summary.json) | [All accounts](metrics.csv) | [Raw records](summary.json)"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "report"), required=True)
    parser.add_argument("--output", default="reports/runs/ppo_attribution_20261007")
    parser.add_argument("--reference-root", default="reports/runs/ppo_exploration_20261006")
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
        for cell in CELLS:
            for seed in SEEDS:
                if args.stage == "report" and not (root / "runs" / f"{fold}_{cell}_{seed}" / "completed.json").exists():
                    raise ValueError("Report cannot start training missing trials")
                records.extend(run_trial(root, protocol, fold, cell, seed))
                if args.stage == "train":
                    write_json(root / "summary.json", records)
    if args.stage == "report" and read_config(root / "summary.json") != records:
        raise ValueError("Global records differ from sealed individual trials")
    write_report(root, records, rules)
    check_frozen(protocol)
    write_json(root / "status.json", dict(status="complete", **validate_records(records),
        gradient_diagnostic_batches=96, test_evaluated=False, actionable=False))


if __name__ == "__main__":
    main()
