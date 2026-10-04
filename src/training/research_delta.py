"""Registered stock-delta study: train-only pilots, fixed snapshots, local validation."""

import argparse
from copy import deepcopy
import gc
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.preprocess import ROOT, file_hash, read_config, resolve_path, write_json
from src.envs.initial_account import MODES
from src.runtime import events
from .env_factory import ResearchData
from .evaluate import evaluate_agent
from .freeze import json_digest, source_hashes
from .matrix_compare import load_matrix
from .matrix_experience import EpisodeAudit
from .matrix_report import _metric_mean, _metric_stats, _stats
from .research_matrix import MatrixCallback, verify_completed


SEEDS = [42, 123, 2026]
BUDGETS = dict(short=8192, long=65536)
PILOT_STEPS = 2048
PILOT_CANDIDATES = [dict(name=f"lr{lr:g}_std{std:g}", learning_rate=lr, initial_log_std=std)
                    for lr in (1e-5, 1e-6) for std in (-3., -1.5)]
DATA_FIELDS = ("dataset_version", "manifest_sha256", "artifacts", "feature_cols", "stock_codes",
               "lookback", "broker", "initial_capital", "price_basis", "data_source")


def config_for(base, root, seed=42, candidate=None):
    config = deepcopy(base)
    config.update(action_mode="stock_delta", hidden_sizes=[256, 128], device="cuda", torch_threads=1,
                  seed=seed, max_delta_weight=.05, actor_output_gain=.01, top_n=20,
                  top_n_choices=[10, 20, 40], prior_feature_index=None, output_root=str(Path(root) / "runs"))
    config.pop("action_top_k", None)
    if candidate is not None:
        config["ppo"]["learning_rate"] = candidate["learning_rate"]
        config["initial_log_std"] = candidate["initial_log_std"]
    if any(config["ppo"].get(key) != value for key, value in [("n_steps", 512), ("batch_size", 128), ("n_epochs", 5)]):
        raise ValueError("Delta study requires n_steps=512, batch_size=128 and n_epochs=5")
    if config.get("learning_rate_schedule", "constant") != "constant":
        raise ValueError("Pilot learning-rate comparison requires the constant schedule")
    return config


def pilot_eligible(report):
    kl = report.get("tail_mean_kl")
    return bool(report.get("status") == "success" and report.get("actual_steps") == PILOT_STEPS
                and report.get("all_parameters_finite") is True and report.get("optimizer_steps", 0) > 0
                and report.get("gross_filled_notional", 0) > 0 and isinstance(kl, (int, float))
                and math.isfinite(kl) and 0 <= kl <= .045)


def choose_pilot(reports):
    eligible = [row for row in reports if pilot_eligible(row)]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (-row["optimizer_steps"], row["tail_mean_kl"],
                math.exp(row["initial_log_std"]), -row["learning_rate"], row["name"]))


def _registered_files(directory):
    return {path.relative_to(directory).as_posix(): file_hash(path)
            for path in directory.rglob("*") if path.is_file() and path.name != "completed.json"}


def _verify_files(directory, files):
    if not isinstance(files, dict) or not files:
        raise ValueError("Registered file hashes must be a nonempty dictionary")
    for name, digest in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Registered delta study artifact changed: " + str(path))


def prepare_study(base, root, reference_roots):
    root = Path(root).resolve()
    if set(reference_roots) != {"prior", "random"}:
        raise ValueError("Both prior and random reference matrices are required")
    for reference in reference_roots.values():
        reference = Path(reference).resolve()
        if root == reference or root.is_relative_to(reference) or reference.is_relative_to(root):
            raise ValueError("Study and reference roots must not contain each other")
    config = config_for(base, root)
    data = ResearchData(config)
    if data.contract.get("data_source") != "sina" or len(data.codes) != 300:
        raise ValueError("Delta study requires the registered local Sina 300-stock data")
    references = {}
    differences = {}
    for mode, directory in reference_roots.items():
        reference = load_matrix(directory, mode)
        old = reference["protocol"]
        if old["seeds"] != SEEDS or any(old[name + "_steps"] != steps for name, steps in BUDGETS.items()):
            raise ValueError("Reference seeds or fixed budgets differ")
        if old["train_end"] != data.manifest["train_end"] or old["val_end"] != data.manifest["val_end"]:
            raise ValueError("Reference chronological boundaries differ")
        if old["evaluation_seed"] != 20260927 or old["evaluation_modes"] != list(MODES):
            raise ValueError("Reference evaluation scenarios differ")
        for field in DATA_FIELDS:
            if old["core_contract"].get(field) != data.contract.get(field):
                raise ValueError("Reference data/execution mismatch: " + field)
        old_ppo, new_ppo = deepcopy(old["base_config"]["ppo"]), deepcopy(config["ppo"])
        old_ppo.pop("learning_rate", None)
        new_ppo.pop("learning_rate", None)
        if old_ppo != new_ppo or old["base_config"].get("learning_rate_schedule", "constant") != "constant":
            raise ValueError("Reference PPO hyperparameters differ beyond pilot learning rate")
        for field in ("initial_portfolio", "training_episode_steps", "recent_episode_every"):
            if old["core_contract"].get(field) != data.contract.get(field):
                raise ValueError("Reference training episode/account sampling differs: " + field)
        references[mode] = dict(root=str(reference["root"]), evidence=reference["evidence"])
        new_core = deepcopy(data.contract)
        new_core.pop("hidden_sizes")
        differences[mode] = {key: dict(reference=old["core_contract"].get(key), delta=new_core.get(key))
            for key in sorted(set(old["core_contract"]) | set(new_core))
            if old["core_contract"].get(key) != new_core.get(key)}
    core = deepcopy(data.contract)
    core.pop("hidden_sizes")
    protocol = dict(schema_version=1, study_mode="delta", action_mode="stock_delta", base_config=config,
        seeds=SEEDS, short_steps=BUDGETS["short"], long_steps=BUDGETS["long"],
        maximum_training_steps=len(SEEDS) * BUDGETS["long"], pilot_steps_each=PILOT_STEPS,
        maximum_pilot_steps=len(PILOT_CANDIDATES) * PILOT_STEPS, pilot_candidates=PILOT_CANDIDATES,
        pilot_rule="Finite parameters, optimizer>0, gross fills>0, last-quarter mean KL<=0.045; "
                   "rank actual optimizer steps, then lower KL, lower initial sigma, higher learning rate; no validation returns",
        core_contract=core, references=references, reference_contract_differences=differences,
        source_hashes=source_hashes(), evaluation_seed=20260927, evaluation_modes=list(MODES),
        main_top_n=20, auxiliary_top_n=[10, 40], auxiliary_used_for_selection=False,
        checkpoint_selection="fixed_budget_shared_prefix", train_end=data.manifest["train_end"],
        val_end=data.manifest["val_end"], validation_already_exposed=True, test_evaluated=False,
        limitations=data.manifest.get("limitations", []) + [
            "Stock-delta and legacy target actors differ in action/state architecture and execution semantics.",
            "Pilot learning rate and exploration are selected using training diagnostics, so comparisons are not isolated causal tests.",
            "HFQ quotes and 100-share lots are research assumptions, not real execution/share quantities.",
            "Already-exposed development validation; no actionable or globally optimal model is established."],
        critic_diagnostic_capture="rollout_end_copy_v2", hidden_sizes=[256, 128])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol.json"
    if path.exists():
        if read_config(path) != protocol:
            raise ValueError("Delta protocol changed; use a new study directory")
    else:
        write_json(path, protocol)
        write_json(root / "status.json", dict(status="prepared", test_evaluated=False))
    return protocol


class DeltaCallback(MatrixCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gross_filled_notional = 0.

    def _on_step(self):
        for info in self.locals.get("infos", []):
            self.gross_filled_notional += sum(abs(order["filled_shares"]) * order["price"]
                                             for order in info.get("orders", []) if order["filled_shares"])
        return super()._on_step()

    def record_update(self):
        sigma = self.model.policy.network.log_std.detach().exp()
        self.model.logger.record("train/stock_sigma", float(sigma.mean().cpu().item()))
        self.model.logger.record("train/gross_filled_notional", self.gross_filled_notional)
        super().record_update()


def _training_setup(config, directory):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    (directory / "events").mkdir()
    events.configure(directory / "events")
    data = ResearchData(config)
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(config["seed"])
    return data, experience, env


def _release():
    events.configure(None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_pilot(directory, config, protocol):
    cached = read_config(directory / "completed.json")
    if cached["config_sha256"] != json_digest(config) or cached["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Pilot cache configuration changed")
    _verify_files(directory, cached["files"])
    report = cached["report"]
    if (report != read_config(directory / "report.json") or report["learning_rate"] != config["ppo"]["learning_rate"]
            or report["initial_log_std"] != config["initial_log_std"] or report["name"] != directory.name):
        raise ValueError("Pilot report identity differs from registered configuration")
    if report["status"] == "success":
        audit = read_config(directory / "pilot_audit.json")
        metadata = read_config(directory / "pilot_model" / "metadata.json")
        digest = file_hash(directory / "pilot_model" / "model.zip")
        expected = dict(actual_steps=audit["actual_timesteps"], all_parameters_finite=audit["all_parameters_finite"],
                        optimizer_steps=audit["optimizer_steps_min"],
                        tail_mean_kl=audit["training_diagnostics"].get("tail_mean/train/approx_kl"),
                        gross_filled_notional=audit["training_diagnostics"].get("last/train/gross_filled_notional"),
                        stock_sigma=audit["training_diagnostics"].get("last/train/stock_sigma"))
        required = {"config.json", "report.json", "pilot_audit.json", "pilot_model/model.zip", "pilot_model/metadata.json"}
        if (not required <= set(cached["files"]) or any(report.get(key) != value for key, value in expected.items())
                or audit["model_sha256"] != digest or metadata["model_sha256"] != digest
                or metadata["num_timesteps"] != PILOT_STEPS or metadata["config"] != config):
            raise ValueError("Pilot summary differs from registered training evidence")
    return report


def run_pilot(config, root, candidate, protocol):
    directory = Path(root) / "pilots" / candidate["name"]
    marker = directory / "completed.json"
    if marker.exists():
        return _load_pilot(directory, config, protocol)
    if directory.exists():
        raise ValueError("Incomplete pilot requires review; refusing to overwrite")
    data, experience, env = _training_setup(config, directory)
    model = callback = None
    report = dict(candidate, status="error", actual_steps=0, all_parameters_finite=False,
                  optimizer_steps=0, gross_filled_notional=0., tail_mean_kl=None,
                  stock_sigma=None, validation_evaluated=False, test_evaluated=False)
    started = perf_counter()
    try:
        model = make_model(env, config)
        initial = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
        PPOAgent(model, data.contract).save(directory / "initial_model", config)
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        callback = DeltaCallback(data, directory, experience, initial, dict(pilot=PILOT_STEPS),
                                 Path(root) / "status.json", candidate["name"])
        torch.cuda.reset_peak_memory_stats(model.device)
        model.learn(total_timesteps=PILOT_STEPS, callback=callback)
        model.logger.dump(step=model.num_timesteps)
        audit = read_config(directory / "pilot_audit.json")
        report.update(status="success", actual_steps=model.num_timesteps,
                      all_parameters_finite=audit["all_parameters_finite"], optimizer_steps=audit["optimizer_steps_min"],
                      gross_filled_notional=callback.gross_filled_notional,
                      tail_mean_kl=audit["training_diagnostics"].get("tail_mean/train/approx_kl"),
                      stock_sigma=float(model.policy.network.log_std.detach().exp().mean().cpu().item()))
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      actual_steps=int(model.num_timesteps) if model is not None else 0)
    finally:
        env.close()
        report["elapsed_seconds"] = perf_counter() - started
        report["eligible"] = pilot_eligible(report)
        write_json(directory / "report.json", report)
        write_json(marker, dict(config_sha256=json_digest(config), protocol_sha256=json_digest(protocol),
                               files=_registered_files(directory), report=report))
        del model, callback
        _release()
    return report


def run_pilots(protocol, root):
    reports = []
    for candidate in PILOT_CANDIDATES:
        config = config_for(protocol["base_config"], root, candidate=candidate)
        reports.append(run_pilot(config, root, candidate, protocol))
    selected = choose_pilot(reports)
    result = dict(protocol_sha256=json_digest(protocol), reports=reports, selected=selected,
                  actual_steps=sum(row["actual_steps"] for row in reports),
                  validation_evaluated=False, test_evaluated=False,
                  files={f"pilots/{row['name']}/completed.json": file_hash(Path(root) / "pilots" / row["name"] / "completed.json")
                         for row in reports})
    write_json(Path(root) / "pilot_selection.json", result)
    write_json(Path(root) / "status.json", dict(status="pilot_complete" if selected else "pilot_failed_closed",
               actual_pilot_steps=result["actual_steps"], selected=selected, test_evaluated=False))
    if selected is None:
        raise RuntimeError("No stable train-only pilot; formal training is disabled")
    return result


def _selected_pilot(protocol, root):
    result = read_config(Path(root) / "pilot_selection.json")
    if result["protocol_sha256"] != json_digest(protocol):
        raise ValueError("Pilot selection belongs to a different protocol")
    _verify_files(Path(root), result["files"])
    reports = []
    for candidate in PILOT_CANDIDATES:
        config = config_for(protocol["base_config"], root, candidate=candidate)
        directory = Path(root) / "pilots" / candidate["name"]
        reports.append(_load_pilot(directory, config, protocol))
    selected = choose_pilot(reports)
    if selected is None or result["selected"] != selected or result["reports"] != reports:
        raise ValueError("Pilot selection changed or has no eligible candidate")
    return selected


def evaluate_initial(data, initial_path, root, run_id):
    directory = Path(root) / "controls" / run_id
    identity = dict(contract=data.contract, training_seed=data.config["seed"], evaluation_seed=20260927,
                    evaluation_modes=list(MODES), top_n=20, initial_model_sha256=file_hash(initial_path / "model.zip"))
    path = directory / "controls.json"
    if path.exists():
        record = read_config(path)
        if record["identity"] != identity:
            raise ValueError("Initialization control identity changed")
        _verify_files(directory, record["files"])
        required = {f"{mode}/{name}" for mode in MODES for name in
                    ("metrics.json", "ppo/daily.csv", "ppo/orders.csv", "ppo/positions.parquet")}
        if not required <= set(record["files"]):
            raise ValueError("Initialization controls lack required registered account artifacts")
        if record["metrics"] != {mode: read_config(directory / mode / "metrics.json") for mode in MODES}:
            raise ValueError("Initialization control metrics changed")
        return record["metrics"]
    if directory.exists():
        raise ValueError("Incomplete initialization controls require review")
    agent = PPOAgent.load(initial_path, data.contract, data.config["device"])
    if agent.model.num_timesteps != 0:
        raise ValueError("Initialization control must be untrained")
    metrics = {}
    for mode in MODES:
        data.settings.update(seed=20260927, initial_portfolio=mode, delta_top_n=20)
        folder = directory / mode
        evaluate_agent(agent, data, folder, compare_baselines=True)
        metrics[mode] = read_config(folder / "metrics.json")
    write_json(path, dict(identity=identity, metrics=metrics, files=_registered_files(directory)))
    del agent
    _release()
    return metrics


def run_trial(protocol, root, seed, candidate):
    config = config_for(protocol["base_config"], root, seed, candidate)
    run_id = f"small_{seed}"
    directory = Path(root) / "runs" / run_id
    cached = verify_completed(directory, config, protocol)
    if cached is not None:
        if not (Path(root) / "controls" / run_id / "controls.json").exists():
            raise ValueError("Completed delta trial lacks initialization controls")
        data = ResearchData(config)
        controls = evaluate_initial(data, directory / "initial_model", root, run_id)
        if any(row["metrics"][mode]["initialized"] != controls[mode]["ppo"] for row in cached for mode in MODES):
            raise ValueError("Completed delta trial initialization evidence changed")
        return cached
    if directory.exists():
        raise ValueError("Incomplete delta trial requires review; refusing to overwrite")
    data, experience, env = _training_setup(config, directory)
    try:
        model = make_model(env, config)
        initial = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
        PPOAgent(model, data.contract).save(directory / "initial_model", config)
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        callback = DeltaCallback(data, directory, experience, initial, BUDGETS, Path(root) / "status.json", run_id)
        torch.cuda.reset_peak_memory_stats(model.device)
        model.learn(total_timesteps=BUDGETS["long"], callback=callback)
        model.logger.dump(step=model.num_timesteps)
        if callback.saved != set(BUDGETS):
            raise AssertionError("Both fixed delta checkpoints must be saved")
    finally:
        env.close()
    del model, callback, env, initial, experience
    _release()
    controls = evaluate_initial(data, directory / "initial_model", root, run_id)
    records = []
    for budget, steps in BUDGETS.items():
        path = directory / (budget + "_model")
        agent = PPOAgent.load(path, data.contract, config["device"])
        metrics, auxiliary = {}, {}
        for top_n in [20, 10, 40]:
            current = {}
            for mode in MODES:
                data.settings.update(seed=20260927, initial_portfolio=mode, delta_top_n=top_n)
                folder = directory / (budget + "_validation" if top_n == 20 else f"{budget}_aux_n{top_n}") / mode
                write_json(Path(root) / "status.json", dict(status="evaluating", run_id=run_id,
                           budget=budget, account=mode, top_n=top_n, test_evaluated=False))
                evaluate_agent(agent, data, folder)
                current[mode] = read_config(folder / "metrics.json")["ppo"]
            if top_n == 20:
                metrics = {mode: dict(ppo=current[mode], initialized=controls[mode]["ppo"],
                    baselines={name: value for name, value in controls[mode].items() if name != "ppo"}) for mode in MODES}
            else:
                auxiliary[str(top_n)] = current
        audit = read_config(directory / (budget + "_audit.json"))
        records.append(dict(size="small", budget=budget, seed=seed, steps=steps, hidden_sizes=[256, 128],
            initialization_mode="delta", parameters=audit["parameters"], elapsed_seconds=audit["elapsed_seconds"],
            training_diagnostics=audit["training_diagnostics"], experience=audit["experience"], metrics=metrics,
            auxiliary_metrics=auxiliary, auxiliary_used_for_selection=False,
            training_updates_path=str((directory / "training_updates.csv").relative_to(root)),
            training_run_id=run_id, prefix_shared=True, model_sha256=audit["model_sha256"],
            model_path=str(path.relative_to(ROOT)), test_evaluated=False))
        del agent
        _release()
    data.settings["delta_top_n"] = 20
    write_json(directory / "completed.json", dict(config_sha256=json_digest(config), protocol_sha256=json_digest(protocol),
               files=_registered_files(directory), records=records))
    return records


def summarize_study(records, references):
    expected = {(budget, seed) for budget in BUDGETS for seed in SEEDS}
    if len(records) != len(expected) or {(row["budget"], row["seed"]) for row in records} != expected:
        raise ValueError("Delta study requires both fixed budgets for all three seeds")
    if (any(row.get("size") != "small" or row.get("hidden_sizes") != [256, 128] for row in records)
            or len({row["parameters"] for row in records}) != 1):
        raise ValueError("Delta study requires one consistent small model capacity")
    cells = {}
    for budget, steps in BUDGETS.items():
        rows = sorted([row for row in records if row["budget"] == budget], key=lambda row: row["seed"])
        if any(row["steps"] != steps or row.get("test_evaluated") or set(row["metrics"]) != set(MODES) for row in rows):
            raise ValueError("Delta checkpoint budget, split or account scenarios changed")
        per_seed = []
        for row in rows:
            scenarios = list(row["metrics"].values())
            metrics = _metric_mean([item["ppo"] for item in scenarios])
            initialized = _metric_mean([item["initialized"] for item in scenarios])
            gaps = dict(initialized=metrics["cumulative_return"] - initialized["cumulative_return"])
            for name in ["momentum_daily", "momentum"]:
                gaps[name] = metrics["cumulative_return"] - _metric_mean([item["baselines"][name] for item in scenarios])["cumulative_return"]
            old_gaps = {}
            for mode, reference in references.items():
                old = next(item for item in reference["records"] if (item["size"], item["budget"], item["seed"]) == ("small", budget, row["seed"]))
                for account in MODES:
                    if scenarios[0]["baselines"].keys() != old["metrics"][account]["baselines"].keys():
                        raise ValueError("Legacy baseline names differ")
                    for name, metric in row["metrics"][account]["baselines"].items():
                        for field, value in metric.items():
                            previous = old["metrics"][account]["baselines"][name].get(field)
                            if (value is None) != (previous is None) or (value is not None and not math.isclose(value, previous, rel_tol=0, abs_tol=1e-10)):
                                raise ValueError("Legacy fixed baseline values differ")
                old_gaps[mode] = metrics["cumulative_return"] - _metric_mean([item["ppo"] for item in old["metrics"].values()])["cumulative_return"]
            per_seed.append(dict(seed=row["seed"], metrics=metrics, initialized=initialized, return_gaps=gaps,
                                 legacy_return_gaps=old_gaps))
        worst = max(abs(row["metrics"][mode]["ppo"]["max_drawdown"]) for row in rows for mode in MODES)
        positive = sum(row["return_gaps"]["initialized"] > 0 for row in per_seed)
        gates = dict(all_accounts_within_10_percent_drawdown=worst <= .10,
                     mean_learning_increment_positive=_stats(row["return_gaps"]["initialized"] for row in per_seed)["mean"] > 0,
                     at_least_two_positive_learning_seeds=positive >= 2,
                     mean_return_above_matched_prior=_stats(row["legacy_return_gaps"]["prior"] for row in per_seed)["mean"] > 0)
        cells[budget] = dict(steps=steps, parameters=rows[0]["parameters"], metrics=_metric_stats([row["metrics"] for row in per_seed]),
            initialized=_metric_stats([row["initialized"] for row in per_seed]),
            return_gaps={name: _stats(row["return_gaps"][name] for row in per_seed) for name in ("initialized", "momentum_daily", "momentum")},
            legacy_return_gaps={mode: _stats(row["legacy_return_gaps"][mode] for row in per_seed) for mode in references},
            worst_account_drawdown=worst, per_seed=per_seed,
            gates=gates, qualified=all(gates.values()), positive_learning_seeds=positive,
            beats_daily_momentum=_stats(row["return_gaps"]["momentum_daily"] for row in per_seed)["mean"] > 0,
            beats_20day_momentum=_stats(row["return_gaps"]["momentum"] for row in per_seed)["mean"] > 0,
            per_account={mode: _metric_stats([row["metrics"][mode]["ppo"] for row in rows]) for mode in MODES},
            auxiliary={str(k): _metric_stats([_metric_mean(list(row["auxiliary_metrics"][str(k)].values())) for row in rows]) for k in (10, 40)})
    paired = [cells["long"]["per_seed"][index]["metrics"]["cumulative_return"]
              - cells["short"]["per_seed"][index]["metrics"]["cumulative_return"] for index in range(len(SEEDS))]
    eligible = [name for name, cell in cells.items() if cell["qualified"]]
    candidate = None
    if eligible:
        budget = min(eligible, key=lambda name: (-cells[name]["metrics"]["cumulative_return"]["mean"],
                     cells[name]["worst_account_drawdown"], cells[name]["steps"]))
        cell = cells[budget]
        representative = min(cell["per_seed"], key=lambda row: (abs(row["metrics"]["cumulative_return"]
                            - cell["metrics"]["cumulative_return"]["mean"]), row["seed"]))
        record = next(row for row in records if row["budget"] == budget and row["seed"] == representative["seed"])
        candidate = dict(budget=budget, main_top_n=20, mean_return=cell["metrics"]["cumulative_return"]["mean"],
                         representative_seed=representative["seed"], model_path=record.get("model_path"),
                         model_sha256=record.get("model_sha256"), beats_daily_momentum=cell["beats_daily_momentum"],
                         beats_20day_momentum=cell["beats_20day_momentum"], rule="closest-to-cell-mean seed, not best seed; main K20 only")
    return dict(seeds=SEEDS, account_scenarios=list(MODES), main_top_n=20, cells=cells,
                qualified_candidate=candidate,
                paired_long_minus_short=_stats(paired), actual_formal_steps=len(SEEDS) * BUDGETS["long"],
                checkpoint_steps_sum=sum(row["steps"] for row in records), total_runs=3, total_checkpoints=6,
                auxiliary_used_for_selection=False, actionable=False, global_optimality_claim=False, test_evaluated=False,
                aggregation="Four-account means per seed, then sample standard deviation across three seeds; old comparisons are paired within seed/budget")


def write_report(root, records, references, protocol):
    summary = summarize_study(records, references)
    summary.update(protocol_sha256=json_digest(protocol), limitations=protocol["limitations"],
                   reference_contract_differences=protocol["reference_contract_differences"],
                   source_hash_differences={mode: {name: dict(old=reference["protocol"]["source_hashes"].get(name), delta=digest)
                       for name, digest in protocol["source_hashes"].items() if reference["protocol"]["source_hashes"].get(name) != digest}
                       for mode, reference in references.items()})
    selection_path = Path(root) / "pilot_selection.json"
    if selection_path.exists():
        summary["selected_train_only_pilot"] = read_config(selection_path)["selected"]
    write_json(Path(root) / "study_summary.json", summary)
    lines = ["# Stock Delta Study", "", "Already-exposed development validation; main K=20. Auxiliary K=10/40 is descriptive only.", "",
        "| Budget | Steps | Net return % +/- seed SD | Own initialization gain pp | Prior gap pp | Random gap pp | Worst drawdown % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def stat(value):
        return f"{100 * value['mean']:.3f} +/- {100 * value['std']:.3f}"
    for budget, cell in summary["cells"].items():
        lines.append(f"| {budget} | {cell['steps']:,} | {stat(cell['metrics']['cumulative_return'])} | "
            f"{stat(cell['return_gaps']['initialized'])} | {stat(cell['legacy_return_gaps']['prior'])} | "
            f"{stat(cell['legacy_return_gaps']['random'])} | {cell['worst_account_drawdown'] * 100:.3f} |")
    candidate = summary["qualified_candidate"]
    lines += ["", "Qualified main-K20 development candidate: " + (candidate["budget"] if candidate else "none") + ".",
        "The gate requires all-account drawdowns <=10%, positive mean own-initialization gain, at least two positive-learning seeds, and higher mean return than matched prior.",
        "Auxiliary K10/K40 never participates in candidate selection; the representative seed is closest to the cell mean.",
        "Learning increments, daily/20-day momentum, costs, cash ratios and each seed/account are recorded in study_summary.json.",
        "Pilot selection used training diagnostics only. These results do not isolate the causal contribution of the new actor or establish an optimal strategy.",
        "", *["- " + value for value in protocol["limitations"]], "", "![Main validation comparison](comparison.png)", "", "[Machine summary](study_summary.json)"]
    (Path(root) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(2)
    for index, (name, color) in enumerate([("prior", "#278d70"), ("random", "#a04b82"), ("delta", "#2874a6")]):
        values = []
        errors = []
        for budget in BUDGETS:
            value = summary["cells"][budget]["metrics"]["cumulative_return"] if name == "delta" else _stats(
                _metric_mean(list(row["metrics"][mode]["ppo"] for mode in MODES))["cumulative_return"]
                for row in references[name]["records"] if row["size"] == "small" and row["budget"] == budget)
            values.append(value["mean"] * 100)
            errors.append(value["std"] * 100)
        ax.bar(x + (index - 1) * .23, values, width=.23, yerr=errors, capsize=4, label=name, color=color)
    ax.set(xticks=x, xticklabels=list(BUDGETS), ylabel="Net return (%), four-account mean per seed",
           title="Exposed development validation; sample SD across three seeds")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.savefig(Path(root) / "comparison.png", dpi=150)
    plt.close(fig)
    return summary


def run_formal(protocol, root):
    selected = _selected_pilot(protocol, root)
    records = []
    try:
        for seed in SEEDS:
            records.extend(run_trial(protocol, root, seed, selected))
            write_json(Path(root) / "summary.json", records)
        references = {mode: load_matrix(value["root"], mode) for mode, value in protocol["references"].items()}
        summary = write_report(root, records, references, protocol)
        write_json(Path(root) / "status.json", dict(status="complete", total_trained_steps=summary["actual_formal_steps"],
                   pilot_steps=sum(row["actual_steps"] for row in read_config(Path(root) / "pilot_selection.json")["reports"]),
                   checkpoints=6, test_evaluated=False, actionable=False))
        return summary
    except Exception as error:
        write_json(Path(root) / "status.json", dict(status="failed", error=str(error), completed_checkpoints=len(records), test_evaluated=False))
        raise
    finally:
        _release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "pilot", "train"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = resolve_path(args.output)
    protocol = prepare_study(read_config(resolve_path(args.config)), root,
                             dict(prior=resolve_path(args.prior_root), random=resolve_path(args.random_root)))
    print(f"STAGE {args.stage}: pilot budget 8,192; formal budget 196,608; test disabled", flush=True)
    if args.stage == "prepare":
        return protocol
    if not torch.cuda.is_available():
        raise RuntimeError("Registered delta training requires working CUDA")
    if args.stage == "pilot":
        return run_pilots(protocol, root)
    return run_formal(protocol, root)


if __name__ == "__main__":
    main()
