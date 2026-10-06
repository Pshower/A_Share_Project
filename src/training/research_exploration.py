"""Fixed historical-fold ablations, native trade traces and staged diagnostics."""

import argparse
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.exploration_policy import ExplorationTradePolicy
from src.agents.ppo_agent import PPOAgent
from src.backtest.metrics import performance
from src.backtest.run import run_policy, git_metadata
from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.envs.initial_account import MODES
from src.runtime import events
from src.runtime.device import resolve_device
from .exploration_env import ExplorationData, EconomicRule, DateSampler
from .freeze import json_digest, source_hashes
from .matrix_experience import EpisodeAudit
from .matrix_report import _metric_mean, _metric_stats, _stats
from .research_delta import _registered_files, _verify_files, _release
from .research_trade import TradeCallback, TradeTrace, _trace_identity, verify_trace_prefix


SEEDS = (42, 123, 2026)
EVAL_SEED = 20260927
BUDGETS = dict(short=2048, long=8192)
FORMAL = {
    "baseline_l1": dict(lookback=1, architecture="baseline", sampling="recent"),
    "lookback5": dict(lookback=5, architecture="baseline", sampling="recent"),
    "split_l1": dict(lookback=1, architecture="split", sampling="recent"),
    "balanced_l1": dict(lookback=1, architecture="baseline", sampling="quarter_balanced"),
}
PILOTS = {
    "log_net": dict(reward_basis="log_net"),
    "gamma_0995": dict(gamma=.995),
    "size_anneal": dict(size_annealing=True),
}


def config_for(base, root, cell, seed):
    if cell not in {*FORMAL, *PILOTS} or seed not in SEEDS:
        raise ValueError("Unregistered exploration cell or seed")
    config = deepcopy(base)
    spec = dict(FORMAL.get(cell, FORMAL["baseline_l1"]))
    pilot = PILOTS.get(cell, {})
    config.update(research_config=str(Path(root) / "folds" / "2025" / "research.json"),
        action_mode="stock_trade", seed=seed, device="cuda", torch_threads=1, hidden_sizes=[256, 128],
        lookback=spec.pop("lookback"), initial_log_std=-.7, actor_output_gain=.1,
        initial_portfolio="mixed", training_episode_steps=256, recent_episode_every=4,
        max_trade_weight=.05, max_commission_ratio=.001, top_n=20, top_n_choices=[10, 20, 40],
        prior_feature_index=None, output_root=str(Path(root) / "runs"))
    config.pop("action_top_k", None)
    config.pop("max_delta_weight", None)
    config.pop("training_date_policy", None)
    config["exploration"] = dict(spec, common_lookback=5, reward_basis=pilot.get("reward_basis", "simple_net"),
                                 size_annealing=pilot.get("size_annealing", False))
    config["ppo"].update(learning_rate=1e-5, n_steps=512, batch_size=128, n_epochs=5,
        gamma=pilot.get("gamma", .99), gae_lambda=.95, clip_range=.2, target_kl=.03,
        ent_coef=0., vf_coef=.5, max_grad_norm=.5)
    config["learning_rate_schedule"] = "constant"
    return config


def make_exploration_model(env, config):
    torch.set_num_threads(config["torch_threads"])
    return PPO(ExplorationTradePolicy, env, seed=config["seed"], device=resolve_device(config["device"]), verbose=0,
        policy_kwargs=dict(hidden_sizes=tuple(config["hidden_sizes"]), initial_log_std=config["initial_log_std"],
            actor_output_gain=config["actor_output_gain"], architecture=config["exploration"]["architecture"]), **config["ppo"])


def load_agent(path, data, device="cpu"):
    agent = PPOAgent.load(path, data.contract, device)
    if (not isinstance(agent.model.policy, ExplorationTradePolicy)
            or agent.model.policy.architecture != data.architecture):
        raise ValueError("Research architecture differs from the saved artifact")
    return agent


class ExplorationCallback(TradeCallback):
    def __init__(self, *args, annealing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.annealing, self.schedule = annealing, []

    def _on_step(self):
        result = super()._on_step()
        for info in self.locals.get("infos", []):
            learning = float(info["reward"])
            raw = float(info["raw_net_return"])
            self.portfolio_totals["raw_return"] += raw - learning
            self.portfolio_totals["learning_reward"] += learning
            self.portfolio_counts["learning_reward"] += 1
        return result

    def _on_rollout_start(self):
        super()._on_rollout_start()
        if self.annealing:
            total = max(self.budgets.values())
            progress = min(1., self.model.num_timesteps / (total - self.model.n_steps))
            target = -.7 + (-2. + .7) * progress
            with torch.no_grad():
                self.model.policy.network.log_std.fill_(target)
            self.schedule.append(dict(timesteps=self.model.num_timesteps, log_std=target, sigma=float(np.exp(target))))
            write_json(self.output / "size_schedule.json", dict(events=self.schedule,
                optimizer_still_updates_std=True, mutations_only_before_rollout=True))


def _check_sources(protocol):
    if source_hashes() != protocol["source_hashes"]:
        raise ValueError("Frozen exploration source identity changed")
    if file_hash(ROOT / "doc" / "ppo_exploration_protocol_20261006.md") != protocol["document_sha256"]:
        raise ValueError("Preregistered protocol document changed during execution")
    for key, item in protocol["folds"].items():
        folder = Path(item["directory"])
        if file_hash(folder / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Fold manifest changed: " + key)
        _verify_files(folder, item["artifacts"])


def prepare(root, old_root):
    root, old_root = Path(root).resolve(), Path(old_root).resolve()
    old = read_config(old_root / "protocol.json")
    for name, digest in old["source_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Original trade implementation changed: " + name)
    folds = {}
    for fold in ("2023", "2024", "2025"):
        folder = root / "folds" / fold
        manifest = read_config(folder / "manifest.json")
        if not manifest.get("protected_period_excluded") or manifest["end_date"] > "2026-06-30":
            raise ValueError("Fold can expose the reserved performance period")
        folds[fold] = dict(directory=str(folder), manifest_sha256=file_hash(folder / "manifest.json"),
            artifacts={name: file_hash(folder / name) for name in
                       ("panel_data.parquet", "prices.parquet", "scaler_params.json", "research.json")})
    models = {}
    for seed in SEEDS:
        for budget in ("short", "long"):
            folder = old_root / "runs" / f"small_{seed}" / (budget + "_model")
            metadata = read_config(folder / "metadata.json")
            if file_hash(folder / "model.zip") != metadata["model_sha256"]:
                raise ValueError("Frozen deployment-reference model changed")
            models[f"{seed}_{budget}"] = dict(path=str(folder), model_sha256=metadata["model_sha256"],
                                              metadata_sha256=file_hash(folder / "metadata.json"))
    protocol = dict(schema_version=1, study="seven_direction_exploration", base_config=old["base_config"],
        seeds=list(SEEDS), formal_cells=FORMAL, pilots=PILOTS, budgets=BUDGETS,
        formal_training_steps=98304, pilot_training_steps=6144, actual_training_steps=104448,
        source_hashes=source_hashes(), document_sha256=file_hash(ROOT / "doc" / "ppo_exploration_protocol_20261006.md"),
        folds=folds, old_root=str(old_root), references=models, old_protocol_sha256=file_hash(old_root / "protocol.json"),
        evaluation_seed=EVAL_SEED, main_top_n=20, auxiliary_top_n=[10, 40], test_evaluated=False,
        actionable=False, original_source_count=len(old["source_hashes"]), git=git_metadata(),
        packages={name: version(name) for name in ("torch", "stable-baselines3", "numpy", "pandas", "gymnasium")})
    path = root / "protocol.json"
    if path.exists():
        saved = read_config(path)
        comparison = deepcopy(protocol)
        comparison["git"] = saved["git"]
        if saved != comparison:
            raise ValueError("Registered exploration identity changed")
        _check_sources(saved)
        return saved
    data = ExplorationData(config_for(protocol["base_config"], root, "baseline_l1", 42))
    env = data.env("train")
    try:
        obs, _ = env.reset(seed=42)
        if not env.observation_space.contains(obs) or env.unwrapped.market.stock_codes != data.codes:
            raise ValueError("Research observations or stock ordering mismatch")
    finally:
        env.close()
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, protocol)
    return protocol


def evaluate_case(agent, data, folder, label):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    identity = _trace_identity(data, label, seed=EVAL_SEED,
        initial_mode=data.settings["initial_portfolio"], top_n=data.settings["trade_top_n"])
    trace = TradeTrace(folder / "trace", identity)
    market = data.market("val")

    def capture(date, observation, action, info):
        log_prob, value = None, None
        if isinstance(agent, PPOAgent):
            tensor, _ = agent.model.policy.obs_to_tensor(observation)
            raw = np.concatenate([action.diagnostics["directions"], action.diagnostics["size_latent"]])
            packed = torch.as_tensor(raw, dtype=torch.float32, device=agent.model.device).unsqueeze(0)
            with torch.no_grad():
                predicted, probability, _ = agent.model.policy.evaluate_actions(tensor, packed)
            value, log_prob = predicted.cpu().numpy()[0], probability.cpu().numpy()[0]
        trace.append(date, observation, action, info, log_prob=log_prob, value=value,
                     reward=info["reward"], done=market.current_index == market.n_dates - 1)

    try:
        daily, holdings, orders, assumptions = run_policy(market, agent, data.settings, trace_sink=capture)
    finally:
        trace.close()
    daily.to_csv(folder / "daily.csv", index=False)
    holdings.to_parquet(folder / "positions.parquet")
    orders.to_csv(folder / "orders.csv", index=False)
    metrics = performance(daily)
    if trace.statistics.violations:
        raise AssertionError("Economic execution violation during evaluation")
    result = dict(metrics=metrics, trace=trace.manifest(), mode_statistics=trace.statistics.snapshot(),
        contract=data.contract, assumptions=assumptions, initial_mode=data.settings["initial_portfolio"],
        top_n=data.settings["trade_top_n"], first_execution=str(daily.date.iloc[1].date()),
        last_execution=str(daily.date.iloc[-1].date()), test_evaluated=False)
    write_json(folder / "result.json", result)
    return result


def rule_controls(root, protocol):
    directory = Path(root) / "rules"
    marker = directory / "completed.json"
    if marker.exists():
        record = read_config(marker)
        if record["protocol_sha256"] != json_digest(protocol):
            raise ValueError("Rule controls belong to another study")
        _verify_files(directory, record["files"])
        for mode in MODES:
            for name in ("hold", "momentum_daily", "momentum_20day"):
                actual = read_config(directory / mode / name / "result.json")["metrics"]
                if actual != record["results"][mode][name]:
                    raise ValueError("Rule marker metrics differ from native evaluation")
        return record["results"]
    if directory.exists():
        raise ValueError("Incomplete economic rule controls require review")
    data = ExplorationData(config_for(protocol["base_config"], root, "baseline_l1", 42))
    results = {}
    for mode in MODES:
        data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        results[mode] = {}
        for name, kind, interval in (("hold", "hold", 1), ("momentum_daily", "momentum_20", 1),
                                     ("momentum_20day", "momentum_20", 20)):
            result = evaluate_case(EconomicRule(data.contract, kind, interval), data, directory / mode / name, "economic_rule")
            results[mode][name] = result["metrics"]
    write_json(marker, dict(protocol_sha256=json_digest(protocol), results=results, files=_registered_files(directory),
        rule="Top K momentum_20; EXIT unselected legal full holdings; BUY selected legal names; sizes linearly 4..-4; interval HOLD.",
        identical_common_history_mask=5, old_target_weight_executor_used=False))
    return results


def run_trial(root, protocol, cell, seed):
    pilot = cell in PILOTS
    directory = Path(root) / ("pilots" if pilot else "runs") / f"{cell}_{seed}"
    config = config_for(protocol["base_config"], root, cell, seed)
    marker = directory / "completed.json"
    if marker.exists():
        saved = read_config(marker)
        if saved["config_sha256"] != json_digest(config) or saved["protocol_sha256"] != json_digest(protocol):
            raise ValueError("Trial cache identity changed")
        _verify_files(directory, saved["files"])
        for record in saved["records"]:
            audit = read_config(directory / (record["budget"] + "_audit.json"))
            metadata = read_config(directory / (record["budget"] + "_model") / "metadata.json")
            if (audit["actual_timesteps"] != record["steps"] or metadata["num_timesteps"] != record["steps"]
                    or audit["model_sha256"] != record["model_sha256"]
                    or metadata["model_sha256"] != record["model_sha256"]):
                raise ValueError("Cached record differs from saved training audit")
            for key, source in (("parameters", "parameters"), ("optimizer_steps_min", "optimizer_steps_min"),
                                ("optimizer_steps_max", "optimizer_steps_max"), ("experience", "experience"),
                                ("training_mode_statistics", "mode_statistics"),
                                ("training_diagnostics", "training_diagnostics")):
                if record[key] != audit[source]:
                    raise ValueError("Cached training diagnostics changed: " + key)
            for mode in MODES:
                native = read_config(directory / (record["budget"] + "_n20") / mode / "result.json")
                initialized = read_config(directory / "initialized" / mode / "result.json")
                if (record["metrics"][mode]["ppo"] != native["metrics"]
                        or record["metrics"][mode]["initialized"] != initialized["metrics"]
                        or record["metrics"][mode]["trace"] != native["trace"]
                        or record["metrics"][mode]["mode_statistics"] != native["mode_statistics"]):
                    raise ValueError("Cached primary metrics differ from registered native evaluation")
            for top_n, accounts in record["auxiliary_metrics"].items():
                for mode, metrics in accounts.items():
                    native = read_config(directory / (record["budget"] + "_n" + top_n) / mode / "result.json")
                    if metrics != native["metrics"]:
                        raise ValueError("Cached auxiliary metrics differ from native evaluation")
            verify_trace_prefix(directory / "training_trace", record["training_trace_prefix"])
        return saved["records"]
    if directory.exists():
        raise ValueError("Incomplete exploration trial requires review")
    _check_sources(protocol)
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    (directory / "events").mkdir()
    events.configure(directory / "events")
    data = ExplorationData(config)
    experience = EpisodeAudit(data.env("train"))
    env = DummyVecEnv([lambda: Monitor(experience)])
    env.seed(seed)
    budgets = dict(pilot=2048) if pilot else BUDGETS
    callback = None
    try:
        model = make_exploration_model(env, config)
        initial = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
        PPOAgent(model, data.contract).save(directory / "initial_model", config)
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        callback = ExplorationCallback(data, directory, experience, initial, budgets, Path(root) / "status.json",
            f"{cell}_{seed}", annealing=config["exploration"]["size_annealing"])
        torch.cuda.reset_peak_memory_stats(model.device)
        model.learn(total_timesteps=max(budgets.values()), callback=callback)
        model.logger.dump(step=model.num_timesteps)
        if callback.saved != set(budgets):
            raise AssertionError("Missing registered trained snapshots")
        if callback.trace.statistics.violations:
            raise AssertionError("Training produced an economic execution violation")
    finally:
        env.close()
        if callback is not None:
            callback.trace.close()
        events.configure(None)
    del model, env, callback, initial, experience
    _release()
    controls = {}
    initialized = load_agent(directory / "initial_model", data)
    for mode in MODES:
        data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=20)
        controls[mode] = evaluate_case(initialized, data, directory / "initialized" / mode, "initialized")
    del initialized
    records = []
    for budget, steps in budgets.items():
        agent = load_agent(directory / (budget + "_model"), data)
        main, auxiliary = {}, {}
        for top_n in ([20] if pilot else [20, 10, 40]):
            cases = {}
            for mode in MODES:
                data.settings.update(seed=EVAL_SEED, initial_portfolio=mode, trade_top_n=top_n)
                write_json(Path(root) / "status.json", dict(status="evaluating", cell=cell, seed=seed,
                    budget=budget, top_n=top_n, account=mode, test_evaluated=False))
                cases[mode] = evaluate_case(agent, data, directory / (budget + f"_n{top_n}") / mode, "trained")
            if top_n == 20:
                main = {mode: dict(ppo=cases[mode]["metrics"], initialized=controls[mode]["metrics"],
                    mode_statistics=cases[mode]["mode_statistics"], trace=cases[mode]["trace"]) for mode in MODES}
            else:
                auxiliary[str(top_n)] = {mode: result["metrics"] for mode, result in cases.items()}
        audit = read_config(directory / (budget + "_audit.json"))
        records.append(dict(cell=cell, seed=seed, budget=budget, steps=steps, pilot=pilot,
            parameters=audit["parameters"], model_sha256=audit["model_sha256"],
            model_path=str((directory / (budget + "_model")).relative_to(ROOT)),
            training_trace_prefix=audit["training_trace_prefix"], experience=audit["experience"],
            training_diagnostics=audit["training_diagnostics"], optimizer_steps_min=audit["optimizer_steps_min"],
            optimizer_steps_max=audit["optimizer_steps_max"], training_mode_statistics=audit["mode_statistics"],
            tensor_delta_l2=audit["tensor_delta_l2"], metrics=main, auxiliary_metrics=auxiliary,
            auxiliary_used_for_selection=False, test_evaluated=False))
        del agent
        _release()
    _check_sources(protocol)
    write_json(marker, dict(protocol_sha256=json_digest(protocol), config_sha256=json_digest(config),
                           records=records, files=_registered_files(directory)))
    return records


def validate_records(records):
    expected = {(cell, seed, budget, steps, False) for cell in FORMAL for seed in SEEDS
                for budget, steps in BUDGETS.items()}
    expected |= {(cell, 42, "pilot", 2048, True) for cell in PILOTS}
    keys = [(row["cell"], row["seed"], row["budget"], row["steps"], row["pilot"]) for row in records]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("A complete summary needs exactly all formal snapshots and all three pilot snapshots")
    runs = {}
    for row in records:
        key = (row["cell"], row["seed"], row["pilot"])
        runs[key] = max(runs.get(key, 0), row["steps"])
    formal = sum(steps for (_, _, pilot), steps in runs.items() if not pilot)
    pilots = sum(steps for (_, _, pilot), steps in runs.items() if pilot)
    return dict(formal_training_steps=formal, pilot_training_steps=pilots, actual_training_steps=formal+pilots,
                trained_runs=len(runs), formal_snapshots=sum(not row["pilot"] for row in records),
                pilot_snapshots=sum(row["pilot"] for row in records))


def summarize(records, rules):
    counts = validate_records(records)
    formal = [record for record in records if not record["pilot"]]
    cells = {}
    for cell in FORMAL:
        for budget, steps in BUDGETS.items():
            rows = sorted([r for r in formal if r["cell"] == cell and r["budget"] == budget], key=lambda r: r["seed"])
            if sorted(row["seed"] for row in rows) != sorted(SEEDS):
                raise ValueError("Formal summary requires every registered seed")
            averaged = [_metric_mean([row["metrics"][mode]["ppo"] for mode in MODES]) for row in rows]
            initialized = [_metric_mean([row["metrics"][mode]["initialized"] for mode in MODES]) for row in rows]
            gain = [current["cumulative_return"] - start["cumulative_return"] for current, start in zip(averaged, initialized)]
            worst = max(abs(row["metrics"][mode]["ppo"]["max_drawdown"]) for row in rows for mode in MODES)
            daily = _metric_mean([rules[mode]["momentum_daily"] for mode in MODES])["cumulative_return"]
            metrics = _metric_stats(averaged)
            cost = _stats([r["total_fees"] + r["total_slippage_cost"] for r in averaged])
            cells[f"{cell}_{budget}"] = dict(cell=cell, budget=budget, steps=steps,
                parameters=rows[0]["parameters"], metrics=metrics, own_initialization_gain=_stats(gain),
                positive_learning_seeds=sum(value > 0 for value in gain), worst_account_drawdown=worst,
                daily_momentum_gap=metrics["cumulative_return"]["mean"] - daily, cost=cost,
                qualified=bool(worst <= .10 and np.mean(gain) > 0 and sum(value > 0 for value in gain) >= 2
                               and metrics["cumulative_return"]["mean"] > daily),
                per_seed=[dict(seed=row["seed"], metrics=mean, own_initialization_gain=delta)
                          for row, mean, delta in zip(rows, averaged, gain)])
    eligible = [value for value in cells.values() if value["qualified"]]
    eligible.sort(key=lambda item: (-item["metrics"]["cumulative_return"]["mean"], item["worst_account_drawdown"],
        item["cost"]["mean"], item["steps"], item["cell"]))
    paired = {}
    for cell in FORMAL:
        if cell == "baseline_l1":
            continue
        for budget in BUDGETS:
            base = cells[f"baseline_l1_{budget}"]["per_seed"]
            other = cells[f"{cell}_{budget}"]["per_seed"]
            paired[f"{cell}_{budget}"] = _stats([b["metrics"]["cumulative_return"] - a["metrics"]["cumulative_return"]
                                                for a, b in zip(base, other)])
    return dict(cells=cells, paired_to_baseline=paired, qualified_candidate=(
        dict(cell=eligible[0]["cell"], budget=eligible[0]["budget"], representative_seed=42) if eligible else None),
        **counts,
        actionable=False, test_evaluated=False, retrospective_historical_research=True,
        aggregation="Mean four paired accounts within seed, then sample SD over three seeds.")


def write_report(root, records, rules):
    root = Path(root)
    summary = summarize(records, rules)
    write_json(root / "study_summary.json", summary)
    table = []
    for record in records:
        for mode, account in record["metrics"].items():
            table.append(dict(cell=record["cell"], seed=record["seed"], budget=record["budget"],
                account=mode, steps=record["steps"], pilot=record["pilot"], **account["ppo"]))
    pd.DataFrame(table).to_csv(root / "metrics.csv", index=False)
    lines = ["# 七方向探索结果", "", "历史回溯开发研究；非独立盲测；没有评估2026年7月后的保留期。", "",
        "| Cell | Steps | ROI% +/- seed SD | Init gain pp | Worst DD% | Mean costs | Qualified |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for value in summary["cells"].values():
        roi = value["metrics"]["cumulative_return"]
        lines.append(f"| {value['cell']} | {value['steps']} | {roi['mean']*100:.4f} +/- {roi['std']*100:.4f} | "
            f"{value['own_initialization_gain']['mean']*100:.4f} | {value['worst_account_drawdown']*100:.4f} | "
            f"{value['cost']['mean']:.2f} | {value['qualified']} |")
    lines += ["", f"Candidate: {summary['qualified_candidate']}", "",
        "Pilot只有一个seed，不参加候选排名。L5增加20480参数，split减少1280参数，均非精确容量匹配。",
        "规则与神经策略共用mapper/broker及5日可用mask，但请求生成不同；不假装旧目标权重规则等价。",
        "源数据固定2026名单与HFQ成交、缺真实公司行为/交易状态等限制未解决。",
        "[Machine summary](study_summary.json) | [All account metrics](metrics.csv) | [Raw records](summary.json)"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def coverage_diagnostic(root, protocol):
    directory = Path(root) / "coverage"
    if directory.exists():
        marker = read_config(directory / "completed.json")
        if marker["protocol_sha256"] != json_digest(protocol):
            raise ValueError("Coverage diagnostic belongs to another registered study")
        _verify_files(directory, marker["files"])
        return read_config(directory / "summary.json")
    directory.mkdir()
    results = []
    for seed in SEEDS:
        modes_by_sampler = {}
        for mode in ("recent", "quarter_balanced"):
            config = config_for(protocol["base_config"], root, "baseline_l1", seed)
            config["exploration"]["sampling"] = mode
            data = ExplorationData(config)
            env = data.env("train")
            dates = env.unwrapped.market.dates
            accounts, indices = [], []
            try:
                for episode in range(32):
                    _, info = env.reset(seed=seed if episode == 0 else None)
                    indices.append(env.unwrapped.market.current_index)
                    accounts.append(dict(date=str(info["date"].date()), mode=info["initial_account"]["mode"],
                        cash=float(info["cash"]), positions_sha256=json_digest(info["positions"].tolist())))
            finally:
                env.close()
            modes_by_sampler[mode] = [account["mode"] for account in accounts]
            for episodes in (8, 32):
                starts = indices[:episodes]
                counts = np.zeros(len(dates), dtype=np.int64)
                for start in starts:
                    counts[start:start + 256] += 1
                results.append(dict(seed=seed, sampler=mode, episodes=episodes, environment_steps=0,
                    projected_decision_visits=episodes*256, unique_dates=int((counts > 0).sum()),
                    repetition_fraction=float(1 - (counts > 0).sum() / counts.sum()),
                    start_dates=[str(dates[index].date()) for index in starts],
                    quarter_starts=pd.Series(dates[starts].to_period("Q").astype(str)).value_counts().sort_index().to_dict(),
                    initial_accounts=accounts[:episodes],
                    date_counts={str(date.date()): int(count) for date, count in zip(dates, counts) if count}))
        if modes_by_sampler["recent"] != modes_by_sampler["quarter_balanced"]:
            raise AssertionError("Account modes changed with the date sampling algorithm")
    report = dict(results=results, no_trades=True, no_parameter_updates=True,
                  reset_count=192, mode_streams_equal_across_samplers=True,
                  projected_date_coverage_not_actual_policy_experience=True)
    write_json(directory / "summary.json", report)
    write_json(directory / "completed.json", dict(protocol_sha256=json_digest(protocol),
                                                  files=_registered_files(directory)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "coverage", "train", "report"), required=True)
    parser.add_argument("--output", default="reports/runs/ppo_exploration_20261006")
    parser.add_argument("--old-root", default="reports/runs/ppo_trade_actor_20261005")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    protocol = prepare(root, args.old_root)
    _check_sources(protocol)
    if args.stage == "prepare":
        return
    if args.stage == "coverage":
        coverage_diagnostic(root, protocol)
        return
    records = []
    rules = rule_controls(root, protocol)
    if args.stage == "train":
        for cell in FORMAL:
            for seed in SEEDS:
                records.extend(run_trial(root, protocol, cell, seed))
                write_json(root / "summary.json", records)
        for cell in PILOTS:
            records.extend(run_trial(root, protocol, cell, 42))
            write_json(root / "summary.json", records)
    else:
        records = read_config(root / "summary.json")
        validate_records(records)
        cached = []
        for cell, seeds in [(cell, SEEDS) for cell in FORMAL] + [(cell, (42,)) for cell in PILOTS]:
            marker = root / ("pilots" if cell in PILOTS else "runs") / f"{cell}_{seeds[0]}" / "completed.json"
            for seed in seeds:
                path = marker.parent.parent / f"{cell}_{seed}" / "completed.json"
                if not path.exists():
                    raise ValueError("Report cannot certify a trial without a completed training artifact")
                cached.extend(run_trial(root, protocol, cell, seed))
        if cached != records:
            raise ValueError("Global summary differs from sealed per-trial records")
    write_report(root, records, rules)
    _check_sources(protocol)
    counts = validate_records(records)
    write_json(root / "status.json", dict(status="complete", **counts, test_evaluated=False, actionable=False))


if __name__ == "__main__":
    main()
