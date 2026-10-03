"""Aggregate the complete offline capacity/budget matrix without training."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


SIZES = ("small", "large")
BUDGETS = ("short", "long")
ACCOUNTS = ("cash", "equal", "concentrated", "random")
MOMENTUM_BASELINES = ("momentum_daily", "momentum")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return dict(mean=None, std=None, min=None, max=None, range=None, count=0)
    return dict(mean=mean(values), std=stdev(values) if len(values) > 1 else None,
                min=min(values), max=max(values), range=[min(values), max(values)], count=len(values))


def _metric_mean(metrics):
    keys = sorted({key for item in metrics for key in item})
    return {key: _stats(item.get(key) for item in metrics)["mean"] for key in keys}


def _metric_stats(metrics):
    keys = sorted({key for item in metrics for key in item})
    return {key: _stats(item.get(key) for item in metrics) for key in keys}


def _numeric_leaves(values, prefix=""):
    leaves = {}
    for key, value in values.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            leaves.update(_numeric_leaves(value, name + "/"))
        elif value is None or _number(value):
            if value is not None and not math.isfinite(value):
                raise ValueError("Nonfinite diagnostic: " + name)
            leaves[name] = value
    return leaves


def _validate_metric(metric, label):
    if not isinstance(metric, dict) or not _number(metric.get("cumulative_return")):
        raise ValueError("Missing cumulative_return: " + label)
    for value in metric.values():
        if value is not None and (not _number(value) or not math.isfinite(value)):
            raise ValueError("Invalid metric: " + label)


def _validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Incomplete matrix: no records")
    grouped = {(size, budget): {} for size in SIZES for budget in BUDGETS}
    baseline_names = None
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Matrix records must be objects")
        size, budget, seed = (record.get(key) for key in ("size", "budget", "seed"))
        if size not in SIZES or budget not in BUDGETS or type(seed) is not int or seed < 0:
            raise ValueError("Invalid matrix cell or seed")
        if seed in grouped[size, budget]:
            raise ValueError(f"Duplicate matrix cell: {size}/{budget}/seed {seed}")
        for name in ("steps", "parameters"):
            if type(record.get(name)) is not int or record[name] < 1:
                raise ValueError("Invalid matrix " + name)
        hidden = record.get("hidden_sizes")
        if (not isinstance(hidden, list) or len(hidden) != 2
                or any(type(width) is not int or width < 1 for width in hidden)):
            raise ValueError("Invalid hidden_sizes")
        elapsed = record.get("elapsed_seconds")
        if not _number(elapsed) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid elapsed_seconds")
        if not all(isinstance(record.get(key), dict) for key in ("training_diagnostics", "experience")):
            raise ValueError("Missing training_diagnostics or experience")
        experience = record["experience"]
        if "total_steps" in experience:
            if type(experience["total_steps"]) is not int or experience["total_steps"] != record["steps"]:
                raise ValueError("Experience steps differ from checkpoint steps")
            for name in ("mode_steps", "decision_date_counts"):
                if name in experience:
                    counts = experience[name]
                    if (not isinstance(counts, dict) or any(type(count) is not int or count < 0 for count in counts.values())
                            or sum(counts.values()) != experience["total_steps"]):
                        raise ValueError("Inconsistent experience counts: " + name)
            if "mode_steps" in experience and not set(experience["mode_steps"]).issubset(ACCOUNTS):
                raise ValueError("Invalid experience account modes")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(ACCOUNTS):
            raise ValueError("Incomplete account scenarios")
        for account, scenario in metrics.items():
            if not isinstance(scenario, dict):
                raise ValueError("Invalid account scenario")
            for strategy in ("ppo", "initialized"):
                _validate_metric(scenario.get(strategy), f"{size}/{budget}/{seed}/{account}/{strategy}")
            baselines = scenario.get("baselines")
            if not isinstance(baselines, dict) or not set(MOMENTUM_BASELINES).issubset(baselines):
                raise ValueError("Missing daily or periodic momentum baseline")
            if baseline_names is None:
                baseline_names = set(baselines)
            if set(baselines) != baseline_names:
                raise ValueError("Inconsistent baseline set")
            for name, metric in baselines.items():
                _validate_metric(metric, f"{size}/{budget}/{seed}/{account}/{name}")
        grouped[size, budget][seed] = record
    seeds = set(grouped["small", "short"])
    if not seeds or any(set(cell) != seeds for cell in grouped.values()):
        raise ValueError("Incomplete matrix: every cell must contain the same seeds")
    for budget in BUDGETS:
        if len({row["steps"] for (size, arm), cell in grouped.items() if arm == budget for row in cell.values()}) != 1:
            raise ValueError("Inconsistent step budget")
    if next(iter(grouped["small", "short"].values()))["steps"] >= next(iter(grouped["small", "long"].values()))["steps"]:
        raise ValueError("Long budget must exceed short budget")
    for size in SIZES:
        shape = {(tuple(row["hidden_sizes"]), row["parameters"])
                 for (arm, budget), cell in grouped.items() if arm == size for row in cell.values()}
        if len(shape) != 1:
            raise ValueError("Inconsistent model capacity")
    return grouped, sorted(seeds), sorted(baseline_names)


def _training_runs(records):
    runs = {}
    for record in records:
        identifier = record.get("training_run_id")
        shared = record.get("prefix_shared", False)
        if type(shared) is not bool or (identifier is not None and (not isinstance(identifier, str) or not identifier)):
            raise ValueError("Invalid training run identity")
        if shared and identifier is None:
            raise ValueError("Shared prefixes require training_run_id")
        key = ("run", identifier) if identifier is not None else ("independent", record["size"], record["budget"], record["seed"])
        runs.setdefault(key, []).append(record)
    result = []
    for rows in runs.values():
        identity = {(row["size"], row["seed"]) for row in rows}
        if len(identity) != 1:
            raise ValueError("Training run identity mixes capacities or seeds")
        if len(rows) > 1 and not all(row.get("prefix_shared", False) for row in rows):
            raise ValueError("Repeated training_run_id requires shared prefixes")
        if any(row.get("prefix_shared", False) for row in rows) and {row["budget"] for row in rows} != set(BUDGETS):
            raise ValueError("Shared training run lacks matching short and long checkpoints")
        row = max(rows, key=lambda item: item["steps"])
        result.append(dict(training_run_id=row.get("training_run_id", f"{row['size']}_{row['budget']}_{row['seed']}"),
                           size=row["size"], seed=row["seed"], steps=row["steps"],
                           checkpoint_count=len(rows), prefix_shared=len(rows) > 1))
    return result


def summarize_matrix(records):
    """Account means form each seed observation; uncertainty is across seeds."""
    grouped, seeds, baseline_names = _validate_records(records)
    training_runs = _training_runs(records)
    cells = {}
    for size in SIZES:
        for budget in BUDGETS:
            rows = [grouped[size, budget][seed] for seed in seeds]
            per_seed = []
            for row in rows:
                scenarios = list(row["metrics"].values())
                metrics = _metric_mean([scenario["ppo"] for scenario in scenarios])
                initialized = _metric_mean([scenario["initialized"] for scenario in scenarios])
                baselines = {name: _metric_mean([scenario["baselines"][name] for scenario in scenarios])
                             for name in baseline_names}
                gaps = dict(initialized=metrics["cumulative_return"] - initialized["cumulative_return"])
                gaps.update({name: metrics["cumulative_return"] - baseline["cumulative_return"]
                             for name, baseline in baselines.items()})
                per_seed.append(dict(seed=row["seed"], training_run_id=row.get("training_run_id"),
                                     prefix_shared=row.get("prefix_shared", False), metrics=metrics, initialized=initialized,
                                     baselines=baselines, return_gaps=gaps))
            per_account = {}
            for account in ACCOUNTS:
                scenarios = [row["metrics"][account] for row in rows]
                per_account[account] = dict(metrics=_metric_stats([scenario["ppo"] for scenario in scenarios]),
                    initialized=_metric_stats([scenario["initialized"] for scenario in scenarios]),
                    baselines={name: _metric_stats([scenario["baselines"][name] for scenario in scenarios])
                               for name in baseline_names})
            diagnostics = [_numeric_leaves(row["training_diagnostics"]) for row in rows]
            experience = [_numeric_leaves(row["experience"]) for row in rows]
            worst = [max(abs(row["metrics"][account]["ppo"]["max_drawdown"]) for account in ACCOUNTS)
                     for row in rows if all(row["metrics"][account]["ppo"].get("max_drawdown") is not None
                                            for account in ACCOUNTS)]
            cells[size + "_" + budget] = dict(size=size, budget=budget, seeds=seeds,
                steps=rows[0]["steps"], hidden_sizes=rows[0]["hidden_sizes"], parameters=rows[0]["parameters"],
                elapsed_seconds=_stats(row["elapsed_seconds"] for row in rows),
                metrics=_metric_stats([row["metrics"] for row in per_seed]),
                initialized=_metric_stats([row["initialized"] for row in per_seed]),
                baselines={name: _metric_stats([row["baselines"][name] for row in per_seed]) for name in baseline_names},
                return_gaps={name: _stats(row["return_gaps"][name] for row in per_seed)
                             for name in ("initialized", *baseline_names)},
                worst_account_drawdown_magnitude=_stats(worst),
                training_diagnostics=_metric_stats(diagnostics), experience=_metric_stats(experience),
                per_account=per_account, per_seed=per_seed)
    paired = []
    for index, seed in enumerate(seeds):
        returns = {key: cell["per_seed"][index]["metrics"]["cumulative_return"] for key, cell in cells.items()}
        budget = {size: returns[size + "_long"] - returns[size + "_short"] for size in SIZES}
        capacity = {arm: returns["large_" + arm] - returns["small_" + arm] for arm in BUDGETS}
        paired.append(dict(seed=seed, budget=budget, capacity=capacity,
                           interaction=budget["large"] - budget["small"]))
    contrasts = dict(budget={size: _stats(row["budget"][size] for row in paired) for size in SIZES},
                     capacity={budget: _stats(row["capacity"][budget] for row in paired) for budget in BUDGETS},
                     interaction=_stats(row["interaction"] for row in paired))
    return dict(seeds=seeds, account_scenarios=list(ACCOUNTS), baseline_names=baseline_names, cells=cells,
                contrasts=contrasts, paired_seed_effects=paired, total_runs=len(training_runs),
                total_checkpoints=len(records), training_runs=training_runs,
                total_steps=sum(row["steps"] for row in training_runs),
                checkpoint_steps_sum=sum(row["steps"] for row in records), test_evaluated=False,
                aggregation="Mean of four account scenarios per seed, then sample standard deviation across seeds; null metrics omitted.")


def _cell_order(summary):
    return [summary["cells"][size + "_" + budget] for budget in BUDGETS for size in SIZES]


def _format_stat(stat, scale=1, suffix=""):
    if stat["mean"] is None:
        return "未定义"
    uncertainty = "" if stat["std"] is None else f" +/- {stat['std'] * scale:.3f}"
    return f"{stat['mean'] * scale:.3f}{uncertainty}{suffix}"


def _cell_label(cell):
    return ("小" if cell["size"] == "small" else "大") + "模型/" + ("短" if cell["budget"] == "short" else "长")


def _table(summary):
    lines = ["| 配置 | 环境步 | 参数数 | 净收益% | 相对初始化pp | 相对每日动量pp | 相对20日动量pp | 最坏账户回撤% |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cell in _cell_order(summary):
        worst = cell["worst_account_drawdown_magnitude"]["max"]
        lines.append(f"| {_cell_label(cell)} | {cell['steps']:,} | {cell['parameters']:,} | "
                     f"{_format_stat(cell['metrics']['cumulative_return'], 100)} | "
                     f"{_format_stat(cell['return_gaps']['initialized'], 100)} | "
                     f"{_format_stat(cell['return_gaps']['momentum_daily'], 100)} | "
                     f"{_format_stat(cell['return_gaps']['momentum'], 100)} | "
                     + (f"{worst * 100:.3f}" if worst is not None else "未记录") + " |")
    return lines


def _write_metrics(root, records):
    fields = sorted({key for row in records for scenario in row["metrics"].values()
                     for metric in [scenario["ppo"], scenario["initialized"], *scenario["baselines"].values()] for key in metric})
    with (root / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["size", "budget", "seed", "steps", "parameters", "elapsed_seconds",
                                                   "training_run_id", "prefix_shared", "hidden_sizes",
                                                   "account", "strategy", *fields])
        writer.writeheader()
        for row in sorted(records, key=lambda item: (item["budget"], item["size"], item["seed"])):
            identity = {key: row[key] for key in ("size", "budget", "seed", "steps", "parameters", "elapsed_seconds")}
            identity.update(training_run_id=row.get("training_run_id"), prefix_shared=row.get("prefix_shared", False),
                            hidden_sizes=json.dumps(row["hidden_sizes"], separators=(",", ":")))
            for account in ACCOUNTS:
                scenario = row["metrics"][account]
                strategies = dict(ppo=scenario["ppo"], initialized=scenario["initialized"], **scenario["baselines"])
                for strategy, metric in strategies.items():
                    writer.writerow(dict(identity, account=account, strategy=strategy, **metric))


def _matrix_plot(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = [("Net return (%)", lambda cell: cell["metrics"]["cumulative_return"]),
              ("Learning increment vs initialization (pp)", lambda cell: cell["return_gaps"]["initialized"]),
              ("Return gap vs daily momentum (pp)", lambda cell: cell["return_gaps"]["momentum_daily"]),
              ("Return gap vs 20-day momentum (pp)", lambda cell: cell["return_gaps"]["momentum"])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, getter) in zip(axes.flat, panels):
        stats = [[getter(summary["cells"][size + "_" + budget]) for size in SIZES] for budget in BUDGETS]
        values = np.array([[stat["mean"] * 100 for stat in row] for row in stats])
        limit = max(float(np.abs(values).max()), .001)
        ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(xticks=[0, 1], xticklabels=["Small", "Large"], yticks=[0, 1], yticklabels=["Short", "Long"], title=title)
        for i, row in enumerate(stats):
            for j, stat in enumerate(row):
                label = f"{stat['mean'] * 100:.3f}"
                if stat["std"] is not None:
                    label += f"\n+/- {stat['std'] * 100:.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=13,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.8, boxstyle="square,pad=.3"))
    fig.suptitle(f"Development validation matrix: {len(summary['seeds'])} seeds; four account scenarios averaged within seed")
    fig.savefig(root / "matrix.png", dpi=160)
    plt.close(fig)


def _training_plot(root, records, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd

    colors = dict(small_short="#30836d", large_short="#c76e37", small_long="#367da2", large_long="#955b91")
    plots = [("train/explained_variance", "Critic explained variance"), ("train/value_loss", "Critic value loss"),
             ("train/approx_kl", "Approximate KL"), ("train/clip_fraction", "Clip fraction"),
             ("optimizer/steps_min", "Actual optimizer steps"), ("time/elapsed_seconds", "Elapsed seconds (includes callbacks)")]
    loaded = []
    for record in records:
        path = record.get("training_updates_path")
        if not path:
            continue
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ValueError("Missing training CSV: " + str(path))
        frame = pd.read_csv(path)
        if "timesteps" not in frame:
            raise ValueError("Training CSV lacks timesteps: " + str(path))
        frame = frame[frame.timesteps <= record["steps"]]
        loaded.append((record, frame))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, plots):
        plotted = False
        for record, frame in loaded:
            if metric not in frame:
                continue
            finite = frame[["timesteps", metric]].dropna()
            finite = finite[finite.timesteps > 0]
            if not finite.empty:
                ax.plot(finite.timesteps, finite[metric], color=colors[record["size"] + "_" + record["budget"]],
                        alpha=.65, linewidth=1.2)
                if record["budget"] == "short":
                    ax.scatter(finite.timesteps.iloc[-1], finite[metric].iloc[-1],
                               color=colors[record["size"] + "_short"], s=20, zorder=4)
                plotted = True
        if plotted:
            ax.set_xscale("log")
            ax.set_xlabel("Environment steps (log scale)")
            ax.grid(alpha=.2)
        else:
            cells = _cell_order(summary)
            values = []
            for cell in cells:
                if metric == "time/elapsed_seconds":
                    stat = cell["elapsed_seconds"]
                else:
                    stat = cell["training_diagnostics"].get(metric)
                values.append(stat["mean"] if stat else None)
            if all(value is not None for value in values):
                ax.bar(range(4), values, color=[colors[cell["size"] + "_" + cell["budget"]] for cell in cells])
                ax.set_xticks(range(4), [cell["size"] + "\n" + cell["budget"] for cell in cells])
                ax.set_xlabel("Recorded diagnostic mean across seeds")
            else:
                ax.text(.5, .5, "No diagnostic recorded", transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
        ax.set_title(title)
    handles = [Line2D([0], [0], color=color, label=key.replace("_", "/")) for key, color in colors.items()]
    axes[0, 0].legend(handles=handles, fontsize=8)
    fig.suptitle("Training diagnostics: individual seed trajectories; not an isolated causal test of the Critic")
    fig.savefig(root / "training_comparison.png", dpi=150)
    plt.close(fig)


def _experience_artifacts(root, records, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for record in records:
        values = _numeric_leaves(record["experience"])
        values = {key: value for key, value in values.items() if not key.startswith("decision_date_counts/")}
        identity = {key: record[key] for key in ("size", "budget", "seed", "steps")}
        identity.update(training_run_id=record.get("training_run_id"), prefix_shared=record.get("prefix_shared", False))
        rows.append(dict(identity, **values))
    fields = ["size", "budget", "seed", "steps", "training_run_id", "prefix_shared"]
    fields += sorted({key for row in rows for key in row} - set(fields))
    with (root / "experience.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cells = _cell_order(summary)
    labels = [cell["size"] + "\n" + cell["budget"] for cell in cells]
    colors = ["#30836d", "#c76e37", "#367da2", "#955b91"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [("unique_decision_dates", "Distinct historical decision dates"),
              ("unique_initial_states", "Distinct initial account/date states"),
              ("unique_initial_accounts", "Distinct initial accounts")]
    for ax, (key, title) in zip(axes.flat, panels):
        plotted = False
        for index, cell in enumerate(cells):
            stat = cell["experience"].get(key)
            if stat is None or stat["mean"] is None:
                continue
            ax.bar(index, stat["mean"], color=colors[index], alpha=.7)
            if stat["std"] is not None:
                ax.errorbar(index, stat["mean"], yerr=stat["std"], color="black", capsize=4, fmt="none")
            raw = [row.get(key) for row in rows if (row["size"], row["budget"]) == (cell["size"], cell["budget"])
                   and row.get(key) is not None]
            ax.scatter(np.linspace(index - .08, index + .08, len(raw)), raw, color="black", s=12, zorder=3)
            plotted = True
        ax.set_xticks(range(4), labels)
        ax.set_title(title)
        if not plotted:
            ax.text(.5, .5, "No experience audit recorded", transform=ax.transAxes, ha="center", va="center")
        ax.grid(axis="y", alpha=.2)
    ax = axes[1, 1]
    bottom = np.zeros(4)
    plotted = False
    for account, color in zip(ACCOUNTS, ["#519d81", "#dda354", "#5596be", "#bc79a3"]):
        stats = [cell["experience"].get("mode_steps/" + account) for cell in cells]
        if any(stat is None or stat["mean"] is None for stat in stats):
            continue
        values = np.array([stat["mean"] for stat in stats])
        ax.bar(range(4), values, bottom=bottom, color=color, label=account)
        bottom += values
        plotted = True
    ax.set_xticks(range(4), labels)
    ax.set_title("Experienced steps by initial account mode")
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "No account-mode counts recorded", transform=ax.transAxes, ha="center", va="center")
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("Training exposure at checkpoint: seed means; shared prefixes are not separate training")
    fig.savefig(root / "experience_comparison.png", dpi=150)
    plt.close(fig)


def write_report(root):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    raw = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    records = raw["records"] if isinstance(raw, dict) else raw
    summary = summarize_matrix(records)
    if protocol.get("test_evaluated") is True:
        raise ValueError("This report only supports development validation")
    if "seeds" in protocol and sorted(protocol["seeds"]) != summary["seeds"]:
        raise ValueError("Matrix seeds differ from the registered protocol")
    (root / "matrix_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_metrics(root, records)
    _matrix_plot(root, summary)
    _training_plot(root, records, summary)
    _experience_artifacts(root, records, summary)
    caveat = ("本轮仅比较已经暴露的开发验证窗口，不是独立最终测试。2026-07-01之后的保留期未做绩效评估。"
              "四种账户是同一种子的账户场景，不是四个独立训练种子；表内先平均四账户，再报告种子间样本标准差。")
    lines = ["# PPO训练预算与模型容量对比", "", caveat, "",
             f"共{summary['total_runs']}次真实训练、{summary['total_checkpoints']}个比较快照，"
             f"实际训练量{summary['total_steps']:,}环境步，种子{summary['seeds']}。"
             "使用同源新浪300股、相同费用与HFQ研究成交假设，按预定最终checkpoint比较。", "",
             "共享前缀按training_run_id分组，每次真实训练只计最大checkpoint步数。"
             f"快照步数之和为{summary['checkpoint_steps_sum']:,}，不把这个和当作训练量。"
             "带prefix_shared标记的短快照是同一训练过程的前缀，其长减短描述继续训练后的变化，"
             "不是两个独立训练样本；没有运行ID的记录按各格独立训练计。", "",
             "收益单位为百分比，差值单位为百分点(pp)，+/-为跨种子样本标准差；最坏回撤取全部账户与种子的最大幅度。", "",
             *_table(summary), "", "![容量与预算矩阵](matrix.png)", "",
             "## 学习、先验与强基线", "",
             "相对初始化先验的差值才用于检查训练后的收益增量。每日动量是同频对照，20日调仓动量保留较强规则对照；"
             "不能仅凭优于旧PPO就宣称强化学习带来收益。费用和滑点已经计入净值，不重复扣减。", "",
             "| 配置 | Sharpe | 平均回撤% | 日均换手% | 实际平均现金% | 费用 | 滑点 | 训练秒 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    empty = _stats([])
    for cell in _cell_order(summary):
        metrics = cell["metrics"]
        values = [_format_stat(metrics.get(name, empty), scale) for name, scale in
                  [("sharpe", 1), ("max_drawdown", 100), ("mean_daily_turnover", 100),
                   ("mean_cash_ratio", 100), ("total_fees", 1), ("total_slippage_cost", 1)]]
        lines.append("| " + _cell_label(cell) + " | " + " | ".join(values + [_format_stat(cell["elapsed_seconds"])]) + " |")
    lines += ["", "## 同种子配对差值", "", "所有差值先在同一训练种子上计算，再跨种子汇总。正值表示前项收益更高。", "",
              "| 对比 | 平均收益差pp +/- 标准差 | 范围pp |", "| --- | ---: | ---: |"]
    effects = [("小模型：长减短", summary["contrasts"]["budget"]["small"]),
               ("大模型：长减短", summary["contrasts"]["budget"]["large"]),
               ("短训练：大减小", summary["contrasts"]["capacity"]["short"]),
               ("长训练：大减小", summary["contrasts"]["capacity"]["long"]),
               ("交互：(大长-大短)-(小长-小短)", summary["contrasts"]["interaction"])]
    for label, stat in effects:
        lines.append(f"| {label} | {_format_stat(stat, 100)} | {stat['min'] * 100:.3f}至{stat['max'] * 100:.3f} |")
    lines += ["", "| Seed | 小模型长减短pp | 大模型长减短pp | 短训练大减小pp | 长训练大减小pp | 交互pp |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for effect in summary["paired_seed_effects"]:
        values = [effect["budget"]["small"], effect["budget"]["large"], effect["capacity"]["short"],
                  effect["capacity"]["long"], effect["interaction"]]
        lines.append(f"| {effect['seed']} | " + " | ".join(f"{value * 100:.3f}" for value in values) + " |")
    lines += ["", "## 账户与种子明细", "", "| 配置 | Seed | 账户 | 净收益% | 相对初始化pp | 相对每日动量pp | 相对20日动量pp |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for record in sorted(records, key=lambda row: (row["budget"], row["size"], row["seed"])):
        for account in ACCOUNTS:
            scenario = record["metrics"][account]
            value = scenario["ppo"]["cumulative_return"]
            gaps = [value - scenario["initialized"]["cumulative_return"],
                    *[value - scenario["baselines"][name]["cumulative_return"] for name in MOMENTUM_BASELINES]]
            lines.append(f"| {_cell_label(record)} | {record['seed']} | {account} | {value * 100:.3f} | "
                         + " | ".join(f"{gap * 100:.3f}" for gap in gaps) + " |")
    lines += ["", "## 训练与Critic诊断", "",
              "![训练诊断](training_comparison.png)", "",
              "value_loss、explained_variance、KL和clip fraction描述各自on-policy训练过程。"
              "这些指标不是固定验证集的Critic预测精度；value_loss下降也不直接证明策略收益改善。"
              "容量与预算同时影响共享编码器、Actor与Critic，本矩阵不能识别Critic单独的因果贡献。"
              "每条曲线仅绘制到相应checkpoint步数，圆点标记短预算终点。", "",
              "机器汇总保留训练诊断、经验覆盖与真实优化器更新计数，逐账户策略指标见metrics.csv。"
              "未定义指标保留null，汇总跳过null并附有效种子数；单种子标准差为null。", "",
              "## 训练经验与覆盖", "", "![训练经验覆盖](experience_comparison.png)", "",
              "经验量记录到各checkpoint为止的实际采样；共享短前缀不另计一份训练量。"
              "不同历史日期、不同初始账户和账户/日期状态分别统计，重复训练步不会自动增加独立历史信息。"
              "图中点表示各训练种子，柱为种子均值；四账户步骤堆叠图显示训练采样覆盖。", "",
              "HFQ与100股整手可能使实际仓位大幅偏离目标仓位；固定2026年股票名单有选择偏差，"
              "晚上市股票反复采样不增加独立信息。开发窗口短，种子数量有限，不据此发布实盘模型或保证未来效果。", "",
              "[预登记协议](protocol.json) · [完整矩阵汇总](matrix_summary.json) · [逐账户指标](metrics.csv) · [经验统计](experience.csv)"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    short = ["# 训练矩阵摘要", "", caveat, "",
             f"{summary['total_runs']}次真实训练、{summary['total_checkpoints']}个快照；"
             f"实际训练量{summary['total_steps']:,}环境步。共享短前缀不重复计入训练量。", "", *_table(summary), "",
             "差值必须与初始化先验、同频每日动量及20日调仓动量一起解释。"
             "本矩阵同时改变共享编码器、Actor与Critic的容量，不能宣称Critic单独产生了收益增量。", "",
             "[完整报告](report.md) · [机器汇总](matrix_summary.json) · [全部指标](metrics.csv)"]
    (root / "summary.md").write_text("\n".join(short) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = write_report(args.output)
    print(json.dumps(dict(total_runs=summary["total_runs"], total_checkpoints=summary["total_checkpoints"],
                          total_steps=summary["total_steps"], checkpoint_steps_sum=summary["checkpoint_steps_sum"],
                          seeds=summary["seeds"], test_evaluated=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
