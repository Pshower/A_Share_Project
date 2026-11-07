"""Compare the validation backtest taken at every training checkpoint.

Two training runs that differ only in their reward belong side by side, but the reader has to
be able to see that for themselves: this tool only tabulates and plots, it does not judge.
The reference column is the daily-rebalanced equal-weight portfolio on the same validation
split, recomputed independently for each run, so an "excess" is selection net of beta and of
the account's own cash drag.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocess import file_hash, write_json

COLUMNS = ("step", "policy_return", "reference_return", "excess_return", "sharpe",
           "max_drawdown", "mean_cash_ratio", "mean_daily_turnover", "total_fees",
           "tracking_error", "information_ratio")


def load(run):
    """One run's checkpoint table, with the run's own label attached."""
    run = Path(run)
    frame = pd.read_csv(run / "checkpoints.csv")
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{run.name} is missing checkpoint columns: {missing}")
    if frame.empty:
        raise ValueError(f"{run.name} has no checkpoints")
    frame = frame.sort_values("step").reset_index(drop=True)
    frame["run"] = run.name
    return frame


def summarise(frame):
    """Headline numbers per run; nothing is selected on, because selection is a decision."""
    excess = frame["excess_return"]
    best = frame.loc[excess.idxmax()]
    worst = frame.loc[excess.idxmin()]
    return dict(run=str(frame["run"].iloc[0]), checkpoints=int(len(frame)),
                first_step=int(frame.step.iloc[0]), last_step=int(frame.step.iloc[-1]),
                reference_return=float(frame.reference_return.iloc[-1]),
                first_excess=float(excess.iloc[0]), final_excess=float(excess.iloc[-1]),
                best_excess=float(best.excess_return), best_step=int(best.step),
                worst_excess=float(worst.excess_return), worst_step=int(worst.step),
                excess_mean=float(excess.mean()), excess_sd=float(excess.std(ddof=1)),
                policy_return_final=float(frame.policy_return.iloc[-1]),
                cash_first=float(frame.mean_cash_ratio.iloc[0]),
                cash_final=float(frame.mean_cash_ratio.iloc[-1]),
                turnover_final=float(frame.mean_daily_turnover.iloc[-1]),
                fees_final=float(frame.total_fees.iloc[-1]))


def plot(frames, path):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for frame in frames:
        label = frame["run"].iloc[0]
        axes[0].plot(frame.step, frame.excess_return * 100, marker="o", markersize=3, label=label)
        axes[1].plot(frame.step, frame.mean_cash_ratio * 100, marker="o", markersize=3, label=label)
        axes[2].plot(frame.step, frame.mean_daily_turnover * 100, marker="o", markersize=3, label=label)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Validation excess over equal weight (%)")
    axes[1].set_title("Mean cash ratio (%)")
    axes[2].set_title("Mean daily turnover (%)")
    for axis in axes:
        axis.set_xlabel("training step")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _markdown(payload):
    lines = ["# 训练 checkpoint 的验证集回测", "",
             f"参考组合：{payload['reference']}（每个 run 独立重算，同一验证切分）。",
             f"验证步数：{payload['steps']}。", "",
             "| run | 检查点 | 最终组合 | 参考 | 最终超额 | 首点超额 | 最好 | 最差 | 平均超额 | 现金(首→末) | 费用 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["runs"]:
        lines.append("| {} | {} | {:.2%} | {:.2%} | {:+.2%} | {:+.2%} | {:+.2%} @{} | {:+.2%} @{} | {:+.2%} | {:.1%} → {:.1%} | {:.2%} |".format(
            row["run"], row["checkpoints"], row["policy_return_final"], row["reference_return"],
            row["final_excess"], row["first_excess"],
            row["best_excess"], row["best_step"], row["worst_excess"], row["worst_step"],
            row["excess_mean"], row["cash_first"], row["cash_final"], row["fees_final"] / 1e6))
    lines += ["", "![checkpoints](checkpoints.png)", ""]
    for row in payload["runs"]:
        lines += [f"## {row['run']}", "",
                  "| step | 组合 | 参考 | 超额 | Sharpe | 回撤 | 现金 | 换手 | 费用 | 跟踪误差 |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for _, point in payload["points"][row["run"]].iterrows():
            lines.append("| {} | {:.2%} | {:.2%} | {:+.2%} | {:.2f} | {:.2%} | {:.1%} | {:.2%} | {:.0f} | {:.2%} |".format(
                int(point.step), point.policy_return, point.reference_return, point.excess_return,
                point.sharpe, point.max_drawdown, point.mean_cash_ratio,
                point.mean_daily_turnover, point.total_fees, point.tracking_error))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", required=True, help="run directories with checkpoints.csv")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    frames = [load(run) for run in arguments.run]
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(reference="equal_weight_daily", steps=int(frames[0].matched_steps.iloc[-1]),
                   runs=[summarise(frame) for frame in frames],
                   points={frame["run"].iloc[0]: frame for frame in frames},
                   source_hashes={str(Path(run).resolve() / "checkpoints.csv"):
                                  file_hash(Path(run) / "checkpoints.csv") for run in arguments.run})
    write_json(output / "checkpoints.json",
               {**payload, "points": {key: value.to_dict("records") for key, value in payload["points"].items()}})
    (output / "checkpoints.md").write_text(_markdown(payload), encoding="utf-8")
    plot(frames, output / "checkpoints.png")
    for row in payload["runs"]:
        print("{}: checkpoints={} final_excess={:+.2%} best={:+.2%}@{} cash={:.1%}->{:.1%}".format(
            row["run"], row["checkpoints"], row["final_excess"], row["best_excess"],
            row["best_step"], row["cash_first"], row["cash_final"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
