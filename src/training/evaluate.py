"""Shared validation and explicitly frozen final-test accounting."""

import argparse
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.ppo_agent import PPOAgent
from src.backtest.metrics import performance
from src.backtest.run import run_policy, run_strategy, plot_report, git_metadata
from src.data.preprocess import ROOT, file_hash, read_config, resolve_path, write_json
from .env_factory import ResearchData
from .freeze import validate_frozen


def evaluate_agent(agent, data, output, compare_baselines=False, *, split="val", frozen=None):
    """No gradients; callers label initialized policies as connectivity checks only."""
    if agent.contract != data.contract:
        raise ValueError("Evaluation data does not match the model contract")
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    if split == "test":
        if frozen is None:
            raise ValueError("Final test requires a frozen selection")
        validate_frozen(agent, data, frozen)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, metrics = {}, {}
    daily, holdings, orders, assumptions = run_policy(data.market(split, allow_test=split == "test"), agent, data.settings)

    def record(name, daily, holdings, orders):
        folder = output / name
        folder.mkdir()
        daily.to_csv(folder / "daily.csv", index=False)
        holdings.to_parquet(folder / "positions.parquet")
        orders.to_csv(folder / "orders.csv", index=False)
        results[name] = daily
        metrics[name] = performance(daily)

    record("ppo", daily, holdings, orders)
    if compare_baselines:
        for name, interval in [("cash", None), ("buy_hold", None), ("equal_weight", None),
                               ("momentum", None), ("equal_weight", 1), ("momentum", 1)]:
            settings = deepcopy(data.settings)
            if interval is not None:
                settings["rebalance_every"] = interval
            daily, holdings, orders, _ = run_strategy(data.market(split, allow_test=split == "test"), name, settings)
            record(name if interval is None else name + "_daily", daily, holdings, orders)
    write_json(output / "metrics.json", metrics)
    pd.DataFrame(metrics).T.to_csv(output / "metrics.csv")
    write_json(output / "experiment.json", dict(contract=data.contract, config=data.config,
               research_config=data.research, split=split, num_timesteps=agent.model.num_timesteps,
               frozen_selection=frozen,
               provenance=agent.provenance, git=git_metadata(),
               packages={name: version(name) for name in ["torch", "stable-baselines3", "gymnasium", "numpy", "pandas", "pyarrow", "matplotlib"]},
               source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "src").rglob("*.py")},
               assumptions=data.manifest["limitations"] + assumptions))
    plot_report(results, output, title=f"PPO {split} - " + data.contract["price_basis"])
    lines = [f"# PPO {split.title()} Report", "", f"Split: {split}. No parameter updates during evaluation.",
             f"Model timesteps: {agent.model.num_timesteps}. Zero means no updates in this instance; inspect transfer provenance.",
             "", "| Strategy | Return | Sharpe | Max drawdown | Fees | Mean cash |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in metrics.items():
        lines.append(f"| {name} | {values['cumulative_return']:.4%} | {values['sharpe']} | "
                     f"{values['max_drawdown']:.4%} | {values['total_fees']:.2f} | {values['mean_cash_ratio']:.2%} |")
    lines += ["", "![Validation curves](curves.png)", "", "## Limitations", "",
              "HFQ research prices, not real execution quotes. All strategies use the same broker settings.",
              "PPO is daily; *_daily controls match its rebalance frequency. Original periodic baselines are retained.",
              "Fees/slippage are already included in equity, and are not subtracted twice.",
              "Annualization: 252 days, zero risk-free rate; undefined Sharpe is null.",
              "No external index comparison is included by this PPO evaluator."]
    lines += [f"- {item}" for item in data.manifest["limitations"] + assumptions]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics["ppo"]


def selection_score(metrics):
    sharpe = metrics["sharpe"]
    if sharpe is None or not np.isfinite(sharpe):
        return None
    return float(sharpe), float(metrics["cumulative_return"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo.json")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--frozen-selection", type=Path)
    args = parser.parse_args(argv)
    if (args.split == "test") != (args.frozen_selection is not None):
        parser.error("--split test requires --frozen-selection; validation must not use it")
    data = ResearchData(read_config(args.config))
    agent = PPOAgent.load(resolve_path(args.model), data.contract, data.config["device"])
    if agent.model.num_timesteps == 0 and not agent.provenance.get("source_num_timesteps", 0):
        raise ValueError("Untrained artifacts are connectivity checks, not validation candidates")
    frozen = read_config(resolve_path(args.frozen_selection)) if args.frozen_selection else None
    evaluate_agent(agent, data, resolve_path(args.output), compare_baselines=True, split=args.split, frozen=frozen)


if __name__ == "__main__":
    main()
