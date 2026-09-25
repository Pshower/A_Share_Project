"""Build local data if absent, then run a reproducible validation baseline suite."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.data.preprocess import ROOT, build_from_config, file_hash, read_config, resolve_path, write_json
from src.envs.broker import BrokerConfig
from src.envs.market import MarketDataProvider
from src.envs.trading import TradingEnv
from src.runtime import events
from .metrics import performance
from .strategies import Baseline


def git_metadata():
    def command(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    try:
        return dict(commit=command("rev-parse", "HEAD"), dirty=bool(command("status", "--porcelain")))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


def run_strategy(market, name, settings):
    broker_config = BrokerConfig(**settings["broker"])
    strategy = Baseline(name, market.feature_cols, max_weight=broker_config.max_weight,
                        rebalance_every=settings["rebalance_every"], top_k=settings["momentum_top_k"])
    return run_policy(market, strategy, settings)


def run_policy(market, strategy, settings):
    """Shared execution and accounting audit for baselines and learned policies."""
    env = TradingEnv(market, initial_capital=settings["initial_capital"],
                     broker_config=BrokerConfig(**settings["broker"]))
    obs, initial = env.reset(seed=settings["seed"])
    daily = [dict(date=initial["date"], equity=initial["equity"], cash=initial["cash"],
                  daily_return=0.0, fee=0.0, slippage_cost=0.0, turnover=0.0,
                  holding_count=0, unfilled_orders=0)]
    positions = [np.zeros(market.n_stocks)]
    previous = initial
    step = 0
    while True:
        events.check_cancel()
        action = strategy.act(obs, step)
        obs, reward, terminated, truncated, info = env.advance() if action is None else env.step(action)
        orders = info["orders"]
        fees = sum(o["fee"] for o in orders)
        slippage = sum(o["slippage_cost"] for o in orders)
        net = sum(o["filled_shares"] * o["price"] for o in orders if o["filled_shares"])
        gross = sum(abs(o["filled_shares"]) * o["price"] for o in orders if o["filled_shares"])
        expected_positions = previous["positions"].copy() + info["event_shares"]
        code_index = {code: i for i, code in enumerate(market.stock_codes)}
        for order in orders:
            expected_positions[code_index[order["code"]]] += order["filled_shares"]
        np.testing.assert_allclose(info["positions"], expected_positions, rtol=0, atol=1e-8)
        np.testing.assert_allclose(info["cash"], previous["cash"] + info["event_cash"] - net - fees, rtol=0, atol=1e-6)
        np.testing.assert_allclose(info["total_fees"], previous["total_fees"] + fees, rtol=0, atol=1e-6)
        closes = market.prices.iloc[market.current_index].to_numpy()
        held = info["positions"] > 0
        np.testing.assert_allclose(info["equity"], info["cash"] + info["receivables"] + np.dot(info["positions"][held], closes[held]),
                                   rtol=0, atol=1e-6)
        if not env.observation_space.contains(obs):
            raise ValueError("Invalid model observation")
        daily.append(dict(date=info["date"], equity=info["equity"], cash=info["cash"], daily_return=reward,
                          fee=fees, slippage_cost=slippage, turnover=gross / previous["equity"],
                          holding_count=int(held.sum()),
                          unfilled_orders=sum(o["status"] != "filled" for o in orders)))
        positions.append(info["positions"].copy())
        previous = info
        step += 1
        if step % 20 == 0 or terminated or truncated:
            events.emit("backtest_progress", step=step, total=market.n_dates - 1)
        if terminated or truncated:
            break
    daily = pd.DataFrame(daily)
    daily["net_value"] = daily.equity / daily.equity.iloc[0]
    daily["drawdown"] = daily.equity / daily.equity.cummax() - 1
    holdings = pd.DataFrame(positions, index=daily.date, columns=market.stock_codes)
    orders = pd.DataFrame(env.broker.orders, columns=["date", "code", "side", "target_weight", "requested_shares",
                         "filled_shares", "price", "fee", "slippage_cost", "status", "reasons", "cash_after", "position_after"])
    return daily, holdings, orders, info["assumptions"]


def plot_report(results, output, title="Validation baselines - HFQ research simulation"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for name, daily in results.items():
        axes[0].plot(daily.date, daily.net_value, label=name)
        axes[1].plot(daily.date, daily.drawdown)
        axes[2].plot(daily.date, daily.fee.cumsum() + daily.slippage_cost.cumsum())
    for axis, label in zip(axes, ["Net value", "Drawdown", "Cumulative fees + slippage"]):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    fig.suptitle(title)
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)


def run_suite(config, run_id=None):
    data_dir = resolve_path(config["dataset"]["clean_dir"])
    if not data_dir.exists():
        build_from_config(config)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    settings = config["backtest"]
    if settings["split"] != "val":
        raise ValueError("This baseline-selection command only runs validation; final test data stays untouched")
    for field in ["train_end", "val_end"]:
        if manifest[field] != config["dataset"][field]:
            raise ValueError(f"Config does not match dataset {field}")
    expected = config["dataset"]
    for field in ["version", "start_date", "end_date", "train_end", "val_end", "volume_multiplier", "raw_pattern"]:
        value = expected.get(field, "{code}_data_*_hfq.csv" if field == "raw_pattern" else None)
        if manifest["build_config"][field] != value:
            raise ValueError(f"Config does not match dataset {field}; build a new version")
    requested_codes = expected.get("train_codes")
    if requested_codes is None:
        requested_codes = [c.strip() for c in resolve_path(expected["stock_list_path"]).read_text(encoding="utf-8-sig").splitlines() if c.strip()]
    if manifest["build_config"]["train_codes"] != requested_codes:
        raise ValueError("Config does not match dataset training stock pool")
    for name in ["panel_data.parquet", "prices.parquet", "scaler_params.json"]:
        if file_hash(data_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"Dataset artifact was modified: {name}")
    features = pd.read_parquet(data_dir / "panel_data.parquet")
    prices = pd.read_parquet(data_dir / "prices.parquet")
    # Include the last training close for the first validation opening decision.
    start = features.index[features.index <= pd.Timestamp(manifest["train_end"])][-1]
    end = pd.Timestamp(manifest["val_end"])
    output_root = resolve_path(settings["output_root"])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    results, metrics = {}, {}
    assumptions = []
    for name in ["cash", "buy_hold", "equal_weight", "momentum"]:
        market = MarketDataProvider(features, manifest["feature_cols"], price_path=prices,
                                    price_basis=manifest["price_basis"], start_date=start, end_date=end,
                                    lookback=settings["lookback"])
        daily, holdings, orders, assumptions = run_strategy(market, name, settings)
        folder = output / name
        folder.mkdir()
        daily.to_csv(folder / "daily.csv", index=False)
        holdings.to_parquet(folder / "positions.parquet")
        orders.to_csv(folder / "orders.csv", index=False)
        results[name] = daily
        metrics[name] = performance(daily)
        print(f"{name}: steps={len(daily)-1} orders={len(orders)} final_equity={daily.equity.iloc[-1]:.2f}")
    benchmark_status = "No benchmark path configured; no index comparison or network request."
    if settings.get("benchmark_path"):
        path = resolve_path(settings["benchmark_path"])
        if not path.exists():
            benchmark_status = f"Local benchmark missing: {path.name}; not downloaded."
        else:
            benchmark = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"]
            aligned = benchmark.reindex(results["cash"].date)
            if aligned.isna().any() or not np.isfinite(aligned).all() or (aligned <= 0).any():
                raise ValueError("Benchmark must supply a valid close for every evaluation date")
            daily = results["cash"].copy()
            daily["equity"] = settings["initial_capital"] * aligned.to_numpy() / aligned.iloc[0]
            daily["cash"] = 0.0
            daily["net_value"] = daily.equity / daily.equity.iloc[0]
            daily["drawdown"] = daily.equity / daily.equity.cummax() - 1
            daily["daily_return"] = daily.equity.pct_change().fillna(0)
            daily.to_csv(output / "index_benchmark.csv", index=False)
            metrics["index_benchmark"] = performance(daily)
            results["index_benchmark"] = daily
            benchmark_status = "Local index close-to-close return; excludes fees and is not a traded portfolio."
    pd.DataFrame(metrics).T.to_csv(output / "metrics.csv")
    write_json(output / "metrics.json", metrics)
    write_json(output / "config.json", config)
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for area in ["src", "configs"]
                     for p in (ROOT / area).rglob("*") if p.is_file() and p.suffix in {".py", ".json"}}
    metadata = dict(git=git_metadata(), source_hashes=source_hashes,
                    dataset_manifest_sha256=file_hash(data_dir / "manifest.json"), dataset_version=manifest["version"],
                    feature_version=manifest["schema_version"], seed=settings["seed"], split="val",
                    initial_date=str(start.date()), final_date=str(end.date()),
                    packages={name: version(name) for name in ["numpy", "pandas", "gymnasium", "pyarrow", "matplotlib"]},
                    benchmark=benchmark_status, assumptions=manifest["limitations"] + assumptions)
    write_json(output / "experiment.json", metadata)
    plot_report(results, output)
    lines = ["# Validation Baseline Report", "", f"Dataset: {manifest['version']}; split: val; test split was not evaluated.",
             "", "| Strategy | Cumulative return | Annual volatility | Max drawdown | Fees | Gross turnover | Mean cash |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value['cumulative_return']:.4%} | {value['annual_volatility']:.4%} | "
                     f"{value['max_drawdown']:.4%} | {value['total_fees']:.2f} | {value['gross_turnover']:.4f} | {value['mean_cash_ratio']:.2%} |")
    lines += ["", "![Validation curves](curves.png)", "", "## Definitions", "",
              "Annualization uses 252 steps/year. Sharpe uses zero risk-free rate; zero volatility returns null.",
              "Drawdown includes starting capital. Turnover is gross traded notional / previous close equity (two-sided).",
              "Equity already includes fees and slippage. Slippage cost is diagnostic and is not deducted twice.",
              "Buy-and-hold submits one equal-weight target then no further orders; rejected quantities are not retried.",
              "Periodic equal weight and top-K 20-day momentum rebalance at the configured step interval.",
              "All baselines share dates, pool and execution settings. Cash allocation follows stable stock order.",
              "HFQ prices plus lot constraints may leave substantial cash; equal-weight targets do not imply equal realized holdings.",
              "", "## Limitations", "", benchmark_status, ""]
    lines += [f"- {note}" for note in metadata["assumptions"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {output / 'report.md'}")
    return output, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research.json")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_suite(read_config(args.config), args.run_id)
