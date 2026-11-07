"""Does the one-day reversal survive the broker? A cost-aware backtest of a fixed construction.

The probe measured a one-day cross-sectional reversal with |t| up to 7 and no overlap penalty.
That is a signal, not a strategy.  The arithmetic that decides the strategy is brutal: a top
book earning ``IC * sigma_cs_1d * E[z]`` about 0.14% a day against roughly 0.31% for one full
round trip.  Whether anything survives therefore depends entirely on turnover, and turnover is
not a property of the signal -- it is a property of how fast the ranking moves through names
while lots, T+1 and slippage are all applied.

So this tool runs the same executions as everything else in the repository: the real broker,
real lot rounding, T+1 availability, commission, stamp duty and slippage.  Every construction
is run twice, with and without friction, so the cost bill is measured rather than assumed.

Three fixed constructions, no search over which feature to trade:

* ``reversal_return`` ranks by minus yesterday's return;
* ``reversal_body`` ranks by minus the candle body ratio;
* ``reversal_composite`` averages minus-return, minus-body, minus five-day momentum and minus
  the five-day close/MA gap.

Picking the single strongest performer from the earlier probe would be selection on a spent
window, so all three are reported side by side and none is declared the winner.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import performance
from src.backtest.run import run_policy, run_strategy
from src.data.preprocess import ROOT, file_hash, read_config, write_json
from src.envs.broker import BrokerConfig
from src.envs.market import MarketDataProvider
from tools import price_feature_ic

REPORT_ROOT = ROOT / "reports" / "runs" / "reversal_backtest"
CONSTRUCTIONS = {
    "reversal_return": ("return",),
    "reversal_body": ("body_ratio",),
    "reversal_composite": ("return", "body_ratio", "momentum_5", "close_ma_ratio_5"),
}
TOP_N = (20, 30)
COST_MODELS = ("broker", "frictionless")
SLIPS = (0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002)
WORKING_CAPITAL = 3_000_000.0


def build_market(window, directory=None, codes_limit=None):
    """A provider over the same panel the probe read, sliced to one spent window."""
    start, end = price_feature_ic.WINDOWS[window]
    directory = price_feature_ic.DATASET if directory is None else Path(directory)
    manifest, panel, prices, codes = price_feature_ic.load_dataset(directory)
    features = list(manifest["feature_cols"])
    codes = list(codes if codes_limit is None else codes[:codes_limit])
    panel = panel.loc[:pd.Timestamp(end), pd.IndexSlice[codes, features]]
    prices = prices.loc[:pd.Timestamp(end), pd.IndexSlice[codes, :]]
    market = MarketDataProvider(panel, features, price_path=prices,
                                price_basis=manifest["price_basis"],
                                start_date=start, end_date=end, lookback=1)
    return market, features


def settings_for(research, *, costs=True):
    """Broker settings; the frictionless arm zeroes every charge, nothing else changes."""
    settings = deepcopy(research["backtest"])
    settings.update(seed=42, lookback=1, initial_capital=WORKING_CAPITAL, rebalance_every=1)
    if not costs:
        settings["broker"] = dict(settings["broker"], commission=0.0, min_commission=0.0,
                                  sell_tax=0.0, slippage=0.0)
    return settings


class ReversalStrategy:
    """Rank eligible names so the biggest loser comes first, then equal weight the top N.

    ``buffer_n`` is the turnover control and nothing else: a name already held stays in the
    book while it ranks inside the wider buffer, and only the vacancies are refilled from the
    top.  Without a buffer the book is rebuilt to exactly the top N every day, which trades
    every rank swap and pays a round trip for it.  ``top_n`` and ``buffer_n`` are the only
    knobs, and every combination is reported rather than the best one.
    """

    def __init__(self, feature_cols, basket, top_n, max_weight, buffer_n=None):
        self.indices = [feature_cols.index(name) for name in basket]
        if not self.indices:
            raise ValueError("Empty reversal basket")
        self.top_n = int(top_n)
        self.max_weight = float(max_weight)
        self.buffer_n = None if buffer_n is None else int(max(buffer_n, top_n))
        self.contract = {}

    def scores(self, observation):
        """Mean cross-sectional rank of the basket, reversed so a loser scores high."""
        eligible = (observation["valid_mask"].astype(bool)
                    & observation["buy_mask"].astype(bool))
        if int(eligible.sum()) < 2:
            return None, eligible
        parts = []
        for index in self.indices:
            values = observation["stock_features"][:, -1, index].astype(float)
            usable = eligible & np.isfinite(values)
            ranks = np.full(values.shape, np.nan)
            if int(usable.sum()) >= 2:
                order = pd.Series(values[usable]).rank(method="average").to_numpy(dtype=float)
                ranks[usable] = order / (usable.sum() + 1.0)
            parts.append(ranks)
        stacked = np.vstack(parts)
        usable = np.isfinite(stacked)
        count = usable.sum(axis=0)
        mean = np.divide(np.where(usable, stacked, 0.0).sum(axis=0), count,
                         out=np.full(count.shape, np.nan, dtype=float), where=count > 0)
        return -mean, eligible

    def act(self, observation, step=None):
        scores, eligible = self.scores(observation)
        weights = np.zeros(len(eligible))
        if scores is None:
            return weights
        usable = eligible & np.isfinite(scores)
        if int(usable.sum()) < 1:
            return weights
        order = np.argsort(-scores, kind="stable")
        ranked = [index for index in order if usable[index]]
        chosen = ranked[:self.top_n]
        if self.buffer_n is not None:
            held = observation["portfolio"][:-1].astype(float) > 0
            keep = [index for index in ranked[:self.buffer_n] if held[index]][:self.top_n]
            room = self.top_n - len(keep)
            keep_set = set(keep)
            additions = [index for index in ranked if index not in keep_set][:max(room, 0)]
            chosen = keep + additions
        if chosen:
            weights[chosen] = min(1.0 / len(chosen), self.max_weight)
        return weights


def describe(daily, reference_daily):
    """Net, gross and the cost bill, on the same close-to-close grid as the reference."""
    metrics = performance(daily)
    reference = performance(reference_daily)
    capital = daily.equity.iloc[0]
    paid = (metrics["total_fees"] + metrics["total_slippage_cost"]) / capital
    paired = pd.concat([daily.set_index("date").equity.pct_change().rename("policy"),
                        reference_daily.set_index("date").equity.pct_change().rename("reference")],
                       axis=1).dropna()
    active = (paired.policy - paired.reference).to_numpy(dtype=float)
    tracking = float(active.std(ddof=1)) if len(active) > 1 else 0.0
    return dict(cumulative_return=metrics["cumulative_return"],
                annual_return=metrics["annual_return"], sharpe=metrics["sharpe"],
                max_drawdown=metrics["max_drawdown"],
                mean_daily_turnover=metrics["mean_daily_turnover"],
                annual_turnover=metrics["mean_daily_turnover"] * 252,
                total_fees=metrics["total_fees"],
                total_slippage=metrics["total_slippage_cost"],
                cost_ratio=float(paid),
                gross_return=metrics["cumulative_return"] + float(paid),
                reference_return=reference["cumulative_return"],
                excess_return=metrics["cumulative_return"] - reference["cumulative_return"],
                gross_excess_return=metrics["cumulative_return"] + float(paid) - reference["cumulative_return"],
                tracking_error=tracking * np.sqrt(252),
                information_ratio=(float(active.mean() / tracking * np.sqrt(252))
                                   if tracking > 1e-15 else None),
                holding_count=float(daily.holding_count.iloc[1:].mean()))


def sweep(window, directory=None, top_n=(20, 30), slips=SLIPS, keep=("reversal_body",)):
    """Net return as a function of the one cost nobody has measured.

    Every other charge is a published rate; slippage is a guess.  Sweeping it turns "this
    hinges on execution cost" from an assertion into a curve, and keeps the equity paths at the
    registered rate and at zero so the gap is visible rather than tabulated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    market, features = build_market(window, directory)
    research = read_config(ROOT / "configs" / "research.json")
    broker = BrokerConfig(**research["backtest"]["broker"])
    reference_daily, _, _, _ = run_strategy(market, "equal_weight",
                                            settings_for(research, costs=True))
    reference_return = performance(reference_daily)["cumulative_return"]
    rows, curves = [], {}
    for name, basket in CONSTRUCTIONS.items():
        for band in top_n:
            strategy_settings = settings_for(research, costs=True)
            for slip in slips:
                settings = deepcopy(strategy_settings)
                settings["broker"] = dict(settings["broker"], slippage=float(slip))
                strategy = ReversalStrategy(features, basket, band, broker.max_weight)
                daily, _, orders, _ = run_policy(market, strategy, settings)
                metrics = performance(daily)
                key = f"{name}_n{band}_s{slip:.4f}"
                rows.append(dict(construction=name, top_n=int(band), slippage=float(slip),
                                 cumulative_return=metrics["cumulative_return"],
                                 excess_return=metrics["cumulative_return"] - reference_return,
                                 mean_daily_turnover=metrics["mean_daily_turnover"],
                                 annual_turnover=metrics["mean_daily_turnover"] * 252,
                                 max_drawdown=metrics["max_drawdown"],
                                 sharpe=metrics["sharpe"],
                                 cost_ratio=(metrics["total_fees"] + metrics["total_slippage_cost"])
                                 / WORKING_CAPITAL))
                if name in keep and band == 30 and slip in (0.0, broker.slippage):
                    curves[f"{name}_s{slip:.4f}"] = daily[["date", "equity"]].copy()

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for band, group in pd.DataFrame(rows).groupby("top_n"):
        for name, part in group.groupby("construction"):
            axes[0].plot(part.slippage * 100, part.cumulative_return * 100,
                         marker="o", markersize=4, label=f"{name} n={band}")
    axes[0].axhline(reference_return * 100, color="black", linestyle="--", linewidth=1,
                    label="equal weight")
    axes[0].axhline(0.0, color="grey", linewidth=0.8)
    axes[0].axvline(broker.slippage * 100, color="red", linestyle=":", linewidth=1)
    axes[0].set_xlabel("slippage per side (%)")
    axes[0].set_ylabel("net cumulative return (%)")
    axes[0].set_title("Net return against slippage")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    for key, frame in curves.items():
        axes[1].plot(pd.to_datetime(frame.date), frame.equity / frame.equity.iloc[0],
                     label=key, linewidth=1.2)
    axes[1].plot(pd.to_datetime(reference_daily.date),
                 reference_daily.equity / reference_daily.equity.iloc[0],
                 color="black", linestyle="--", linewidth=1, label="equal weight")
    axes[1].set_title("Equity: slip 0 (blue) vs registered 0.10% (orange), both pay commission+stamp")
    axes[1].set_ylabel("growth of 1")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(REPORT_ROOT / "slippage_sweep.png", dpi=120)
    plt.close(figure)
    return dict(window=window, start=price_feature_ic.WINDOWS[window][0],
                end=price_feature_ic.WINDOWS[window][1], capital=WORKING_CAPITAL,
                registered_slippage=broker.slippage, reference_return=reference_return,
                rows=rows,
                curves={key: {str(stamp.date()): float(value)
                              for stamp, value in frame.set_index("date").equity.items()}
                        for key, frame in curves.items()},
                warning="V 已花掉：本图说明成本敏感性，不授权部署。")


def evaluate(window, directory=None, top_n=TOP_N, buffers=(None, "double")):
    market, features = build_market(window, directory)
    research = read_config(ROOT / "configs" / "research.json")
    broker = BrokerConfig(**research["backtest"]["broker"])
    reference_settings = settings_for(research, costs=True)
    reference_daily, _, _, _ = run_strategy(market, "equal_weight", reference_settings)
    results = {}
    for cost_model in COST_MODELS:
        settings = settings_for(research, costs=cost_model == "broker")
        for name, basket in CONSTRUCTIONS.items():
            for band in top_n:
                if band * broker.max_weight < 1 - 1e-9:
                    raise ValueError(f"top_n={band} cannot reach full investment under the cap")
                for buffer in buffers:
                    width = None if buffer is None else (band * 2 if buffer == "double" else int(buffer))
                    strategy = ReversalStrategy(features, basket, band, broker.max_weight, width)
                    daily, _, orders, assumptions = run_policy(market, strategy, settings)
                    key = f"{name}_n{band}_b{width or 0}_{cost_model}"
                    results[key] = dict(
                        construction=name, basket=list(basket), top_n=int(band),
                        buffer_n=width, cost_model=cost_model,
                        **describe(daily, reference_daily))
                    results[key]["orders"] = int(len(orders))
    return dict(window=window, start=price_feature_ic.WINDOWS[window][0],
                end=price_feature_ic.WINDOWS[window][1],
                codes=int(market.n_stocks), dates=int(market.n_dates),
                capital=WORKING_CAPITAL, max_weight=broker.max_weight,
                reference=dict(name="equal_weight_daily",
                               **{key: performance(reference_daily)[key] for key in
                                  ("cumulative_return", "annual_return", "sharpe",
                                   "max_drawdown", "mean_daily_turnover", "total_fees",
                                   "total_slippage_cost")}),
                strategies=results,
                warning="V 已花掉：本表说明成本能吃多少，不授权部署。",
                assumptions=assumptions)


def _fmt(value, spec="+.2%"):
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return "—"


def _markdown(payload):
    lines = [f"# 1 日反转的成本感知回测 — {payload['window']}"
             f"（{payload['start']} ~ {payload['end']}）", "",
             f"> {payload['warning']}", "",
             f"- {payload['codes']} 只，{payload['dates']} 个交易日，本金 {payload['capital']:,.0f}",
             f"- 参考 `equal_weight_daily`：累计 {payload['reference']['cumulative_return']:.2%}，"
             f"日均换手 {payload['reference']['mean_daily_turnover']:.2%}", "",
             "| 构造 | 只数 | 缓冲 | 成本 | 净收益 | 净超额 | 毛超额 | 年化换手 | 成本占比 | Sharpe | 回撤 |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, row in sorted(payload["strategies"].items()):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {:.0f}× | {:.2%} | {} | {} |".format(
            row["construction"], row["top_n"], row["buffer_n"] or "—", row["cost_model"],
            _fmt(row["cumulative_return"]), _fmt(row["excess_return"]),
            _fmt(row["gross_excess_return"]),
            row["annual_turnover"], row["cost_ratio"],
            _fmt(row["sharpe"], "+.2f"), _fmt(row["max_drawdown"])))
    lines += ["", "成本占比 = （佣金 + 印花税 + 滑点）/ 期初本金；毛超额 = 净超额 + 成本占比。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=sorted(price_feature_ic.WINDOWS), default="V")
    parser.add_argument("--dataset", default=str(price_feature_ic.DATASET))
    parser.add_argument("--top-n", type=int, nargs="+", default=list(TOP_N))
    parser.add_argument("--sweep", action="store_true",
                        help="sweep slippage instead of running the cost grid")
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if arguments.sweep:
        payload = sweep(arguments.window, Path(arguments.dataset), tuple(arguments.top_n))
        payload["inputs"] = {"reversal_backtest.py": file_hash(Path(__file__))}
        if arguments.output:
            output = Path(arguments.output)
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "slippage_sweep.json", payload)
        for row in sorted(payload["rows"], key=lambda item: (item["construction"], item["top_n"], item["slippage"])):
            print("{:<20s} n={:<3d} slip={:.2%} net={:+.2%} turnover={:.0f}x".format(
                row["construction"], row["top_n"], row["slippage"],
                row["cumulative_return"], row["annual_turnover"]))
        return 0
    payload = evaluate(arguments.window, Path(arguments.dataset), tuple(arguments.top_n))
    payload["inputs"] = {"reversal_backtest.py": file_hash(Path(__file__)),
                         "price_feature_ic.py": file_hash(ROOT / "tools" / "price_feature_ic.py")}
    if arguments.output:
        output = Path(arguments.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "reversal_backtest.json", payload)
        (output / "reversal_backtest.md").write_text(_markdown(payload), encoding="utf-8")
    for key, row in sorted(payload["strategies"].items()):
        print("{:34s} net={:+.2%} ref={:+.2%} excess={:+.2%} gross={:+.2%} "
              "turnover={:.0f}x cost={:.2%}".format(
                  key, row["cumulative_return"], row["reference_return"], row["excess_return"],
                  row["gross_excess_return"], row["annual_turnover"], row["cost_ratio"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
