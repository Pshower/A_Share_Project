"""Does a fold's result depend on the day the portfolio was opened?

A 59-day validation window has 59 possible entry dates, and the registered result uses exactly
one of them. If the return collapses when the same rule starts a few days later, the window is
a story about one session, not about a rule.

For every shift ``k`` the window becomes the suffix ``val_dates[k:]``: identical data, identical
lot rules, identical 20-day rebalance cadence, only a later start. The same suffix is also
replayed by ``draws`` random 20-name portfolios, so the shift curve can be read as selection
(excess over the null) rather than as exposure to a shorter window.

Nothing here is a new experiment on unseen data: every shift stays inside the registered
validation window.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import performance
from src.backtest.run import run_policy, run_strategy
from src.data.preprocess import read_config, write_json
from src.envs.broker import BrokerConfig
from src.training.env_factory import ResearchData
from tools.daily_advice import RandomTopK


class ShiftedMarket:
    """Start the trading loop at ``offset`` instead of the first row of the window.

    ``TradingEnv.reset`` calls ``market.reset()`` and then reads ``current_index``, so shifting
    there keeps the whole provider - lookback history included - exactly as registered. Only
    the opening date moves.
    """

    def __init__(self, market, offset):
        if offset < 0 or offset >= market.n_dates - 1:
            raise ValueError("Shift leaves no room to trade")
        self._market, self._offset = market, offset

    def __getattr__(self, name):
        return getattr(self._market, name)

    def reset(self):
        self._market.reset()
        self._market.current_index = self._offset

    @property
    def window(self):
        return self._market.dates[self._offset:]


def evaluate_fold(config, fold, *, shifts=20, draws=100, top_k=20):
    """Momentum and a random-20 null over every suffix of the fold's validation window."""
    folds_root = Path(config["research_config"]).parent.parent
    fold_config = Path(folds_root) / str(fold) / "research.json"
    if not fold_config.exists():
        raise ValueError(f"Missing fold config {fold_config}")
    local = deepcopy(config)
    local["research_config"] = str(fold_config)
    data = ResearchData(local)
    settings = deepcopy(data.settings)
    settings["rebalance_every"] = 20
    settings["momentum_top_k"] = top_k
    market = data.market("val")
    control_weight = BrokerConfig(**settings["broker"]).max_weight
    rows = []
    for shift in range(min(shifts, market.n_dates - 1)):
        view = ShiftedMarket(market, shift)
        daily, _, _, _ = run_strategy(view, "momentum", settings)
        momentum = performance(daily)["cumulative_return"]
        reference, _, _, _ = run_strategy(view, "equal_weight", settings)
        values = []
        for seed in range(draws):
            sample, _, _, _ = run_policy(
                view, RandomTopK(seed, top_k=top_k, max_weight=control_weight, interval=20), settings)
            values.append(performance(sample)["cumulative_return"])
        median = float(np.median(values))
        rows.append(dict(
            shift=shift, start=str(view.window[0].date()), end=str(view.window[-1].date()),
            steps=len(view.window) - 1, momentum=float(momentum),
            equal_weight=float(performance(reference)["cumulative_return"]),
            random_median=median,
            random_p05=float(np.percentile(values, 5)),
            random_p95=float(np.percentile(values, 95)),
            excess=float(momentum) - median,
            percentile=float(np.mean(np.asarray(values) < momentum)),
        ))
    return dict(fold=str(fold), version=data.manifest["version"], draws=int(draws),
                top_k=int(top_k), rows=rows)


def _markdown(payload):
    lines = [f"# 调仓日敏感性 {payload['fold']} 折（{payload['version']}）", "",
             f"每行把建仓日往后推若干个交易日，窗口是验证期的后缀；随机对照每行 {payload['draws']} 抽。", "",
             "| 后移 | 起始日 | 交易日 | 动量 | 随机中位 | 超额 | 分位 |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["rows"]:
        lines.append("| {} | {} | {} | {:+.4%} | {:+.4%} | {:+.4%} | {:.0%} |".format(
            row["shift"], row["start"], row["steps"], row["momentum"],
            row["random_median"], row["excess"], row["percentile"]))
    first = payload["rows"][0]
    lines += ["", f"后移 0 天：动量 {first['momentum']:+.4%}，随机中位 {first['random_median']:+.4%}。"]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo.json")
    parser.add_argument("--research-config", required=True,
                        help="any fold research.json; the fold directory is taken from its parent")
    parser.add_argument("--folds", default="2023,2024")
    parser.add_argument("--shifts", type=int, default=20)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    config = read_config(arguments.config)
    config["research_config"] = arguments.research_config
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    payloads = []
    for fold in [item.strip() for item in arguments.folds.split(",") if item.strip()]:
        payload = evaluate_fold(config, fold, shifts=arguments.shifts,
                                draws=arguments.draws, top_k=arguments.top_k)
        payloads.append(payload)
        (output / f"fold_{fold}.md").write_text(_markdown(payload), encoding="utf-8")
        rows = payload["rows"]
        print(f"{fold}: shift0 {rows[0]['momentum']:+.4f} | shift1 {rows[1]['momentum']:+.4f} | "
              f"shift3 {rows[3]['momentum']:+.4f} | excess0 {rows[0]['excess']:+.4f} | "
              f"excess median {np.median([row['excess'] for row in rows]):+.4f}")
    write_json(output / "sensitivity.json", dict(folds=payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
