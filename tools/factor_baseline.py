"""Independent cross-sectional factor baseline through the shared broker.

Fits a fixed-alpha ridge model on training-period rows only, scores the current
observation, and holds the top-N eligible names at equal weight. Execution goes
through ``run_policy`` so fees, slippage, lot size, T+1 and accounting are
identical to every other baseline and to a learned policy.

This answers a question the repository has never posed: can a learned
cross-sectional factor model, traded with no learning, beat PPO? It is a
baseline, not a candidate model: it performs no PPO update, no gradient step and
no parameter search, and it never evaluates the protected period.

Validation only. Fitting never uses rows or labels after ``train_end``;
scoring and trading are produced only on the validation split.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.metrics import performance
from src.backtest.run import git_metadata, run_policy, run_strategy
from src.data.preprocess import ROOT, file_hash, read_config, resolve_path, write_json
from src.training.env_factory import ResearchData
from src.training.exploration_signal import (RIDGE_ALPHA, fit_ridge, forward_labels,
                                            predict_ridge, split_examples)

BASELINE_NAMES = ("cash", "buy_hold", "equal_weight", "momentum")
SCHEMA_VERSION = 1


class FactorBaseline:
    """Top-N equal weight on a fixed ridge score; no learning at decision time."""

    def __init__(self, model, feature_cols, codes, *, max_weight=0.05,
                 rebalance_every=20, top_k=20):
        if rebalance_every < 1 or top_k < 1:
            raise ValueError("rebalance_every and top_k must be positive")
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        expected = list(model["feature_cols"])
        if list(feature_cols) != expected:
            raise ValueError("Feature order differs from the fitted factor model")
        if len(codes) != len(set(codes)):
            raise ValueError("Stock codes must be unique")
        self.model = deepcopy(model)
        self.feature_cols = expected
        self.codes = list(codes)
        self.max_weight = float(max_weight)
        self.interval = int(rebalance_every)
        self.top_k = int(top_k)
        self.contract = {}
        self.invested = False
        self.decisions = []

    def act(self, observation, step):
        n = len(observation["valid_mask"])
        if n != len(self.codes):
            raise ValueError("Observation stock count differs from the fitted pool")
        if step % self.interval:
            return None
        features = np.asarray(observation["stock_features"], dtype=np.float64)
        if features.ndim != 3 or features.shape[0] != n or features.shape[2] != len(self.feature_cols):
            raise ValueError("Observation features do not match the factor contract")
        latest = features[:, -1, :]
        if not np.isfinite(latest).all():
            raise ValueError("Factor scoring requires finite features; masks must precede filling")
        eligible = (np.asarray(observation["valid_mask"], dtype=bool)
                    & np.asarray(observation["buy_mask"], dtype=bool))
        weights = np.zeros(n)
        scores = predict_ridge(self.model, latest)
        if not np.isfinite(scores).all():
            raise ValueError("Factor scores must be finite")
        chosen = np.zeros(n, dtype=bool)
        if eligible.any():
            indices = np.flatnonzero(eligible)
            # Descending score, ascending code as the recorded stable tie-break.
            order = np.lexsort((np.asarray(self.codes)[indices], -scores[indices]))
            chosen[indices[order[:self.top_k]]] = True
            weights[chosen] = min(1.0 / int(chosen.sum()), self.max_weight)
            self.invested = True
        self.decisions.append(dict(step=int(step), eligible=int(eligible.sum()),
                                   chosen=int(chosen.sum()), invested=bool(self.invested),
                                   score_min=float(scores[eligible].min()) if eligible.any() else None,
                                   score_max=float(scores[eligible].max()) if eligible.any() else None))
        return weights


def fit_factor(panel, prices, codes, feature_cols, train_end, *, horizon=20, alpha=RIDGE_ALPHA):
    """Past-only fit on training rows; labels ending after ``train_end`` are purged."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    codes, feature_cols = list(codes), list(feature_cols)
    if not codes or len(codes) != len(set(codes)):
        raise ValueError("codes must be a nonempty unique list")
    if not feature_cols or len(feature_cols) != len(set(feature_cols)):
        raise ValueError("feature_cols must be a nonempty unique list")
    missing = [c for c in codes if c not in set(panel.columns.get_level_values(0))]
    if missing:
        raise ValueError("Panel lacks requested codes: " + ", ".join(missing[:5]))
    labels, endpoint = forward_labels(prices, codes, horizon)
    x, y, frame, audit = split_examples(panel, labels, endpoint, codes, feature_cols,
                                        None, pd.Timestamp(train_end))
    if len(x) < 2:
        raise ValueError("Too few complete training labels to fit a factor model")
    model = fit_ridge(x, y, alpha=alpha)
    model.update(feature_cols=feature_cols, codes=codes, horizon=int(horizon),
                 train_end=str(pd.Timestamp(train_end).date()),
                 label_definition="close[t+h] / open[t+1] - 1; complete path; endpoint <= train_end")
    model["fit"] = dict(rows=int(len(x)), features=len(feature_cols), horizon=int(horizon),
                        alpha=float(alpha), **audit)
    return model


def _paired(daily_a, daily_b, metrics_a, metrics_b):
    paired = pd.concat([daily_a.set_index("date").equity.pct_change().rename("a"),
                        daily_b.set_index("date").equity.pct_change().rename("b")], axis=1).dropna()
    active = paired.a - paired.b
    tracking = float(active.std(ddof=1)) if len(active) > 1 else 0.0
    return dict(matched_steps=int(len(paired)),
                return_gap=metrics_a["cumulative_return"] - metrics_b["cumulative_return"],
                daily_return_correlation=float(paired.a.corr(paired.b)) if len(paired) > 1
                and paired.a.std() > 1e-15 and paired.b.std() > 1e-15 else None,
                tracking_error=tracking * np.sqrt(252),
                information_ratio=float(active.mean() / tracking * np.sqrt(252)) if tracking > 1e-15 else None,
                drawdown_magnitude_gap=abs(metrics_a["max_drawdown"]) - abs(metrics_b["max_drawdown"]),
                cost_gap=(metrics_a["total_fees"] + metrics_a["total_slippage_cost"]
                          - metrics_b["total_fees"] - metrics_b["total_slippage_cost"]))


def evaluate(config, output, *, horizon=20, alpha=RIDGE_ALPHA, rebalance_every=20, top_k=20):
    data = ResearchData(config)
    settings = deepcopy(data.settings)
    settings["rebalance_every"] = rebalance_every
    settings["momentum_top_k"] = top_k
    train_end = pd.Timestamp(data.manifest["train_end"])
    model = fit_factor(data.features, data.prices, data.codes, data.manifest["feature_cols"],
                       train_end, horizon=horizon, alpha=alpha)
    broker = data.settings["broker"]
    strategy = FactorBaseline(model, data.manifest["feature_cols"], data.codes,
                              max_weight=broker["max_weight"], rebalance_every=rebalance_every,
                              top_k=top_k)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, metrics, assumptions = {}, {}, {}

    def record(name, daily, holdings, orders, assumption):
        folder = output / name
        folder.mkdir()
        daily.to_csv(folder / "daily.csv", index=False)
        holdings.to_parquet(folder / "positions.parquet")
        orders.to_csv(folder / "orders.csv", index=False)
        results[name], metrics[name], assumptions[name] = daily, performance(daily), assumption

    market = data.market("val")
    daily, holdings, orders, assumption = run_policy(market, strategy, settings)
    record("factor_ridge", daily, holdings, orders, assumption)
    for name in BASELINE_NAMES:
        daily, holdings, orders, assumption = run_strategy(data.market("val"), name, settings)
        record(name, daily, holdings, orders, assumption)
    comparisons = {name: _paired(results["factor_ridge"], results[name],
                                 metrics["factor_ridge"], metrics[name])
                   for name in BASELINE_NAMES}
    write_json(output / "metrics.json", metrics)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "factor_model.json", model)
    write_json(output / "decisions.json", strategy.decisions)
    pd.DataFrame(metrics).T.to_csv(output / "metrics.csv")
    write_json(output / "experiment.json", dict(
        schema_version=SCHEMA_VERSION, split="val", test_evaluated=False,
        contract=data.contract, config=data.config, research_config=data.research,
        evaluation_settings=settings, factor=dict(horizon=int(horizon), alpha=float(alpha),
        rebalance_every=int(rebalance_every), top_k=int(top_k)),
        assumptions=assumptions, git=git_metadata(),
        source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "src").rglob("*.py")},
        limitations=data.manifest["limitations"] + ["Factor baseline is fitted on training rows only; "
        "it performs no PPO update and is not a candidate model.",
        "Top-N equal weight reuses the shared broker; ranking ties break by ascending stock code."]))
    lines = ["# Cross-sectional factor baseline", "",
             f"Split: val. Horizon {horizon}, alpha {alpha}, rebalance every {rebalance_every}, top {top_k}.",
             f"Fit rows: {model['fit']['rows']}. Max decision date: {model['fit']['max_decision_date']}.",
             "No PPO update, no gradient step, no protected-period evaluation.", "",
             "| Strategy | Return | Sharpe | Max drawdown | Fees | Mean cash |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in metrics.items():
        lines.append(f"| {name} | {values['cumulative_return']:.4%} | {values['sharpe']} | "
                     f"{values['max_drawdown']:.4%} | {values['total_fees']:.2f} | {values['mean_cash_ratio']:.2%} |")
    lines += ["", "## Paired comparison against the factor baseline", "",
              "| Baseline | Return gap | Max-DD gap | Cost gap | Corr | Info ratio |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in comparisons.items():
        lines.append(f"| {name} | {value['return_gap']:+.4%} | {value['drawdown_magnitude_gap']:+.4%} | "
                     f"{value['cost_gap']:+.2f} | {value['daily_return_correlation']} | {value['information_ratio']} |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in data.manifest["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(output=str(output), metrics=metrics, comparisons=comparisons,
                fit=model["fit"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo.json",
                        help="PPO-style config; its research_config supplies dataset and backtest settings")
    parser.add_argument("--research-config", type=Path,
                        help="override research_config, e.g. a registered fold research.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.research_config is not None:
        config["research_config"] = str(resolve_path(args.research_config))
    research = read_config(resolve_path(config["research_config"]))
    if research["backtest"]["split"] != "val":
        parser.error("This command only runs validation; final test data stays untouched")
    root = resolve_path(args.output_root or Path("reports/runs/factor_baseline"))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("run-id must be one directory name")
    result = evaluate(config, root / run_id, horizon=args.horizon, alpha=args.alpha,
                      rebalance_every=args.rebalance_every, top_k=args.top_k)
    print(json.dumps(dict(output=result["output"], fit=result["fit"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
