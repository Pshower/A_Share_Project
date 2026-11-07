"""Cross-sectional rank IC for every stored feature on a frozen fold dataset.

The question this answers: is ``momentum_20`` the strongest cross-sectional signal in the
feature set, or only the one this repository happens to trade? Nothing is fitted here, so a
window that ends at ``train_end`` can rank candidates without seeing the validation period.

For every decision date the tool correlates each feature with a tradable forward return
(buy the next open, exit ``horizon`` rows later at the close, skipping any window with a
missing quote), using a Spearman rank correlation over the names that have both a finite
feature row and a finite label. Per-date correlations are then aggregated into mean IC, IC
stability and hit rate; a high mean with a low stability is luck, not information.

Caveats that belong with any answer this produces: one IC number covers one market regime,
rank correlation says nothing about the tails a top-k portfolio actually holds, and picking
the winner of a 20-candidate sweep is a selection, not a validated strategy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import read_config, write_json

MIN_NAMES = 30


def tradable_forward(prices, codes, horizon):
    """Buy the next open, exit ``horizon`` rows later at the close; never jump a gap."""
    from src.training.exploration_signal import forward_labels
    return forward_labels(prices, codes, horizon)


def feature_matrices(panel, codes, features):
    """One ``dates x codes`` float matrix per feature, aligned to ``codes``."""
    return {name: panel.xs(name, axis=1, level=1).reindex(columns=codes).to_numpy(dtype=float)
            for name in features}


def daily_ic(panel, labels, *, min_names=MIN_NAMES):
    """Spearman IC per decision date: returns a frame of dates by feature."""
    codes = list(panel.columns.get_level_values(0).unique())
    features = list(panel.columns.get_level_values(1).unique())
    matrices = feature_matrices(panel, codes, features)
    targets = labels.reindex(index=panel.index, columns=codes).to_numpy(dtype=float)
    rows, kept = [], []
    for position, date in enumerate(panel.index):
        values = np.stack([matrices[name][position] for name in features])
        target = targets[position]
        usable = np.isfinite(values).all(axis=0) & np.isfinite(target)
        if usable.sum() < min_names:
            continue
        ranked = pd.DataFrame(values[:, usable].T).rank().to_numpy(dtype=float).T
        ranked = ranked - ranked.mean(axis=1, keepdims=True)
        response = pd.Series(target[usable]).rank().to_numpy(dtype=float)
        response = response - response.mean()
        scale = np.sqrt((ranked ** 2).sum(axis=1) * (response ** 2).sum())
        rows.append(np.divide(ranked @ response, scale, out=np.full(len(features), np.nan),
                              where=scale > 1e-12))
        kept.append(date)
    if not rows:
        return pd.DataFrame(columns=features, dtype=float)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(kept, name="date"), columns=features)


def summarize(daily):
    """Mean IC, IC standard deviation, stability (mean/std) and positive-rate per feature."""
    if daily.empty:
        return pd.DataFrame()
    frame = pd.DataFrame({
        "dates": daily.notna().sum(),
        "mean_ic": daily.mean(),
        "ic_std": daily.std(ddof=1),
        "hit_rate": (daily > 0).sum() / daily.notna().sum(),
    })
    frame["stability"] = frame["mean_ic"] / frame["ic_std"]
    return frame.sort_values("mean_ic", ascending=False)


def windows(manifest):
    """Train and validation windows from the stored split boundaries."""
    train = (pd.Timestamp(manifest["start_date"]), pd.Timestamp(manifest["train_end"]))
    val = (pd.Timestamp(manifest["train_end"]) + pd.Timedelta(days=1), pd.Timestamp(manifest["val_end"]))
    return {"train": train, "val": val}


def evaluate(directory, horizon):
    """Per-window IC tables for one fold dataset."""
    directory = Path(directory)
    manifest = read_config(directory / "manifest.json")
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    codes = list(panel.columns.get_level_values(0).unique())
    labels, endpoint = tradable_forward(prices, codes, horizon)
    labels, endpoint = labels.reindex(panel.index), endpoint.reindex(panel.index)
    results = {}
    for name, (start, end) in windows(manifest).items():
        # A decision only counts when its label has finished inside the window.
        eligible = (panel.index >= start) & (panel.index <= end)
        eligible &= endpoint.fillna(pd.Timestamp.max).to_numpy() <= end
        block = daily_ic(panel.loc[eligible], labels.loc[eligible])
        table = summarize(block).reset_index(names="feature")
        results[name] = dict(start=str(start.date()), end=str(end.date()),
                             table=json.loads(table.to_json(orient="records")))
    return dict(dataset=str(directory), horizon=horizon, version=manifest["version"], windows=results)


def _markdown(payload):
    lines = [f"# 单因子截面 IC（前 {payload['horizon']} 交易日，{payload['version']}）", ""]
    for name, block in payload["windows"].items():
        lines += [f"## {name} {block['start']} ~ {block['end']}", "",
                  "| 特征 | 均值 IC | IC 波动 | 稳定性 | 正比例 | 决策日 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in block["table"]:
            lines.append("| {} | {:+.4f} | {:.4f} | {:+.2f} | {:.1%} | {} |".format(
                row["feature"], row["mean_ic"], row["ic_std"], row["stability"],
                row["hit_rate"], row["dates"]))
        if not block["table"]:
            lines.append(f"| — | 窗口短于 {payload['horizon']} 交易日，没有可结算的标签 |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="fold dataset directory")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--output", required=True, help="directory for ic.json and ic.md")
    arguments = parser.parse_args(argv)
    payload = evaluate(arguments.dataset, arguments.horizon)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ic.json", payload)
    (output / "ic.md").write_text(_markdown(payload), encoding="utf-8")
    for name, block in payload["windows"].items():
        if not block["table"]:
            print(f"{name}: {block['start']}..{block['end']} no settled labels for horizon {payload['horizon']}")
            continue
        best = block["table"][0]
        print(f"{name}: {block['start']}..{block['end']} best={best['feature']} "
              f"ic={best['mean_ic']:+.4f} stability={best['stability']:+.2f} dates={best['dates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
