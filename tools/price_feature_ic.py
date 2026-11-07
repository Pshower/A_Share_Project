"""Measure the twenty stored price features under the corrected holding-period IC framework.

The claim on the table is that all twenty features sit at ``|IC| <= 0.035``.  That number was
produced with a naive standard error and a lag-5 Newey-West correction, both of which
understate uncertainty when the labels overlap.  This tool re-measures every feature with the
correction that was later shown to matter -- Newey-West at lag ``max(5, h)`` -- and reports the
implied number of independent decision dates beside it.

Three things follow from how the sample is chosen:

* The clean dataset ends **2024-12-31**, so this tool cannot read the reserve or protected
  samples even by accident; there is no truncation to forget.
* ``V`` has already been spent on the exposure work.  These numbers therefore cannot license a
  decision to deploy anything.  They *can* falsify a previously published statistic, which is
  the only question being asked here.
* The panel is standardised with pooled training statistics.  That is a single affine map per
  feature, so within-date ordering -- and therefore every rank IC below -- is unchanged.

No direction is pre-registered for these twenty features, so this report is descriptive:
magnitude against the economic threshold and two-sided significance.  Calling a sign correct
after the fact would be a registration the protocol never made.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import file_hash, read_config, write_json
from src.training.exploration_signal import forward_labels
from tools import alpha_ic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "clean" / "research_v3"
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "price_feature_ic"
HORIZONS = (20, 60)
MIN_NAMES = 30
# The full panel starts in 2017; "V" is the spent validation segment used everywhere else.
WINDOWS = {"V": ("2019-01-01", "2024-12-31"), "full": ("2017-01-02", "2024-12-31")}


def load_dataset(directory=DATASET):
    """Panel, prices and manifest for the dataset the PPO actor actually reads."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    codes = [code for code in manifest["stock_codes"] if code in set(panel.columns.get_level_values(0))]
    if not codes or not set(codes) <= set(prices.columns.get_level_values(0)):
        raise ValueError("Panel and price stock sets do not overlap")
    expected = set(pd.MultiIndex.from_product([codes, manifest["feature_cols"]]))
    if not expected <= set(panel.columns):
        raise ValueError("Panel is missing a registered feature column")
    return manifest, panel, prices, codes


def feature_matrices(panel, codes, features):
    """One ``dates x codes`` float matrix per feature, aligned to the panel index."""
    matrices = {}
    for name in features:
        block = panel.xs(name, axis=1, level=1)
        matrices[name] = block.reindex(index=panel.index, columns=codes).to_numpy(dtype=float)
    return matrices


def assess(stats, threshold):
    """Descriptive: magnitude against the economic bar, and two-sided significance."""
    if stats.get("mean_ic") is None or stats.get("t_nw_h") is None:
        return dict(magnitude=False, significant=False)
    return dict(magnitude=bool(abs(stats["mean_ic"]) >= threshold),
                significant=bool(abs(stats["t_nw_h"]) >= 2.0))


def evaluate(window, horizons=HORIZONS, directory=DATASET):
    start, end = WINDOWS[window]
    alpha_ic.guard(start, end)
    manifest, panel, prices, codes = load_dataset(directory)
    features = list(manifest["feature_cols"])
    matrices = feature_matrices(panel, codes, features)
    primary = alpha_ic._budget_block(alpha_ic.BUDGET / "budget_000300_2005_2026.json")
    index = panel.index
    inside = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    results = {}
    for horizon in horizons:
        labels, endpoint = forward_labels(prices, codes, horizon)
        labels = labels.reindex(index)
        settled = endpoint.reindex(index).fillna(pd.Timestamp.max).to_numpy() <= np.datetime64(pd.Timestamp(end))
        mask = inside & settled
        dates = index[mask]
        target = labels.to_numpy(dtype=float)[mask]
        threshold = alpha_ic.required_ic(primary, horizon)
        table = {}
        for name in features:
            series = alpha_ic.rank_ic(matrices[name][mask], target, dates)
            stats = alpha_ic.ic_stats(series, horizon, threshold)
            stats["assessment"] = assess(stats, threshold)
            stats["mean_rank_corr_log_cap"] = None
            table[name] = stats
        ranked = sorted(table.items(), key=lambda item: -abs(item[1].get("mean_ic") or 0.0))
        results[f"h{horizon}"] = dict(
            rows=int(mask.sum()), first=str(dates[0].date()) if len(dates) else None,
            last=str(dates[-1].date()) if len(dates) else None,
            required_ic=threshold, required_alpha=primary["required_alpha"],
            signals=table,
            ranking=[name for name, _ in ranked],
            strongest=ranked[0][0] if ranked else None,
            strongest_abs_ic=abs(ranked[0][1]["mean_ic"]) if ranked and ranked[0][1].get("mean_ic") is not None else None,
            magnitude_count=sum(row.get("assessment", {}).get("magnitude", False) for row in table.values()),
            significant_count=sum(row.get("assessment", {}).get("significant", False) for row in table.values()),
            # The registered-but-wrong test, kept only to show what it would have claimed.
            lag5_significant_count=sum(
                abs((row.get("lag5") or {}).get("t") or 0.0) >= 2.0 for row in table.values()))
    return dict(window=window, start=start, end=end, dataset=str(directory.relative_to(PROJECT_ROOT)),
                version=manifest["version"], codes=len(codes), features=features,
                rows=len(index), label_definition="close[t+h] / open[t+1] - 1, complete positive path",
                warning=("V 已花掉：本表不能授权任何部署决定，只用于检验旧统计量是否站得住。"
                         if window == "V" else
                         "该窗口含设计段；同样只用于检验旧统计量。"),
                primary_alpha=primary["required_alpha"], horizons=results)


def _fmt(value, spec="+.4f"):
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return "—"


def _markdown(payload):
    lines = [f"# 20 个价格特征在新框架下的截面 IC — {payload['window']}"
             f"（{payload['start']} ~ {payload['end']}）", "",
             f"> {payload['warning']}", "",
             f"- 数据集 `{payload['dataset']}`（{payload['version']}），{payload['codes']} 只，"
             f"面板 {payload['rows']} 行",
             f"- 标签：`{payload['label_definition']}`",
             f"- 经济门槛：{payload['primary_alpha']:+.2%}/年，30 只组合、70% 敞口", ""]
    for key, block in payload["horizons"].items():
        lines += [f"## {key}（{block['rows']} 个已结算决策日，{block['first']} ~ {block['last']}）", "",
                  f"经济门槛 IC ≥ **{block['required_ic']:.4f}**；"
                  f"量级达标 {block['magnitude_count']}/{len(payload['features'])} 个，"
                  f"在 lag-h 下 |t| ≥ 2 的 **{block['significant_count']}/{len(payload['features'])}** 个"
                  f"（若仍用 lag-5 会是 {block['lag5_significant_count']} 个）。", "",
                  "| 特征 | 均值 IC | \\|t\\|(NW h) | t(NW 5) | 有效日 | 达标量级 | 显著 |",
                  "| --- | ---: | ---: | ---: | ---: | :---: | :---: |"]
        for name in block["ranking"]:
            row = block["signals"][name]
            if row.get("mean_ic") is None:
                lines.append(f"| {name} | — | — | — | — | — | — |")
                continue
            mark = row["assessment"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                name, _fmt(row["mean_ic"]), _fmt(abs(row["t_nw_h"] or 0.0), ".2f"),
                _fmt(abs((row.get("lag5") or {}).get("t") or 0.0), ".2f"),
                _fmt(row.get("effective_dates"), ".0f"),
                "是" if mark["magnitude"] else "否",
                "**是**" if mark["significant"] else "否"))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=sorted(WINDOWS), default="V")
    parser.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    payload = evaluate(arguments.window, tuple(arguments.horizon), Path(arguments.dataset))
    payload["inputs"] = {
        "price_feature_ic.py": file_hash(Path(__file__)),
        "alpha_ic.py": file_hash(PROJECT_ROOT / "tools" / "alpha_ic.py"),
        "nw_lags": ["5", "horizon", "2*horizon"],
        "direction_registered": False,
    }
    if arguments.output:
        output = Path(arguments.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "price_feature_ic.json", payload)
        (output / "price_feature_ic.md").write_text(_markdown(payload), encoding="utf-8")
    for key, block in payload["horizons"].items():
        print(f"{key}: rows={block['rows']} threshold={block['required_ic']:.4f} "
              f"magnitude={block['magnitude_count']}/{len(payload['features'])} "
              f"significant={block['significant_count']}/{len(payload['features'])} "
              f"lag5_would_claim={block['lag5_significant_count']}")
        for name in block["ranking"][:6]:
            row = block["signals"][name]
            if row.get("mean_ic") is None:
                continue
            print("   {:18s} ic={:+.4f} |t(nw h)|={:.2f} t(lag5)={:+.2f} eff_n={:.0f}".format(
                name, row["mean_ic"], abs(row["t_nw_h"] or 0.0),
                (row.get("lag5") or {}).get("t") or 0.0, row.get("effective_dates") or 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
