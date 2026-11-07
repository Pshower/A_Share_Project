"""Is there any autoregressive structure in the panel at all?

The cross-sectional question ("does feature X rank names correctly") and the time-series
question ("does a name's own past predict its future") are different, and they have wildly
different power.  A 20-day cross-sectional IC throws away overlap: adjacent decision dates
share 19/20 of the return window, so 1436 days amount to about 100 independent observations.
A one-day autocorrelation has no overlap at all, so the same window is worth roughly 1400.
This tool asks the cheap, well-powered question first.

Three measurements, all on the clean research panel (which ends 2024-12-31 and therefore
cannot read the reserve or protected sample even by accident):

* pooled autocorrelations ``rho_k`` of daily close-to-close returns, run through a moving-block
  bootstrap over dates so that within-day cross-sectional correlation is respected;
* variance ratios ``VR(q) = 1 + 2 * sum (1 - k/q) rho_k`` -- below one means reversal, above
  one means continuation;
* short-horizon cross-sectional ICs at h = 1, 2, 5, 20 for the twenty stored features.

No direction is pre-registered and ``V`` has already been spent, so nothing here licenses a
decision.  What it can do is close a direction: if there is no autocorrelation at one day, no
autoregressive trading model has anything to fit, no matter how large the network.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import file_hash, write_json
from src.training.exploration_signal import forward_labels
from tools import alpha_ic
from tools import price_feature_ic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "autocorrelation"
HORIZONS = (1, 2, 5, 20)
AUTOCORR_LAGS = 59
VR_ORDER = (5, 20, 60)
BOOTSTRAP = 400
BLOCK = 20
VOL_WINDOW = 60
CLIP = 5.0
MIN_EXECUTABLE_HORIZON = 2
SEED = 20261106


def returns_matrix(prices, codes, index):
    """Daily close-to-close returns on the panel's own grid, per name."""
    close = prices.xs("close", axis=1, level=1).reindex(index=index, columns=list(codes))
    close = close.where(close > 0)
    return close / close.shift(1) - 1.0


def volatility_standardise(returns, window=VOL_WINDOW, clip=CLIP):
    """Divide by a trailing volatility that stops one day before the return it scales.

    Without this the pooled autocorrelation is dominated by whatever names happened to be
    volatile, which is a property of the sample rather than of the return process.  The clip
    is not decoration: a name that sat still for months has a near-zero trailing scale, and one
    such observation drags a pooled ratio towards one.  A scaled return is meant to be roughly
    unit variance, so a bounded tail is the correct domain for the estimator.
    """
    scale = returns.rolling(window, min_periods=max(10, window // 4)).std(ddof=0).shift(1)
    scaled = returns / scale.where(scale > 0)
    return scaled.replace([np.inf, -np.inf], np.nan).clip(-clip, clip)


def demean(values):
    """Subtract each name's own mean; daily means are tiny but they bias rho downward."""
    frame = pd.DataFrame(values)
    return frame.sub(frame.mean(axis=0), axis=1).to_numpy(dtype=float)


def per_date_products(values, lags=AUTOCORR_LAGS):
    """Sufficient statistics that a bootstrap can aggregate cheaply.

    ``cross[k][t] = sum_i r[t,i] * r[t-k,i]`` and ``square[t] = sum_i r[t,i]^2``, each with an
    accompanying count so a date with partial coverage is not double counted.
    """
    array = np.asarray(values, dtype=float)
    rows = array.shape[0]
    cross = np.zeros((rows, lags + 1))
    counts = np.zeros((rows, lags + 1))
    for lag in range(1, lags + 1):
        left = array[lag:]
        right = array[:-lag]
        usable = np.isfinite(left) & np.isfinite(right)
        cross[lag:, lag] = np.where(usable, left * right, 0.0).sum(axis=1)
        counts[lag:, lag] = usable.sum(axis=1)
    square = np.where(np.isfinite(array), array ** 2, 0.0).sum(axis=1)
    observed = np.isfinite(array).sum(axis=1)
    return dict(cross=cross, counts=counts, square=square, observed=observed)


def pooled_autocorrelations(stats, rows):
    """rho_k for every lag the cached statistics actually hold."""
    available = stats["cross"].shape[1] - 1
    total_square = stats["square"][:rows].sum()
    total_observed = stats["observed"][:rows].sum()
    rhos = np.zeros(available + 1)
    if total_square <= 0 or total_observed <= 0:
        return rhos
    variance = total_square / total_observed
    for lag in range(1, available + 1):
        numerator = stats["cross"][:rows, lag].sum()
        pairs = stats["counts"][:rows, lag].sum()
        if pairs > 0:
            rhos[lag] = (numerator / pairs) / variance
    return rhos


def variance_ratio(rhos, order):
    """VR(q) = 1 + 2 * sum_{k<q} (1 - k/q) rho_k; below one is reversal."""
    total = 1.0
    for lag in range(1, min(order, len(rhos) - 1) + 1):
        total += 2.0 * (1.0 - lag / order) * rhos[lag]
    return float(total)


def block_bootstrap(stats, rows, *, replications=BOOTSTRAP, block=BLOCK, seed=SEED):
    """Moving-block resampling of dates, so a resample keeps its own cross-section."""
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(rows / block))
    width = stats["cross"].shape[1]
    samples = np.empty((replications, width))
    ratios = np.empty((replications, len(VR_ORDER)))
    for index in range(replications):
        starts = rng.integers(0, max(1, rows - block + 1), blocks)
        picked = (starts[:, None] + np.arange(block)[None, :]).ravel()[:rows]
        picked = picked[picked < rows]
        subset = {key: value[picked] for key, value in stats.items()}
        rhos = pooled_autocorrelations(subset, len(picked))
        samples[index] = rhos
        ratios[index] = [variance_ratio(rhos, order) for order in VR_ORDER]
    return samples, ratios


def summarise(estimate, samples, name, null=0.0):
    """Point estimate with a bootstrap standard error and a percentile interval.

    ``null`` matters: an autocorrelation is tested against zero but a variance ratio is tested
    against one, and reporting ``estimate / se`` for both would make every ratio look
    impossibly significant.
    """
    spread = float(np.std(samples, ddof=1))
    low, high = np.percentile(samples, [2.5, 97.5])
    return dict(name=name, estimate=float(estimate), bootstrap_se=spread, null=float(null),
                t=(float(estimate - null) / spread if spread > 0 else None),
                ci_low=float(low), ci_high=float(high))


def per_name_autocorrelation(values):
    """Each name's own lag-one coefficient; the dispersion is as informative as the mean."""
    array = np.asarray(values, dtype=float)
    left, right = array[:-1], array[1:]
    usable = np.isfinite(left) & np.isfinite(right)
    numerator = np.where(usable, left * right, 0.0).sum(axis=0)
    denominator = np.where(usable, left ** 2, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def open_to_open_labels(prices, codes, horizon=1):
    """Buy the next open, sell ``horizon`` opens later: the shortest T+1-legal round trip.

    The standard label exits at a close, so ``h=1`` means buying and selling inside the same
    session -- which A-share T+1 forbids.  Open-to-open keeps the same decision grid and skips
    the cash: a position opened at ``t+1`` is closed at ``t+1+h``, which needs ``h >= 1``.
    """
    dates = prices.index
    labels = pd.DataFrame(np.nan, index=dates, columns=list(codes), dtype=float)
    endpoint = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    span = int(horizon) + 1
    if len(dates) <= span:
        return labels, endpoint
    endpoint.iloc[:-span] = dates[span:].to_numpy()
    rows = len(dates) - span
    for code in codes:
        opening = prices[(code, "open")].to_numpy(dtype=float)
        valid = np.isfinite(opening) & (opening > 0)
        path = np.ones(rows, dtype=bool)
        for offset in range(1, span + 1):
            path &= valid[offset:offset + rows]
        values = np.full(rows, np.nan)
        values[path] = opening[span:][path] / opening[1:1 + rows][path] - 1
        labels.iloc[:-span, labels.columns.get_loc(code)] = values
    return labels, endpoint


def short_horizon_ic(panel, codes, prices, features, index, start, end, horizons):
    """Cross-sectional IC at horizons short enough that overlap stops mattering."""
    inside = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    results = {}
    plans = [("h%d" % horizon, forward_labels(prices, codes, horizon), horizon,
              ("买入次日开盘、**当日收盘卖出**：A 股 T+1 不允许，不可执行"
               if horizon < MIN_EXECUTABLE_HORIZON else "买入次日开盘，h 个交易日后收盘卖出"))
             for horizon in horizons]
    plans.append(("o2o_h1", open_to_open_labels(prices, codes, 1), 1,
                  "买入次日开盘、再下一次开盘卖出：**T+1 合法的最短往返**"))
    for key, (labels, endpoint), horizon, note in plans:
        labels = labels.reindex(index)
        settled = endpoint.reindex(index).fillna(pd.Timestamp.max).to_numpy() <= np.datetime64(pd.Timestamp(end))
        mask = inside & settled
        dates = index[mask]
        target = labels.to_numpy(dtype=float)[mask]
        table = {}
        for name in features:
            matrix = panel.xs(name, axis=1, level=1).reindex(index=index, columns=codes).to_numpy(dtype=float)
            series = alpha_ic.rank_ic(matrix[mask], target, dates)
            table[name] = alpha_ic.ic_stats(series, horizon, 0.0)
        results[key] = dict(
            rows=int(mask.sum()), first=str(dates[0].date()) if len(dates) else None,
            last=str(dates[-1].date()) if len(dates) else None, signals=table,
            executable=not key.startswith("h") or horizon >= MIN_EXECUTABLE_HORIZON,
            note=note)
    return results


def evaluate(window, horizons=HORIZONS, directory=None):
    start, end = price_feature_ic.WINDOWS[window]
    alpha_ic.guard(start, end)
    directory = price_feature_ic.DATASET if directory is None else Path(directory)
    manifest, panel, prices, codes = price_feature_ic.load_dataset(directory)
    index = panel.index
    inside = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    rows = int(inside.sum())
    keep = index[inside]
    raw = returns_matrix(prices, codes, keep)
    scaled = volatility_standardise(raw)

    blocks = {}
    for label, frame in (("raw", raw), ("volatility_scaled", scaled)):
        stats = per_date_products(demean(frame.to_numpy(dtype=float)))
        rhos = pooled_autocorrelations(stats, len(keep))
        samples, ratios = block_bootstrap(stats, len(keep))
        names = np.array([np.nan] * (AUTOCORR_LAGS + 1))
        per_name = per_name_autocorrelation(frame.to_numpy(dtype=float))
        blocks[label] = dict(
            lag_one=summarise(rhos[1], samples[:, 1], "rho_1"),
            lags=[summarise(rhos[lag], samples[:, lag], f"rho_{lag}") for lag in (1, 2, 3, 5, 10, 20)],
            variance_ratio={f"q{order}": summarise(variance_ratio(rhos, order), ratios[:, position],
                                                   f"vr_{order}", null=1.0)
                            for position, order in enumerate(VR_ORDER)},
            per_name=dict(median=float(np.nanmedian(per_name)),
                          share_negative=float(np.nanmean(per_name < 0)),
                          count=int(np.isfinite(per_name).sum())))

    return dict(window=window, start=start, end=end,
                dataset=str(directory.relative_to(PROJECT_ROOT)),
                version=manifest["version"], codes=len(codes), rows=rows,
                return_definition="close_t / close_{t-1} - 1",
                bootstrap=dict(replications=BOOTSTRAP, block=BLOCK, seed=SEED),
                autocorrelation=blocks,
                feature_ic=short_horizon_ic(panel, codes, prices, list(manifest["feature_cols"]),
                                            index, start, end, horizons),
                warning=("V 已花掉：本表只回答\"有没有自回归结构\"，不授权任何部署决定。"
                         if window == "V" else "该窗口含设计段；同样只用于诊断。"))


def _fmt(value, spec="+.4f"):
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return "—"


def _markdown(payload):
    lines = [f"# 自回归结构诊断 — {payload['window']}（{payload['start']} ~ {payload['end']}）", "",
             f"> {payload['warning']}", "",
             f"- 数据集 `{payload['dataset']}`，{payload['codes']} 只，{payload['rows']} 个交易日",
             f"- 收益定义 `{payload['return_definition']}`（**不是**前面 IC 用的次日开盘买入）",
             f"- 自助法：移动区块 {payload['bootstrap']['block']} 日 × {payload['bootstrap']['replications']} 次，"
             f"种子 {payload['bootstrap']['seed']}", ""]
    for label, block in payload["autocorrelation"].items():
        title = "原始日收益" if label == "raw" else "波动率标准化后"
        lines += [f"## {title}", "",
                  "| 统计量 | 估计 | 自助 SE | t | 95% 区间 |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in block["lags"]:
            lines.append("| {} | {} | {} | {} | [{} , {}] |".format(
                row["name"], _fmt(row["estimate"]), _fmt(row["bootstrap_se"], ".4f"),
                _fmt(row["t"], "+.2f"), _fmt(row["ci_low"]), _fmt(row["ci_high"])))
        for key, row in block["variance_ratio"].items():
            lines.append("| {}（<1 反转） | {} | {} | {} | [{} , {}] |".format(
                key, _fmt(row["estimate"]), _fmt(row["bootstrap_se"], ".4f"),
                _fmt(row["t"], "+.2f"), _fmt(row["ci_low"]), _fmt(row["ci_high"])))
        lines += ["", f"- 逐名滞后一阶系数中位 {_fmt(block['per_name']['median'])}，"
                      f"负值占比 {block['per_name']['share_negative']:.1%}"
                      f"（{block['per_name']['count']} 只）", ""]
    lines += ["## 短持有期截面 IC", "",
              "方差比的 t 是相对 **1** 检验；自相关的 t 是相对 0。", ""]
    for key, block in payload["feature_ic"].items():
        ranked = sorted(block["signals"].items(), key=lambda item: -abs(item[1].get("mean_ic") or 0.0))
        top = ranked[0]
        mark = "" if block.get("executable", True) else " ⚠️ **不可执行**"
        lines.append(f"- **{key}**（{block['rows']} 日）{mark}：`{block.get('note', '')}`；"
                     f"最强 `{top[0]}` IC {_fmt(top[1].get('mean_ic'))}，"
                     f"\\|t\\|(NW h) {_fmt(abs(top[1].get('t_nw_h') or 0), '.2f')}，"
                     f"有效日 {_fmt(top[1].get('effective_dates'), '.0f')}；"
                     f"\\|t\\| ≥ 2 的个数 "
                     f"{sum(abs(r.get('t_nw_h') or 0) >= 2 for _, r in ranked)}/20")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=sorted(price_feature_ic.WINDOWS), default="V")
    parser.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--dataset", default=str(price_feature_ic.DATASET))
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    payload = evaluate(arguments.window, tuple(arguments.horizon), Path(arguments.dataset))
    payload["inputs"] = {"autocorrelation_probe.py": file_hash(Path(__file__)),
                         "price_feature_ic.py": file_hash(PROJECT_ROOT / "tools" / "price_feature_ic.py")}
    if arguments.output:
        output = Path(arguments.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "autocorrelation.json", payload)
        (output / "autocorrelation.md").write_text(_markdown(payload), encoding="utf-8")
    for label, block in payload["autocorrelation"].items():
        print(f"== {label}")
        for row in block["lags"]:
            print("   {:8s} {:+.4f}  se={:.4f} t={:+.2f}".format(
                row["name"], row["estimate"], row["bootstrap_se"], row["t"] or 0.0))
        for key, row in block["variance_ratio"].items():
            print("   {:8s} {:+.4f}  se={:.4f} t={:+.2f}".format(
                key, row["estimate"], row["bootstrap_se"], row["t"] or 0.0))
        print("   per-name AR1 median {:.4f}, negative share {:.1%}".format(
            block["per_name"]["median"], block["per_name"]["share_negative"]))
    for key, block in payload["feature_ic"].items():
        ranked = sorted(block["signals"].items(), key=lambda item: -abs(item[1].get("mean_ic") or 0.0))
        top = ranked[0]
        print("{} rows={} top={} ic={:+.4f} |t|={:.2f} significant={}/20".format(
            key, block["rows"], top[0], top[1]["mean_ic"], abs(top[1]["t_nw_h"] or 0.0),
            sum(abs(r.get("t_nw_h") or 0) >= 2 for _, r in ranked)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
