"""Holding-period rank IC for the pre-registered alpha families, plus a machinery self-test.

This tool answers one question per signal: measured over a holding period of ``h`` rows, does
the daily cross-sectional rank IC clear the threshold that the exposure arithmetic in
``doc/alpha_program_20261106.md`` section 2.3 says is economically required?  The threshold is
not a taste; it is ``alpha * sqrt(h/243) / (sigma_cs * E[z | top share])`` evaluated at the
70% constant-weight 000300 ladder from ``tools/alpha_budget.py``.

Three things are worth stating before any number is read:

1. **Overlapping labels.** With a daily decision grid and an ``h``-row holding period, adjacent
   daily ICs share ``h-1`` rows of return, so the IC series is autocorrelated by construction.
   The registered lag-5 Newey-West correction cannot repair that at ``h = 20``, so the
   significance test is run at lag ``max(5, h)`` and the milder lags are reported beside it.
   A signal that clears only the lag-5 bar has not been shown to clear anything.
2. **Fundamental inputs change once a quarter.** The effective number of independent
   cross-sections is far below the number of decision dates.  The tool reports the implied
   independent count ``(ic_std / nw_se)**2`` rather than pretending the days are independent.
3. **The stored window decides what may be reported.** ``WINDOWS`` holds only samples that
   have already been spent.  ``UNLICENSED`` holds the reserve and protected samples, and the
   guard here refuses to touch them until the protocol is executed deliberately.

``--inject`` exists so the machinery can be proven correct on a real panel before any real
factor is judged: a factor set equal to the realised forward return must score IC 1.0, a noise
factor must score about zero, and a factor built from a *past* return window must collapse.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.data.preprocess import file_hash, write_json
from src.training.exploration_signal import forward_labels
from tools import alpha_budget
from tools import alpha_factors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY = PROJECT_ROOT / "data" / "online" / "history" / "h_13cd6e0169124b2e"
FUNDAMENTALS = PROJECT_ROOT / "data" / "online" / "fundamentals" / "f_2f8ef1efe22c430a"
BUDGET = PROJECT_ROOT / "reports" / "runs" / "alpha_budget"
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "alpha_ic"
PROTOCOL = PROJECT_ROOT / "doc" / "alpha_program_20261106.md"

MIN_NAMES = 30
UNIVERSE = 300
BOOK = 30
EXPOSURE = 0.7
HORIZONS = (20, 60)
HIT_MIN_IC = 0.0

WINDOWS = {"V": ("2019-01-01", "2024-12-31")}
# Neither of these may be read until the protocol is executed as a deliberate, single step.
UNLICENSED = {"R": ("2025-01-01", "2026-06-30"), "P": ("2026-07-01", "2026-10-30")}
# The reserve's decision-day count, taken from the ledger in the protocol rather than counted
# from the panel, because counting it would mean reading rows this tool is not allowed to read.
RESERVE_DAYS = 359
INJECTIONS = ("perfect", "noise", "placebo")


def _budget_block(path):
    """Required annual alpha at the registered exposure, and the dispersion it divides by."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["policies"]["rebalanced"]["rows"]
    match = [row for row in rows if abs(row["exposure"] - EXPOSURE) < 1e-9]
    if not match or match[0].get("required_alpha") is None:
        raise ValueError("Budget report has no required alpha at the registered exposure")
    return dict(required_alpha=float(match[0]["required_alpha"]),
                sigma_cs=float(payload["transmission"]["sigma_cs"]),
                start=payload["start"], end=payload["end"])


def required_ic(block, horizon, share=BOOK / UNIVERSE):
    return alpha_budget.required_ic_at_horizon(block["required_alpha"], share,
                                               block["sigma_cs"], horizon)


def rank_ic(values, target, dates, *, min_names=MIN_NAMES):
    """Spearman correlation per row over names that have both a value and a label."""
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    out = np.full(len(values), np.nan)
    for row in range(len(values)):
        usable = np.isfinite(values[row]) & np.isfinite(target[row])
        if usable.sum() < min_names:
            continue
        left = pd.Series(values[row][usable]).rank().to_numpy()
        right = pd.Series(target[row][usable]).rank().to_numpy()
        left = left - left.mean()
        right = right - right.mean()
        scale = math.sqrt(float((left ** 2).sum() * (right ** 2).sum()))
        if scale > 1e-12:
            out[row] = float(left @ right / scale)
    return pd.Series(out, index=dates)


def nw_se(values, lag):
    """Newey-West standard error of the mean with a Bartlett kernel of ``lag`` lags."""
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 4:
        return float("nan")
    lag = int(min(max(int(lag), 0), n - 2))
    centred = series - series.mean()
    variance = float(centred @ centred) / n
    for step in range(1, lag + 1):
        covariance = float(centred[:-step] @ centred[step:]) / n
        variance += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    return math.sqrt(max(variance, 0.0) / n)


def _acf(values, lag):
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) <= lag + 2:
        return None
    centred = series - series.mean()
    scale = float(centred @ centred)
    if scale <= 0:
        return None
    return float(centred[:-lag] @ centred[lag:] / scale)


def ic_stats(series, horizon, threshold):
    """Mean IC with three Newey-West horizons plus the power to see ``threshold``."""
    clean = pd.Series(series).dropna()
    dates = len(clean)
    if dates < 4:
        return dict(dates=dates, mean_ic=None, ic_std=None, threshold=threshold)
    values = clean.to_numpy(dtype=float)
    mean = float(values.mean())
    spread = float(values.std(ddof=1))
    recorded = dict(dates=dates, mean_ic=mean, ic_std=spread,
                    naive_se=spread / math.sqrt(dates), threshold=threshold,
                    hit_rate=float((values > HIT_MIN_IC).mean()),
                    acf_1=_acf(values, 1), acf_5=_acf(values, 5), acf_h=_acf(values, horizon))
    errors = {"lag5": nw_se(values, 5), "lag_h": nw_se(values, horizon),
              "lag_2h": nw_se(values, 2 * horizon)}
    for label, error in errors.items():
        recorded[label] = dict(se=error, t=(mean / error if error and error > 0 else None))
    decisive = errors["lag_h"]
    recorded["t_nw_h"] = recorded["lag_h"]["t"]
    recorded["effective_dates"] = (float((spread / decisive) ** 2)
                                   if decisive and decisive > 0 else None)
    recorded["threshold_over_se"] = (float(threshold / decisive)
                                     if decisive and decisive > 0 else None)
    return recorded


def verdict(stats):
    """Criterion 1 (magnitude), 2 (significance at lag h) and 3 (registered direction)."""
    if stats.get("mean_ic") is None or stats.get("t_nw_h") is None:
        return dict(c1_magnitude=False, c2_significance=False, c3_direction=False, pass_=False)
    c1 = stats["mean_ic"] >= stats["threshold"] > 0
    c2 = stats["t_nw_h"] >= 2.0
    c3 = stats["mean_ic"] > 0
    return dict(c1_magnitude=bool(c1), c2_significance=bool(c2), c3_direction=bool(c3),
                pass_=bool(c1 and c2 and c3))


def family_false_positive(families=len(alpha_factors.FAMILIES), signals=len(alpha_factors.SIGNALS),
                          alpha=0.05):
    """Family-wise false-positive rate under a complete null, for the record.

    The registration quotes 23% for five families at a 5% per-family level.  The criterion
    is actually ``t >= 2`` one-sided on each signal, which is 2.28% per test over seven
    signals; both numbers are reported so the difference is visible rather than assumed away.
    """
    one_sided = NormalDist().cdf(-2.0)
    return dict(registered=float(1.0 - (1.0 - alpha) ** families),
                implied_by_t2=float(1.0 - (1.0 - one_sided) ** signals),
                families=families, signals=signals, per_test_alpha=alpha)


def guard(start, end):
    """Refuse any window that would consume the reserve or the protected sample."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for name, (low, high) in UNLICENSED.items():
        low, high = pd.Timestamp(low), pd.Timestamp(high)
        if start <= high and end >= low:
            raise ValueError(f"Window intersects the {name} sample; the protocol forbids it")
    if start > end:
        raise ValueError("Window start is after its end")


def resolve_window(name):
    if name not in WINDOWS:
        known = ", ".join(sorted(set(WINDOWS) | set(UNLICENSED)))
        raise ValueError(f"Unknown window {name!r}; known: {known}")
    return WINDOWS[name]


def _inject(name, labels, horizon, shape, seed=20261106):
    """Synthetic factors that must come out at a known IC if the pipeline is honest."""
    if name == "perfect":
        return np.asarray(labels, dtype=float)
    if name == "noise":
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape)
    if name == "placebo":
        # The label of a return window that has already closed: nothing forward-looking.
        return np.asarray(labels.shift(horizon), dtype=float)
    raise ValueError(f"Unknown injection {name!r}")


def control_match(name, stats):
    """A control is judged against its known answer, not against the alpha criteria."""
    if name == "perfect":
        ok = stats.get("mean_ic") is not None and abs(stats["mean_ic"] - 1.0) <= 1e-6
        return dict(expected=1.0, tolerance=1e-6, criterion="abs(IC − 1) ≤ 1e-6", matches=bool(ok))
    error = (stats.get("lag_h") or {}).get("se")
    if not error:
        return dict(expected=0.0, tolerance=None, criterion="没有标准误", matches=False)
    ok = abs(stats["mean_ic"]) <= 4.0 * error
    return dict(expected=0.0, tolerance=4.0 * error,
                criterion="abs(IC) ≤ 4 SE(NW h)", matches=bool(ok))


def power_floor(horizons_block, reserve_days):
    """How small an IC the remaining samples could actually resolve.

    The measured IC dispersion and the measured overlap factor are extrapolated to a sample of
    ``reserve_days`` decision dates.  If the economically required IC sits far inside one
    standard error, no amount of care in the test will make the test informative -- that has to
    be said before the sample is spent, not after it comes back insignificant.
    """
    rows = {}
    for key, block in horizons_block.items():
        horizon = int(key[1:])
        spreads, ratios = [], []
        for name, row in block["signals"].items():
            if row.get("family") == "control" or row.get("mean_ic") is None:
                continue
            if row.get("ic_std") and row.get("effective_dates"):
                spreads.append(row["ic_std"])
                ratios.append(row["effective_dates"] / block["rows"])
        if not spreads:
            continue
        spread = float(np.median(spreads))
        ratio = float(np.median(ratios))
        effective = ratio * reserve_days
        error = spread / math.sqrt(effective)
        threshold = block["required_ic"]["primary"]
        rows[key] = dict(
            horizon=horizon, window_rows=block["rows"], median_ic_sd=spread,
            independent_share=ratio, overlap_factor=1.0 / ratio,
            reserve_rows=reserve_days, reserve_effective_rows=effective,
            reserve_se=error, detectable_2sigma=2.0 * error, required_ic=threshold,
            required_over_se=threshold / error,
            rows_needed_for_t2=float((2.0 * spread / threshold) ** 2 / ratio),
            years_needed_for_t2=float((2.0 * spread / threshold) ** 2 / ratio / 243.0))
    return rows


def evaluate(window, horizons=HORIZONS, *, injections=()):
    """Every registered signal plus optional injected controls, on one spent window."""
    start, end = resolve_window(window)
    guard(start, end)
    primary = _budget_block(BUDGET / "budget_000300_2005_2026.json")
    modern = _budget_block(BUDGET / "budget_000300_2017_2026.json")

    codes = sorted(path.stem for path in (HISTORY / "hfq").glob("*.parquet"))
    # Truncating the panel at the window end makes the reserve sample unreadable rather than
    # merely unused: no row after ``end`` ever enters memory, so no statistic can depend on it.
    index = alpha_factors.union_calendar(HISTORY / "hfq", end=end)
    fields = alpha_factors.load_fields(HISTORY / "hfq", index, codes)
    unadjusted = alpha_factors.load_fields(HISTORY / "unadjusted", index, codes, ("close",))

    from tools.signal_expansion import latest_by_code, share_events
    financials = pd.read_parquet(FUNDAMENTALS / "financials.parquet")
    events = alpha_factors.financial_events(financials, index)
    signals, returns = alpha_factors.build_signals(index, codes, fields, events)
    shares = latest_by_code(index, codes, share_events(FUNDAMENTALS / "shareholders.parquet", index))
    with np.errstate(divide="ignore", invalid="ignore"):
        market_cap = np.where((unadjusted["close"] > 0) & (shares.to_numpy() > 0),
                              unadjusted["close"] * shares.to_numpy(), np.nan)
    log_cap = pd.DataFrame(np.log(market_cap), index=index, columns=codes)
    first_seen = alpha_factors.listing_dates(HISTORY / "hfq", codes)
    age = alpha_factors.listing_age(index, codes, first_seen)

    prices = alpha_factors.to_price_frame(fields, codes)
    inside = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    results = {}
    for horizon in horizons:
        labels, endpoint = forward_labels(prices, codes, horizon)
        settled = endpoint.fillna(pd.Timestamp.max).to_numpy() <= np.datetime64(pd.Timestamp(end))
        mask = inside & settled
        dates = index[mask]
        target = labels.to_numpy(dtype=float)[mask]
        block = dict(rows=int(mask.sum()), first=str(dates[0].date()) if len(dates) else None,
                     last=str(dates[-1].date()) if len(dates) else None,
                     required_ic=dict(
                         primary=required_ic(primary, horizon),
                         primary_alpha=primary["required_alpha"],
                         modern=required_ic(modern, horizon),
                         modern_alpha=modern["required_alpha"],
                         book=BOOK, exposure=EXPOSURE))
        table = {}
        for name, matrix in signals.items():
            series = rank_ic(matrix.to_numpy(dtype=float)[mask], target, dates)
            stats = ic_stats(series, horizon, block["required_ic"]["primary"])
            stats["family"] = next(key for key, value in alpha_factors.FAMILIES.items()
                                   if name in value["signals"])
            stats["construction"] = alpha_factors.CONSTRUCTION[name]
            stats["verdict"] = verdict(stats)
            stats["versus_modern"] = verdict(dict(stats, threshold=block["required_ic"]["modern"]))
            stats["mean_rank_corr_log_cap"] = _mean_rank_corr(
                matrix.to_numpy(dtype=float)[mask], log_cap.to_numpy(dtype=float)[mask], dates)
            stats["mean_rank_corr_listing_age"] = _mean_rank_corr(
                matrix.to_numpy(dtype=float)[mask], age.to_numpy(dtype=float)[mask], dates)
            table[name] = stats
        for injection in injections:
            matrix = _inject(injection, labels, horizon, labels.shape)
            series = rank_ic(matrix[mask], target, dates)
            stats = ic_stats(series, horizon, block["required_ic"]["primary"])
            stats["family"] = "control"
            stats["construction"] = f"注入：{injection}"
            stats["control"] = control_match(injection, stats)
            table[f"control_{injection}"] = stats
        block["signals"] = table
        results[f"h{horizon}"] = block

    reserve_days = RESERVE_DAYS
    mix = alpha_factors.period_mix(index, codes, events)
    sampled = mix.loc[index[inside]]
    step = max(1, len(sampled) // 12)
    return dict(window=window, start=start, end=end, codes=len(codes), rows=len(index),
                signals=list(alpha_factors.SIGNALS),
                families={key: dict(value) for key, value in alpha_factors.FAMILIES.items()},
                coverage=alpha_factors.coverage_report(signals),
                period_mix={str(date.date()): {key: int(value) for key, value in row.items()}
                            for date, row in sampled.iloc[::step].iterrows()},
                budget=dict(primary=primary, modern=modern),
                false_positive=family_false_positive(),
                power=power_floor(results, reserve_days),
                warning="此窗口已花掉；本表只用于验证度量机件，不构成任何证据。",
                horizons=results)


def _mean_rank_corr(values, other, dates):
    """Average cross-sectional rank correlation; a diagnostic, not a criterion."""
    series = rank_ic(values, other, dates)
    clean = series.dropna()
    return float(clean.mean()) if len(clean) else None


def _fmt(value, spec="+.4f"):
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return "—"


def _markdown(payload):
    lines = [f"# Alpha 族持有期 IC — 窗口 {payload['window']} "
             f"（{payload['start']} ~ {payload['end']}）", "",
             f"> {payload['warning']}", "",
             f"- 代码 {payload['codes']} 只，面板 {payload['rows']} 行",
             f"- 经济门槛：{payload['budget']['primary']['required_alpha']:+.2%}/年，"
             f"σ_cs {payload['budget']['primary']['sigma_cs']:.4f}，"
             f"基准样本 {payload['budget']['primary']['start']}~{payload['budget']['primary']['end']}"
             f"（现代窗口对照门槛 {payload['budget']['modern']['required_alpha']:+.2%}/年）",
             f"- 全零假设下的族错误率：注册口径 {payload['false_positive']['registered']:.1%}"
             f"（{payload['false_positive']['families']} 族 / α=5%），"
             f"按 t≥2 实际口径 {payload['false_positive']['implied_by_t2']:.1%}"
             f"（{payload['false_positive']['signals']} 个信号）", ""]
    lines += ["## 覆盖率", "", "| 信号 | 非空占比 | 有值日 | 中位只数 |", "| --- | ---: | ---: | ---: |"]
    for name, row in payload["coverage"].items():
        lines.append(f"| {name} | {row['share']:.1%} | {row['dates']} | {row['median_names']:.0f} |")
    lines.append("")
    lines += _control_section(payload)
    lines += _mix_section(payload)
    lines += _power_section(payload)
    for key, block in payload["horizons"].items():
        lines += [f"## {key}（{block['rows']} 个已结算决策日，"
                  f"{block['first']} ~ {block['last']}）", "",
                  f"动态门槛 IC ≥ **{block['required_ic']['primary']:.4f}**"
                  f"（对照现代窗口 {block['required_ic']['modern']:.4f}）", "",
                  "| 信号 | 族 | 均值 IC | 阈值 | t(NW h) | t(NW 5) | 有效日 | IC>0 | 判据 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
        for name, row in block["signals"].items():
            if row.get("family") == "control":
                continue
            if row.get("mean_ic") is None:
                lines.append(f"| {name} | {row.get('family', '—')} | — | — | — | — | — | — | — |")
                continue
            mark = "通过" if row["verdict"]["pass_"] else "未过"
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {:.1%} | {} |".format(
                name, row.get("family", "—"), _fmt(row["mean_ic"]),
                _fmt(row["threshold"], ".4f"), _fmt(row.get("t_nw_h"), "+.2f"),
                _fmt(row.get("lag5", {}).get("t"), "+.2f"),
                _fmt(row.get("effective_dates"), ".0f"), row["hit_rate"], mark))
        lines.append("")
        lines += ["| 信号 | IC 波动 | 一阶自相关 | h 阶自相关 | 与 log 市值秩相关 | 与上市年长秩相关 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, row in block["signals"].items():
            if row.get("family") == "control" or row.get("mean_ic") is None:
                continue
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                name, _fmt(row.get("ic_std"), ".4f"), _fmt(row.get("acf_1"), "+.3f"),
                _fmt(row.get("acf_h"), "+.3f"), _fmt(row.get("mean_rank_corr_log_cap")),
                _fmt(row.get("mean_rank_corr_listing_age"))))
        lines.append("")
    return "\n".join(lines)


def _control_section(payload):
    """The self-check must be read before any real factor number."""
    rows = [(key, name, row) for key, block in payload["horizons"].items()
            for name, row in block["signals"].items() if row.get("family") == "control"]
    if not rows:
        return []
    lines = ["## 机件自检（注入已知答案的因子）", "",
             "这些不是信号，是校准。若度量有前视或对齐错误，前两行会立刻暴露。", "",
             "| 持有期 | 对照 | 期望 IC | 实测 IC | 判据 | 结论 |", "| --- | --- | ---: | ---: | --- | :---: |"]
    for key, name, row in rows:
        control = row.get("control", {})
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            key, name.replace("control_", ""), _fmt(control.get("expected"), "+.1f"),
            _fmt(row.get("mean_ic")), control.get("criterion", "—"),
            "符合" if control.get("matches") else "**不符合**"))
    lines.append("")
    return lines


def _mix_section(payload):
    """Show which report period the cross-section is actually carrying."""
    mix = payload.get("period_mix") or {}
    if not mix:
        return []
    lines = ["## 报告期构成（每只股票当前用的是哪一期）", "",
             "基本面信号按最近一期已披露报告构造；年报与一季报同一天（4/30）可用，"
             "每日期望接近单一季度。", "",
             "| 日期 | 一季报 | 半年报 | 三季报 | 年报 | 无 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for date, row in mix.items():
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            date, row.get("03", 0), row.get("06", 0), row.get("09", 0), row.get("12", 0),
            row.get("none", 0)))
    lines.append("")
    return lines


def _power_section(payload):
    """Say out loud whether the reserved sample could resolve the required IC at all."""
    power = payload.get("power") or {}
    if not power:
        return []
    lines = ["## 功效下限（用已测得的离散度外推到 R）", "",
             "| 持有期 | IC 波动 | 重叠系数 | R 有效日 | R 的 SE | 2σ 可测 IC | 所需 IC | 所需 IC / SE | 达到 t=2 需要的年数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, row in power.items():
        lines.append("| {} | {} | {:.1f}× | {:.1f} | {} | {} | {} | {:.2f} | {:.0f} |".format(
            key, _fmt(row["median_ic_sd"], ".3f"), row["overlap_factor"],
            row["reserve_effective_rows"], _fmt(row["reserve_se"], ".4f"),
            _fmt(row["detectable_2sigma"], ".4f"), _fmt(row["required_ic"], ".4f"),
            row["required_over_se"], row["years_needed_for_t2"]))
    lines.append("")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="V", help="spent sample to measure on")
    parser.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--inject", nargs="*", default=[], choices=INJECTIONS,
                        help="machinery self-test: synthetic factors with a known IC")
    parser.add_argument("--output", help="directory for alpha_ic.json and alpha_ic.md")
    arguments = parser.parse_args(argv)
    payload = evaluate(arguments.window, tuple(arguments.horizon),
                       injections=tuple(arguments.inject))
    payload["inputs"] = {
        "alpha_ic.py": file_hash(Path(__file__)),
        "alpha_factors.py": file_hash(PROJECT_ROOT / "tools" / "alpha_factors.py"),
        "protocol": file_hash(PROTOCOL) if PROTOCOL.exists() else None,
        "label_definition": "close[t+h] / open[t+1] - 1, complete positive path",
        "nw_lags": ["5", "horizon", "2*horizon"],
    }
    if arguments.output:
        output = Path(arguments.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "alpha_ic.json", payload)
        (output / "alpha_ic.md").write_text(_markdown(payload), encoding="utf-8")
    for key, block in payload["horizons"].items():
        for name, row in block["signals"].items():
            if row.get("mean_ic") is None:
                print(f"{key} {name}: no settled dates")
                continue
            if row.get("family") == "control":
                control = row.get("control", {})
                print("{} {:20s} ic={:+.4f} expected={:+.1f} {}".format(
                    key, name, row["mean_ic"], control.get("expected") or 0.0,
                    "OK" if control.get("matches") else "MISMATCH"))
                continue
            print("{} {:20s} ic={:+.4f} thr={:.4f} t(nw)={:+.2f} t(lag5)={:+.2f} "
                  "eff_n={:6.1f} {}".format(
                      key, name, row["mean_ic"], row["threshold"],
                      row.get("t_nw_h") or float("nan"),
                      (row.get("lag5") or {}).get("t") or float("nan"),
                      row.get("effective_dates") or float("nan"),
                      "PASS" if row["verdict"]["pass_"] else "fail"))
    for key, row in (payload.get("power") or {}).items():
        print("{} power: required_ic={:.4f} se={:.4f} required_over_se={:.2f} "
              "years_for_t2={:.0f}".format(
                  key, row["required_ic"], row["reserve_se"], row["required_over_se"],
                  row["years_needed_for_t2"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
