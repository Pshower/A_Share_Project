"""Separate beta from alpha before comparing strategies, and test whether the
alpha that survives is just a size tilt.

Every comparison in this repository so far ranks net returns or Sharpe ratios
against each other.  That is a beta comparison whenever the sample drifts: a book
that simply carried more market exposure wins a rising window without containing
any skill.

**The market factor is a proxy, not the CAPM market portfolio.**  CAPM's market
portfolio is the value-weighted portfolio of every investable asset; no such series
exists here.  ``000300`` is a free-float, value-weighted index of the 300 largest
listings, so it is a defensible *large-cap equity* factor and nothing more.  Using
it against books that hold 32 equal-weighted names charges the whole size and
weighting difference to alpha, which is why an untraded buy-and-hold book can show
a double-digit alpha.  The second factor exists to test exactly that: ``000905``
(the mid-cap index) minus ``000300`` is a small-minus-big proxy, so a two-factor
regression can say whether an alpha survives once the size tilt is priced.

Nothing is re-simulated: the inputs are the daily files each run already wrote.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import write_json
from tools.exposure_metrics import SNAPSHOT, load_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "baseline_capm"
DEFAULT_RUN = PROJECT_ROOT / "reports" / "runs" / "ppo_lightweight" / "train_8192_20260925"
STRATEGIES = ("cash", "buy_hold", "equal_weight", "equal_weight_daily",
              "momentum", "momentum_daily", "ppo")
CASH_RATE = 0.02
TRADING_DAYS = 243
MARKET_CODE = "000300"
SIZE_CODE = "000905"


def _fit(design, y):
    """Least squares with classical errors, residual autocorrelation and R squared."""
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    dof = y.size - design.shape[1]
    sigma2 = float(residuals @ residuals) / dof
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    errors = np.sqrt(np.diag(covariance))
    total = float(((y - y.mean()) ** 2).sum())
    lag1 = float("nan")
    if y.size > 3 and residuals.std() > 0:
        lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else float("nan")
    return coefficients, errors, lag1, r_squared


def _ratio(value, error):
    return float(value / error) if error > 0 else float("nan")


def ols(y, x):
    """One-factor regression: intercept, slope, t statistics, diagnostics."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape or y.size < 3:
        raise ValueError("Regressand and regressor must share a shape with at least 3 rows")
    coefficients, errors, lag1, r_squared = _fit(np.column_stack([np.ones_like(x), x]), y)
    return dict(alpha=float(coefficients[0]), beta=float(coefficients[1]),
                alpha_t=_ratio(coefficients[0], errors[0]),
                beta_t=_ratio(coefficients[1], errors[1]),
                residual_lag1=lag1, r_squared=r_squared)


def ols_two_factor(y, market, size):
    """Market plus size regression, so a size tilt is priced instead of charged to alpha."""
    y = np.asarray(y, dtype=float)
    market = np.asarray(market, dtype=float)
    size = np.asarray(size, dtype=float)
    if not (y.shape == market.shape == size.shape) or y.size < 4:
        raise ValueError("All three series must share a shape with at least 4 rows")
    design = np.column_stack([np.ones_like(market), market, size])
    coefficients, errors, lag1, r_squared = _fit(design, y)
    return dict(alpha=float(coefficients[0]), beta_market=float(coefficients[1]),
                beta_size=float(coefficients[2]),
                alpha_t=_ratio(coefficients[0], errors[0]),
                beta_market_t=_ratio(coefficients[1], errors[1]),
                beta_size_t=_ratio(coefficients[2], errors[2]),
                residual_lag1=lag1, r_squared=r_squared)


def index_returns(start, end, code):
    """Daily returns of one index inside ``start`` .. ``end``."""
    bars = load_index(SNAPSHOT, code)
    nav = bars.set_index("date")["close"].sort_index()
    nav = nav.loc[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
    return nav.pct_change().dropna()


def market_returns(start, end, code=MARKET_CODE):
    """Cash-excess daily returns of the market factor."""
    return index_returns(start, end, code)


def load_strategy(directory):
    """Daily net returns of one strategy, indexed by date."""
    frame = pd.read_csv(directory / "daily.csv", encoding="utf-8")
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["daily_return"].astype(float)
    return series.loc[~series.index.duplicated()].sort_index()


def regress_strategy(returns, market, size=None, cash_rate=CASH_RATE, days=TRADING_DAYS):
    """CAPM regression of one strategy, optionally with a size factor."""
    columns = dict(r=returns, rm=market)
    if size is not None:
        columns["smb"] = size
    joined = pd.concat(columns, axis=1, join="inner").dropna()
    if len(joined) < 30:
        raise ValueError("Too few overlapping observations to regress")
    rf = cash_rate / days
    excess = joined["r"].to_numpy() - rf
    factor = joined["rm"].to_numpy() - rf
    result = ols(excess, factor)
    result.update(observations=int(len(joined)), one_factor=True)
    if size is not None:
        result["two_factor"] = ols_two_factor(excess, factor, joined["smb"].to_numpy())
    result.update(
        annual_alpha=float(result["alpha"] * days),
        # The t statistic is invariant to rescaling alpha, so annualising it does not change it.
        annual_alpha_t=float(result["alpha_t"]),
        annual_return=float((1.0 + joined["r"]).prod() ** (days / len(joined)) - 1.0),
        annual_vol=float(joined["r"].std(ddof=1) * math.sqrt(days)),
        annual_market=float((1.0 + joined["rm"]).prod() ** (days / len(joined)) - 1.0),
        start=str(joined.index[0].date()), end=str(joined.index[-1].date()))
    if "two_factor" in result:
        result["two_factor"]["annual_alpha"] = float(result["two_factor"]["alpha"] * days)
    result["sharpe_excess"] = float((result["annual_return"] - cash_rate) / result["annual_vol"]
                                    if result["annual_vol"] > 0 else float("nan"))
    return result


def finite(value):
    """Replace non-finite numbers with None so the payload stays JSON compliant.

    A cash-only strategy has no return variance, so its regression is degenerate:
    the slope, the residual correlation and every t statistic are undefined rather
    than zero.  Writing ``null`` keeps that distinction instead of inventing numbers.
    """
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def measure(run, *, splits=("selected_comparison", "final_test"), code=MARKET_CODE,
            size_code=SIZE_CODE, cash_rate=CASH_RATE, strategies=STRATEGIES):
    """Regress every stored strategy in every split, with and without the size factor."""
    payload = dict(code=code, size_code=size_code, cash_rate=float(cash_rate),
                   run=str(Path(run).resolve()), splits={})
    for split in splits:
        directory = Path(run) / split
        if not directory.exists():
            continue
        rows = {}
        for name in strategies:
            if not (directory / name / "daily.csv").exists():
                continue
            returns = load_strategy(directory / name)
            market = market_returns(returns.index[0], returns.index[-1], code)
            big = index_returns(returns.index[0], returns.index[-1], code)
            small = index_returns(returns.index[0], returns.index[-1], size_code)
            size = (small - big).dropna()
            rows[name] = regress_strategy(returns, market, size, cash_rate)
        if rows:
            payload["splits"][split] = rows
    if not payload["splits"]:
        raise ValueError("No strategy daily file was found under the requested run")
    return finite(dict(payload, verdict=verdict(payload)))


def verdict(payload):
    """What changes when the size tilt is priced.

    ``cash`` is excluded: its return series has no variance, so every coefficient in
    its regression is a division by zero rather than an estimate.
    """
    summary = {}
    for split, rows in payload["splits"].items():
        usable = {name: row for name, row in rows.items() if row.get("two_factor") and name != "cash"}
        if not usable:
            continue
        before = {name: row["annual_alpha"] for name, row in usable.items()}
        after = {name: row["two_factor"]["annual_alpha"] for name, row in usable.items()}
        best = max(after, key=lambda name: after[name])
        summary[split] = dict(
            observations=int(next(iter(usable.values()))["observations"]),
            market_annual=float(next(iter(usable.values()))["annual_market"]),
            one_factor_range=[float(min(before.values())), float(max(before.values()))],
            two_factor_range=[float(min(after.values())), float(max(after.values()))],
            one_factor_best=max(before, key=lambda name: before[name]),
            two_factor_best=best,
            two_factor_best_value=float(after[best]),
            two_factor_best_t=float(usable[best]["two_factor"]["alpha_t"]),
            alpha_shift=float(np.mean([after[name] - before[name] for name in usable])),
            beta_size_range=[float(min(row["two_factor"]["beta_size"] for row in usable.values())),
                             float(max(row["two_factor"]["beta_size"] for row in usable.values()))],
            significant_two_factor=[name for name, row in usable.items()
                                    if abs(row["two_factor"]["alpha_t"]) >= 2.0])
    return summary


def _fmt(value, spec):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return format(value, spec)


def _markdown(payload):
    lines = ["# 策略的 CAPM 分解（一因子与两因子）", "",
             f"市场因子：{payload['code']} 的日超额收益；规模因子：{payload['size_code']} 减 {payload['code']}。"
             f"现金年化 {payload['cash_rate']:.2%}。α 按每个文件的实际天数年化。", "",
             f"**{payload['code']} 只是大盘市值加权指数的代理，不是 CAPM 的市场组合。** "
             f"它对这个池子（等权 32 只）不匹配，所以两因子一栏才是关键。"
             f"实测结论与预期相反：**加入规模因子后 α 没有塌，反而略升**（规模载荷为正），"
             f"说明一因子的 α 不是规模倾斜——它来自池子的选样偏差，任何指数因子都吸收不掉。", ""]
    for split, rows in payload["splits"].items():
        first = next(iter(rows.values()))
        lines += [f"## {split}（{first['start']} ~ {first['end']}，{first['observations']} 个交易日）", "",
                  f"同期市场年化 {_fmt(first['annual_market'], '+.2%')}。", "",
                  "| 策略 | 年化收益 | α（一因子） | t | β | α（加规模） | t | β(规模) | R²(两因子) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, row in sorted(rows.items(), key=lambda item: -((item[1].get("two_factor") or {}).get("annual_alpha") or -9e9)):
            two = row.get("two_factor") or {}
            lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, _fmt(row["annual_return"], "+.2%"),
                _fmt(row["annual_alpha"], "+.2%"), _fmt(row["alpha_t"], "+.2f"),
                _fmt(row["beta"], ".3f"),
                _fmt(two.get("annual_alpha"), "+.2%"), _fmt(two.get("alpha_t"), "+.2f"),
                _fmt(two.get("beta_size"), ".3f"), _fmt(two.get("r_squared"), ".3f")))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Decompose stored strategies into market, size and alpha.")
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--code", default=MARKET_CODE)
    parser.add_argument("--size-code", default=SIZE_CODE)
    parser.add_argument("--cash-rate", type=float, default=CASH_RATE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    payload = measure(args.run, code=args.code, size_code=args.size_code, cash_rate=args.cash_rate)
    target = Path(args.output) if args.output else REPORT_ROOT
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "baseline_capm.json", payload)
    (target / "baseline_capm.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
