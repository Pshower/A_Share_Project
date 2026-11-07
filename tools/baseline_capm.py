"""Separate beta from alpha before comparing strategies.

Every comparison in this repository so far ranks net returns or Sharpe ratios
against each other.  That is a beta comparison whenever the sample drifts: a book
that simply carried more market exposure wins a rising window without containing
any skill.  This tool regresses each stored strategy on the same market factor, so
the ranking can be restated as alpha with beta held constant.

The factor is the cash-excess return of 000300 over exactly the strategy's own
dates.  The regression is ``r - rf = alpha + beta * (rm - rf) + e`` on daily
returns, estimated by least squares, with ``alpha`` annualised by the number of
observations in the file.  The t statistic for alpha uses the classical
homoskedastic standard error and, as a check on that assumption, the first-order
autocorrelation of the residuals is reported.

Nothing is re-simulated and no return is recomputed: the inputs are the daily
files each run already wrote.  The point is to say whether "1/N wins" means "1/N
has alpha" or only "1/N had more beta".
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


def ols(y, x):
    """Slope, intercept, standard errors, residual autocorrelation and R squared."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape or y.size < 3:
        raise ValueError("Regressand and regressor must share a shape with at least 3 rows")
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted
    dof = y.size - design.shape[1]
    sigma2 = float(residuals @ residuals) / dof
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    errors = np.sqrt(np.diag(covariance))
    total = float(((y - y.mean()) ** 2).sum())
    lag1 = float("nan")
    if y.size > 3 and residuals.std() > 0:
        lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    return dict(alpha=float(coefficients[0]), beta=float(coefficients[1]),
                alpha_t=float(coefficients[0] / errors[0]) if errors[0] > 0 else float("nan"),
                beta_t=float(coefficients[1] / errors[1]) if errors[1] > 0 else float("nan"),
                residual_lag1=lag1,
                r_squared=1.0 - float(residuals @ residuals) / total if total > 0 else float("nan"))


def market_returns(start, end, code="000300"):
    """Cash-excess daily returns of the index inside ``start`` .. ``end``."""
    bars = load_index(SNAPSHOT, code)
    nav = bars.set_index("date")["close"].sort_index()
    nav = nav.loc[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
    return nav.pct_change().dropna()


def load_strategy(directory):
    """Daily net returns of one strategy, indexed by date."""
    frame = pd.read_csv(directory / "daily.csv", encoding="utf-8")
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["daily_return"].astype(float)
    return series.loc[~series.index.duplicated()].sort_index()


def regress_strategy(returns, market, cash_rate=CASH_RATE, days=TRADING_DAYS):
    """CAPM regression of one strategy on the market factor."""
    joined = pd.concat([returns.rename("r"), market.rename("rm")], axis=1, join="inner").dropna()
    if len(joined) < 30:
        raise ValueError("Too few overlapping observations to regress")
    rf = cash_rate / days
    result = ols(joined["r"].to_numpy() - rf, joined["rm"].to_numpy() - rf)
    # The t statistic is invariant to rescaling alpha, so annualising it does not change it.
    result.update(observations=int(len(joined)),
                  annual_alpha=float(result["alpha"] * days),
                  annual_alpha_t=float(result["alpha_t"]),
                  annual_return=float((1.0 + joined["r"]).prod() ** (days / len(joined)) - 1.0),
                  annual_vol=float(joined["r"].std(ddof=1) * math.sqrt(days)),
                  annual_market=float((1.0 + joined["rm"]).prod() ** (days / len(joined)) - 1.0),
                  start=str(joined.index[0].date()), end=str(joined.index[-1].date()))
    result["sharpe_excess"] = float((result["annual_return"] - cash_rate) / result["annual_vol"]
                                    if result["annual_vol"] > 0 else float("nan"))
    result["information_ratio"] = float(result["annual_alpha"] / result["annual_vol"]
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


def measure(run, *, splits=("selected_comparison", "final_test"), code="000300",
            cash_rate=CASH_RATE, strategies=STRATEGIES):
    """Regress every stored strategy in every split against the same factor."""
    payload = dict(code=code, cash_rate=float(cash_rate), run=str(Path(run).resolve()), splits={})
    for split in splits:
        directory = Path(run) / split
        if not directory.exists():
            continue
        rows = {}
        for name in strategies:
            path = directory / name / "daily.csv"
            if not path.exists():
                continue
            returns = load_strategy(directory / name)
            market = market_returns(returns.index[0], returns.index[-1], code)
            rows[name] = regress_strategy(returns, market, cash_rate)
        if rows:
            payload["splits"][split] = rows
    if not payload["splits"]:
        raise ValueError("No strategy daily file was found under the requested run")
    return finite(dict(payload, verdict=verdict(payload)))


def verdict(payload):
    """Which statements the regressions support."""
    best = {}
    for split, rows in payload["splits"].items():
        ranked = sorted(rows.items(), key=lambda item: item[1]["annual_alpha"], reverse=True)
        significant = [name for name, row in ranked if abs(row["alpha_t"]) >= 2.0]
        best[split] = dict(
            observations=next(iter(rows.values()))["observations"],
            market_annual=next(iter(rows.values()))["annual_market"],
            best_alpha=ranked[0][0],
            best_alpha_value=ranked[0][1]["annual_alpha"],
            best_alpha_t=ranked[0][1]["alpha_t"],
            best_sharpe=max(rows, key=lambda name: rows[name]["sharpe_excess"]),
            significant_alpha=significant,
            alpha_range=[float(ranked[-1][1]["annual_alpha"]), float(ranked[0][1]["annual_alpha"])],
            beta_range=[float(min(row["beta"] for row in rows.values())),
                        float(max(row["beta"] for row in rows.values()))])
    return best


def _fmt(value, spec):
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def _markdown(payload):
    lines = ["# 策略的 CAPM 分解", "",
             f"因子：{payload['code']} 的日超额收益；现金年化 {payload['cash_rate']:.2%}。"
             f"回归 `r − rf = α + β(rm − rf) + e`，α 按每个文件的实际天数年化。", ""]
    for split, rows in payload["splits"].items():
        first = next(iter(rows.values()))
        lines += [f"## {split}（{first['start']} ~ {first['end']}，{first['observations']} 个交易日）", "",
                  f"同期市场年化 {_fmt(first['annual_market'], '+.2%')}。", "",
                  "| 策略 | 年化收益 | 年化 α | α 的 t | β | β 的 t | 残差 lag-1 | R² | 超额 Sharpe |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, row in sorted(rows.items(), key=lambda item: -(item[1]["annual_alpha"] or -9e9)):
            lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, _fmt(row["annual_return"], "+.2%"), _fmt(row["annual_alpha"], "+.2%"),
                _fmt(row["alpha_t"], "+.2f"), _fmt(row["beta"], ".3f"),
                _fmt(row["beta_t"], "+.2f"), _fmt(row["residual_lag1"], "+.3f"),
                _fmt(row["r_squared"], ".3f"), _fmt(row["sharpe_excess"], "+.3f")))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regress stored strategies against the market factor.")
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--code", default="000300")
    parser.add_argument("--cash-rate", type=float, default=CASH_RATE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    payload = measure(args.run, code=args.code, cash_rate=args.cash_rate)
    target = Path(args.output) if args.output else REPORT_ROOT
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "baseline_capm.json", payload)
    (target / "baseline_capm.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
