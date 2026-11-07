"""CAPM tests on the full-market panel: market portfolio, beta pricing, and 1/N.

Three questions, in the order the protocol registers them:

* **T1** do beta-sorted deciles earn the same alpha, as CAPM says they must?
* **T2** is the security market line's slope equal to the market's own excess
  return?  This is CAPM's only falsifiable core, because a flat or negative slope
  rejects it outright.
* **T3** is holding the value-weighted market portfolio actually competitive with
  the naive ``1/N`` book on the same universe and the same costs?

Everything is measured on **non-overlapping 20-day periods**, so the Fama-MacBeth
standard errors need no Newey-West correction: each rebalance reads a fresh block
of returns.  The sample starts in 2019 so 2017-2018 can warm up the beta window,
which means the formal sample is the spent V segment: this is a structural
description, not a deployment test, and the protocol says so before any number
was seen.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import write_json
from tools.full_market_panel import DEFAULT_OUTPUT, load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "capm_full_market"

CASH_RATE = 0.02
TRADING_DAYS = 243
WINDOW = 60
MIN_OBS = 40
STEP = 20
DECILES = 10
SAMPLE_START = "2019-01-02"
SAMPLE_END = "2024-11-30"


def daily_frame(prices):
    """Simple returns from a wide price frame, without forward-filling gaps."""
    returns = prices.astype(float).pct_change(fill_method=None)
    return returns.replace([np.inf, -np.inf], np.nan)


def market_portfolio(prices, traded, shares, cash_rate=CASH_RATE, days=TRADING_DAYS):
    """Value-weighted market return from traded price times free-float shares.

    Capitalisation must use the **traded** price: the adjusted series is not a
    price anyone can transact at.  Weights lag one day, and names without a valid
    return that day are dropped and the rest renormalised, so a suspension cannot
    silently hold a stale weight at zero return.
    """
    capitalisation = (traded * shares).where(prices.notna())
    raw = capitalisation.div(capitalisation.sum(axis=1), axis=0)
    returns = daily_frame(prices)
    weights = raw.shift(1).where(returns.notna())
    weights = weights.div(weights.sum(axis=1), axis=0)
    market = (weights * returns.fillna(0.0)).sum(axis=1)
    market = market.where(weights.sum(axis=1) > 0)
    market.name = "market"
    return market, weights, capitalisation


def rolling_beta(returns, market, cash_rate=CASH_RATE, days=TRADING_DAYS, window=WINDOW, min_obs=MIN_OBS):
    """Rolling market beta of every column on the market excess return."""
    rf = cash_rate / days
    excess = returns.sub(rf)
    factor = (market - rf).reindex(returns.index)
    covariance = excess.rolling(window, min_periods=min_obs).cov(factor)
    variance = factor.rolling(window, min_periods=min_obs).var()
    return covariance.div(variance, axis=0)


def forward_returns(returns, step=STEP):
    """Return over the ``step`` sessions that follow the decision date.

    ``step`` sessions after ``t`` means sessions ``t+1 .. t+step``, and since
    ``growth[t]`` already contains ``r[t]`` the correct ratio is
    ``growth[t+step] / growth[t] - 1``.  Dividing by ``growth[t+1]`` instead would
    silently measure ``step - 1`` sessions.
    """
    growth = (1.0 + returns).cumprod()
    return growth.shift(-step) / growth - 1.0


def rebalance_dates(index, start=SAMPLE_START, end=SAMPLE_END, step=STEP):
    """Every ``step``-th session inside the sample, so the blocks do not overlap."""
    inside = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    return inside[::step]


def cross_section(betas, forward, market_forward, dates, rf_period):
    """Decile statistics and the Fama-MacBeth slope, one row per rebalance."""
    rows = []
    for date in dates:
        if date not in betas.index or date not in forward.index:
            continue
        beta = betas.loc[date]
        ret = forward.loc[date] - rf_period
        usable = beta.notna() & ret.notna()
        if usable.sum() < DECILES * 5:
            continue
        beta, ret = beta[usable], ret[usable]
        order = beta.rank(method="first")
        bucket = np.minimum((order - 1) * DECILES // len(order), DECILES - 1) + 1
        members = {int(b): ret[bucket == b] for b in bucket.unique()}
        if len(members) < DECILES:
            continue
        design = np.column_stack([np.ones(len(beta)), beta.to_numpy()])
        coefficients, *_ = np.linalg.lstsq(design, ret.to_numpy(), rcond=None)
        rows.append(dict(date=date, gamma0=float(coefficients[0]), gamma1=float(coefficients[1]),
                         names=int(usable.sum()),
                         market=float(market_forward.loc[date] - rf_period),
                         mean_beta=float(beta.mean()),
                         deciles={key: float(value.mean()) for key, value in members.items()},
                         decile_beta={key: float(beta[bucket == key].mean()) for key in members}))
    return rows


def equal_weight_portfolio(returns, step=STEP):
    """Equal weight rebalanced every ``step`` sessions, drifting in between.

    A daily-rebalanced ``1/N`` is not the strategy anyone runs: it pays the full
    round trip on every name every day.  Holding the same weights and resetting
    them monthly is both cheaper and what the repository's own ``equal_weight``
    baseline does, so that is the book the market portfolio has to beat.

    Returns the daily return series and the turnover charged at each rebalance,
    where turnover is half the summed absolute weight change.
    """
    values = returns.to_numpy(dtype=float)
    n = values.shape[1]
    weights = np.full(n, 1.0 / n)
    out = np.zeros(len(values))
    turnover = np.zeros(len(values))
    for position in range(len(values)):
        row = np.nan_to_num(values[position], nan=0.0)
        out[position] = float(weights @ row)
        grown = weights * (1.0 + row)
        total = grown.sum()
        weights = grown / total if total > 0 else np.full(n, 1.0 / n)
        if (position + 1) % step == 0:
            target = np.full(n, 1.0 / n)
            turnover[position] = float(np.abs(target - weights).sum() / 2.0)
            weights = target
    return pd.Series(out, index=returns.index), pd.Series(turnover, index=returns.index)


def net_of_costs(returns, turnover, round_trip=0.0031):
    """Subtract the transaction cost implied by a turnover series."""
    return returns - turnover * round_trip


def performance(series, cash_rate=CASH_RATE, days=TRADING_DAYS):
    """Annualised return, volatility, drawdown and excess Sharpe of a return series."""
    clean = series.dropna()
    if clean.empty:
        return dict(observations=0)
    growth = (1.0 + clean).cumprod()
    years = len(clean) / days
    annual = float(growth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    volatility = float(clean.std(ddof=1) * math.sqrt(days))
    peak = growth.cummax()
    drawdown = float((growth / peak - 1.0).min())
    return dict(observations=int(len(clean)), annual_return=annual, annual_vol=volatility,
                max_drawdown=drawdown,
                sharpe_excess=float((annual - cash_rate) / volatility) if volatility > 0 else float("nan"),
                cumulative=float(growth.iloc[-1] - 1.0))


def summarize(rows, names):
    """Turn the per-rebalance rows into the three statistics the protocol names."""
    if not rows:
        return dict(status="empty")
    gamma1 = np.array([row["gamma1"] for row in rows])
    gamma0 = np.array([row["gamma0"] for row in rows])
    market = np.array([row["market"] for row in rows])
    decile_means = {key: float(np.mean([row["deciles"][key] for row in rows])) for key in range(1, DECILES + 1)}
    decile_betas = {key: float(np.mean([row["decile_beta"][key] for row in rows])) for key in range(1, DECILES + 1)}
    # Alpha per decile: realised excess minus the decile's beta times the market excess.
    decile_alpha = {key: float(np.mean([row["deciles"][key] - decile_betas[key] * row["market"] for row in rows]))
                    for key in range(1, DECILES + 1)}
    se = gamma1.std(ddof=1) / math.sqrt(len(gamma1)) if len(gamma1) > 1 else float("nan")
    return dict(status="ready", periods=len(rows),
                gamma1_mean=float(gamma1.mean()), gamma1_se=float(se),
                gamma1_t=float(gamma1.mean() / se) if se and se > 0 else float("nan"),
                gamma0_mean=float(gamma0.mean()),
                market_excess_mean=float(market.mean()),
                slope_gap=float(gamma1.mean() - market.mean()),
                slope_gap_t=float((gamma1.mean() - market.mean()) / se) if se and se > 0 else float("nan"),
                decile_excess=decile_means, decile_beta=decile_betas, decile_alpha=decile_alpha,
                alpha_spread=float(decile_alpha[DECILES] - decile_alpha[1]),
                mean_names=float(np.mean([row["names"] for row in rows])),
                universe=int(names))


def verdict(summary, market_perf, equal_perf):
    """The protocol's mechanical reading of the slope, in the registered order.

    The ladder is ordered so that each case is reachable exactly once: a slope
    significantly *below* the market premium rejects CAPM outright; a slope that is
    significant and close to the premium does not; a slope close to the premium but
    imprecise is "cannot distinguish"; and a slope significantly *above* the premium
    is reported as its own outcome rather than forced into one of the three.
    """
    if summary.get("status") != "ready":
        return dict(status="empty")
    gap_t = summary["slope_gap_t"]
    slope_t = summary["gamma1_t"]
    if gap_t <= -2:
        reading, rejected = "slope significantly BELOW the market premium: CAPM rejected", True
    elif abs(gap_t) < 2 and slope_t >= 2:
        reading, rejected = "beta is priced and the slope matches the market premium", False
    elif abs(gap_t) < 2:
        reading, rejected = "slope has the right sign but is too imprecise to call", None
    else:
        reading, rejected = "slope significantly ABOVE the market premium", False
    return dict(status="ready", slope_reading=reading, capm_rejected=rejected,
                low_vol_anomaly=bool(summary["alpha_spread"] < 0),
                market_vs_equal=float(market_perf.get("annual_return", float("nan"))
                                      - equal_perf.get("annual_return", float("nan"))),
                market_sharpe=market_perf.get("sharpe_excess"),
                equal_sharpe=equal_perf.get("sharpe_excess"))


def measure(panel_dir=DEFAULT_OUTPUT, *, start=SAMPLE_START, end=SAMPLE_END, cash_rate=CASH_RATE):
    """Run T1, T2 and T3 on a built panel."""
    manifest, prices, traded, shares = load(panel_dir)
    market, weights, capitalisation = market_portfolio(prices, traded, shares, cash_rate)
    market = market.dropna()
    returns = daily_frame(prices)
    betas = rolling_beta(returns, market, cash_rate)
    forward = forward_returns(returns)
    growth = (1.0 + market).cumprod()
    market_forward = growth.shift(-STEP) / growth - 1.0
    dates = rebalance_dates(returns.index, start, end)
    rf_period = cash_rate / TRADING_DAYS * STEP
    rows = cross_section(betas, forward, market_forward, dates, rf_period)
    summary = summarize(rows, manifest["shape"][1])

    window = returns.loc[(returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))]
    market_window = market.reindex(window.index)
    equal, turnover = equal_weight_portfolio(window, STEP)
    equal = equal.reindex(market_window.index)
    turnover = turnover.reindex(market_window.index).fillna(0.0)
    equal_net = net_of_costs(equal, turnover)
    market_perf = performance(market_window, cash_rate)
    equal_perf = performance(equal, cash_rate)
    equal_net_perf = performance(equal_net, cash_rate)
    return dict(panel=dict(path=str(Path(panel_dir)), shape=manifest["shape"],
                           codes=manifest["shape"][1], missing=len(manifest.get("codes_missing", [])),
                           capitalisation=manifest.get("capitalisation")),
                start=start, end=end, step=STEP, window=WINDOW, deciles=DECILES,
                summary=summary,
                performance=dict(market=market_perf, equal_weight=equal_perf,
                                 equal_weight_net=equal_net_perf,
                                 annual_turnover=float(turnover.sum() * TRADING_DAYS / len(turnover))),
                verdict=verdict(summary, market_perf, equal_net_perf),
                periods=[dict(date=str(row["date"].date()), gamma1=row["gamma1"],
                              market=row["market"], names=row["names"]) for row in rows])


def _fmt(value, spec):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return format(value, spec)


def _markdown(payload):
    summary, perf = payload["summary"], payload["performance"]
    lines = ["# 全市场 CAPM 检验", "",
             f"面板：`{payload['panel']['path']}`，{payload['panel']['shape'][0]} 个交易日 × "
             f"{payload['panel']['shape'][1]} 只（缺 {payload['panel']['missing']} 只）。",
             f"样本 {payload['start']} ~ {payload['end']}，β 窗口 {payload['window']} 日，"
             f"每 {payload['step']} 日一个不重叠区间。", ""]
    if summary.get("status") != "ready":
        return "\n".join(lines + ["样本不足，无法给出检验结果。"])
    lines += ["## T1/T2 β 定价", "",
              "| 指标 | 值 |", "| --- | ---: |",
              f"| 区间数 | {summary['periods']} |",
              f"| 平均股票数 | {summary['mean_names']:.0f} |",
              f"| SML 斜率 γ1（均值） | {summary['gamma1_mean']:+.4f} |",
              f"| γ1 的 t | {summary['gamma1_t']:+.2f} |",
              f"| 同期市场区间超额 | {summary['market_excess_mean']:+.4f} |",
              f"| γ1 − 市场超额 | {summary['slope_gap']:+.4f}（t {summary['slope_gap_t']:+.2f}） |",
              f"| 截距 γ0 | {summary['gamma0_mean']:+.4f} |", "",
              "| 十分位 | 平均 β | 平均超额 | α（对市场） |", "| ---: | ---: | ---: | ---: |"]
    for key in range(1, DECILES + 1):
        lines.append("| D{} | {:.3f} | {:+.4f} | {:+.4f} |".format(
            key, summary["decile_beta"][key], summary["decile_excess"][key], summary["decile_alpha"][key]))
    lines += ["", f"α 极差 D10−D1：**{summary['alpha_spread']:+.4f}**（每 {payload['step']} 日）。", "",
              "## T3 市场组合 vs 1/N", "",
              "市值加权组合是**买入并持有**：价格变动按比例改变权重，不需要再平衡，因此几乎没有成本。"
              "1/N 每 20 日回到等权，按 31bp 往返扣费。", "",
              "| 组合 | 年化收益 | 年化波动 | 最大回撤 | 超额 Sharpe | 累计 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, block in (("市值加权 M（买入持有）", perf["market"]),
                        ("1/N 月频（毛）", perf["equal_weight"]),
                        ("1/N 月频（净）", perf.get("equal_weight_net", {}))):
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            name, _fmt(block.get("annual_return"), "+.2%"), _fmt(block.get("annual_vol"), ".2%"),
            _fmt(block.get("max_drawdown"), ".2%"), _fmt(block.get("sharpe_excess"), "+.3f"),
            _fmt(block.get("cumulative"), "+.2%")))
    lines += ["", f"1/N 的年化换手：**{_fmt(perf.get('annual_turnover'), '.2f')} 倍**，"
                  f"按 31bp 往返计成年化成本 {_fmt(_cost_drag(perf), '.2%')}。", ""]
    return "\n".join(lines)


def _cost_drag(perf):
    """Annual cost implied by the equal-weight turnover at a 31bp round trip."""
    turnover = perf.get("annual_turnover")
    return None if turnover is None else turnover * 0.0031


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the CAPM tests on the full-market panel.")
    parser.add_argument("--panel", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start", default=SAMPLE_START)
    parser.add_argument("--end", default=SAMPLE_END)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    payload = measure(args.panel, start=args.start, end=args.end)
    target = Path(args.output) if args.output else REPORT_ROOT
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "capm_full_market.json", payload["summary"] | {"verdict": payload["verdict"],
                                                                      "performance": payload["performance"]})
    (target / "capm_full_market.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
