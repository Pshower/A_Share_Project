"""Audit every feature family the same way: profile it, then test whether it predicts.

Three blocks, one rule set:

* **price/volume** -- the 20+ features on the 298-stock panel, cross-sectional rank IC against
  forward 1/5/20-day returns.
* **fundamental** -- index valuation (CSI 300 PE/PB) against forward index returns at 1/3/6/12/36
  months, plus a statement about what cannot be tested (stock-level fundamentals exist only as a
  snapshot here, so they cannot be given a time series IC).
* **macro** -- monthly series against forward index returns at 1/3/12 months, plus a regime split.

The bar is the repository's own: |IC| >= 0.031 and |t| >= 2.5.  This is a **screening audit**, not an
adoption: nothing here is added to any book, and a pass is only permission to write a
pre-registration later.  Monthly data with ~200 observations rarely clears that bar, and the report
says so rather than pretending a t of 1.2 is a finding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "data" / "clean" / "all_stocks_features.parquet"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "feature_audit"
BAR_IC, BAR_T = 0.031, 2.5
MIN_WINDOWS = 30      # a time-series claim needs at least this many non-overlapping windows

SKIP = {"date", "code", "股票名称"}
HORIZON_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "12M": 12, "36M": 36}   # months, not per year


def hac_slope_t(regressor, target, lags):
    """Newey-West t of the univariate slope; ``lags`` should cover the overlap in the target.

    With ranks and both sides standardised, the regressor has mean zero and ``X'X = n``, so

        Var(b) = n^-2 * [ sum(e_t^2 x_t^2) + 2 sum_l w_l sum(e_t e_{t-l} x_t x_{t-l}) ]

    and ``t = b / sqrt(Var(b))``.  This is the number that decides whether an apparent IC survives
    overlapping windows and a persistent regressor.
    """
    frame = pd.concat([regressor.rename("x"), target.rename("y")], axis=1).dropna()
    if len(frame) < 24:
        return float("nan")
    x = frame["x"].rank().values.astype(float)
    y = frame["y"].rank().values.astype(float)
    x = (x - x.mean()) / (x.std() or 1.0)
    y = (y - y.mean()) / (y.std() or 1.0)
    n = len(x)
    slope = float((x * y).mean())
    residual = y - slope * x
    meat = float((residual ** 2 * x ** 2).mean())
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        autoregressive = float((residual[lag:] * residual[:-lag] * x[lag:] * x[:-lag]).mean())
        meat += 2.0 * weight * autoregressive
    variance = meat / n
    if variance <= 0:
        return float("nan")
    return float(slope / (variance ** 0.5) * (n ** 0.5) / (n ** 0.5))



def effective_n(n, months):
    """How many independent windows a series of ``n`` monthly prints actually offers."""
    return max(1, int(round(n / max(1, months))))


def spearman(left, right):
    """Rank correlation without scipy: rank both series, then a plain Pearson correlation."""
    frame = pd.concat([left.rename("a"), right.rename("b")], axis=1).dropna()
    if len(frame) < 5:
        return float("nan")
    return float(frame["a"].rank().corr(frame["b"].rank()))


def newey_t(series, lags=0):
    """t-stat of the mean with Newey-West standard errors; overlapping horizons need this."""
    values = pd.Series(series).dropna()
    n = len(values)
    if n < 5:
        return np.nan
    centred = values - values.mean()
    variance = float((centred ** 2).sum()) / n
    for lag in range(1, min(lags, n - 1) + 1):
        covariance = float((centred.iloc[lag:].values * centred.iloc[:-lag].values).sum()) / n
        variance += 2 * (1 - lag / (lags + 1)) * covariance
    if variance <= 0:
        return np.nan
    return float(values.mean() / np.sqrt(variance / n))


def profile(frame, name):
    values = pd.to_numeric(frame, errors="coerce").dropna()
    if values.empty:
        return None
    returns = values.diff()
    autocorrelation = float(returns.autocorr(1)) if len(returns.dropna()) > 5 else np.nan
    return dict(feature=name, n=int(len(values)), coverage=round(float(values.notna().mean()), 4),
                mean=round(float(values.mean()), 4), std=round(float(values.std()), 4),
                skew=round(float(values.skew()), 3) if len(values) > 3 else None,
                min=round(float(values.min()), 4), max=round(float(values.max()), 4),
                ar1_diff=round(autocorrelation, 3) if np.isfinite(autocorrelation) else None)


def cross_sectional_ic(panel, feature, horizons=(1, 5, 20), min_names=30):
    """Mean rank IC per horizon, with a Newey-West t over the horizon's overlap."""
    rows = []
    wide = panel.pivot_table(index="date", columns="code", values=feature)
    close = panel.pivot_table(index="date", columns="code", values="close")
    for horizon in horizons:
        forward = close.shift(-horizon) / close - 1.0
        ranked = wide.rank(axis=1)
        ic = ranked.corrwith(forward.rank(axis=1), axis=1)
        ic = ic[forward.notna().sum(axis=1) >= min_names].dropna()
        if len(ic) < 30:
            continue
        rows.append(dict(feature=feature, horizon=horizon, days=len(ic),
                         mean_ic=round(float(ic.mean()), 4),
                         t=round(newey_t(ic, lags=max(1, horizon)), 2),
                         positive=round(float((ic > 0).mean()), 3),
                         first_half=round(float(ic.iloc[: len(ic) // 2].mean()), 4),
                         second_half=round(float(ic.iloc[len(ic) // 2:].mean()), 4)))
    return rows


def price_volume_block(panel, log=print):
    features = [column for column in panel.columns if column not in SKIP
                and pd.api.types.is_numeric_dtype(panel[column])]
    profiles, results = [], []
    for feature in features:
        shape = profile(panel[feature], feature)
        if shape:
            profiles.append(shape)
        results += cross_sectional_ic(panel, feature)
    for row in results:
        row["passes"] = bool(abs(row["mean_ic"]) >= BAR_IC and abs(row["t"]) >= BAR_T)
        row["stable"] = bool(np.sign(row["first_half"]) == np.sign(row["second_half"]))
    log("price/volume: %d features, %d IC rows, %d pass the bar" % (
        len(profiles), len(results), sum(row["passes"] for row in results)))
    return profiles, results


def index_returns(daily, horizon):
    close = daily.set_index("date")["close"].sort_index()
    return close.pct_change(horizon).dropna()


def fundamental_block(log=print):
    import akshare as ak

    results, notes = [], []
    try:
        daily = ak.stock_zh_index_daily(symbol="sh000300")
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        pe = ak.stock_index_pe_lg(symbol="沪深300")
    except Exception as error:                              # noqa: BLE001
        log("fundamental: index data unavailable (%s)" % type(error).__name__)
        return [], [dict(block="fundamental", note="index data unavailable")]

    pe.columns = [str(column) for column in pe.columns]
    pe = pe.rename(columns={column: "指数点位" for column in pe.columns if column.strip() == "指数"})
    date_column = next(column for column in pe.columns if "日期" in column)
    pe[date_column] = pd.to_datetime(pe[date_column])
    value_columns = [column for column in pe.columns
                     if column != date_column and pd.api.types.is_numeric_dtype(pe[column])]
    for column in value_columns:
        series = pe.set_index(date_column)[column].dropna()
        for horizon, months in HORIZON_MONTHS.items():
            forward = daily.set_index("date")["close"].shift(-int(months * 21))
            forward = forward / daily.set_index("date")["close"] - 1.0
            joined = pd.concat([series.rename("value"), forward.rename("fwd")], axis=1).dropna()
            if len(joined) < 30:
                continue
            ic = spearman(joined["value"], joined["fwd"])
            span = max(1, int(months * 21))            # trading days
            t = hac_slope_t(joined["value"], joined["fwd"], lags=min(36, months))
            results.append(dict(block="fundamental", feature=column, horizon=horizon,
                                n=int(len(joined)), n_eff=effective_n(len(joined), max(1, months)),
                                ic=round(float(ic), 4), t=round(float(t), 2) if np.isfinite(t) else None,
                                power=bool(effective_n(len(joined), max(1, months)) >= MIN_WINDOWS),
                                passes=bool(abs(ic) >= BAR_IC and np.isfinite(t) and abs(t) >= BAR_T
                                            and effective_n(len(joined), max(1, months)) >= MIN_WINDOWS)))
    notes.append(dict(block="fundamental",
                      note="stock-level fundamentals exist only as today's snapshot; a time-series "
                           "IC needs point-in-time history, which this repository does not keep"))
    log("fundamental: %d rows" % len(results))
    return results, notes


MACRO = {
    "CPI 同比": lambda ak: ak.macro_china_cpi(),
    "PPI 同比": lambda ak: ak.macro_china_ppi(),
    "PMI": lambda ak: ak.macro_china_pmi(),
    "M2 同比": lambda ak: ak.macro_china_money_supply(),
    "GDP 同比": lambda ak: ak.macro_china_gdp(),
    "LPR 5Y": lambda ak: ak.macro_china_lpr(),
}

MONTHS = {"月份": None, "日期": None, "季度": None, "时间": None}


def pick_value(frame, hints):
    for hint in hints:
        for column in frame.columns:
            if hint in column and pd.api.types.is_numeric_dtype(frame[column]):
                return column
    return None


MACRO_HINTS = {"CPI 同比": ["全国-同比增长", "同比增长"], "PPI 同比": ["当月同比增长", "同比增长"],
               "PMI": ["制造业-指数"], "M2 同比": ["同比增长"],
               "GDP 同比": ["同比增长"], "LPR 5Y": ["LPR5Y", "5Y"]}


def macro_block(log=print):
    import akshare as ak
    from tools import data_catalog

    results, notes = [], []
    try:
        daily = ak.stock_zh_index_daily(symbol="sh000300")
        daily["date"] = pd.to_datetime(daily["date"])
        index = daily.set_index("date")["close"].sort_index()
    except Exception as error:                              # noqa: BLE001
        log("macro: index unavailable (%s)" % type(error).__name__)
        return [], [dict(block="macro", note="index unavailable")]

    for name, fetch in MACRO.items():
        try:
            frame = fetch(ak)
        except Exception as error:                          # noqa: BLE001
            notes.append(dict(block="macro", feature=name, note="fetch failed: %s" % type(error).__name__))
            continue
        date_column = next((column for column in frame.columns
                            if any(key in str(column) for key in MONTHS)), None)
        value_column = pick_value(frame, MACRO_HINTS.get(name, []))
        if date_column is None or value_column is None:
            notes.append(dict(block="macro", feature=name, note="no usable date/value column"))
            continue
        parsed = data_catalog.to_dates(frame[date_column])
        series = pd.Series(pd.to_numeric(frame[value_column], errors="coerce").values,
                           index=parsed.values).dropna().sort_index()
        series = series[~series.index.duplicated(keep="last")]
        monthly = index.resample("M").last()
        for horizon, months in (("1M", 1), ("3M", 3), ("12M", 12)):
            forward = monthly.shift(-months) / monthly - 1.0
            aligned = pd.concat([series.resample("M").last().rename("value"),
                                 forward.rename("fwd")], axis=1).dropna()
            if len(aligned) < 24:
                notes.append(dict(block="macro", feature=name, note="sample too short"))
                continue
            ic = spearman(aligned["value"], aligned["fwd"])
            t = hac_slope_t(aligned["value"], aligned["fwd"], lags=min(36, months))
            results.append(dict(block="macro", feature=name, horizon=horizon, n=int(len(aligned)),
                                n_eff=effective_n(len(aligned), months), ic=round(float(ic), 4),
                                t=round(float(t), 2) if np.isfinite(t) else None,
                                power=bool(effective_n(len(aligned), months) >= MIN_WINDOWS),
                                passes=bool(abs(ic) >= BAR_IC and np.isfinite(t) and abs(t) >= BAR_T
                                            and effective_n(len(aligned), months) >= MIN_WINDOWS)))
            median = aligned["value"].median()
            high = aligned[aligned["value"] > median]["fwd"]
            low = aligned[aligned["value"] <= median]["fwd"]
            if len(high) > 5 and len(low) > 5:
                spread = float(high.mean() - low.mean())
                error = np.sqrt(high.var() / len(high) + low.var() / len(low))
                results.append(dict(block="macro", feature=name, horizon=horizon + " regime",
                                    n=int(len(aligned)), ic=round(spread, 4),
                                    t=round(spread / error, 2) if error > 0 else None,
                                    passes=False))
    log("macro: %d rows, %d notes" % (len(results), len(notes)))
    return results, notes


def run(panel_path=PANEL, output=OUTPUT, log=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "code"])

    profiles, price_rows = price_volume_block(panel, log=log)
    fundamental_rows, fundamental_notes = fundamental_block(log=log)
    macro_rows, macro_notes = macro_block(log=log)

    pd.DataFrame(profiles).to_csv(output / "price_profiles.csv", index=False)
    pd.DataFrame(price_rows).to_csv(output / "price_ic.csv", index=False)
    pd.DataFrame(fundamental_rows + macro_rows).to_csv(output / "index_and_macro.csv", index=False)
    summary = dict(
        generated_at=pd.Timestamp.now().isoformat(),
        panel=dict(rows=int(len(panel)), codes=int(panel["code"].nunique()),
                   start=str(panel["date"].min().date()), end=str(panel["date"].max().date())),
        bar=dict(ic=BAR_IC, t=BAR_T),
        price=dict(features=len(profiles), rows=len(price_rows),
                   passes=sum(1 for row in price_rows if row["passes"]),
                   stable=sum(1 for row in price_rows if row["stable"])),
        fundamental=dict(rows=len(fundamental_rows), notes=fundamental_notes),
        macro=dict(rows=len(macro_rows), notes=macro_notes),
        notes=["screening audit only: a pass is permission to pre-register, not adoption"],
    )
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    log("wrote %s" % output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit every feature family.")
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    summary = run(args.panel, args.output)
    print(json.dumps({key: summary[key] for key in ("price", "fundamental", "macro")},
                     ensure_ascii=False, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
