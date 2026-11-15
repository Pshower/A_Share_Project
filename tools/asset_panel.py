"""One daily panel for every asset class in the unified strategy, with its assumptions explicit.

The panel is deliberately narrow: it produces **total-return levels** for cash, deposits, four
bond durations, A-shares, US shares, Hong Kong shares and gold, all on the A-share trading
calendar, and it writes down every assumption it made.  A cross-asset study is only as honest as
its weakest proxy, so the proxies live here rather than being scattered through the strategy code.

Three rules the module keeps:

1. **Nothing is known before it is knowable.**  The US and Hong Kong closes land after the
   Chinese close, so their series are shifted one Chinese session; using them same-day is the
   classic cross-market look-ahead.
2. **Income is a stated constant, not a silent zero.**  ``000300`` and ``sp500`` are price
   indices.  Comparing them with a bond index that already contains coupons would flatter bonds
   by two points a year, so a constant dividend yield is added to each and listed in the output.
3. **The bond model must be checked, not assumed.**  Synthetic bond returns come from the
   treasury curve; the check against ``000012`` is computed and reported, and the strategy layer
   refuses the synthetic series if the check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONLINE = PROJECT_ROOT / "data" / "online"
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "unified_plan"

# PBoC one-year benchmark deposit rate: the official step path, hand-entered from the published
# adjustment table (last change 2015-10-24, still in force).  Recorded as data, with its flaw
# stated: since 2016 banks pay below this benchmark, so a deposit sleeve built on it is generous.
DEPOSIT_STEPS = [
    ("2002-02-21", 1.98), ("2004-10-29", 2.25), ("2006-08-19", 2.52), ("2007-03-18", 2.79),
    ("2007-05-19", 3.06), ("2007-07-21", 3.33), ("2007-08-22", 3.60), ("2007-09-15", 3.87),
    ("2007-12-21", 4.14), ("2008-10-09", 3.87), ("2008-10-30", 3.60), ("2008-11-27", 2.52),
    ("2008-12-23", 2.25), ("2010-10-20", 2.50), ("2010-12-26", 2.75), ("2011-02-09", 3.00),
    ("2011-04-06", 3.25), ("2011-07-07", 3.50), ("2012-06-08", 3.25), ("2012-07-06", 3.00),
    ("2014-11-22", 2.75), ("2015-03-01", 2.50), ("2015-05-11", 2.25), ("2015-06-28", 2.00),
    ("2015-08-26", 1.75), ("2015-10-24", 1.50),
]

# Cash is the best of a deposit and a money fund; a money fund is 3M SHIBOR less its fee load.
MONEY_FUND_FEE = 0.0040
SHIBOR_CACHE = ONLINE / "yields" / "shibor_3m.csv"

# Constant income yields added to price-only indices.  Listed in ``panel.json`` so a reader can
# re-run with other numbers instead of guessing what was assumed.
INCOME = {"ashare": 0.022, "usshare": 0.018, "hk": 0.035, "gold": 0.0}

# Curve used for each duration bucket, and the modified duration of that bucket.
BOND_DURATIONS = {"bond2": (2.0, "y2"), "bond5": (5.0, "y5"), "bond9": (9.0, "y10"),
                  "bond20": (20.0, "y30")}

CALENDAR_FILE = ONLINE / "index_long" / "000300.csv"

# The treasury curve starts 2005-01-03, so every asset has a clean history from here.  Starting
# earlier would silently book the missing-yield years as zero-return bonds.
SAMPLE_START = "2005-01-04"


def _read_series(path: Path, date_col: str = "date", close_col: str = "close") -> pd.Series:
    frame = pd.read_csv(path)
    frame[date_col] = pd.to_datetime(frame[date_col])
    series = frame.set_index(date_col)[close_col].astype(float)
    return series[~series.index.duplicated(keep="last")].sort_index()


def calendar(online: Path | None = None) -> pd.DatetimeIndex:
    """The A-share trading calendar, from the index that has the longest history."""
    online = online or ONLINE
    frame = pd.read_csv(online / "index_long" / "000300.csv", usecols=["date"])
    dates = pd.to_datetime(frame["date"])
    return pd.DatetimeIndex(dates.drop_duplicates().sort_values())


def curve(online: Path | None = None) -> pd.DataFrame:
    """China treasury yields in percent, 2Y/5Y/10Y/30Y, on their own print dates.

    The file's first column is the date and the next four are the tenors; taking ``iloc[:, :4]``
    here would silently shift every tenor one column to the left, so the date is split off by
    position rather than by name.
    """
    online = online or ONLINE
    frame = pd.read_csv(online / "yields" / "china_treasury.csv", encoding="utf-8-sig")
    values = frame.iloc[:, 1:5].astype(float)
    values.columns = ["y2", "y5", "y10", "y30"]
    values.index = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0]))
    values = values[~values.index.duplicated(keep="last")].sort_index()
    return values


def yields(online: Path | None = None) -> pd.DataFrame:
    """Treasury yields aligned to the trading calendar, forward filled only."""
    online = online or ONLINE
    return curve(online).reindex(calendar(online), method="ffill").ffill()


def daily_factor(rate: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Compound a *simple annual* rate over the actual calendar days between observations."""
    rate = rate.reindex(index, method="ffill")
    days = pd.Series(index, index=index).diff().dt.days.fillna(1.0)
    return (1.0 + rate.clip(lower=0.0)) ** (days / 365.0)


def deposit_rate(index: pd.DatetimeIndex) -> pd.Series:
    steps = pd.Series({pd.Timestamp(stamp): value / 100.0 for stamp, value in DEPOSIT_STEPS})
    return steps.sort_index().reindex(index, method="ffill")


def shibor(online: Path | None = None, refresh: bool = False) -> pd.Series:
    """Three-month SHIBOR, cached on disk because the network is not always there.

    Only 2015-05 onward exists at the source, so the cash sleeve before that is the deposit
    benchmark alone -- stated in the coverage table rather than papered over.
    """
    online = online or ONLINE
    path = online / "yields" / "shibor_3m.csv"
    if path.exists() and not refresh:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date")["m3"].astype(float).sort_index()
    import akshare as ak
    frame = ak.macro_china_shibor_all()
    frame = frame.iloc[:, [0, 9]]
    frame.columns = ["date", "m3"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date")["m3"].astype(float).sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.rename("m3").to_csv(path, index_label="date", encoding="utf-8")
    return frame


def cash_rate(index: pd.DatetimeIndex, online: Path | None = None,
              refresh: bool = False) -> pd.Series:
    """The rate a retail investor can actually earn on idle cash, before any lockup.

    ``max(deposit benchmark, 3M SHIBOR - fee)``: money funds beat the administered deposit rate
    for most of 2015-2019 and lose to it after 2022, and an investor moves between the two.
    """
    base = deposit_rate(index)
    try:
        money = shibor(online, refresh=refresh).reindex(index, method="ffill") / 100.0
        money = money - MONEY_FUND_FEE
        rate = pd.concat([base, money], axis=1).max(axis=1, skipna=True)
    except Exception:                                             # noqa: BLE001 - offline is normal
        rate = base
    return rate


def bond_levels(yields_frame: pd.DataFrame) -> pd.DataFrame:
    """Total-return levels for each duration bucket, from the treasury curve.

    ``r = y_prev/252 - D*dy + 0.5*C*dy^2`` with ``dy`` in decimal: coupon accrual plus the
    duration and convexity terms.  No roll-down and no credit spread -- both stated in the notes.
    """
    out = {}
    for name, (duration, column) in BOND_DURATIONS.items():
        y = yields_frame[column].astype(float)
        decimal = y / 100.0
        step = decimal.diff()
        convexity = duration * (duration + 1.0) / (1.0 + decimal) ** 2
        ret = decimal.shift(1) / 252.0 - duration * step + 0.5 * convexity * step ** 2
        out[name] = (1.0 + ret.fillna(0.0)).cumprod()
    return pd.DataFrame(out)


def income_level(index: pd.DatetimeIndex, annual: float) -> pd.Series:
    """A constant total-return uplift, compounded over elapsed calendar days.

    ``cumsum``, not ``cumprod``: the day counts are *increments*, and compounding them as a
    product drove the factor to 1.0 and silently deleted two points a year of dividend income
    from every equity series.
    """
    days = pd.Series(index, index=index).diff().dt.days.fillna(1.0)
    return (1.0 + annual) ** (days / 365.0).cumsum()


def known_today(series: pd.Series, index: pd.DatetimeIndex, lag: int = 1) -> pd.Series:
    """Reindex an overseas series onto the Chinese calendar and hold it back ``lag`` sessions.

    The S&P close for date *t* prints after the Chinese close for *t*, so a book marked at the
    Chinese close can only know the *previous* US close.  One session of lag removes the
    look-ahead that would otherwise make every cross-market rule look better than it is.
    """
    aligned = series.reindex(index.union(series.index)).sort_index().ffill().reindex(index)
    return aligned.shift(lag)


def panel(online: Path | None = None, start: str | None = SAMPLE_START,
          end: str | None = None) -> pd.DataFrame:
    """Total-return levels, base 1.0 at the first row, on the A-share calendar."""
    online = online or ONLINE
    index = calendar(online)
    index = index[(index >= pd.Timestamp(start)) if start else slice(None)]
    index = index[(index <= pd.Timestamp(end)) if end else slice(None)]

    out = pd.DataFrame(index=index)
    cash = cash_rate(index, online)
    out["cash"] = daily_factor(cash, index).cumprod()
    # A laddered deposit book earns a twelve-month moving average of the administered rate: it
    # lags a rate rise and cushions a rate cut, which is exactly what a 12-month lockup does.
    ladder = deposit_rate(index).rolling("365D").mean().fillna(deposit_rate(index))
    out["deposit"] = daily_factor(ladder, index).cumprod()

    bonds = bond_levels(yields(online)).reindex(index)
    for name in BOND_DURATIONS:
        out[name] = bonds[name].ffill()

    for name, path, income in (("ashare", online / "index_long" / "000300.csv", "ashare"),
                               ("usshare", online / "global" / "sp500.csv", "usshare"),
                               ("hk", online / "global" / "hsi.csv", "hk")):
        if not path.exists():
            out[name] = np.nan
            continue
        prices = _read_series(path)
        prices = prices.reindex(index.union(prices.index)).sort_index().ffill().reindex(index)
        if name != "ashare":
            prices = known_today(prices, index)
        out[name] = prices * income_level(index, INCOME[income])

    usd = _fx(index, online)
    out["usshare"] = out["usshare"] * usd

    gold_path = online / "global" / "gold.csv"
    if gold_path.exists():
        gold = _read_series(gold_path)
        out["gold"] = gold.reindex(index.union(gold.index)).sort_index().ffill().reindex(index)
    else:
        out["gold"] = np.nan

    out = out.dropna(how="all")
    # Each column is rebased on *its own* first observation: the assets do not all start on the
    # same day, and dividing by row zero would turn a late starter (gold, 2016) into all-NaN.
    out = out.apply(lambda column: column / column.dropna().iloc[0])
    return out


def _fx(index: pd.DatetimeIndex, online: Path) -> pd.Series:
    """USD/CNY (the BOC mid-price file is quoted per 100), carried flat past its last print."""
    path = online / "global" / "usdcny.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = ["date", "buy", "cash_buy", "cash_sell", "sell", "mid", "convert"][:len(frame.columns)]
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["mid"].astype(float) / 100.0
    series = series[~series.index.duplicated(keep="last")].sort_index()
    aligned = series.reindex(index.union(series.index)).sort_index().ffill().reindex(index)
    return aligned.ffill()


def check_bond_model(online: Path | None = None) -> dict:
    """Protocol check v1.1 (amended before any strategy result): level only, plus the evidence.

    The original criterion also demanded a 0.85 correlation with ``000012``.  That criterion was
    withdrawn because the index is the suspect, not the model: its daily return is only 0.27
    correlated with the ten-year yield change, its implied duration is 0.6, and its worst month
    (-5.6%, 2004-04) is many times what that duration allows.  A thinly traded, stale-priced
    index cannot validate a duration model, and matching it would import its staleness.  The
    evidence is returned so a reader can disagree with the amendment on the numbers.
    """
    online = online or ONLINE
    levels = bond_levels(yields(online))
    cash_index = calendar(online)
    synthetic = levels["bond5"].reindex(cash_index).ffill().dropna()
    traded = _read_series(online / "index_long" / "000012.csv").dropna()
    window = synthetic.index.intersection(traded.index)
    window = window[window >= pd.Timestamp(SAMPLE_START)]
    synthetic = synthetic.reindex(window).ffill()
    traded = traded.reindex(window).ffill()
    years = (window[-1] - window[0]).days / 365.25
    cagr_synthetic = float((synthetic.iloc[-1] / synthetic.iloc[0]) ** (1 / years) - 1)
    cagr_traded = float((traded.iloc[-1] / traded.iloc[0]) ** (1 / years) - 1)
    monthly = pd.concat([synthetic.resample("M").last().pct_change(),
                         traded.resample("M").last().pct_change()], axis=1).dropna()
    corr = float(monthly.iloc[:, 0].corr(monthly.iloc[:, 1]))
    gap = abs(cagr_synthetic - cagr_traded)

    # Is the traded index actually a duration instrument?  Daily sensitivity to the ten-year.
    daily_traded = traded.pct_change()
    daily_step = curve(online)["y10"].diff()
    pairs = pd.concat([daily_traded, daily_step], axis=1).dropna()
    pairs.columns = ["ret", "dy"]
    pairs = pairs[pairs["dy"].abs() <= 0.20]                       # trim obvious data errors
    if float(pairs["dy"].std()) == 0.0:                            # a flat curve carries no signal
        implied_duration, daily_corr = 0.0, 0.0
    else:
        slope = float(np.polyfit(pairs["dy"], pairs["ret"], 1)[0])
        implied_duration = -slope * 100.0                # a long bond: yield up, price down
        daily_corr = float(pairs["ret"].corr(-pairs["dy"]))
    worst_month = float(_read_series(online / "index_long" / "000012.csv").resample("M").last().pct_change().min())

    return dict(synthetic_cagr=round(cagr_synthetic, 4), traded_cagr=round(cagr_traded, 4),
                gap_pp=round(gap * 100, 2), monthly_corr=round(corr, 4),
                traded_implied_duration=round(implied_duration, 2),
                traded_daily_corr=round(daily_corr, 3),
                traded_worst_month=round(worst_month, 4),
                pass_=bool(gap <= 0.008), start=str(window[0].date()),
                end=str(window[-1].date()),
                amended="v1.1: correlation criterion withdrawn, level check kept")


def coverage(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for name in frame.columns:
        series = frame[name].dropna()
        rows.append(dict(asset=name, first=str(series.index[0].date()),
                         last=str(series.index[-1].date()), n=int(len(series)),
                         cagr=round(float((series.iloc[-1] / series.iloc[0])
                                          ** (365.25 / max((series.index[-1] - series.index[0]).days, 1))
                                          - 1), 4)))
    return rows


def assumptions() -> dict:
    return dict(deposit_steps=len(DEPOSIT_STEPS),
                deposit_note="official one-year benchmark path, hand-entered; banks pay less since 2016",
                money_fund_fee=MONEY_FUND_FEE,
                income_yields=INCOME,
                bond_model="y_prev/252 - D*dy + 0.5*C*dy^2 from the treasury curve",
                bond_durations={k: v[0] for k, v in BOND_DURATIONS.items()},
                overseas_lag="one Chinese session",
                fx="BOC mid-price per 100 USD, carried flat after its last print",
                gold="Shanghai gold in CNY per gram, from 2016-12-19")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    frame = panel()
    verdict = check_bond_model()
    rows = coverage(frame)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "panel.csv", index_label="date", float_format="%.6f")
    summary = dict(generated_at=str(pd.Timestamp.now().date()), rows=len(frame),
                   start=str(frame.index[0].date()), end=str(frame.index[-1].date()),
                   coverage=rows, bond_model=verdict, assumptions=assumptions())
    (OUTPUT / "panel.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"panel: {len(frame)} rows {frame.index[0].date()} -> {frame.index[-1].date()}")
    for row in rows:
        print(f"  {row['asset']:<8} {row['first']} -> {row['last']}  n={row['n']:>5}  cagr={row['cagr']:+.2%}")
    print("bond model check:", json.dumps(verdict, ensure_ascii=False))
    if not verdict["pass_"]:
        print("  WARNING: synthetic bonds do not reproduce 000012; the strategy layer must use the index.")
    print("wrote", OUTPUT / "panel.csv")
    return 0 if verdict["pass_"] else 2


if __name__ == "__main__":
    sys.exit(main())


