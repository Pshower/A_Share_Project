"""Auditable macro snapshots with conservative release availability."""

from datetime import date, datetime
import argparse
import json
from pathlib import Path
import re
import time

import pandas as pd

from src.data.online import CHINA, retry
from src.data.preprocess import ROOT
from src.data.rate_limit import AdaptivePacer
from src.data.snapshots import Snapshots, utc_now
from src.runtime import events

SPECS = {
    "pmi": dict(api="macro_china_pmi", source="eastmoney", date="月份", frequency="monthly",
                lag=1, metrics={"制造业-指数": ("pmi_manufacturing", "index", 1),
                                "非制造业-指数": ("pmi_non_manufacturing", "index", 1)}),
    "cpi": dict(api="macro_china_cpi", source="eastmoney", date="月份", frequency="monthly",
                lag=10, metrics={"全国-同比增长": ("cpi_yoy", "percent", 1),
                                 "全国-环比增长": ("cpi_mom", "percent", 1)}),
    "ppi": dict(api="macro_china_ppi", source="eastmoney", date="月份", frequency="monthly",
                lag=10, metrics={"当月同比增长": ("ppi_yoy", "percent", 1)}),
    "money_supply": dict(api="macro_china_money_supply", source="eastmoney", date="月份", frequency="monthly",
                         lag=15, metrics={"货币和准货币(M2)-数量(亿元)": ("m2_level", "100m_cny", 1),
                                          "货币和准货币(M2)-同比增长": ("m2_yoy", "percent", 1),
                                          "货币(M1)-同比增长": ("m1_yoy", "percent", 1)}),
    "credit": dict(api="macro_china_new_financial_credit", source="eastmoney", date="月份",
                   frequency="monthly", lag=15,
                   metrics={"当月": ("new_rmb_loan", "100m_cny", 1),
                            "当月-同比增长": ("new_rmb_loan_yoy", "percent", 1)}),
    "social_financing": dict(api="macro_china_shrzgm", source="eastmoney", date="月份",
                             frequency="monthly", lag=15,
                             metrics={"社会融资规模增量": ("social_financing", "100m_cny", 1),
                                      "其中-人民币贷款": ("social_financing_rmb_loan", "100m_cny", 1)}),
    "gdp": dict(api="macro_china_gdp", source="eastmoney", date="季度", frequency="quarterly",
                lag=20, metrics={"国内生产总值-绝对值": ("gdp_level", "100m_cny", 1),
                                 "国内生产总值-同比增长": ("gdp_yoy", "percent", 1)}),
    "industrial_production": dict(api="macro_china_gyzjz", source="eastmoney", date="月份",
                                  frequency="monthly", lag=20,
                                  metrics={"同比增长": ("industrial_production_yoy", "percent", 1),
                                           "累计增长": ("industrial_production_ytd", "percent", 1)}),
    "trade": dict(api="macro_china_hgjck", source="eastmoney", date="月份", frequency="monthly",
                  lag=15, metrics={"当月出口额-同比增长": ("exports_yoy", "percent", 1),
                                   "当月进口额-同比增长": ("imports_yoy", "percent", 1)}),
    "retail": dict(api="macro_china_consumer_goods_retail", source="eastmoney", date="月份",
                   frequency="monthly", lag=20,
                   metrics={"同比增长": ("retail_yoy", "percent", 1),
                            "累计-同比增长": ("retail_ytd", "percent", 1)}),
    "investment": dict(api="macro_china_gdzctz", source="eastmoney", date="月份",
                       frequency="monthly", lag=20,
                       metrics={"同比增长": ("fixed_asset_investment_yoy", "percent", 1)}),
    "lpr": dict(api="macro_china_lpr", source="eastmoney", date="TRADE_DATE", frequency="daily",
                lag=1, metrics={"LPR1Y": ("lpr_1y", "percent", 1),
                                "LPR5Y": ("lpr_5y", "percent", 1)}),
    "bond_yield": dict(api="bond_zh_us_rate", source="eastmoney", date="日期", frequency="daily",
                       lag=1, metrics={"中国国债收益率10年": ("cn_10y_yield", "percent", 1),
                                       "中国国债收益率10年-2年": ("cn_10y_2y_slope", "percent", 1),
                                       "美国国债收益率10年": ("us_10y_yield", "percent", 1)}),
    "shibor": dict(api="macro_china_shibor_all", source="jin10", date="日期", frequency="daily",
                   lag=1, metrics={"O/N-定价": ("shibor_on", "percent", 1),
                                   "1W-定价": ("shibor_1w", "percent", 1),
                                   "3M-定价": ("shibor_3m", "percent", 1)}),
    "usdcny": dict(api="currency_boc_safe", source="safe", date="日期", frequency="daily",
                   lag=1, metrics={"美元": ("usdcny", "CNY_per_USD", 0.01)}),
}


class MacroProvider:
    name = "macro"

    def __init__(self, root=ROOT):
        self.pacer = AdaptivePacer(Path(root) / "data/online/providers/macro")

    def fetch(self, api):
        self.pacer.acquire()
        import akshare as ak
        return getattr(ak, api)()


def _month_end(year, month):
    return pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)


def _quarter_end(year, quarter):
    return _month_end(year, quarter * 3)


def parse_period(value):
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).normalize()
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})年(\d{1,2})月份?", text)
    if match:
        return _month_end(int(match.group(1)), int(match.group(2)))
    match = re.fullmatch(r"(\d{4})年第(\d)(?:-(\d))?季度", text)
    if match:
        return _quarter_end(int(match.group(1)), int(match.group(3) or match.group(2)))
    match = re.fullmatch(r"(\d{4})(\d{2})", text)
    if match:
        return _month_end(int(match.group(1)), int(match.group(2)))
    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if match:
        return _month_end(int(match.group(1)), int(match.group(2)))
    match = re.fullmatch(r"(\d{4})年?", text)
    if match:
        return _month_end(int(match.group(1)), 12)
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed.normalize() if not pd.isna(parsed) else pd.NaT


def normalize_macro(raw, spec, start, end):
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Provider macro table is empty")
    date_column = spec["date"]
    if date_column not in raw:
        raise ValueError("Provider macro date column changed")
    frame = raw.copy()
    frame["period"] = frame[date_column].map(parse_period)
    invalid = int(frame["period"].isna().sum())
    if invalid:
        raise ValueError(f"Provider macro table has {invalid} invalid periods")
    missing = [column for column in spec["metrics"] if column not in frame]
    if missing:
        raise ValueError("Provider macro metric columns changed")
    for column, (_, _, factor) in spec["metrics"].items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor
    frame = frame.dropna(subset=["period"]).sort_values("period")
    frame = frame.drop_duplicates("period", keep="last")
    frame["available_from"] = (frame["period"] + pd.Timedelta(days=spec["lag"])).dt.strftime("%Y-%m-%d")
    frame["period"] = frame["period"].dt.strftime("%Y-%m-%d")
    frame = frame[frame.period.between(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))]
    frame = frame[["period", "available_from"] + list(spec["metrics"])].dropna(
        subset=list(spec["metrics"]), how="all").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No macro rows in requested range")
    return frame


def to_long(frame, spec):
    rows = []
    for column, (series_id, unit, factor) in spec["metrics"].items():
        part = frame[["period", "available_from", column]].copy()
        part["value"] = pd.to_numeric(part[column], errors="coerce")
        part = part.dropna(subset=["value"])
        part["series_id"] = series_id
        part["metric"] = column
        part["unit"] = unit
        part["frequency"] = spec["frequency"]
        part["source"] = spec["source"]
        part["api"] = spec["api"]
        rows.append(part.drop(columns=[column]))
    return pd.concat(rows, ignore_index=True)


def capture_macro(start="2017-01-01", end=None, *, allow_network=False, root=ROOT, provider=None, series=None):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    selected = list(SPECS) if series is None else list(series)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(SPECS):
        raise ValueError("Unknown or duplicate macro series")
    end = str(datetime.now(CHINA).date()) if end is None else str(end)
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date or end_date.date() > datetime.now(CHINA).date():
        raise ValueError("Invalid or future macro range")
    provider = provider or MacroProvider(root)
    store = Snapshots(root)
    identifier, directory = store.begin("macro")
    deadline = time.monotonic() + 600
    entries, errors, frames = [], [], {}
    for key in selected:
        spec = SPECS[key]
        try:
            events.check_cancel()
            raw = retry(lambda: provider.fetch(spec["api"]),
                        limiter=getattr(provider, "pacer", None), deadline=deadline,
                        context=dict(series=key, api=spec["api"]))
            frame = normalize_macro(raw, spec, start_date, end_date)
            frames[key] = frame
            directory.joinpath("series").mkdir(exist_ok=True)
            frame.to_parquet(directory / "series" / f"{key}.parquet", index=False)
            frame.to_csv(directory / "series" / f"{key}.csv", index=False, encoding="utf-8-sig")
            entries.append(dict(series=key, api=spec["api"], source=spec["source"],
                                frequency=spec["frequency"], status="ready", rows=len(frame),
                                first_period=str(frame.period.iloc[0]),
                                last_period=str(frame.period.iloc[-1]),
                                last_available_from=str(frame.available_from.iloc[-1]),
                                metrics=list(spec["metrics"])))
        except events.Cancelled:
            raise
        except Exception as error:
            item = dict(series=key, api=spec["api"], source=spec["source"],
                        frequency=spec["frequency"], status="failed", rows=0,
                        error=str(error)[:500])
            entries.append(item)
            errors.append(dict(series=key, error=item["error"]))
        events.emit("macro_progress", completed=len(entries), total=len(selected), series=key)

    long_frames = [to_long(frame, SPECS[key]) for key, frame in frames.items()]
    if long_frames:
        combined = pd.concat(long_frames, ignore_index=True).sort_values(["series_id", "period"])
        combined.to_parquet(directory / "macro.parquet", index=False)
        combined.to_csv(directory / "macro.csv", index=False, encoding="utf-8-sig")
        coverage = dict(rows=len(combined), series=int(combined.series_id.nunique()),
                        first_period=str(combined.period.min()), last_period=str(combined.period.max()),
                        last_available_from=str(combined.available_from.max()))
    else:
        coverage = dict(rows=0, series=0)
    result = store.publish("macro", identifier, directory, dict(
        request=dict(provider="mixed", start_date=str(start_date.date()), end_date=str(end_date.date()),
                     series=selected, specs={key: dict(api=spec["api"], source=spec["source"],
                                                          frequency=spec["frequency"], lag_days=spec["lag"],
                                                          units={metric: spec["metrics"][metric][1]
                                                                 for metric in spec["metrics"]})
                                                for key, spec in SPECS.items()}),
        entries=entries, errors=errors, coverage=coverage,
        status="ready" if not errors else "partial", captured_at=utc_now(),
        model_ready=False,
        release_policy="Conservative fixed lag after period end; not an official release calendar",
        timestamp_semantics="period end plus conservative availability lag; not real-time release timestamps",
        revision_policy="Latest provider snapshot only; historical vintages are not preserved",
        unit_policy="Provider units retained, except SAFE USD quote converted from CNY per 100 USD to CNY per USD"))
    events.emit("download_snapshot", snapshot_id=identifier, snapshot_kind="macro", status=result["status"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=list(SPECS))
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=str(datetime.now(CHINA).date()))
    parser.add_argument("--allow-network", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(capture_macro(args.start, args.end, allow_network=args.allow_network, series=args.only),
                     ensure_ascii=True, indent=2))
