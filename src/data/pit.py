"""PIT trading calendar, quarterly financial reports and shareholder-count snapshots."""

from datetime import datetime
import argparse
import json
from pathlib import Path
import re
import time

import pandas as pd

from src.data.online import CHINA, retry, valid_codes
from src.data.preprocess import ROOT, write_json
from src.data.rate_limit import AdaptivePacer
from src.data.snapshots import Snapshots, utc_now
from src.runtime import events

CALENDAR_PROVIDER = "sina"
PIT_PROVIDER = "eastmoney"
QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))
FINANCIAL_DATE = "最新公告日期"
FORECAST_DATE = "公告日期"
HOLDER_DATE = "公告日期"


class CalendarProvider:
    name = CALENDAR_PROVIDER

    def __init__(self, root=ROOT):
        self.pacer = AdaptivePacer(Path(root) / "data/online/providers/sina")

    def trade_dates(self):
        self.pacer.acquire()
        import akshare as ak
        return ak.tool_trade_date_hist_sina()


class PitProvider:
    name = PIT_PROVIDER

    def __init__(self, root=ROOT):
        self.pacer = AdaptivePacer(Path(root) / "data/online/providers/eastmoney")

    def _call(self, function, **kwargs):
        self.pacer.acquire()
        import akshare as ak
        return getattr(ak, function)(**kwargs)

    def financials(self, period):
        return self._call("stock_yjbb_em", date=period)

    def forecasts(self, period):
        return self._call("stock_yjyg_em", date=period)

    def shareholders(self, period):
        return self._call("stock_zh_a_gdhs", symbol=period)


def _period(period):
    if not re.fullmatch(r"[0-9]{8}", str(period)):
        raise ValueError("Report periods must use YYYYMMDD")
    value = pd.Timestamp(str(period))
    if value.strftime("%Y%m%d") != str(period) or (value.month, value.day) not in QUARTER_ENDS:
        raise ValueError("Report periods must be quarter ends")
    return str(period)


def report_periods(start="2017-01-01", end=None):
    end = str(datetime.now(CHINA).date()) if end is None else str(end)
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date or end_date.date() > datetime.now(CHINA).date():
        raise ValueError("Invalid or future reporting range")
    periods = []
    for year in range(start_date.year, end_date.year + 1):
        for month, day in QUARTER_ENDS:
            candidate = pd.Timestamp(year=year, month=month, day=day)
            if start_date <= candidate <= end_date:
                periods.append(candidate.strftime("%Y%m%d"))
    if not periods:
        raise ValueError("No quarter-end report periods in the requested range")
    return periods


def normalize_calendar(raw):
    if not isinstance(raw, pd.DataFrame) or "trade_date" not in raw or raw.empty:
        raise ValueError("Trading calendar is empty or missing trade_date")
    dates = pd.to_datetime(raw["trade_date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Trading calendar contains invalid or duplicate dates")
    return pd.DataFrame({"trade_date": dates.dt.strftime("%Y-%m-%d")}).sort_values("trade_date").reset_index(drop=True)


def _codes(series):
    values = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if (values.str.fullmatch(r"[0-9]{6}") == False).any():
        raise ValueError("Provider contains an invalid stock code")
    return values


def _announcement(series):
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%Y-%m-%d")


def _normalize_table(raw, *, period, code_column, date_column, codes=None, unique_keys=("code",), dedupe=False):
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(dict(period=[], code=[], announcement_date=[])), 0, 0
    if not {code_column, date_column} <= set(raw.columns):
        raise ValueError("Provider table schema changed")
    frame = raw.copy()
    frame["code"] = _codes(frame[code_column])
    frame["period"] = _period(period)
    frame["announcement_date"] = _announcement(frame[date_column])
    if codes is not None:
        frame = frame[frame.code.isin(codes)].copy()
    keys = ["code"] + [column for column in unique_keys if column != "code"]
    if not set(keys) <= set(frame.columns):
        raise ValueError("Provider table is missing a unique-key column")
    duplicates = 0
    if frame.duplicated(keys).any():
        if not dedupe:
            raise ValueError("Provider table contains duplicate keys")
        before = len(frame)
        frame = (frame.sort_values(keys + ["announcement_date"])
                 .drop_duplicates(keys, keep="last"))
        duplicates = before - len(frame)
    missing = int(frame["announcement_date"].isna().sum())
    return frame.reset_index(drop=True), missing, duplicates


def _entry(frame, missing, duplicates=0):
    dates = frame["announcement_date"].dropna()
    return dict(status="ready" if len(frame) else "empty", rows=len(frame),
                unique_codes=int(frame["code"].nunique()) if len(frame) else 0,
                announcement_missing=missing, duplicates_removed=duplicates,
                first_announcement=str(dates.min()) if len(dates) else None,
                last_announcement=str(dates.max()) if len(dates) else None)


def _announcement_range(frame):
    dates = frame["announcement_date"].dropna()
    return (str(dates.min()), str(dates.max())) if len(dates) else (None, None)


def _write_table(directory, name, frame):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    return path.with_suffix(".parquet")


def capture_calendar(*, allow_network=False, root=ROOT, provider=None, source=CALENDAR_PROVIDER):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    if source != CALENDAR_PROVIDER:
        raise ValueError("Trading calendar supports Sina only")
    provider = provider or CalendarProvider(root)
    if getattr(provider, "name", source) != source:
        raise ValueError("Calendar provider differs from declared source")
    store = Snapshots(root)
    identifier, directory = store.begin("calendar")
    deadline = time.monotonic() + 120
    frame = normalize_calendar(retry(lambda: provider.trade_dates(),
                                     limiter=getattr(provider, "pacer", None), deadline=deadline,
                                     context=dict(kind="calendar")))
    _write_table(directory, "calendar", frame)
    today = str(datetime.now(CHINA).date())
    result = store.publish("calendar", identifier, directory, dict(
        request=dict(provider=source), status="ready", rows=len(frame),
        first_date=str(frame.trade_date.iloc[0]), last_date=str(frame.trade_date.iloc[-1]),
        future_dates=int((frame.trade_date > today).sum()), captured_at=utc_now(),
        model_ready=False, timestamp_semantics="Trading dates only; no intraday session detail",
        limitation="Calendar is reference data; it does not establish that a vendor bar is complete"))
    events.emit("download_snapshot", snapshot_id=identifier, snapshot_kind="calendar", status=result["status"])
    return result


def capture_fundamentals(codes=None, start="2017-01-01", end=None, *,
                         include_forecasts=True, include_holders=True, allow_network=False,
                         root=ROOT, provider=None, source=PIT_PROVIDER):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    if source != PIT_PROVIDER:
        raise ValueError("Financial and shareholder data currently support Eastmoney only")
    selected = None if not codes else list(valid_codes(list(codes)))
    periods = report_periods(start, end)
    provider = provider or PitProvider(root)
    if getattr(provider, "name", source) != source:
        raise ValueError("PIT provider differs from declared source")
    store = Snapshots(root)
    identifier, directory = store.begin("fundamentals")
    deadline = time.monotonic() + max(900, len(periods) * 3 * 15)
    entries, errors = [], []
    financial_frames, forecast_frames, holder_frames = [], [], []
    cancelled = False

    def capture(period, kind, function):
        nonlocal cancelled
        try:
            events.check_cancel()
            raw = retry(lambda: function(period), limiter=getattr(provider, "pacer", None),
                        deadline=deadline, context=dict(period=period, kind=kind))
            column = {"financials": FINANCIAL_DATE, "forecasts": FORECAST_DATE, "shareholders": HOLDER_DATE}[kind]
            code_column = "代码" if kind == "shareholders" else "股票代码"
            unique_keys = ("code", "预测指标") if kind == "forecasts" else ("code",)
            frame, missing, duplicates = _normalize_table(raw, period=period, code_column=code_column,
                                                          date_column=column, codes=selected,
                                                          unique_keys=unique_keys,
                                                          dedupe=kind == "forecasts")
            _write_table(directory, f"{kind}/{period}", frame)
            return frame, _entry(frame, missing, duplicates)
        except events.Cancelled:
            cancelled = True
            return None, dict(status="failed", rows=0, error="Cancelled during capture")
        except Exception as error:
            item = dict(status="failed", rows=0, error=str(error)[:500])
            errors.append(dict(period=period, kind=kind, error=item["error"]))
            return None, item

    for period in periods:
        entry = dict(period=period)
        frame, entry["financials"] = capture(period, "financials", provider.financials)
        if frame is not None:
            financial_frames.append(frame)
        if include_forecasts:
            frame, entry["forecasts"] = capture(period, "forecasts", provider.forecasts)
            if frame is not None:
                forecast_frames.append(frame)
        if include_holders:
            frame, entry["shareholders"] = capture(period, "shareholders", provider.shareholders)
            if frame is not None:
                holder_frames.append(frame)
        entries.append(entry)
        events.emit("pit_progress", period=period, completed=len(entries), total=len(periods),
                    financial_rows=entry["financials"].get("rows", 0))

    combined = {}
    for name, frames in (("financials", financial_frames), ("forecasts", forecast_frames),
                         ("shareholders", holder_frames)):
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            _write_table(directory, name, merged)
            first, last = _announcement_range(merged)
            combined[name] = dict(rows=len(merged), periods=len(frames),
                                  unique_codes=int(merged["code"].nunique()),
                                  first_announcement=first, last_announcement=last)
        else:
            combined[name] = dict(rows=0, periods=0)

    result = store.publish("fundamentals", identifier, directory, dict(
        request=dict(provider=source, codes=selected or [], start_date=str(pd.Timestamp(start).date()),
                     end_date=str(pd.Timestamp(end or datetime.now(CHINA).date()).date()), periods=periods,
                     include_forecasts=include_forecasts, include_holders=include_holders),
        entries=entries, combined=combined, status="ready" if not errors else "partial",
        errors=errors, captured_at=utc_now(), model_ready=False,
        pit_policy="Use announcement_date, never period end. Values are the vendor's latest revision; "
                   "original report vintage is not preserved. Shift availability to the next trading day.",
        timestamp_semantics="Date-only announcement fields; no intraday release time",
        calendar_dependency="Join with a calendar snapshot before constructing features"))
    events.emit("download_snapshot", snapshot_id=identifier, snapshot_kind="fundamentals", status=result["status"])
    if cancelled:
        raise events.Cancelled("PIT capture stopped; completed files are saved in snapshot " + identifier)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["calendar", "fundamentals"], required=True)
    parser.add_argument("--codes", nargs="*", default=[])
    parser.add_argument("--codes-file", help="UTF-8 text/CSV containing six-digit stock codes")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=str(datetime.now(CHINA).date()))
    parser.add_argument("--skip-forecasts", action="store_true")
    parser.add_argument("--skip-holders", action="store_true")
    parser.add_argument("--allow-network", action="store_true", required=True)
    args = parser.parse_args()
    codes = list(dict.fromkeys(args.codes))
    if args.codes_file:
        path = Path(args.codes_file)
        if path.stat().st_size > 200000:
            parser.error("Stock code file exceeds 200 KB")
        codes = list(dict.fromkeys(codes + re.findall(r"(?<!\d)\d{6}(?!\d)", path.read_text(encoding="utf-8-sig"))))
    result = (capture_calendar(allow_network=args.allow_network)
              if args.kind == "calendar" else
              capture_fundamentals(codes or None, args.start, args.end,
                                   include_forecasts=not args.skip_forecasts,
                                   include_holders=not args.skip_holders,
                                   allow_network=args.allow_network))
    print(json.dumps(result, ensure_ascii=True, indent=2))
