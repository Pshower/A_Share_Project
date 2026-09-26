"""Explicit auxiliary market captures, isolated from model-ready daily data."""

from datetime import datetime
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import pandas as pd

from src.data.online import Eastmoney, CHINA, valid_codes
from src.data.preprocess import ROOT
from src.data.snapshots import Snapshots, utc_now
from src.runtime import events

TYPES = {
    "profile": dict(label="个股资料", api="stock_individual_info_em", provider="eastmoney", required=["item", "value"], date_column=None, note="采集时资料快照，不能回填历史基本面"),
    "fund_flow": dict(label="个股资金流", api="stock_individual_fund_flow", provider="eastmoney", required=["日期", "主力净流入-净额"], date_column="日期", note="近 100 个交易日；供应商分类统计"),
    "minute": dict(label="分钟行情", api="stock_zh_a_hist_min_em", provider="eastmoney", required=["时间", "收盘", "成交量"], date_column="时间", note="近期数据；1 分钟近 5 个交易日且不复权，部分开盘可能为 0"),
    "index_daily": dict(label="指数日线", api="index_zh_a_hist", provider="eastmoney", required=["日期", "开盘", "收盘", "最高", "最低"], date_column="日期", note="指数点位，不是股票成交价格"),
    "dividend": dict(label="分红记录", api="stock_history_dividend_detail", provider="sina", required=["公告日期", "派息"], date_column="公告日期", note="供应商分红记录；税前派息，尚未适配真实账户"),
}


def validate_request(request):
    allowed = {"data_type", "codes", "market", "start_date", "end_date", "period"}
    if set(request) != allowed or request["data_type"] not in TYPES:
        raise ValueError("Unknown supplemental request")
    valid_codes(request["codes"])
    if len(request["codes"]) > 20 or request["market"] not in {"sh", "sz", "bj"} or request["period"] not in {"1", "5", "15", "30", "60"}:
        raise ValueError("Invalid market, period or more than 20 codes")
    start, end = pd.Timestamp(request["start_date"]), pd.Timestamp(request["end_date"])
    if pd.isna(start) or pd.isna(end) or start > end or end.date() > datetime.now(CHINA).date():
        raise ValueError("Invalid or future dates")
    if request["data_type"] == "minute" and (end - start).days > 31:
        raise ValueError("Minute capture is limited to a 31-day request window")


def api_arguments(kind, code, request):
    if kind == "profile":
        return dict(symbol=code, timeout=12)
    if kind == "fund_flow":
        return dict(stock=code, market=request["market"])
    if kind == "minute":
        return dict(symbol=code, start_date=request["start_date"] + " 00:00:00",
                    end_date=request["end_date"] + " 23:59:59", period=request["period"], adjust="")
    if kind == "index_daily":
        return dict(symbol=code, period="daily", start_date=request["start_date"].replace("-", ""),
                    end_date=request["end_date"].replace("-", ""))
    if kind == "dividend":
        return dict(symbol=code, indicator="分红")
    raise ValueError("Unknown capture type")


def _sdk_child(kind, code, request, output):
    def parent_guard():
        parent = mp.parent_process()
        while parent is not None and parent.is_alive():
            time.sleep(0.5)
        os._exit(2)
    threading.Thread(target=parent_guard, daemon=True).start()
    try:
        import akshare as ak
        frame = getattr(ak, TYPES[kind]["api"])(**api_arguments(kind, code, request))
        if not isinstance(frame, pd.DataFrame) or len(frame) > 20000 or len(frame.columns) > 100:
            raise ValueError("Unexpected or oversized provider table")
        payload = frame.to_json(orient="split", date_format="iso", force_ascii=False)
        if len(payload.encode("utf-8")) > 20_000_000:
            raise ValueError("Provider table exceeds 20 MB")
        Path(output).write_text(payload, encoding="utf-8")
    except Exception as error:
        Path(output).write_text(json.dumps(dict(error=str(error)[:500])), encoding="utf-8")


class SupplementProvider:
    def __init__(self, root=ROOT):
        self.pacer = Eastmoney(root)
        self.temp = Path(root) / "data/online/.requests"
        self.temp.mkdir(parents=True, exist_ok=True)

    def fetch(self, kind, code, request):
        self.pacer.pace()
        with tempfile.TemporaryDirectory(dir=self.temp) as directory:
            output = Path(directory) / "response.json"
            process = mp.get_context("spawn").Process(target=_sdk_child, args=(kind, code, request, str(output)), daemon=True)
            process.start()
            started = time.monotonic()
            try:
                while process.is_alive():
                    events.check_cancel()
                    if time.monotonic() - started > 45:
                        raise TimeoutError("SDK call exceeded 45 seconds")
                    process.join(timeout=0.1)
                if not output.exists() or output.stat().st_size > 20_000_000:
                    raise ValueError("SDK worker returned no bounded table")
                data = json.loads(output.read_text(encoding="utf-8"))
                if "error" in data:
                    raise ValueError(data["error"])
                return pd.DataFrame(data["data"], columns=data["columns"])
            finally:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                process.close()


def validate_table(frame, kind, code, request):
    if not isinstance(frame, pd.DataFrame) or len(frame) > 20000 or len(frame.columns) > 100 or frame.columns.has_duplicates:
        raise ValueError("Unexpected provider table")
    if frame.empty:
        return frame, None, None
    if not set(TYPES[kind]["required"]) <= set(frame.columns):
        raise ValueError("Provider table schema changed")
    if kind == "profile":
        identity = frame.loc[frame.item.eq("股票代码"), "value"]
        if len(identity) != 1 or str(identity.iloc[0]).zfill(6) != code:
            raise ValueError("Provider stock identity mismatch")
    date_column = TYPES[kind]["date_column"]
    first = last = None
    if date_column:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        if kind in {"minute", "index_daily", "fund_flow"}:
            if dates.isna().any() or dates.duplicated().any():
                raise ValueError("Invalid or duplicate provider timestamps")
            if kind in {"minute", "index_daily"}:
                selected = dates.between(pd.Timestamp(request["start_date"]), pd.Timestamp(request["end_date"]) + pd.Timedelta(days=1), inclusive="left")
                frame, dates = frame.loc[selected].copy(), dates.loc[selected]
        if dates.notna().any():
            first, last = str(dates.min()), str(dates.max())
    return frame, first, last


def capture_supplement(request, *, allow_network=False, root=ROOT, provider=None, resume_id=None):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    validate_request(request)
    store = Snapshots(root)
    old, old_dir = store.get("supplements", resume_id) if resume_id else ({}, None)
    if old and old["request"] != request:
        raise ValueError("Retry must use the original request")
    kind = request["data_type"]
    identifier, directory = store.begin("supplements")
    previous = {r["code"]: r for r in old.get("entries", [])}
    provider = provider or SupplementProvider(root)
    entries, failures, cancelled = [], 0, False
    started = time.monotonic()
    for code in request["codes"]:
        prior = previous.get(code)
        try:
            if prior and prior["status"] in {"ready", "empty"}:
                for extension in ["json", "csv"]:
                    shutil.copyfile(old_dir / f"{code}.{extension}", directory / f"{code}.{extension}")
                item = dict(prior, reused_from=resume_id)
            else:
                events.check_cancel()
                if cancelled or failures >= 3 or time.monotonic() - started > 1000:
                    raise ValueError("Batch cancelled or circuit/time limit reached")
                frame, first, last = validate_table(provider.fetch(kind, code, request), kind, code, request)
                (directory / f"{code}.json").write_text(frame.to_json(orient="split", date_format="iso", force_ascii=False), encoding="utf-8")
                def csv_text(value):
                    return "'" + value if isinstance(value, str) and (value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n"))) else value
                safe = frame.copy()
                for column in safe.columns:
                    safe[column] = safe[column].map(csv_text)
                safe.columns = [csv_text(c) for c in safe.columns]
                safe.to_csv(directory / f"{code}.csv", index=False, encoding="utf-8-sig")
                item = dict(code=code, status="ready" if len(frame) else "empty", rows=len(frame),
                            columns=list(frame.columns), first_date=first, last_date=last, received_at=utc_now())
                failures = 0
        except events.Cancelled:
            cancelled = True
            item = dict(code=code, status="failed", error="Cancelled before capture completed")
        except Exception as error:
            failures += 1
            item = dict(code=code, status="failed", error=str(error)[:500])
        entries.append(item)
        events.emit("supplement_progress", completed=len(entries), total=len(request["codes"]), code=code,
                    saved=sum(e["status"] == "ready" for e in entries), failed=sum(e["status"] == "failed" for e in entries))
    result = store.publish("supplements", identifier, directory, dict(request=request, entries=entries,
                           status="partial" if any(e["status"] == "failed" for e in entries) else "ready",
                           api=TYPES[kind]["api"], provider=TYPES[kind]["provider"], akshare_version=version("akshare"),
                           unit_policy="Provider fields and units retained; no conversion", timezone="Asia/Shanghai",
                           model_ready=False, history_complete=False, cancelled=cancelled,
                           limitation=TYPES[kind]["note"]))
    events.emit("supplement_snapshot", snapshot_id=identifier)
    if cancelled:
        raise events.Cancelled("Completed supplementary files saved in " + identifier)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=list(TYPES), required=True)
    parser.add_argument("--codes", nargs="+", required=True)
    parser.add_argument("--market", choices=["sh", "sz", "bj"], default="sz")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--period", choices=["1", "5", "15", "30", "60"], default="5")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    result = capture_supplement(dict(data_type=args.type, codes=args.codes, market=args.market,
                                    start_date=args.start, end_date=args.end, period=args.period),
                                allow_network=args.allow_network, resume_id=args.resume)
    print(json.dumps(dict(id=result["id"], status=result["status"], entries=result["entries"]), ensure_ascii=False))
