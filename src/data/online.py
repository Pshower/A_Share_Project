"""Explicit Eastmoney access with bounded requests and auditable snapshots."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.data.preprocess import ROOT, RENAME, file_hash, write_json
from src.data.snapshots import Snapshots, utc_now
from src.runtime import events

CHINA = ZoneInfo("Asia/Shanghai")


def valid_codes(codes):
    if not codes or len(codes) > 500 or len(set(codes)) != len(codes) or any(not re.fullmatch(r"[0-9]{6}", c) for c in codes):
        raise ValueError("Expected unique six-digit stock codes")
    return codes


class Eastmoney:
    name = "eastmoney"

    def __init__(self, root=ROOT):
        from src.data.data_socket import DataFetcher
        self.root = Path(root)
        self.fetcher = DataFetcher(self.root / "data/online/.cache")

    def pace(self):
        events.check_cancel()
        path = self.root / ".workbench/network.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=10) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS clock (id INTEGER PRIMARY KEY, next REAL)")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT next FROM clock WHERE id=1").fetchone()
            slot = max(time.time(), row[0] if row else 0)
            conn.execute("INSERT OR REPLACE INTO clock VALUES(1,?)", (slot + 1,))
        while time.time() < slot:
            events.check_cancel()
            time.sleep(max(0, min(0.2, slot - time.time())))

    def history(self, code, start, end, basis):
        self.pace()
        return self.fetcher.get_stock_hist(code, start.replace("-", ""), end.replace("-", ""),
                    adjust="hfq" if basis == "hfq" else "", use_cache=False, timeout=12, retries=1)

    def quote(self, code):
        self.pace()
        response = requests.get("https://push2.eastmoney.com/api/qt/stock/get", params={
            "secid": ("1." if code.startswith("6") else "0.") + code, "fltt": "2", "invt": "2",
            "fields": "f43,f44,f45,f46,f47,f48,f51,f52,f57,f58,f60,f86"}, timeout=(5, 12))
        response.raise_for_status()
        body = response.json()
        if body.get("rc") != 0 or not body.get("data"):
            raise ValueError("Provider returned no quote")
        return body["data"]


def normalize_history(raw, code, start, end):
    data = raw.rename(columns=RENAME).copy()
    if "code" not in data:
        data["code"] = code
    columns = ["date", "code", "open", "close", "high", "low", "volume", "amount", "turnover"]
    if not set(columns) <= set(data) or data.empty:
        raise ValueError("History is empty or missing required fields")
    data = data[columns]
    data["date"] = pd.to_datetime(data["date"])
    if data.date.isna().any() or not data.code.astype(str).eq(code).all() or data.duplicated(["code", "date"]).any():
        raise ValueError("History identity/date validation failed")
    for column in columns[2:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (not np.isfinite(data[columns[2:]].to_numpy(dtype=float)).all()
            or (data[["open", "close", "high", "low"]] <= 0).any().any()
            or (data[["volume", "amount", "turnover"]] < 0).any().any()
            or (data.low > data[["open", "close"]].min(axis=1)).any()
            or (data.high < data[["open", "close"]].max(axis=1)).any()):
        raise ValueError("Invalid OHLC, volume, amount or turnover")
    data = data[data.date.between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("date")
    if data.empty:
        raise ValueError("No history in requested range")
    data["volume"] *= 100  # Eastmoney history is in lots; published files are in shares.
    return data.reset_index(drop=True)


def retry(call, attempts=2):
    for attempt in range(attempts):
        events.check_cancel()
        try:
            return call()
        except events.Cancelled:
            raise
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(1)


def download_history(codes, start, end, bases, *, allow_network=False, root=ROOT, provider=None, resume_id=None):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    valid_codes(codes)
    if not bases or len(set(bases)) != len(bases) or not set(bases) <= {"hfq", "unadjusted"}:
        raise ValueError("Explicit, unique price bases required")
    if pd.Timestamp(start) > pd.Timestamp(end) or pd.Timestamp(end).date() > datetime.now(CHINA).date():
        raise ValueError("Invalid or future download range")
    store = Snapshots(root)
    request = dict(provider="eastmoney", codes=codes, start_date=start, end_date=end, bases=bases)
    old, old_dir = (store.get("history", resume_id) if resume_id else ({}, None))
    if old and old["request"] != request:
        raise ValueError("Retry must use the original request, including stock order and price bases")
    identifier, directory = store.begin("history")
    provider = provider or Eastmoney(root)
    previous = {(r["code"], r["basis"]): r for r in old.get("entries", [])}
    entries, errors = [], []
    failures = 0
    started = time.monotonic()
    for code in codes:
        for basis in bases:
            events.check_cancel()
            events.emit("download_progress", code=code, basis=basis, completed=len(entries), total=len(codes) * len(bases))
            target = directory / basis
            target.mkdir(exist_ok=True)
            prior = previous.get((code, basis))
            if prior and prior["status"] == "ready":
                for suffix in ["parquet", "csv"]:
                    shutil.copyfile(old_dir / basis / f"{code}.{suffix}", target / f"{code}.{suffix}")
                entries.append(dict(prior, reused_from=resume_id))
                continue
            if failures >= 3 or time.monotonic() - started > 600:
                item = dict(code=code, basis=basis, status="failed", error="Batch circuit/time limit reached")
                entries.append(item); errors.append(item)
                continue
            try:
                frame = normalize_history(retry(lambda: provider.history(code, start, end, basis)), code, start, end)
                frame.to_parquet(target / f"{code}.parquet", index=False)
                frame.to_csv(target / f"{code}.csv", index=False, encoding="utf-8-sig")
                item = dict(code=code, basis=basis, status="ready", rows=len(frame),
                            first_date=str(frame.date.iloc[0].date()), last_date=str(frame.date.iloc[-1].date()),
                            received_at=utc_now())
                failures = 0
            except events.Cancelled:
                raise
            except Exception as error:
                failures += 1
                item = dict(code=code, basis=basis, status="failed", error=str(error)[:500])
                errors.append(item)
            entries.append(item)
            with (directory / "requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return store.publish("history", identifier, directory, dict(request=request, entries=entries,
                         status="ready" if not errors else "partial", errors=errors, volume_unit="shares",
                         complete_bar_policy="Only dates strictly before capture day may be used for plans; exchange calendar not verified"))


def normalize_quote(raw, code, received_at):
    if str(raw.get("f57")) != code:
        raise ValueError("Quote stock code mismatch")

    def numeric(key, positive=True):
        try:
            value = float(raw[key])
            return value if np.isfinite(value) and (value > 0 if positive else value >= 0) else None
        except (ValueError, KeyError, TypeError):
            return None

    last = numeric("f43")
    if last is None:
        raise ValueError("Missing or invalid latest price")
    timestamp = None
    epoch = numeric("f86")
    if epoch:
        try:
            candidate = datetime.fromtimestamp(epoch, timezone.utc)
            age = (datetime.fromisoformat(received_at) - candidate).total_seconds()
            if candidate.year >= 2000 and age >= -5:
                timestamp = candidate.isoformat()
        except (ValueError, OverflowError, OSError):
            pass
    return dict(code=code, name=str(raw.get("f58", "")), last=last, previous_close=numeric("f60"),
                limit_up=numeric("f51"), limit_down=numeric("f52"),
                provider_timestamp=timestamp, received_at=received_at, time_field="f86",
                timestamp_semantics="unverified_vendor_epoch_field_f86", price_basis="unadjusted")


def capture_quotes(codes, *, allow_network=False, root=ROOT, provider=None, permission_check=None):
    if not allow_network:
        raise ValueError("Explicit network consent required")
    valid_codes(codes)
    if len(codes) > 50:
        raise ValueError("Quote snapshots are limited to 50 stocks")
    store = Snapshots(root)
    identifier, directory = store.begin("quotes")
    provider = provider or Eastmoney(root)
    rows, errors, raw_results = [], [], {}
    failures = 0
    started = time.monotonic()
    for code in codes:
        events.check_cancel()
        if failures >= 3 or time.monotonic() - started > 90:
            errors.append(dict(code=code, error="Quote circuit/time limit reached"))
            continue
        try:
            def fetch():
                if permission_check:
                    permission_check()
                return provider.quote(code)
            raw = retry(fetch)
            received = utc_now()
            raw_results[code] = dict(received_at=received, data=raw)
            rows.append(normalize_quote(raw, code, received))
            failures = 0
        except events.Cancelled:
            raise
        except Exception as error:
            failures += 1
            errors.append(dict(code=code, error=str(error)[:500]))
        events.emit("quote_progress", completed=len(rows) + len(errors), total=len(codes))
    if permission_check:
        permission_check()
    write_json(directory / "quotes.json", rows)
    write_json(directory / "provider_response.json", raw_results)
    return store.publish("quotes", identifier, directory, dict(provider="eastmoney", codes=codes,
                         status="ready" if not errors else "partial", errors=errors,
                         timestamp_source="f86 parsed as Unix seconds; vendor semantics and exchange trade time are not certified",
                         price_basis="unadjusted", rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["history", "quotes"], required=True)
    parser.add_argument("--codes", nargs="+", required=True)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=str(datetime.now(CHINA).date()))
    parser.add_argument("--allow-network", action="store_true", required=True)
    args = parser.parse_args()
    result = (download_history(args.codes, args.start, args.end, ["hfq", "unadjusted"], allow_network=args.allow_network)
              if args.kind == "history" else capture_quotes(args.codes, allow_network=args.allow_network))
    print(json.dumps(result, ensure_ascii=True, indent=2))
