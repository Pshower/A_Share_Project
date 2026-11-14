"""Probe the data sources this project could actually use, and say which ones answer.

The objective is an inventory, not an experiment: every free interface in ``akshare`` (1,095 of them)
is too many to reason about, so this file lists the ones that matter to a macro / fundamental /
technical split, calls each with a small request, and records whether it answered, how many rows,
how fresh the last row is, and how long it took.

Two rules learned the hard way in this repository:

* **A probe that returns nothing is a finding**, not a failure to hide.  ``status`` says ``empty``,
  ``blocked`` or ``error`` and the row is kept, because "this source is gone" is information.
* **Nothing is inferred.**  A source is not credited with a signal because its name sounds
  promising; the verdict column only says whether it *answered*, and the document that consumes
  this JSON is where effectiveness is argued from prior tests.

Probes run in a thread pool with a hard per-call timeout, because several of these endpoints hang
rather than fail when they are rate-limited.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "runs" / "data_catalog"
TODAY = date.today()
RECENT = (TODAY - timedelta(days=30)).strftime("%Y%m%d")
START = (TODAY - timedelta(days=120)).strftime("%Y%m%d")

# (id, category, family, module, function, kwargs, what the row means)
CATALOG = [
    # ---------------------------------------------------------------- 宏观
    ("gdp", "宏观", "增长", "macro_china_gdp", {}, "GDP（季度）"),
    ("cpi", "宏观", "通胀", "macro_china_cpi", {}, "CPI（月度）"),
    ("ppi", "宏观", "通胀", "macro_china_ppi", {}, "PPI（月度）"),
    ("pmi", "宏观", "景气", "macro_china_pmi", {}, "官方 PMI（月度）"),
    ("m2", "宏观", "货币", "macro_china_money_supply", {}, "货币供应量（月度）"),
    ("shrzgm", "宏观", "社融", "macro_china_shrzgm", {}, "社会融资规模增量"),
    ("lpr", "宏观", "利率", "macro_china_lpr", {}, "贷款市场报价利率"),
    ("yield", "宏观", "利率", "bond_zh_us_rate", {"start_date": "20240101"}, "中美国债收益率曲线"),
    ("margin_sh", "宏观", "杠杆", "macro_china_market_margin_sh", {}, "沪市两融余额"),
    ("margin_sz", "宏观", "杠杆", "macro_china_market_margin_sz", {}, "深市两融余额"),
    ("cnbs", "宏观", "杠杆", "macro_cnbs", {}, "宏观杠杆率"),
    ("fx", "宏观", "汇率", "currency_boc_sina", {"symbol": "美元", "start_date": RECENT, "end_date": TODAY.strftime("%Y%m%d")}, "人民币汇率"),
    ("trade", "宏观", "外贸", "macro_china_trade_balance", {}, "贸易差额"),

    # ---------------------------------------------------------------- 技术
    ("stock_daily", "技术", "个股日线", "stock_zh_a_hist", {"symbol": "600519", "period": "daily", "start_date": RECENT, "end_date": TODAY.strftime("%Y%m%d"), "adjust": "qfq"}, "个股日线"),
    ("stock_daily_sina", "技术", "个股日线", "stock_zh_a_daily", {"symbol": "sh600519", "start_date": RECENT, "end_date": TODAY.strftime("%Y%m%d"), "adjust": "qfq"}, "个股日线（新浪）"),
    ("stock_min", "技术", "分钟线", "stock_zh_a_hist_min_em", {"symbol": "600519", "period": "5", "adjust": "qfq"}, "5 分钟线"),
    ("stock_spot", "技术", "全市场快照", "stock_zh_a_spot_em", {}, "A 股实时快照"),
    ("index_daily", "技术", "指数日线", "stock_zh_index_daily", {"symbol": "sh000300"}, "指数日线"),
    ("index_min", "技术", "指数分钟", "index_zh_a_hist_min_em", {"symbol": "000300", "period": "5"}, "指数分钟线"),
    ("market_activity", "技术", "市场情绪", "stock_market_activity_legu", {}, "赚钱效应"),
    ("index_pe", "技术", "指数估值", "stock_index_pe_lg", {"symbol": "沪深300"}, "指数 PE 序列"),
    ("zt_pool", "技术", "市场情绪", "stock_zt_pool_em", {"date": (TODAY - timedelta(days=1)).strftime("%Y%m%d")}, "涨停板池"),
    ("sse_summary", "技术", "市场统计", "stock_sse_summary", {}, "上交所概况"),
    ("szse_summary", "技术", "市场统计", "stock_szse_summary", {"date": (TODAY - timedelta(days=7)).strftime("%Y%m%d")}, "深交所统计"),

    # ---------------------------------------------------------------- 基金与 ETF
    ("etf_daily", "技术", "ETF 日线", "fund_etf_hist_sina", {"symbol": "sh510300"}, "ETF 日线（新浪）"),
    ("etf_spot", "技术", "ETF 快照", "fund_etf_spot_em", {}, "ETF 实时快照（含 IOPV）"),
    ("etf_nav", "基本面", "ETF 净值", "fund_etf_fund_info_em", {"fund": "510300"}, "ETF 历史净值"),
    ("etf_category", "技术", "ETF 目录", "fund_etf_category_sina", {"symbol": "ETF基金"}, "ETF 清单"),
    ("etf_dividend", "基本面", "ETF 分红", "fund_etf_dividend_sina", {"symbol": "sh510300"}, "ETF 分红记录"),
    ("fund_open", "基本面", "基金净值", "fund_open_fund_info_em", {"symbol": "000001", "indicator": "单位净值走势"}, "开放式基金净值"),

    # ---------------------------------------------------------------- 基本面
    ("valuation", "基本面", "估值", "stock_zh_valuation_baidu", {"symbol": "600519", "indicator": "市盈率(TTM)", "period": "近一年"}, "PE/PB 等估值序列"),
    ("fin_indicator", "基本面", "财务指标", "stock_financial_analysis_indicator", {"symbol": "600519", "start_year": "2023"}, "财务分析指标"),
    ("fin_report", "基本面", "三大报表", "stock_financial_report_sina", {"stock": "sh600519", "symbol": "资产负债表"}, "资产负债表"),
    ("yjbb", "基本面", "业绩", "stock_yjbb_em", {"date": "20260930"}, "业绩报表"),
    ("yjyg", "基本面", "业绩预告", "stock_yjyg_em", {"date": "20260930"}, "业绩预告"),
    ("fhps", "基本面", "分红送配", "stock_fhps_em", {"date": "20251231"}, "分红送配"),
    ("gdhs", "基本面", "股东户数", "stock_zh_a_gdhs", {"symbol": "20260930"}, "股东户数"),
    ("lhb", "基本面", "龙虎榜", "stock_lhb_detail_em", {"start_date": "20261001", "end_date": "20261031"}, "龙虎榜明细"),
    ("unlock", "基本面", "限售解禁", "stock_restricted_release_queue_em", {"symbol": "600519"}, "限售解禁"),
    ("index_cons", "基本面", "指数成分", "index_stock_cons", {"symbol": "000300"}, "沪深300成分"),
    ("industry", "基本面", "行业分类", "stock_board_industry_name_em", {}, "行业板块"),
    ("fund_hold", "基本面", "基金持仓", "fund_portfolio_hold_em", {"symbol": "510300", "date": "2026"}, "基金持仓明细"),
]


def probes():
    """A list of (entry, callable) so a probe can run without knowing the argument shapes."""
    return list(CATALOG)


def _call(entry):
    name, category, family, function, kwargs, note = entry
    import akshare as ak

    started = time.time()
    frame = getattr(ak, function)(**kwargs)
    elapsed = time.time() - started
    return frame, elapsed


DATE_HINTS = ("date", "日期", "时间", "月份", "交易日", "报告期", "季度", "年份", "统计时间", "time")
CHINESE_DATE = re.compile(r"(\d{4})\D+?(\d{1,2})")


def to_dates(series):
    """Parse a column into dates, including the Chinese forms these sources actually use.

    "2026年10月份" and "2026年第1-3季度" defeat ``pd.to_datetime`` and used to make a current
    series look two years stale.  Falling back to the year/month pair keeps the freshness reading
    honest.
    """
    import pandas as pd

    text = series.astype(str)
    parsed = pd.to_datetime(text, errors="coerce")
    if parsed.notna().mean() >= 0.5:
        return parsed
    found = text.str.extract(CHINESE_DATE)
    rebuilt = found[0] + "-" + found[1].str.zfill(2) + "-01"
    retry = pd.to_datetime(rebuilt, errors="coerce")
    return retry if retry.notna().any() else parsed


def shape(frame):
    """Rows, column names and the freshest date, without mistaking a code or a price for a date.

    The first version parsed every column as a timestamp, so a code like 20261231 or a price read as
    an epoch and every source appeared to end in 1970.  Dates are only taken from columns that look
    like dates, and only if the parsed values land in a plausible window.
    """
    import pandas as pd

    rows, columns, last = 0, 0, None
    if frame is None:
        return 0, 0, None, []
    names = []
    if isinstance(frame, pd.DataFrame):
        rows, columns = int(len(frame)), int(frame.shape[1])
        names = [str(column) for column in frame.columns][:10]
        candidates = [column for column in frame.columns
                      if any(hint in str(column).lower() for hint in DATE_HINTS)]
        for column in candidates:
            series = to_dates(frame[column])
            if not series.notna().any():
                continue
            newest = series.max()
            if 1990 <= newest.year <= 2100:
                stamp = str(newest.date())
                last = stamp if last is None or stamp > last else last
    elif isinstance(frame, pd.Series):
        rows, columns = int(len(frame)), 1
        names = [str(frame.name)] if frame.name else []
    else:
        try:
            rows = len(frame)
        except TypeError:
            rows = 0
    return rows, columns, last, names


def classify(problem):
    """A refused connection is not the same finding as an interface that no longer exists."""
    text = "%s: %s" % (type(problem).__name__, problem)
    if any(marker in text for marker in ("ConnectionError", "RemoteDisconnected", "URLError",
                                         "Timeout", "timed out", "10060")):
        return "blocked", text
    return "error", text


def probe(entry, timeout=30):
    """One source, hard-capped.

    The executor is shut down without waiting: ``with ThreadPoolExecutor(...)`` waits for the worker
    on exit, so a request that hangs past the timeout would still block the whole run -- the timeout
    would be decorative.  The abandoned thread dies with the process.
    """
    name, category, family, function, kwargs, note = entry
    started = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_call, entry)
    try:
        frame, elapsed = future.result(timeout=timeout)
        rows, columns, last, names = shape(frame)
        status = "ok" if rows else "empty"
        return dict(id=name, category=category, family=family, function=function, kwargs=kwargs,
                    note=note, status=status, rows=rows, columns=columns, last_date=last,
                    column_names=names, seconds=round(elapsed, 2), error=None)
    except FutureTimeout:
        return dict(id=name, category=category, family=family, function=function, kwargs=kwargs,
                    note=note, status="timeout", rows=0, columns=0, last_date=None,
                    column_names=[], seconds=round(time.time() - started, 2),
                    error="no answer in %ss" % timeout)
    except Exception as problem:                      # noqa: BLE001 - the message is the finding
        status, message = classify(problem)
        return dict(id=name, category=category, family=family, function=function, kwargs=kwargs,
                    note=note, status=status, rows=0, columns=0, last_date=None,
                    column_names=[], seconds=round(time.time() - started, 2),
                    error=message[:180])
    finally:
        pool.shutdown(wait=False)


def summarise(rows):
    by_category = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], dict(total=0, ok=0, empty=0, blocked=0, failed=0))
        bucket["total"] += 1
        if row["status"] == "ok":
            bucket["ok"] += 1
        elif row["status"] == "empty":
            bucket["empty"] += 1
        elif row["status"] == "blocked":
            bucket["blocked"] += 1
        else:
            bucket["failed"] += 1
    return by_category


def run(entries=None, timeout=30, output=OUTPUT, log=print):
    entries = entries if entries is not None else probers()
    rows = []
    for entry in entries:
        row = probe(entry, timeout=timeout)
        rows.append(row)
        log("%-14s %-6s %-10s %5d rows  %-10s %s" % (
            row["id"], row["category"], row["status"], row["rows"],
            row["last_date"] or "-", (row["error"] or "")[:60]))
    result = dict(generated_at=date.today().isoformat(), probes=len(rows),
                  summary=summarise(rows), rows=rows)
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        (path / "catalog.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log("wrote %s" % (path / "catalog.json"))
    return result


def probers():
    return CATALOG


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probe the data sources and record what answers.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--only", default=None, help="Probe a single id.")
    args = parser.parse_args(argv)
    entries = CATALOG
    if args.only:
        entries = [entry for entry in CATALOG if entry[0] == args.only]
        if not entries:
            raise SystemExit("no such id: %s" % args.only)
    result = run(entries, timeout=args.timeout, output=args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
