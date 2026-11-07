"""Read-only inventory of the local A-share data assets.

The repository has grown several independent data families (raw CSVs, dense
research panels, fold panels, online fundamentals, macro series, index and
breadth snapshots).  This module walks those directories and reports what is
actually on disk: which codes, which exchange/board, which industry, which
date span, and which fields.  It reads no model artifact and makes no research
decision, so its output can be regenerated at any time without touching a
frozen version.

Every number comes with the file or directory it was derived from.  Where a
field is genuinely absent it is reported as a gap instead of being filled from
an unrelated snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs"

LOT_SIZE = 100

# Longest prefixes first so "688"/"689" wins over "60".
_BOARDS = (
    ("688", "上交所", "科创板"),
    ("689", "上交所", "科创板"),
    ("920", "北交所", "北交所"),
    ("300", "深交所", "创业板"),
    ("301", "深交所", "创业板"),
    ("302", "深交所", "主板"),
    ("000", "深交所", "主板"),
    ("001", "深交所", "主板"),
    ("002", "深交所", "主板"),
    ("003", "深交所", "主板"),
    ("60", "上交所", "主板"),
    ("43", "北交所", "北交所"),
    ("83", "北交所", "北交所"),
    ("87", "北交所", "北交所"),
)


def board_of(code: str) -> dict:
    """Map a six-digit code to exchange and board; unknown prefixes stay 其他."""
    text = str(code).strip()
    if len(text) != 6 or not text.isdigit():
        return {"code": text, "exchange": "其他", "board": "其他", "lot": LOT_SIZE}
    for prefix, exchange, board in _BOARDS:
        if text.startswith(prefix):
            return {"code": text, "exchange": exchange, "board": board, "lot": LOT_SIZE}
    return {"code": text, "exchange": "其他", "board": "其他", "lot": LOT_SIZE}


def board_label(code: str) -> str:
    """One readable label: exchange-qualified for the two mainland exchanges."""
    entry = board_of(code)
    if entry["board"] in ("北交所", "其他"):
        return entry["board"]
    return f"{entry['exchange']}{entry['board']}"


def _rel(path: Path) -> str:
    """Prefer a workspace-relative label so reports stay portable."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stock_list(path: Path) -> list[str]:
    """The HS300 snapshot is one bare code per line."""
    return [line.strip() for line in path.read_text(encoding="utf-8").split() if line.strip()]


def read_names(breadth_path: Path) -> dict[str, str]:
    frame = pd.read_parquet(breadth_path, columns=["code", "name"])
    return {str(code).zfill(6): str(name).strip() for code, name in zip(frame.code, frame.name)}


def read_industry(financials_path: Path) -> dict[str, str]:
    """Latest non-null 所处行业 per code, which is the vendor's SW level-2 label."""
    frame = pd.read_parquet(financials_path, columns=["code", "period", "所处行业"])
    frame = frame.sort_values(["code", "period"]).dropna(subset=["所处行业"])
    return {str(code).zfill(6): str(name).strip() for code, name in zip(frame.code, frame.所处行业)}


def read_fundamental_periods(financials_path: Path) -> dict[str, dict]:
    frame = pd.read_parquet(financials_path, columns=["code", "period", "announcement_date"])
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    grouped = frame.groupby("code")
    return {
        code: dict(periods=int(part.period.nunique()), first=str(part.period.min()),
                   last=str(part.period.max()), announced=int(part.announcement_date.notna().sum()))
        for code, part in grouped
    }


def history_index(history_root: Path) -> dict[str, dict]:
    """Merge the repeated per-code downloads into one span per code."""
    merged: dict[str, dict] = {}
    for path in sorted(history_root.glob("*/hfq/*.parquet")):
        code = path.stem
        dates = pd.read_parquet(path, columns=["date"])["date"]
        item = merged.setdefault(code, dict(code=code, batches=0, rows=0))
        item["batches"] += 1
        item["rows"] = max(item["rows"], int(len(dates)))
        item["start"] = min(filter(None, [item.get("start"), dates.min()]))
        item["end"] = max(filter(None, [item.get("end"), dates.max()]))
    for path in history_root.glob("*/unadjusted/*.parquet"):
        merged.setdefault(path.stem, dict(code=path.stem, batches=0, rows=0))["unadjusted"] = True
    for item in merged.values():
        item.setdefault("start", pd.NaT)
        item.setdefault("end", pd.NaT)
        item.setdefault("unadjusted", False)
    return merged


def history_summary(history_root: Path, codes: dict[str, dict]) -> dict:
    """Aggregate the repeated online history downloads into one span."""
    files = sorted(history_root.glob("*/hfq/*.parquet"))
    unadjusted = sorted(history_root.glob("*/unadjusted/*.parquet"))
    if not files:
        return dict(codes=0, hfq_files=0, unadjusted_files=0, columns=[], start="", end="")
    columns = [str(column) for column in pd.read_parquet(files[0]).columns]
    starts = [item["start"] for item in codes.values() if pd.notna(item.get("start"))]
    ends = [item["end"] for item in codes.values() if pd.notna(item.get("end"))]
    return dict(codes=len(codes), hfq_files=len(files), unadjusted_files=len(unadjusted),
                columns=columns, start=str(min(starts).date()), end=str(max(ends).date()),
                batches=len({path.parent.parent.name for path in files}))


_RAW_PER_CODE = re.compile(r"^\d{6}_data_.*\.csv$")


def _csv_span(paths: list[Path], column: str = "日期") -> tuple:
    start, end = None, None
    for path in paths:
        dates = pd.to_datetime(pd.read_csv(path, usecols=[column])[column])
        start = dates.min() if start is None else min(start, dates.min())
        end = dates.max() if end is None else max(end, dates.max())
    if start is None:
        return "", ""
    return str(start.date()), str(end.date())


def raw_index(raw_root: Path) -> dict:
    """`data/raw` holds three unrelated naming families with no manifest."""
    per_code = sorted(path for path in raw_root.glob("*.csv") if _RAW_PER_CODE.match(path.name))
    plain = [path for path in per_code if not path.name.endswith("_hfq.csv")]
    hfq = [path for path in per_code if path.name.endswith("_hfq.csv")]
    combined = sorted(path for path in raw_root.glob("stock_data*.csv"))
    mirrors = sorted(raw_root.glob("stock_hist_*.parquet"))

    columns = []
    if plain:
        columns = [str(column) for column in pd.read_csv(plain[0], nrows=2).columns]
    start, end = _csv_span(plain)
    mirror_start, mirror_end = "", ""
    for path in mirrors:
        dates = pd.to_datetime(pd.read_parquet(path, columns=["日期"])["日期"])
        mirror_start = dates.min() if not mirror_start else min(mirror_start, dates.min())
        mirror_end = dates.max() if not mirror_end else max(mirror_end, dates.max())
    return dict(
        per_code_plain=len(plain), per_code_hfq=len(hfq),
        combination=[dict(file=path.name, start=_csv_span([path])[0], end=_csv_span([path])[1])
                     for path in combined],
        mirrors=len(mirrors),
        mirror_start="" if not mirror_start else str(pd.Timestamp(mirror_start).date()),
        mirror_end="" if not mirror_end else str(pd.Timestamp(mirror_end).date()),
        columns=columns,
        start=start, end=end,
        files=len(per_code) + len(combined) + len(mirrors),
    )


def supplementary_tables(directory: Path) -> dict:
    """Earnings preannouncements and shareholder counts sit beside the statements."""
    specs = {
        "forecasts": ("业绩预告", ["预测指标", "业绩变动幅度", "预告类型", "预测数值", "上年同期值"]),
        "shareholders": ("股东户数", ["股东户数-本次", "股东户数-增减比例", "户均持股数量", "总股本", "总市值"]),
    }
    tables = {}
    for key, (label, fields) in specs.items():
        path = directory / f"{key}.parquet"
        if not path.exists():
            tables[key] = dict(label=label, rows=0, codes=0, periods=0, fields=fields, present=[])
            continue
        frame = pd.read_parquet(path)
        tables[key] = dict(label=label, rows=int(len(frame)), codes=int(frame["code"].nunique()),
                           periods=int(frame["period"].nunique()),
                           start=str(frame["period"].min()), end=str(frame["period"].max()),
                           fields=fields, present=[name for name in fields if name in frame.columns])
    return tables


def snapshot_counts(online_root: Path) -> dict:
    """Real-time quote and decision-plan snapshots are single-day artefacts."""
    quotes = [path for path in (online_root / "quotes").glob("*") if path.is_dir()]
    plans = [path for path in (online_root / "plans").glob("*") if path.is_dir()]
    providers = sorted(path.name for path in (online_root / "providers").glob("*") if path.is_dir())
    return dict(quotes=len(quotes), plans=len(plans), providers=providers)


def feature_names(report_root: Path, clean_root: Path) -> list[str]:
    """The model input list as the newest manifest records it."""
    candidates = list(sorted(report_root.glob("*/folds/*/manifest.json")))
    candidates.extend(sorted(clean_root.glob("*/manifest.json")))
    for manifest_path in candidates:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("feature_cols"):
            return [str(name) for name in payload["feature_cols"]]
    return []


def dataset_table(clean_root: Path) -> list[dict]:
    rows = []
    for manifest_path in sorted(clean_root.glob("*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(dict(
            version=payload.get("version", manifest_path.parent.name),
            directory=_rel(manifest_path.parent),
            codes=len(payload.get("stock_codes") or []),
            dates=payload.get("n_dates"),
            rows=payload.get("rows"),
            start=payload.get("start_date"),
            end=payload.get("end_date"),
            train_end=payload.get("train_end"),
            val_end=payload.get("val_end"),
            features=len(payload.get("feature_cols") or []),
            price_basis=payload.get("price_basis"),
        ))
    return rows


def fold_table(report_root: Path) -> list[dict]:
    """Fold panels are the production decisions' data contract; list them once each."""
    seen: dict[tuple, dict] = {}
    for manifest_path in sorted(report_root.glob("*/folds/*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        codes = payload.get("stock_codes") or []
        key = (payload.get("version"), payload.get("train_end"), payload.get("n_dates"), len(codes))
        seen.setdefault(key, dict(
            fold=manifest_path.parent.name,
            version=payload.get("version"),
            directory=_rel(manifest_path.parent),
            codes=len(codes),
            dates=payload.get("n_dates"),
            rows=payload.get("rows"),
            start=payload.get("start_date"),
            end=payload.get("end_date"),
            train_end=payload.get("train_end"),
            val_end=payload.get("val_end"),
            features=len(payload.get("feature_cols") or []),
            price_basis=payload.get("price_basis"),
            protected_period_excluded=payload.get("protected_period_excluded"),
        ))
    return sorted(seen.values(), key=lambda item: (str(item["train_end"]), str(item["version"])))


def macro_table(macro_root: Path) -> dict:
    dumps = sorted(macro_root.glob("*/macro.parquet"))
    if not dumps:
        return dict(series=[], files=0)
    frame = pd.read_parquet(dumps[-1])
    frame["period"] = frame["period"].astype(str)
    grouped = frame.groupby("series_id")
    series = [
        dict(series_id=str(name), rows=int(len(part)), start=part.period.min(), end=part.period.max(),
             frequency=str(part.frequency.iloc[0]), source=str(part.source.iloc[0]),
             api=str(part.api.iloc[0]), unit=str(part.unit.iloc[0]), metric=str(part.metric.iloc[0]),
             available_from=bool(part.available_from.notna().all()))
        for name, part in grouped
    ]
    return dict(files=len(dumps), directory=_rel(dumps[-1].parent),
                fields=[str(column) for column in frame.columns],
                snapshots=int(frame.period.nunique()),
                available_from_nulls=int(frame.available_from.isna().sum()),
                series=sorted(series, key=lambda item: item["series_id"]))


def index_table(market_root: Path) -> list[dict]:
    rows = []
    for path in sorted(market_root.glob("*/indices/*.parquet")):
        frame = pd.read_parquet(path, columns=["date", "code"])
        rows.append(dict(code=path.stem, rows=int(len(frame)),
                         start=str(pd.Timestamp(frame.date.min()).date()),
                         end=str(pd.Timestamp(frame.date.max()).date())))
    return rows


def breadth_summary(market_root: Path) -> dict:
    snapshots = sorted(market_root.glob("*/breadth.parquet"))
    if not snapshots:
        return dict(codes=0)
    frame = pd.read_parquet(snapshots[-1])
    boards = pd.Series([board_label(code) for code in frame.code]).value_counts().to_dict()
    limits = frame["limit_rate"].round(4).value_counts().to_dict()
    return dict(directory=_rel(snapshots[-1].parent),
                codes=int(len(frame)), boards=boards,
                limit_rates={f"{key:.2f}": int(value) for key, value in sorted(limits.items())},
                suspended=int(frame["suspended"].sum()),
                price_basis=sorted({str(value) for value in frame["price_basis"]}),
                fields=[str(column) for column in frame.columns])


def calendar_summary(calendar_root: Path) -> dict:
    paths = sorted(calendar_root.glob("*/calendar.parquet"))
    if not paths:
        return dict(days=0)
    frame = pd.read_parquet(paths[-1])
    dates = pd.to_datetime(frame["trade_date"])
    return dict(days=int(len(dates)), start=str(dates.min().date()), end=str(dates.max().date()),
                source=_rel(paths[-1].parent))


def build(data_root: Path = DATA_ROOT, report_root: Path = REPORT_ROOT) -> dict:
    """Walk the local tree and return the inventory payload."""
    stock_lists = sorted((data_root / "stock_list").glob("hs300_*.txt"))
    stock_list_path = stock_lists[-1]
    codes = read_stock_list(stock_list_path)

    online = data_root / "online"
    breadth_path = sorted(online.glob("market/*/breadth.parquet"))[-1]
    fundamentals = sorted(online.glob("fundamentals/*/financials.parquet"),
                          key=lambda path: (pd.read_parquet(path, columns=["code"]).code.nunique(),
                                            path.stat().st_size))[-1]
    names = read_names(breadth_path)
    industry = read_industry(fundamentals)
    periods = read_fundamental_periods(fundamentals)
    history = history_index(online / "history")

    rows = []
    for code in codes:
        entry = board_of(code)
        item = history.get(code, {})
        detail = periods.get(code, {})
        rows.append(dict(
            code=code,
            name=names.get(code, ""),
            exchange=entry["exchange"],
            board=entry["board"],
            industry=industry.get(code, ""),
            history_start=str(item["start"].date()) if pd.notna(item.get("start")) else "",
            history_end=str(item["end"].date()) if pd.notna(item.get("end")) else "",
            history_rows=int(item.get("rows", 0)),
            hfq_downloads=int(item.get("batches", 0)),
            unadjusted=bool(item.get("unadjusted", False)),
            fundamental_periods=int(detail.get("periods", 0)),
            fundamental_first=detail.get("first", ""),
            fundamental_last=detail.get("last", ""),
        ))

    universe = dict(source=_rel(stock_list_path),
                    sha256=file_sha256(stock_list_path), codes=len(codes))
    counts = {
        "exchange": pd.Series([row["exchange"] for row in rows]).value_counts().to_dict(),
        "board": pd.Series([board_label(row["code"]) for row in rows]).value_counts().to_dict(),
        "industry": pd.Series([row["industry"] or "未获取" for row in rows]).value_counts().to_dict(),
    }

    gaps = []
    for label, missing in (
        ("缺少股票简称", [row["code"] for row in rows if not row["name"]]),
        ("缺少行业分类", [row["code"] for row in rows if not row["industry"]]),
        ("缺少后复权历史", [row["code"] for row in rows if not row["history_rows"]]),
        ("缺少不复权历史", [row["code"] for row in rows if not row["unadjusted"]]),
        ("缺少财务数据", [row["code"] for row in rows if not row["fundamental_periods"]]),
    ):
        if missing:
            gaps.append(dict(label=label, codes=missing))

    macro = macro_table(online / "macro")
    if macro.get("series"):
        latest = pd.Timestamp(max(item["end"] for item in macro["series"]))
        # One monthly publication cycle is late; beyond that the series is behind its peers.
        stale = [item["series_id"] for item in macro["series"]
                 if latest - pd.Timestamp(item["end"]) > pd.Timedelta(days=45)]
        if stale:
            gaps.append(dict(label="落后最新发布期超过 45 天的宏观序列", codes=stale))

    return dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_root=_rel(data_root),
        universe=universe,
        counts=counts,
        codes=rows,
        datasets=dataset_table(data_root / "clean"),
        folds=fold_table(report_root),
        history=history_summary(online / "history", history),
        features=feature_names(report_root, data_root / "clean"),
        raw=raw_index(data_root / "raw"),
        fundamentals=dict(
            directory=_rel(fundamentals.parent),
            file=fundamentals.name,
            codes=len(periods),
            periods=int(pd.read_parquet(fundamentals, columns=["period"]).period.nunique()),
            fields=["每股收益", "营业总收入-营业总收入", "营业总收入-同比增长", "营业总收入-季度环比增长",
                    "净利润-净利润", "净利润-同比增长", "净利润-季度环比增长", "每股净资产", "净资产收益率",
                    "每股经营现金流量", "销售毛利率", "所处行业", "最新公告日期", "announcement_date"],
            pit="周期字段是报告期；可用日取 announcement_date，不是 period 末尾",
            supplementary=supplementary_tables(fundamentals.parent),
        ),
        macro=macro,
        snapshots=snapshot_counts(online),
        market=dict(indices=index_table(online / "market"), breadth=breadth_summary(online / "market"),
                    calendar=calendar_summary(online / "calendar")),
        gaps=gaps,
    )


_FEATURE_GROUPS = (
    ("return", "收益"), ("log_return", "收益"), ("range_ratio", "形态"), ("body_ratio", "形态"),
    ("volume_ratio", "量能"), ("turnover", "量能"), ("rsi_14", "强度"),
    ("macd_ratio", "趋势"), ("macd_signal_ratio", "趋势"), ("bb_position", "通道"),
    ("momentum_5", "动量"), ("momentum_10", "动量"), ("momentum_20", "动量"), ("momentum_60", "动量"),
    ("close_ma_ratio_5", "均线比"), ("close_ma_ratio_10", "均线比"),
    ("close_ma_ratio_20", "均线比"), ("close_ma_ratio_60", "均线比"),
    ("volatility_20", "波动"), ("volatility_60", "波动"),
)


def _feature_group(name: str) -> str:
    for feature, group in _FEATURE_GROUPS:
        if name == feature:
            return group
    return "其他"


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _markdown(payload: dict) -> str:
    counts = payload["counts"]
    lines = [
        "# 本地数据清单",
        "",
        f"生成时间：{payload['generated_at']}，根目录 `{payload['data_root']}`。",
        "本文件由 `tools/data_inventory.py` 从磁盘直接读取生成，不读模型、不做研究判断。",
        "",
        "## 1. 股票池",
        "",
        f"- 名单文件：`{payload['universe']['source']}`",
        f"- 代码数：**{payload['universe']['codes']}**，sha256 `{payload['universe']['sha256'][:16]}…`",
        f"- 交易所：{', '.join(f'{k} {v}' for k, v in counts['exchange'].items())}",
        f"- 板块：{', '.join(f'{k} {v}' for k, v in counts['board'].items())}",
        f"- 行业（申万二级）：{len(counts['industry'])} 个标签，"
        f"最大三类 {', '.join(f'{name} {count}' for name, count in list(counts['industry'].items())[:3])}",
        "",
        "## 2. 代码、市场与行业",
        "",
    ]
    top = list(counts["industry"].items())[:12]
    lines.extend(_table(["行业", "只数"], [[name, count] for name, count in top]))
    lines.extend(["", f"完整 {len(payload['codes'])} 行见同目录 `codes.csv`。", ""])

    dataset_rows = [[item["version"], item["codes"], item["start"], item["end"], item["features"],
                     f"{item['train_end']} / {item['val_end']}"] for item in payload["datasets"]]
    lines.extend(["## 3. 数据集版本（`data/clean`）", ""])
    lines.extend(_table(["version", "代码", "起", "止", "特征", "train_end / val_end"], dataset_rows))
    lines.append("")

    fold_rows = [[item["fold"], item["codes"], item["start"], item["end"], item["train_end"],
                  item["val_end"], item["protected_period_excluded"]] for item in payload["folds"]]
    lines.extend(["## 4. 研究 fold 面板（`reports/runs/*/folds`）", ""])
    lines.extend(_table(["fold", "代码", "起", "止", "train_end", "val_end", "保护期排除"], fold_rows))
    lines.append("")

    history = payload["history"]
    raw = payload["raw"]
    lines.extend(["## 5. 个股行情", "",
                  "### 5.1 联网历史快照（`data/online/history`）", "",
                  f"- 代码 {history['codes']} 只，后复权文件 {history['hfq_files']} 个"
                  f"（{history['batches']} 个下载批次重复覆盖），不复权文件 {history['unadjusted_files']} 个",
                  f"- 覆盖：{history['start']} 至 {history['end']}",
                  f"- 字段：{'、'.join(history['columns'])}", "",
                  "### 5.2 本地 CSV / Parquet 存档（`data/raw`）", "",
                  f"- 逐代码 CSV：不复权 {raw['per_code_plain']} 个、后复权 {raw['per_code_hfq']} 个，"
                  f"{raw['start']} 至 {raw['end']}",
                  f"- 哈希命名 Parquet 镜像：{raw['mirrors']} 个，{raw['mirror_start']} 至 {raw['mirror_end']}",
                  f"- 全市场合并转储：" + ("；".join(
                      f"`{item['file']}` {item['start']} 至 {item['end']}" for item in raw["combination"]) or "无"),
                  f"- 字段（CSV）：{'、'.join(raw['columns'])}",
                  "- 同一股票在同一目录里有三份副本，且 CSV 的股票代码列被存成整数、前导零丢失"
                  "（`000001` 写作 `1`）。", ""])

    if payload["features"]:
        lines.extend(["## 6. 模型输入特征（量价，20 个）", ""])
        lines.extend(_table(["序号", "特征", "组"], [
            [index + 1, name, _feature_group(name)] for index, name in enumerate(payload["features"])]))
        lines.append("")
    else:
        lines.extend(["## 6. 模型输入特征（量价，20 个）", "", "未找到记录特征列表的 manifest。", ""])

    fund = payload["fundamentals"]
    lines.extend(["## 7. 基本面（`data/online/fundamentals`）", "",
                  f"- 覆盖代码 {fund['codes']} 只，{fund['periods']} 个报告期",
                  f"- 字段：{'、'.join(fund['fields'])}",
                  f"- 时点：{fund['pit']}", ""])

    for key, table in fund["supplementary"].items():
        if not table["rows"]:
            lines.extend([f"### {table['label']}（`{key}.parquet`）：未找到", ""])
            continue
        lines.extend([f"### {table['label']}（`{key}.parquet`）", "",
                      f"- {table['rows']} 行 × {table['codes']} 只 × {table['periods']} 期，"
                      f"{table['start']} 至 {table['end']}",
                      f"- 可用字段：{'、'.join(table['present'])}", ""])
    lines.extend([
        "**时点陷阱**：`shareholders` 里 `最新价`、`涨跌幅`、`区间涨跌幅` 是抓取时点的快照，"
        "在全部历史行上是同一个值；只有 `总股本`、`总市值`、`户均持股数量`、`户均持股市值`、`股东户数` 是当期值。", ""])

    lines.extend(["## 8. 宏观（`data/online/macro`）", "",
                  f"- {len(payload['macro']['series'])} 条序列，字段 {'、'.join(payload['macro']['fields'])}", ""])
    series_rows = [[item["series_id"], item["frequency"], item["start"], item["end"], item["rows"],
                    item["source"]] for item in payload["macro"]["series"]]
    lines.extend(_table(["series", "频率", "起", "止", "行数", "来源"], series_rows))
    lines.append("")

    market = payload["market"]
    lines.extend(["## 9. 指数、宽度与日历", ""])
    lines.extend(_table(["指数", "起", "止", "行数"],
                        [[item["code"], item["start"], item["end"], item["rows"]]
                         for item in market["indices"]]))
    breadth = market["breadth"]
    lines.extend(["", f"- 宽度快照：{breadth['codes']} 只全市场，板块 "
                      f"{', '.join(f'{k} {v}' for k, v in breadth['boards'].items())}；"
                      f"涨跌停幅度 {', '.join(f'{k} {v}' for k, v in breadth['limit_rates'].items())}",
                  f"- 交易日历：{market['calendar']['start']} 至 {market['calendar']['end']}，"
                  f"{market['calendar']['days']} 个交易日",
                  f"- 单日快照：实时报价 {payload['snapshots']['quotes']} 次、决策计划 "
                  f"{payload['snapshots']['plans']} 次；来源目录 "
                  f"{', '.join(payload['snapshots']['providers']) or '无'}", ""])

    lines.extend(["## 10. 已知缺口", ""])
    if payload["gaps"]:
        lines.extend(f"- {item['label']}：{', '.join(item['codes'])}" for item in payload["gaps"])
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPORT_ROOT / "data_inventory_20261105"))
    args = parser.parse_args(argv)

    payload = build()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    (output / "inventory.json").write_text(
        json.dumps({key: value for key, value in payload.items() if key != "codes"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(payload["codes"]).to_csv(output / "codes.csv", index=False, encoding="utf-8-sig")
    (output / "inventory.md").write_text(_markdown(payload), encoding="utf-8")

    print(f"{len(payload['codes'])} codes, {len(payload['datasets'])} dataset versions, "
          f"{len(payload['macro']['series'])} macro series -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
