"""Unit checks for the read-only data inventory."""

import json

import pandas as pd
import pytest

from tools import data_inventory as inventory


@pytest.mark.parametrize("code, exchange, board", [
    ("600519", "上交所", "主板"),
    ("601059", "上交所", "主板"),
    ("688981", "上交所", "科创板"),
    ("000001", "深交所", "主板"),
    ("002594", "深交所", "主板"),
    ("302132", "深交所", "主板"),
    ("300750", "深交所", "创业板"),
    ("301165", "深交所", "创业板"),
    ("920992", "北交所", "北交所"),
    ("830799", "北交所", "北交所"),
])
def test_board_of_maps_every_a_share_prefix(code, exchange, board):
    entry = inventory.board_of(code)
    assert (entry["exchange"], entry["board"], entry["lot"]) == (exchange, board, 100)


def test_board_of_keeps_an_unknown_code_out_of_a_real_board():
    assert inventory.board_of("abcdef")["board"] == "其他"
    assert inventory.board_of("12345")["board"] == "其他"


def _tree(root):
    """Smallest tree that exercises every section of the report."""
    data = root / "data"
    reports = root / "reports" / "runs"

    (data / "stock_list").mkdir(parents=True)
    (data / "stock_list" / "hs300_20260629.txt").write_text("000001\n688981\n302132\n", encoding="utf-8")

    market = data / "online" / "market" / "m_1"
    (market / "indices").mkdir(parents=True)
    pd.DataFrame({"code": ["000001", "688981", "302132"], "name": ["平安银行", "中芯国际", "中航成飞"],
                  "limit_rate": [0.1, 0.2, 0.1], "suspended": [False, False, True],
                  "price_basis": ["unadjusted"] * 3}).to_parquet(market / "breadth.parquet", index=False)
    pd.DataFrame({"date": pd.to_datetime(["2017-01-03", "2024-12-31"]), "code": "000300"}).to_parquet(
        market / "indices" / "000300.parquet", index=False)

    fundamentals = data / "online" / "fundamentals" / "f_1"
    fundamentals.mkdir(parents=True)
    pd.DataFrame({
        "code": ["000001", "688981", "302132"],
        "period": ["20240930"] * 3,
        "所处行业": ["银行Ⅱ", "半导体", None],
        "announcement_date": ["2024-10-30"] * 3,
    }).to_parquet(fundamentals / "financials.parquet", index=False)
    pd.DataFrame({"code": ["000001"], "period": ["20240930"], "预测指标": ["归属于上市公司股东的净利润"],
                  "业绩变动幅度": [12.0], "预告类型": ["预增"], "预测数值": [1e9],
                  "上年同期值": [8e8]}).to_parquet(fundamentals / "forecasts.parquet", index=False)
    pd.DataFrame({"code": ["000001"], "period": ["20240930"], "股东户数-本次": [1000],
                  "股东户数-增减比例": [-2.0], "户均持股数量": [500.0], "总股本": [1e9],
                  "总市值": [1.1e10]}).to_parquet(fundamentals / "shareholders.parquet", index=False)

    history = data / "online" / "history" / "h_1"
    (history / "hfq").mkdir(parents=True)
    (history / "unadjusted").mkdir(parents=True)
    bars = pd.DataFrame({"date": pd.to_datetime(["2017-01-03", "2024-12-31"])})
    bars.to_parquet(history / "hfq" / "000001.parquet", index=False)
    bars.to_parquet(history / "unadjusted" / "000001.parquet", index=False)
    bars.assign(date=pd.to_datetime(["2019-01-02", "2024-12-31"])).to_parquet(
        history / "hfq" / "688981.parquet", index=False)

    macro = data / "online" / "macro" / "e_1"
    macro.mkdir(parents=True)
    pd.DataFrame({
        "period": ["2024-09-30", "2024-09-30", "2024-06-30"],
        "available_from": ["2024-10-31", "2024-10-31", "2024-07-31"],
        "value": [50.0, 2.0, 1.0],
        "series_id": ["pmi", "cpi", "ppi"],
        "metric": ["制造业", "同比", "同比"],
        "unit": ["index", "percent", "percent"],
        "frequency": ["monthly"] * 3,
        "source": ["eastmoney"] * 3,
        "api": ["macro_china_pmi", "macro_china_cpi", "macro_china_ppi"],
    }).to_parquet(macro / "macro.parquet", index=False)

    calendar = data / "online" / "calendar" / "c_1"
    calendar.mkdir(parents=True)
    pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"])}).to_parquet(
        calendar / "calendar.parquet", index=False)

    clean = data / "clean" / "v1"
    clean.mkdir(parents=True)
    (clean / "manifest.json").write_text(json.dumps({
        "version": "v1", "stock_codes": ["000001", "688981"], "n_dates": 2, "rows": 4,
        "start_date": "2017-01-03", "end_date": "2024-12-31",
        "train_end": "2022-08-05", "val_end": "2023-10-20",
        "feature_cols": ["return"], "price_basis": "hfq_research",
    }), encoding="utf-8")

    raw = data / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"日期": ["2017-01-03", "2024-12-31"], "开盘": [9.11, 11.0]}).to_csv(
        raw / "000001_data_2017_2024.csv", index=False)
    pd.DataFrame({"日期": ["2017-01-03"], "开盘": [1035.6]}).to_csv(
        raw / "000001_data_2017_2024_hfq.csv", index=False)
    pd.DataFrame({"日期": ["2017-01-03", "2026-06-26"], "股票代码": ["000001"] * 2,
                  "开盘": [9.11, 12.0]}).to_parquet(raw / "stock_hist_deadbeef.parquet", index=False)

    fold = reports / "study" / "folds" / "2023"
    fold.mkdir(parents=True)
    (fold / "manifest.json").write_text(json.dumps({
        "version": "fold_v1", "stock_codes": ["000001", "688981", "302132"], "n_dates": 2,
        "rows": 6, "start_date": "2017-01-03", "end_date": "2026-06-30",
        "train_end": "2023-03-31", "val_end": "2023-06-30", "feature_cols": ["return"],
        "price_basis": "hfq_research", "protected_period_excluded": True,
    }), encoding="utf-8")
    return data, reports


def test_build_joins_names_industries_and_reports_missing_fields(tmp_path):
    data, reports = _tree(tmp_path)
    payload = inventory.build(data, reports)

    assert [row["code"] for row in payload["codes"]] == ["000001", "688981", "302132"]
    by_code = {row["code"]: row for row in payload["codes"]}

    assert by_code["000001"]["name"] == "平安银行"
    assert by_code["000001"]["industry"] == "银行Ⅱ"
    assert by_code["000001"]["history_rows"] == 2
    assert by_code["000001"]["unadjusted"] is True
    assert by_code["688981"]["history_start"] == "2019-01-02"
    assert by_code["688981"]["unadjusted"] is False

    # 302132 has no industry in the vendor table: it must stay blank and be listed as a gap.
    assert by_code["302132"]["industry"] == ""
    assert by_code["302132"]["fundamental_periods"] == 1
    gaps = {item["label"]: item["codes"] for item in payload["gaps"]}
    assert gaps["缺少行业分类"] == ["302132"]
    assert gaps["缺少不复权历史"] == ["688981", "302132"]
    assert gaps["缺少后复权历史"] == ["302132"]

    assert payload["counts"]["board"] == {"深交所主板": 2, "上交所科创板": 1}
    assert payload["raw"]["per_code_plain"] == 1 and payload["raw"]["per_code_hfq"] == 1
    assert payload["raw"]["mirrors"] == 1
    assert payload["raw"]["start"] == "2017-01-03" and payload["raw"]["end"] == "2024-12-31"
    assert payload["raw"]["mirror_end"] == "2026-06-26"
    assert payload["fundamentals"]["codes"] == 3
    supplementary = payload["fundamentals"]["supplementary"]
    assert supplementary["forecasts"]["rows"] == 1
    assert supplementary["shareholders"]["codes"] == 1
    assert "总市值" in supplementary["shareholders"]["present"]
    assert payload["datasets"][0]["version"] == "v1"
    assert payload["folds"][0]["protected_period_excluded"] is True
    assert payload["market"]["breadth"]["codes"] == 3
    assert payload["market"]["calendar"]["days"] == 2


def test_build_flags_macro_series_that_stop_early(tmp_path):
    data, reports = _tree(tmp_path)
    payload = inventory.build(data, reports)
    stale = {item["label"]: item["codes"] for item in payload["gaps"]}
    assert stale["落后最新发布期超过 45 天的宏观序列"] == ["ppi"]


def test_markdown_carries_every_section_and_the_code_count(tmp_path):
    data, reports = _tree(tmp_path)
    text = inventory._markdown(inventory.build(data, reports))

    for heading in ("## 1. 股票池", "## 2. 代码、市场与行业", "## 3. 数据集版本",
                    "## 4. 研究 fold 面板", "## 5. 个股行情", "## 6. 模型输入特征",
                    "## 7. 基本面", "## 8. 宏观", "## 9. 指数、宽度与日历", "## 10. 已知缺口"):
        assert heading in text
    assert "**3**" in text          # universe size
    assert "codes.csv" in text      # points at the full table
    assert "银行Ⅱ" in text
    assert "缺少行业分类：302132" in text
    assert "业绩预告" in text and "股东户数" in text
    assert "| 1 | return | 收益 |" in text
