"""How effective is each data family, with the arithmetic shown.

The survey (``doc/data_sources_20261114.md``) says what can be fetched.  This file answers the next
question -- *how much is it worth* -- and it refuses two shortcuts:

* **Availability is not effectiveness.**  A source that answers every day scores full marks on
  availability and can still be worthless; the two are computed separately and never added into one
  "quality" number that would hide which one failed.
* **Untested is not average.**  A family nobody has measured gets ``unknown``, not a middle score.
  Scoring it 5/10 would quietly promote it above families that were tested and failed.

Evidence comes from the repository's own pre-registered tests; each row cites the report that
contains the number, so a reader can check the grade rather than trust it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = PROJECT_ROOT / "reports" / "runs" / "data_catalog"
OUTPUT = CATALOG_DIR / "scorecard.json"

# family: (grade, evidence, tradable, cited report)
#   grade  A = tested and demonstrably useful
#          B = tested, statistically present, economically not worth it
#          C = tested and not distinguishable from zero
#          D = cannot be obtained
#          U = never tested (no verdict -- not a low score)
EVIDENCE = {
    "利率 / 国债收益率": dict(tradable_grade="是",
        grade="A", sources=["yield", "lpr", "chinabond_yield"],
        evidence="方案的 carry 直接是它：10Y 1.59% / 2Y 1.22% 时预期只有 +4.4~4.6%，"
                 "7% 需要 carry 回到 2.5–3%",
        tradable="是：决定安全腿收益预期与仓位刻度",
        cited="doc/final_plan_20261109.md"),
    "融资融券（个股）": dict(tradable_grade="否",
        grade="B", sources=["margin_sh", "margin_sz", "szse_margin"],
        evidence="IC −0.0403（t −4.64）过预登记门槛；做成 30 只多头书净超额 **−1.06%（t −0.149）**，不如随机",
        tradable="否：有预测力、不值钱——IC 门槛对 30 只多头偏松",
        cited="doc/archive/margin_portfolio_result_20261111.md"),
    "融资融券（市场级杠杆）": dict(tradable_grade="否",
        grade="C", sources=["margin_sh", "margin_sz"],
        evidence="四条判据 T1 t −0.083、T3 t 0.307、T4 t −0.537，三条不过；增速形式整轮作废",
        tradable="否：整族关闭",
        cited="doc/market_leverage_result_20261114.md"),
    "股东户数变化": dict(tradable_grade="否",
        grade="C", sources=["gdhs"],
        evidence="IC −0.0282（t −7.01），方向六年一致，但只有门槛 0.031 的 **91%**",
        tradable="否：量级不够",
        cited="doc/archive/gdhs_ic_20261111.md"),
    "龙虎榜": dict(tradable_grade="否",
        grade="C", sources=["lhb"],
        evidence="IC 0.0308（差 0.0002 到门槛），**t 0.83**——这个量级与零无法区分",
        tradable="否",
        cited="doc/archive/lhb_ic_20261111.md"),
    "业绩报表": dict(tradable_grade="否",
        grade="C", sources=["yjbb"],
        evidence="IC **+0.0041**（t 0.24），门槛的 13%，连方向都谈不上",
        tradable="否",
        cited="doc/archive/perf_ic_20261111.md"),
    "业绩预告": dict(tradable_grade="否",
        grade="C", sources=["yjyg"],
        evidence="IC 0.0216（t 0.98），低于门槛 0.031，与零无法区分",
        tradable="否",
        cited="doc/archive/information_sources_result_20261108.md"),
    "限售解禁": dict(tradable_grade="否",
        grade="C", sources=["unlock"],
        evidence="条件化 IC −0.0463（t −2.82）看着强，但**覆盖率只有 4%**、全截面 IC −0.0100（t −1.80）、"
                 "且没扣成本、没做样本外——按多重检验应打折",
        tradable="否：覆盖太窄，30 只组合里平均只有 1 只可交易",
        cited="doc/archive/unlock_ic_20261108.md"),
    "价格 / 量（20 个技术特征）": dict(tradable_grade="否",
        grade="C", sources=["stock_daily", "stock_daily_sina", "etf_daily"],
        evidence="最强 body_ratio IC 0.0093（t −2.20），其余更弱；|IC| 前 5 名建的书测试窗 −4.97%，与随机无差别",
        tradable="否：1 日反转的毛额被成本吃掉",
        cited="doc/archive/information_sources_20261108.md"),
    "估值（PE/PB、bp）": dict(tradable_grade="部分",
        grade="B", sources=["valuation", "index_pe"],
        evidence="bp 有正 IC +0.0387（t 2.95），但**集中在 2022-2024**（2019-21 t 1.09 vs 2022-24 t 6.09），"
                 "CAPM 研究判定为**风格暴露**而不是 alpha",
        tradable="作为风格/风险预算可用；作为选股信号不能主张",
        cited="doc/archive/dividend_sleeve_20261108.md"),
    "资金流（个股）": dict(tradable_grade="未知",
        grade="U", sources=["fund_flow"],
        evidence="归档只有 **0.42 年**，相关可算、IC 不可算；归档仍在每周累积",
        tradable="未知：样本不足",
        cited="doc/archive/fund_flow_20261108.md"),
    "宏观月度（CPI/PPI/PMI/M2/GDP/社融）": dict(tradable_grade="未知",
        grade="U", sources=["cpi", "ppi", "pmi", "m2", "gdp", "shrzgm"],
        evidence="从未在本仓库检验；月/季频与日频决策的期限不匹配，且样本只有约 200 个月",
        tradable="未知",
        cited="—"),
    "指数 PE": dict(tradable_grade="未知",
        grade="U", sources=["index_pe"],
        evidence="可得且日频，但从未检验；理论上解释**多年**收益而非次日收益",
        tradable="未知",
        cited="—"),
    "ETF 折溢价": dict(tradable_grade="部分",
        grade="B", sources=["etf_daily", "fund_pingzhong"],
        evidence="513500 在 2024 年后中位溢价 **4.00%**（p95 9.52%）；20 日持有成本均值 −0.09pp，"
                 "**溢价均值回复，硬闸门不成立**",
        tradable="是：作为买入时点的熔断条件，不是收益信号",
        cited="doc/etf_premium_20261114.md"),
    "ETF 净价 / 分红": dict(tradable_grade="是",
        grade="A", sources=["etf_daily", "etf_dividend", "fund_open"],
        evidence="方案的执行价与分红入账都建立在它上面；账本 replay 与快照可逐位对上",
        tradable="是：已落地",
        cited="doc/account_ledger_20261111.md"),
    "分钟线 / 实时快照 / IOPV": dict(tradable_grade="否",
        grade="D", sources=["stock_min", "index_min", "stock_spot", "etf_spot"],
        evidence="东财 push2 在本机被拒；历史 IOPV 不对外发布，只能前向累积",
        tradable="否：拿不到",
        cited="doc/data_sources_20261114.md"),
    "行业板块 / 基金持仓": dict(tradable_grade="未知",
        grade="U", sources=["industry", "fund_hold"],
        evidence="接口部分被拒或返回非 JSON；从未检验",
        tradable="未知",
        cited="—"),
}

WEIGHTS = dict(evidence=0.5, tradable=0.3, available=0.2)
GRADE_SCORE = {"A": 10.0, "B": 6.5, "C": 2.5, "D": 0.0}
TRADABLE_SCORE = {"是": 10.0, "部分": 6.0, "未知": None, "否": 0.0}


def load_probes():
    """Both probe files, keyed by id."""
    probes = {}
    for name in ("catalog.json", "primary.json"):
        path = CATALOG_DIR / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            probes[row["id"]] = row
    return probes


def availability(probe, today):
    """0-10 for "can this be fetched, and is it current", computed from the probe record."""
    from datetime import date

    if probe is None:
        return None
    if probe["status"] in ("blocked", "timeout"):
        return 0.0
    if probe["status"] == "empty":
        return 6.0                      # answered, just no row for the sample asked about
    if probe["status"] != "ok":
        return 0.0
    last = probe.get("last_date")
    if not last:
        return 5.0                      # answers, but the row carries no date to judge freshness
    days = (date.fromisoformat(today) - date.fromisoformat(last)).days
    if days <= 5:
        return 10.0
    if days <= 45:
        return 7.0                      # monthly series, still current
    if days <= 200:
        return 3.0
    return 1.0


def score_family(name, entry, probes, today):
    graded = GRADE_SCORE.get(entry["grade"])
    tradable = TRADABLE_SCORE.get(entry.get("tradable_grade", "未知"))
    avail_values = [availability(probes.get(source), today) for source in entry["sources"]]
    avail_values = [value for value in avail_values if value is not None]
    available = max(avail_values) if avail_values else None
    if graded is None or tradable is None or available is None:
        total = None
    else:
        total = (WEIGHTS["evidence"] * graded + WEIGHTS["tradable"] * tradable
                 + WEIGHTS["available"] * available)
    return dict(family=name, grade=entry["grade"], evidence=entry["evidence"],
                tradable=entry["tradable"], cited=entry["cited"], sources=entry["sources"],
                evidence_score=graded, tradable_score=tradable, available_score=available,
                effectiveness=round(total, 2) if total is not None else None)


def run(today=None, output=OUTPUT, log=print):
    from datetime import date

    today = today or date.today().isoformat()
    probes = load_probes()
    rows = [score_family(name, entry, probes, today) for name, entry in EVIDENCE.items()]
    ranked = sorted(rows, key=lambda row: (row["effectiveness"] is None, -(row["effectiveness"] or 0)))
    for row in ranked:
        score = "未知" if row["effectiveness"] is None else "%.1f" % row["effectiveness"]
        log("%-34s 等级 %s  有效程度 %-5s  （证据 %s / 可交易 %s / 可得 %s）" % (
            row["family"], row["grade"], score,
            _fmt(row["evidence_score"]), _fmt(row["tradable_score"]), _fmt(row["available_score"])))
    result = dict(generated_at=today, weights=WEIGHTS, rows=ranked)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log("wrote %s" % output)
    return result


def _fmt(value):
    return "—" if value is None else ("%.1f" % value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Effectiveness scorecard for the data families.")
    parser.add_argument("--today", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    result = run(args.today, args.output)
    print(json.dumps([{key: row[key] for key in ("family", "grade", "effectiveness")}
                      for row in result["rows"]], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
