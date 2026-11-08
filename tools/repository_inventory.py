"""Inventory the repository: what each script is for and whether anything refers to it.

Deleting a research script is cheap until the day a report cites its output, so the rule
here is to classify and report rather than to remove anything.  A tool counts as live if
a test imports it, a document names it, or another tool imports it; otherwise it is a
candidate for retirement, and the decision is left to a human who knows whether the
experiment it belonged to is finished.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[1] / "tools").exists() is False else Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT

# Which research line each tool belongs to, decided by its name and the documents that
# describe it.  Anything unmatched falls through to "other" and is flagged.
LINES = {
    "数据建设": ("data_inventory", "full_market_panel", "download_full_market", "fetch_missing",
                 "derive_pit_availability", "feature_panel", "archive_fund_flow", "tradability_ledger"),
    "特征与信号": ("price_feature_ic", "signal_expansion", "signal_leaderboard", "alpha_factors",
                   "alpha_ic", "alpha_budget", "feature_backtest", "unlock_ic", "preannounce_ic",
                   "fund_flow_collinearity"),
    "回测与归因": ("capm_backtest", "equal_weight_crosscheck", "selection_diagnostics", "baseline_capm",
                   "prior_attribution", "exposure_metrics", "exposure_stage2", "rebalance_sensitivity",
                   "reversal_backtest", "factor_baseline", "e1_reversal", "allocation_study",
                   "tail_exposure", "autocorrelation_probe"),
    "执行与撮合": ("daily_advice", "shadow_run"),
    "按时点干预（已结束）": ("research_timing", "analyze_timing", "audit_timing", "seal_timing",
                              "benchmark_timing_policy", "timing_phase", "timing_counterfactual",
                              "timing_action_value_learning", "timing_causal_data", "timing_skip_sidecar"),
    "反事实学习（已结束）": ("research_counterfactual", "analyze_counterfactual", "audit_counterfactual",
                              "counterfactual_storage_check", "counterfactual_exact_schedule",
                              "run_counterfactual", "plot_counterfactual"),
    "市场摘要（已结束）": ("analyze_market_summary", "plot_market_summary", "run_market_summary"),
    "其它分析": ("analyze_delta_actor", "analyze_priority_execution", "analyze_trade_projection",
                 "analyze_credit_counterfactual", "plot_attribution", "plot_exploration",
                 "plot_priority_rank", "checkpoint_compare", "critic_initialization_probe",
                 "research_timing_phase_budget_repaired"),
}
FINISHED = {"按时点干预（已结束）", "反事实学习（已结束）", "市场摘要（已结束）"}


def line_of(name):
    for line, members in LINES.items():
        for member in members:
            if name == member or (member.endswith("_") is False and name.startswith(member + "_")):
                return line
            if name.startswith(member):
                return line
    return "其它分析"


def references(root=ROOT):
    """Every (file, text) pair that could mention a tool, excluding the tool itself."""
    corpus = []
    for folder in ("tests", "doc", "src", "tools", "configs"):
        base = root / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in (".py", ".md", ".json", ".txt"):
                continue
            try:
                corpus.append((path, path.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
    return corpus


def classify(root=ROOT):
    corpus = references(root)
    tools = sorted(path.stem for path in (root / "tools").glob("*.py"))
    rows = []
    for name in tools:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        tests = doc = other = src = 0
        for path, text in corpus:
            hits = len(pattern.findall(text))
            if not hits:
                continue
            if path.parent.name == "tests":
                tests += hits
            elif path.parent.name == "doc":
                doc += hits
            elif path.parent.name == "tools" and path.stem != name:
                other += hits
            elif path.parent.name == "src":
                src += hits
        rows.append(dict(tool=name, line=line_of(name), tests=tests, doc=doc, other=other, src=src,
                         lines=len((root / "tools" / f"{name}.py").read_text(encoding="utf-8",
                                                                             errors="ignore").splitlines()),
                         retired=line_of(name) in FINISHED))
    return rows


def modules(root=ROOT):
    rows = []
    for path in sorted((root / "src").rglob("*.py")):
        rows.append(dict(module=str(path.relative_to(root)).replace("\\", "/"),
                         lines=len(path.read_text(encoding="utf-8", errors="ignore").splitlines())))
    return rows


def documents(root=ROOT):
    rows = []
    for path in sorted((root / "doc").glob("*.md")):
        rows.append(dict(document=path.name,
                         lines=len(path.read_text(encoding="utf-8", errors="ignore").splitlines())))
    return rows


def tests(root=ROOT):
    rows = []
    for path in sorted((root / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        count = len(re.findall(r"^def test_", text, re.M))
        rows.append(dict(test=path.name, tests=count))
    return rows


def markdown(rows, module_rows, doc_rows, test_rows):
    total_tools = len(rows)
    retired = [row for row in rows if row["retired"]]
    by_line = defaultdict(list)
    for row in rows:
        by_line[row["line"]].append(row)
    lines = ["# 仓库清点", "",
             "工具 %d 个、`src` 模块 %d 个、文档 %d 篇、测试文件 %d 个。"
             % (total_tools, len(module_rows), len(doc_rows), len(test_rows)), ""]
    if retired:
        lines += ["**属于已结束研究线的工具：%d 个，合计 %d 行。** 引用计数不能判定它们该删："
                  "同一研究线内部互相 import，会让每个脚本都显得\"被引用\"。判定依据是这条线有没有结束。"
                  % (len(retired), sum(row["lines"] for row in retired)), ""]
    for line in sorted(by_line, key=lambda name: (name in FINISHED, name)):
        members = by_line[line]
        alive = [row for row in members if not row["retired"]]
        stamp = "已结束" if members[0]["retired"] else "在用"
        lines += ["## %s — %s（%d 个，合计 %d 行）" % (
            line, stamp, len(members), sum(row["lines"] for row in members)), "",
                  "| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
        for row in sorted(members, key=lambda item: -item["lines"]):
            state = "**已结束**" if row["retired"] else "在用"
            lines.append("| `%s` | %d | %d | %d | %d | %s |" % (
                row["tool"], row["lines"], row["tests"], row["doc"], row["other"], state))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inventory the repository.")
    parser.add_argument("--output", default=str(ROOT / "doc" / "repository_inventory_20261108.md"))
    args = parser.parse_args(argv)
    rows = classify()
    module_rows, doc_rows, test_rows = modules(), documents(), tests()
    Path(args.output).write_text(markdown(rows, module_rows, doc_rows, test_rows), encoding="utf-8")
    retired = [row["tool"] for row in rows if row["retired"]]
    print(json.dumps(dict(tools=len(rows), modules=len(module_rows), docs=len(doc_rows),
                          tests=len(test_rows), retired=len(retired),
                          retired_names=retired), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
