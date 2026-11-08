# 仓库清点

工具 80 个、`src` 模块 98 个、文档 78 篇、测试文件 146 个。

**属于已结束研究线的工具：36 个，合计 10053 行。** 引用计数不能判定它们该删：同一研究线内部互相 import，会让每个脚本都显得"被引用"。判定依据是这条线有没有结束。

## 其它分析 — 在用（10 个，合计 1924 行）

| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `analyze_delta_actor` | 428 | 1 | 2 | 1 | 在用 |
| `analyze_priority_execution` | 278 | 0 | 1 | 1 | 在用 |
| `analyze_trade_projection` | 242 | 0 | 1 | 3 | 在用 |
| `plot_priority_rank` | 217 | 0 | 1 | 3 | 在用 |
| `critic_initialization_probe` | 179 | 1 | 3 | 1 | 在用 |
| `repository_inventory` | 168 | 0 | 1 | 0 | 在用 |
| `checkpoint_compare` | 130 | 1 | 5 | 1 | 在用 |
| `analyze_credit_counterfactual` | 112 | 3 | 1 | 4 | 在用 |
| `plot_attribution` | 99 | 0 | 1 | 1 | 在用 |
| `plot_exploration` | 71 | 0 | 1 | 1 | 在用 |

## 回测与归因 — 在用（14 个，合计 4051 行）

| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `capm_backtest` | 778 | 2 | 9 | 9 | 在用 |
| `reversal_backtest` | 349 | 1 | 11 | 1 | 在用 |
| `autocorrelation_probe` | 342 | 1 | 5 | 1 | 在用 |
| `prior_attribution` | 329 | 1 | 7 | 1 | 在用 |
| `exposure_metrics` | 290 | 1 | 5 | 5 | 在用 |
| `baseline_capm` | 269 | 1 | 8 | 8 | 在用 |
| `selection_diagnostics` | 252 | 1 | 6 | 2 | 在用 |
| `e1_reversal` | 243 | 0 | 6 | 2 | 在用 |
| `factor_baseline` | 241 | 1 | 7 | 1 | 在用 |
| `equal_weight_crosscheck` | 239 | 2 | 13 | 2 | 在用 |
| `exposure_stage2` | 222 | 2 | 9 | 2 | 在用 |
| `allocation_study` | 185 | 0 | 5 | 1 | 在用 |
| `tail_exposure` | 171 | 0 | 4 | 1 | 在用 |
| `rebalance_sensitivity` | 141 | 1 | 4 | 1 | 在用 |

## 执行与撮合 — 在用（2 个，合计 905 行）

| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `daily_advice` | 751 | 1 | 7 | 2 | 在用 |
| `shadow_run` | 154 | 0 | 2 | 1 | 在用 |

## 数据建设 — 在用（8 个，合计 1783 行）

| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `data_inventory` | 584 | 1 | 10 | 1 | 在用 |
| `download_full_market` | 221 | 1 | 2 | 1 | 在用 |
| `tradability_ledger` | 219 | 1 | 4 | 1 | 在用 |
| `full_market_panel` | 217 | 1 | 3 | 7 | 在用 |
| `fetch_missing` | 147 | 6 | 3 | 1 | 在用 |
| `derive_pit_availability` | 143 | 1 | 5 | 1 | 在用 |
| `feature_panel` | 129 | 0 | 3 | 4 | 在用 |
| `archive_fund_flow` | 123 | 0 | 5 | 1 | 在用 |

## 特征与信号 — 在用（10 个，合计 2527 行）

| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `alpha_ic` | 515 | 1 | 10 | 21 | 在用 |
| `alpha_budget` | 404 | 1 | 12 | 5 | 在用 |
| `signal_expansion` | 353 | 1 | 11 | 3 | 在用 |
| `alpha_factors` | 250 | 2 | 4 | 20 | 在用 |
| `feature_backtest` | 247 | 0 | 3 | 1 | 在用 |
| `price_feature_ic` | 199 | 1 | 10 | 21 | 在用 |
| `signal_leaderboard` | 148 | 2 | 4 | 2 | 在用 |
| `preannounce_ic` | 145 | 0 | 4 | 1 | 在用 |
| `unlock_ic` | 139 | 0 | 6 | 1 | 在用 |
| `fund_flow_collinearity` | 127 | 0 | 4 | 1 | 在用 |

## 反事实学习（已结束） — 已结束（9 个，合计 2205 行）

| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `research_counterfactual_learning` | 702 | 1 | 6 | 4 | **已结束** |
| `audit_counterfactual_learning` | 623 | 1 | 1 | 0 | **已结束** |
| `analyze_counterfactual_pilot` | 373 | 1 | 1 | 1 | **已结束** |
| `plot_counterfactual_learning` | 158 | 1 | 1 | 0 | **已结束** |
| `plot_counterfactual_pilot` | 110 | 1 | 1 | 0 | **已结束** |
| `counterfactual_storage_check` | 81 | 1 | 1 | 3 | **已结束** |
| `counterfactual_exact_schedule` | 69 | 5 | 1 | 6 | **已结束** |
| `run_counterfactual_control` | 46 | 1 | 4 | 0 | **已结束** |
| `run_counterfactual_holdout` | 43 | 1 | 4 | 0 | **已结束** |

## 市场摘要（已结束） — 已结束（4 个，合计 326 行）

| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `analyze_market_summary_projection` | 115 | 1 | 1 | 0 | **已结束** |
| `plot_market_summary` | 100 | 0 | 1 | 1 | **已结束** |
| `run_market_summary_tail` | 64 | 1 | 1 | 0 | **已结束** |
| `run_market_summary_worker` | 47 | 2 | 1 | 1 | **已结束** |

## 按时点干预（已结束） — 已结束（23 个，合计 7522 行）

| 工具 | 行数 | 测试 | 文档 | 其它工具 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `research_timing_phase_budget` | 754 | 2 | 7 | 4 | **已结束** |
| `research_timing_credit` | 665 | 2 | 2 | 7 | **已结束** |
| `research_timing_margin` | 572 | 2 | 3 | 7 | **已结束** |
| `timing_counterfactual_v2` | 531 | 1 | 1 | 2 | **已结束** |
| `research_timing_counterfactual_pilot` | 499 | 1 | 6 | 3 | **已结束** |
| `audit_timing_study` | 492 | 1 | 2 | 3 | **已结束** |
| `timing_counterfactual` | 424 | 4 | 2 | 4 | **已结束** |
| `audit_timing_phase_budget` | 386 | 1 | 1 | 4 | **已结束** |
| `seal_timing_credit` | 362 | 1 | 2 | 1 | **已结束** |
| `timing_causal_data` | 311 | 1 | 1 | 4 | **已结束** |
| `audit_timing_credit` | 301 | 3 | 1 | 6 | **已结束** |
| `timing_action_value_learning` | 284 | 1 | 1 | 4 | **已结束** |
| `seal_timing_phase_budget` | 278 | 1 | 2 | 2 | **已结束** |
| `audit_timing_margin` | 275 | 1 | 1 | 1 | **已结束** |
| `analyze_timing_phase_budget` | 240 | 3 | 1 | 5 | **已结束** |
| `research_timing_phase_budget_repaired` | 225 | 2 | 6 | 1 | **已结束** |
| `timing_phase_agent` | 193 | 5 | 3 | 3 | **已结束** |
| `analyze_timing_credit` | 187 | 3 | 1 | 5 | **已结束** |
| `analyze_timing` | 171 | 1 | 1 | 4 | **已结束** |
| `analyze_timing_margin` | 157 | 1 | 1 | 2 | **已结束** |
| `benchmark_timing_policy` | 89 | 0 | 1 | 1 | **已结束** |
| `timing_phase_policy` | 67 | 4 | 2 | 4 | **已结束** |
| `timing_skip_sidecar` | 59 | 0 | 1 | 1 | **已结束** |
