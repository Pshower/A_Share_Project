# 本地产物目录 `reports/runs/`

更新日期：2026-11-07。

这个目录保存每一轮实验的**机器产物**：逐检查点回测曲线、组合净值、交易流水、因子面板、
训练轨迹和模型权重。它**不进 Git**（`.gitignore` 中的 `reports/runs/`），因此：

- 其他机器**没有**这些文件，只能按运行说明重跑；
- **结论不在这个目录里**。每一轮的人类可读报告在 [`doc/`](../doc/README.md)：2026-11 之后的
  研究写在 `doc/*.md`，2026-10 及更早的写在 `doc/reports/*.md`；
- 这里的路径在文档中作为**本地证据位置**引用，不是可分发资源。

## 1. 一个 run 目录的约定

绝大多数 PPO 研究使用同一套结构，认识一次就够：

| 名称 | 内容 | 重建难度 |
| --- | --- | --- |
| `report.md` / `report.json` | 自动生成的报告与汇总 | 由本轮小文件重算，秒级 |
| `protocol.json` / `frozen_experiment.json` | 预登记的配置与门槛 | **不可重建**，是证据本身 |
| `summary.json` / `study_summary.json` / `acceptance.json` | 跨配置汇总与验收判定 | 由本轮小文件重算 |
| `*_audit.json` / `independent_*.json` / `manifest.json` / `*seal.py` | 冷审计、封存与 CRC | **不可重建** |
| `metrics.csv` / `model_index.csv` / `comparison.png` | 汇总表与图 | 由本轮小文件重算 |
| `runs/<config>/` | 每个配置一次训练 | 需重训同一 seed 与预算 |
| `runs/<config>/training_trace/index.json` | 轨迹清单：contract 哈希、300 只代码、逐 chunk sha256 | 需重训 |
| `runs/<config>/training_trace/steps_*.npz` | 每 512 步一块，**67 个数组 × (步数 × 300 只)**：观测、原始与映射后动作、开/收盘账户状态、逐只成交结果、`reward`、`value`、`log_prob` | 需重训 |
| `runs/<config>/training_trace/orders.jsonl.gz` | 每步一行的**可读**下单记录：side、目标权重、申报/成交股数、价、费、滑点、状态、拒因 | 需重训 |
| `runs/<config>/{initial,long,short,pilot}_model/` | 保存的 torch 权重 | 需重训 |
| `runs/<config>/events/`、`logs/` | 事件流与日志 | 需重跑 |

**判据是：能被直接读的是证据，只能被矩阵运算消费的是中间产物。**
名字里带 `report`/`summary`/`audit`/`protocol`/`manifest`/`seal`/`crc` 的属于前者；
`steps_*.npz`、`*_model/`、`events/`、`logs/` 属于后者。
`orders.jsonl.gz` 与 `index.json` 名字里带 `training_trace`，但属于前者。

## 2. 三层保留策略

| 层 | 内容 | 处置 |
| --- | --- | --- |
| **T1 证据** | 上表所有 json/csv/md/png 汇总、审计、封存与协议，以及 `training_trace/index.json` 与 `orders.jsonl.gz` | **永久保留**。文档中的结论靠它复核 |
| **T2 稠密中间产物** | `training_trace/steps_*.npz`、`*_model/` 权重 | 可由同配置重训重建；方向关闭时可清理 |
| **T3 现役** | 2026-11 之后的全部 study（合计 < 200 MB） | 全留，含中间产物 |

`index.json` 与 `orders.jsonl.gz` 放在 T1 而不放 T2，是因为它们**不是稠密数组**：
一个是清单，一个是每步一行、可以直接读的下单流水（成交股数、价、费、滑点、拒因）。
真正占空间的 `steps_*.npz` 是"每一步 × 每一只 × 67 个字段"的观测/动作矩阵——它有审计价值，
但没有阅读价值。

### 当前占用（2026-11-07 实测）

| 分类 | 文件 | 大小 | 占总量 |
| --- | ---: | ---: | ---: |
| 全部 | 39,281 | **28.7 GB** | 100% |
| 逐更新快照 `steps_*.npz` | 4,412 | 16.0 GB | 56% |
| 模型权重 `*_model/` | 816 | 2.5 GB | 9% |
| 交易流水 `orders.jsonl.gz` | 182 | 3.1 GB | 11% |
| 清单 `index.json` | 182 | 2.8 MB | ~0% |
| 其余（报告、审计、面板、rollout 缓冲） | 33,689 | 7.1 GB | 25% |
| 其中报告与表格类证据（T1） | 19,171 | 1.57 GB | 5% |

```text
steps_*.npz       4,412 个文件  16.0 GB   ← 87% 的 T2 体积
*_model/            816 个文件   2.5 GB
orders.jsonl.gz     182 个文件   3.1 GB   ← 可读，保留
```

单个配置的量级：`ppo_priority_rank_20261011_16k/runs/2024_decoupled_ppo_42` 的
`training_trace/` 是 32 个 `steps_*.npz`（约 5.5 MB/块）+ 1 个 50.7 MB 的
`orders.jsonl.gz` + `index.json`，合计约 227 MB；其中可读的下单流水占 22%。

### 一个必须写下来的例外

`ppo_priority_rank_20261011_16k` 与 `ppo_counterfactual_learning_20261017` 做过**冷审计与封存**，
其 `independent_archive_crc.json`、`manifest.json`、`sampling_plan_seal.json` 记录的是
`training_trace/` 的 CRC。**清掉轨迹后这些清单无法再复核**——清单本身仍在，但它证明的对象没了。
这不改变那两轮的研究结论（结论写在 `doc/reports/` 里、由报告正文承载），但会退化成
"曾经校验过、现在无法重校验"。若打算清理，这两轮应单独决定。

## 3. 完整目录表

按大小降序。"轨迹/权重" 列是 `training_trace/` 与 `*_model/` 在该目录内的大小，
**其中约 16% 是可读的 `orders.jsonl.gz`**（属 T1）；把这一列减去 orders 才是该目录可清理的快照量。

| 目录 | 文件 | MB | 其中轨迹/权重 MB | 最后写入 |
| --- | ---: | ---: | ---: | --- |
| `ppo_priority_rank_20261011_16k` | 3142 | 5,924.6 | 5,424.9 | 2026-10-11 |
| `ppo_counterfactual_learning_20261017` | 8541 | 5,212.4 | 2,760.9 | 2026-10-18 |
| `ppo_exploration_20261006` | 4210 | 2,281.5 | 1,291.1 | 2026-10-07 |
| `ppo_market_summary_20261007` | 1981 | 2,055.6 | 1,807.8 | 2026-10-07 |
| `ppo_timing_20261014` | 2130 | 1,715.7 | 1,553.5 | 2026-10-14 |
| `ppo_timing_phase_state_repair_20261015` | 1715 | 1,702.7 | 1,582.1 | 2026-10-16 |
| `ppo_attribution_20261007` | 2569 | 1,688.3 | 1,243.0 | 2026-10-07 |
| `ppo_trade_actor_20261005` | 1569 | 1,452.7 | 1,284.3 | 2026-10-05 |
| `ppo_timing_phase_budget_20261015` | 1165 | 1,257.2 | 1,183.2 | 2026-10-15 |
| `ppo_random_matrix_20261003` | 1187 | 931.9 | 776.6 | 2026-10-03 |
| `ppo_timing_credit_20261015` | 1420 | 878.3 | 776.6 | 2026-10-15 |
| `ppo_matrix_20261003` | 652 | 852.4 | 776.6 | 2026-10-03 |
| `ppo_timing_margin_20261014` | 977 | 669.4 | 594.7 | 2026-10-15 |
| `agent_improvement_20260927` | 2164 | 466.6 | 7.9 | 2026-09-27 |
| `ppo_random_matrix_smoke_20261003` | 403 | 301.4 | 258.8 | 2026-10-03 |
| `ppo_matrix_smoke_20261003` | 298 | 285.0 | 258.8 | 2026-10-03 |
| `ppo_delta_actor_20261004` | 1165 | 154.7 | 14.1 | 2026-10-04 |
| `workbench` | 1561 | 146.9 | 4.3 | 2026-11-03 |
| `capacity_fixed_20260927` | 77 | 141.6 | 0.0 | 2026-09-27 |
| `sina_hs300_study_20260927` | 573 | 137.5 | 2.2 | 2026-09-27 |
| `capacity_20260927` | 122 | 119.0 | 0.0 | 2026-09-27 |
| `ppo_counterfactual_pilot_20261016` | 214 | 86.1 | 0.0 | 2026-10-16 |
| `ppo_excess_small` | 197 | 34.8 | 0.4 | 2026-11-06 |
| `ppo` | 169 | 34.5 | 0.6 | 2026-11-02 |
| `ppo_return_small` | 146 | 28.5 | 0.2 | 2026-11-06 |
| `ppo_excess_3m` | 142 | 27.7 | 0.2 | 2026-11-06 |
| `ppo_lightweight` | 197 | 27.5 | 0.6 | 2026-09-25 |
| `ppo_full_invest_3m` | 146 | 25.4 | 0.2 | 2026-11-06 |
| `ppo_full_invest_small` | 144 | 24.5 | 0.2 | 2026-11-06 |
| `tradability_ledger_20261102` | 3 | 15.0 | 0.0 | 2026-11-02 |
| `ppo_random_matrix_comparison_20261003` | 4 | 13.9 | 0.0 | 2026-10-03 |
| `factor_baseline` | 66 | 3.7 | 0.0 | 2026-10-23 |
| `pit_availability_20261102` | 4 | 3.2 | 0.0 | 2026-11-02 |
| `stage3_validation` | 18 | 1.6 | 0.0 | 2026-09-24 |
| `stage3_validation_reproduced` | 18 | 1.6 | 0.0 | 2026-09-24 |
| `stage3_validation_final` | 18 | 1.6 | 0.0 | 2026-09-24 |
| `stage4_baseline_regression` | 18 | 1.6 | 0.0 | 2026-09-25 |
| `20261101T160741219059Z` | 18 | 1.6 | 0.0 | 2026-11-02 |
| `advice_validation` | 26 | 1.3 | 0.0 | 2026-11-03 |
| `ppo_market_summary_20261007_diagnostics` | 1 | 1.1 | 0.0 | 2026-10-07 |
| `matrix_report_qa_j7bz7asy` | 22 | 0.6 | 0.0 | 2026-10-03 |
| `multiseed_20260926` | 2 | 0.4 | 0.0 | 2026-09-26 |
| `execution_source_audit_20260927` | 1 | 0.3 | 0.0 | 2026-09-27 |
| `advice_ppo_2023` | 2 | 0.3 | 0.0 | 2026-11-02 |
| `advice_20261102_lots` | 2 | 0.3 | 0.0 | 2026-11-04 |
| `advice_20261102` | 2 | 0.3 | 0.0 | 2026-11-02 |
| `reversal_backtest` | 4 | 0.3 | 0.0 | 2026-11-07 |
| `autocorrelation` | 4 | 0.2 | 0.0 | 2026-11-06 |
| `ppo_checkpoints_20261106` | 3 | 0.2 | 0.0 | 2026-11-06 |
| `ppo_checkpoints_3m_20261106` | 3 | 0.2 | 0.0 | 2026-11-06 |
| `sina_hs300_download_20260927` | 1 | 0.1 | 0.0 | 2026-09-27 |
| `alpha_budget` | 6 | 0.1 | 0.0 | 2026-11-06 |
| `data_inventory_20261105` | 3 | 0.1 | 0.0 | 2026-11-05 |
| `price_feature_ic` | 4 | 0.1 | 0.0 | 2026-11-06 |
| `exposure_stage2` | 10 | 0.0 | 0.0 | 2026-11-06 |
| `alternative_probe_20260927` | 5 | 0.0 | 0.0 | 2026-09-27 |
| `validations` | 5 | 0.0 | 0.0 | 2026-11-04 |
| `exposure_stage1` | 4 | 0.0 | 0.0 | 2026-11-05 |
| `signal_expansion` | 12 | 0.0 | 0.0 | 2026-11-05 |
| `signal_ic` | 7 | 0.0 | 0.0 | 2026-11-03 |
| `rebalance_sensitivity_20261104` | 3 | 0.0 | 0.0 | 2026-11-04 |
| `alpha_ic` | 2 | 0.0 | 0.0 | 2026-11-06 |

## 4. 清理 T2（本次**未执行**）

只删 `steps_*.npz` 与模型权重，保留 `index.json` 和 `orders.jsonl.gz`。先只列出目标：

```powershell
$snapshots = Get-ChildItem reports/runs -Recurse -File -Filter 'steps_*.npz'
$models = Get-ChildItem reports/runs -Recurse -Directory |
  Where-Object { $_.Name -in 'initial_model','long_model','short_model','pilot_model' }

'快照 {0:N1} GB / {1} 个文件' -f ((($snapshots | Measure-Object Length -Sum).Sum) / 1e9), $snapshots.Count
'权重 {0:N1} GB / {1} 个文件' -f ((($models | ForEach-Object { (Get-ChildItem -Recurse -File $_.FullName | Measure-Object Length -Sum).Sum } |
  Measure-Object -Sum).Sum / 1e9), $models.Count
```

确认后逐个删除（先跑 `-WhatIf` 看一遍）：

```powershell
$snapshots | Remove-Item -WhatIf
$models | Remove-Item -Recurse -WhatIf
```

预期释放 **18.5 GB**（16.0 GB 快照 + 2.5 GB 权重），保留 10.2 GB，其中可读交易流水 3.1 GB、
报告与表格类证据 1.57 GB 全在。

代价必须写清楚：`verify_trace_prefix` / `verify_priority_trace` 之类的冷审计是**对 chunk 的
sha256** 比对，快照删掉后这些审计无法重跑（审计结论仍在 `*_audit.json` 里，但它证明的对象没了）。
`ppo_priority_rank_20261011_16k`、`ppo_counterfactual_learning_20261017` 与
`ppo_delta_actor_20261004` 的 `sealed_inputs.json` 属于这一类，应逐轮决定是否整体保留。

## 5. 硬规则

- **不删 T1**：`*.md`、`report*.json`、`summary*.json`、`*audit*.json`、`*seal*`、
  `manifest.json`、`*crc*.json`、`metrics.csv`、`model_index.csv`、`*.png`，
  以及 `training_trace/index.json` 与 `orders.jsonl.gz`。它们是文档结论的凭据。
- 清理只针对 T2，且**逐轮决定**，不按目录名批量套用。
- 新的一项研究落盘后，若总体积变化超过 1 GB，更新本文件第 2 节的占用表。
- 本文件随 Git 分发；`reports/runs/` 不随 Git 分发。
