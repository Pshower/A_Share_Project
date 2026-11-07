# 本地产物目录 `reports/runs/`

更新日期：2026-11-07（**已执行一次分层清理**，见第 4 节）。

这个目录保存每一轮实验的**机器产物**：逐检查点回测曲线、组合净值、交易流水、因子面板、
训练轨迹和模型权重。它**不进 Git**（`.gitignore` 中的 `reports/runs/`），因此：

- 其他机器**没有**这些文件，只能按运行说明重跑；
- **结论不在这个目录里**。每一轮的人类可读报告在 [`doc/`](../doc/README.md)：2026-11 之后的
  研究写在 `doc/*.md`，2026-10 及更早的写在 `doc/reports/*.md`；
- 这里的路径在文档中作为**本地证据位置**引用，不是可分发资源。

单位说明：本文一律用**十进制 GB**（÷10⁹）。此前版本写的 "28.7 GB" 实际是 28.7 GiB，
换成十进制是 30.1 GB；两种口径不要混用。

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
| `runs/<config>/training_trace/steps_*.npz` | 每 512 步一块，**67 个数组 × (步数 × 300 只)**：观测、原始与映射后动作、开/收盘账户状态、逐只成交结果、`reward`、`value`、`log_prob` | 需重训；**2026-11-07 已清理** |
| `runs/<config>/training_trace/orders.jsonl.gz` | 每步一行的**可读**下单记录：side、目标权重、申报/成交股数、价、费、滑点、状态、拒因 | 需重训 |
| `runs/<config>/{initial,long,short,pilot}_model/model.zip` | 保存的 torch 权重 | 需重训；**保留** |
| `runs/<config>/events/`、`logs/` | 事件流与日志 | 需重跑 |

**判据是：能被直接读的是证据，只能被矩阵运算消费的是中间产物。**
名字里带 `report`/`summary`/`audit`/`protocol`/`manifest`/`seal`/`crc` 的属于前者；
`steps_*.npz`、`events/`、`logs/` 属于后者。`orders.jsonl.gz` 与 `index.json` 名字里带
`training_trace`，但属于前者。

## 2. 三层保留策略

| 层 | 内容 | 处置 |
| --- | --- | --- |
| **T1 证据** | 汇总 json/csv/md/png、审计、封存与协议，`index.json`、`orders.jsonl.gz`、`model.zip` | **永久保留** |
| **T2 稠密中间产物** | `steps_*.npz`（训练/滚动/探测/基准各类 trace 目录内的全部） | 可由同配置重训重建；**已于 2026-11-07 清理** |
| **T3 现役** | 2026-11 之后的全部 study | 全留，含中间产物 |

`model.zip` 从 T2 提升到 T1 的理由是实测出来的：2026-11-07 的
[先验归因](../doc/prior_attribution_20261107.md) 只用权重就回答了"策略到底学到了什么"——
12 个 run 训练后的排序与先验完全一致（Spearman 0.9999997、top-20 20/20）。
**权重能回答的问题，`steps_*.npz` 回答不了；而 `steps_*.npz` 占的体积是权重的 6 倍。**

### 当前占用（清理后，2026-11-07 实测）

| 分类 | 文件 | 大小 | 占总量 |
| --- | ---: | ---: | ---: |
| **全部** | 32,563 | **11.94 GB** | 100% |
| 可读下单流水 `orders.jsonl*` | 2,490 | 3.44 GB | 29% |
| 模型权重 `model.zip` | 810 | 3.07 GB | 26% |
| 其余 `.gz`（事件流、标签） | 4,032 | 1.88 GB | 16% |
| 面板与切片 `.parquet` | 3,436 | 1.37 GB | 11% |
| 表格 `.csv` | 8,966 | 1.06 GB | 9% |
| 其余 `.npz`（rollout 缓冲等） | 2,583 | 0.45 GB | 4% |
| 轨迹清单 `index.json` | 2,490 | 0.02 GB | ~0% |
| （其中报告与表格类证据合计） | 19,173 | 1.64 GB | 14% |

体积已经不在"训练轨迹"上了：剩下的大头是**可读流水、权重和事件流**，都属 T1 或与 T1 同级。

## 3. 完整目录表

按大小降序（清理后）。

| 目录 | 文件 | GB | 最后写入 |
| --- | ---: | ---: | --- |
| `ppo_counterfactual_learning_20261017` | 7229 | 2.62 | 2026-10-18 |
| `ppo_priority_rank_20261011_16k` | 2110 | 1.62 | 2026-10-11 |
| `ppo_exploration_20261006` | 3490 | 1.07 | 2026-10-07 |
| `ppo_random_matrix_20261003` | 1187 | 0.98 | 2026-10-03 |
| `ppo_matrix_20261003` | 652 | 0.89 | 2026-10-03 |
| `ppo_attribution_20261007` | 2113 | 0.68 | 2026-10-07 |
| `ppo_market_summary_20261007` | 1501 | 0.66 | 2026-10-07 |
| `agent_improvement_20260927` | 2164 | 0.49 | 2026-09-27 |
| `ppo_trade_actor_20261005` | 1273 | 0.47 | 2026-10-05 |
| `ppo_random_matrix_smoke_20261003` | 403 | 0.32 | 2026-10-03 |
| `ppo_matrix_smoke_20261003` | 298 | 0.30 | 2026-10-03 |
| `ppo_timing_phase_state_repair_20261015` | 1107 | 0.21 | 2026-10-16 |
| `ppo_timing_20261014` | 1426 | 0.19 | 2026-10-14 |
| `ppo_delta_actor_20261004` | 1165 | 0.16 | 2026-10-04 |
| `ppo_timing_phase_budget_20261015` | 725 | 0.15 | 2026-10-15 |
| `capacity_fixed_20260927` | 77 | 0.15 | 2026-09-27 |
| `workbench` | 1561 | 0.15 | 2026-11-03 |
| `sina_hs300_study_20260927` | 573 | 0.14 | 2026-09-27 |
| `capacity_20260927` | 122 | 0.12 | 2026-09-27 |
| `ppo_timing_credit_20261015` | 1036 | 0.11 | 2026-10-15 |
| `ppo_timing_margin_20261014` | 689 | 0.09 | 2026-10-15 |
| `ppo_counterfactual_pilot_20261016` | 214 | 0.09 | 2026-10-16 |
| `ppo_excess_small` | 197 | 0.04 | 2026-11-06 |
| `ppo` | 169 | 0.04 | 2026-11-02 |
| `ppo_lightweight` | 197 | 0.03 | 2026-09-25 |
| `ppo_return_small` | 146 | 0.03 | 2026-11-06 |
| `ppo_full_invest_3m` | 146 | 0.03 | 2026-11-06 |
| `ppo_full_invest_small` | 144 | 0.03 | 2026-11-06 |
| `ppo_excess_3m` | 142 | 0.03 | 2026-11-06 |
| `tradability_ledger_20261102` | 3 | 0.02 | 2026-11-02 |
| `ppo_random_matrix_comparison_20261003` | 4 | 0.01 | 2026-10-03 |
| 其余 32 个（`reversal_backtest`、`signal_*`、`price_feature_ic`、`alpha_*`、`prior_attribution`、`advice_*`、`stage3*`、`stage4*`、`validations`、`exposure_*`、`factor_baseline`、`ppo_checkpoints*`、`pit_availability*`、`autocorrelation`、`data_inventory*` 等） | 合计 300 | 均 < 0.01 | 2026-09~11 |

## 4. 清理记录：2026-11-07（**已执行**）

删掉的只有一类：**所有 trace 目录下的 `steps_*.npz`**。

| 项 | 数值 |
| --- | ---: |
| 删除文件 | **6,720** |
| 释放空间 | **18.16 GB** |
| 清理前 | 39,283 个文件 / 30.10 GB |
| 清理后 | 32,563 个文件 / **11.94 GB** |

范围核对（删除前逐条校验，任何一条不符即中止）：

- 全部位于 `reports/runs/` 之下；
- 文件名匹配 `steps_*.npz`；
- 路径中存在一个名字含 `trace` 的目录（`training_trace/`、`trace/`、`*_trace/`，
  以及它们下面的 `benchmark/`）。

保留且**未受任何改动**的：

| 类 | 清理前 | 清理后 |
| --- | ---: | ---: |
| `orders.jsonl*` | 2,490 | **2,490** |
| `index.json` | 2,490 | **2,490** |
| `model.zip` | 810 | **810** |

### 代价必须写清楚

`verify_trace_prefix` / `verify_priority_trace` / `verify_timing_trace` 之类的冷审计是
**对 chunk 的 sha256 逐块比对**。删除后：

- **1,951 个 trace 目录现在只剩 `orders.jsonl.gz` 和 `index.json`**，没有任何 chunk；
- 因此 `ppo_priority_rank_20261011_16k`、`ppo_counterfactual_learning_20261017`、
  `ppo_delta_actor_20261004` 等做过封存的研究，其 `independent_archive_crc.json`、
  `manifest.json`、`sampling_plan_seal.json` **无法再复核**——清单还在，它证明的对象没了；
- 这不改变那些研究的结论（结论由报告正文与 `*_audit.json` 承载），但审计状态从
  "可重校验"降级为"曾校验过"。

若将来需要重跑这些审计，只能按对应 `config.json` 重训，重训的轨迹**不会**与旧 chunk 逐字节相同，
所以旧 CRC 永远无法恢复。

## 5. 硬规则

- **不删 T1**：`*.md`、`report*.json`、`summary*.json`、`*audit*.json`、`*seal*`、
  `manifest.json`、`*crc*.json`、`metrics.csv`、`model_index.csv`、`*.png`，
  以及 `index.json`、`orders.jsonl.gz` 和 `model.zip`。它们是文档结论的凭据。
- 再要释放空间，下一站是**其余 `.gz` 事件流（1.88 GB）与 `parquet` 面板（1.37 GB）**；
  两者都还有二次分析价值，须逐轮决定，不按目录名批量套用。
- 新的一项研究落盘后，若总体积变化超过 1 GB，更新本文件第 2 节的占用表。
- 本文件随 Git 分发；`reports/runs/` 不随 Git 分发。
