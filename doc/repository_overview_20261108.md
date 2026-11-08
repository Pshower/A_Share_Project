# 仓库功能总览

日期：2026-11-08
状态：**清点完成**。工具 `tools/repository_inventory.py`，
明细见[仓库清点](repository_inventory_20261108.md)。

---

## 1. 规模

| 层 | 数量 |
| --- | ---: |
| `tools/` 研究脚本 | **80** |
| `src/` 模块 | **98** |
| `doc/` 文档 | 78 |
| `tests/` 测试文件 | 146（**2,739 个测试**全绿） |

## 2. `src/` 的六个子系统

### `data`（21 个模块）——数据层
下载（`download`、`sina`、`online`、`supplement`）、抓取节流（`rate_limit`）、
版本化特征构建（`preprocess`）、**时点可得性**（`pit`，按法定披露截止日推导 `available_from`）、
覆盖度核查（`coverage`、`check_missing_shares`）、历史重建（`rebuild_history`）、快照（`snapshots`）。

### `envs`（6 个）——撮合与持仓
`broker`（先卖后买、整手、部分成交、佣金/最低佣金/印花税/滑点）、`trading`（Gym 环境）、
`market`（特征与价格提供者）、`portfolio`、`initial_account`。

### `agents`（16 个）——策略与动作映射
PPO（`ppo_agent`/`ppo_policy`）、三种动作空间（权重 `action_mapping`、增量 `delta_mapping`、直接交易 `trade_mapping`）、
三条研究线各自的 agent（时序、优先级、市场摘要）。

### `training`（31 个）——训练与评估
训练循环（`train`）、环境工厂（`env_factory`）、评估（`evaluate`）、冻结（`freeze`）、
实验矩阵（`matrix_*`、`benchmark_capacity`）、以及各研究线的环境与轨迹（`research_*`、`timing_*`）。

### `backtest`（4 个）——回测入口
`run_strategy`/`run_policy`（统一执行与记账审计）、`strategies`（cash / buy_hold / equal_weight / momentum 基线）、`metrics`。

### `inference`（4 个）+ `web`（11 个）——落地
`eod`/`live`/`predict` 出当日建议；`web` 是一个本地服务（`serve`、`app`、`jobs`、`worker`、`schemas`），带任务队列与实验目录。

### `runtime`（4 个）
设备选择、取消事件（`events.check_cancel`）、SQLite 连接。

## 3. `tools/` 按研究线分（8 条线）

### 在用（44 个，11,190 行）

| 线 | 数量 | 行数 | 代表工具 |
| --- | ---: | ---: | --- |
| **数据建设** | 8 | 1,783 | `download_full_market`（分块可续）、`full_market_panel`（拼接）、`fetch_missing`（小批补漏）、`feature_panel`（1,800 只特征）、`archive_fund_flow`（滚动归档） |
| **特征与信号** | 10 | 2,527 | `price_feature_ic`（20 特征 IC）、`alpha_ic`（秩 IC + Newey-West）、`unlock_ic`（解禁）、`preannounce_ic`（业绩预告）、`feature_backtest`（样本外筛特征） |
| **回测与归因** | 14 | 4,051 | `capm_backtest`（CAPM 三检）、`equal_weight_crosscheck`（指数对照）、`selection_diagnostics`（基准规范）、`baseline_capm`、`prior_attribution`、`e1_reversal`、`allocation_study`、`tail_exposure` |
| **执行与撮合** | 2 | 905 | `daily_advice`、`shadow_run`（S2 前瞻影子运行） |
| **其它分析** | 10 | 1,924 | `checkpoint_compare`、`critic_initialization_probe`、绘图类 |

### 已结束（36 个，10,053 行，**占工具数 45%、代码行数约 47%**）

| 线 | 数量 | 行数 | 为什么结束 |
| --- | ---: | ---: | --- |
| **按时点干预** | 23 | 7,522 | 多次训练未动先验 |
| **反事实学习** | 9 | 2,205 | 未产生可认证效果 |
| **市场摘要** | 4 | 326 | 同上 |

## 4. 已确立的结论（这些是仓库现在的知识）

| 命题 | 证据 |
| --- | --- |
| **1/N 的优势在成本与宽度，不在收益** | 净 12.85% vs 市值加权 8.86%，但差距在下文被推翻 |
| **等权溢价是假的** | 静态分档 +7.92pp → PIT 动态 **+0.37pp**，指数对 **+0.59pp** |
| **CAPM 无法区分** | γ1 t = 0.67；合并 gap t = −0.89 |
| **价格特征无效** | 20 个方向全负、无一显著；样本外 −4.97% vs 随机 −4.43% |
| **E1 不通过** | V +1.85%（t 0.42）、full +0.25%（t 0.07） |
| **只有解禁过门槛** | 条件化 IC −0.0686（t −5.26），但覆盖仅 5.1% |
| **可执行基准是 5.27%，不是 8.02%** | 三 ETF 混合 TE 0.98%/2.57%，差 2.7pp 全在微盘 |

## 5. 清理：已删除（分支 `cleanup/archive-closed-lines`）

三条已结束的研究线**共 38 个工具、45 个测试文件已删除**（`git rm`）。删除前曾移入
`tools/archive/`、`tests/archive/` 并保留一个提交，确认主回归全绿后才执行删除。

| 项 | 清理前 | 清理后 |
| --- | ---: | ---: |
| `tools/` 脚本 | 80 | **43** |
| 测试文件 | 146 | **102** |
| 默认回归 | 2,739 passed | **1,715 passed** |

判断依据不是"有没有人引用"——同一研究线内部互相 import，80 个里只有 6 个引用为 0；
而是**这条线有没有结束**。

### 删除的可逆性

被删的代码仍在 `main` 的历史里，不在当前 `HEAD`。取回任何一个文件：

```powershell
git log --oneline --all -- 'tools/research_timing_credit.py'
git show <commit>:tools/research_timing_credit.py > 恢复.py
```

所以"删掉就不可复现"这个反对理由**在这个仓库里不成立**——它只在没有版本控制时成立。

### 保留了文档

12 篇协议与结果文档**没有删**（`doc/ppo_timing_*`、`doc/ppo_counterfactual_*`、
`doc/ppo_market_summary_*` 等）。它们是"没效果"这条结论**本身**的依据；
删了文档，删代码这件事就变成了没有记录的决定。

文档里的 `tools/<name>.py` 路径已失效，指向的是历史提交里的文件。
## 6. 复现

```powershell
python -B -m tools.repository_inventory
```
