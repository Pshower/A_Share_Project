# A 股执行层与影子运行 Pipeline

本文件描述**当前分支（`execution`）**的实际结构。PPO 训练、交易环境、实验工作台这些已关闭的
研究线在 2026-11-14 从本分支移除，代码与历史完整保留在 `research/supply-pressure`。
结论与判断不在本文件，在 [`doc/implementation_plan.md`](doc/implementation_plan.md) 与
[`doc/README.md`](doc/README.md) 的索引里。

## 1. 这个仓库现在做什么

三层，彼此不共享代码：

| 层 | 做什么 | 入口 |
| --- | --- | --- |
| **数据层** | 抓取、清洗、构建特征面板（前复权 + 不复权 + 流通股） | `src/data/` |
| **执行层** | 三本账（live / paper / backtest）、CPPI 目标权重、盘前闸门、每日检查 | `tools/account_ledger.py`、`tools/plan_orders.py`、`tools/plan_backtest.py` |
| **影子层** | 把 E1 反转规则放到前向面板上逐日记录，只花日历时间、不花样本 | `tools/shadow_run.py`、`tools/forward_panel.py` |

研究侧的读数工具只有一个还在活跃：`tools/market_leverage.py`（市场级杠杆闸门，已按预登记关闭）。

## 2. 目录

| 路径 | 内容 |
| --- | --- |
| `src/data/preprocess.py` | `DataPreprocessor`：全市场面板构建、特征、scaler、manifest |
| `src/data/online.py`、`snapshots.py`、`rate_limit.py` | 在线抓取、快照登记、限速与重试 |
| `src/data/feature_definitions.py` | 特征定义与校验 |
| `tools/` | 执行层 5 个、影子 2 个、研究 2 个、支撑 1 个（见下） |
| `src/web/` + `web/` | 只读工作台：三本账与影子运行的一个页面 |
| `data/` | `clean/`（面板与池）、`online/`（抓取归档）、`raw/`（原始） |
| `reports/account/<book>/` | 每本账的 `account.json` + `trades.jsonl` + `snapshots.jsonl` |
| `reports/shadow/` | `e1_shadow.jsonl`（只追加）与 `e1_state.json` |

工具清单：`account_ledger`（账本/闸门/报表）、`plan_orders`（CPPI → 整手订单）、
`plan_backtest`（真实 ETF 回测与导出）、`mark_when_ready`（等报价发布再补当日净值）、
`audit_marks`（按当日真实收盘价重算并修复）、`shadow_run`（S2 记录器）、`forward_panel`（前向面板）、
`market_leverage`（杠杆闸门）、`stats`（Newey-West）、`trading_calendar`。

## 3. 执行层的不变量

1. **成交流水是唯一真相源**。现金与持仓一律 `replay(trades)` 推出；快照只存当日估值。
   任何"快照现金无法由流水重放"的书都算坏账（2026-11-14 就出过一次，见 `doc/implementation_plan.md`）。
2. **账本之间绝不共享文件**。`--account live|paper|backtest` 决定路径；测试写真实账本会被
   `tests/conftest.py` 直接判失败。
3. **不给历史补一天没发生过的净值**。`mark` 核对每个持仓腿的报价日期，不符就拒绝；
   报价未发布时用 `mark_when_ready` 等，不用 `--allow-stale` 硬写。
4. **回测账是封口件**（`sealed: true`），`mark` 拒绝给它加日期；要更新就重跑导出。
5. **只出信号、不下单**。没有任何代码路径会把订单发出去。

## 4. 影子运行（S2）

- 前向面板每个交易日全量重下（`forward_panel --end <date>`），池子 297 只，`601059` 是已知缺口
  （新浪无 K 线、东财只到 2026-09-14），**不是失败**。旧面板由 `prune(keep=3)` 轮换。
- 记录器 `shadow_run` 的构造参数**冻结**：20 个价格特征全取负号、top 30、跌出前 60 才换、
  20 日持有（研究面板与前向面板同为 298 只量级，所以缓冲带含义一致）。
- **节奏**：只在距入场满 20 个交易日时重判（`decision="rebalance"`），其余交易日沿用原名单
  （`decision="hold"`）。此前每天重判，实测 105× 年换手，价差会吃掉全部收益——见
  [影子节奏](doc/shadow_cadence_20261114.md)。
- 账本 `reports/shadow/e1_shadow.jsonl` 只追加，每天一行 `decided` / `stale` / `no_score`。
- `equal_weight_nav` 是持仓的**等权平均价格**（成分一变就跳），**不是收益**；
  要看收益读记录里的 `interval.return` 复合出的**链式净值**（页面上的"链式净值"）。

## 5. 每日与每周的自动任务

| 时间 | 任务 | 说明 |
| --- | --- | --- |
| 交易日 20:00 | S2 影子记录 | 先全量重下前向面板，再记录 |
| 交易日 **21:00** | 账户每日检查 | `sync-dividends → mark → check → curve`，live 与 paper 各一遍；报价未发布时改用 `mark_when_ready` |
| 每周六 18:00 | 资金流归档 | 为日后研究积累，暂不分析 |

21:00 这个时点是用实测定的：20:15 时四只 ETF 的当日收盘价还没落地，数据约 20:30 发布。

## 6. 数据口径

- **两条价格路径**：研究用前复权（`hfq`），执行与估值用不复权 + 真实流通股。两者不能混用。
- 新浪 ETF 历史**不做拆分调整**（513500 在 2022-03-30 按 1:2 拆分、510500 在 2022-08-29 丢失过一次分红），
  `price_and_payouts` 按比例修复并记入分红流水。
- 分红走 `action: "dividend"` 的现金流水，不改股数；`sync-dividends` 只给"除息日前严格持有"的份额入账。
- 保证金：`data/online/margin/` 有逐股归档（2019-2024）与两市官方合计（2010-2026，深交所缺日已补 150 天）。

## 7. 本地工具链

```powershell
# 解释器（本机）
D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe

# 测试：全离线，任何联网或被改动的真实账本都会让用例失败
& $py -B -m pytest -q -p no:cacheprovider

# 每日检查（只出信号）
& $py -B -m tools.account_ledger sync-dividends --account paper
& $py -B -m tools.account_ledger mark --date 2026-11-14 --account paper
& $py -B -m tools.account_ledger check --date 2026-11-14 --account paper
& $py -B -m tools.account_ledger curve --account paper

# 影子记录
& $py -B -m tools.forward_panel --end 2026-11-14
& $py -B -m tools.shadow_run --dataset data/clean/forward_pool_20261114

# 工作台（只读）
cd web; npm run build
& $py -B -m src.web.serve     # http://127.0.0.1:8765
```

## 8. 改动的边界

- 执行参数（楼层 0.90、倍数 6、不带走 5% 带宽）与影子构造参数**在预登记之外不得调整**。
- 任何新信号要先写预登记（判据在看到结果之前定死），并按 V / R / P 的样本账本记账；
  P 段（2026-07 起）**已被触碰**：plan_backtest 的 129 个起点中有 2026-07/08/09/10 四个落在这里，另有一次 topK 冒烟读过 2026-08~11 的净值（见 [方案敏感性](doc/plan_sensitivity_20261114.md#3-一个必须披露的样本问题)）。方案层的样本外验证改为**前向**：只用 2026-11 之后每天新落地的数据。
- 守卫与门槛的阈值必须先用**自变量自己的分布**校准再写死；查分布不算偷看结果。
