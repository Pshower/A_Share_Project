# 全仓审查与修改意见：执行层、每日推荐、回测与文档一致性

日期：2026-11-15（**第二轮外部审查已并入第 7 节**）。状态：**已完成两轮全仓审查**（第一轮覆盖 `src/`、`tools/`、`web/`、`tests/`、`reports/`、`pipeline.md` 与 `doc/`，定位 14 项并给出修改意见；第二轮基于提交 `7b47760`，针对每日唯一入口、全资产建议、中债指数、载体测量与特征审计再定位 11 项，含 5 项 P0）。
复现检查：`python -B -m pytest -q -p no:cacheprovider`、`python -B -m tools.doc_lint`。

---

## 0. 结论与问题总表

仓库近期完成了从「PPO 训练与横截面 Alpha 探索」向「执行层（四只 ETF 的 CPPI 三本账）+ 每日个股推荐与 S2 影子运行」的分支收敛（`execution` 分支）：移除了已关闭的 PPO 训练栈与 61 个旧研究脚本，提炼出三本独立账本（`live` / `paper` / `backtest`）、盘前闸门、每日推荐/影子记录页面及 207 项离线测试（含 `tests/conftest.py` 对真实账本的防写守卫）。整体架构清晰、边界诚实。

但在逐行审查新增的**每日推荐与影子运行链路**、**CPPI 回测与账本执行链路**、**周度资金流归档工具**以及**现行文档集**时，发现 **14 项具体问题**（其中 **3 项 P0 缺陷**会在下一个交易日 `2026-11-16` 或下周六定时任务中直接导致前端数据错位、推荐 `topK` 失序或归档空转）：

| 编号 | 级别 | 模块 / 文件 | 问题摘要 | 影响 |
| --- | --- | --- | --- | --- |
| **1.1** | **P0** | `src/web/execution.py` | 链式净值下标 `index` 仅在 `decided` 行自增，而 `chain` 按全部行（含 4 行 `stale`）追加 | 自 `2026-11-16` 起，「每日推荐」页的单日 `chained_nav` / `chained_pool` / `chained_hs300` 恒滞后 4 行（前 4 个交易日卡在 `1.0000`） |
| **1.2** | **P0** | `tools/shadow_run.py`、`tools/e1_rule.py`、`src/web/execution.py` | 20 日节奏下 `hold` 日（19/20 的交易日）写入 `score=None`，且 `decide()` 返回的 `book` 为「缓冲保留股 + 新补入股」未按分数全局排序 | 自 `2026-11-16` 起，「每日推荐」页在所有 `hold` 日选 `topK`（前 5/10/20）会切出错误的股票（漏掉排名前列的换入股），且分数与截断值全显示 `—` |
| **3.1** | **P0** | `tools/archive_fund_flow.py` | `latest = current["date"].max()` 取本地旧库最大日期做跳过条件，且 `rows_added` 在去重前统计 | **直接解释了 11-14 周六归档「跳过 1,797 只、仅抓 2 只停牌股、净增 0 行、停在 11-06」的真实原因**：不是东财滞后 5 天，而是本地跳过逻辑让周度任务永远不再抓取正常股票 |
| **1.3** | **P1** | `web/src/ExecutionPage.tsx`、`src/web/execution.py` | 前端读取 `selected.interval_from`，后端只返回 `selected.interval`（含 `"from"` 键） | 「每日推荐」页「本段收益」卡片下方起点日期恒显示 `自 —` |
| **1.4** | **P1** | `tools/shadow_run.py`、`reports/shadow/e1_state.json` | `interval["sessions"]` 误填累计持有天数而非本段交易日数；`e1_state.json` 持 `11-13` 名单却标 `entry_date="2026-11-10"` | 区间元数据语义混淆；第一期 20 日持有期实际只持 `11-13` 名单 17 个交易日就会触发再平衡，且漏记 `11-10~11-13` 的区间收益 |
| **1.5** | **P1** | `tools/shadow_run.py` | 再平衡日 $t$ 收盘出信号后，次日区间收益直接按新书算 `close[t+1]/close[t]-1` | 隐含假设在 $t$ 日收盘价零成本瞬间换仓，未对齐 `e1_topk.py` 的「$t+1$ 开盘价成交 + 整手 + 费用」口径 |
| **2.1** | **P1** | `tools/account_ledger.py`、`tools/plan_backtest.py` | `record_trade` 与 `gate` 对四只 ETF 卖出征收 `0.05%` 印花税；`fee_for` 未计印花税使 `plan_backtest` 卖出不收税 | 场内 ETF（`510300`/`510500`/`513500`/`511010`）法定免征印花税；当前实盘/模拟盘账本多扣卖出税且与回测账口径不一致 |
| **2.2** | **P1** | `tools/plan_backtest.py` | `simulate_book` 用 $t$ 日收盘价算目标权重并立即按 $t$ 日收盘价成交 | 实盘/模拟盘为「$t$ 日 21:00 收盘价决策、$t+1$ 日开盘价成交」，回测漏掉了 $t$ 收盘至 $t+1$ 开盘的隔夜跳空风险 |
| **2.3** | **P1** | `tools/plan_orders.py`、`tools/account_ledger.py` | `plan_orders.orders()` 按 `100%` 净值向下取整手，未预留买入手续费；`book()` 逐笔写文件无前置整批现金校验 | 当四腿取整余量小于总佣金（或次日开盘价微涨）时，`gate` 会因差几块钱拦截订单，`book()` 则会在写完前 3 笔后第 4 笔抛错留下半截脏流水 |
| **2.4** | **P1** | `tools/account_ledger.py` | `state()` 中 `peak = peak_of(snapshots, capital)` 未与当日实时估值 `value` 取 `max` | 若按手册顺序先跑 `check`/`book` 再跑 `mark`，创新高当日会出现 `nav > peak`、`drawdown > 0`，且传给 CPPI 的峰值偏低 |
| **2.5** | **P2** | `tools/plan_backtest.py` | `price_and_payouts` 与 `total_return` 在拆分回溯缩放历史价格时，未同步缩放拆分前的每份分红 | 若同一 ETF 在拆分前有过现金分红，拆分前股数翻倍而每份分红未折半，会高估拆分前分红收益 |
| **3.2** | **P2** | `tools/market_leverage.py` | `signal="growth"` 时 `pre_v_2017_2018` 仍回归全 `NaN` 的 `r20`，向 JSON 写入非法字面量 `NaN`；输出文件名未区分 `signal` | `reports/runs/market_leverage/gate_both_long.json` 第 107–109 行含非标准 JSON `NaN`，严格 JSON 解析器会报错 |
| **3.3** | **P2** | `tools/survivorship.py`、`src/runtime/device.py`、`tools/backup.py` | `survivorship.py` 存在 `return` 后死代码；`device.py` 残留未使用的 `import torch`；`backup.py` 状态探针与多目标校验细节 | 代码整洁度与无 `torch` 环境导入兼容性问题；新插移动盘在首次备份前 `--status` 报 `reachable=False` |
| **4.1–4.2** | **P1/P2** | `tools/doc_lint.py`、`doc/*.md`、`pipeline.md` | `doc_lint` 仅匹配 `tools/<name>.py` 与 `.md` 链接，漏检 `python -m tools.<name>` / `src.<module>`；多篇现行文档残留 `+5.3%` 旧值、已删命令与错误链接 | 现行手册（`research_workflow.md`、`online_market_runbook.md` 等）仍指导读者执行已删除模块，文档间关键数字存在冲突 |

---

## 1. 每日股票推荐与影子运行（S2）审查意见

### 1.1 [P0] `src/web/execution.py` 链式净值下标错位（`chain[index]` 滞后于真实行）

- **位置**：[`src/web/execution.py`](../src/web/execution.py) 第 142–196 行。
- **现象与根因**：
  第一个循环遍历 `rows` 的**每一行**（包含 `decided` 与 `stale`）构建 `chain`，因此 `len(chain) == len(rows)`。但第二个循环在遇到非 `decided` 行时直接 `continue`，把 `index += 1` 放在了循环末尾：
  ```python
  # src/web/execution.py:144-196
  for row in rows:
      ...
      chain.append((level, pool_level, index_level))
  index = 0
  for row in rows:
      if row.get("status") != "decided" or not row.get("book"):
          continue
      ...
          chained_nav=round(chain[index][0], 6),
          chained_pool=round(chain[index][1], 6),
          chained_hs300=round(chain[index][2], 6),
      ...
      index += 1
  ```
  在真实账本 `reports/shadow/e1_shadow.jsonl` 中，第 0 行是 `decided`（`2024-12-31`），第 1~4 行是 `stale`，第 5~8 行是 `decided`（`2026-11-10` ~ `11-13`）。由于前 9 行都是旧记录（无 `interval`），`chain[0..8]` 恰好全是 `(1.0, 1.0, 1.0)`，掩盖了错位；但从下一个交易日 **`2026-11-16`（第 9 行，第 6 个 `decided`）** 起，`rows[9]` 写入第一笔真实 `interval.return`，此时顶层 `shadow["chained_nav"]` 会更新，而 `days[-1]["chained_nav"]` 读取的却是 `chain[5]`（`1.0000`），**在接下来 4 个交易日里一直卡在 `1.0000`，且此后永远滞后 4 个交易日（若未来再有 `stale` 行则滞后更多）**。
- **修改建议**：
  删除手动维护的 `index = 0` 与 `index += 1`，直接用 `for index, row in enumerate(rows):`，并在 `tests/test_web_execution.py` 的 `test_the_api_chains_the_benchmark_columns_too` 中插入 `stale` 行断言 `days[-1]["chained_nav"] == data["chained_nav"]`。

### 1.2 [P0] `hold` 日 `score=None` 叠加 `book` 未按分数排序，导致「每日推荐」页 `topK` 切片失序

- **位置**：[`tools/shadow_run.py`](../tools/shadow_run.py) 第 229–240 行、[`tools/e1_rule.py`](../tools/e1_rule.py) 第 74–81 行、[`src/web/execution.py`](../src/web/execution.py) 第 166–187 行、[`web/src/ExecutionPage.tsx`](../web/src/ExecutionPage.tsx) 第 89 行。
- **现象与根因**：
  1. `e1_rule.decide()` 先保留仍在前 60 名缓冲区内的旧持仓 `keep`，再从全池榜首依次补足 30 只：
     ```python
     keep = [code for code in held if rank_of.get(code, 10 ** 9) < buffer]
     keep = sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))[:top_k]
     for code in order.index:
         if len(keep) >= top_k:
             break
         if code not in keep:
             keep.append(code)
     return keep
     ```
     因此返回的 `book` 列表顺序是 **「缓冲区保留股（名次可到第 59 名）在前、新换入的高排名股（名次在第 1~30 名）在后」**。例如 `reports/shadow/e1_state.json` 中第 20 位 `300896` 的分数是 `0.3283`（第 30 名），而排在列表第 21 位的换入股 `603260` 分数是 `0.2516`（第 7 名）。
  2. 在 `rebalance` 日，`shadow_run.py` 记录了 `score` 字典，`src/web/execution.py` 第 167 行会用 `sorted(book, key=lambda code: scores.get(code, float("inf")))` 重新按分数排序。
  3. 但在改为 20 日一判后，**每 20 个交易日里有 19 天是 `decision="hold"`**。在 `hold` 日，`shadow_run.py` 第 240 行执行：
     ```python
     else:
         score, book = None, list(held)
     ```
     此时 `scores` 为空字典，`scores.get(code, float("inf"))` 对所有股票都返回 `inf`，`sorted()` 退化为保持 `book` 原顺序（即保留股在前、换入股在后）。
  4. 用户在「每日推荐」页点击 `前 5 / 前 10 / 前 20` 时，前端 `selected.names.slice(0, topK)` 切出的是未按分数排序的前 $K$ 只，且所有股票的「分数」列和顶部「分数截断」卡片全部变成 `—`。
  5. 现有测试 `test_each_days_names_are_ordered_so_a_topk_slice_is_meaningful` 之所以未报错，是因为它写了 `scores = [item["score"] for item in day["names"] if item["score"] is not None]`——当 `score` 全为 `None` 时 `scores == []`，`[] == sorted([])` 恒为真。
- **修改建议**：
  1. 在 `tools/e1_rule.py::decide` 返回前按当日 `rank_of` 对最终 30 只 `keep` 统一排序：`return sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))`；
  2. 在 `tools/shadow_run.py` 中，即使 `due=False`（`hold` 日），也调用 `score = composite(panel, codes, features, latest)` 记录当日 30 只持仓的最新截面分（或保留入场日分数），使「每日推荐」页每天都有可核对的分数与 `topK` 截断值；
  3. 在 `tests/test_web_execution.py` 中断言 `len(scores) == 30`（不允许 `score` 全空蒙混过关）。

### 1.3 [P1] `web/src/ExecutionPage.tsx` 字段名不一致导致「本段收益」起点恒显示 `自 —`

- **位置**：[`web/src/ExecutionPage.tsx`](../web/src/ExecutionPage.tsx) 第 330 行、[`src/web/execution.py`](../src/web/execution.py) 第 176–177 行。
- **现象与根因**：
  `ExecutionPage.tsx` 第 330 行写的是：
  ```tsx
  <small>自 {selected.interval_from || '—'}</small>
  ```
  而 `src/web/execution.py` 只返回了 `interval=interval`（结构为 `{"from": ..., "to": ..., "sessions": ..., "return": ...}`）和 `int_return=interval.get("return")`，从未生成 `interval_from` 字段。
- **修改建议**：
  在 `src/web/execution.py` 的 `days.append(...)` 中补上 `interval_from=interval.get("from")`，或将前端改为 `selected.interval?.from || '—'`（建议两端同时对齐，并重新执行 `npm --prefix web run build` 更新 `web/dist`）。

### 1.4 [P1] `tools/shadow_run.py` 中 `interval["sessions"]` 口径混淆与 `e1_state.json` 过渡期起点错位

- **位置**：[`tools/shadow_run.py`](../tools/shadow_run.py) 第 225、244–247 行，[`reports/shadow/e1_state.json`](../reports/shadow/e1_state.json)，[`doc/shadow_cadence_20261114.md`](shadow_cadence_20261114.md) 第 32–35 行。
- **现象与根因**：
  1. `interval` 记录的是相邻两次面板之间的单段收益（`from: state.get("last_panel_date")` 到 `to: str(latest.date())`，正常为 1 个交易日），但第 245 行把 `"sessions": sessions` 赋成了 `sessions_between(panel.index, state.get("entry_date"), latest)`（即自入场日 `entry_date` 起算的**累计持有交易日数** `sessions_held`），导致 `interval` 字典内 `from/to`（1 天）与 `sessions`（1~20 天）矛盾。
  2. `reports/shadow/e1_state.json` 当前保存的 `holdings` 是 **`2026-11-13` 日频重判后的 30 只名单**（相比 `11-10` 已换掉 18 只），但 `entry_date` 仍停留在 `"2026-11-10"`。这意味着：
     - `chain` 从 `2026-11-13` 开始起算链式净值（`1.0000`），不含 `11-10 ~ 11-13` 的损益；
     - 但 `sessions_held` 却从 `2026-11-10` 起算，导致 `2026-11-13` 这批股票实际上只持有 **17 个交易日**（到 `11-10` 之后第 20 个交易日）就会触发第一次 `rebalance`。这与 `doc/shadow_cadence_20261114.md` 第 32 行「11-13 的书成为第一期的起点：`entry_date` = 2026-11-10 不成立」的书面说明不一致。
- **修改建议**：
  1. `interval["sessions"]` 改为 `sessions_between(panel.index, state.get("last_panel_date"), latest)`，累计持有天数继续由顶层字段 `sessions_held` 表达；
  2. 在 `2026-11-16` 20:00 任务运行前，明确决定 `reports/shadow/e1_state.json` 的 `entry_date` 是设为 `"2026-11-13"`（让第一期链式净值完整覆盖 `11-13` 名单的 20 个交易日）还是保留 `"2026-11-10"`（若保留，需在 `doc/shadow_cadence_20261114.md` 中注明第一期链式净值仅含后 17 个交易日）。

### 1.5 [P1] `tools/shadow_run.py` 再平衡日换仓时点与成交成本未对齐回测引擎

- **位置**：[`tools/shadow_run.py`](../tools/shadow_run.py) 第 244–260 行 vs [`tools/e1_topk.py`](../tools/e1_topk.py) 第 127–168 行。
- **现象与根因**：
  在 `e1_topk.py` 中，$t$ 日收盘算出的信号在 **$t+1$ 日开盘**按 `open` 价成交（含双边 5bp 滑点、佣金与卖出印花税）。而在 `shadow_run.py` 中，$t$ 日收盘触发 `rebalance` 后立即将新名单写入 `state["holdings"]`；到了 $t+1$ 日，`basket_return` 直接对新名单计算 `close[t+1] / close[t] - 1`。这相当于假设在 **$t$ 日收盘时刻已知 $t$ 日收盘价并以该收盘价零成本完成换仓**，吃到了新入选反转股从 `close[t]` 到 `open[t+1]` 的隔夜跳空。
- **修改建议**：
  在 `doc/shadow_cadence_20261114.md` 中明确声明 `chained_nav` 与 `chained_pool` 均为「收盘对收盘、零摩擦的纯理论纸面监控指数」；或者在 `shadow_run.py` 中对再平衡次日首段采用 `open[t+1] -> close[t+1]`（原持仓吃 `close[t] -> open[t+1]`）并扣除固定换手费率估算。

---

## 2. CPPI 方案回测与执行层账本审查意见

### 2.1 [P1] `tools/account_ledger.py` 对场内 ETF 误征卖出印花税（`0.05%`），且与回测引擎不一致

- **位置**：[`tools/account_ledger.py`](../tools/account_ledger.py) 第 56、94–97、268、383 行 vs [`tools/plan_backtest.py`](../tools/plan_backtest.py) 第 184 行。
- **现象与根因**：
  1. 最终方案交易的四只标的（`510300`、`510500`、`513500`、`511010`）全部是**场内 ETF**。按沪深交易所规定，**ETF 买卖双方均免征印花税**（仅股票卖出征收 `0.05%` 印花税）。
  2. 但在 `account_ledger.py` 中：
     - `SELL_TAX = 0.0005`（第 56 行）；
     - `record_trade` 在 `action == "sell"` 时额外加收了 `fee += amount * 0.0005`（第 268 行）；
     - `gate` 在核算卖出回笼资金时扣除了 `proceeds * SELL_TAX`（第 383 行）。
  3. 同时，`account_ledger.py::fee_for(amount)` 的文档字符串写着 `"Commission floor plus the sell-side stamp duty"`，字典里也定义了 `"sell_tax": 0.0005`，但函数体只返回了 `max(broker["min_commission"], amount * broker["commission"])`。因此调用 `fee_for` 的 `plan_backtest.py::simulate_book`（第 184 行）在卖出时**没有扣印花税**，而 `account_ledger.py` 的 `record_trade` 和 `gate` **扣了 `0.05%` 印花税**。
- **修改建议**：
  场内 ETF 账本应统一将 `SELL_TAX` 设为 `0.0`（或按代码前缀区分：5/1 开头的基金代码 `sell_tax = 0.0`，6/0/3 开头的股票代码 `sell_tax = 0.0005`），并修正 `fee_for()` 的注释与 `doc/account_ledger_20261111.md` 第 3 节、第 191 行的「卖出印花税」表述。

### 2.2 [P1] `tools/plan_backtest.py` 采用「当日收盘出信号、当日收盘立即成交」，未反映隔夜执行跳空

- **位置**：[`tools/plan_backtest.py`](../tools/plan_backtest.py) 第 169–203 行。
- **现象与根因**：
  `simulate_book`（以及 `simulate`）只读取了新浪 ETF 历史的 `close` 列：在交易日 $t$，它用 $t$ 日收盘价算出净值 `value` 与目标权重 `target`，并在 `abs(target - held) >= band` 时**直接按同一个 $t$ 日收盘价成交**。
  而实际运行中（见 `pipeline.md` §5 与 `paper` 账本 11-12 的 `model_open` 成交），信号在 $t$ 日晚 21:00 生成，订单只能在 **$t+1$ 日开盘（`open`）** 成交。对 CPPI 而言，楼层保护最怕的正是「$t$ 日大跌触发减仓信号、$t+1$ 日开盘继续大幅低开」的隔夜跳空。
- **修改建议**：
  `ak.fund_etf_hist_sina` 本身同时返回 `open` 与 `close`。建议在 `plan_backtest.py` 中增加 `execution="next_open"` 模式（$t$ 日收盘算目标股数 delta，$t+1$ 日按 `open` 价成交、按 `close` 价记当日净值），核对 129 个起点的最差回撤（当前日频收盘成交为 `-8.24%`）在次日开盘成交下是否仍 100% 守住 `-10%` 上限。

### 2.3 [P1] `tools/plan_orders.py` 未预留买入手续费，且 `account_ledger.py::book` 缺整批事务校验

- **位置**：[`tools/plan_orders.py`](../tools/plan_orders.py) 第 69–83 行、[`tools/account_ledger.py`](../tools/account_ledger.py) 第 370–390、409–434 行。
- **现象与根因**：
  1. `plan_orders.orders()` 将 `100%` 的 `nav` 分配给四条腿（权重之和为 `1.0`），每条腿仅做 `math.floor(target_value / price / lot) * lot` 整手向下取整，**完全没有扣除买入手续费**。
  2. 当四条腿整手取整省下的零头恰好小于四笔买单的佣金合计（例如净值刚好是各腿整手价格的公倍数附近），`plan_orders` 生成的订单在过 `account_ledger.py::gate` 时会触发 `"orders need X more cash"`（`safe_to_send: false`）。
  3. 更危险的是 `account_ledger.py::book()`（第 426–433 行）：它不调 `gate()`，而是直接循环调用 `record_trade()` 逐笔追加写入 `trades.jsonl`。若前 3 笔买单把现金耗尽、第 4 笔买单因差几十元手续费在 `record_trade()` 第 271 行抛出 `ValueError("Order needs ... but only ... cash")`，**前 3 笔成交已经落盘写入 `trades.jsonl`**，账本会停留在只买了 3 条腿的半成交损坏状态。
- **修改建议**：
  1. 在 `plan_orders.py::orders` 中，若全部买单加手续费超过可用现金，将最后一腿（安全腿 `511010` 或金额最大腿）下调一手（`lot`）；
  2. 在 `account_ledger.py::book` 中，写入任何一笔交易前先调用 `gate()`（或先在内存中对全部 `ordered` 试跑一遍 `replay`），确认最后一笔成交后 `cash >= 0` 再落盘。

### 2.4 [P1] `tools/account_ledger.py::state` 计算 `peak` 时未包含当日实时估值 `value`

- **位置**：[`tools/account_ledger.py`](../tools/account_ledger.py) 第 146–170 行。
- **现象与根因**：
  `state()` 中 `peak = peak_of(snapshots, cfg["capital"])` 只对历史已保存的 `snapshots` 和初始本金取最大值，没有把第 161 行刚用当日 `prices` 算出的 `value` 纳入 `peak`。
  若用户在未跑 `mark` 前先跑 `check` 或 `book`（`doc/account_ledger_20261111.md` 第 60–68 行命令示例正是先 `check` 后 `mark`），一旦当日净值创新高，就会出现 `value > peak`、回报一个**正数的回撤**（`drawdown = value / peak - 1.0 > 0`），并且传给 `plan_orders.decide(row["nav"], row["peak"], ...)` 的峰值偏低（使缓冲垫 `nav - 0.9 * peak` 被高估）。
- **修改建议**：
  将 `account_ledger.py` 第 166 行改为：
  ```python
  peak = max(peak_of(snapshots, cfg["capital"]), value)
  ```
  使其与 `curve()`（第 341 行）及 `plan_backtest.py::simulate_book`（第 170 行 `peak = max(peak, value)`）严格一致。

### 2.5 [P2] `tools/plan_backtest.py` 拆分回溯未同步缩放拆分前的每份分红

- **位置**：[`tools/plan_backtest.py`](../tools/plan_backtest.py) 第 81–91、129–136 行。
- **现象与根因**：
  在 `price_and_payouts`（及 `total_return`）中，当检测到 `date` 发生简单比例拆分（如 1:2 拆分，`move = 0.5`）时，代码将 `prior`（拆分前所有日期）的价格乘以 `move`，但**在此之前已经写入 `payouts[d]`（$d < \text{date}$）的每份分红金额没有乘以 `move`**。若某只 ETF 在拆分前曾派发分红，回测在拆分前按减半后的价格买入双倍份额，却按未减半的每份分红领现金，会导致拆分前分红翻倍。（目前四只 ETF 中唯一拆分的 `513500` 无分红记录，故未触发，但属潜在计算隐患。）
- **修改建议**：
  当 `move <= 1 - split and looks_like_a_split(move)` 触发时，同步执行：
  ```python
  for past_day in list(payouts):
      if past_day < date:
          payouts[past_day] *= move
  ```

---

## 3. 数据归档、研究与备份工具审查意见

### 3.1 [P0] `tools/archive_fund_flow.py` 跳过逻辑缺陷：用旧归档最大日期当目标日，导致周度任务永久空转

- **位置**：[`tools/archive_fund_flow.py`](../tools/archive_fund_flow.py) 第 70–95 行；关联记录 [`doc/implementation_plan.md`](implementation_plan.md) 第 35–43 行。
- **现象与根因**：
  `doc/implementation_plan.md` 记录了 `2026-11-14`（周六）资金流归档的异常现象：`requested 1,799 / fetched 2 / skipped 1,797 / failed 0 / rows_added 240 / rows_total 215,880 / last_date 2026-11-06`，并推测可能是东财数据源滞后 5 个交易日。
  核查代码与 `data/online/fund_flow/flow.parquet` 后确认，**真实原因是本地跳过条件写错了**：
  ```python
  # tools/archive_fund_flow.py:69-77
  current = existing(index_path)
  latest = current["date"].max() if len(current) else None
  targets = codes or universe_codes(limit)
  fresh, failed, skipped = [], [], []
  for position, code in enumerate(targets, 1):
      have = current.loc[current["code"] == code, "date"]
      if latest is not None and len(have) and have.max() >= latest:
          skipped.append(code)
          continue
  ```
  - `flow.parquet` 在上周（11-08）归档后，1,797 只正常交易股票的最大日期是 `2026-11-06`，另有 2 只停牌股最大日期是 `2026-10-30`；因此 `latest = current["date"].max()` 算出来就是 **`2026-11-06`**！
  - 进入循环后，所有 1,797 只正常股票都满足 `have.max() (2026-11-06) >= latest (2026-11-06)`，**在发起任何网络请求之前就被全部跳过（`skipped = 1797`）**！
  - 脚本唯一去请求网络的只有那 **2 只**上次停在 `2026-10-30` 的股票（`fetched = 2`），它们每次返回 120 行旧数据，第 89 行在 `drop_duplicates` **之前**计算 `added = sum(len(frame) for frame in fresh) = 240`，去重后实际净增 **0 行**（`rows_total` 仍为 `215,880`）。
  - 一旦所有股票的最大日期对齐，该脚本此后每周六都会 100% 跳过全部股票。
- **修改建议**：
  1. 目标日期 `target_date` 应取自交易日历 `trading_calendar.load()` 中 `<= today` 的最近一个交易日（或先请求第一只活跃股票获取上游最新交易日 `upstream_latest`），只有 `have.max() >= target_date` 时才作为「今日已抓过」跳过；
  2. `rows_added` 应改为去重后的真实净增行数：`added = int(len(merged) - len(current))`；
  3. 在 `tests/test_archive_fund_flow.py` 中增加一条模拟跨周重跑的离线测试（旧库停在 $T_1$，新抓取返回到 $T_2 > T_1$，断言旧库中已到 $T_1$ 的股票不会被跳过）。

### 3.2 [P2] `tools/market_leverage.py` 在 `signal="growth"` 下生成非标准 JSON `NaN` 且输出文件名冲突

- **位置**：[`tools/market_leverage.py`](../tools/market_leverage.py) 第 469–491 行；产物 `reports/runs/market_leverage/gate_both_long.json` 第 106–110 行。
- **现象与根因**：
  1. 当 `--signal growth` 时，`need_cap=False` 使 `basket_frame["lev"]` 全为 `NaN`；但第 472 行固定计算 `pre_v = nw_slope(early["r20"], early["fwd"])`（而非 `g20`），得到 `slope: NaN, t: NaN, corr: NaN`，并直接写入 `gate_both_long.json`（标准 JSON 规范不支持 `NaN` 字面量）。
  2. 第 488 行输出文件名固定为 `gate_%s_%s.json % (source, window)`，未包含 `signal`，若对同一 `(source, window)` 跑不同 `signal` 会互相覆盖。
- **修改建议**：
  第 472 行按 `col = "g20" if signal == "growth" else "r20"` 选择列，并在 `nw_slope` 或序列化时将非有限浮点数转为 `None`（JSON `null`）。

### 3.3 [P2] `tools/survivorship.py`、`src/runtime/device.py` 与 `tools/backup.py` 的清理项

- **位置**：
  1. [`tools/survivorship.py`](../tools/survivorship.py) 第 115–118 行：在 `return result` 之后残留了 3 行不可达的重复循环代码，应直接删除。
  2. [`src/runtime/device.py`](../src/runtime/device.py) 第 1–21 行：全仓已无任何模块引用 `src.runtime.device`，且该文件顶层 `import torch`，而 `requirements.txt` 已移除 `torch`。建议删除 `src/runtime/device.py`。
  3. [`tools/backup.py`](../tools/backup.py) 第 102–115、191–218 行：
     - `reachable_now(path)` 在 `not path.exists()` 时直接返回 `False`。若用户新插入一块从未备份过的移动硬盘 `E:` 并运行 `python -B -m tools.backup --status`，因为 `E:/a_share_backups/a_share_project` 尚未创建，状态会误报 `reachable=False`（应探测盘符根目录 `Path(path.anchor).exists()`）；
     - `--verify` 目前只校验 `default_dest()`（第一个可达目录），建议遍历 `destinations()` 校验所有已写入的目标目录。

---

## 4. 文档体系与 `tools/doc_lint.py` 审查意见

### 4.1 [P1] `tools/doc_lint.py` 检查规则盲区与现行文档中的失效命令

- **位置**：[`tools/doc_lint.py`](../tools/doc_lint.py) 第 24–25、58–71 行。
- **根因**：
  1. `TOOL = re.compile(r"\btools[/\\]([a-z0-9_]+)\.py")` 只匹配文件路径写法 `tools/<name>.py`，**完全不匹配模块调用写法 `python -B -m tools.<name>` 和 `python -B -m src.<module>`**；
  2. `LINK = re.compile(r"\]\(([^)\s]+?\.md)(?:#[^)]*)?\)")` 只检查 `.md` 链接，不检查指向 `.txt` 等其他本地文件的相对链接。
- **漏检的现行文档问题清单**：
  1. [`doc/research_workflow.md`](research_workflow.md)（在 `doc/README.md` 阅读顺序中排第 3 位，属现行主文档）：
     - 第 10 行：`python -B -m src.backtest.run --config configs/research.json`（`src/backtest` 已删除）；
     - 第 15 行：引用 `requirements-ppo.txt`（文件已删除）；
     - 第 40–41 行：`python -B -m src.data.clean_data_analysis` 与 `python -B -m src.data.check_missing_shares`（均已删除）。
     - **建议**：将该文移入 `doc/archive/`，或重写为当前 `execution` 分支的数据面板构建与回测复现流程（`src.data.preprocess`、`tools.forward_panel`、`tools.plan_backtest`、`tools.e1_topk`）。
  2. [`doc/online_market_runbook.md`](online_market_runbook.md)（现行手册）：
     - 第 14–28、158–238 行：仍在描述旧工作台的「联网行情」5 个二级标签页及 `/api/market/*` 路由（当前 `src/web/app.py` 仅保留 `/api/execution/*`）；
     - 第 130–133 行：`python -B -m src.data.pit ...`、`python -B -m src.data.macro ...`（`pit.py` 与 `macro.py` 已删除）；
     - 第 140–142 行：`npm --prefix web run test:unit` 与 `npm --prefix web run test:e2e`（`web/package.json` 中已无这两个脚本）；
     - 第 253–254 行：`python -B -m tools.tradability_ledger` 与 `python -B -m tools.derive_pit_availability`（工具已删除）。
     - **建议**：在文首增加与 `doc/README.md` 一致的分支范围说明，标注 `pit`/`macro`/`tradability_ledger` 及旧网页标签属 `research/supply-pressure` 分支，当前分支仅保留 `src.data.online` 与执行层工作台。
  3. [`doc/cppi_strategy_20261109.md`](cppi_strategy_20261109.md) 第 102 行与 [`doc/etf_menu_20261110.md`](etf_menu_20261110.md) 第 161 行：
     - 文首已注明代码在分支裁剪时删除，但文末「复现」小节仍保留着未加说明的 `python -B -m tools.cppi_strategy` 和 `python -B -m tools.etf_menu`。
  4. **修改 `tools/doc_lint.py` 的建议**：
     将 `TOOL` 扩展为同时匹配 `r"\btools[./\\]([a-z0-9_]+)(?:\.py)?\b"`，并增加对 `r"\bpython(?:\.exe)?\s+(?:-B\s+)?-m\s+(src\.[a-z0-9_.]+)"` 的模块存在性检查；将 `LINK` 扩展为检查所有非 `http`/`#` 的本地相对文件链接。

### 4.2 [P1/P2] 现行文档间的数值口径冲突与链接错位

1. **「暴跌当分红」bug 修复（`0fc04f0`，中位年化 `5.27% → 4.43%`）后的旧数字残留**：
   - [`doc/README.md`](README.md) 第 32 行（`plan_sensitivity_20261114.md` 摘要）：仍写着「**本金 30 万~500 万中位年化 5.22%~5.31%**」，而 `plan_sensitivity_20261114.md` 第 1 节表内修正后实际为 **`4.39%~4.46%`**。
   - [`doc/plan_sensitivity_20261114.md`](plan_sensitivity_20261114.md) 第 5 行仍写「与已公布的 **+5.3% 中位**可比」，第 44 行仍写「`5.57% 与 5.27% 的差落在里面`」，与同页第 1、2 节修正后的表格（`4.43%`、`4.48% 对 4.43%`）自相矛盾。
   - [`doc/survivorship_20261114.md`](survivorship_20261114.md) 第 15、18、80 行（及 `doc/README.md` 第 45 行）、[`doc/ppo_value_20261114.md`](ppo_value_20261114.md) 第 54 行：仍引用旧数字 `+5.3%` / `+5.27%/年`，应统一加注或更新为修正后的 `+4.43%`（日检查）/ `+4.60%`（月检查）。
   - [`doc/roadmap_20261114.md`](roadmap_20261114.md) 第 52 行：仍写「`final_plan` 说『当前环境预期 +4.9%~+6.6%』」（`final_plan` 已修正为 `+4.4%~+4.6%`）；且第 1.1 节（影子对照基准）与第 1.2 节（每日备份 `tools/backup.py`）已在 11-14 当天实现上线，文档仍标为待做。
2. **错误或已删除的文件链接**：
   - [`doc/implementation_plan.md`](implementation_plan.md) 第 18 行：`[`reports/pruned_20261114.txt`](../reports/pruned_20261114.txt)`——该文件已在提交 `a3b80e9` 中删除（见 `reports/README.md` 第 17 行）。
   - [`doc/implementation_plan.md`](implementation_plan.md) 第 19、284、307 行、[`doc/ppo_value_20261114.md`](ppo_value_20261114.md) 第 8 行、[`doc/repository_cleanup_20261114.md`](repository_cleanup_20261114.md) 第 14、82 行：文字写 `[`reports/README.md`](README.md)` 或 `[本地产物目录](README.md)`，在 `doc/` 目录下 `(README.md)` 实际指向的是 `doc/README.md`，应改为 `(../reports/README.md)`。
3. **[`pipeline.md`](../pipeline.md) 工具清单与实际目录不同步**：
   - `pipeline.md` 第 18、27、33–36 行写「研究侧的读数工具只有一个还在活跃：`tools/market_leverage.py`」「`tools/`：执行层 5 个、影子 2 个、研究 2 个、支撑 1 个」（仅列 10 个工具）；实际 `tools/` 下共有 **20 个 Python 工具**（含 `backup.py`、`doc_lint.py`、`e1_rule.py`、`e1_topk.py`、`etf_premium.py`、`plan_sensitivity.py`、`survivorship.py`、`data_catalog.py`、`data_scorecard.py`、`archive_fund_flow.py`）。
4. **[`doc/implementation_plan.md`](implementation_plan.md) 首尾状态撕裂**：
   - 文件顶部（第 1–194 行）已更新至 `2026-11-14`（执行层三本账、S2 影子运行、两融市场杠杆关闭、11.94 GB 误删事故），但：
     - 第 88–90 行「2026-11-14：仓库清理」仍写着「`reports/runs/` 一个没动；保留 810 个 `model.zip`」，未加批注说明同日稍晚该目录已被误删清空；
     - 第 579–662 行「阶段范围与剩余任务」「下一步优先级」「维护与验证规则」仍完全停留在 **2026-11-08**（例如把早已跑完并关闭的 E1 列为「下一步唯一实验」、把离线测试数记为 `2731 passed` 而非当前分支的 `207 passed`、把已删除的 `src.backtest.run` 和 `src.training.train` 列为默认验证命令）。

---

## 5. 建议修改执行顺序

1. **第一优先级（在 `2026-11-16` 周一 20:00 / 21:00 自动任务前完成，P0）**：
   - 修复 `src/web/execution.py` 的 `chain` 下标错位（§1.1）与 `interval_from` 字段（§1.3）；
   - 修复 `tools/e1_rule.py` 与 `tools/shadow_run.py` 在 `hold` 日的排序与分数记录（§1.2），校准 `interval["sessions"]` 与 `e1_state.json` 的 `entry_date`（§1.4），重新构建 `web/dist`；
   - 修复 `tools/archive_fund_flow.py` 的跳过日期基准与 `rows_added` 去重后统计（§3.1）。
2. **第二优先级（执行层与回测口径对齐，P1）**：
   - 在 `tools/account_ledger.py` 中去掉对场内 ETF 的 `0.05%` 卖出印花税（§2.1），修正 `state()` 的 `peak` 计算（§2.4），并为 `plan_orders.py` / `account_ledger.py::book` 补上买入手续费预留与整批现金前置校验（§2.3）；
   - 在 `tools/plan_backtest.py` 中补充次日开盘价成交（`next_open`）对照与拆分前分红缩放（§2.2、§2.5）。
3. **第三优先级（工具与文档卫生收口，P2）**：
   - 清理 `tools/survivorship.py` 死代码、移除未使用的 `src/runtime/device.py`、修复 `tools/market_leverage.py` 的 `NaN` 序列化与 `tools/backup.py` 的新盘状态探测（§3.2、§3.3）；
   - 升级 `tools/doc_lint.py` 规则并同步修正 `doc/` 与 `pipeline.md` 中的过期数字、链接与尾部章节（§4.1、§4.2）。

---

## 6. 第二轮复核（2026-11-16，整理仓库时做的增量审查）

本轮触发：仓库整理时发现"每日任务"从三条命令串行变成了一条 `tools.daily_ops`，同时新增了
`/api/recommend/unified` 与 live 开账。以下只写**上一轮之后新增或新暴露**的问题。

### 6.1 上一轮 14 项的现状

| 编号 | 级别 | 现状 | 依据 |
| --- | --- | --- | --- |
| 1.1 | P0 | **已修** | `src/web/execution.py::_chain` 改为逐行累计，`days` 用 `enumerate` 的位置索引 |
| 1.2 | P0 | **已修** | `shadow_run.py` 每次运行都打分（第 230 行注释写明"Scored every session"），`book` 按分数排序 |
| 1.3 | P1 | **已修** | 后端同时返回 `interval` 与 `interval_from` |
| 1.4 | P1 | **已修** | `e1_state.json` 的 `last_panel_date`/`entry_date` 已对齐 2026-11-13，并加 `cadence_from` |
| 1.5 | P1 | **按计划延后** | 再平衡日仍按收盘价换仓；与 2.2 同源，一起做才有意义 |
| 2.1 | P1 | **已修** | `sell_tax_for()`：代码以 5/1 开头的 ETF 免征，其余 0.05% |
| 2.2 | P1 | **按计划延后** | `next_open` 对照未做；会改动最差回撤 −8.24%，需要单独一轮 |
| 2.3 | P1 | **已修** | `plan_orders.orders()` 会把最大买单减一手直到佣金塞进余量；`book()` 先全量干跑再落笔 |
| 2.4 | P1 | **已修** | `state()` 里 `peak = max(peak_of(...), value)` |
| 2.5 | P2 | **按计划延后** | 拆分前分红缩放未做（513700 无分红，暂不影响） |
| 3.1 | P0 | **已修** | `archive_fund_flow.last_trading_day()`（第 67 行）替换了旧的最大日期取法 |
| 3.2 | P2 | **已修** | `clean_numbers()` + `allow_nan=False`，文件名按 `signal` 区分 |
| 3.3 | P2 | **已修** | `device.py` 已删除、`survivorship` 死代码已清、`backup` 探测已改 |
| 4.1–4.2 | P1/P2 | **已修** | `doc_lint` 现有 links/titles/tools/duplicates/index 五组规则，`COMMAND` 正则覆盖 `python -m tools.<name>`；本轮又把 6 篇文档里失效的 `schedule-08fde740` 引用清掉 |

**10 项已修、3 项按计划延后、1 项部分**。上一轮的判断是准的：三个 P0 确实都会在下一个交易日暴露。

### 6.2 新发现（6 项）

#### 6.2.1 [P1] `/api/recommend/unified` 的 `book` 参数没有白名单——可读任意目录，且错值返回 500

`src/web/app.py:72` 直接把查询参数传给 `daily_allocation.recommendation(book=book)`，
而 `daily_allocation.book_state()` 在 `daily_allocation.py:102` 做的是
`base = (base or ACCOUNTS) / book` —— **一次朴素的路径拼接**。

已实测：

```
book_state('../paper')  -> 去读 <root>/paper/account.json（报 FileNotFoundError，路径已穿出去）
GET /api/recommend/unified?book=../../doc -> HTTP 500
```

写入路径是安全的（`apply()` 先调 `ledger.use()`，它会 `ValueError: Unknown account`），
但**读路径没有**。影响限于本机回环用户，属于低危信息暴露与错误处理问题，但修起来很便宜：

- `app.py` 用 `Literal["live","paper","backtest"]` 或显式校验，非法值返回 **400/409**；
- `book_state()` 里加一道 `if book not in ledger.ACCOUNTS: raise ValueError(...)`，让两处口径一致。

#### 6.2.2 [P1] `tools.daily_ops` 一次刷新 9 只载体，而每天只用 4 只

`daily_ops.py:35` 遍历 `fund_vehicles.CARRIERS`（9 只）。但落账只用到
`510300` / `510500` / `513500` / `511010`（`cash_safe` 时再加 `511880`）= **4–5 只**。
多出来的 `159919` / `511260` / `511090` / `518880` 每天各拉一次行情 + 一次分红表，
而且**其中任何一只失败都会让整轮 failed**——一只与当日决策无关的黄金 ETF 断网，就能让两本账都不落账。

建议：`refresh_prices(codes)` 只刷新当日需要的腿；其余留着手动的 `fund_vehicles --refresh`。

#### 6.2.3 [P2] 同一次运行里两个价格源

`daily_ops.prepare()` 走 `ledger.mark()`，它用 `_prices_with_dates()` **直连新浪**；
`daily_ops.book()` 走 `daily_allocation`，它用 **`data/online/fund/hist_*.csv` 缓存**。
两条路各自独立，若新浪可用但缓存拉取失败（或反过来），会出现
"快照按新浪定价、建议按缓存定价"的错配，而 `as_of == day` 的校验只覆盖后者。
建议：`prepare` 也接受缓存价格（或在 `mark` 之后断言 `price_dates == day` 且
`daily_allocation` 用的是同一批价）。

#### 6.2.4 [P2] `hold` 日把整手取整漂移显示成"差额"，看起来像要下单

`daily_allocation.recommendation()` 在 `hold` 分支用 `risky_now` 反推目标权重，
于是 `plan_orders.orders()` 算出的 `delta` 全是取整噪声。页面实测：

```
动作：持有（未触发带宽）
510300 +100   510500 +500   513500 −1,600   511010 0
```

**这四行的"差额"不是要下的单**，但页面上和真正触发调仓时长得一模一样。
建议：`hold` 时把 `delta` 全部置 0、另存 `rounding_delta` 供参考，或在页面该列标注"取整漂移（不下单）"。

#### 6.2.5 [P2] `daily_ops` 没有互斥锁

定时任务 21:30 启动，手工再跑一次就可能并发：两个进程各自读到"带宽已触发"，
各自 `record_trade`，第二笔会在现金校验上失败或**重复下单**（`record_trade` 只校验单笔现金）。
建议加一个 `reports/runs/daily_ops/.lock`（含 pid 与时间戳），存在且未过期就拒绝启动。

#### 6.2.6 [P2] 页面数据源没有接口级测试；`web/dist` 不入库

`tests/` 里没有任何用例打到 `/api/recommend/unified`（`test_daily_ops.py` 全部 monkeypatch），
`book_state()` 也只有间接覆盖。本轮修 6.2.1 时会顺手补 2 条：非法 `book` 返回 4xx、
正常 `book` 的 `assets` 权重合计为 1。
另外 `web/dist` 在 `.gitignore` 里，**新克隆必须 `npm run build` 才有页面**——这一条在
`online_market_runbook` 里没有写，值得补一句。

### 6.3 建议顺序

1. **明天 21:30 之前**：6.2.1（白名单 + 4xx）、6.2.2（只刷需要的腿）——前者是输入校验，后者是
   减少"无关资产断网导致整轮失败"的概率，两者都只需几行。
2. **本周内**：6.2.3、6.2.5、6.2.4（一个价格源、一把锁、一个不误导的列）。
3. **与上一轮延后项一起**：1.5 + 2.2（次日开盘成交）、2.5（拆分前分红），一次做完再重跑回测。

---

## 7. 本轮整理记录（不是审查意见）

- `execution` 分支从 `d2caeac` 快进到 `7b47760`（14 个提交），研究结论与执行代码回到主线；
  其余 4 个本地分支全部已并入、0 提交差异，按仓库规则保留。
- `.workbench/` 下 19 个一次性探针脚本已删除（本来也在 `.gitignore` 里）。
- 6 篇文档里对已删除的 `schedule-08fde740` 的引用已改为"已删除/已并入 21:30"。
- 复核基线：`pytest` **264 passed**、`doc_lint` **0 issue**、工作树干净。

---

## 7. 第二轮外部审查全文（2026-11-16，已并入本文）

> 这一节是另一轮独立审查的原文，原先单独放在 `repository_review_20261116.md`，
> 现并入本文以保持「一个仓库只有一份审查台账」。标题层级已降两级以适配本章。
> **本轮提出的 5 项 P0 我已逐条抽查，详见下表；原文的修改建议一字未改。**

| 编号 | 级别 | 我的抽查结论（2026-11-16） |
| --- | --- | --- |
| 1.1 | P0 | **成立**：`daily_ops.DEFAULT_ACCOUNTS = ("live", "paper")`，落账对实盘同样生效；
`live/trades.jsonl` 已有 4 笔 `unified_open` 收盘价模拟单。违反「实盘只记真实成交」的设计前提 |
| 1.2 | P0 | **成立**：`reports/account/backtest/trades.jsonl` 仍含 `2025-04-07` 的 `513500`
假分红 `23,389.00`，终点净值 `1,687,442.92`；修正后的
`plan_backtest_daily.json` 为 `1,582,725.46`，虚高 **10.47pp** |
| 1.3 | P0 | **成立**：`audit_marks.py:102` 用 `source.endswith("open")`，
而 `daily_allocation` 记的是 `unified_open`（收盘价），会被误当成开盘价核对 |
| 2.1 | P0 | **成立**：`feature_audit.py:27` 读 `data/clean/all_stocks_features.parquet`，
第 114–116 行直接 `close.shift(-h)/close-1`；该列若为 Z-score 值，符号会翻转 |
| 2.2 | P0 | **待复核**：需要补两笔除权事件（`159919` 2019-01-14、`510500` 2022-08-29）后重算，
在此之前 v5 的「0.58pp 载体差」只能当**待验证结论** |
| 1.4–1.5、2.3、3.1–3.3 | P1/P2 | 未逐条复算；其中 3.3「`doc_lint.COMMAND` 定义了没用」已抽查为**成立** |

### 全仓审查与修改意见（第二轮）：每日唯一入口、多资产研究、载体测量与特征审计

日期：2026-11-16。状态：**已完成第二轮全仓审查**（基于提交 `7b47760`，覆盖 `tools/daily_ops.py`、`tools/daily_allocation.py`、`tools/asset_panel.py`、`tools/unified_plan.py`、`tools/fund_vehicles.py`、`tools/feature_audit.py`、`tools/audit_marks.py`、`src/web/app.py`、`web/src/`、三本账与现行文档，共定位 11 项问题并给出具体修改意见）。
复现检查：`python -B -m pytest -q -p no:cacheprovider`、`python -B -m tools.doc_lint`。

> **与上一轮的关系**：[`repository_review_20261115.md`](repository_review_20261115.md) 中的 14 项建议已有 9 项在提交 `66ff07e` 中收口；本轮针对的是该修复之后新合入的「每日唯一入口 + 全资产建议」与「统一策略 v1–v6 / 全特征审计」模块。

---

#### 0. 结论与问题总表

本轮的核心风险是：**执行层的第一不变量（模型成交绝不进入实盘账本）已被打破**，且**统一多资产研究的若干关键数字建立在未核实的 ETF 除权/分红数据之上**。全部 11 项问题的清单如下：

| 编号 | 级别 | 模块 / 文件 | 问题摘要 | 影响 |
| --- | --- | --- | --- | --- |
| **1.1** | **P0** | `tools/daily_ops.py`、`tools/daily_allocation.py`、`reports/account/live/trades.jsonl` | 21:30 定时任务默认对 `("live", "paper")` 同时调 `daily_allocation.apply()` 自动写入模型收盘价成交 | **破坏「实盘账本只记真实成交」的第一不变量**：11-15 已向 `live` 写入 4 笔收盘价模拟单（`unified_open`），未来每次触发 5% 带宽都会自动向 `live` 追加模型单 |
| **1.2** | **P0** | `reports/account/backtest/` vs `reports/runs/plan_backtest/plan_backtest_daily.json` | `0fc04f0` 修复「2025-04-07 暴跌当分红」后未重导 `reports/account/backtest` | 网页「回测账」仍含 `2025-04-07` `513500` 的 `23,389` 元假分红，终点净值显示 **`1,687,442.92`（+68.74%）**，比同起点修正后的 `plan_backtest_daily.json`（**`1,582,725.46`，+58.27%**）虚高 **10.47pp（104,717 元）** |
| **1.3** | **P0** | `tools/audit_marks.py`（第 102 行）vs `tools/daily_allocation.py`（第 213 行） | `daily_allocation` 按**收盘价**记建仓单并标 `source="unified_open"`；`audit_marks` 用 `source.endswith("open")` 判断开盘/收盘 | 对 `live` 运行 `audit_marks` 会把按 11-13 **收盘价**记的 4 笔 `unified_open` 拿去比对 11-13 **开盘价**，误报 `fill price does not match the open` |
| **2.1** | **P0** | `tools/feature_audit.py`（第 27、114–116 行）、`doc/feature_audit_20261115.md` | 读取已废弃的 2026-07-01 顶层 `data/clean/all_stocks_features.parquet`，将**已做 Z-score 标准化（均值 0、含负数）的 `close` 列**相除算远期收益，且将 `股票代码` 当作特征 | Z-score 正负相除导致比值符号翻转：1 日 `return` 真实截面 IC 为 **`-0.0283`（强反转）**，被算成 **`+0.0095`（假动量）**；报告第 3 节还把 `指数点位` 的 1M IC（`-0.23`）误抄到两个 PE 口径名下 |
| **2.2** | **P0** | `tools/fund_vehicles.py`（第 57–75 行）、`tools/plan_backtest.py`、`doc/fund_vehicles_20261115.md` | 新浪分红表漏记 `159919` 的 `2019-01-14` 除息（单日 `-10.09%`）并将 `510500` 的 `2022-08-29` 折算（单日 `-12.74%`）记为 `0.000`，代码将两处除权跳空当成真亏损 | **v5 核心结论失真**：补回这两笔漏记事件后，`159919` 对指数差从 `-1.00pp` 变为 **`-0.23pp/年`**（优于 `510300` 的 `-0.42pp`），`510500` 从 `-0.99pp` 变为 **`+0.06pp/年`**（跟踪误差 `3.81pp → 1.81pp`）；所谓「同指数选错载体白丢 0.58pp/年」实为新浪早期分红缺口 |
| **1.4** | **P1** | `tools/daily_ops.py`、`tools/daily_allocation.py`、`tools/plan_orders.py` | ① `refresh_prices` 后 `prepare`/`check` 仍重复发 24 次新浪请求；② `--day` 未透传进 `daily_allocation` 且 `apply()` 硬编码 `now().date()`；③ `apply()` 缺写前整批现金校验；④ `plan_orders` 修剪手数未计卖出佣金 | 跨日补跑 `daily_ops --day` 或长假（>4 天）后运行会直接报错；单跑 `daily_allocation --apply` 若第 4 腿缺现金会留下前 3 腿半截脏流水 |
| **1.5** | **P1** | `tools/daily_allocation.py`（第 130–170 行） | ① `--cash-safe` 在带宽内漏配 `511880`（已持仓时抛 `ValueError`）；② `--cash-safe` 下 12 类资产表权重之和仅 `0.80`；③ `action="hold"` 时 `sleeves` 仍返回非零 `delta` 与 `fee` | 开启 `--cash-safe` 后次日进入带宽即崩溃，且资产表丢掉 20% 权重；网页显示「持有」的同时表格却显示 `+100/+500/-1600` 份差额与 `15` 元费用 |
| **2.3** | **P1** | `tools/unified_plan.py`（第 127–161 行）、`tests/test_unified_plan.py`（第 96–106 行） | 第 152 行先把 `book` 推到 `cum[end+2]`（`end+1` 日收盘），再用 `end+1` 日净值算 `target_weights` 并在同一个 `cum[end+2]` 成交；带宽按全组合 L1 算而非风险腿合计 | 实际执行的是「`end+1` 日同日收盘决策 + 同日收盘成交」（零延迟），与文档声称的「月末决策、次日成交」不符；单测因用了静态组合且跳空后序列持平而未能检出 |
| **3.1** | **P1** | `tools/shadow_run.py`（第 242 行）、`reports/shadow/e1_state.json` | `hold` 日按**当日新分数**对 30 只持仓重排并写回 `state`；且当前 `e1_state.json` 仍为未排序旧顺序 | 在 19/20 的 `hold` 日里，虽然总名单 30 只不动，但「每日推荐」页的 `前 5 / 前 10 / 前 20` 每天都在洗牌（变相日频换手） |
| **3.2** | **P2** | `src/web/app.py`（第 71–79 行）、`web/src/ExecutionPage.tsx` | `/api/recommend/unified` 未校验 `book` 参数且未注入 `root`；前端 `bookNotes` 与实际成交口径矛盾 | 非法 `book` 参数返回 HTTP 500；`create_app(tmp_root)` 测试时读穿到全局真实目录；页面切换 `ExecutionPage` 每次重挂载重拉全部 4 个接口 |
| **3.3** | **P2** | `tools/doc_lint.py`（第 28、65–75 行）、`doc/daily_recommendation_20261115.md`、`pipeline.md` | `doc_lint.py` 定义了 `COMMAND` 正则却从未调用；`daily_recommendation` 表内把 `paper` 的 `55.49%` 误抄给 `live`；`pipeline.md` 工具数仍写 10 个（实为 27 个） | `doc_lint` 仍漏检 `python -m tools.<deleted>`；`live` 真实目标为 `59.84%`（偏离 `-0.09%`）而非文中的 `55.49%`（`-4.44%`） |

---

#### 1. 每日唯一入口（`daily_ops` / `daily_allocation`）审查意见

##### 1.1 [P0] 停止 `daily_ops.py` 向 `live` 实盘账本自动写入模型收盘价成交

- **位置**：[`tools/daily_ops.py`](../tools/daily_ops.py) 第 29、55–63、118–127 行；[`tools/daily_allocation.py`](../tools/daily_allocation.py) 第 209–226 行；[`reports/account/live/trades.jsonl`](../reports/account/live/trades.jsonl)。
- **现象与根因**：
  `account_ledger.py` 第 41–45 行与 `pipeline.md` §3 的首要设计原则是「模型模拟成交绝不进入实盘账本（`live` 只记 `manual`）」。但 `daily_ops.py` 把 `DEFAULT_ACCOUNTS = ("live", "paper")` 一视同仁地送进 `book(account, day)` → `daily_allocation.apply()`。这不仅在 11-15 把 4 笔 `unified_open` 模型收盘价买单写进了 `live/trades.jsonl`，而且使每晚 21:30 的定时任务在未来任何触发 5% 带宽的日子都会**自动把模型收盘价模拟单写进 `live`**，导致 `live` 与 `paper` 同质化（甚至比 `paper` 的 `model_open` 更偏离实盘）。
- **修改建议**：
  1. 在 `daily_ops.py` 中区分账本职责：`live` 和 `paper` 均执行 `prepare`（`sync_dividends` + `mark`）、`check` 和只读生成 `daily_allocation.recommendation()`；但**自动落账 `daily_allocation.apply()` 默认只对 `paper` 执行**（或增加 `--book-accounts=paper`，`live` 仅输出建议与预检结论，真实成交由人工通过 `account_ledger trade --account live` 录入）。
  2. 若 `live` 尚未在真实券商下单，应清空 `reports/account/live/trades.jsonl` 中的 4 笔 `unified_open` 模拟单并将 `live/snapshots.jsonl` 恢复为 `1,000,000` 纯现金状态；若已在券商下单，则用真实交割单价格与费用替换为 `source: "manual"`。

##### 1.2 [P0] 重新导出 `reports/account/backtest`，消除残留的 `2025-04-07` 假分红与 `10.47pp` 净值虚高

- **位置**：[`reports/account/backtest/trades.jsonl`](../reports/account/backtest/trades.jsonl) 第 444 行 vs [`reports/runs/plan_backtest/plan_backtest_daily.json`](../reports/runs/plan_backtest/plan_backtest_daily.json) 第 407–415 行。
- **实测证据**：
  提交 `0fc04f0` 修复了「把 2025-04-07 关税暴跌当分红」的缺陷并重跑了 `plan_backtest_daily.json`（`2019-01-02` 起点的终点净值已更新为 **`1,582,725.46`**、年化 `+6.01%`、最差回撤 `-8.23%`、621 笔成交）。
  但 **`reports/account/backtest/` 从未重导**，其 `trades.jsonl` 里至今保留着 `{"action": "dividend", "amount": 23389.0, "cash_per_share": 0.19, "code": "513500", "date": "2025-04-07"}`，导致网页「三本账 → 回测账」仍显示旧净值 **`1,687,442.92`（虚高 `104,717.46` 元 / `+10.47pp`）**。
- **修改建议**：
  1. 在修复下文 §2.2 的 `510500`（`2022-08-29`）折算识别后，重跑 `plan_backtest.py` 与 `plan_backtest.export_account('2019-01-02', name='backtest')`；
  2. 在 `tests/test_backtest_export.py` 中增加一致性测试：断言 `reports/account/backtest` 的终点净值与 `reports/runs/plan_backtest/plan_backtest_daily.json` 中 `2019-01-02` 起点的 `final` 完全相等；
  3. 同步更新 `doc/account_ledger_20261111.md` 第 26 行与 `doc/implementation_plan.md` 中的旧回测账终点数字。

##### 1.3 [P0] 修复 `tools/audit_marks.py` 对 `source="unified_open"` 的开盘/收盘后缀冲突

- **位置**：[`tools/audit_marks.py`](../tools/audit_marks.py) 第 102 行、[`tools/daily_allocation.py`](../tools/daily_allocation.py) 第 213 行。
- **根因**：
  `daily_allocation.apply()` 写入建仓成交时用 `source = "unified_open" if opening else "unified"`，其中 `"open"` 指「开账（opening）」，**价格实际取自收盘价 `close`**；而 `audit_marks.audit_fills()` 第 102 行写的是 `frame = opens if source.endswith("open") else closes`，把 `"unified_open"` 误判为按开盘价成交。
- **修改建议**：
  将 `daily_allocation.py` 的 `source` 命名为 `"unified_init"` / `"unified_rebalance"`（或 `"unified_close"`），并将 `audit_marks.py` 第 102 行改为 `frame = opens if source == "model_open" else closes`。

##### 1.4 [P1] 修复 `daily_ops.py` / `daily_allocation.py` / `plan_orders.py` 的四处工程裂缝

- **位置**：[`tools/daily_ops.py`](../tools/daily_ops.py) 第 32–64 行、[`tools/daily_allocation.py`](../tools/daily_allocation.py) 第 55–73、184–226 行、[`tools/plan_orders.py`](../tools/plan_orders.py) 第 91 行。
- **修改建议**：
  1. **消除 24 次重复联网请求并保证 `prepare` 原子性**：`daily_ops.refresh_prices()` 已将最新行情与分红写入 `data/online/fund/{hist,dist}_<code>.csv`，`prepare()` 与 `check()` 应直接复用该本地缓存（或将 prices/dividends 一次性读入内存传给 `sync_dividends`、`mark`、`check`），避免在已写完 `live` 快照后因 `paper` 的重复新浪请求超时而留下半边快照。
  2. **透传 `day` 参数**：为 `daily_allocation.cached_prices()`、`recommendation()` 和 `apply()` 增加 `as_of: str | None = None` 参数；当 `daily_ops --day <date>` 指定日期时，截取 `<= as_of` 的缓存行并以 `as_of`（而非 `pd.Timestamp.now().date()`）做交易日校验，使历史重跑与次日凌晨补跑正常工作；过期阈值改按**距今未更新的交易日数（如 >1 个交易日）**而非自然日 `MAX_AGE_DAYS = 4`（自然日 4 天在五一/国庆/春节长假后会误拒）。
  3. **`daily_allocation.apply()` 增加写前整批现金校验**：与 `account_ledger.book()` 一致，在循环调用 `ledger.record_trade()` 落盘前先在内存中试算整批 `ordered` 订单后的现金余额。
  4. **`plan_orders.orders()` 计入卖出佣金**：将第 91 行 `sum(fee_for(row) for row in plan if row["delta"] > 0)` 改为 `sum(fee_for(row) for row in plan if row["delta"] != 0)`，并按 `row["delta"] * row["price"]`（买入金额最大腿）下调一手。

##### 1.5 [P1] 修复 `daily_allocation.py` 的 `--cash-safe` 崩溃、资产表丢权重与 `hold` 日非零差额

- **位置**：[`tools/daily_allocation.py`](../tools/daily_allocation.py) 第 130–170 行。
- **修改建议**：
  1. **修复 `--cash-safe` 在带宽内的崩溃**：当 `abs(drift) < BAND`（`action == "hold"`）时，不应重算 `25/25/50` 也不应丢掉 `CASH_ETF`，而应直接保持当前各腿持仓股数不变（`delta = 0, fee = 0.0, action = "hold"`）。这样既修复了持有 `511880` 时进入带宽抛 `ValueError: account holds 511880` 的崩溃，也解决了网页在显示「持有，未触发带宽」时下方表格却列出 `510300 +100 / 510500 +500 / 513500 -1600` 差额与 `15` 元费用的自相矛盾。
  2. **修复 `--cash-safe` 下 12 类资产表权重之和为 `0.80` 的 bug**：在构建 `assets` 表（第 160–170 行）时，将 `CASH_ETF`（`511880`）的目标权重与目标市值并入「现金 / 货基」行（即 `weight = idle_weight + cash_etf_weight`，`value = idle_cash + cash_etf_value`），使 `cash_safe=True` 时 12 类资产权重之和仍恒等于 `1.0000`。

---

#### 2. 多资产研究、载体测量与特征审计审查意见

##### 2.1 [P0] 修复 `tools/feature_audit.py` 误用 Z-score `close` 相除算收益率及 `doc/feature_audit_20261115.md` 表格错位

- **位置**：[`tools/feature_audit.py`](../tools/feature_audit.py) 第 27、32、110–145 行；[`reports/runs/feature_audit/price_ic.csv`](../reports/runs/feature_audit/price_ic.csv) 第 1–38 行；[`doc/feature_audit_20261115.md`](feature_audit_20261115.md) 第 28–46 行。
- **实证验证**：
  - `data/clean/all_stocks_features.parquet` 是 2026-07-01 的早期遗留文件，其中连 `股票代码`（如 `0.724061`）和 `close`（如 `-0.006905`）都被做了均值 0、标准差 1 的 Z-score 标准化（见 `reports/runs/feature_audit/price_profiles.csv` 全部 `std = 1.0`、`mean = 0.0`）。
  - `feature_audit.cross_sectional_ic()` 执行 `forward = close.shift(-horizon) / close - 1.0`，用含负数的 Z-score 相除，导致 1 日 `return` 的截面秩 IC 从真实的 **`-0.0283`（强反转）** 变成了 **`+0.0095`（假动量）**，且 `price_ic.csv` 把 `股票代码`、`open`、`close`、`ma_5..60` 等 12 个非特征列也算进了「32 个量价特征」。
  - 此外，`doc/feature_audit_20261115.md` §3 表格把 `index_and_macro.csv` 第 2 行 `指数点位` 的 1M IC（`-0.2292`）误填到了 `静态市盈率中位数`（实为 `-0.0554`）和 `等权静态市盈率`（实为 `-0.0335`）的 1 月栏。
- **修改建议**：
  1. 将 `feature_audit.py` 的输入改为 `data/clean/research_v3/panel_data.parquet`（20 个标准特征）与 `data/clean/research_v3/prices.parquet`（未标准化的真实后复权 `close`）；
  2. 重跑生成 `reports/runs/feature_audit/`，并在 `doc/feature_audit_20261115.md` 顶部按规范添加「更正」块（20 日持有期下 20 个量价特征最大 `|IC|` 仍低于 `0.031` 门槛，故「0 个过门槛」的最终裁决不变，但 1 日反转方向与估值 1M IC 数字需更正）。

##### 2.2 [P0] 修复 `tools/fund_vehicles.py` 与 `tools/plan_backtest.py` 中漏记的两处 ETF 除权/折算事件，更正 v5 载体结论

- **位置**：[`tools/fund_vehicles.py`](../tools/fund_vehicles.py) 第 57–75 行、[`tools/plan_backtest.py`](../tools/plan_backtest.py) 第 60–91、110–136 行、[`doc/fund_vehicles_20261115.md`](fund_vehicles_20261115.md)、[`doc/unified_strategy_final_20261115.md`](unified_strategy_final_20261115.md)。
- **实证验证**：
  新浪 `fund_etf_dividend_sina` 接口存在两处历史数据缺失：
  1. **`159919`（嘉实沪深300ETF）**：接口仅返回 2020 年以后的 5 次分红，漏掉了 **`2019-01-14`** 的除息（`2019-01-11` 停牌半日，`2019-01-14` 收盘从 `3.408` 跌至 `3.064`，单日 `-10.09%`，而当日沪深300指数仅 `-0.10%`）。这单日 `-10.09%` 的未入账除息在 14.4 年里造成了 `-0.73pp/年` 的假落后。**补齐该笔 `0.344` 元除息后，`159919` 年化为 `5.12%`，对指数差为 `-0.23pp/年`（跟踪误差 `2.11pp`），反而优于 `510300` 的 `-0.42pp/年`；在分红完整的 2019-01-15 至今回看，`159919`（`+5.43%`）也略高于 `510300`（`+5.32%`）。**
  2. **`510500`（南方中证500ETF）**：**`2022-08-29`** 发生份额折算/分红，收盘从 `7.220` 跳至 `6.300`（单日 `-12.74%`，当日中证500指数 `+0.39%`），接口将该行分红记为 `0.000`。提交 `0fc04f0` 移除 10% 跳空规则后，把这次折算当成了 `-12.74%` 的真暴跌。**补齐该事件后，`510500` 年化从 `5.50%` 升至 `6.56%`，对指数差从 `-0.99pp/年` 变为 `+0.06pp/年`（跟踪误差从 `3.81pp` 降至 `1.81pp`）。**
- **修改建议**：
  1. 区分「真市场暴跌（如 `2025-04-07`）」与「接口漏记的除息/折算（如 `2022-08-29` `510500`、`2019-01-14` `159919`）」的客观判据是**相对标的指数的超额跳空**：当 ETF 单日跌幅 $\ge 8\%$ 且与同日标的指数涨跌幅之差 $|r_{\text{ETF}} - r_{\text{index}}| \ge 8\%$ 时（或显式登记这两笔经交易所公告确认的历史事件），才作为漏记分红/折算修复；同时在 `repair()` 缩放折算前历史价格时，同步缩放折算日之前的 `paid` 每份分红。
  2. 在 `doc/fund_vehicles_20261115.md` 与 `doc/unified_strategy_final_20261115.md` 顶部补充更正：A 股宽基 ETF（`510300`/`159919`/`510500`）真实载体差仅 `-0.42pp ~ +0.06pp/年`，真正需要计入 `-0.7 ~ -1.2pp/年` 显著载体差的是**债券 ETF（交易所国债 vs 银行间中债指数口径差）与 QDII `513500`**。

##### 2.3 [P1] 修复 `tools/unified_plan.py` 的「月末决策、次日成交」时点错位与带宽定义

- **位置**：[`tools/unified_plan.py`](../tools/unified_plan.py) 第 127–161 行、[`tests/test_unified_plan.py`](../tests/test_unified_plan.py) 第 96–106 行。
- **根因**：
  `unified_plan.simulate()` 第 152–160 行先用 `book = book * (cum[end + 2] / cum[base])` 把组合推到 `end + 1` 日收盘，再用 `end + 1` 日的 `nav`、`peak` 和 `y_at(end + 1)` 计算 `wanted = target_weights(...)`，并立即在同一个 `cum[end + 2]` 按 `wanted` 调仓——导致**决策与成交发生在同一天（`end + 1` 日收盘），没有任何成交滞后**。
- **修改建议**：
  1. 在 `end` 日收盘（即 `nav_decision = float(navs[-1])`、`peak_decision = peak`、`y_at(end)`）计算目标权重 `wanted`；随后推进到 `cum[end + 2]`（`end + 1` 日收盘）按 `wanted` 成交并扣除交易成本；
  2. 将第 132 行初始峰值设为 `peak = 1.0`（含建仓成本回撤），并将 CPPI 规则的不交易带判据对齐为风险腿合计偏离 `abs(risky_target - risky_now) >= band`；
  3. 将 `tests/test_unified_plan.py::test_fills_happen_one_session_after_the_decision` 改为使用动态 CPPI 规则（`C1`）并在决策日次日构造二次下跌，真正检验 1 日成交滞后。

---

#### 3. 影子运行、Web 接口与文档检查审查意见

##### 3.1 [P1] `tools/shadow_run.py` 在 `hold` 日应保持入场排序，避免 `topK` 每日洗牌

- **位置**：[`tools/shadow_run.py`](../tools/shadow_run.py) 第 239–243 行、[`reports/shadow/e1_state.json`](../reports/shadow/e1_state.json)。
- **根因**：
  在 `hold` 日（19/20 的交易日），第 242 行 `book = sorted(held, key=lambda code: float(score.get(code, float("inf"))))` 用**当日新算的截面分**对持有的 30 只股票重排并写回 `e1_state.json`。这会导致虽然 30 只总持仓未变（`decision="hold"`），但网页「每日推荐」的 `前 5 / 前 10 / 前 20` 切片每天都在发生日频轮动；同时 `reports/shadow/e1_state.json` 中现存的 30 只 `holdings` 仍是 `66ff07e` 之前的未排序状态（第 20 位 `300896` 分数 `0.3283` > 第 21 位 `603260` 分数 `0.2516`）。
- **修改建议**：
  - 将 `reports/shadow/e1_state.json` 的 `holdings` 按 `2026-11-13` 的入场分数排好序，并存入 `entry_score`；
  - 在 `hold` 日保持 `book = list(held)` 的入场日排序不变（使 20 日持有期内 `book[:K]` 真正固定持有不动），当日新分数可单独记入 `current_score` 字段供参考。

##### 3.2 [P2] `src/web/app.py` 路由参数校验与 `root` 注入、前端说明同步

- **位置**：[`src/web/app.py`](../src/web/app.py) 第 71–79 行、[`tools/daily_allocation.py`](../tools/daily_allocation.py) 第 95–105 行、[`web/src/ExecutionPage.tsx`](../web/src/ExecutionPage.tsx) 第 13–17、44–54 行。
- **修改建议**：
  1. 在 `daily_allocation.book_state()` 中增加 `if book not in ("live", "paper", "backtest"): raise ValueError(...)`（由 FastAPI 统一转为 HTTP 409，防止未捕获的 `FileNotFoundError` 500 错误或路径穿越）；
  2. 在 `app.py` 的 `recommend_unified` 路由中传入 `online=root / "data" / "online"` 与 `base=root / "reports" / "account"`，与其余路由保持一致；
  3. 在 `ExecutionPage.tsx` 的 `load()` 中接受可选 `targetRecBook = recBook` 参数，避免切换 `live`/`paper` 后刷新读到旧状态。

##### 3.3 [P2] 启用 `tools/doc_lint.py` 中未调用的 `COMMAND` 正则并修正三处文档笔误

- **位置**：[`tools/doc_lint.py`](../tools/doc_lint.py) 第 28、65–75 行；[`doc/daily_recommendation_20261115.md`](daily_recommendation_20261115.md) 第 44–46 行；[`pipeline.md`](../pipeline.md) 第 18、27、33–36、89–95 行；[`doc/implementation_plan.md`](implementation_plan.md) 第 18–19、579–662 行。
- **修改建议**：
  1. 在 `tools/doc_lint.py::check_tools` 第 71 行将 `for name in TOOL.findall(line):` 改为 `for name in TOOL.findall(line) + COMMAND.findall(line):`（让第 28 行已定义的 `COMMAND` 真正生效），并清理被检出的 4 处失效命令引用；
  2. 修正 `doc/daily_recommendation_20261115.md` 第 44–46 行表格中 `live` 的「风险腿目标」（应为 **`59.84%`**，误写成 `paper` 的 `55.49%`）与「偏离」（应为 **`−0.09%`**，误写成 `−4.44%`），同步修正 `doc/README.md` 第 50 行摘要；
  3. 更新 `pipeline.md` 的工具清单（纳入 `daily_ops`、`daily_allocation`、`asset_panel`、`unified_plan`、`fund_vehicles`、`feature_audit` 等）与第 7 节每日执行命令，清理 `doc/implementation_plan.md` 第 18–19 行的死链与尾部 2026-11-08 过期章节。

---

#### 4. 建议修改执行顺序

1. **第一优先级（今日 `2026-11-16` 20:00 S2 与 21:30 定时任务前完成，P0）**：
   - 让 `daily_ops.py` 不再向 `live` 自动落账（§1.1），并修复 `audit_marks` 的 `source` 后缀冲突（§1.3）；
   - 重导 `reports/account/backtest`（§1.2），修复 `fund_vehicles` / `plan_backtest` 对 `510500` `2022-08-29` 与 `159919` `2019-01-14` 的漏记事件（§2.2）；
   - 重跑 `feature_audit` 并更正 `doc/feature_audit_20261115.md`（§2.1）。
2. **第二优先级（每日入口的健壮性，P1）**：
   - 补齐 `daily_allocation` 的 `--day` 透传、写前整批现金校验与 `--cash-safe` 崩溃修复（§1.4、§1.5）；
   - 修正 `unified_plan.py` 的成交时点（§2.3）与 `shadow_run.py` 的 `hold` 日排序（§3.1）。
3. **第三优先级（接口与文档卫生，P2）**：
   - 修正 `app.py` 路由注入与前端提示（§3.2）；
   - 启用 `doc_lint` 的 `COMMAND` 正则并修正 `pipeline.md`、`implementation_plan.md`、`daily_recommendation` 中的过期数字与链接（§3.3）。
