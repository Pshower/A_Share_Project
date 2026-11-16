# 全仓代码审查（第三轮代码 / 2026-11-17）：建议生成、交易日历与运行环境

承接 [全仓审查与修改意见](repository_review_20261115.md)（第一、二轮代码）与
[投资建议审查](investment_advice_review_20261116.md)（第三轮，审建议）。本轮只审代码：不重跑研究，
不改任何策略参数，也不改账本。**所有发现都有可复现的命令**，文末给复现清单。

## 0. 结论与问题总表

一句话：**账本本身是可信的，可信到测试会用内容哈希守着它；不可信的是"建议怎么送到人手里"这一段**——
引擎说「持有」，页面照样给出可下单的差额；日历每天重下一遍；文档里的入口命令在本机根本跑不起来。

| # | 问题 | 位置 | 影响 | 级别 |
| --- | --- | --- | --- | --- |
| 1 | **持有日仍生成可下单差额**：`action=hold` 与 `sleeves[].delta != 0` 同时出现在同一个响应里 | `tools/daily_allocation.py:130-138`、`tools/plan_orders.py:45-52,76-86` | 用户按页面操作会白付佣金并制造反向漂移；引擎侧不落账，两边状态从此分叉 | **P0** |
| 2 | **文档入口命令在本机不可执行**：`python -B -m tools.daily_ops` 解析到无 pandas 的 Python 3.9 | `doc/README.md:50`、`doc/daily_recommendation_20261115.md` 等 | `ModuleNotFoundError: pandas`，退出码 1；只有定时任务里写死的 AShareGPU 路径能用 | **P1** |
| 3 | **交易日历每次调用都联网**（`HORIZON=365` 永远不满足，实测 1.15 s/次），且**失败点在 try 之外** | `tools/trading_calendar.py:22,33`、`tools/daily_ops.py:109`（对照 `:114`） | 每天 3 次多余请求；断网时 `run()` 直接抛出，退出码 1、**不写失败 JSON**，与模块自己的承诺相反 | **P1** |
| 4 | **周度资金流归档的合并对 datetime 单位敏感** | `tools/archive_fund_flow.py:110` | pandas 2.1.4 下 `ValueError`（测试红）；生产 2.0.3 不触发；`requirements.txt` 不锁版本，等于把风险交给安装当天 | **P1** |
| 5 | **每日刷新 9 只载体的缓存**：无节流、非原子写、分红截止日硬编码 | `tools/fund_vehicles.py:78-92,104` | 每次 18 个无节流请求；中断可留半截缓存；2027 年起新分红静默不进缓存 | **P1** |
| 6 | 预检与落账的价格来自**两套独立抓取** | `tools/account_ledger.py:541` vs `tools/daily_allocation.py:122` | 预检通过不代表落账用的是同一份价；失败时信息指向不明 | P2 |
| 7 | 负仓处理**前后不一致** | `tools/daily_allocation.py:120`（`held` 过滤 `>0`）vs `:123`（`nav` 用未过滤的 `positions`） | 四腿里的负持仓会悄悄压低 NAV，却不影响下单量；`risky_now` 因此对不上 | P2 |
| 8 | `/api/execution/books/{name}` 与 `/api/recommend/unified?book=` **无白名单** | `src/web/app.py:62,72` | 本地只读服务，但读路径可越出 `reports/account` | P2 |
| 9 | 两个源文件带 UTF-8 BOM | `tools/unified_plan.py`、`tests/test_restate_book.py` | Python 能跑；`ast`/linters/部分工具读文本会报错 | P2 |
| 10 | `replay` 不设下界 | `tools/account_ledger.py:126-136` | 负仓与负现金只在 `record_trade` 挡；流水被手改后账本"照常重放" | P2 |
| 11 | **`mark(path=...)` 只换了快照路径，定价仍读全局流水** | `tools/account_ledger.py:316` | 用别的账本调用时，会把 A 账的快照配上 B 账的成交（实测见 §4） | P2 |
| 12 | **cash_safe 模式三处不闭环**：hold 分支重建权重时丢掉 511880，`orders()` 直接抛「账上有 511880，先卖掉」；而 511880 一旦买过，非 cash_safe 模式也永久抛 | `tools/daily_allocation.py:130-138`、`tools/plan_orders.py:68-71` | 两种模式互相把对方锁死：切到 cash_safe 之后就不能再切回来 | **P1** |
| 13 | 资产表在 cash_safe 下**少报一半净值**：12 行 UNIVERSE 里没有 511880 | `tools/daily_allocation.py:159-170` | 实测 `sum(sleeve)=1.000000` 而 `sum(asset)=0.500000`；页面渲染的是 assets。测试只覆盖了 cash_safe=False | **P1** |
| 14 | **决策阶段不是原子的**：`apply()` 先逐笔 `record_trade` 再 `mark`，mark 一失败就留下「成交写了、净值没写」 | `tools/daily_allocation.py:215-230`、`tools/daily_ops.py:70` | 与「两阶段写入」的承诺不符；事后再跑会重复记账 | **P1** |
| 15 | 手工成交后、下次记账前，页面把持仓显示成价格 0 | `src/web/execution.py:103` | `prices.get(code) or 0.0`：市值 0、权重 0%，无任何提示 | P2 |
| 16 | `mark_when_ready` 轮询无异常处理；`audit_marks --repair` 给未核验的腿也盖当日戳；`backup verify` 不抓 `BadZipFile` | `tools/mark_when_ready.py:38`、`tools/audit_marks.py:139`、`tools/backup.py:188` | 一次取价异常就整轮崩；修补能抹掉 `provisional`；截断包直接抛栈而不是 `ok=false` | P2 |
| 17 | 浮点取整少一手；`fee_for` 的 docstring 说了印花税但函数不算 | `tools/plan_orders.py:79`、`tools/account_ledger.py:105-108` | `10.7/1.07` 得 99.999… → 0 股；无 `fee` 字段的卖出行在 `replay` 里少收税（299,970 vs 299,920） | P2 |
| 18 | 杂项：`cash_after` 死变量、`simulate` 未防 `nav<=0`、3 份 JSONL 读取、7 处 `resample("M")`、`doc_lint` 的 `COMMAND` 正则从未使用、`package.json` 缺 runbook 里的 `test:unit`/`test:e2e` | `tools/plan_orders.py:75,86`、`tools/unified_plan.py:90`、`src/web/execution.py:49` 等 | 维护性与前向兼容 | P2 |

上一轮报过、**本轮复核仍未修**的（不是新发现，列出来只为对上账）：

| 上一轮编号 | 问题 | 现状 |
| --- | --- | --- |
| 第二轮 P0 | `audit_marks` 用 `source.endswith("open")` 判断价源，把 `unified_open` 当成开盘价 | 仍在：`tools/audit_marks.py:97,102`；`daily_allocation.py:213` 仍写 `unified_open` |
| 第二轮 P0 | `feature_audit` 用截面 Z-score 的 `close` 相除算远期收益 | 仍在：`tools/feature_audit.py:114-116` |
| 第二轮 P0 只修一半 | `apply()` 仍接受 `book="live"` | 仍在：`tools/daily_allocation.py:209` 直接 `ledger.use(result["book"])` |
| 第三轮 P0 | 持有日页面给出订单 | 仍在，见 §2（本轮补上了根因） |
| 第三轮 P1 | 日历每次联网 | 仍在，见 §3.2（本轮补上了耗时） |

## 1. 怎么跑的

### 1.1 先找回正确的解释器（这本身就是第一个发现）

这台机器上有**三个**能叫 python 的东西，装的东西完全不同：

| 解释器 | 版本 | 关键包 | 用它跑仓库会怎样 |
| --- | --- | --- | --- |
| PATH 上的 `python` | 3.9.13 | **无 pandas、无 pytest、无 akshare** | `ModuleNotFoundError: pandas`（§3.1） |
| `D:\ProgramFiles\Anaconda3\python.exe` | 3.11.7 | pandas 2.1.4、numpy 1.26.4、pytest 7.4.0，**无 akshare、无 fastapi** | 能跑测试，但网络路径全废 |
| `D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe` | **3.11.16** | **pandas 2.0.3**、akshare 1.18.64、fastapi 0.115.12、pytest 9.0.3 | **生产环境**：三个定时任务的提示词里都硬编码了这条路径 |

仓库**没有** `pyproject.toml` / `setup.py` / `python_requires`，`requirements.txt` 全是区间
（`pandas>=2,<3`）。"用哪个 Python"只存在于定时任务的提示词里，文档里写的是 `python`。

### 1.2 测试：两组结果，差异全在环境

| 解释器 | 结果 | 说明 |
| --- | --- | --- |
| Anaconda 3.11.7 / pandas 2.1.4 / pytest 7.4 | **268 passed, 2 failed** | `TMP` 指向工作区后可正常建 `tmp_path` |
| AShareGPU 3.11.16 / pandas 2.0.3 / pytest 9.0.3 | 184 passed, 1 failed, **85 errors** | 85 个 error 全是 pytest 9 的 `tmp_path` 锁文件 `Permission denied`（沙箱），不是代码 |

两个 failed 的性质不同：

* `test_backup.py::test_the_default_destination_is_outside_the_project`：`default_dest()` 找不到工作区外可写目录。沙箱环境的必然结果，换一台正常机器就过。
* `test_archive_fund_flow.py::test_the_skip_target_is_the_trading_day_not_the_stores_own_maximum`：**pandas 2.1.4 独有**，2.0.3 正常，见 §3.3。

> 顺带说明：pytest 9 的 `make_numbered_dir_with_cleanup` 会建 `.lock` 并 chmod，DSH 沙箱拒绝；
> pytest 7.4 不建锁，所以同一个沙箱里一个能跑一个不能。**这不是仓库 bug，但说明"测试全绿"这句话
> 依赖于环境**。

### 1.3 端到端跑了一遍每日流程

在 AShareGPU 里跑 `daily_ops.run(dry=True, refresh=False)`（不写任何文件）：状态 `failed`，停在预检，
原因是四条腿的报价都停在 `2026-11-16` 而今天是 `2026-11-17`。**这是对的**——现在 00:50，今天的 K 线
还没出来，闸门拒绝按昨天的价记账，正是设计意图（`account_ledger.py:320-325`）。同一时刻
`daily_allocation.recommendation("paper")` 从本地缓存出建议：`action=hold`，然后给了 `513500 -100` 手。
这就是 §2。

### 1.4 静态扫描（`tools/` + `src/` + `tests/`，约 8,000 行）

| 检查 | 结果 |
| --- | --- |
| 裸 `except:` | 0 |
| `shell=True` / `eval` / `exec` / `pickle.load` | 0 |
| 带类型的 `except` 里静默默认值 | 32 处，其中 15 处标了 `# noqa: BLE001` 并写明理由（多为"离线不算失败"） |
| `TODO` / `FIXME` | 0 |
| 可变默认参数 | 0 |
| 硬编码未来日期 | 2 处真隐患：`fund_vehicles.py:104`（`2026-12-31`）、`plan_backtest.py:46`（`END="2026-11-11"`） |
## 2. [P0] 持有日的"幽灵订单"：根因在 `target_weights`，不在页面

### 2.1 现象：同一个响应里自相矛盾

取今天的 paper 账（只读）：

    action: hold          drift: 0.0003（带宽 0.05 之内）
    sleeve deltas: 510300 0 / 510500 0 / 513500 -100 / 511010 0
    fees: 5.0

落账代码只在 `action == "rebalance"` 时才写成交（`tools/daily_allocation.py:215`），所以**账本一分不动**；
但同一份 JSON 的 `sleeves` 已经带上「卖 100 份 513500」，页面的「下单差额」表照单渲染。
11-16 的实跑记录里 paper 也是 `action=hold, fills=1`，就是这一行。

### 2.2 根因：带宽只钉住"风险腿合计"，腿内结构每轮都被重排

`target_weights` 在带宽命中时把 `applied` 设成 `held_risky`——**从今天净值反算的实际风险占比**
（`tools/plan_orders.py:45-52`）。这一步只保证"风险腿占多少"，没有保证四条腿各自占多少；
随后 `orders()` 无条件按固定的 25/25/50 重算每条腿的目标股数（`:76-86`）。于是：

* 腿内比例一旦不是 25/25/50，hold 日就会"纠正"腿内结构——**这不是取整零头，是大幅调仓**；
* 即使恰好是 25/25/50，整手取整的零头也会让每条腿各差一手。

纯函数复现（不需要行情）：风险腿 60% 全放在 510300、安全腿 40%，CPPI 目标正好 0.6：

    held = {"510300": 150000, "511010": 3636}      # 风险腿合计 = 60% NAV，正好等于目标
    d = plan_orders.decide(1_000_000, 1_000_000, prices, held=held)
    d["traded"]          # False —— 带宽说"不动"
    # 但订单是：510300 -112500（45% NAV）、510500 +25000、513500 +166600、511010 -36
    # 换手 90.4% NAV

也就是说：**带宽说"不动"，订单说"把九成净值换一遍"**。真实 paper 账上的
`+100 / +500 / -1,600 / 0` 只是"腿内恰好 25/25/50、只差取整零头"的特例。

更麻烦的是闸门：`check()` 把 `targets["orders"]` 直接交给 `gate()`，而 `gate()` 只查报价日期、
整手、现金与下限（`account_ledger.py:368-403`），不看 `traded`。于是同一个响应会同时出现
`band_triggered=False`（带宽内）与 `safe_to_send=True`（可以发单）。

### 2.3 代价

* 每笔至少 5 元佣金（`max(5, 0.03%)`），而一手 510300 才 400 元；
* 真按页面执行，第二天持仓又偏离，下一轮给反向单——2026-11-12 到 11-16 的 paper 记录已经是
  `+100 / +500 / -1,600 / 0` 的来回；
* 页面文案还在鼓励执行（「差额为 0 才不需要下单」）。这一点第三轮已报过，**代码侧至今没修**。

### 2.4 建议改法（最小改动，不动模型）

`recommendation()` 在 `action == "hold"` 时不要再调 `orders()`，直接输出 `delta=0` 的四行
（保留 `target_shares` 供展示）；或让 `orders()` 接受 `traded=False` 并回填 `held`。
两处合计不超过 6 行，加一条「带宽内不得有 delta」的测试即可锁住。

## 3. 七个 P1（按影响排序）

### 3.1 文档里的入口命令在这台机器上起不来

`doc/README.md:50`、`doc/daily_recommendation_20261115.md` 以及若干运行记录都写
`python -B -m tools.daily_ops`。实测：

    $ python -B -m tools.daily_ops --dry
    ModuleNotFoundError: No module named 'pandas'      # exit=1

PATH 上的 `python` 是 3.9.13，没有 pandas。真正的解释器只在三个定时任务的提示词里
（`& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -m tools.daily_ops`）。

后果：照文档执行必然失败，且错在 `ModuleNotFoundError`，看起来像代码坏了而不是环境没配。
建议在 `pipeline.md` 顶部加"运行环境"一节，并把文档里的 `python` 统一成一个可执行的绝对路径，
或提供一个 `tools/py.cmd` 包装。

### 3.2 交易日历：每天重下一遍，失败点还不在 try 里

`tools/trading_calendar.py:22` 的 `HORIZON = 365` 要求缓存覆盖到「今天 + 365 天」，而新浪的日历
只到当年年底（现在 `data/online/calendar/trade_dates.csv` 最大 `2026-12-31`）。于是：

    load() 用时 1.15 s，返回后文件被重写，horizon 判断 = False   # 下次还会再下

每天至少三次调用：`daily_ops.py:109`（交易日闸门）、`daily_allocation.py:198`（落账前再判一次）、
`daily_ops.py:84`（月报窗口）。即每天 3 次、每次 1 秒多的网络依赖，缓存形同虚设。

更关键的是位置：`daily_ops.run()` 的 `is_trading_day` 在 `try` **之前**（`:109` 对 `:114`）。
用模拟断网验证：

    trading_calendar.is_trading_day = boom      # raise ConnectionError
    daily_ops.run(...)   # -> ConnectionError 直接抛出 run()，没有 report

而 `main()` 只在拿到 report 之后才写 `reports/runs/daily_ops/<day>.json`（`:232-235`），所以**断网时
既没有失败 JSON，退出码也不是 2 而是 1**，与模块开头「任何一步失败……失败点写进 json」的承诺相反。
定时任务在 21:30 遇到一次网络抖动，看到的是一个 traceback，而不是"停在哪一步"。

建议：`HORIZON` 改成"覆盖到今年年底即可"，另给 `refresh()` 显式刷新；把 `is_trading_day` 挪进
`try`，或包一层并记 `error="calendar"`。

### 3.3 周度归档的合并对 datetime 单位敏感（且 requirements 不锁版本）

`tools/archive_fund_flow.py:110`：

    merged = pd.concat([current] + fresh, ignore_index=True)

当 `current["date"]` 是 `datetime64[ns]`、而新抓的一行是 `datetime64[s]`（pandas 2.x 里"把标量
Timestamp 赋给 datetime 列"就会这样），pandas 在 `NDArrayBacked._concat_same_type` 里抛：

    ValueError: all the input array dimensions ... array at index 0 has size 2 and ... size 1

实测：pandas **2.1.4 复现**，pandas **2.0.3（生产）正常**。所以：

* 生产今天没事，但 `requirements.txt` 写的是 `pandas>=2,<3`，装到 2.1+ 的机器上周度任务会挂；
* CI 用 `pip install -r requirements.txt`（`.github/workflows/ci.yml:18`），解析到哪个 2.x 取决于当天；
* 测试 `test_the_skip_target_is_the_trading_day_not_the_stores_own_maximum` 在 2.1.4 下就是红的。

这行本身也该防一手：合并前把两边 `date` 统一 `.astype("datetime64[ns]")`。**一行代码，消掉一个只在
特定版本上出现的生产事故。**

### 3.4 每日刷新 9 只载体：没有节流、不是原子写、分红有截止日

`daily_ops.refresh_prices()`（`:42-48`）对 9 只载体各调一次 `load_prices(refresh=True)` 与
`load_distributions(refresh=True)`，共 **18 次 HTTP**：

1. **没走仓里自己的节流器**。`src/data/rate_limit.py` 的 `AdaptivePacer` 只在 `Sina.history` 里通过
   patch `Session.request` 生效；`fund_vehicles` 直接调 `ak.fund_etf_hist_sina`，完全绕过。
   新浪一旦限流，整天的任务就停在刷新这一步。
2. **非原子写**。`series.to_csv(path)` 直接覆盖 `data/online/fund/hist_<code>.csv`（`:91`），
   返回空表时也会把好文件覆盖成空；`cached_prices()` 随后读到短表，而 `common` 日期交集会把
   **所有腿**的 `as_of` 一起往回拖。
3. **截止日硬编码**。`load_distributions` 调 `account_ledger.distributions(code, "2000-01-01", "2026-12-31")`
   （`:104`），2027 年起的新分红会被 `end` 过滤掉且不报错——缓存"看起来在更新"，其实不再增加。

建议：给 `fund_vehicles` 接上 `AdaptivePacer`；写临时文件再 `os.replace`；`end` 改成 `date.today()`。

### 3.5 cash_safe 模式把两种模式互相锁死，并且少报一半净值

`weights_for(cash_safe=True)` 把安全腿拆成 `511010` + `511880`（`daily_allocation.py:76-85`），
但 hold 分支重建权重时只写：

    plan_weights = {code: risky_now * part for code, part in RISKY.items()}
    plan_weights[SAFE] = 1.0 - risky_now          # :131-132

`511880` 直接消失，紧接着 `orders()` 的"计划外持仓"检查（`plan_orders.py:68-71`）抛
「account holds 511880, which the plan does not allocate; sell it first」。反过来，只要账本买过
511880，非 cash_safe 模式也永久抛同一句——**两种模式各自把对方变成死路**。

同一段还有第二处：资产表由 12 行 `UNIVERSE` 驱动（`:39-52`），里面没有 511880，于是 cash_safe 下
`sum(sleeve)=1.0` 而 `sum(asset)=0.5`，页面渲染的是后者。测试只覆盖了 `cash_safe=False`
（`tests/test_daily_allocation.py`）。

修法：hold 分支复用 `weights_for` 的实际腿内占比，别再手写一遍；`UNIVERSE` 补一行 511880；
加一条 cash_safe 的往返测试。

### 3.6 决策阶段不是原子的：成交写了，净值可能没写

`apply()` 的顺序是「逐笔 `record_trade` -> `mark`」（`daily_allocation.py:215-230`）。`mark` 会因为
报价过期、sealed、无价格而拒绝（`account_ledger.py:312-325`），而这时成交**已经落盘**：

    record_trade(...)   # 3 笔卖出 + 1 笔买入，已经写进 trades.jsonl
    mark(...)           # ValueError: quotes are not from 2026-11-17
    # -> trades 有 4 行、snapshots 0 行、decisions.jsonl 没有这一轮

下一轮再跑，`replay` 会把这 4 行算进持仓，`recommendation` 于是给出反向单——两份记录从此对不上。
`daily_ops.py:70` 直接调用它，没有任何补偿。

修法：先算好全部成交与当日价格，`mark` 成功后再写 trades；或给 `apply` 加"预演—提交"结构
（`account_ledger.py:441-467` 的 `book()` 已经是这么写的，照抄即可）。

## 4. 其余 P2

| 问题 | 位置 | 说明 |
| --- | --- | --- |
| 预检与落账的价格不同源 | `account_ledger.py:541`（akshare 现抓）vs `daily_allocation.py:122`（本地缓存） | 闸门判的是它自己刚抓到的那份，落账用的是刷新时的缓存；两边都要求"等于今天"，多数时候同时过/同时挡，但失败信息会指错方向 |
| `mark(path=...)` 读的是全局流水 | `account_ledger.py:311`（写 `path`）vs `:316`（`replay(_read(TRADES), ...)`） | 签名允许传另一个账本，实现却只换了快照落盘位置。实测：传 10 万本金、并已记入「买 900 份 511010」，`mark` 仍算出 NAV 100,000（因为它读的是 live 的空流水）。树内调用者都先 `ledger.use()`，所以今天不会错；但这条正对着「账本之间绝不共享文件」的承诺 |
| 所有 `ValueError` 都被当成 409 | `src/web/app.py:43-45` | 路由里任何一个 `ValueError`（包括代码 bug）都返回 409「冲突」而不是 500，把内部故障伪装成客户端错误 |
| 手工成交后页面价格 0 | `src/web/execution.py:103` | `prices.get(code) or 0.0`：快照之后新买的腿显示 0 价、0 市值、0 权重，不提示 |
| `mark_when_ready` 无重试 | `tools/mark_when_ready.py:38` | 一次取价异常就整轮崩，而它的本意正是"等到发布" |
| `--repair` 盖戳过宽 | `tools/audit_marks.py:139` | 给未核验、当日无 bar 的腿也写当日 `price_dates`，并把 `provisional` 抹掉 |
| `backup verify` 不抓坏包 | `tools/backup.py:188` | 截断包抛 `BadZipFile`、缺 manifest 抛 `KeyError`，没有 `ok=false` 的 JSON |
| 浮点取整 | `tools/plan_orders.py:79`（同 `plan_backtest.py:180,307`、`e1_topk.py:151`） | `10.7/1.07 = 99.999…`，`floor` 后得 0 股；上界一手 |
| `fee_for` 名不副实 | `tools/account_ledger.py:105-108` | docstring 与默认 broker 都写印花税，函数只算佣金；`replay` 对无 `fee` 字段的卖出行少收 |
| `doc_lint` 的死正则 | `tools/doc_lint.py:28` | `COMMAND` 定义后从未使用，`python -m tools.x` 形式的失效引用查不到 |
| 文档指向不存在的脚本 | `doc/online_market_runbook.md:140-142` | `npm run test:unit` / `test:e2e` 在 `web/package.json` 里不存在 |
| `feature_audit` 只能读旧面板 | `tools/feature_audit.py:27,114` | 当前 builder 的面板是 MultiIndex、没有 `close` 列，直接 `KeyError`；"32 个特征"是旧特征集 |
| 路径参数无白名单 | `src/web/app.py:62,72` | `name`/`book` 直接拼进 `reports/account/<name>`；本地只读，风险有限，但 `book=../..` 可读到仓库外 |
| BOM | `tools/unified_plan.py`、`tests/test_restate_book.py` | Python 容忍；`ast.parse` 直接报 `invalid non-printable character U+FEFF`，静态检查会跳过这两个文件 |
| `replay` 无下界 | `account_ledger.py:126-136` | `record_trade` 挡住了超卖与超额买入，但 `replay` 本身对负仓/负现金照单全收，流水被手改后不会报错 |
| 死变量 | `plan_orders.py:75,86` | `cash_after` 算了没用，`orders()` 里也没有地方用 |
| 除零 | `unified_plan.py:90` | `(nav - floor*peak)/nav` 未防 `nav<=0`；`plan_orders.risky_weight` 是防了的，两份实现不一致 |
| JSONL 读取三份 | `account_ledger.py:60`、`daily_allocation.py:88`、`execution.py:49` | 语义相同、错误处理不同，坏行都会抛 `json.JSONDecodeError` |
| `resample("M")` | `asset_panel.py:346,347,363`、`feature_audit.py:250,253`、`fund_vehicles.py:143,144` | pandas 2.2 起弃用、3.0 报错；`requirements` 写的是 `<3`，属于"下次升级就炸" |

## 5. 复核后确认没问题的

不是客套——这几条我都试着攻过：

1. **真实账本写不坏**：`tests/conftest.py` 对 `reports/account` 做 sha256 指纹，测试写了一个字节就整条失败；
   还有一条把 `socket.connect` 全禁掉。这是我在这个仓库里见过最硬的守卫。
2. **"模型单不进实盘"现在是真的**：`DEFAULT_BOOK_ACCOUNTS=("paper",)` 加上 `book()` 里再判一次
   `as_of == day`，`live` 只出建议不写账。11-16 的报告里 live 是 `advisory: true, trades: 0`，对得上。
3. **闸门四腿都查**（含安全腿）、卖单收益先进现金再检查买单（`account_ledger.py:374-402`），
   以及 `book()` 的"先全量试算再落账"——上一轮报的"第四笔失败留下三笔"确实修了。
4. **滑点记账自洽**：`simulate_book` 把成交价与 `decision_price` 分开存，导出账能由流水重放；
   注释里记着当初重放差 28,220 / 69,742 的两次事故，现在的写法对得上。
5. **没有裸 `except`、没有 `shell=True`/`eval`**，异常基本都带上下文；8,000 行里算少见。
6. **写入守卫真的挡得住**：`mark` 同一天写两次只留一行（幂等）、`sealed` 账本拒绝补日期、
   超卖与超额买入在 `record_trade` 就被拒（实测 `Selling 100 of 510300 but only 0 held`、
   `Order needs 400120.00 but only 970.30 cash`）。

## 6. 建议顺序

**今天（都是小改动，改完就能测）**

1. §2：`hold` 时不要走 `orders()`——直接回填现有持仓、`delta=0`；顺手让 `gate()` 看 `traded`。
   加一条「带宽内不得有 delta」的测试，这是本轮唯一会让人下错单的问题。
2. §3.6：`apply()` 改成"先算、后写"，`mark` 失败时不留成交。
3. §3.2：`is_trading_day` 挪进 `try`；`HORIZON` 改为"覆盖到年底"。
4. §3.1：文档里的运行命令换成真实解释器，或加 `tools/py.cmd`。

**本周**

5. §3.5：cash_safe 的 hold 分支复用 `weights_for`；`UNIVERSE` 补 511880；补一条往返测试。
6. §3.3 / §3.4：合并前统一 dtype；`fund_vehicles` 接节流、原子写、`end=date.today()`。
7. 第二轮遗留的 `audit_marks`/`unified_open` 与 `feature_audit` 目标函数（不在本文范围，但比上面更影响数字）。

**下次重构**

8. 三份 JSONL 读取合一；`resample("M")` 改 `"ME"`；`replay` 加下界校验；`book`/`name` 参数白名单；
   整手取整改用整数运算（`Decimal`），别再让浮点决定一手。

## 7. 复现

    # 环境
    python --version                                          # 3.9.13，无 pandas
    D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe -c "import pandas,akshare;print(pandas.__version__)"

    # 测试（TMP 必须指到工作区，否则 tmp_path 建不出来）
    $env:TMP="$PWD\.tmp-pytest"; $env:TEMP=$env:TMP
    D:\ProgramFiles\Anaconda3\python.exe -B -m pytest -q -p no:cacheprovider                  # 268 passed, 2 failed
    D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe -B -m pytest -q -p no:cacheprovider   # 1 failed, 85 errors

    # §2 持有日幽灵订单（纯函数，不需要行情）
    D:\ProgramFiles\Anaconda3\python.exe -B reports/runs/code_review_20261117/probe_hold2.py
    D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe -B reports/runs/code_review_20261117/probe_hold3.py   # 腿内结构被重排：90.4% NAV 换手

    # §3.6 / §4 账本侧（mark 幂等、sealed、超卖与超买拦截、mark 读全局流水）
    D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe -B reports/runs/code_review_20261117/probe_mark.py

    # §3.2 日历每次联网 + 断网直接抛
    D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe -B reports/runs/code_review_20261117/probe_cal.py
    D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe -B reports/runs/code_review_20261117/probe_calendar_fail.py

    # §3.3 pandas 版本差异
    D:\ProgramFiles\Anaconda3\python.exe -B reports/runs/code_review_20261117/probe_concat.py                  # 2.1.4 抛 ValueError
    D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe -B reports/runs/code_review_20261117/probe_concat.py   # 2.0.3 通过

证据都在 `reports/runs/code_review_20261117/`（gitignored）：`pytest_conda.log`、`pytest_gpu.log`、`probe_*.py`。

## 8. 本轮**没有**做的事

* 没有改任何代码、参数、账本；没有重跑回测。
* 没有做券商对账、税务建模、涨跌停/部分成交模拟。
* `src/web/execution.py` 的链式净值、`tools/shadow_run.py` 的 20 日节奏只做了静态检查，没有构造反例；
  它们的正确性仍以 `tests/test_web_execution.py`、`tests/test_shadow_run.py` 为准。
