# 全仓代码审查（第四轮，2026-11-19）：上一轮修复的验收，与两处没关上的门

> 本轮只审代码。上一轮（[repository_review_20261117.md](repository_review_20261117.md)）报了 1 项 P0、7 项 P1；
> 这之间落了 8 个提交，其中 3 个是修复。结论是：**P0 修好了，5 项 P1 修好了，2 项只修了一半**——
> 而"只修一半"的两项，恰好一个在"钱进哪本账"上，一个在"账能不能继续跑"上。

## 0. 结论

> **追记（2026-11-19 01:15）**：`aca9fad` 已收掉下表的第 1、2、3、5、6、7、8、9、10、11 项（其中第 6 项改的是预登记文档）；第 4 项只修了 state 原子写，第 12、13 项仍开。全套测试 289 passed / 1 skipped。剩下的运营项见[运营检查](operations_20261119.md)。

| # | 级别 | 位置 | 是什么 | 状态 |
| ---: | --- | --- | --- | --- |
| 1 | **P1** | `tools/daily_ops.py:77` + `tools/daily_allocation.py:227` | 新增的实盘闸门被它唯一的调用者绕过：`book()` 传 `allowed=(account,)`，account 是 live 时就放行 | 上轮 P1 的修复引入 |
| 2 | **P1** | `tools/daily_allocation.py:143` / `tools/plan_orders.py:68` | cash_safe 关掉后账上仍有 511880，**调仓日**直接抛错，整轮 failed 且此后每天 failed | 上轮 P1-5 的残余 |
| 3 | P2 | `tools/daily_ops.py:136` | 预检失败只报"未通过"，把 gate 已经算好的 4 条原因丢掉 | 新 |
| 4 | P2 | `tools/shadow_variants.py:112,122` | 新模块复制 `shadow_run.run` 的骨架且已漂移；state 非原子写，写坏即整本账卡死 | 新 |
| 5 | P2 | `tests/test_daily_allocation.py:120` | `test_last_quarter(): pass` ——287 个测试里有一个不断言任何东西 | 新 |
| 6 | P2 | `doc/shadow_variants_20261118.md:56` | 判据 4 用持仓名单比对，但 A 与 E1 的 20 日周期永远差 3 个交易日 | 新 |
| 7 | P2 | `tools/trading_calendar.py:49` | 缓存仍非原子写；12-25 起每天重下（COVER=7，发布止于 12-31） | 上轮 P1-2 残余 |
| 8 | P2 | `tools/daily_allocation.py:259` | 回滚只备份 trades；开账首日 backup=None；快照重写非原子 | 上轮 P1-7 残余 |
| 9 | P2 | `tests/test_backup.py:89` | 依赖本机 OneDrive，任何没有可写目标的机器都红 | 上轮已见 |
| 10 | P2 | `tools/doc_lint.py:28` | `COMMAND` 正则定义了从不调用 | 上轮 P2，未修 |
| 11 | P3 | `doc/implementation_plan.md:192,198` | 标题紧贴上一个列表项，缺空行 | 新 |
| 12 | P3 | `src/web/app.py:66` | 网页只有 E1 一本影子账，四本变体账没有端点 | 新 |
| 13 | P3 | `tools/daily_allocation.py:206` | 持有日的 `invested` 在安全腿改目标之前算，现金重的账上会虚高 | 新 |

**没有 P0。** 资金安全的两条硬约束（模型单不写实盘、持有日不下单）在正常路径上都还在；
上面的 #1 是"把默认值当闸门"的漏洞，#2 是"一个标志位把账锁死"的老问题换了个方向。

## 1. 这轮怎么跑的

**提交。** `a826d7d..HEAD`：8 个提交、31 个文件、**+1520 / −77**。三个修复
`30380c0`（持有日）、`2c0e260`（日历/落账闸门/开收盘判定）、`7c434d8`（cash_safe/回滚/原子写），
一个新模块 `fa925f8`（`tools/shadow_variants.py`，210 行），其余是日志与文档。

**测试。** 这台机器仍是三个解释器，所以测试也分两遍：

| 环境 | 命令 | 结果 |
| --- | --- | --- |
| Anaconda base（3.11.7 / **pandas 2.1.4** / pytest 7.4） | `python -B -m pytest -q` | **286 passed, 1 failed, 19.32 s**（上一轮同一环境 268 passed / 2 failed） |
| 生产 env AShareGPU（3.11.16 / pandas 2.0.3 / pytest 9） | 同 | 不重复跑：85 个 `tmp_path` 错误是 pytest 9 的锁目录被沙箱拒，与代码无关 |

唯一那个失败仍是 `test_backup.py::test_the_default_destination_is_outside_the_project`：本机没有可写的
备份目标（OneDrive 与 E: 都被拒），属环境。定向复跑 `test_archive_fund_flow` / `test_fund_vehicles` /
`test_trading_calendar`：**22 passed**——上一轮那个 pandas 2.1 的 `concat` 崩溃确实修好了。

**探针。** 都写在 gitignored 的 `reports/runs/review_20261119/`：

- 四本变体账 vs 主账本；A 与 `e1_rule.composite` 的逐位差异；四本账一次跑完的耗时；
- cash_safe 开/关 × 持有日/调仓日的四种组合；
- `apply()` 的 `allowed` 参数两侧各试一次；
- 日历 `load()` 三次计时与缓存 mtime；今天 00:13 的 `daily_ops --dry --no-refresh`。

**dry run 的结果是"正确的失败"**：status `failed`，因为四腿报价停在 2026-11-18 而今天是 11-19
（00:13 还没开盘）。拒绝是对的行为，但摘要只说一句"预检未通过"（见 §5.1）。

**文档 lint。** `python -B -m tools.doc_lint` → `0 issue(s): {}`：链接、标题、工具引用、重名、索引五项全过。

## 2. 上一轮清单的验收

| 上一轮 | 结论 | 证据 |
| --- | --- | --- |
| **P0** 持有日把风险腿按 25/25/50 重排 | **修好** | `30380c0` 让持有日直接返回 0 差额；真实 paper 账今天 `action=hold`、四腿 `delta=[0,0,0,0]`、资产表合计 **1.000100**；新测试 `test_a_hold_day_never_proposes_a_trade` 用 45/18/57 的歪腿持仓验证 |
| P1-1 文档入口跑不起来 | **只加了说明** | `doc/README.md` 新增解释器段落；`python -B -m tools.daily_ops` 仍然是 `ModuleNotFoundError: pandas` |
| P1-2 日历每次联网、休市误判 | **修好** | `load()` 三次 **0.008 / 0.006 / 0.006 s**（上轮实测 1.15 s），缓存 mtime 不变；断网回退与 COVER 都有测试；`is_trading_day` 已挪进 `run()` 的 try |
| P1-3 周度归档对 pandas 版本敏感 | **修好** | 2.1.4 下定向 22 passed；`dated()` 把两边都归一到 `datetime64[ns]` |
| P1-4 载体刷新（节流/原子/截止日） | **修好** | `_write_atomic` + 每只 0.3 s 间隔 + 分红窗口 `end=today`，三条都有测试 |
| P1-5 cash_safe 把两种模式互相锁死 | **一半** | 持有日已通（`test_cash_safe_does_not_lock_the_book_in_either_mode`）；**调仓日仍抛错**，见 §4 |
| P1-6 资产表漏 511880 | **修好** | 资产表合计 1.0001；`cash_safe` 关闭时不再丢行 |
| P1-7 `apply()` 先写成交再 mark | **一半** | `mark` 失败回滚成交有测试；快照与开账首日仍在备份之外，见 §5 表第 8 行 |

净结果：**1 个 P0 + 5 个 P1 修好，2 个 P1 只修一半**。两次"只修一半"都不是粗心，而是补丁只覆盖了
报告里举例的那个方向（持有日、mark 立刻抛），没有覆盖对称的那一半（调仓日、mark 写完之后）。

## 3. P1-1：实盘闸门被唯一的调用者绕过

`2c0e260` 给 `apply()` 加了硬闸门（`tools/daily_allocation.py:227`）：

```python
if result.get("book") not in tuple(allowed):
    raise ValueError("模型单不允许写进 %r（只允许 %s）；实盘成交请用 account_ledger trade"
                     % (result.get("book"), ", ".join(allowed)))
```

默认 `allowed=("paper",)`，测试 `test_apply_refuses_to_write_a_model_fill_into_the_real_money_book`
（`tests/test_daily_allocation.py:188`）验的就是这个默认值。但生产路径上唯一的调用者是
`tools/daily_ops.py:77`：

```python
log = daily_allocation.apply(result, allowed=(account,))
```

而 `account` 来自 `book_accounts`，也就是 `--book-accounts`（`:232-233`，help 里写着
"which books the model may write to; live must be recorded by hand"）。于是：
**只要把 live 传进 book_accounts，闸门收到的允许列表就是 ("live",)，自己放自己过去。**

探针（dry，不写任何文件）：

- `apply(probe, dry=True)` → `ValueError: 模型单不允许写进 'live'（只允许 paper）`
- `apply(probe, dry=True, allowed=("live",))` → **accepted, book='live'**

现在挡住实盘模型单的，只剩 `DEFAULT_BOOK_ACCOUNTS = ("paper",)` 这个默认值（`daily_ops.py:34`）。
而文档明确教了怎么改它——`doc/daily_recommendation_20261115.md:68-69`：

> 要临时改变落账对象必须显式传 `--book-accounts=paper,live`，而脚本会在汇报里标出哪些账是「仅建议」。

这句话本身就是矛盾的：真把 live 传进去，live 就**不是**"仅建议"了，汇报里也没有任何东西可标。
`apply` 的新注释写着 "A default is not a gate"——这句判词对它自己的修复同样成立。

**修法（两行）。** `book()` 不该传 `(account,)`，应传模块常量：

```python
log = daily_allocation.apply(result, allowed=DEFAULT_BOOK_ACCOUNTS)
```

这样 `--book-accounts=paper,live` 会在 live 上**失败关闭**，错误信息直接告诉操作者改用
`account_ledger trade --account live`。再补一条测试，把"命令行不能加宽闸门"钉住：

```python
def test_the_cli_cannot_widen_the_gate(monkeypatch):
    ...  # ops.book("live", day) 必须抛，而不是写进 live
```

顺带把上面那句文档改成"该参数只对 paper 有效；传 live 会被拒绝"。

## 4. P1-2：cash_safe 在调仓日仍然锁死

`7c434d8` 的修法只在**持有日**成立：持有分支把账上已有的每个代码都放进 `plan_weights`
（`tools/daily_allocation.py:135-140`），所以 511880 不会触发 `orders()` 的"计划外持仓"检查。
调仓分支（`:143`）直接 `plan_weights = weights`，`weights_for()` 在 `cash_safe=False` 时只产出
风险腿 + 511010，**没有 511880**。

探针：账上 60% 风险腿（25/25/50）+ 2000 份 511880 + 20 万现金，风险腿跌 10% 使偏离 > 5%：

```
cash_safe=True  -> rebalance   nav=939,980  risky 0.575->0.255
cash_safe=False -> ValueError: account holds 511880, which the plan does not allocate; sell it first
```

后果不是"这次不行"，而是**每天都 failed**：`daily_ops` 把异常记成 failed、退出码 2，下一次跑
仍停在同一处；错误信息也没给出手工卖出 511880 的命令。方向还是不对称的——**先带
`--cash-safe` 开账、后来不带，会锁死；反过来不会**。这正是上一轮"两种模式互相锁死"的镜像。

**修法（推荐 a）。**

- **a. 调仓日把它当卖出。** 在 `recommendation` 里给账上非计划的代码补一个 **0 权重**：
  `plan_weights.setdefault(code, 0.0)`。`orders()` 会给出 `target=0 / delta=−held` 的卖单，
  先卖后买（`apply` 已按 sell-first 排序）的现金正好供同一轮买入，且符合 `plan_orders.py:65-67`
  自己的注释——"the caller either sells it or tells the tool it is part of NAV deliberately"。
  这里 caller 就是 `recommendation`，它有能力做这个决定。
- **b. 若坚持不自动卖**：至少把错误信息改成可直接复制的一行
  `account_ledger trade --account paper --code 511880 --action sell --shares 2000 --price <价>`，
  并让 `daily_ops` 把这种情形标成 `needs_manual`（仍退出 2），而不是与"数据缺失"共用一句
  "预检未通过"。

## 5. P2 / P3

| # | 位置 | 细节 | 修法 |
| ---: | --- | --- | --- |
| 3 | `daily_ops.py:136-138` | 只把「哪些账没过」放进异常，`gates[account]["problems"]` 没进错误串；step 明细也只记 `safe_to_send/band_triggered/nav` | 把 problems 拼进 `RuntimeError`，summary 带第一条 |
| 4 | `shadow_variants.py:112,118,122` | 见 §5.2 | 复用 session 逻辑；state 用 `_write_atomic` |
| 5 | `tests/test_daily_allocation.py:120` | 空测试 | 删除，或补成真正断言 |
| 6 | `doc/shadow_variants_20261118.md:56` | 判据 4 用名单，两个账本相位永远差 3 个交易日 | 改比 score 向量（§5.3） |
| 7 | `trading_calendar.py:49` | 缓存 `to_csv` 直写；`COVER=7` 而缓存上限固定在 2026-12-31，12-25 起每次调用都重下 | 写临时文件再 `os.replace`；`COVER` 降到 3 或按"发布范围本身短"降级 |
| 8 | `daily_allocation.py:259-282` | 回滚只还原 `trades.jsonl`；文件不存在时 `backup=None`，开账首日的半截成交不会被清；`mark` 重写整个快照文件，不在备份里 | `backup = ""` 兜底并在异常时写回空文件；快照同样走原子写 |
| 9 | `tests/test_backup.py:89-92` | 直接调 `default_dest()`，没有可写目标就抛 | `if not destinations(): pytest.skip(...)` |
| 10 | `doc_lint.py:28` | `COMMAND` 正则从未被调用；README 新教的命令形式同样查不到 | 在 `check_tools` 里加一条 `COMMAND` 检查 |
| 11 | `implementation_plan.md:192,198` | 标题紧贴上一个列表项（11-18 的日更段有正确空行） | 补空行 |
| 12 | `src/web/app.py:66-69` | 只读 `e1_shadow.jsonl`；四本变体账没有端点或页面 | 加 `/api/execution/shadow?book=A`，或一个 variants 列表 |
| 13 | `daily_allocation.py:206` | 持有日 `invested` 在安全腿改写之前算 | 无消费方；要么在改写后重算，要么删掉这个字段 |

### 5.1 预检的原因被丢掉了

今天 00:13 的 dry run 摘要全文是：

```
2026-11-19 FAILED RuntimeError: 预检未通过：live、paper；已停在落账之前
```

而 `ledger.check()` 已经把四条原因算好了：

```
510300 quote is from 2026-11-18, not 2026-11-19
510500 quote is from 2026-11-18, not 2026-11-19
513500 quote is from 2026-11-18, not 2026-11-19
511010 quote is from 2026-11-18, not 2026-11-19
```

21:30 的正常运行时，如果新浪晚发布十分钟，操作者看到的就是同一句没有原因的红字。把
`problems` 拼进异常只多两行，却把"该等十分钟"与"该查账"区分开了。

### 5.2 变体模块是第二份骨架，而且已经开始漂移

`shadow_variants.run_one`（`:122-185`）与 `shadow_run.run`（`:184-264`）做的是同一件事：
读 `manifest/panel/prices` → `bad_data` → `stale` → `due` → `decide/hold` → 记录 → 存 state。
仓库自己在 `tools/e1_rule.py:5-6` 写着 "Keeping two copies would let the backtest drift away from
what the recorder writes"——新模块复制了它要防的那种结构。

已经漂移的三处：

- **没有休市判断。** `shadow_run.run` 在 `:204-216` 先问日历，休市日什么都不追加；变体没有这一步，
  所以每个休市日会往四本账各追加一行 `stale`。
- **记录缺 `sessions_held`。** `shadow_run.py:257` 有，变体没有；页面的"已持有 N 日"没法复用。
- **state 不是原子写。** `shadow_variants.py:112` 用 `write_text` 直写，而同一批提交刚给
  `fund_vehicles` 加了 `_write_atomic`（`:79-86`）。这不只是卫生问题：state 是"开账日 + 持有名单"
  的唯一记忆，写到一半断电会留下半截 JSON，`load_state` 的 `json.loads` 不捕获，
  **第二天起每次运行都抛**，直到有人手工删掉 `A_state.json`。同样的写法在 `e1_state.json` 上已存在，
  新模块把它复制了一遍。

### 5.3 判据 4 比的是名单，而两个账本永远不同相

`doc/shadow_variants_20261118.md:56` 预登记："同一交易日，A 的名单与 `e1_shadow.jsonl` 的名单
**在共同持有期内应当逐项相同**；若不同，先查 harness。"

但两个账本的开账日不同：E1 的 `entry_date` 是 2026-11-13，A 是 2026-11-18（`A_state.json`）。
两者都是 20 个交易日后才重判，**周期永远差 3 个交易日**，相位不会合并。今天实测
A ∩ E1 = **11/30**；而 A ∩ B = 16、A ∩ C = 16、A ∩ D = 4（随机期望 3）。11 不是 harness 错了，
是两本不同持有期的书。

判据 4 想守的是"同一套规则"，精确可验证的量是**打分**而不是名单：

- 同日 `A.blend` vs `e1_rule.composite`：297 只全有分数，**max|Δ| = 3.331e-16**，前 30 名完全一致；
- 这条已经在 `tests/test_shadow_variants.py:41` 钉住，运行时也复现了。

建议把判据 4 改成"同日 score 向量 max|Δ| < 1e-12，且 `decide(score, [])` 的前 30 完全一致"，
名单比对降级为"相位对齐的日期上做 sanity check"。否则 60 个交易日后第一次比较时，
读者会把一个正确的差异当成 harness 故障。

## 6. 复核后确实没问题的

| 项 | 证据 |
| --- | --- |
| 持有日不交易（上轮 P0） | paper 四腿差额全 0；资产表合计 1.000100；专门测试用歪腿持仓构造 |
| A 是同一套规则 | 最大差 3.331e-16，前 30 完全一致，297 只无缺失 |
| 四本变体账互不相同 | A∩B=16、B∩C=24、A∩D=4（期望 3）；一次跑完 **2.65 s**，与文档"2–3 秒"相符 |
| 日历不再联网 | 6–8 ms、缓存 mtime 不变、8797 行、末 2026-12-31；断网有回退测试 |
| 落账回滚 | `mark` 立刻失败时 `trades.jsonl` 逐字节不变（新测试）；正常路径先卖后买 |
| 模型单默认不写实盘 | 默认路径仍只写 paper（`DEFAULT_BOOK_ACCOUNTS`），advisory 分支不写任何东西 |
| 测试基线 | pandas 2.1.4 下 286 passed；上一轮的两个失败只剩一个环境项 |

## 7. 建议顺序

1. **今天（半小时，三处小改）**：`book()` 传 `DEFAULT_BOOK_ACCOUNTS` 并加一条"CLI 不能加宽闸门"的测试；
   预检异常带上 `problems`；删掉空测试。
2. **本周**：调仓日把非计划持仓按 0 权重卖出（§4a）；变体 state 改原子写；日历缓存改原子写。
3. **下次重构**：变体模块复用 `shadow_run` 的 session；判据 4 改比打分；`doc_lint` 接上 `COMMAND`；
   网页暴露四本变体账。

## 8. 复现

```powershell
# 全量测试（这台机器上必须指到有 pandas 的解释器）
$env:TMP="$PWD\reports\runs\review_20261119\tmp"; $env:TEMP=$env:TMP
& 'D:\ProgramFiles\Anaconda3\python.exe' -B -m pytest -q

# 持有日
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -m tools.daily_allocation --book=paper

# 今天的 dry run
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -m tools.daily_ops --dry --no-refresh
```

判据 4 的量化（同一段代码也在 `reports/runs/review_20261119/probe_variants.py`）：

```python
panel = pd.read_parquet("data/clean/forward_pool_20261118/panel_data.parquet")
ref = e1_rule.composite(panel, codes, features, panel.index.max())
mix = shadow_variants.blend(panel, codes, shadow_variants.VARIANTS["A"]["spec"], panel.index.max())
print((ref - mix).abs().max())          # 3.331e-16
```

## 9. 边界

- 没有联网跑 akshare：所有涉及网络的路径是静态读 + 用本地缓存验证，未验证的会明说。
- 变体只跑了 2026-11-18 这一份面板；四本账目前各一行，**任何收益结论都还不成立**（文档自己也这么说）。
- 没有尝试把任何一笔成交写进 live；§3 的探针全部 `dry=True`。
- 本轮的产出只有这份文档与索引行；代码、账本、参数一个字未改。
