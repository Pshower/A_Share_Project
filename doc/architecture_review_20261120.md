# 架构与设计审查（2026-11-20）：框架层

日期：2026-11-20（00:20–01:10）。
分支：`execution`，验收起点 `3a0691a`（干跑写自己的记录，§4 顺手验收）。
范围：**框架与设计，不重跑逐行代码**——第三、四轮代码审查与运营检查的逐条结论仍然有效，本文只回答"结构对不对、设计靠不靠得住、文档说的架构和实际是不是同一个"。方法：实测 import 图与 tracked 清单、git 与远端状态、逐条读关键模块的设计陈述，能在完全权限下验证的都验证了。
改动：**代码 1 处**（`tools/backup.py`，P1，§3-F1），其余全部是文档。

---

## 0. 一句话

三根大梁——**账本是唯一真相、每日链失败关闭、参数冻结在代码**——逐条验证都站得住（§2）。本轮抓到的不是结构问题，是**执行面**问题：一个只在完全权限下才现形的 P1（`backup.status` 把缺失目录报可达，沙箱里假绿）、一套写对了但**从没跑过的 CI**、一份落后系统一周的架构地图（已顺手刷新）。

## 1. 实测结构（凌晨的 `git ls-files`，不是文档说的）

| 层 | 位置 | 规模 | 依赖方向（grep 实测） |
| --- | --- | --- | --- |
| 数据层 | `src/data/`（9 模块）+ `src/runtime/`（下载事件与限速记账，**不是遗留**） | src 16 文件 | 被 `src.data` 自用；不 import `tools` |
| 执行层 | `tools/` 每日链 8 模块 | tools 29 模块 + 1 脚本 | **不 import `src/`**：全部 `tools/` 里唯一的跨层 import 是 `tools/forward_panel.py:28-29`（属影子层） |
| 影子层 | `shadow_run` / `shadow_variants` / `forward_panel` / `e1_rule` | 4 模块 | 复用数据层的下载与预处理 |
| 表现层 | `src/web/`（只读 API）+ `web/`（vite/react，13 个 tracked 文件） | — | **import 执行层**：`src/web/execution.py:17` 用 `account_ledger`，`src/web/app.py:78` 在路由内懒加载 `daily_allocation`；`src/web` 里 `ledger.use()` 零调用 |
| 记录 | `reports/account/`（3 本账）+ `reports/shadow/`（主账 + 4 分册）**入 Git**（23 文件）；`reports/runs/` 观测面**不入 Git**（`.gitignore:24`） | — | 页面与 CLI 共用同一套读函数 |

口径倒挂值得点一句：**不能重算的（账本、影子账）在 Git 里，能重算的（运行记录、体检 JSON）在本地**——备份 zip 管前者加配置，但 Git 历史本身哪份 zip 都没有（§3-F2）。

## 2. 五条设计支柱逐条验证

| 支柱 | 设计陈述 | 实测证据 |
| --- | --- | --- |
| 单一真相 | 净值/现金一律 `replay(trades)`；页面不得自己算 | `src/web/execution.py:7-10` 写明"仪表盘自己凑整就是第二真相"；`tests/test_web_execution.py` 断言页面=CLI；`tests/conftest.py:52-60` 指纹守卫让任何测试写真实账本直接红——这条守卫是 2026-11-14 一次真实坏账换来的（conftest 文档串记着 1,613,595 vs 1,687,443） |
| 失败关闭的每日链 | 一条命令、两阶段写入、exit 2 + JSON、不回补 | `tools/daily_ops.py:106-191`；闸门是策略不是参数（`:77-79`）；health/regime 降级为 note 不挂账（`:173-187`）；干跑今晚起写自己的记录（`record_path()` `:271`、`:295`） |
| 参数冻结在代码 | 执行参数没有配置旋钮，想改只能改代码、进 Git 被审查 | CPPI 常量在 `tools/plan_orders.py:34-42`；利率档 `REGIMES` 在 `tools/daily_ops.py:38-40` 且带出处注释；`configs/` 只有 research 数据集配置 |
| 预登记纪律 | 判据先写死；harness 被对照测试锁住 | `tests/test_shadow_variants.py::test_variant_a_is_exactly_the_frozen_rule`（A 与主规则差 3.3e-16）；修正一律追记（本文 §4 又是一次） |
| 只读 web | 页面永远不下单 | `src/web` 无 `ledger.use`（grep 零命中）；`src/web/serve.py:16-30` 只绑 127.0.0.1 并从 8765 起扫 20 个端口；`daily_allocation` 懒加载避免模块级全局 |

日历的"休市≠故障"设计单独记一笔：`tools/trading_calendar.py:8-10` 把第三轮那个 P1 的教训写成了设计规则（"要求覆盖整年永远无法满足，于是每次调用都重新下载"）；现在的实现是覆盖余量 7 天（`:24`）、每天最多试一次下载（`:42-46`）、断网时只要缓存还答得了今天就不致命（`:50-54`）、越过日历末端**raise 而不是答"休市"**（`:67-79`）——配合 `calendar_left()` 的倒计时，第三轮的 P1 在设计层面收干净了。

## 3. 发现

| # | 级 | 发现 | 证据 | 处置 |
| --- | --- | --- | --- | --- |
| F1 | **P1（代码）** | `backup.status()` 的 `reachable` 对不存在的目录**探盘根**：固定盘上任何拼错的路径都报 True；测试钉的语义是"缺失=不可达"，两者只在沙箱（盘根写入被拒）里不冲突 | 失败现场 `assert [True, True] == [False, True]`（`tests/test_backup.py:134`）；语义与测试同出 `cc4ee4e`，验证跑在沙箱里 → 假绿 | **本轮已修**：新增 `_on_removable_drive()`（`tools/backup.py:103`），盘根探针只保留给它注释里写的场景（可移动盘刚插上）；固定盘缺失就是不可达。修后全套 **298 passed / 0 failed / 0 skipped** |
| F2 | P2（运维） | **离线安全网从未执行**：`.github/workflows/ci.yml` 三件套（离线 pytest + doc_lint + web 构建）选得对，但 `execution` 无 upstream、`master` 落后 `origin/master` **117 个提交**、远端只有 master——近期没有任何提交过过 CI；每日备份 zip 的 48 个文件**不含 Git 历史** | `git branch -vv` / `git branch -r` | 二选一，都需要你点头：推 `execution`（+master）让 CI 真跑；或 `backup.py` 加一行 `git bundle`，历史随每日 zip 走 |
| F3 | P2（设计债） | **双影子引擎**是仓库唯一明知故犯的重复：`shadow_run` 15 个 def、`shadow_variants` 11 个 def，状态存取 / 只追加 / 区间链 / 判读节奏是两份拷贝；A-parity 测试只锁**打分**，不锁**记账** | 两文件 def 清单对比；第四轮已标"骨架漂移" | 2027-02 首次判读前合并成"同一引擎 + spec 注入"，或至少补一条记账 parity 测试（同一 panel 日上 A 分册与主账的 `interval` 字段逐位一致） |
| F4 | P2（文档） | **pipeline.md 的地图停在 11-15**（`7b47760`）：工具清单缺每日链全部 8 个模块；§4 没有四本分册；§5 的 20:00 任务没提变体；§2 目录表没有 `variants/` 与 `reports/runs/`；§1"三层彼此不共享代码"与实测不符；§7 的 `$py` 未定义 | `git ls-files tools` = 29 模块 + 1 脚本 | **本轮已修**（八处；doctrine 未动：三不变量 / 数据口径 / 改动边界逐条核对仍准确） |
| F5 | P3（文档） | implementation_plan 维护段教了本分支跑不了的命令（`src.backtest.run`、`src.training.train`；`src/backtest`、`src/training`、`configs/ppo.json` 都不在 `execution`） | `git ls-files src` | **本轮已修**：归属注释 |
| F6 | P3（判据） | "沙箱下 `0 failed` 仍可用"这条维护判据**双向不成立**：0700 临时目录让好测试假红（运营那轮），盘根探针被拒让坏语义假绿（本轮 F1） | 两轮实测 | **本轮已修**：implementation_plan 判据改为"完全权限下的全绿才算" |
| F7 | P3（数据） | `data/online/history` 3.05 GB 原始归档无保留策略（前向面板有 `prune(keep=3)`，它没有） | [运营检查](operations_20261119.md) §5 实测 | 维持运营检查结论：先体检（已接入每日链），生长率出来再定 |
| F8 | P3（配置/卫生） | `configs/research.json` 的 `backtest`/`download` 段本分支无读者（只有 `src/data/preprocess.py:307` 用 dataset 段）；README 一度同一文档两行索引（`3a0691a` 与快照各一行） | grep / README | 配置无害不动；**重复行本轮已合并** |

> F1 的教训值得单独写：**沙箱里的绿不等于绿**。11-19 那轮"296 passed / 0 failed"跑在沙箱 + shim 下，其中就藏着这个假绿；换成完全权限它自己现了形。同理 F2：CI 从没跑过，意味着 `requirements.txt` 注释里那条 pandas 2.1 concat 警告没有任何常年守夜人。

## 4. 本轮顺手修掉的

1. **F1**：`tools/backup.py` 新增 `_on_removable_drive()` 并把盘根探针限定在可移动盘；`tests/test_backup.py` 一字未动（它钉的语义是对的）。修前 1 failed / 修后 298 全绿。
2. **`3a0691a` 验收**：干跑写 `<日期>-dry.json`（`record_path()`，`tools/daily_ops.py:271`），守卫测试在；运营检查 §6 追记已加，快照文档 §5 同步。
3. **F4**：pipeline.md 八处刷新（三层表述、29 模块清单、`variants/` 与 `reports/runs/` 目录行、分册 bullet、20:00 任务、`$py` 定义）。
4. **F5/F6**：implementation_plan 两条过期命令加归属注释；"沙箱 0 failed"判据更正为"完全权限全绿才算"。
5. **F8**：README 里建议文档的重复索引行合并为一行。

## 5. 留给决策与下一轮

| 项 | 谁定 | 备注 |
| --- | --- | --- |
| F2 二选一（推远端 or `git bundle` 进备份） | 你 | 都是一条命令的事，但推远端是外发动作，不替你定 |
| F3 合并时点 | 2027-02 判读前 | 60 天窗口还长，合并别动主账本的代码路径 |
| 下一轮框架层之后 | — | 值得查：`web/` 构建产物与 `src/web/serve` 的耦合（bundle 名已换过一次）；38 个测试文件对 29 模块的覆盖缺口表；周六资金流任务的实际运行记录抽查 |

## 6. 验证

| 检查 | 结果 |
| --- | --- |
| 全套（完全权限，无 shim） | **298 passed / 0 failed / 0 skipped**，32.7 s |
| `backup.status` 修复前 / 后 | 前 `assert [True, True] == [False, True]`；后 `tests/test_backup.py` 10/10 |
| import 图 | `tools→src` 仅 `forward_panel.py:28-29`；`src→tools` 仅 `src/web` 两处；`src/web` 无 `ledger.use` |
| `git show 3a0691a` | 逐行读：`record_path` + 测试 + README 行，验收通过 |
| `python -B -m tools.doc_lint` | `0 issue(s): {}` |

## 7. 追记（2026-11-20 凌晨，验收时补）

1. **F2 的方向写反了**。实测 `git rev-list --left-right --count origin/master...master` = `0  117`：**master 比 origin/master 多 117 个提交**，不是"落后"。远端最后一个提交是 `b9eddf1`（2026-09-26 18:28）。这让 F2 更严重而不是更轻——**9 月 26 日之后的所有工作只存在于这台机器上**，CI 从未见过它们，异地副本（备份 zip）也不含 Git 历史。
2. **F3 已收一半**。分册补上 `sessions_held`、显式 `null` 的 `interval`、`benchmark` 触发条件，以及交易日闸门（非交易日两边都不写行）；新增 `test_variant_a_books_its_intervals_exactly_like_the_main_ledger`、`test_a_holiday_appends_nothing_in_either_ledger`、`test_the_first_recorded_row_books_an_explicit_null_interval`。**引擎合并**仍按 §5 留在 2027-02 首次判读前，别在主账本代码路径上动刀。追记写在[并行分册](shadow_variants_20261118.md) §6。
3. **§5 第三项（周六资金流抽查）已做**：`schedule-f41e0275` 于 2026-11-14 18:00 交付并执行成功，`data/online/fund_flow/status.json` 记 `requested 1799 / fetched 2 / skipped 1797 / failed 0`、`rows_added 240`、`last_date 2026-11-06`，下一次 2026-11-21——不是"从没跑过"。
4. 追记后全套（完全权限）**301 passed / 0 failed / 0 skipped**。