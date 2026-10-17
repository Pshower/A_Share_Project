# 因果动作价值与 PPO 运行记录

记录更新：2026-10-17 10:07 UTC。研究仍在执行，目标保持active；没有本轮完整收益、筛选或确认结论，不把部分训练当作达标。

## 冻结与工程证据

- 生成/学习框架提交：`95a27b0`；[预注册协议](../ppo_counterfactual_learning_protocol_20261017.md)。生成13输入、原96源码、父封存数据未改。
- root最终生成联合验收：session66504、chunk4f9620、exit0，326 passed/44.21s；含真实SB3合成训练、原生回调、TimeLimit/GAE、保存加载和确认集合守卫。不是历史金融经历。
- 真实派生矩阵只读预检：session15617、chunkb63032、exit0。两折更早inner scaler的实际float32 M20有效股排序不变；没有环境、teacher标签或新模型。
- 正式prepare：session22776、chunkc593e5、实际exit0。机器协议：`reports/runs/ppo_counterfactual_learning_20261017/protocol.json`，文件SHA `2d0685eb7ef2eeefd6fefdf1ca04dbbe3bef07adba35a3c139919938c4786d58`。
- 独立冷审计/图表提交：`77c7084`；root后处理验收session7707、chunk4a0f88、exit0，85 passed/33.83s。尚未对本轮完整结果运行真实冷审计或绘图，不将这些合成检查当作收益证据。

## 实际任务归属

以下是已启动的同一批任务，进程号与完整命令行曾由Win32_Process核对。它们是本记录时的运行状态，不是永久liveness凭证；后续必须重新轮询相应handle/进程，不能仅凭此表重启或宣布退出。

| 任务 | Python PID | root exec session | 当时状态 |
| --- | ---: | ---: | --- |
| screen 2023主worker | 33616 | 48055 | 正在生成inner-train配对标签 |
| screen 2024主worker | 37208 | 55789 | 正在生成inner-train配对标签 |
| 提前2023 PPO-only | 39556 | 10311 | 已越过8K保存点，继续固定32K |
| 提前2024 PPO-only | 20412 | 92968 | 已越过8K保存点，继续固定32K |
| 只读阶段观察器 | 不作金融执行 | 10469 | 监测产物，不增加步数 |

两折第一遍240次source reset均已完成，480状态清单/输入封口后才开始第二遍真实重演和H/R标签。inner-holdout尚待执行，Ridge尚未真实fit，预热PPO尚未开始。每段目标480标签，四段合1,920；这些是预登记目标而不是已完成数。

## 并行安排

详见[操作补充](../counterfactual_learning_scheduling_20261017.md)。`55f91f8`提前调用冻结runner的两个PPO-only案例，每个仍是原case目录、seed42、32,768步和8K/32K快照；原worker之后只读复用完整封口，不再训练一次。

调度入口5项合成验收：root88011、chunkede634、exit0，5 passed/8.98s。未增加模型、方法、seed、标签或RL预算。

`3b10839`另登记对照实际完成后复用空闲CPU槽生成原定inner-holdout的入口；root97327、chunka00080、exit0，两个调度工具合计10 passed/10.43s。该入口在此记录时尚未真实启动，不预填成功或成本。若原worker遇到仍在另进程执行的partial组件，它会明确拒绝；保留实际退出，等独立组件实际exit0并封口后，才可显式重入，只读复用完整组件，不续接或重采partial。

一次提交命令的自动审批超时发生在命令执行前，未影响任何worker；唯一重试取得commit `3b10839`、工具chunkef59eb、exit0。这不是金融重训或因观察超时重启。

## 真实 8K 保存点

root只读两个 `runs/screen_<fold>_ppo_only_42/snapshots/8192/training_audit.json`，工具chunk1ea537、实际exit0，直接核对：

| 字段 | 2023对照 | 2024对照 |
| --- | ---: | ---: |
| 实际PPO步 | 8,192 | 8,192 |
| 实际Adam步 | 320 | 320 |
| Adam覆盖参数tensor | 8 | 8 |
| 原生rollout buffer | 16 | 16 |
| 实际bootstrap侧证buffer | 16 | 16 |
| 相对PPO起点Actor权重变化 | 是 | 是 |
| 相对PPO起点Critic权重变化 | 是 | 是 |
| 保存/加载前后训练状态不变 | 是 | 是 |

原生rollout/credit验算在保存点执行后，两个进程均继续输出8704及之后的真实更新。因此不是只创建了模型目录或计划执行更新。短模型未作为提前选优依据，两个对照继续完整32K；短、长快照的验证都在物理训练结束后执行。

训练前各四种初始账户的θ0与冻结M20对照已由实际run逐项相等核验，否则代码不能进入上述PPO更新。此处不以初始化相等宣称PPO收益相等或已提高。

## 03:38 UTC 运行回执补充

以下更新保留前文的历史快照，不把早前状态当作当前状态。两个提前对照已完成原定训练、两个保存点的回测及 case 封口，取得原生进程退出回执：

| 案例 | session | 实际退出码 / chunk | 唯一实际 RL 步 |
| --- | ---: | --- | ---: |
| 2023 PPO-only | 10311 | 0 / `52029c` | 32,768 |
| 2024 PPO-only | 92968 | 0 / `4f3bda` | 32,768 |

两个启动器均明确输出 `registered_rl_steps=32768 extra_rl_steps=0`。8K 是同一次物理训练的前缀快照，不另计 RL。完整训练步数合计65,536，另有同量 shadow；最终精确更新、回测和收益字段仍须从各 case 封口及独立冷审计核验，不以进程退出代替收益达标。

按已登记调度释放这两个 CPU 槽后，分别启动原定留出段：

| 片段 | 启动时间（北京时间） | Python PID | session | 本次观察 |
| --- | --- | ---: | ---: | --- |
| 2023 inner-holdout | 2026-10-17 11:34:38 | 48972 | 46351 | 运行中，尚无完成回执 |
| 2024 inner-holdout | 2026-10-17 11:37:14 | 55540 | 32212 | 运行中，尚无完成回执 |

命令均为 `python -B -u -m tools.run_counterfactual_holdout --fold <fold>`。03:37 UTC 左右逐一轮询原 session，两个训练段 worker 仍在运行；其日志已分别报告240/480和220/480配对标签。没有重启或复采。上述标签数只是该次日志进度，不是片段最终封口数。

此时四个正式计算进程为两个原 inner-train worker 加两个提前 inner-holdout worker，Win32_Process 已核对 PID 与完整命令行。Ridge 和预热 PPO 尚未启动，全部八个开发快照未闭合，故不执行筛选或确认，也不根据提前对照结果修改方案。

## 04:44 UTC 训练段封口与依赖守卫

两个原 worker 均已完成其唯一 inner-train 片段，各480配对标签。随后在进入另进程正在执行的 inner-holdout 之前，按登记规则抛出 `ValueError: Never resume a partial teacher segment`，不是观察超时或静默重训：

| 原 worker | 实际退出码 | 工具 chunk | 训练段完成 marker SHA256 |
| --- | ---: | --- | --- |
| 2023 / session48055 | 1 | `066743` | `f96bab2f3825292789e739926082d0dd1861a2716b1bfc4c6ea1df234c57b0b1` |
| 2024 / session55789 | 1 | `b33533` | `5bf62bba84965e19d7c7499bf14b02af9505e3078584bbd329a2dd2dd42a845e` |

root只读计数核验 `2908f6`、exit0：每段collection为28,800 Agent及同量shadow，branch为19,200 Agent及同量shadow，合96,000 book-step/段。两个训练段共960标签、192,000 book-step；没有RL、NN或拟合。canonical协议SHA均为 `9a2a66336e4a2a472566f492d14e3ba9daada10051209fd7411d57f25d743968`，与原protocol文件SHA属于不同计算口径，不应互相替代。

该非零退出发生在holdout目录存在检查处、该worker进入holdout金融执行前，故没有额外采样、RL或失败尾部金融步骤。留出session46351/32212保持原进程继续，不因主worker退出而重启。仍需等待它们各自实际exit0及封口，再显式重入；Ridge/预热PPO此时尚未执行。

等待期间可开发下一轮候选的独立合成验收工具，但不更改本轮任何冻结输入，不运行新金融实验，不用未闭合B结果调整本轮方法。独立工具的代码/测试成本不算新增市场经历。

04:52 UTC：[Critic初始化/裁剪探针](../critic_initialization_probe_20261017.md)仅完成独立工程准备，root session25703/f0835c实际exit0，58 passed/24.20s。当前冻结runner及调度入口没有导入该工具，正式金融研究仍只有本轮原任务。

## 后续收尾

先取得两个控制进程实际退出及完成清单，再按内存/CPU情况复用其空闲槽；继续所有teacher、唯一Ridge fit和预热PPO，不因offline诊断不好取消任一方法。全过程源模型/数据/训练代码不改，重入及失败成本另列。

全部四个物理run和八个开发快照封口后，运行冻结report与独立cold audit，并把原生NAV/费用/股数、GAE、archive参数/Adam、真实收益增量和选择重新核对。只有开发已登记资格成立才固定唯一方法/预算并执行2025/2026两折、三seed、两臂共12条确认记录。原+0.5pp和风险条件不变、不碰保护期、不挑幸运seed。目标未证明前不标记complete。

## 10:07 UTC inner-holdout 双折封口

以下为新增只读封口登记，保留前文历史快照。两个提前 inner-holdout worker 均已完成其唯一片段，四段 teacher 至此全部封口：

| 片段 | 封口回执 | 回执写入（北京时间） | completed_sha256 | 额外步数 |
| --- | --- | --- | --- | ---: |
| 2023 inner-holdout | `early_holdout_2023.json`，status=complete | 14:06:20 | `2f18ee14592995d785e6f3436453b6de340ade6d45a8382b6b0ad6ddc76bb3ca` | 0 |
| 2024 inner-holdout | `early_holdout_2024.json`，status=complete | 14:18:43 | `80fa15bb1788a556bb11ec54feea9cab81ee192cc21376c6a9ffaf3d188b0084` | 0 |

回执由 `tools/run_counterfactual_holdout.py` 在段关闭核验（`closed()` 要求计数恰等于预登记教师计数）通过后写出，含 `launcher_sha256=8d7e0ffc…`、`protocol_sha256=9a2a6633…786d58`、`scheduling_only=true`；不续接或重采任何 partial。

四段计数器（只读 `counters.json`）完全一致：每段 480 配对标签、collection 28,800 Agent + branch 19,200 Agent 及同量 shadow；合计 1,920 标签、192,000 Agent step + 同量 shadow = 384,000 book-step，与预登记计划 1,920 标签/384,000 book-step 相符。各段目录 `completed.json`/`counters.json`/`progress.json`/`targets.npz` 落盘（mtime 14:05:42 / 14:18:06 北京时间），480 个标签文件齐全。

17:57 北京时间以 Win32_Process 逐项核验：无任何 python 进程，无未完金融执行、无失败尾部。

当前四项物理 run 中：A 臂两折 PPO-only 已封口（03:38 UTC 节）、teacher 两折 inner-train 已封口（04:44 UTC 节）、teacher 两折 inner-holdout 本节封口。唯一 Ridge fit 的输入前提（sealed teacher segments）已满足；下一步为显式重入冻结 runner：唯一 Ridge fit + 两折 value_warmup 臂，补全全部八个开发快照后再评估筛选资格。协议、门槛、保护期不变，本节不预填收益或筛选结论。

## 11:21 UTC 开发筛选完成、冷审计通过（首次准入）

以下只登记本节新增事实，保留前文历史快照；前文"预热臂待显式重入、八个开发快照未闭合"以本节取代。数字来自已封口封存物、冻结 report 与独立冷审计；本节不预填确认结果。

重入执行（18:11–18:36 北京时间）：预检 FROZEN_OK 后显式重入两折 `--stage screen`，唯一 Ridge fit 于 18:11:21 落盘（`fits/{2023,2024}/fit.json`），两折 value_warmup 训练与 8K/32K 快照回测分别于 18:35:45 / 18:36:07 封口；此前 A 臂 ppo_only run 未被重训（mtime 保持 11:31/11:32）。两折 worker 回执 `worker_screen_2023.json`（SHA `03755152…`，18:36:22）、`worker_screen_2024.json`（SHA `123f51c1…`，18:36:41）status=complete，八个开发快照全部闭合；无失败尾部、无补采，训练期间冻结输入未改。

八个开发快照实际结果（baseline_gap=部署相对 M20；ppo_gain=相对自身辅助起点的 PPO 纯增量；已换算为百分点）：

| 折 | 方法 | 快照 | baseline_gap | ppo_gain | changes / vs_aux（4 账户） |
| --- | --- | ---: | ---: | ---: | --- |
| 2023 | ppo_only | 8,192 | 0.000000 | 0.000000 | 全无 / 全无 |
| 2023 | ppo_only | 32,768 | -7.009290 | -7.009290 | 全有 / 全有 |
| 2023 | value_warmup | 8,192 | +7.220747 | +13.618915 | 全有 / 全有 |
| 2023 | value_warmup | 32,768 | +11.698993 | +18.097161 | 全有 / 全有 |
| 2024 | ppo_only | 8,192 | 0.000000 | 0.000000 | 全无 / 全无 |
| 2024 | ppo_only | 32,768 | -2.895338 | -2.895338 | 全有 / 全有 |
| 2024 | value_warmup | 8,192 | +1.383572 | +0.768440 | 全有 / 全有 |
| 2024 | value_warmup | 32,768 | +1.175065 | +0.559933 | 全有 / 全有 |

8K 与 32K 是同一次物理训练的前缀/终点快照，不重复计 RL 步。冻结 report 于 18:39:04 写出 `selection.json`（SHA `a937e64c…`）、`selection_proof.json`（SHA `66ab692e…`）、`summary.json`：status=`screen_admitted`，admitted=True；候选资格：ppo_only 8K 不达标（000.000000）、ppo_only 32K 不达标（-7.009290）、value_warmup 8K 达标（min +1.383572、mean +4.302160）、value_warmup 32K 达标（min +1.175065、mean +6.437029）。按登记规则取最小 gap 优先，固定唯一方法/预算 `value_warmup/8,192`（minimum_gap 1.383572pp、mean_gap 4.302160pp、worst_drawdown 14.239191%、mean_cost 6,578.007653），确认阶段授权 98,304 RL 步（12×8,192）。本项目首次通过开发准入；最终合格仍取决于 2025/2026 确认与既定风险门槛。

独立冷审计首次真实运行先暴露工具缺陷：审计工具两处 `verify_timing_trace` 期望值误用紧凑分隔符的 `runner.digest`，而冻结评估/训练 trace 由 `src.training.freeze.json_digest`（默认分隔符）签署；实测存盘 SHA `01f6191b…` 等于 freeze 摘要、紧凑摘要 `98ab6eb4…` 不等，全部 32 个 role 用例误报。审计工具不在冻结 INVENTORY 内，修复仅把这两处期望值改为与冻结 writer 完全相同的签署函数（修复前工具 SHA `36bd6bac…`，修复后 `bca56e50…`）；工具自身 67 项合成测试前/后均通过，其未覆盖签署路径是本缺陷逃逸原因，作为覆盖缺口记录。第二遍通过：status=`verified_saved_evidence`，计数器 teacher 1,920 配对标签/384,000 book-step、Ridge fit 2、物理 RL 131,072+同量 shadow、Adam 5,120、评价 3,776/3,776，审计自身 environment/neural/optimizer/fit 全 0（cold 性质成立）；独立重算 outcome 与已保存 `selection.json` 完全一致。sidecar：`diagnostics/cold_audit/screen/audit.json`（SHA `5c04ab80…`）与 `manifest.json`（SHA `acfe1595…`，status=`passed_cold_artifact_audit`、version=`closed_causal_teacher_native_ppo_cold_v1`）；图表 `diagnostics/figures_screen/learning_comparison.png`（SHA `2a442392…`）及 manifest（SHA `7c448e66…`）于 19:19 写出。

下一步（协议、门槛、保护期不变）：按冻结选择显式启动 2025/2026 两折 `--stage confirm`，每折新建数据产物与两段 teacher（各 480 标签/96,000 book-step），3 seeds × 2 臂 × 8,192 步共 12 条确认记录、合计 98,304 RL 步；不挑 seed、不换臂、不重训任何已封口开发 run。确认完成并冷审计通过后才按原门槛判定最终资格。

## 12:30 UTC confirm 首启被窗口关闭中止、无窗口重入

以下登记操作事故与重入事实，保留前文历史快照；不改变协议、选择、阈值与保护期。时间均为北京时间。

19:25:12 以附着控制台窗口的方式启动两折 confirm worker（PID 31156/43108）。两折先把 `data/2025`、`data/2026` 产物落盘并封口，随后进入 `teacher/inner_train` 采集；运行至 20:19–20:24 被控制台窗口关闭中止，stderr 记录 `forrtl: error (200): program aborting due to window-CLOSE event`（MKL/Fortran 运行时对窗口关闭事件的响应），无正常退出码可记，非金融或数值失败。

中止时的实际状态：两折 data 产物完整可复用；`teacher/2025/inner_train` 完成 139/480 标签、`teacher/2026/inner_train` 完成 129/480 标签，均未封口（无 `completed.json`）。按冻结规则 `Never resume a partial teacher segment`，20:26 删除这两个未封口部分目录（2025：143 文件/66.1MB；2026：133 文件/61.5MB），旧日志改名 `counterfactual_confirm_*.failed_window_close.*` 保留；被作废的部分采集不计入研究计数。

20:27:46 以 `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`（无控制台窗口、stdin 置空、stdout/stderr 重定向）分离方式重入两折 worker：PID 40028/2025、23180/2026，日志重新写回 `.workbench/counterfactual_confirm_{2025,2026}.{out,err}.log`。data 产物按原身份复用，两段 teacher 从零重采；本次速度与既有 inner-holdout 段（约 2.5 小时/段，96,000 book-step）同量级，属登记预算内。后续长任务的启动方式统一改为无窗口分离，避免窗口关闭误杀；确认完成前不碰冻结件、不预填确认结果。
