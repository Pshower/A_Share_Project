# 因果动作价值与 PPO 运行记录

记录更新：2026-10-17 03:11:07 UTC。研究仍在执行，目标保持active；没有本轮完整收益、筛选或确认结论，不把部分训练当作达标。

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

## 待完成工作

## 04:44 UTC 训练段封口与依赖守卫

两个原 worker 均已完成其唯一 inner-train 片段，各480配对标签。随后在进入另进程正在执行的 inner-holdout 之前，按登记规则抛出 `ValueError: Never resume a partial teacher segment`，不是观察超时或静默重训：

| 原 worker | 实际退出码 | 工具 chunk | 训练段完成 marker SHA256 |
| --- | ---: | --- | --- |
| 2023 / session48055 | 1 | `066743` | `f96bab2f3825292789e739926082d0dd1861a2716b1bfc4c6ea1df234c57b0b1` |
| 2024 / session55789 | 1 | `b33533` | `5bf62bba84965e19d7c7499bf14b02af9505e3078584bbd329a2dd2dd42a845e` |

root只读计数核验 `2908f6`、exit0：每段collection为28,800 Agent及同量shadow，branch为19,200 Agent及同量shadow，合96,000 book-step/段。两个训练段共960标签、192,000 book-step；没有RL、NN或拟合。canonical协议SHA均为 `9a2a66336e4a2a472566f492d14e3ba9daada10051209fd7411d57f25d743968`，与原protocol文件SHA属于不同计算口径，不应互相替代。

该非零退出发生在holdout目录存在检查处、该worker进入holdout金融执行前，故没有额外采样、RL或失败尾部金融步骤。留出session46351/32212保持原进程继续，不因主worker退出而重启。仍需等待它们各自实际exit0及封口，再显式重入；Ridge/预热PPO此时尚未执行。

等待期间可开发下一轮候选的独立合成验收工具，但不更改本轮任何冻结输入，不运行新金融实验，不用未闭合B结果调整本轮方法。独立工具的代码/测试成本不算新增市场经历。

## 后续收尾

先取得两个控制进程实际退出及完成清单，再按内存/CPU情况复用其空闲槽；继续所有teacher、唯一Ridge fit和预热PPO，不因offline诊断不好取消任一方法。全过程源模型/数据/训练代码不改，重入及失败成本另列。

全部四个物理run和八个开发快照封口后，运行冻结report与独立cold audit，并把原生NAV/费用/股数、GAE、archive参数/Adam、真实收益增量和选择重新核对。只有开发已登记资格成立才固定唯一方法/预算并执行2025/2026两折、三seed、两臂共12条确认记录。原+0.5pp和风险条件不变、不碰保护期、不挑幸运seed。目标未证明前不标记complete。
