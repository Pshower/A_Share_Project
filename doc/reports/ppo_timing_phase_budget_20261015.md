# 调仓结构与训练预算研究记录

日期：2026-10-15。状态：框架已完成合成验收并正式登记，两个fold worker已启动，筛选运行中；尚无worker退出或封口结果，不填中途收益或满预算完成次数。首次登记点真实金融步骤为0，启动后的实际步骤待封口核对，不把登记点0冒充实时计数。

依据：[Credit详细报告](ppo_timing_credit_20261015.md)及[摘要](ppo_timing_credit_20261015_summary.md)。上一轮relative_log/lambda1在2023比基线高0.977594pp、2024持平，未达准入，确认0。本轮不改判该结果，另行检验架构与预算，规则见[训练前协议](../ppo_timing_phase_budget_protocol_20261015.md)。

## 已实现

- 129维输入、HOLD/REBALANCE_M20两动作与原股票排序/股数/费用执行不变；linear共12,805参数，due残差增加109参数，共12,914。
- 两架构的映射Actor/Critic初始权重、概率、greedy和同种子采样一致；新增参数在九张量Adam中实际更新，元数据与归档使用明确类白名单和严格类型合同。
- 四次32,768步物理训练、八个8,192/32,768快照；短预算是已优化前缀，保留RNG、模式和环境状态，全部评价延后长训练结束。真实RL只计131,072步，不把短快照重复相加。
- 全快照完成后固定排序；每开发折至少+1bp及同路径macro/经济变化才准入。条件确认六run使用同选定预算，本轮最多327,680 RL步；未准入则确认0，不换基线或fallback。
- 缓存先校冻结，再冷读模型、参数、完整Adam覆盖与实际权重差值；两fresh linear短快照需复现Credit对应真实8K控制，不借旧样本。

原96源码及Credit已签4,676文件未改，未下载数据或观察2026年7月以后的保护期性能。新增框架和组件在tools中隔离，默认目录 `reports/runs/ppo_timing_phase_budget_20261015`。

## 训练前设计

| Actor | 8,192步快照 | 32,768步快照 |
| --- | --- | --- |
| linear | 同一新训练的已更新短前缀 | 该物理训练长终点 |
| due_phase_residual | 同一新训练的零初始化残差短前缀 | 该物理训练长终点 |

2023/2024各两架构、seed42，计划四次物理训练，各32,768步。两架构映射后的初始Actor/Critic相同，新增109维向量为零；架构效应按同fold/同预算配对，长短效应按同一训练链配对。短快照不是额外独立样本，也不是独立训练seed。只在8K对应Adam完成后保存，保持训练RNG、模式及账户/游标/待观察状态；短长收益评价在该完整32K训练结束后进行，固定选择还须等四个物理run和八快照全部封口。

隔离范围按当前正式CPU单线程执行披露：保存/恢复的是Python、NumPy及Torch CPU随机状态，不宣称 `preserve_training_rng(cpu)` 保护所有潜在GPU RNG。checkpoint digest核对PPO观察/进度/update、两book的Portfolio/Broker/history/cursor/phase及三个schedule RNG，不是对 `TimingEnv.__dict__` 中所有 `last_*` 字段逐项快照。当前capture没有环境step/reset/observe或金融评价路径，评价在长训练后；fresh linear短前缀的实际数值冷复现将作为保护证据。未发现真实状态漏写，不因此改变本轮方法或冻结协议。

固定relative_log、gamma.995、GAE lambda1、margin2、n_steps512、batch128、epochs5、恒定LR3e-4/clip.2、targetKL.03、entropy0、vf.5、grad clip.5和CPU单线程。观测、两动作及费用执行不变，due残差只在phase0调整REBALANCE的context斜率，不是第三动作、神经选股或新奖励。训练/评价provider仍先按各fold train_end/val_end物理裁剪，不自动延长2026窗口。

固定选择按最差开发折净ROI基线差、两折平均差、最坏账户回撤、费用加滑点排序，完全平局优先linear，再优先8K。每开发折须至少+1bp，并至少一条同路径有greedy macro与经济账本变化；不准入则确认0且无fallback。准入后六个2025/2026确认run重新从头训练所选Actor，预算必须同所选cell，为8K或32K，不比较未选架构确认收益。最终还须满足三seed/跨折+0.5pp/自身初始化增量/风险门槛；已有曝光历史不称首次盲测。

## 准入测试

组件根代理实际43 passed、45.44秒、exit0；最终框架/独审根代理实际74 passed、32.35秒、exit0。两组是不同范围，不相加为金融经历。旧33或64项属于加固前版本，不当作最终验收。

修复包括nested指标role索引、archive float32、8/9张量Adam唯一完整覆盖、初始/快照真实theta差值重算、标量step精度与三public入口冻结早拒。所有模型更新只发生在人工环境；保存加载和伪归档测试不证明收益。

## 正式进度

实现与冻结协议提交 `3bb394e07375dcfd4a6585b60bf144d165f7382d`；根代理正式prepare进程43093实际exit0。机器协议文件SHA `f8fba8ec1616f5f23dfc15ba18153ccacce3bc108c772f0d019a7f1a10e717c3`，父Credit终审硬钉 `fd3449e3d0fcb2df87fa9cd3d87274fa3d93290bb77a3b8d111c5c04751e8955`。prepare完整复核4,676父签名输入，注册原96源码、五个component（两个policy/agent工具与三份组件测试）、runner、两套框架测试、两个旧helper与冻结协议；不能在已登记后重写这些输入。

身份依据[实际机器协议](../../reports/runs/ppo_timing_phase_budget_20261015/protocol.json)，不是从草案推定：runner SHA `9c54115cff7a88e4b0c668fff073d7aaaa3c3336c6e40ae5cc80b42c3db7826a`；冻结人类协议SHA `21596c50dc0aa50b1eaee326abb7fd5fc54ecdcb5c200a9abf66abbdc29a1de9`。五个component输入确切为两个policy/agent工具与三份组件测试；此外两份框架/独审测试SHA分别为 `ce8fda8b34799d9f13cd9014c09b57db1784471109a6db32689366fab01f62fb`、`ef9ef2cf12447685008b2c1a3ddb5120d22c74ab7311e30e21ebec0ca571575a`。所有component/helper逐文件身份均在机器协议中，不以96份src库存替代tools依赖。注册Git如实dirty=true，保留并发用户文件，身份冻结以登记SHA为准。

首次登记后只读核对时正式目录仅有 `protocol.json` 和 `status.json`，状态prepared，未创建run或模型。随后根代理已启动2023 worker session50482及2024 worker session90201，尚无退出或封口结果；不读取未封口收益。下表0仅记录登记点，不代表worker运行时的实时状态：

| 计数 | 计划 | 首次登记点实际 |
| --- | ---: | ---: |
| 筛选物理训练 | 4 | 0 |
| 配对评价快照 | 8 | 0 |
| Agent RL日步 | 131,072 | 0 |
| 独立训练基线日步 | 131,072 | 0 |
| 快照clock之和，不计为RL | 163,840 | 0 |
| 确认run | 条件准入后6 | 0 |
| 确认RL日步 | 49,152或196,608 | 0 |
| 全轮RL日步 | 未准入131,072；准入180,224或327,680 | 0 |

正式Adam、账户/日期/phase覆盖、评价路径、耗时/内存及模型索引待实际封口统计，不按计划反推完成次数。尚无筛选收益或选择结果，确认是否获授权仍待固定门槛，不提前称本轮screen_failed或准入成功。

screen两独占fold工作进程已启动，真实累计步骤、Adam和快照数待其封口统计，不能预填131,072步。匹配共享前缀和九张量模型的独立冷审工具另行准备。仅在四个长run及八快照完整封口后报告开发收益、固定选择和条件确认，不读取未封口中途收益选模。后续补详细账户、资源与独立审计；尚未有本轮候选模型或实盘建议。
