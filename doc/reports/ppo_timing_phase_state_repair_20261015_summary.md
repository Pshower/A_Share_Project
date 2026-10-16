# 调仓架构与预算修复研究摘要

登记日期2026-10-15，结果更新2026-10-16。本批研究执行、四次32K物理训练/八配对快照、固定聚合、机器report、分析、最终独立冷审与唯一签章均已完成。正式选中linear 8K，但2024没有+1bp，`admitted=false`、`screen_failed`、确认0，无合格候选，不部署。技术审计通过不等于收益目标达成。

## 核心结果

以下为2023/2024各自Q2验证窗口（非全年）的四账户平均净ROI，费用与滑点已扣。2023未修改M20基线为5.541507%，2024为1.208622%；仅开发seed42，不是多seed确认。

| Actor/快照 | 2023净ROI % | 基线差 pp | 2024净ROI % | 基线差 pp |
| --- | ---: | ---: | ---: | ---: |
| linear 8K | 6.519102 | +0.977594 | 1.208622 | 0.000000 |
| due 8K | 6.445540 | +0.904033 | 1.208622 | 0.000000 |
| linear 32K | 2.060353 | -3.481154 | -1.499981 | -2.708603 |
| due 32K | -1.467782 | -7.009290 | -0.314241 | -1.522863 |

8K在2023有局部真实增量，2024持平；32K在两架构、两折都低于各自8K与基线。两折平均的32K减8K为linear -3.583676pp、due -4.718093pp。长训练验证退化是实际观察，但不能仅凭一个seed和两个已曝光Q2归因为过拟合、Critic或单纯成本问题，也不能据此说模型没有学习。

due 32K在2024比linear 32K少亏1.185740pp，但仍落后基线1.522863pp；2023 due 32K最坏账户回撤16.565240%。模型容量增加或训练步数增加没有自动带来更好结果。

所有cell均未满足“两开发折各至少+1bp”。linear 8K与due 8K的最差折差均0，linear两折平均增量+0.488797pp高于due的+0.452016pp，由固定第二排序项入选，非tie或事后选模；但2024基线差0，正式确认授权及实际均0。无fallback、无合格候选，不自动部署或晋升默认模型。三seed/跨折+0.5pp最终门槛未验证。

## 实际行为

最终确定性评价的所有变化均为额外非日历REBALANCE，无跳过到期调仓。每账户额外次数8K→32K：2023 linear 2→19、due 3→38；2024 linear 0→16、due 0→25。长预算平均现金降至2023 linear29.856%/due22.034%（基线43.126%），2024为40.804%/39.220%（基线53.464%）。

32K比基线多的费用（`agent_fees - benchmark_fees`，不含slippage）为2023 linear1,549.62/due3,570.69元、2024为2,784.85/3,975.50元；净ROI本身已扣费用和滑点。更频繁调仓、更低现金和更高成本伴随退化，但未做动作反事实拆分，不能说手续费解释全部亏损。旧value explained variance约-.0017至-.2875，均为pre-update旧V对bootstrap目标；目标尺度同时变小，原始MSE降低不等于未来预测提升或Critic因果诊断。

## 方法与计数

修复只对合法拒单history中非有限scalar做typed bit状态哈希编码，不改价格、特征、动作、奖励或模型finite检查。使用原新浪300股HFQ研究代理/F20、训练期scaler、129维因果两book摘要；linear 12,805参数，due增加109维phase0残差至12,914。两动作仍HOLD/REBALANCE_M20，股票执行、T1、冻结持仓、整手、资金和费用沿用旧规则。

四个新physical run实际131,072 Agent RL步及同量训练基线步，八快照clock和163,840不重复计为RL。两fresh linear 8K控制记录逐tensor、Agent/基线原生数组及原八字段buffer精确复现父Credit，不借旧样本。原失败尝试98,304 RL/同量基线、记录3,840 Adam单独记成本，不算修复样本。

实际5,120 Adam，256份unique PPO buffer、320次快照引用；初始16条评价944步、训练后32条1,888步，每book评价2,832步，全native各133,904步。12模型归档为初始/短/长各4；short320、long1,280 Adam按physical去重。1,678登记叶SHA、两个fresh控制及完整科学冷审均通过，decoder抽样范围另行披露。

四个长run的训练/保存/快照诊断耗时累计7,333.82秒，已含其短前缀，不能再加短耗时；不是并行墙钟，不含之后金融评价。新旧工程成本合计229,376 RL、记录8,960 Adam，含旧失败尾部未持久化960，不混作新研究样本。

## 证据边界

机器协议文件SHA `24e7e54419bbc98c97c232187d765ff041be02cbe10f4709fc7db86205785a46`，正式selection文件SHA `deb5327ed3249becfa76bde4d10642db37d3d46f65bf96d8ee88ca28e77a9fb5`；selection中的协议规范化JSON身份 `cd1065dee86e8e037d1a05daa0d45e5372c8a4edf20de49a1006828f0f261dc7`不与文件字节SHA混用。正式输出 `reports/runs/ppo_timing_phase_state_repair_20261015`。原96源码与父4,676签署输入不变；联合工程验收237项/63.98秒是合成验收，不是收益证明。

机器report83062 actual exit0，候选null、qualified/actionable/test_evaluated均false；analyzer87639 actual exit0，[离线诊断](../../reports/runs/ppo_timing_phase_state_repair_20261015/diagnostics/phase_budget_all/diagnostics.json)SHA `7e3f241830f020e65ba4e27c05a1e376dae4767053ccec85d769214cfdb3d2c3`，posthoc未改变选择，增加模型前向/环境步/梯度均0。

阶段冷审24111与最终冷审67704均actual exit0；后者scope为 `complete_authorized_sealed_physical_study`，最终进度SHA `814f0a169770a0458b258888d08b5e3e2070fcb0e608d58c8d70f54332eaa6c7`，不混作同一阶段。四physical/八快照、1,678叶、双book各133,904native、256unique GAE/131,072步均核对，advantage/return递推最大误差0；368次TimeLimit、176自然终止、240活跃tail/16done。Box/账本/旧macro概率/HOLD/CRC全核；decoder仅固定首末抽样Agent608/36,245、基线608/6,696，不冒充全量重解码。

唯一seal39659 actual exit0，[final_audit](../../reports/runs/ppo_timing_phase_state_repair_20261015/final_audit.json) SHA `af731fb246da9dfb15999a73a1761d1e4c804a7525f3865e7395398d9e04b78d`，签6,712文件，含父4,676再核、失败318保留文件及本批输入/结果，原96源码不变。验收保存真实237测试、阶段进程、分析和native托管回执；训练退出来自supervisor WaitForExit/ExitCode，不冒root直接Wait。审计/seal新增NN、环境步、梯度和optimizer均0，不是重跑金融路径；人类最终报告不纳签避免循环。

固定2026股票池、HFQ经济代理、缺完整历史交易状态/公司行为及历史窗口多次曝光限制仍在。未下载新数据、未评价2026年7月后保护期，无合格候选。下一步仅建议另行预注册经济必要性调仓约束、费用后动作级/配对Q信用和真实执行数据核验，均未实施在本轮。完整数据处理、模型、账户、模型路径及事故成本见[详细报告](ppo_timing_phase_state_repair_20261015.md)。
