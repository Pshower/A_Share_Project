# 调仓状态修复与架构预算研究记录

登记日期：2026-10-15；结果更新：2026-10-16。状态：本批研究执行、固定选择、机器report、离线分析、完整独立冷审及唯一签章均已实际完成。正式选中linear 8K，但2024未达到+1bp，`admitted=false`、`screen_failed`、确认授权与实际执行均为0。实际新增131,072 Agent RL步、同量独立训练基线步及5,120次Adam；8K局部增益未跨折重复，32K验证结果退化。工程修复与技术审计通过不等于金融目标达成，无合格候选、不部署、不晋升默认模型。

本报告独立于原尝试，可变的人类进度不回写冻结协议或正式叶文件。设计依据[工程修复附录](../ppo_timing_phase_state_repair_protocol_20261015.md)，原方法遵循[架构与预算协议](../ppo_timing_phase_budget_protocol_20261015.md)。原事故与进度见[原研究记录](ppo_timing_phase_budget_20261015.md)；本轮从头执行，不能将原已有模型或32K经历作为新样本。

## 研究问题与工程边界

已签[Credit结果](ppo_timing_credit_20261015.md)的relative_log/lambda1控制在2023四账户平均净ROI为6.519102%，未修改M20基线为5.541507%，学习增量+0.977594pp；2024两者均1.208622%，增量0。该批固定准入失败、确认0。这些只是已签父研究背景，不是本修复研究的收益，也不改判旧结果成功。

新研究仍只检验两个因素：linear与due_phase_residual架构；同一训练链的8,192与32,768步预算。工程修复只让状态哈希能够记录合法拒单history中的非有限标量，不改变输入、价格、动作、奖励、优化或准入门槛。

## 数据来源与处理

沿用已签本地新浪日线与固定300股研究队列，HFQ用于本轮研究的价量特征和经济执行代理。不下载、不切换数据源、不扩充原始历史、不新增未来标签；具体股票顺序、数据文件与manifest/scaler身份继承原已冻结契约。固定2026股票名单不是各历史日期实时成分股，会存在静态队列及生存者选择限制。

F20由已有日线因果计算：return/log_return、振幅/实体/量比/换手、RSI14、MACD及signal比例、布林带位置，5/10/20/60日动量与均线偏离，20/60日波动率。滚动指标只用决策收盘时已知历史；未成熟、无效数据沿用原valid及交易可用性口径，不填入未来价格。神经网络使用标准化特征，交易估值、股数和费用使用单独保留的研究价格，不把标准化值直接当成交价格。

每fold的scaler只拟合其训练期，验证和推理仅使用该冻结参数transform，不利用验证分布重新拟合。本轮复用已有scaler，不重复构造新特征集。构造train/val provider前分别物理裁到原train_end/val_end，随后才创建环境；2026训练不晚于3月31日、评价不晚于6月30日，不调用旧自动延长窗口路径。2026年7月以后保护期不读取性能或参与选择。

HFQ代理不是历史未复权真实成交价与完整公司行为处理；冻结100股整手、费用/滑点、T1和交易状态口径也不代表完整复原各历史制度与板块。即使研究门槛通过，仍不直接视为实盘交易验收。股票池、feature次序和账户契约固定，不能任意换入新股票或删减输入后声称同一模型合同仍有效。

## 129维观测与两种Actor

| 观测块 | 维数 | 信息 |
| --- | ---: | --- |
| phase one-hot | 20 | 20日周期当前phase |
| market_mean | 20 | 当前有效行情的F20摘要 |
| agent_held | 20 | Agent当前持仓加权F20摘要 |
| benchmark_held | 20 | 独立基线当前持仓加权F20摘要 |
| candidate_mean | 20 | 当前M20 top20候选F20摘要 |
| Agent及基线账户 | 14+14 | 现金、应收、股票敞口、锁仓、可卖、槽位、集中度及数据/交易可用性等 |
| log equity ratio | 1 | 当时Agent/基线净值比 |

这是收盘时当前/过去的macro状态，不是将300股完整lookback张量直接交给新Actor，更不含次日收益或配对Q标签。候选集合仍按当前有效momentum20、股票代码稳定tie及top20旧规则产生，不由神经网络另行选股。

linear Actor是129→2线性Categorical，260参数；Critic为129→64→64→1，共12,545参数，合计12,805。Actor输出0=HOLD、1=REBALANCE_M20的概率/采样token，Critic估计当前策略下折扣未来relative reward，用于PPO value loss和GAE基线。每个run各自联合更新Actor/Critic，不跨run共享同一权重；评价模型参数仍须全部finite。

due_phase_residual只给REBALANCE logit增加 `phase_0 * (u_due · context)`，context为后109维，新增向量初始全0，总12,914参数。其他phase仍用原共同context斜率；不是新增第三动作、股票方向或仓位神经头。两架构映射后的初始Actor/Critic、概率和同种子采样函数一致，新增向量不消费后续采样RNG。linear的8张量与due的9张量须完整注册Adam并实际更新。

初始Actor仅给phase0的REBALANCE与其他phase的HOLD设置margin2，其余Actor输入权重/bias为0。greedy精确复现20日M20基线，Categorical仍允许训练期随机探索；日历初始化只发生一次，所有权重随后可训练，不是永久规则prior或BC。REBALANCE的股票排名、数量、冻结持仓、T1、整手、资金和费用都复用旧decoder/Broker，现金是实际股票成交后的余额；HOLD确实不下单。

## 训练方法与经历

使用标准SB3 PPO，reward为Agent净NAV日log收益减去独立20日M20基线同日净log收益。两book从相同真实Portfolio状态开始，随后独立执行、扣费和推进，不二次扣费用，也不把relative reward冒充净ROI。相同初始greedy与基线意味着净基线差亦是自身初始化增量，二者不是两个独立收益证明。

固定gamma.995、GAE lambda1、LR3e-4、n_steps512、batch128、epochs5、clip.2、targetKL.03、entropy0、vf.5、max_grad_norm.5、CPU单线程。lambda1仍受rollout tail、自然终止和TimeLimit bootstrap约束，不等于240步完整真实Monte Carlo回报。保留原PPO八字段buffer与真实SB3已有tail/terminal sidecar，不为记录增加预测。

240步episode使用cash/equal/concentrated/random四种实际初始持仓、early/middle/latest三日期层、四phase层，共48种分层。初始账户不仅全现金；股票股数、可卖/锁仓、应收及基线book按原规则构造。各32K物理run实际覆盖全部48组合，137个开始episode、136个完整episode和128步尾段、94个去重初始账户；2023去重决策日1,422个、2024为1,657个，两架构同fold排程一致。四run实际合计548个开始、544个完整episode及512尾段步；去重日期/账户不跨run简单相加。反复采样同一原始行情上的不同起点/账户是新增交互经历，不是新增原始数据。

## 配对预算与固定选择

| 架构 | 8K快照 | 32K快照 |
| --- | --- | --- |
| linear | 新物理训练的已优化前缀 | 同一训练长终点 |
| due_phase_residual | 新物理训练的已优化前缀 | 同一训练长终点 |

2023/2024各两架构、seed42，共四次物理训练，各32,768步，新增131,072 Agent RL步及同量训练基线步。八快照clock之和163,840必须另列，不能把四个短前缀32,768步再次计入RL。每physical run只评价一次初始化、一次8K及一次32K，初始化与Adam按physical id去重，不按两快照重复相加。

8K只在其PPO Adam更新完成后保存；当前CPU训练的Python/NumPy/Torch CPU RNG、policy mode及已披露双book/调度状态隔离，不声称所有GPU RNG或环境每一字段均已保护。金融评价延后该32K训练结束；全部四run/八快照封口后才选择，不读短收益早停。两fresh linear 8K控制必须冷读逐tensor、Agent/基线native及原八字段buffer精确复现父Credit对应条件，旧样本不计新训练。

选模按最差开发折净ROI基线差、两折平均差、最坏账户回撤、费用加滑点排序；完全平局优先linear，再8K。所选cell每开发折须至少+1bp，并至少一条同路径同时有greedy macro和实际经济账本变化，才授权确认。未准入则确认0且无fallback；不能以loss、参数变化或单账户收益替代门槛。

准入后2025/2026各seed42/123/2026从头训练选中架构，预算与选中cell相同：六run共49,152或196,608 RL步；本修复正式研究总RL为180,224或327,680，未准入仅131,072。最终各确认折平均正增量、至少2/3seed胜基线、跨折平均优势至少+0.5pp、自身初始化增量正、最坏账户回撤≤10%且相对基线恶化≤2pp必须全部满足。已有曝光Q2仅称本轮选择未使用的时间确认，不是首次盲测；不比较未选架构确认收益。

## 已封口筛选结果

以下为2023/2024各自Q2验证窗口（非全年）的四账户平均净ROI，只读四份 `completed.json` 的八条快照记录；先平均cash/equal/concentrated/random四账户，净ROI已扣费用与滑点。2023未修改M20基线净ROI为5.5415074463%，2024为1.2086218544%，也与各物理run初始化评价完全相同。仅开发seed42，四账户不是四个独立训练seed，不报告多seed置信结论。

| 折 | Actor | 快照步数 | 净ROI % | 基线差/初始化增量 pp | 最坏账户回撤 % | 平均费用+滑点 元 | 平均现金比例 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | linear | 8,192 | 6.519102 | +0.977594 | 13.053802 | 4,308.48 | 37.033280 |
| 2023 | due_phase_residual | 8,192 | 6.445540 | +0.904033 | 12.915895 | 4,333.60 | 36.794170 |
| 2023 | linear | 32,768 | 2.060353 | -3.481154 | 13.244449 | 8,055.97 | 29.855833 |
| 2023 | due_phase_residual | 32,768 | -1.467782 | -7.009290 | 16.565240 | 13,627.07 | 22.034124 |
| 2024 | linear | 8,192 | 1.208622 | 0.000000 | 3.776816 | 3,356.53 | 53.463874 |
| 2024 | due_phase_residual | 8,192 | 1.208622 | 0.000000 | 3.776816 | 3,356.53 | 53.463874 |
| 2024 | linear | 32,768 | -1.499981 | -2.708603 | 5.336066 | 10,990.56 | 40.804035 |
| 2024 | due_phase_residual | 32,768 | -0.314241 | -1.522863 | 4.487119 | 14,260.41 | 39.219575 |

工程修复让所有run走完保存与评价，不等于策略质量提升。两架构8K在2023都有扣成本的真实正增量，2024均与基线相同；两架构32K在两个开发折都低于自身8K与基线。32K成本、换手和持仓路径也改变；仅凭本表不能把退化全部归因于成本，更不能未经训练/验证与梯度诊断就断言过拟合、Critic失效或某个单一原因。

### 配对架构与预算效应

| 净ROI差，pp | 2023 | 2024 | 两折平均 |
| --- | ---: | ---: | ---: |
| linear：32K减8K | -4.458749 | -2.708603 | -3.583676 |
| due：32K减8K | -7.913322 | -1.522863 | -4.718093 |
| 8K：due减linear | -0.073562 | 0.000000 | -0.036781 |
| 32K：due减linear | -3.528135 | +1.185740 | -1.171198 |

长短是同一物理训练链的快照配对，架构比较按同fold/预算配对；两折平均仍只有同一个训练seed，不是n=2独立重复或统计显著性。due在2024长预算少亏1.185740pp，但仍落后基线1.522863pp，不能据此称新架构胜出。观察不支持“步数越多效果必然越好”，也不支持自动追加训练。

### 四账户净ROI

下表直接列32条训练后路径，各账户起点净值100万元，均59个执行日。每physical run的初始化只实际评价一次，8K/32K记录中对其引用不重复计为新路径。

| 折/架构/快照 | cash % | equal % | concentrated % | random % |
| --- | ---: | ---: | ---: | ---: |
| 2023 linear 8K | 6.624865 | 6.545016 | 6.304948 | 6.601579 |
| 2023 due 8K | 6.546959 | 6.475298 | 6.227042 | 6.532862 |
| 2023 linear 32K | 2.168735 | 2.109655 | 1.797705 | 2.165317 |
| 2023 due 32K | -1.398352 | -1.356482 | -1.714524 | -1.401770 |
| 2024 linear 8K | 1.184343 | 1.188374 | 1.260184 | 1.201587 |
| 2024 due 8K | 1.184343 | 1.188374 | 1.260184 | 1.201587 |
| 2024 linear 32K | -1.516890 | -1.520933 | -1.462454 | -1.499645 |
| 2024 due 32K | 0.069868 | -0.734239 | -0.679707 | 0.087113 |

### 两个fresh控制与快照资源

两份linear `reproduction.json` 均记录且经最终独立冷审核对：初始/8K policy tensor、完整Agent与基线native数组、原八字段PPO buffer与Credit对应条件精确相同；`reused_as_sample=false`、`no_extra_environment_steps=true`。这是重新实际训练的控制，不是取父研究模型充样本。保存的新协议/来源元数据与旧研究不同，数值复现不等于整个ZIP字节相同。

| 物理run | 8K训练/保存/诊断累计秒 | 32K训练/保存/诊断累计秒 |
| --- | ---: | ---: |
| 2023 linear | 489.02 | 1,763.06 |
| 2023 due | 496.09 | 1,935.79 |
| 2024 linear | 484.22 | 1,731.77 |
| 2024 due | 494.40 | 1,903.20 |

每行32K耗时已包含该行8K前缀，不能再把短耗时相加；四个32K audit累计7,333.82秒，不是两个worker并行墙钟，也不包含之后金融评价。正式为CPU单线程，无同过程GPU对照。以下实际去重计数已经封口数值复核、机器report对账及最终独立冷审，签章范围见下文。

| 已核对实际计数 | 新修复研究 |
| --- | ---: |
| 物理训练 / 配对快照 | 4 / 8 |
| Agent RL / 独立训练基线步 | 131,072 / 131,072 |
| 实际Adam | 5,120，按四个物理长终点各1,280去重 |
| 共享短前缀 / 快照clock之和 | 32,768 / 163,840，均不再加进RL |
| 原PPO buffer unique / 快照引用views | 256 / 320，短前缀引用不是新经历 |
| 初始化评价 / 训练后评价 | 16路径944步 / 32路径1,888步 |
| 每book全部评价 | 48路径2,832步 |
| Agent / 基线全量native | 各133,904步，即131,072训练加2,832评价 |
| 模型归档 | 12，初始化4、short4、long4 |
| 模型clock / Adam clock | 0/8,192/32,768；初始0、short320、long1,280 |
| linear / due注册参数与Adam张量 | 12,805/8；12,914/9 |
| 新修复确认run / 确认RL | 0 / 0 |

1,678份登记叶文件SHA、两fresh linear控制、完整账本/概率/GAE及已披露的decoder抽样均已通过最终独立冷审。旧尝试98,304 RL、同量训练基线及记录3,840 Adam仍独立列工程成本；新旧合计工程支出229,376 RL及记录8,960 Adam，其中旧失败尾部960 Adam未持久化。这个工程成本合计不构成一条训练链，也不计为四个新样本的训练步数。

### 模型索引

八快照仅供研究复核，不是部署建议；同一行两模型属于一条训练链。对应目录还保留初始化模型、完整配置和原生账本。

| 折/Actor | 8K模型 | 32K模型 |
| --- | --- | --- |
| 2023 linear | [short_model](../../reports/runs/ppo_timing_phase_state_repair_20261015/runs/screen_2023_linear_42/short_model/model.zip) | [final_model](../../reports/runs/ppo_timing_phase_state_repair_20261015/runs/screen_2023_linear_42/final_model/model.zip) |
| 2023 due | [short_model](../../reports/runs/ppo_timing_phase_state_repair_20261015/runs/screen_2023_due_phase_residual_42/short_model/model.zip) | [final_model](../../reports/runs/ppo_timing_phase_state_repair_20261015/runs/screen_2023_due_phase_residual_42/final_model/model.zip) |
| 2024 linear | [short_model](../../reports/runs/ppo_timing_phase_state_repair_20261015/runs/screen_2024_linear_42/short_model/model.zip) | [final_model](../../reports/runs/ppo_timing_phase_state_repair_20261015/runs/screen_2024_linear_42/final_model/model.zip) |
| 2024 due | [short_model](../../reports/runs/ppo_timing_phase_state_repair_20261015/runs/screen_2024_due_phase_residual_42/short_model/model.zip) | [final_model](../../reports/runs/ppo_timing_phase_state_repair_20261015/runs/screen_2024_due_phase_residual_42/final_model/model.zip) |

### 正式固定选择与确认0

根代理新driver固定screen聚合session30301 actual exit0、输出2a5c8e；[正式selection](../../reports/runs/ppo_timing_phase_state_repair_20261015/selection.json)文件SHA为 `deb5327ed3249becfa76bde4d10642db37d3d46f65bf96d8ee88ca28e77a9fb5`。其协议规范化JSON身份为 `cd1065dee86e8e037d1a05daa0d45e5372c8a4edf20de49a1006828f0f261dc7`，不同于协议文件字节SHA `24e7e54419bbc98c97c232187d765ff041be02cbe10f4709fc7db86205785a46`。

driver选中linear/8,192：两个8K cell的最差折差均0，linear两折平均差+0.488797pp高于due的+0.452016pp，因此由第二排序项决定，不是事后择优或依靠tie规则。两个32K cell最差折差均为负。

| 选中linear 8K准入项 | 实际 | 判定 |
| --- | --- | --- |
| 2023平均基线差至少+1bp | +0.977594pp | 通过 |
| 2024平均基线差至少+1bp | 0pp | 失败 |
| 同路径greedy macro及经济账本改变 | 2023四账户均有 | 通过 |
| 四物理run/八快照固定预算完整 | 全部已封口 | 通过 |

正式状态为 `admitted=false`、`screen_failed`，确认授权0、确认实际0，不fallback、不改选其它cell。2025/2026未执行本修复研究确认；三seed/跨折+0.5pp与最终风险门槛尚未验证，不能借前轮确认样本补足。机器report session83062 actual exit0、输出2f4e64；[study_summary](../../reports/runs/ppo_timing_phase_state_repair_20261015/study_summary.json)已确认 `qualified=false`、候选null、`test_evaluated=false`、`actionable=false`，计数与上表一致；不自动训练、下载或晋升。

## 封口行为与信用分配诊断

官方离线analyzer session87639 actual exit0、输出145bd9，只读已封口数据和权重，不实例化模型、不增加神经前向、环境步、梯度或trial叶写入。它是posthoc解释，不更改selection或准入。四份产物的身份如下：

| 产物 | SHA256 |
| --- | --- |
| [diagnostics.json](../../reports/runs/ppo_timing_phase_state_repair_20261015/diagnostics/phase_budget_all/diagnostics.json) | `7e3f241830f020e65ba4e27c05a1e376dae4767053ccec85d769214cfdb3d2c3` |
| [paired_effects.csv](../../reports/runs/ppo_timing_phase_state_repair_20261015/diagnostics/phase_budget_all/paired_effects.csv) | `efad1d5c042106938da419cad96ebb579f1c84965bfa8012809d1bf6a7fe9751` |
| [phase_mode.csv](../../reports/runs/ppo_timing_phase_state_repair_20261015/diagnostics/phase_budget_all/phase_mode.csv) | `23a5a714be1af3b1e4b3a6b12c794775d9c0562eca3cff5c574b2c35c9afe4f5` |
| [rollout_scales.csv](../../reports/runs/ppo_timing_phase_state_repair_20261015/diagnostics/phase_budget_all/rollout_scales.csv) | `f7dd33c2f1f47d6eb858a6383b213c4acf65ba46bc605a2f836f9707bb6fc479` |

### 最终greedy策略学成更多额外调仓

下表次数均为每账户59个执行日的最终确定性策略，四账户宏次数相同，基线各3次REBALANCE。全部最终动作变化都是非日历额外调仓，到期phase0跳过次数全为0；这不表示随机训练时从未探索到期HOLD。

| 折/架构 | 总REBALANCE 8K→32K | 额外非日历调仓 8K→32K | 32K比基线多费用 元 | 32K比基线多滑点 元 | 32K现金比例 / 基线 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 linear | 5→22 | 2→19 | 1,549.62 | 2,761.92 | 29.855833 / 43.125845 |
| 2023 due | 6→41 | 3→38 | 3,570.69 | 6,311.95 | 22.034124 / 43.125845 |
| 2024 linear | 3→19 | 0→16 | 2,784.85 | 4,849.17 | 40.804035 / 53.463874 |
| 2024 due | 3→28 | 0→25 | 3,975.50 | 6,928.38 | 39.219575 / 53.463874 |

费用与滑点分别按四账户平均；“多费用”为 `agent_fees - benchmark_fees`，不含slippage，不能误称含滑点总成本；净ROI本身已扣两者。长预算观察到更频繁调仓、更低现金和更高成本，与净收益退化同时出现；没有做逐动作HOLD/REBALANCE配对反事实或终值拆分，因此不能说费用解释了全部损失，或证明某一个调仓造成全部差异。相同原始历史上更长的PPO训练确实改变了行为，不是模型根本未更新。

### 旧value误差下降不等于Critic效果提升

| 折/架构 | pre-update旧V MSE 8K→32K | 旧V explained variance 8K→32K | 每rollout目标std均值 8K→32K |
| --- | ---: | ---: | ---: |
| 2023 linear | .04090697→.01310433 | -.01244880→-.04982722 | .07901146→.05106106 |
| 2023 due | .04089980→.01313179 | -.00165470→-.05033162 | .07917080→.05103708 |
| 2024 linear | .03981948→.01305390 | -.28745388→-.28590201 | .06976188→.04766786 |
| 2024 due | .03983460→.01321267 | -.27906955→-.27334181 | .07030428→.04743898 |

旧V与该次实际采样时保存的bootstrap GAE目标比较，不是训练后统一θ的拟合质量，也不是真实未来净ROI预测。8K汇总16个rollout，32K汇总64个且包含同一短前缀；各批目标尺度、状态和当时θ都会变化。目标std也下降，因此不能把原始MSE降低约三分之二等同预测能力变好。这里explained variance均负，只反映该存储旧V/估计目标口径的解释度有限；不足以单独证明Critic是策略退化原因。

## 冷审与最终签章

已封口阶段独审session24111 actual exit0、输出bf147b，当时进度SHA `acfffd9325cb2d551f15942493f43f7ae6cd2a109658f8f6e83ee8f4b2ed396f`。这是阶段性scope，不冒充官方final审核或唯一签章。

| 最终冷审核验范围，阶段核验亦涵盖 | 覆盖 |
| --- | --- |
| 物理run / 快照 / 登记叶 | 4 / 8 / 1,678 |
| Agent / 基线native步 | 各133,904 |
| unique实际GAE buffer / 步数 | 256 / 131,072，短引用不重算作新经历 |
| advantage / return最大逐值递推误差 | 0 / 0 |
| unique TimeLimit / 自然终止 | 368 / 176 |
| 活跃tail / done tail | 240 / 16 |
| 129维Box、经济账本、存储旧macro概率/logP、全部HOLD与CRC | 完整核对通过 |
| Agent股票decoder | 固定每chunk首末真实REBALANCE，608 / 36,245行 |
| 基线股票decoder | 同一固定首末抽样，608 / 6,696行 |

decoder明确是抽样，不说逐REBALANCE全量重解码。官方 `--stage audit --sealed --final` session67704 actual exit0、输出37b5f7，状态passed、scope为 `complete_authorized_sealed_physical_study`；最终进度SHA为 `814f0a169770a0458b258888d08b5e3e2070fcb0e608d58c8d70f54332eaa6c7`。此前24111只是阶段核验，不与最终67704混称同一次审计。

唯一seal session39659 actual exit0、输出2516a5；[final_audit.json](../../reports/runs/ppo_timing_phase_state_repair_20261015/final_audit.json)文件SHA为 `af731fb246da9dfb15999a73a1761d1e4c804a7525f3865e7395398d9e04b78d`，状态passed，签署6,712份文件。范围包含父4,676签输入再核及其final审计自身、旧失败318保留文件、本批1,678登记叶与协议/工具/机器证据等，原96源码不变。本人类详细报告与摘要不在签署库存内，避免引用审计SHA的循环签名。

实际acceptance收据纳入237项工程测试、各root阶段进程、87639分析、最终独审及8份native托管回执原文/SHA。后台退出仍来自supervisor的 `WaitForExit/ExitCode`，`root_observed_exit=false` 原样保留，不冒称root直接等待到训练进程退出。进程验收和科学结果验收分别披露。

最终独立审计新增模型实例、NN前向、环境步、梯度、optimizer、登记叶/冻源写入均为0；seal亦为这些新增模型/金融/梯度操作0，只核已有冷审证据与身份/SHA，不声称又重跑全部金融路径。技术scope passed不表示策略准入或收益成功：`screen_admitted=false`、确认0、`gate_passed=false`、候选null、保护期未评价、默认模型未晋升。6,712份已签输入此后不再改写。

## 后续研究方向，尚未实施

1. 对非日历调仓建立可检验的经济需要约束：只用当前目标持仓差异、可执行净调整量和预估费用/滑点定义门槛，比较原方法与新约束，不用未来收益判定必要性。训练采样、概率/logP支持及部署decoder应一致，不只在回测端过滤。净reward已经扣费，不能无说明再重复扣费用。
2. 动作级费用后信用与配对反事实训练：从同一完整账户deepcopy分支HOLD/REBALANCE，按预先固定continuation比较20/60步净log终值差；标签终点只到train_end，不进入观察或验证标签。需明确这是固定continuation Q而非Qπ，设置同初始化PPO-only对照，监督曝光、分支环境步、PPO经历和Adam分别计数。本轮未生成这些标签或开展新训练。
3. 数据执行保真：独立版本核验同源未复权报价、公司行为与到账状态，并与HFQ代理使用同口径基线比较；同时审计PIT股票池、缺失状态与训练期覆盖。改善可信度不保证收益更高，也不把价格口径切换后的收益差直接当作RL增量。

这些方向需要新协议、预算、数据/特征合同和独立对照，不实施在当前被冻结研究，不回改失败准入或以更多预算补挑结果。

## 原工程事故与保留成本

原2023 worker50482实际exit1、输出chunk984001；linear已经32,768 RL步、1,280次Adam，最终capture因 `_json_state`→strict `json_digest` 的非有限scalar失败。final_model未save、无completed；8K模型、64个buffer、64个sidecar及双book各32K trace存在。原2024 worker90201已实际exit0、退出结果81fbdc，linear/due均completed并保存32K。仅记录已确认事实，不读取未封口收益或推断原完整矩阵绩效。

合法拒单history可含 `price=NaN`、`filled=0`。工程修复仅增加状态哈希scalar的typed NaN/+Inf/-Inf bit编码，不补行情、不将NaN换成0、不改变财务奖励，也不放宽模型finite检查。通过新显式CLI context临时安装codec与增强guard，结束或异常必须恢复原函数，原bytes不改。

旧失败manifest固定318份文件，SHA `8b2c1948e89094ab22267c8dff96e2caef27a8fd2cd89815b5e70e8f3eb39b2b`，不含当时仍写入的2024运行。它是工程证据清单，不是完整绩效签章。旧三次物理尝试实际Agent RL与训练基线各98,304步、记录Adam3,840次；最高持久化端点合计只证明2,880次，失败尾部960次未落盘。各book评价1,652步、现存模型归档CRC通过；详见原报告和工程成本记录。不抹掉、不借入新四run，不补保存旧最终模型，不重放梯度。

## 注册与当前进度

修复附录与原失败人类记录先行提交 `65d40e1`，新driver为 `tools/research_timing_phase_budget_repaired.py`，配套 `tests/test_timing_phase_state_repair.py` 与 `tests/test_timing_phase_state_repair_review.py` 已完成。根代理十份新旧框架及辅助测试实际session35708、exit0（d17d2c）：237 passed、63.98秒。作者/独审另一次61项、51.39秒不与该运行相加；均非金融经历。根代理prepare session30574实际exit0、输出306698，正式root `reports/runs/ppo_timing_phase_state_repair_20261015` 已登记为prepared。

实际[修复版机器协议](../../reports/runs/ppo_timing_phase_state_repair_20261015/protocol.json)文件SHA为 `24e7e54419bbc98c97c232187d765ff041be02cbe10f4709fc7db86205785a46`，注册Git为 `29fdc89a6f29878313ce412c895b74d63f17b5a1`、dirty=true。保留用户IDE/PDF等已有变更，不擅自回退；同提交中的用户建议文档与索引不是本轮新特征、数据或科学输入。协议身份以实际登记逐文件SHA为准，不以dirty标记代替来源核验。

`engineering_repair` 版本为 `typed_nonfinite_scalar_bits_v2`，已绑定10份新driver/两测试/修复附录/原辅助审计分析封存工具及其测试，以及318文件失败manifest和原core身份。关键身份如下：

| 登记输入 | 实际SHA256 |
| --- | --- |
| 新driver，亦为新协议顶层tool_sha256 | `8850bd2cb1180c2eb811958dd84a244bd450664ecebe57f7984c71b0212c26d6` |
| 修复作者测试 | `85a911031e2c1e70eb52df1ffc0267baf424e95a71a2f3b3e5f68704f88bbc89` |
| 修复独立测试 | `192acc6810fc6bb48f1bc02d6716982a141b28d7d01e33b64eddcb1a4572fa43` |
| 修复附录 | `b2cb5a3959dff02db57fde01701fce3d1e05fba04a36c313da051dd9e3d5f540` |
| 原core架构预算人类协议，仍为顶层document_sha256 | `21596c50dc0aa50b1eaee326abb7fd5fc54ecdcb5c200a9abf66abbdc29a1de9` |
| 原core机器协议 | `f8fba8ec1616f5f23dfc15ba18153ccacce3bc108c772f0d019a7f1a10e717c3` |
| 原core runner | `9c54115cff7a88e4b0c668fff073d7aaaa3c3336c6e40ae5cc80b42c3db7826a` |
| 父Credit唯一终审 | `fd3449e3d0fcb2df87fa9cd3d87274fa3d93290bb77a3b8d111c5c04751e8955` |

全部96份原源码、原component/helper/test库存及4,676父签输入保持不变；10份repair库存的其余逐文件身份见机器协议，不用新driver SHA替代旧core身份，也不混用机器协议文件字节SHA与规范化JSON身份。

新协议顶层使用新driver SHA，旧普通入口拒绝。只有显式context以不写文件/不传模型的只读core视图验证旧runner身份，随后验证完整repair扩展；真实配置和模型绑定完整新协议及新driver，不把两身份混用。以上身份来自prepare后实际机器协议，不从草案推定。

登记独立审计session77863实际exit0、输出5ae775，状态passed、scope为 `frozen_registration`，协议SHA与上述一致；进度文件SHA为 `3904a414113af0316f812c08ed477abc9f2ddfdee85eb7de02f8781743a28005`。父4,676输入、原96源码、repair10库存及失败318保留文件均核对通过。此次模型实例、神经前向、环境步、梯度、optimizer和登记leaf写入均为0，不是正式训练或收益验收。

该prepare/登记核实点曾为 `runs_exists=false`：当时目录有协议、prepared状态及只读登记审计进度记录，尚无训练run；Agent RL、训练基线、Adam、模型与评价计数均为0。这是已发生的起点记录，不是以下启动后的实时状态。旧尝试Agent/基线各98,304步仍另列，不借作新样本。

### 正式后台筛选启动

持久化launcher与prepared报告提交 `fc450a7`。首次裸PowerShell脚本因本机RemoteSigned限制被拒，根代理输出c1c298、exit1；脚本未执行，没有遗留后台worker。仅作过程级ExecutionPolicy设置的修正提交 `5128c9a` 后，两个启动过程均实际exit0，不修改系统或用户全局执行策略。

最终 `tools/start_phase_state_repair.ps1` SHA为 `d24717d2455703b8d81ca609d23e5d4ef1e6667ab226a1b99e0da924b260ff7a`，不是作者早期8131开头版本。启动命令为：

```powershell
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File .\tools\start_phase_state_repair.ps1 -Fold 2023
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File .\tools\start_phase_state_repair.ps1 -Fold 2024
```

| fold | 根代理启动过程 | supervisor PID | 实际Python PID | started UTC |
| --- | --- | ---: | ---: | --- |
| 2023 | session60388，exit0，输出878c0e | 6836 | 46492 | 2026-10-15T15:17:35.503399+00:00 |
| 2024 | session95935，exit0，输出229abd | 34772 | 46676 | 2026-10-15T15:18:24.050988+00:00 |

根代理native CIM核对输出de1612确认当时上述两Python进程以 `-B -u` 运行已冻结的 `tools/research_timing_phase_budget_repaired.py`，`--stage` 为screen、`--output` 指向本轮修复root、`--fold` 为对应年份。两个 `background/screen_<fold>.started.json` 存在；该启动核实点尚无exit回执，stdout/stderr为空只是当时startup状态，不是本次结果更新时的状态，也不证明当时预算已完成。

日志相对本轮正式root位于 `background/screen_2023.stdout.log`、`background/screen_2024.stdout.log`，各有对应stderr日志。supervisor持续托管了各fold的linear、due及对应本地评价，仅执行screen，不自动aggregate、确认或晋升。真实子进程终止后才写 `background/screen_<fold>.exit.json`；该托管回执的 `root_observed_exit=false` 不冒充根代理亲自观察到训练退出。回执及封口身份现已纳入最终验收核对。

根代理实际监控5a03b4退出0时，2023 linear已2,048/32,768步、Adam80，2024已1,536步、Adam60；均alive、stderr0、无exit或supervisor_error。随后独立只读核验39811c/2cd75e退出0，2023为3,584步/Adam140、2024为3,072步/Adam120，均已越过首次512步/Adam20；恰好两Python及两supervisor，claim/token/脚本SHA/协议匹配，无重复进程。两个初始模型metadata实际clock0、CPU、96源码、完整新协议与driver身份一致。这些是各自读取时的进度点，不是自动更新的实时总量或完成结果；未额外加载网络、读取ROI或执行环境步。

在worker运行期间，全局 `status.json` 曾保留prepare阶段状态，screen worker独占各自run及回执，不能据此误判训练未启动；当时以后台日志、实际进程和worker回执核对。全部四个物理run封口后，冻结driver才汇总screen及report，并将机器状态记为 `screen_failed`、确认0。

上述是独立外部进程托管机制，以脚本Git/SHA和启动回执绑定，不是新增模型方法或改写repair科学库存10。机器协议文件SHA仍为 `24e7e54419bbc98c97c232187d765ff041be02cbe10f4709fc7db86205785a46`，原96源码与冻结输入不变。启动阶段0步/存活记录不覆盖最终实际结果。

两个实际后台exit回执均为 `child_exited`、exit0、`System.Diagnostics.Process.WaitForExit/ExitCode` 来源：2023结束UTC `2026-10-15T16:25:06.4236125+00:00`，2024结束UTC `2026-10-15T16:24:40.3701903+00:00`，均为北京时间2026-10-16凌晨；`root_observed_exit=false` 仍如实保留。它们证明物理进程结束，不单独替代固定聚合或完整科学审计；本批另有已完成的官方聚合和独立冷审。

八快照经济表、逐账户净ROI、模型与原始耗时见已封口结果；固定聚合、机器report、官方离线分析、最终独立冷审及唯一签章均actual exit0，正式确认0。本批研究执行完成但筛选准入失败；无合格候选，不改变默认模型或自动部署，后续配对Q/额外动作不纳入。

## 对外部数据建议的吸收

用户提供[日期关联说明](../training_data_linkage.md)，同目录另有[训练数据改进设计稿](../training_data_improvement_plan.md)。已结合本地实现阅读，未修改用户稿件、未联网复测接口、未采集或接入新数据。设计稿的接口探测属于单机单时点证据，不能据此证明其他历史数据普遍不可得。

优先考虑独立的数据审计分支：区分无源行、指标预热、字段缺失、确定上市状态与未知状态；核查训练期有效覆盖和股票选择偏差，再设计仅用训练期规则的股票留出。`raw_start/raw_end`只是已观测报价边界，不能冒充权威上市/退市日；无源行也不能直接确认为停牌，未知成交量或换手率不能擅自置零。

“填0等于普通行情”的判断不适用于完整现有输入：MarketDataProvider同时输出valid/buy/sell掩码，SharedStockNetwork显式输入标记并排除invalid市场池；本轮macro摘要也只汇总有效行情。近期缺失率、上市年龄等仍可作为增量假设，但须用消融证明其价值，不把已有掩码说成不存在。

截面rank/zscore、广度和离散度值得另行比较，必须限定于当日已知且合格股票、处理样本数与零方差，并明确留出股票是否参与截面统计。现有全局Z-score本身并非错误，它保留长期绝对风险差异；截面量应先作为补充信息做消融，而非认定替换必然更优。新增或替换特征需独立版本、scaler、特征顺序、模型契约和重训；不能原位修改F20或把旧129维模型宣称兼容新语义。新特征不能为了保留原维数而暗中替换输入。

不复权价格与公司行为账本优先服务执行保真，双口径应按同源、同日、同股票和单位校验；事件数量、到账日和基准须独立对账。HFQ/不复权收盘比值不作为权威事件反推，日内触及价格限制也不等于下一次开盘必定不能成交。当前名单加退市名单仍不能自动证明完整PIT宇宙；股票池、执行口径或数据源变化均须重新比较同口径基线，不能将与旧HFQ不同的收益直接视为学习进步。

这些改进可能提高测评可信度，也可能降低原有表观收益，不能承诺因此战胜M20。与原工程修复矩阵分开登记，暂不混入当前科学协议或放宽跨年份、多种子及风险门槛。
