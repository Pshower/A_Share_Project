# 调仓状态修复与架构预算研究记录

日期：2026-10-15。状态：工程修复附录已提交，独立wrapper及作者/独立测试仍在开发，新研究目录尚未prepare。此预注册核实点的正式训练、评价、确认步骤均为0；它不是未来worker启动后的实时计数。未填测试通过数、收益、模型索引或候选，不读取原目录任何未封口ROI。

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

240步episode使用cash/equal/concentrated/random四种实际初始持仓、early/middle/latest三日期层、四phase层，共48种分层。初始账户不仅全现金；股票股数、可卖/锁仓、应收及基线book按原规则构造。反复采样同一原始行情上的不同起点/账户是新增交互经历，不是新增原始数据。实际覆盖、完整/尾段episode和去重日期/账户数待封口统计。

## 配对预算与固定选择

| 架构 | 8K快照 | 32K快照 |
| --- | --- | --- |
| linear | 新物理训练的已优化前缀 | 同一训练长终点 |
| due_phase_residual | 新物理训练的已优化前缀 | 同一训练长终点 |

2023/2024各两架构、seed42，共四次物理训练，各32,768步，新增131,072 Agent RL步及同量训练基线步。八快照clock之和163,840必须另列，不能把四个短前缀32,768步再次计入RL。每physical run只评价一次初始化、一次8K及一次32K，初始化与Adam按physical id去重，不按两快照重复相加。

8K只在其PPO Adam更新完成后保存；当前CPU训练的Python/NumPy/Torch CPU RNG、policy mode及已披露双book/调度状态隔离，不声称所有GPU RNG或环境每一字段均已保护。金融评价延后该32K训练结束；全部四run/八快照封口后才选择，不读短收益早停。两fresh linear 8K控制必须冷读逐tensor、Agent/基线native及原八字段buffer精确复现父Credit对应条件，旧样本不计新训练。

选模按最差开发折净ROI基线差、两折平均差、最坏账户回撤、费用加滑点排序；完全平局优先linear，再8K。所选cell每开发折须至少+1bp，并至少一条同路径同时有greedy macro和实际经济账本变化，才授权确认。未准入则确认0且无fallback；不能以loss、参数变化或单账户收益替代门槛。

准入后2025/2026各seed42/123/2026从头训练选中架构，预算与选中cell相同：六run共49,152或196,608 RL步；本修复正式研究总RL为180,224或327,680，未准入仅131,072。最终各确认折平均正增量、至少2/3seed胜基线、跨折平均优势至少+0.5pp、自身初始化增量正、最坏账户回撤≤10%且相对基线恶化≤2pp必须全部满足。已有曝光Q2仅称本轮选择未使用的时间确认，不是首次盲测；不比较未选架构确认收益。

## 原工程事故与保留成本

原2023 worker50482实际exit1、输出chunk984001；linear已经32,768 RL步、1,280次Adam，最终capture因 `_json_state`→strict `json_digest` 的非有限scalar失败。final_model未save、无completed；8K模型、64个buffer、64个sidecar及双book各32K trace存在。原2024 worker90201的linear已completed/final32K，随后due仍活跃。仅记录根代理已确认事实，不读取未封口收益或推断原完整矩阵绩效。

合法拒单history可含 `price=NaN`、`filled=0`。工程修复仅增加状态哈希scalar的typed NaN/+Inf/-Inf bit编码，不补行情、不将NaN换成0、不改变财务奖励，也不放宽模型finite检查。通过新显式CLI context临时安装codec与增强guard，结束或异常必须恢复原函数，原bytes不改。

旧失败manifest固定318份文件，SHA `8b2c1948e89094ab22267c8dff96e2caef27a8fd2cd89815b5e70e8f3eb39b2b`，不含仍写入的2024运行。它是工程证据清单，不是完整绩效签章。原已付RL、训练基线、Adam与耗时在旧worker结束后单独结算，不抹掉、不借入新四run，不补保存旧最终模型，不重放梯度。

## 注册与当前进度

修复附录与原失败人类记录已提交 `65d40e1`，新driver为 `tools/research_timing_phase_budget_repaired.py`，配套 `tests/test_timing_phase_state_repair.py` 与 `tests/test_timing_phase_state_repair_review.py` 仍在开发。当前未宣称验收通过，正式root `reports/runs/ppo_timing_phase_state_repair_20261015` 尚未prepare，无本轮正式训练、评价、确认模型或收益。

拟登记的 `engineering_repair` 将绑定新driver、两测试、冻结附录、原辅助审计/分析/封存工具及测试、失败manifest和原core身份。原机器协议SHA `f8fba8ec1616f5f23dfc15ba18153ccacce3bc108c772f0d019a7f1a10e717c3`，原core runner SHA `9c54115cff7a88e4b0c668fff073d7aaaa3c3336c6e40ae5cc80b42c3db7826a`，父Credit终审 `fd3449e3d0fcb2df87fa9cd3d87274fa3d93290bb77a3b8d111c5c04751e8955`；原96源码与4,676父签输入保持不变。

新协议顶层使用新driver SHA，旧普通入口应拒绝。只有显式context以不写文件/不传模型的只读core视图验证旧runner身份，随后验证完整repair扩展；真实配置和模型绑定完整新协议及新driver，不把两身份混用。实际协议文件SHA、Git、测试退出与新root身份须在正式prepare成功后填入，不从草案推定。

最终报告待四physical/八paired snapshot全部封口后写架构×预算经济表、逐账户净ROI/成本/回撤、原生行为、模型路径、RL/基线/Adam/评价去重计数、资源、控制复现和条件确认实际次数。独立冷审核对全Box/账本/macro概率/HOLD/CRC/GAE，decoder首末固定抽样另披露，不冒称全量重解码；最终唯一签章尚不存在。本批尚无候选，不改变默认模型或自动部署，后续配对Q/额外动作不纳入。
