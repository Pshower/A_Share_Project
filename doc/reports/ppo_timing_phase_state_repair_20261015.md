# 调仓状态修复与架构预算研究记录

日期：2026-10-15。状态：新研究目录已prepare、登记独立核验通过，2023/2024正式筛选worker已实际启动，日志已确认采样和Adam更新；两Python进程存活、均未收到真实退出回执。当前仍是linear训练，尚未完成四run/八快照，不预填未来预算；prepare时0步仅是起点记录。尚无本轮封口收益、模型索引或候选，不读取任何未封口ROI。

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

根代理native CIM核对输出de1612确认上述两Python进程正在以 `-B -u` 运行已冻结的 `tools/research_timing_phase_budget_repaired.py`，`--stage` 为screen、`--output` 指向本轮修复root、`--fold` 为对应年份。两个 `background/screen_<fold>.started.json` 已存在；尚无真实exit回执，stdout/stderr在该启动核实点为空只是startup状态，不说明失效，也不证明任何预算已完成。

日志相对本轮正式root位于 `background/screen_2023.stdout.log`、`background/screen_2024.stdout.log`，各有对应stderr日志。supervisor会持续托管各fold的linear、due及对应本地评价，仅执行screen，不自动aggregate、确认或晋升。真实子进程终止后才写 `background/screen_<fold>.exit.json`；该托管回执的 `root_observed_exit=false` 不冒充根代理亲自观察到训练退出，后续还需核验回执与封口文件。

根代理实际监控5a03b4退出0时，2023 linear已2,048/32,768步、Adam80，2024已1,536步、Adam60；均alive、stderr0、无exit或supervisor_error。随后独立只读核验39811c/2cd75e退出0，2023为3,584步/Adam140、2024为3,072步/Adam120，均已越过首次512步/Adam20；恰好两Python及两supervisor，claim/token/脚本SHA/协议匹配，无重复进程。两个初始模型metadata实际clock0、CPU、96源码、完整新协议与driver身份一致。这些是各自读取时的进度点，不是自动更新的实时总量或完成结果；未额外加载网络、读取ROI或执行环境步。

全局 `status.json` 暂保留prepare阶段状态，screen worker独占各自run及回执，不据此误判训练未启动；运行中以后台日志、实际进程和worker回执核对。全部四个物理run封口后，才通过冻结driver汇总screen和决定是否授权确认。

上述是独立外部进程托管机制，以脚本Git/SHA和启动回执绑定，不是新增模型方法或改写repair科学库存10。机器协议文件SHA仍为 `24e7e54419bbc98c97c232187d765ff041be02cbe10f4709fc7db86205785a46`，原96源码与冻结输入不变。本次仅记录启动证据，暂不填写实际RL、Adam、评价步数或中途收益。

最终报告待四physical/八paired snapshot全部封口后写架构×预算经济表、逐账户净ROI/成本/回撤、原生行为、模型路径、RL/基线/Adam/评价去重计数、资源、控制复现和条件确认实际次数。独立冷审核对全Box/账本/macro概率/HOLD/CRC/GAE，decoder首末固定抽样另披露，不冒称全量重解码；最终唯一签章尚不存在。本批尚无候选，不改变默认模型或自动部署，后续配对Q/额外动作不纳入。

## 对外部数据建议的吸收

用户提供[日期关联说明](../training_data_linkage.md)，同目录另有[训练数据改进设计稿](../training_data_improvement_plan.md)。已结合本地实现阅读，未修改用户稿件、未联网复测接口、未采集或接入新数据。设计稿的接口探测属于单机单时点证据，不能据此证明其他历史数据普遍不可得。

优先考虑独立的数据审计分支：区分无源行、指标预热、字段缺失、确定上市状态与未知状态；核查训练期有效覆盖和股票选择偏差，再设计仅用训练期规则的股票留出。`raw_start/raw_end`只是已观测报价边界，不能冒充权威上市/退市日；无源行也不能直接确认为停牌，未知成交量或换手率不能擅自置零。

“填0等于普通行情”的判断不适用于完整现有输入：MarketDataProvider同时输出valid/buy/sell掩码，SharedStockNetwork显式输入标记并排除invalid市场池；本轮macro摘要也只汇总有效行情。近期缺失率、上市年龄等仍可作为增量假设，但须用消融证明其价值，不把已有掩码说成不存在。

截面rank/zscore、广度和离散度值得另行比较，必须限定于当日已知且合格股票、处理样本数与零方差，并明确留出股票是否参与截面统计。现有全局Z-score本身并非错误，它保留长期绝对风险差异；截面量应先作为补充信息做消融，而非认定替换必然更优。新增或替换特征需独立版本、scaler、特征顺序、模型契约和重训；不能原位修改F20或把旧129维模型宣称兼容新语义。新特征不能为了保留原维数而暗中替换输入。

不复权价格与公司行为账本优先服务执行保真，双口径应按同源、同日、同股票和单位校验；事件数量、到账日和基准须独立对账。HFQ/不复权收盘比值不作为权威事件反推，日内触及价格限制也不等于下一次开盘必定不能成交。当前名单加退市名单仍不能自动证明完整PIT宇宙；股票池、执行口径或数据源变化均须重新比较同口径基线，不能将与旧HFQ不同的收益直接视为学习进步。

这些改进可能提高测评可信度，也可能降低原有表观收益，不能承诺因此战胜M20。与原工程修复矩阵分开登记，暂不混入当前科学协议或放宽跨年份、多种子及风险门槛。
