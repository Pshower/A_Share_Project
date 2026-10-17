# 配对动作价值辅助与 PPO 正式学习协议

日期：2026-10-17。状态：数据、隔离、Ridge与完整运行器联合合成验收已完成；真实派生矩阵只读预检通过。尚未 prepare、生成本轮真实 teacher、拟合真实标签或执行正式 PPO，不预填金融收益或候选。正式输出独立为 `reports/runs/ppo_counterfactual_learning_20261017`。本轮必须实际进行两种学习方法的 PPO 对照，不再将仅有 oracle 诊断当作策略进展。

## 研究依据和不变项

父[训练期配对诊断](reports/ppo_counterfactual_pilot_20261016.md)已完整保存两折192条标签，但不是新模型回测：188个非到期状态中90条额外调仓信号为负；phase0每折仅2条且来自同一 episode，不能据此判断跳过到期调仓普遍有效。父[状态修复研究](reports/ppo_timing_phase_state_repair_20261015.md)的32K模型较8K出现更多额外调仓和验证退化，不能据此承诺加大网络或交易频率会提高收益。

继续使用现有本地新浪300股、因果F20、HFQ研究执行代理、129维当前观察、原 `CalendarTimingPolicy` 12,805参数、Discrete(2) HOLD/REBALANCE_M20、原M20排序与订单解码、费用/滑点/整手/T1/冻结/应收、独立shadow账本。HOLD不下单，现金是执行后的余额。价格和模型标准化特征保持分离，不重新下载、不增加股票或未来观察、不换容易基线、不自动部署或实盘交易。

全部既有封存产物和96份src不修改。新数据、teacher、学习、运行和测试在独立tools/输出目录显式冻结；任何未来目标、成交、费用或NAV仅进入标签/证据，不回填策略当前观察。2026-07-01及之后保护期不访问；2025/2026 Q2属于已有研究曝光、但未用于本轮选择的时间确认，不称首次盲测。

## 数据分层与标准化

开发fold为2023、2024，outer train_end为当年3月31日，Q2评价不晚于6月30日。每fold inner_end为 `outer_year - 2` 年12月31日，即2021-12-31、2022-12-31；确认fold2025、2026对应2023-12-31、2024-12-31。所有provider先物理裁掉各自结束日之后的行。

新scaler仅拟合inner-train原始因果特征，inner hold、后续outer PPO和Q2两臂始终使用同一inner-train scaler，禁止在teacher训练后更换坐标并直接复用Actor权重。此选择会使PPO可使用更多outer训练日期，但标准化统计仍来自较早的训练段；两臂和M20完全相同。不沿用父outer-train scaler后声称inner验证无泄漏。相同scaler的正仿射变换应保持M20单特征排序，但仍须用真实数据逐日核对候选排序及同路径基线，不能只凭数学预期宣布核验通过。

inner-train与hold均要求完整H20标签终点属于自身片段；分割处至少留20个交易session purge/embargo，并核查源episode、标签区间和实际决策日期，而不是只按reset日期切分。两段分别将可用episode起点分成early/middle/latest三个连续区间，latest为最后1/3而非单一末日起点。日期、股票历史、两个offset及fold嵌套仍有关联；480或1,920均不是独立市场实验数。

## 精确phase teacher和标签

每fold每段使用固定seed42，4初始账户cash/equal/concentrated/random ×3日期区间×20精确phase，完整240次reset；每episode在offset0和真实推进60步后各留一个状态，共480条。每个精确phase恰有24条，源episode专用80步，为offset60后的完整H20留空间，不是改变PPO的240步episode。

源Agent使用θ0解析Bernoulli行为：phase0的R概率为sigmoid(2)，其余为sigmoid(-2)。独立、显式保存在env状态中的action RNG与账户/date/schedule RNG隔离；不依赖神经前向、不读取旧模型权重。shadow始终依原phase执行日历M20，两本源账户在offset60允许不同。这样同时包含四种原始持仓和真实经历过偏离的持仓，而不从持仓比例拼造账户。

第一遍完整采集全部观察、日期、phase、真实全执行状态/RNG指纹和输入身份，全部sampling plan及输入SHA封口后，第二遍fresh env按同seed、相同源动作流和同reset计划真实重演。逐状态比对指纹和观察后，才从当前完整状态独立克隆两分支。不新增任意pickle或capsule恢复路径，不在看到标签后补采、挑正负、挑绝对值或换seed。

两分支首次分别HOLD/R；后19步均按原phase日历续行，不在首次R后重置20日时钟，各自真实生成订单、执行Broker。两支shadow全账本路径须一致；源状态和全局/局部RNG不变，克隆不得共享可变存储。固定主标签：

```text
Delta = sum(j=0..19) 0.995^j * (relative_log_reward_R[j] - relative_log_reward_H[j])
```

成本已扣入NAV，不重复罚费。未折扣奖励差与终值log(NAV_R/NAV_H)只作审计。该标签是固定规则续行的历史差，不是Q_pi、无偏PPO advantage或未来收益保证。禁止放入PPO rollout buffer、old logP、GAE或Critic回报目标。保留全部正/负/近零样本；提前终止、越界、不完整H20、非有限、指纹差异或clone别名失败均保留实际已付Agent/shadow步数并停止，不补采。H20最后一步合法终止单独标记，无bootstrap。

## 固定Ridge和Actor映射

只用每fold inner-train的480条拟合一次Ridge，hold480条只做固定诊断，不调alpha、epoch、阈值、模型或预算。训练X的各列mean/std只从这480条计算，常量列scale取1。令 `s=max(RMS(Delta_train),1e-8)`、`Z=(X-mean)/std`、`y=Delta/s`，固定闭式解：

```text
min_(beta,b) mean((Z beta + b - y)^2) + 1.0 * ||beta||^2
```

截距不惩罚。保存训练输入/目标hash、mean/std、s、beta/b、正规方程残差和映射回原129坐标的系数。Ridge是真实监督模型fit，不是零训练；每开发两折共960条训练标签曝光，另960条hold诊断，不算NN forward、RL、梯度或Adam步。

原Actor两头设置为以下margin的负/正一半：

```text
q_hat = predicted_Delta / s
logit_R - logit_H = 8 * (q_hat + 0.25 * (2*phase_0 - 1))
```

0.25是固定日历deadband，不是统计置信界。非到期状态需q_hat>0.25才R，到期状态需q_hat<=-0.25才H，tie按普通argmax为HOLD；所有系数随后均由PPO正常训练，不在forward持续叠加冻结teacher。零q时与原±2日历margin的动作概率相同，但两头共同平移的参数gauge不同，不将参数差冒充行为进步。

映射只能作用于真实PPO步数0、更新0、buffer空、Adam状态空的原12,805参数策略；optimizer须覆盖每个参数恰一次。只改两个Actor tensor，Critic及其他tensor逐bit不变，不进行Critic辅助拟合、不创建神经梯度。θ0与θaux分别保存并真实评价，保留变换审计，不用旧模型/旧梯度或旧经验。

hold固定报告原Delta单位MSE、训练目标均值常数预测器MSE、精确phase覆盖、额外/跳过调仓数和单决策 `J_cf=mean[(warmup_greedy_R-calendar_R)*Delta]`。J_cf不是可部署策略ROI，重叠标签收益不串接成净值；离线好坏仅解释，不取消两臂真实PPO。

## 两方法与真实训练矩阵

| 方法 | PPO前状态 | 8,192步已更新快照 | 32,768步终点 |
| --- | --- | --- | --- |
| ppo_only | 原日历θ0 | 同一物理run前缀 | 固定长终点 |
| value_warmup | θ0→θaux | 同一物理run前缀 | 固定长终点 |

开发2023/2024、seed42，共四次fresh物理训练，每次32,768 Agent RL步和同量独立shadow步，合计131,072/本。短快照不重复计经历，全部run和八快照封口后才选；不看8K收益早停、加步、改teacher或挑checkpoint。两臂θ0逐tensor精确相同；相同seed、源日期/reset流、原240步分层episode、PPO配置和评价初始账户。学习导致的实际动作及持仓可以不同。

固定relative_log、gamma0.995、GAE lambda1、n_steps512、batch128、epochs5、LR3e-4、clip0.2、targetKL0.03、entropy0、vf0.5、grad clip0.5、CPU单线程。保存真实SB3 old value/logP/动作、129输入、原8-key rollout buffer和实际TimeLimit/tail bootstrap，不把lambda1称完整Monte Carlo。Ridge额外计算单列，不宣称两臂计算量完全相等；PPO真实RL预算相等，Adam须分别真实计数并核查实际更新。

## 筛选、时间确认与风险门槛

先对全部四个method/budget cell按预登记学习条件过滤：每开发折四账户平均净ROI基线差至少+1bp，且每折 `theta_PPO-theta_0` 平均净ROI增量严格为正；hybrid另须每折 `theta_PPO-theta_aux` 平均净ROI增量严格为正。原生轨迹须分别相对日历基线和PPO起点至少有一条路径同时出现greedy macro与真实经济账本变化。此过滤在任何本轮真实数据或模型之前固定，不是看结果后寻找fallback。

仅对合格cell沿用父固定排序：两开发折平均净ROI基线差的最小值、两折均值、最坏账户回撤、费用加滑点；完全平局先ppo_only再短预算，唯一选一个method/budget。若无合格cell则确认0。筛选不另加10%回撤硬门槛，回撤照实报告并参与排序；最终确认风险门槛不变。确认开始后绝不换臂/预算，不按确认结果fallback。

准入后先固定选择文件及协议/四个开发完成清单的哈希绑定，再在2025/2026×seed42/123/2026从头执行两方法、所选相同8,192或32,768预算。确认仅保存和评价所选终点，严格12条唯一fold/method/seed记录；选32K时也不另评价8K。确认两臂全部封口，但只有开发已选method/budget可成为候选，不看确认换臂、改预算、挑幸运seed。每确认fold另建同规格inner-train/hold teacher；每fold同一个固定teacher fit可共享给三PPO seed，披露共享标签不是三套独立teacher。不得调用未选预算或旧模型权重续训。

最终至少同时满足：各确认折三seed/四账户平均净ROI基线差正；各折至少2/3 seed胜M20；跨折平均优势至少0.5pp；相对各自θ0平均增量正；最坏账户回撤不超过10%，且较同路径基线恶化不超过2pp。若候选为hybrid，还须每确认折 `theta_PPO-theta_aux` 平均增量正，且各折至少2/3 seed该增量正。保持匹配PPO-only结果用于方法归因，不用监督赢家冒充正RL增量；两fold及seed都不是独立市场重抽样，不作不当统计显著性声称。

收益下限及回撤上限的等号比较固定使用绝对容差 `1e-12`，仅处理如 `.0201-.02` 的二进制舍入；不是增加经济容忍区间。严格正的各折/各seed增量仍用 `>0`。准入时从四账户metrics重算三种增量，不信任独立填入的gap scalar；所有12个确认键须完整、唯一且只含所选预算。

## 预算、核验和结果保留

每teacher片段：两遍240×60 collection为28,800 Agent步，480×2×20配对为19,200 Agent步；合计48,000 Agent及同量shadow。每fold两片段为96,000/本，开发两fold为192,000/本，即384,000 book-step，1,920标签、2个真实Ridge fit；其中960标签训练、960只诊断。加开发PPO为646,144 book-step，真实评价另计，不混作RL。

未准入时确认teacher/PPO均为0。准入后确认两fold teacher额外同样1,920标签、384,000 book-step；两方法×两fold×三seed确认PPO在8K预算共98,304 RL/本，32K共393,216 RL/本。故开发加确认最大PPO分别229,376或524,288 Agent RL，加同量shadow；teacher不增加PPO时钟，Ridge/评价/失败尝试另记，不将上限写成已执行。

合成验收应覆盖闭式解与mean-MSE alpha约定、未惩罚截距、常量列/RMS floor、无hold参与fit、129坐标映射、phase/deadband/tie、Critic bitwise、Adam/时钟/buffer前置条件、冻结输入早拒，以及实际rollout/GAE/封口兼容。模型fit、真实金融经历、神经forward、Adam、检查和冷读分别计数。全部真实命令/退出码、工具/数据/模型身份、父签输入核验、错误成本、详细报告和摘要在完成后如实保存；失败也提交，不重写旧研究。

当前尚无本轮真实标签fit、PPO结果、准入或确认结论，不预写正式命令的通过记录。

入口已实现为 `tools/research_counterfactual_learning.py`；数据层静态支持2023至2026四fold，但prepare只构造2023/2024身份，确认worker先校验冻结选择才构造2025/2026。命令如下，仅显式prepare可创建新研究目录；残缺teacher或PPO run拒绝静默续写，完整封口可只读复用。

```powershell
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.research_counterfactual_learning --stage prepare
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.research_counterfactual_learning --stage screen --fold 2023
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.research_counterfactual_learning --stage screen --fold 2024
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.research_counterfactual_learning --stage report
```

仅selection实际admitted后，才执行相同入口的 `--stage confirm --fold 2025`、`--stage confirm --fold 2026` 及最后 `--stage report`；本协议没有预填这些命令已执行。

窄模块作者已用 `AShareGPU` 实际运行 `tests/test_timing_action_value_learning.py`，最新工具结果chunk `1b5323`、exit0：37 passed，6.69秒。范围包括独立增广正规方程、纯训练统计/常量列/目标RMS、129坐标迁移、诊断deadband、未知/tampered fit、非fresh PPO早拒、Critic/全局RNG不变，以及合成buffer上原PPO随后实际更新Actor和Critic且8参数tensor均有Adam state。该合成更新不是历史金融经历，不计正式研究RL/Adam；37不与先前31或36项运行相加，也不冒充数据与运行器联合验收。

冻结前root最终联合命令如下，实际session `66504`、最终chunk `4f9620`、exit `0`：`326 passed in 44.21s`。包括真实SB3/原生TimingTrace/CreditCallback的32步合成市场集成，两个16步rollout、TimeLimit及GAE/returns重算、checkpoint前后完整状态不变、Actor/Critic八个参数tensor实际更新。此为工程验收，不算真实研究RL或独立市场证据。

```powershell
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -m pytest tests/test_timing_causal_data.py tests/test_counterfactual_exact_schedule.py tests/test_timing_counterfactual_v2.py tests/test_timing_action_value_learning.py tests/test_counterfactual_learning_runner.py tests/test_timing_counterfactual.py tests/test_timing_counterfactual_review.py tests/test_counterfactual_storage_check.py -q -p no:cacheprovider
```

root另用纯只读构造器检查两个真实派生矩阵（session `15617`、最终chunk `b63032`、exit0），未创建环境、teacher标签、模型或金融步。更早inner scaler的实际float32 M20全有效股顺序一致，2023检查1,513个有支持日期，2024检查1,754个。实际inner hold起点分别为2022-02-08、2023-02-07，结束2023-03-31、2024-03-29。此项不是收益回测，也不证明新模型效果；完整数据身份由正式prepare保存。
