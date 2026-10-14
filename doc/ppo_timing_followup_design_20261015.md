# 调仓控制器后续候选设计

日期：2026-10-15。**状态：候选设计，不是新预注册；未实现、未生成标签、未训练，不承诺收益。** 当前Credit研究继续按其冻结协议运行，本文件不判定它成功或失败。选择下一正式协议必须等当前八trial完整封口、固定选择与真实结论产生后进行。

原96份src、已签产物、当前Credit协议/runner/tests均不改。未来若采用候选，应新增隔离实现、工具与测试，完整登记代码/数据/模型/优化器身份，不能动态替换旧模块全局变量或修改正在运行的模型。

## 方向一：到期日行情交互

### 现有结构与假设

现有[CalendarTimingPolicy](../src/agents/timing_policy.py#L20)使用129维观察：前20维phase one-hot、后109维行情/账户/候选摘要。Actor是线性两logit，Critic为64/64 MLP，共12,805参数；[初始化](../src/agents/timing_policy.py#L55)使phase0选择REBALANCE，其余选择HOLD，参数之后均可训练。

令 `c` 为109维context，`j`为phase，现有二分类logit差可写为：

```text
d(s) = logit_REBALANCE - logit_HOLD
     = phase_intercept[j] + common_context_slope · c
```

phase已能改变阈值，不是完全无法区分调仓日；限制是各phase共用context斜率。假设：原定到期日决定是否跳过，与非到期日决定是否提前调仓，可能需要不同的行情/账户响应。此假设尚未被当前Credit结果验证，不能先认定它是失败根因。

### 最小候选

仅增加一个109维可训练向量 `u_due`，不新增bias、观察或动作：

```text
d_new(s) = d(s) + phase_onehot[0] * (u_due · c)
u_due = zeros(109) at initialization
```

到期日斜率变为 `common_context_slope + u_due`，其他phase仍共用原斜率。候选总参数为 **12,914 = 12,805 + 109**；Actor由260增至369，Critic仍12,545。它不是每个phase独立109维网络，也不是第三动作、神经选股或时间序列模型。

HOLD/REBALANCE、categorical概率、M20排名/数量、Broker/冻结/T1/整手/成本和129维输入保持不变。[phase观察与递进](../src/training/timing_env.py#L107)继续使用原日历锚点，不因某天提前或跳过调仓重置phase。

### 同初始化与优化器

- 用全零 `nn.Parameter` 初始化新向量，避免随机Linear初始化后再清零所消耗的RNG。相同seed的原Actor/Critic张量逐值一致，并核验构造后Python/NumPy/Torch RNG状态；新残差为0时，整个初始策略函数、value、categorical logP及确定性订单/净值应与控制精确一致。
- 原 `_build` 已构造优化器。不能事后新增Parameter却继续用旧optimizer；未来实现须在任何更新前完成新向量注册，并由新policy显式构造覆盖全部12,914参数的优化器。不能移植旧模型的Adam状态冒充新架构训练。
- 核对optimizer参数ID与全部trainable参数集合相等，报告真实active Adam状态。合成due-phase样本应让新向量出现梯度及实际更新；纯非due样本的新向量梯度应为0。增加参数可能改变全局梯度裁剪，不能称训练动态也与控制相同。
- [forward/evaluate_actions/get_distribution](../src/agents/timing_policy.py#L80)必须走同一新logit路径，否则采样与PPO likelihood会不一致。Critic功能不改；仍保存原macro token/logP/value，不把派生股票请求当神经概率。

### 持久化与对照

[TimingAgent](../src/agents/timing_agent.py#L78)严格要求原policy类、线性结构和参数合同，新候选不能直接冒用其身份。未来需要显式白名单中的新policy/agent及合法PPO加载路径，不从metadata任意动态import。保存并严格校验新类名、129维特征顺序、109维残差定义、12,914参数、零初始化版本、state_dict、数据/执行契约与真实算法；默认加载只推理，保存加载需逐tensor及logP/动作一致。

最小正式对照为原12,805参数Actor与due残差12,914参数Actor，同seed、原共同初始θ、数据、奖励、GAE、RL日步和账户排程。原模型必须本轮重新训练，旧样本只作冷读复现。仅检验这一架构因素，不同时改margin、超参数或加入配对标签；仍按完整封口与固定终点判断收益增量。

风险：109参数仍只是局部线性交互；稀少phase0数据可能不足，context缩放或高相关性会影响优化；更易跳过调仓不等于更会赚钱，额外参数和active梯度也不等于收益改进。

## 方向二：同状态配对经济Q标签

### 标签定义

仅在训练期真实状态 `s_t`，从同一完整状态创建两个独立分支。第一步分别HOLD和REBALANCE，随后都执行固定的原phase20日M20延续。候选H为20或60个交易日，正式采用前须固定H，或将H本身作为单独因素登记。

```text
Q_rule,H(s_t, a) = log(NAV_branch[t+H] / NAV_branch[t])
DeltaQ_H(s_t) = Q_rule,H(s_t, REBALANCE) - Q_rule,H(s_t, HOLD)
             = log(NAV_REBALANCE[t+H] / NAV_HOLD[t+H])
```

两分支初始NAV相同，收益已扣各自实际成本，不二次扣费。正DeltaQ支持该状态下第一步调仓，负值支持HOLD。后续从 `phase_t+1` 继续原日历，不以第一步动作重启20日计时；[EconomicRule](../src/training/exploration_env.py#L163)的排名、代码tie-break、rank-dependent数量和同mapper/broker保持原样。

这是**固定规则延续、指定历史路径与有限H**的动作价值差，不是未来市场分布下的期望保证，也不是当前学习策略的 `Q_pi`。本候选的净log目标是未折扣H期净值比，与PPO的gamma0.995折扣目标也不完全相同，潜在目标冲突必须披露，不能偷换概念。

### 必须复制完整状态

不能从trace汇总仓位/现金比例重建账户，更不能假定两分支都从空仓开始。至少保留：

- [Portfolio](../src/envs/portfolio.py#L18)的现金、实际positions、available_positions、cost_basis、current_date、累计费用、receivables、share_locks及事件现金/股数。
- [市场游标](../src/envs/market.py#L98)、环境 `_steps/_done`、历史/事件处理进度、Broker配置与状态、完整股票代码顺序、价格及特征/scaler身份、phase、当前Agent与独立基线book。
- [真实执行流程](../src/envs/trading.py#L145)中的次日开盘、可卖量解锁、公司行为、冻结/名额/整手/参与率/资金和费用处理，不能用简化价格乘权重代替。

先用完整deepcopy或经审计的完整快照还原，核验分支开始的所有账户/时钟状态及订单能力精确一致，两个分支无共享可变Portfolio/Broker/游标。[现双book初始化](../src/training/timing_env.py#L312)已示范复制生成后的实际Portfolio，而非只复制初始权重。

市场数组体积很大。后续可以在严格验证后共享只读行情数组、分离可变游标，但不能先以性能优化为由省略状态。禁止克隆PPO模型来额外预测未来动作；延续规则不需要神经前向。

### 时间与数据边界

决策特征只能来自 `t` 收盘及之前；标签未来价格可以用于训练监督，但只存在独立标签表，不能加入Actor观察、标准化拟合或推理输入。每条标签保存决策日、实际终点、H、完整起点state摘要/哈希、phase、两终点NAV/费用/订单、规则版本、源数据/scaler/Broker/工具SHA。

必须满足 `label_endpoint <= train_end`，并由物理裁剪的训练provider保证，不能跨入val/test/保护期。H按真实交易日期数组推进，不按20/60个自然日。val只允许按新协议作模型评价，不能生成标签、调label温度/阈值或更新scaler。

人工240步TimeLimit不能使标签分支reset后继续凑足H。最小方案只选原环境剩余可合法推进天数足够H的状态；若另用label专用经济延续环境，必须预先登记其horizon语义、保留全部账户/事件/phase时钟并独立验收。自然末日、非正NAV或不完整状态不能静默缩短H冒作完整标签。

### 监督或Q辅助如何接PPO

可检验两个不同用途，下一协议只能明确选择，不先混在一起：

- 监督预热：用DeltaQ及预先固定的温度/阈值训练Actor两logit差或单独优势头，得到 `theta_aux`，之后按同环境继续PPO。
- 联合Q辅助：在PPO更新中增加单独DeltaQ头/损失。它必须与原Critic的 `V_pi` 区分，不能把动作Q差直接当作Critic状态价值回报目标。新增头、参数和损失系数在PPO-only控制中也要同架构初始化，所有优化器参数显式登记。

需要保留三个固定审计端点：基线初始化 `theta_0`、监督后 `theta_aux`、最终 `theta_PPO`，并有同初始θ/同RL预算的PPO-only对照。

```text
supervised_gain = ROI(theta_aux) - ROI(theta_0)
PPO_increment = ROI(theta_PPO) - ROI(theta_aux)
total_gain = ROI(theta_PPO) - ROI(theta_0)
```

端点只用于归因，不从中挑最高收益checkpoint。若总增益只来自监督而PPO增量为0或负，不能把监督/规则收益归因于RL。监督预热转PPO时，建议从全量参数构建新PPO optimizer，清空监督Adam动量以隔离初始化因素；若保留状态，它就是另一个必须登记的因素。PPO-only控制同样报告真实优化器与active状态。

### 计算量与风险

标签生成是额外历史模拟，不算PPO经历。M个状态、每个两分支H步，新增Agent分支步骤为 `2*M*H`；若每分支保留独立基线book，基线另 `2*M*H`，合计 `4*M*H`账本步骤。状态采集的金融步骤还要单列，不能把同一日期重复模拟称为新原始数据。

监督optimizer steps、含辅助损失的joint Adam steps、真正PPO optimizer steps、真实RL日步、标签曝光次数和所有金融/基线/评价步骤分别计量。若辅助并入同一PPO Adam，只报告其中含aux的次数，不虚构额外优化器；若另有监督optimizer，确实单列。额外算力预算须与PPO-only公平披露。

风险包括重叠H窗口与同日多账户高度相关、反复曝光同一价格路径、初始组合分布和phase覆盖偏差、固定延续Q与实际PPO目标不一致、净log与折扣目标冲突，以及零/微小DeltaQ。REBALANCE因冻结、名额或费用门槛而没有成交时，两分支可能经济等价；不能强行给BUY标签或以浮点噪声选方向。阈值、低信号权重、截断/标准化和去重规则只能用训练数据预先固定，不能为评价收益事后调整。

## 采用前的共同条件

两方向目前只是假设，不修改当前Credit训练。下一次先完成代码/合成验收、同初始化与保存加载、真实费用/冻结/整手、标签日期防泄漏、完整预算与白名单合同，然后另立正式协议。选择哪条方向、H、辅助方法和参数，应等当前八trial完整封口结论后决定。

不能直接把两个候选同时合入一轮再声称识别了单一改进；也不以更少交易、较低Critic loss或开发折个别获利替代强基线与自身初始化增量。原HFQ经济代理、固定2026股票池、历史窗口曝光和真实未复权成交缺口仍在，不自动部署，不承诺下一轮收益。
