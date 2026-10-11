# 排序动作与辅助训练研究

日期：2026-10-11。**当前为工程准入阶段，正式训练尚未执行。本文记录已实现的研究设计、合成验收和待完成的结果项，不提前填写收益、候选或研究完成结论。** 用户选择每个正式 trial 最多 16,384 个 RL 步；本轮不自动扩展到 32K。

## 研究问题

上一轮 [纯行情摘要研究](ppo_market_summary_20261007.md) 没有找到跨两历史折稳定改善收益的新摘要。本轮保持 pure mean 行情摘要，转向两个更直接的瓶颈：交易请求的数量和排序共用同一随机坐标，以及只通过组合收益学习个股编码时的监督稀疏性。

上一轮 144 条训练模型验证路径中，BUY 名义金额仅 2.8707% 穿过映射，映射后非零成交率为 99.835%。这是旧研究的描述性执行诊断，不是本轮结果，也不证明减少过滤必然增加收益。新研究必须以真实执行后的扣成本收益、风险和初始化增量验证，而不是只提高请求利用率。

协议 [预注册](../ppo_priority_rank_protocol_20261011.md) 已以 `b8a4237` 先行提交。工程提交及正式 source/data/标签池冻结身份仍待登记。完成进度写在报告，不在正式冻结后回改协议。

## 二乘二对照

| cell | Actor | 训练方法 |
| --- | --- | --- |
| coupled_ppo | BUY priority 是 quantity 的拷贝原子 | PPO，辅助系数 0 |
| coupled_rank | 同上 | PPO + 五日相对收益 rank 辅助任务 |
| decoupled_ppo | BUY priority 是独立 Normal latent | PPO，辅助系数 0 |
| decoupled_rank | 同上 | PPO + 同一 rank 辅助任务 |

同 seed 四 cell 的初始参数张量一致。所有 cell 都声明 priority head、priority log_std 和 signal head；工程已核实总参数为 **125,322**，hidden256/128。priority head 初值复制 quantity head，priority log_std 复制 quantity log_std；signal head 权重和 bias 为零。闲置参数与实际 Adam 状态必须分别披露，总参数相等不等于有效容量相等。

同一 Actor 内，PPO/rank 两方法从相同随机策略函数开始；辅助任务在训练更新中才产生差异。跨 Actor 的确定性初始 quantity、排序、成交及 Critic 输出相同，但 decoupled 增加独立随机坐标，**两者初始随机执行函数不等价**。分别保留初始化控制，不能把初始探索差异全部称为 PPO 学习增益。

保持纯行情/账户分离：market encoder 只读行情特征；市场摘要是有效个股行情编码的均值，不读持仓、现金或交易权限。账户信息仍通过 account encoder、个股融合、held context 和约束影响 Actor/Critic。本轮没有修改摘要为 attention/dispersion，也没有新增股票行情。

## 动作与概率

训练、推理和原生轨迹统一保存 **3N 个原始动作**：

```text
[N 个 direction, N 个 conditional quantity latent, N 个 BUY priority latent]

HOLD   : 不主动改变股数；quantity/priority 为零
BUY    : quantity 决定整手数量；priority 决定资金和名额竞争顺序
REDUCE : quantity 决定合法区间内减仓；priority 为零
EXIT   : 申请全部可售股份退出；quantity/priority 为零
```

coupled 的 BUY priority 必须与原始 quantity 完全相同，是确定性拷贝原子，不重复计入 Normal 密度；decoupled 的 BUY priority 单独采样，其余股票 priority 都是零原子。quantity 沿用旧支持：BUY 一直保留 quantity 坐标，只有 varying-bound REDUCE 保留数量随机性，HOLD/EXIT/fixed REDUCE 为零原子。没有同时加入固定 BUY 坐标删减的另一种干预。

```text
logP_coupled = sum(logP_direction + active_quantity * logP_quantity)
logP_decoupled = logP_coupled + sum(is_BUY * logP_priority)
```

entropy 按方向概率对条件 Normal entropy 加权；coupled copy/zero atom 贡献零。PPO ratio、clip 和 KL 使用完整 joint logP，不除以 N，不使用实际成交股数或裁剪后的 latent 替代采样请求。KL 是请求分布的 old/new 差异，不是持仓变化量。

新 mapper 复用旧合法方向、整手区间和原始股份检查，仅让 BUY 排序读取独立 priority。现金/名额过滤、卖出、冻结、部分可售股份、T+1、完整 EXIT 及残余现金仍保留，Broker 在实际模拟执行日独立复核。训练采样动作可能被 SB3 Box 裁剪后进入环境，但 buffer 必须保存原始未裁剪动作和对应 old logP。确定性推理使用同一裁剪和 tie-break 规则。

**这不是完整的稀疏、无放回、预算感知 selector。** 仍然先生成每股请求，再由 mapper 处理 topN/现金；本轮只检验 quantity/priority 解耦。未实现 autoregressive stock-or-STOP、全局交易 gate 或独立现金动作，不能将过滤后的稀疏成交倒称为已训练的稀疏选择策略。

## 辅助训练

`RankAuxPPO` 以本机 Stable-Baselines3 2.7.1 的 PPO train 逻辑为基础，在**同一次 Adam 更新**中合并目标：

```text
loss = clipped_PPO + 0.5 * value_MSE + entropy_coefficient * entropy_loss
       + lambda_rank * Huber(signal_head(market_encoder(x)), rank_target)

lambda_rank = 0 或 0.05
Huber delta = 0.1
每次被接受的更新抽取 1,024 个训练 stock-date 样本
```

监督只更新 market encoder 和 signal head，不读账户，不把未来标签加进 policy observation，也不直接转换为买卖先验。signal head 的零初始化意味着首个辅助更新中 encoder 的辅助路径可能为零；不能只凭存在辅助 loss 就声称从第一步获得非零 encoder 辅助梯度。

保留原 PPO 优势归一化、GAE、value target、joint ratio/clip、target KL 和全局 gradclip。KL 检查通过后才抽辅助 batch；跳过的更新没有辅助抽样或额外 Adam。辅助使用独立 NumPy RNG，不能消费 PPO shuffle、日期、账户或 K 随机流。

辅助任务会改变 encoder 梯度、全局裁剪和后续 Actor/Critic 数据分布，这是实际训练方法干预。同 RL 步数不等于同监督曝光或计算成本；原始/加权辅助 loss、梯度范数、实际 Adam 与 stock-date 抽样必须单独记录。不追加单独的监督 optimizer，不在训练结束后偷偷继续更新 encoder。

模型加载默认是 inference-only，禁止直接继续训练。显式恢复注册的 `RankAuxPPO` 类后，rank 模型仍必须重新挂载与保存身份一致的 train-only 标签池，缺池时在产生任何 rollout 前拒绝 learn。archive、metadata、config 和 contract 的算法/方法身份及 Actor 变体必须一致，不能把 rank 模型当普通 PPO 静默续训。

## 数据与标签

仅复用封存新浪 300 股、20 特征、lookback1、2023/2024 两折；每折训练到当年 3 月 31 日，验证 Q2。旧训练 scaler 不重拟合，各 cell 使用共同五日特征完整性 mask。两折已暴露，不是新盲测。不下载，不重放旧 2025 绩效，不评估 2026-07-01 后保护期。

构造标签之前先物理截断 features/prices 到该折训练边界：

```text
gross_forward_return = close[t+5] / open[t+1] - 1
rank_target = 同决策日有效股票集合的 average-tie rank，映射到 [-1,1]
```

五日是本地 union-calendar 的五行，要求全部中间 open/close 为完整、正、有限报价，不跳过缺失报价。决策日须满足过去共同五日特征有效；最后五个训练决策日 purge，标签 endpoint 不越训练边界。单日不足两个有效标签股票不构造 rank 样本。登记决策/标签端点、stock/date 索引、特征顺序、原输入 SHA 和池逻辑身份。

rank 预测的是 gross 相对收益，金融 reward 仍是每日扣成本净收益；低辅助 loss、预测 IC 或更多标签曝光不能替代交易收益证明。真实未复权成交、公司行为及历史交易状态资料的缺口仍在，本轮研究价格与冻结股票名单不构成现实实盘验收。

## 固定预算

| 项目 | 已预注册安排；尚非完成量 |
| --- | ---: |
| folds / seeds | 2023、2024 / 42、123、2026 |
| 正式 trial | 4 cell × 2 fold × 3 seed = 24 |
| short / long | 8,192 / 16,384，共享训练前缀 |
| 新金融 RL 总量 | 393,216 步 |
| 快照 | 48 份 |
| 快照步数和 | 589,824；其中 196,608 是已含的 short 前缀 |
| episode / 512 步 chunk | 预期 1,536 / 768，实际待记录 |
| Adam | 无 early KL 时 long 总 15,360；short 7,680 已含 |
| rank 监督曝光 | 无 early KL 时 12 rank trial 共 7,864,320 次抽样 |
| 训练模型 / 初始化 / 规则路径 | 192 / 48 / 24，共 264 |
| 验证步数 | 若每路径 59 步则 15,576，最终记录实际值 |

训练初始账户 mixed cash/equal/concentrated/random，256 步 episode、recent_every4；训练 K10/20/40，主评估 K20。初始化控制按 fold/actor/seed 复用两方法，共 12 组四账户；不跨 Actor 复用随机函数声明。

CUDA、torch_threads1，actor gain0.1，quantity/priority initial log_std=-0.7。共同 LR=3e-5，n_steps512、batch128、epochs5、gamma0.99、GAE0.95、clip0.2、targetKL0.03、ent0、vf0.5、gradclip0.5、常数学习率。LR 与旧研究不同，旧研究不能充当新矩阵严格配对基线。

资本100万元、佣金0.0003/最低5元、卖税0.0005、滑点0.001、量参与率0.01、lot100、单股及 BUY/REDUCE 单步名义额5%、max_commission_ratio0.001。成本在净值中扣一次，没有额外微单罚项；回撤10%仅是候选验收门槛，不是本轮 reward 或 Critic 中新增的风险惩罚。

## 工程验收与硬件

当前新增动作/模型/加载测试 **100 项**，辅助训练测试 **54 项**，联合实际 **154 passed，25.85 秒，exit 0**。root 集成测试另实际 **4 passed，27.98 秒**。最终全仓离线验收已实际完成：**1,293 passed、13 deselected、11 warnings，442.59 秒**；13 项联网测试未执行。针对性和集成验收不是新增独立测试样本，不与全仓通过数相加。

以上都是工程及合成数据验收，不计入金融训练预算，不证明策略收益改善。正式 prepare 登记与金融训练尚未执行，实际 RL/Adam/辅助曝光和收益结果仍待记录。

覆盖 3N 原子与 joint logP/entropy、原始未裁剪动作、无效/冻结与部分 T+1、整手/费用/完整 EXIT、N=1/7/300、topN/cash 及股票代码 tie-break、coupled mapper 对旧映射逐字段精确正例、初始公共参数/确定性函数、保存加载和方法身份负例。辅助验收包含未来扰动不变性、五日 purge、独立 RNG、零辅助系数对原 PPO 参数/Adam 精确重放，以及合成 tiny 联合更新。

合成硬件 benchmark 的 batch128 更新测得：CPU4 线程 **371.83 ms**，CUDA **20.309 ms**，该内核 GPU 约快 **18.3 倍**。它不使用金融数据，不是金融 pilot，也不保证环境推进、账本、轨迹写盘和评估的整体墙钟同样加速。本轮按协议使用 CUDA/torch_threads1，并记录实际运行耗时及设备。

## 评估规则

先平均同 seed 四账户，再平均同 seed 两折，最后对三个 seed 求均值和样本 SD。分别报告每折/每 seed、short/long、初始化增量、动作和方法主效应、四个条件效应及交互；六个 fold/seed、48 个快照不能伪装成独立种子。

同执行机制保留 HOLD、每日动量、20 日动量规则。各 cell/预算须在两折分别通过：最坏12条路径 DD≤10%、自身初始化增量均值>0、至少2/3 seed 增量为正、平均净 ROI 超过20日动量。无合格配置时候选为 `null`；正式结果出来前既不声明合格，也不预判必然失败。代表模型固定 fold2024/seed42，不挑幸运种子，不发布默认模型或真实订单。

执行诊断另列 active quantity/priority、joint KL/clip fraction/entropy、请求/映射/成交、size 分位筛选、quantity-priority 相关、采样训练与确定性验证差异。改进排序利用率并不自动改善净收益；更多训练步也不是改进证明。

## 进度与待填项

- 已提交预注册；3N 动作、严格加载与辅助训练工程已实现，针对性、root 集成及全仓离线验收均已通过。
- 待正式 prepare；source/data/协议/标签池身份冻结后再执行全部24 trial，每 trial 长预算16,384步，总金融RL393,216步。
- 正式训练尚未执行，实际 RL/Adam/辅助曝光、48快照、原生曲线、收益/风险/初始化对照和候选均未填写。
- 待正式产物封存后生成模型索引、对照图、完整汇总和独立审计，再更新摘要与本报告。

本轮没有实现真实 autoregressive 稀疏选择、风险 reward、BC 预训练、GRU 或条件 attention，也没有新增下载数据。即使研究完成，未见时期、未见股票与真实成交/公司行为账本仍须补齐，不能将工程通过当成每日实盘配置目标达成。

[报告摘要](ppo_priority_rank_20261011_summary.md) · [冻结协议](../ppo_priority_rank_protocol_20261011.md)
