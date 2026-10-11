# 排序动作与辅助训练研究

日期：2026-10-11。**本轮研究已完成：24次训练、48快照、实际金融RL393,216步；八配置未通过收益/风险门槛，候选null，不替换默认模型。** 最终全仓回归、只读工具及独立最终审计均通过。每trial最长16,384步，未追加32K，研究完成不表示每日实盘配置目标达成。

## 主要结论

1. **排序/数量解耦提供了短预算增益，但未获得合格模型。** 跨折paired三seed的动作主效应在8,192步为+2.179137pp，16,384步为+0.296290pp；PPO方法下分别+2.256539/+0.552637pp。小样本及跨Actor随机探索差异仍限制归因。
2. **加长训练不稳定增收。** decoupled PPO从短预算+1.190440%降至长预算-0.608901%，long-minus-short为-1.799341pp，三个paired seed均下降；不能以更多步数直接替代方法改进。
3. **辅助rank效果依赖预算和年份。** 方法主效应短预算-0.231141pp、长预算+0.317047pp，但2023/2024长预算效应方向相反，不是稳定跨年收益改善。
4. **观察最高的配置不合格。** decoupled PPO短预算跨折ROI为+1.190440%±3.271436pp，2023最坏DD12.071505%；固定2024/seed42代表ROI为-3.335610%，不能挑平均值或幸运seed晋升。
5. **八配置全部未通过。** 2023每配置最坏DD超过10%且净ROI低于同机制20日动量；2024每配置平均学习增量为负。工程和账本一致性不能替代收益/风险门槛。

## 研究问题

上一轮 [纯行情摘要研究](ppo_market_summary_20261007.md) 没有找到跨两历史折稳定改善收益的新摘要。本轮保持 pure mean 行情摘要，转向两个更直接的瓶颈：交易请求的数量和排序共用同一随机坐标，以及只通过组合收益学习个股编码时的监督稀疏性。

上一轮 144 条训练模型验证路径中，BUY 名义金额仅 2.8707% 穿过映射，映射后非零成交率为 99.835%。这是旧研究的描述性执行诊断，不是本轮结果，也不证明减少过滤必然增加收益。新研究必须以真实执行后的扣成本收益、风险和初始化增量验证，而不是只提高请求利用率。

协议 [预注册](../ppo_priority_rank_protocol_20261011.md) 已以 `b8a4237` 先行提交；工程基线为 `9a52851`。prepare exit0，注册Git为`9a52851cd52120f5abdedcd02c82416c5ec79f72`、dirty=true，冻结 **91份`src` Python源码，原85份未改**，不计tests/tools。24initial模型Git为9a52851×2/c7998904×4/09c65fe2×18；48trained为c7998904×8/09c65fe2×40，共72模型。全部source bytes同注册；不同进度文档提交造成Git身份不同，不声称所有模型来自9a或工作树全净。

[机器登记](../../reports/runs/ppo_priority_rank_20261011_16k/protocol.json) 已固定两折原输入、标签池、预算和源码 SHA；冻结协议文档 SHA256 为 `c0557670ab546f686a115b584be01d55bb09f2e87dfe699d064f049b204f81d1`。完成进度写在报告，不回改已冻结协议或源码。

## 二乘二对照

| cell | Actor | 训练方法 |
| --- | --- | --- |
| coupled_ppo | BUY priority 是 quantity 的拷贝原子 | PPO，辅助系数 0 |
| coupled_rank_aux | 同上 | PPO + 五日相对收益 rank 辅助任务 |
| decoupled_ppo | BUY priority 是独立 Normal latent | PPO，辅助系数 0 |
| decoupled_rank_aux | 同上 | PPO + 同一 rank 辅助任务 |

同seed四cell初始参数张量一致，总参数均 **125,322**，hidden256/128。priority head初值复制quantity head，priority log_std复制quantity log_std；signal head权重和bias为零。实际long快照Adam tensor状态数为coupled PPO17、coupled aux19、decoupled PPO20、decoupled aux22；分别对应125,063/125,192/125,193/125,322个纳入有效路径的参数，总参数相等不等于有效容量相等。

同一 Actor 内，PPO/rank 两方法从相同随机策略函数开始；辅助任务在训练更新中才产生差异。跨 Actor 的确定性初始 quantity、排序、成交及 Critic 输出相同，但 decoupled 增加独立随机坐标，**两者初始随机执行函数不等价**。分别保留初始化控制，不能把初始探索差异全部称为 PPO 学习增益。

保持纯行情/账户分离：market encoder只读行情；市场摘要是有效个股行情编码均值，不读持仓/现金/权限；账户仍通过account encoder、融合、held context和约束影响Actor/Critic。Critic结构和损失口径一致，但**每trial独立训练自己的Actor和Critic**，只是模型内共享编码，不是四cell跨trial共用同一Critic参数。未改成attention/dispersion或下载新行情。

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

**辅助表示限制：** signal head 只读取该股票的 market encoder，不读取同日市场摘要或其他股票特征。虽然 target 是同日未来收益截面 rank，网络学到的是给定单股特征的条件期望排名，并未显式建模该股票在当前完整截面的动态相对位置。不能把rank监督称为已实现完整截面交互；真正相对特征或市场摘要条件辅助head应另轮独立对照，不中途修改本轮输入或源码。

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

正式 prepare 已保存并登记以下 train-only 标签池：

| fold | stock-date 样本 | 有标签决策日 | 有标签股票 / 注册池 | 最早决策日 | 最后决策日 | 最大标签端点 | train_end |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 2023 | 354,930 | 1,449 | 292 / 300 | 2017-04-12 | 2023-03-24 | 2023-03-31 | 2023-03-31 |
| 2024 | 425,912 | 1,690 | 297 / 300 | 2017-04-12 | 2024-03-22 | 2024-03-29 | 2024-03-31 |

两个 manifest 都声明最后五行 purge、common lookback5、horizon5、validation_labels_used=false。2024年3月31日为周末，标签端点止于实际3月29日，不虚构报价。两池历史区间重叠，不能把跨折样本和称为新增独立行情或金融 RL 经历。

池逻辑身份分别为2023 `fd25be314de3c25dc11e53c6ee434e6b608f51fd6877d2dfc62ac0fee4e478b5`、2024 `cb75ee4f54d253908a21dfd5865a66fc305587ede1d9f9f6aaca54a5d1d576e8`；manifest与输入/数组SHA写入机器登记。见[2023标签登记](../../reports/runs/ppo_priority_rank_20261011_16k/labels/2023/manifest.json)和[2024标签登记](../../reports/runs/ppo_priority_rank_20261011_16k/labels/2024/manifest.json)。实际12rank run共7,680次辅助联合更新、7,864,320次stock-date抽样，均由long快照audit累计，不是新增行情或额外RL步。

## 实际预算

| 项目 | 实际完成量 |
| --- | ---: |
| folds / seeds | 2023、2024 / 42、123、2026 |
| 正式 trial | 4 cell × 2 fold × 3 seed = 24 |
| short / long | 8,192 / 16,384，共享训练前缀 |
| 新金融 RL 总量 | 393,216 步 |
| 快照 | 48 份 |
| 快照步数和 | 589,824；其中 196,608 是已含的 short 前缀 |
| episode / 512 步 chunk | 1,536 / 768 |
| joint Adam | 15,360；short 7,680已含，不重复相加 |
| 含aux的joint更新 / rank监督曝光 | 7,680 / 7,864,320次stock-date抽样；额外监督optimizer为0 |
| 训练模型 / 初始化 / 规则路径 | 192 / 48 / 24，共 264 |
| 验证步数 | 264×59=15,576 |

训练初始账户 mixed cash/equal/concentrated/random，256 步 episode、recent_every4；训练 K10/20/40，主评估 K20。初始化控制按 fold/actor/seed 复用两方法，共 12 组四账户；不跨 Actor 复用随机函数声明。

全部24个long记录每trial64episode、cash/equal/concentrated/random各16，unique initial accounts为44至47、unique initial states为57至62，不把1,536片段称为1,536个独立新行情或初始账户。初始mode由独立`mode_rng`随机four-mode bag调度，**不存在recent必然绑定random或单一mode**；总体账户模式均衡，但时间与账户没有分层均匀交叉。

| seed | scheduled_recent片段 | cash | equal | concentrated | random | 实际latest起点片段 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 16 | 2 | 3 | 6 | 5 | 17 |
| 123 | 16 | 1 | 3 | 6 | 6 | 16 |
| 2026 | 16 | 3 | 6 | 4 | 3 | 16 |

同seed的两fold/方法/Actor账户调度一致。seed42有一个uniform时间起点碰巧等于latest且mode=random，所以actual-latest为17，不算第17次scheduled_recent。recent cash每seed仅1至3片段，近期时间×账户覆盖有限；总体各mode16不代表每时间子集均衡。分层时间×账户调度留待后续独立研究，本轮未改源或增步。

CUDA、torch_threads1，actor gain0.1，quantity/priority initial log_std=-0.7。共同 LR=3e-5，n_steps512、batch128、epochs5、gamma0.99、GAE0.95、clip0.2、targetKL0.03、ent0、vf0.5、gradclip0.5、常数学习率。LR 与旧研究不同，旧研究不能充当新矩阵严格配对基线。

资本100万元、佣金0.0003/最低5元、卖税0.0005、滑点0.001、量参与率0.01、lot100、单股及 BUY/REDUCE 单步名义额5%、max_commission_ratio0.001。成本在净值中扣一次，没有额外微单罚项；回撤10%仅是候选验收门槛，不是本轮 reward 或 Critic 中新增的风险惩罚。

**经济规则验收边界：** 费用/lot账本通过指内部冻结的commission/minimum5、sellTax0.0005、slippage0.001及100股整手规则数学执行一致，不表示逐历史日期税费制度或不同板块最小交易单位都已真实复原，不能称实盘成交验收。此限制与HFQ经济代理、固定2026股票池及开发窗口反复曝光并列保留，不追加下载或临时改变本轮经济规则。

## 工程验收与硬件

新增动作/模型/加载100项、辅助54项，针对性联合实际154passed/25.85秒/exit0；root集成实际4passed/27.98秒。训练前全仓准入为1,293passed、13deselected、11warnings、442.59秒；正式封存后的**最终离线回归为1,293passed、13deselected、11warnings、389.43秒、exit0**。13项联网测试未执行，各次重验不累计为不同测试数量。

工程/合成验收不计入金融预算。两worker、全量cached汇总及最终回归真实exit0；独立最终审计actualexit0、status=passed，核24native trial、标签/规则、1,104archives（1,032NPZ+72ZIP）和70,728CRC member。审计签3,140个study文件、91src及外部原fold/旧85/协议MD/tools，candidate替代0，额外NN/env/grad/Adam0；人工最终MD不纳签，冻结预注册MD的c055 SHA纳签，避免引用audit SHA形成循环。审计完整性通过不是收益合格证明。

[独立最终审计](../../reports/runs/ppo_priority_rank_20261011_16k/final_audit.json) SHA256：`89bc5f465eada6a845812b9f538886fc510dfcaecd392e54100b3f1f93379ce2`。

覆盖 3N 原子与 joint logP/entropy、原始未裁剪动作、无效/冻结与部分 T+1、整手/费用/完整 EXIT、N=1/7/300、topN/cash 及股票代码 tie-break、coupled mapper 对旧映射逐字段精确正例、初始公共参数/确定性函数、保存加载和方法身份负例。辅助验收包含未来扰动不变性、五日 purge、独立 RNG、零辅助系数对原 PPO 参数/Adam 精确重放，以及合成 tiny 联合更新。

合成batch128更新CPU4线程371.83ms/CUDA20.309ms，GPU内核约快18.3倍；非金融pilot，不保证整体同幅加速。24正式long audit均actual_device=cuda；每trial elapsed1,232.12至1,376.70秒，累计31,446.26秒。`long_audit.elapsed_seconds`含训练/保存/诊断，**不含后续金融eval，也不是两worker并行总墙钟**。

| cell（各6trial） | 平均elapsed秒 | 范围秒 | 平均FPS | peak allocated MiB | reserved MiB |
| --- | ---: | --- | ---: | --- | --- |
| coupled PPO | 1,269.25 | 1,245.31–1,307.88 | 12.91 | 287.07–288.23 | 402–404 |
| coupled aux | 1,306.44 | 1,284.71–1,324.57 | 12.54 | 288.38 | 404 |
| decoupled PPO | 1,309.35 | 1,232.12–1,358.48 | 12.53 | 269.49 | 344 |
| decoupled aux | 1,356.00 | 1,329.44–1,376.70 | 12.08 | 269.64 | 346 |

每trial PyTorch CUDA peak allocated269.49至288.38MiB、reserved344至404MiB；这不是整卡或两进程合计峰值。只读单点nvidia-smi记录RTX3050Laptop/driver566.36/4,096MiB、整卡已用1,002MiB、util10%、56C，仅为时点检查，含系统/其他进程，不能称研究峰值/均值。没有正式同预算CPU金融对照，因此不把合成18.3倍当整个训练实际加速倍数。

## 轨迹完整性事故

**当前状态：两worker真实exit0后，旧登记/receipt已透明更正并重新seal；canonical NPZ写入0，根因仍未知。** 三轮cached/NO_BUFFERING只读均为原SHA/有效CRC，不称人工恢复数据或根因定位。

`2023_coupled_ppo_42` 的首个 training chunk `steps_000001_000512.npz` 出现压缩内容单 bit 损坏，67 个 ZIP member 中只有 `opening_valuation_prices.npy` CRC 失败。损坏文件 SHA256 为 `58b2299be29129525bc9155a4763bfe1f495802377152fdea0493ac2d8dec16f`；训练时 index、short/long 快照及两条 row-prefix 记录一致要求原 SHA256 `c88750f4a6f30fd93505b26d6a9a93010a0dc0710936a23a722aa0c9a0e1f5df`。

runtime 构造非权威恢复候选，仅将文件 offset5,221,314 的一个字节由 `0x75` 改为 `0xF5`（xor `0x80`，1 bit）。候选完整 SHA **精确恢复上述训练时原 SHA**，不是修改索引 hash 来接受不同 payload。独立审计核验全部512×300价格与冻结 prices parquet及CRC有效同轨迹字段一致、67 member CRC全部通过、5个原登记锚一致，已批准原字节恢复。

扩大健康扫描检查161个稳定NPZ（112已封口、49未封口），仅发现这一文件异常；未封口文件的健康扫描不能算正式完成审计。损坏原因仍未确定，不归咎GPU、存储或其他硬件，也不将确切原字节重建等同于根因定位。

候选、证明和独立审批均保存在 [forensics重建证明](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/2023_first_chunk_recovery_proof.json) 与 [独立审批](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/independent_recovery_approval.json)。先前原字节恢复计划未由root/runtime/review执行，不把候选审批称为人工恢复完成。已确认当时**没有raw坏文件现场capture**，不能宣称保留了损坏原件；实际原件是当时读取记录、证明和旧封口凭据。

后续只读复核出现新的状态：此前稳定读取损坏SHA `58b229...` 且该member CRC失败，随后同canonical路径重新读得完整原SHA `c88750...e1f5df`、offset5,221,314为`0xF5`，inode/mtime未变。runtime连续6次buffered读取均得到原SHA且67 CRC有效；候选有独立inode、nlink=1，不是与canonical互相影响的硬链接；helper确认只写forensics候选和proof。root/runtime/review均未执行canonical改写，因此**状态来源仍undetermined**，不能将其归因于已实施的人工恢复，也不能声称已定位硬件或软件原因。

正式只读`NO_BUFFERING`诊断已实际exit0，三轮cached前/后与uncached读取均精确为原SHA `c88750...e1f5df`，67 member CRC有效，原file ID/size/mtime不变。没有canonical写、flush、系统修改或裸设备访问，见[只读稳定性观察](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/canonical_uncached_observations.json)。用户已明确回复“没有处理过”该原NPZ；root/runtime/review和用户均报告未手动改原文件。当前cached/cold读取稳定，但变化根因仍unknown，不作用户或硬件归因，也不是人工恢复证明。

runtime已由原candidate将同一offset字节反向`0xF5→0x75`，生成[事后重建观察坏variant记录](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/reconstructed_observed_corrupt_variant.json)及NPZ；完整SHA精确为此前观察的`58b229...`，同member CRC失败。**这是事后重建已观测variant，不是当时raw现场保留。** root独立核验canonical/candidate均原`c88750...`、重建坏variant为`58b229...`；这些事实不确定先前读取变化的原因或时间。

[透明reseal凭据](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/incident_lineage_receipt.json)已实际exit0，SHA为`c311ce67f6d3bdf3ff2d0eea2247f375390d6d199cfeb1ebd7ece88116d1fb72`。仅更正completed registry该leaf的旧58→原c887，以及worker_2023对该completed的hash；原completed/workerreceipt bytes已备份，模型/曲线/其他JSON不变，canonical NPZ写0。

reseal凭据的4个worker路径标签因helper局部变量遮蔽误指最后NPZ，原hash实际来自正确worker JSON。已追加[路径标签勘误](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/receipt_path_correction.json)，SHA `8b47f70114ce003a9424511f343180c006eed43efc43c18fcded05b0e6e948b4`，独审exit0通过；原c311凭据及已执行23ccb脚本完整保留，不回写原证据。这里只修标签，不再次改变登记hash或训练产物，无额外RL/推理/梯度/optimizer，预算/source/protocol不变。

## 评估规则

先平均同 seed 四账户，再平均同 seed 两折，最后对三个 seed 求均值和样本 SD。分别报告每折/每 seed、short/long、初始化增量、动作和方法主效应、四个条件效应及交互；六个 fold/seed、48 个快照不能伪装成独立种子。

同执行机制保留HOLD、每日动量、20日动量规则。每cell/预算在两折分别须满足：最坏12路径DD≤10%、自身初始化增量均值>0、至少2/3seed增量正、平均净ROI超过20日动量。全量结果候选为null；代表固定fold2024/seed42，不挑幸运种子，不发布默认模型或真实订单。

执行诊断另列 active quantity/priority、joint KL/clip fraction/entropy、请求/映射/成交、size 分位筛选、quantity-priority 相关、采样训练与确定性验证差异。改进排序利用率并不自动改善净收益；更多训练步也不是改进证明。

**logP审计覆盖限制：** 原生轨迹保存运行时old logP；独立审计检查逐step有限性、3N布局、copy/zero原子、合法支持、来源和prefix登记。未保存每个时点的全部旧theta，且审计禁止新增模型forward，因此不能宣称逐步旧神经密度已全部重算。合成概率/集成测试已验证源码的采样、evaluate_actions和原始likelihood路径，但源码路径验收与正式逐step神经密度数值复算是不同coverage，不互相冒充。

## 完整结果

以下来自完整24trial/48checkpoint的[机器汇总](../../reports/runs/ppo_priority_rank_20261011_16k/study_summary.json)，runner作者只读数学重验一致；作者复算与独立审计角色分开。事故旧receipt透明修订已完成，未追加模型评估。

### 八配置收益表

先求每seed/fold四账户均值；每折列三seed均值，跨折列同seed两折先平均后的三seed均值/样本SD，单位为%。初始化增量另列百分点。

| Actor | 方法 | 步数 | 2023净ROI | 2024净ROI | 跨折净ROI均值±SD | 自身初始化增量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| coupled | PPO | 8,192 | -0.338257 | -1.793941 | -1.066099±1.160042 | -0.425652 |
| coupled | PPO | 16,384 | +0.161418 | -2.484495 | -1.161539±1.895211 | -0.521092 |
| coupled | rank_aux | 8,192 | -0.789760 | -1.649917 | -1.219838±1.412212 | -0.579392 |
| coupled | rank_aux | 16,384 | +1.451153 | -2.627442 | -0.588144±1.840187 | +0.052302 |
| decoupled | PPO | 8,192 | +4.062268 | -1.681389 | +1.190440±3.271436 | +1.830886 |
| decoupled | PPO | 16,384 | +0.351271 | -1.569074 | -0.608901±1.643927 | +0.031545 |
| decoupled | rank_aux | 8,192 | +2.988993 | -1.225199 | +0.881897±2.742447 | +1.522343 |
| decoupled | rank_aux | 16,384 | +0.832417 | -1.928818 | -0.548201±1.505629 | +0.092246 |

共同跨折初始化ROI为-0.640447%。各fold/seed四cell确定性初始化ROI精确相同；这不消除跨Actor初始随机探索函数差异。最高观察均值不是候选：2023提升不能覆盖2024损失，global均值也不能代替逐fold风险和规则门槛。

![二乘二收益及配对效应](../../reports/runs/ppo_priority_rank_20261011_16k/comparison.png)

### 配对归因

记cp/ca/dp/da为coupled PPO/aux、decoupled PPO/aux，同seed配对计算。下表为跨fold三seed净ROI差；所有初始化效应严格0，因此learning gain效应与净效应一致。机器汇总保留逐fold、逐seed及全部三种分解；单位pp，n=3不声称显著性或普遍泛化。

| 效应 | 定义 | 8,192步均值±SD | 16,384步均值±SD |
| --- | --- | ---: | ---: |
| 动作主效应 | ((dp-cp)+(da-ca))/2 | +2.179137±1.864056 | +0.296290±0.439851 |
| 方法主效应 | ((ca-cp)+(da-dp))/2 | -0.231141±0.114459 | +0.317047±0.169494 |
| 交互 | (da-dp)-(ca-cp) | -0.154803±0.855180 | -0.512694±0.919976 |
| PPO下动作简单效应 | dp-cp | +2.256539±2.290784 | +0.552637±0.277400 |
| aux下动作简单效应 | da-ca | +2.101735±1.437840 | +0.039943±0.856251 |
| coupled下方法简单效应 | ca-cp | -0.153740±0.332716 | +0.573394±0.559891 |
| decoupled下方法简单效应 | da-dp | -0.308543±0.530253 | +0.060700±0.408847 |

`net effect = initialization effect + learning-gain effect`逐seed对齐；初始化效应0并非证明跨Actor随机探索无影响。short/long是同一轨迹共享前缀，不是独立训练样本。PPO下解耦增益在两个预算跨折三seed均正；aux下长预算只有1/3seed为正，不能忽略交互称两干预叠加必更好。

| 主效应 | 2023短 / 长 | 2024短 / 长 |
| --- | ---: | ---: |
| 动作 | +4.089639 / -0.214442 | +0.268635 / +0.807022 |
| 方法 | -0.762389 / +0.885441 | +0.300107 / -0.251346 |
| 交互 | -0.621773 / -0.808590 | +0.312167 / -0.216798 |

辅助方法长预算主效应2023正、2024负；短预算2024正效应只来自seed123，其余两个为0。不得用平均微增遮蔽不同年份和种子结果。

| cell | 2023 long-short | 2024 long-short | 跨折配对均值±SD |
| --- | ---: | ---: | ---: |
| coupled PPO | +0.499675 | -0.690555 | -0.095440±0.738578 |
| coupled aux | +2.240913 | -0.977525 | +0.631694±0.514612 |
| decoupled PPO | -3.710997 | +0.112315 | -1.799341±1.812623 |
| decoupled aux | -2.156576 | -0.703619 | -1.430098±1.695499 |

因此本轮不支持训练步数越多收益越好：coupled aux跨折加长三seed均改善，仍然净亏损且不合格；decoupled PPO三seed加长均下降。

### 门槛与代表模型

同机制20日动量四账户均值：2023 +5.541507%，2024 +1.208622%。以下单位DD为%、增量/规则差为pp；每行qualified均false。

| fold | cell | 预算 | 最坏DD | 自身初始化增量 | 正增量seed | 超过20日动量 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2023 | coupled PPO | short | 14.786304 | +0.066581 | 2/3 | -5.879765 |
| 2023 | coupled PPO | long | 12.744846 | +0.566256 | 2/3 | -5.380090 |
| 2023 | coupled aux | short | 14.784018 | -0.384921 | 2/3 | -6.331267 |
| 2023 | coupled aux | long | 10.845480 | +1.855992 | 2/3 | -4.090354 |
| 2023 | decoupled PPO | short | 12.071505 | +4.467107 | 3/3 | -1.479239 |
| 2023 | decoupled PPO | long | 10.354873 | +0.756110 | 2/3 | -5.190236 |
| 2023 | decoupled aux | short | 12.071505 | +3.393831 | 3/3 | -2.552515 |
| 2023 | decoupled aux | long | 10.294922 | +1.237255 | 2/3 | -4.709091 |
| 2024 | coupled PPO | short | 7.255014 | -0.917886 | 0/3 | -3.002562 |
| 2024 | coupled PPO | long | 9.172157 | -1.608441 | 0/3 | -3.693117 |
| 2024 | coupled aux | short | 7.255014 | -0.773863 | 0/3 | -2.858539 |
| 2024 | coupled aux | long | 9.563182 | -1.751388 | 0/3 | -3.836064 |
| 2024 | decoupled PPO | short | 8.324567 | -0.805335 | 0/3 | -2.890011 |
| 2024 | decoupled PPO | long | 9.453848 | -0.693020 | 1/3 | -2.777696 |
| 2024 | decoupled aux | short | 8.324567 | -0.349145 | 1/3 | -2.433821 |
| 2024 | decoupled aux | long | 7.802417 | -1.052764 | 1/3 | -3.137440 |

最高均值配置decoupled PPO short仍被2023风险/规则及2024学习增量/规则否决。候选null，不放宽门槛、替换默认模型或发布真实订单。

固定2024/seed42代表的四账户平均ROI如下；不是配置均值或最优seed。全部48路径/hash/ROI见[模型索引](../../reports/runs/ppo_priority_rank_20261011_16k/model_index.csv)，索引不构成部署推荐。

| cell | 8,192步ROI | 16,384步ROI |
| --- | ---: | ---: |
| coupled PPO | -2.143520% | -3.231918% |
| coupled aux | -2.143520% | -4.590423% |
| decoupled PPO | -3.335610% | -1.369205% |
| decoupled aux | -3.335610% | -0.900798% |

### 执行与资源

实际393,216RL/15,360joint Adam/1,536episode/768chunk已核；其中7,680个joint更新含aux，额外监督optimizer0。192模型路径11,328步、48初始化2,832步、24规则1,416步，共264路径15,576步。训练short前缀与long不重复累计。

[执行诊断](../../reports/runs/ppo_priority_rank_20261011_16k/execution_diagnostics.json)重读24完整long训练路径与192模型/48初始化/24规则路径；工具真实exit0、model loading/inference/env/gradient/optimizer新增均0。原生统计如下；每种路径分开，不混成策略收益或新市场经历。

| 轨迹 | 步数 | raw BUY请求 | 映射BUY | 非零BUY成交 |
| --- | ---: | ---: | ---: | ---: |
| 24训练 | 393,216 | 39,753,451 | 3,622,571 | 3,599,004 |
| 192训练模型验证 | 11,328 | 2,036,389 | 82,670 | 82,344 |
| 48初始化验证 | 2,832 | 436,198 | 12,894 | 12,862 |
| 24规则验证 | 1,416 | 2,799 | 2,799 | 2,771 |

模型验证raw BUY最终成交约4.04%，每决策平均179.77个请求、7.27个成交；映射后的成交率约99.61%。因此全局名额/现金投影仍主导，不能只看Broker后的高成交率称执行瓶颈已解决。

以下是训练8个fold×cell原生合计组范围，率以raw BUY为分母；数量rank bias是同决策raw quantity latent分位与按同决策选中数匹配的请求基线之差，单位0至1，**不是资金/持仓比例或跨日期盈利效应**。

| Actor | raw qty-prio相关 | 同日中心化相关 | 映射数量rank偏移 | topN过滤 | cash过滤 |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupled | 1（copy） | 1 | +0.3505至+0.3511 | 70.0%至70.7% | 20.9%至21.3% |
| decoupled | 0.00152至0.00157 | 0.00108至0.00116 | +0.00035至+0.00042 | 85.6%至86.1% | 4.3%至4.5% |

独立priority采样确实基本移除了**训练时由quantity兼任排序产生的数量上尾选择偏差**；名额/资金过滤结构随动作变化，这是机制证据，不是盈利因果证明。不能由降低cash过滤推导收益自动上升，也不放宽经济约束来制造成交。

**确定性部署仍高度相关：** decoupled验证同日quantity/priority相关0.9449至0.9706，映射数量rank偏移+0.1066至+0.3301。独立随机坐标不等于学到独立的确定性均值函数；两个head复制初始化、共享body，均值仍高度相似。不能声称推理已完全解耦，也不能仅由相关性断言利润来源。下一轮需区分采样噪声结构与学得排序函数，而非只继续加步。

### 后续研究

1. **优先研究真正稀疏、预算/slots感知的selector。** 现验证约96%的BUY请求未成交，独立priority只解开采样排序/数量噪声，没有取消全局投影。下一轮masked stock-or-STOP无放回选择必须进入真实条件joint logP并teacher-force重放，不只回测末端换排序。
2. **区分探索与确定性排序学习。** 新随机坐标确实减少训练数量选择偏差，但部署均值相关仍高；另立head结构/初始化或排序训练对照，保留自身初始化及确定性/采样评价，不将相关性变化直接称alpha。
3. **辅助head加入真实相对/市场条件时另做对照。** 本轮单股head的条件期望rank和弱长预算平均增益不足以证明完整截面理解。标签仍需train-only purge，不偷用Q2标签择超参。
4. **分层时间×账户覆盖和风险目标独立研究。** recent cash覆盖少；同时现reward无风险项，DD门槛不是学习惩罚。时间账户分层、显式风险状态/目标和真实价格/制度账本分开检验，不追改本轮门槛。

不基于同两个反复暴露窗口追加32K或盲目堆参数；下一轮预注册资源和对照，并补未见时期/股票验证。以上是待验证研究，不保证收益。

## 完成进度

- 已提交预注册和工程基线；针对性、root 集成及全仓离线验收均已通过。
- prepare、两worker、全量cached汇总exit0；24trial/48快照、393,216RL和全部计数已核实，候选null。
- 初始原生bit读取事故已透明更正登记/receipt并reseal，canonical写0、原凭据保留；4path标签追加勘误独审通过，根因未知，不称人工恢复。
- 比较图/48模型索引已只读生成、root视觉核验图非空清晰；最终全仓回归1,293pass/389.43秒通过。
- 只读图/index/执行诊断均exit0；原生/标签/规则和全部CRC核过，完整数学对账最大误差0；独立最终audit已actualexit0封存，SHA登记于本报告，不签最终人工MD以避免循环。

本轮没有实现真实 autoregressive 稀疏选择、风险 reward、BC 预训练、GRU 或条件 attention，也没有新增下载数据。即使研究完成，未见时期、未见股票与真实成交/公司行为账本仍须补齐，不能将工程通过当成每日实盘配置目标达成。

[报告摘要](ppo_priority_rank_20261011_summary.md) · [冻结协议](../ppo_priority_rank_protocol_20261011.md) · [机器汇总](../../reports/runs/ppo_priority_rank_20261011_16k/study_summary.json) · [192模型账户指标](../../reports/runs/ppo_priority_rank_20261011_16k/metrics.csv) · [48模型索引](../../reports/runs/ppo_priority_rank_20261011_16k/model_index.csv)
