# 排序动作与辅助训练研究预注册

日期：2026-10-11。用户选择长预算16,384步。本协议在正式训练前冻结，完成进度和结果另写报告，不回写本协议。

## 研究问题

上一轮纯行情摘要没有跨历史窗口稳定收益增益。执行诊断发现，大量独立股票请求被名额和现金过滤；BUY的同一个size还同时负责数量及排序，训练映射买单偏向随机size正尾。本轮检验：分开排序和数量，能否改善执行后的收益；训练期辅助收益任务，能否给个股行情编码提供更密集的学习信号。

固定纯行情均值摘要、共享Critic和原净值收益奖励。不会同时加入风险奖励、BC预训练、动量先验、GRU、多头注意力或新下载数据。本轮动作扩展是独立排序latent，不把末端topN过滤称为已经实现的稀疏选择。真正无放回、预算感知的逐股selector留作后续独立研究。

## 二乘二设计

| cell | 动作 | 方法 |
| --- | --- | --- |
| coupled_ppo | 数量与BUY排序共享latent | PPO |
| coupled_rank | 同上 | PPO + 五日相对收益rank辅助loss |
| decoupled_ppo | BUY排序为独立latent | PPO |
| decoupled_rank | 同上 | PPO + 同一辅助loss |

所有cell使用同一共享stock encoder、纯行情/账户分离网络及Critic；都构造独立priority head/log_std和signal head。priority head初始复制quantity head，priority log_std复制quantity log_std；signal head权重/bias为0。默认hidden256/128，预计总参数125,322，以工程准入核实固定。闲置参数和实际Adam状态另列，不将总参数相等称为有效容量相等。

同seed初始参数一致，各动作结构内部两训练方法的初始随机策略一致。两动作结构的确定性初始排序和数量也应一致，但decoupled增加独立随机坐标，**初始随机执行策略不等价**。方法比较和动作比较均须披露初始化控制；不把随机探索或初始化差异全部当成PPO学习收益。

## Agent动作

统一存储3N原始动作：

```text
[N个方向代码, N个条件quantity latent, N个BUY priority latent]

HOLD   : 不主动改变该股票股数
BUY    : quantity确定整手数量，priority参与名额/资金排序
REDUCE : quantity确定允许范围内的整手减仓，priority为0
EXIT   : 申请全部可售股份退出，quantity/priority为0
```

coupled中BUY的priority是quantity的拷贝原子，其余priority为0；decoupled中仅BUY有独立Normal priority，其余为0原子。保留原conditional quantity支持，包括固定BUY数量时仍保留同原策略的quantity随机坐标，不另混入支持缩减干预。

joint logP严格求和：方向categorical、有效quantity Normal、decoupled有效BUY priority Normal。coupled的copy/zero支持必须校验，不能在错误原子上给出有限概率。不得把logP改成每股平均来放宽KL，也不能用已成交股数或clip后的latent替代原始采样动作计算概率。KL仍是动作请求分布KL，不是仓位变化。

新mapper只改BUY排序来源：整手数量由quantity计算，topN/现金排序使用priority；卖出、T+1、冻结、实际完整EXIT例外和现金剩余保留。coupled模式须与旧2N mapper逐案例完全一致。次日真实模拟价格和可用资金仍由Broker复核，不向Actor输入未来执行价。训练与推理使用同一新动作契约，旧模型保留旧布局，不静默迁移。

## 训练方法

新PPO subclass以本机Stable-Baselines3 2.7.1的train逻辑为基础，在同一次Adam中合并：

```text
loss = clipped_PPO + vf_coef*value_MSE + ent_coef*entropy_loss
       + lambda_rank*rank_Huber
```

plain方法lambda_rank=0；rank方法=.05，每次实际被接受的PPO更新从独立train-only池抽1,024个stock-date样本。辅助预测仅用market_encoder和signal head，不读账户，不将未来标签放入观察或直接变成交易先验。

保留优势归一化、联合概率ratio/clip、GAE、value loss、targetKL和全局gradclip。辅助抽样在KL检查通过之后，使用独立NumPy RNG，不消耗PPO shuffle、日期、账户或K随机流。没有单独aux Adam，也不在PPO训练后偷偷追加encoder更新。lambda=0的新class应与原PPO合成重放等价。

辅助loss会改变encoder、global clip和后续Actor/Critic目标；这是训练方法干预，不是完全相同优化路径。相同RL步数不代表相同计算或监督曝光。若KL提前停止，以实际Adam和监督样本记录为准，跳过的更新不算成功优化。

## 标签与数据

只读复用已封存新浪300股、20特征、lookback1，以及2023/2024历史折。每折训练截止当年3月31日，验证Q2；旧训练scaler不重拟合，所有cell共用5日特征完整性mask。

构造辅助标签前，features和prices先物理截断到该折train_end。标签为：

```text
gross_forward_return = close[t+5] / open[t+1] - 1
target = 当日有效标签集合的相对rank，映射到[-1,1]
```

这里5表示本地union-calendar的五行，要求所有中间open/close报价完整、正且有限，不能跳过缺失报价。决策日还须过去共同五日特征有效；最后五个训练决策日purge。记录最大decision/label endpoint、日期/stock索引、特征顺序、pool及全部原输入SHA。标签不跨训练边界，不用验证数据拟合任何标准化或标签池。

辅助rank是gross预测目标，金融reward仍是每日扣成本净收益。预测IC、低rank loss或更多监督样本不是净收益改善证明。两历史窗口已暴露，不是独立盲测；不下载、不重跑旧2025绩效或评估2026-07-01以后的保护期。

## 固定预算与运行

| 项目 | 本轮安排 |
| --- | --- |
| seeds / folds | 42、123、2026 / 2023、2024 |
| cell数与正式trial | 4cell × 2fold × 3seed = 24 |
| short / long | 8,192 / 16,384，共享前缀 |
| 真实金融RL | 393,216步 |
| 快照 | 48份；步数和589,824，含196,608前缀，不算额外训练 |
| episode / 512步chunks | 预期1,536 / 768，实际记录 |
| Adam | 无earlyKL时每run320/640；long合计15,360，short7,680已含 |
| rank曝光 | 无earlyKL时12rank run共7,864,320次stock-date抽样，非新行情或RL步 |
| 训练初始账户 | mixed cash/equal/concentrated/random、256步episode、recent_every4 |
| train / evaluation K | 10/20/40 / 主K20 |
| 初始化控制 | 各fold/actor/seed内复用方法，共12组×4账户=48路径 |
| 训练模型验证 | 48模型×4账户=192路径 |
| 规则验证 | 2fold×3规则×4账户=24路径 |
| 验证总数 | 264路径；若每条59步为15,576步，报告实际值 |

CUDA、torch_threads1、初actor gain.1、quantity/priority log_std-.7。共同PPO LR=3e-5，n_steps512、batch128、epochs5、gamma.99、GAE.95、clip.2、targetKL.03、ent0、vf.5、gradclip.5、常数学习率。LR相对旧研究改变，旧收益只能作历史描述，不冒充新矩阵严格配对基线。

所有组合执行完整预算；不按中途收益追加32K、停掉失败cell或更改超参数。用户已选择16K，本轮不执行扩展32K。工程中合成测试/硬件benchmark不是金融pilot，分别披露。

原模拟资本100万元，佣金.0003/最低5元、卖税.0005、滑点.001、量参与率.01、lot100、单股及BUY/REDUCE单步名义额5%、max_commission_ratio.001。费用在净值中扣一次，不增加微单处罚或放宽10%研究风险门槛。

只新增隔离模块，不修改旧85源码及旧产物。原PPO、旧2N动作和旧模型不能冒充新3N契约。候选仅为历史研究候选，不发布默认或真实订单。source/Data/标签/artifact身份登记后冻结；阻断或偏离必须记录，不边跑边改实现。

## 评估与审计

先平均每seed的四账户，再平均同seed两折，最后计算三seed均值和样本SD。报告每fold、每seed、短/长预算、初始化增量、动作主效应、方法主效应及交互，不能把六fold/seed或48快照当独立种子。

沿用HOLD、每日动量和20日动量经济规则，相同数据、数量与执行机制；规则priority与原数量排序信号一致，用于固定基准，不悄悄更换容易胜出的规则。

每cell/预算两个fold均须最坏12路径DD≤10%、自身初始化增量均值>0且≥2/3seed正、平均ROI超过20日动量，才可成为合格候选。按ROI、最坏DD、成本、短预算及cell名排序；无合格者为null。代表fold2024/seed42固定，不挑最盈利seed。训练奖励本轮没有风险项，不能把验收回撤门槛说成Critic自动学习的惩罚。

原生3N未clip动作、真实old logP/value/reward/done、整手/费用/T1账本、episode随机流、prefix、实际Adam和aux曝光全部留档。另报告active quantity/priority坐标、joint KL/clipfraction/entropy、请求与映射/成交、size筛选偏移、qty-priority相关及训练/确定性部署差异。方法效果不能仅用rank IC或loss替代。

旧可信参考audit SHA：`fbb932aea3e269b3b46308e0bda21db6ef082bc9d92db033c1c42df26fdfbe41`。prepare需验证参考锚点、旧85源码、两折文件，再登记本轮源码、协议、标签池和预算。

两个worker按fold隔离，trial独立status/event/control目录，完成只依赖completed marker和注册hash；全局报告在两fold全部封存后合并，不再用共享status或partial冲突重入调度。最多两个GPU trial并发，检查显存，但不将调度加速当策略效果。

## 验收与交付

离线覆盖3N概率/原子/未clip支持、coupled旧mapper等价、独立priority排序与qty不混、冻结/EXIT/T1/整手/费用、保存加载/假契约拒绝、初始化方法等价、aux标签purge/未来扰动/独立RNG/lambda0原PPO重放、真实tiny PPO更新、完整缓存和预算计数。先提交并通过工程准入，再登记正式训练。

最终提供详细报告、摘要、模型索引、收益图及独立原生账本/监督样本审计。HFQ研究价格、固定2026名单、真实未复权/公司行为/交易状态以及独立未见股票验证缺口仍保留；代码与研究完成不等于已达到每日实盘配置目标。
