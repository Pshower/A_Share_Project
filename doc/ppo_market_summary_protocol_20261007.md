# 纯行情摘要优化研究预注册

日期：2026-10-07。本文在正式训练前冻结；实际进度和结果另写报告，不回写本协议。目标是检验纯行情摘要能否提高扣成本后的收益，不保证找到盈利模型，不更换工作台默认。

## 问题与假设

上一轮同参数研究中，纯行情摘要在4,096步优于融合摘要，但2,048步反向，且全部配置未通过两折收益/风险门槛。Critic detach没有稳定改善，不能据此移除Critic。

当前摘要为有效股票编码的masked mean。它给所有有效股票相同权重，只保留一个中心位置；两组股票即使均值相近，市场分化程度也可能不同。本轮只研究两项可学习改进：是否应关注不同股票，以及是否应补充横截面离散度。它们是待验证假设，不是已证实的收益来源。

其他值得研究但本轮不同时实施的方向包括：行业分组摘要、个股相对市场的残差特征、多时间尺度摘要、价格/成交量状态门控，以及预算感知的稀疏成交。行业历史信息缺失、时序与特征改动、Actor执行改动需要独立对照，不能全部叠加后宣布摘要有效。

## 三臂设计

所有股票共享原有market/account encoder及交易头；摘要只使用行情编码和valid_mask，不能读取持仓、现金、可买卖标志或资金约束。账户状态仍通过个股融合与held context传入Actor和Critic。

记有效股票编码为m_i，有效mask为v_i，c=max(sum(v),1)，原摘要为mu=sum(v_i*m_i)/c。

| arm | 摘要 | 可学习新增参数 |
| --- | --- | --- |
| mean | mu | 无；保留同构但闲置的q/W |
| attention | mu + sum((a_i - v_i/c)*m_i) | q，128参数 |
| dispersion | mu + W*var | 无bias W，128×128参数 |

```text
score_i = dot(q, m_i) / sqrt(128)
a = valid-only normalized exp(score - max_valid_score)
var = sum(v_i * (m_i - mu)^2) / c
```

attention的无效股票权重为0；全无效时mu、var和权重均为有限0。var是总体中心二阶矩，非无偏样本方差，不对仅一只有效股票制造除零；不使用未来股票排序或收益标签。股票置换应仅置换对应预测，增加无效padding不得改变已有股票结果；有效池变化允许改变市场摘要，不要求股票池扩大时预测完全不变。

三臂全部包含同名q和无bias W，均初始化为0；总参数均为141,575，原125,063加16,512。attention使用相对同口径uniform的残差，确保q=0时精确回到原mean；dispersion的W=0同样回到原mean。三臂同seed的全部初始参数、Actor方向/size/log_std及Critic前向相同，不再接受仅相同theta但初始函数不同的混淆。

**相同总参数不等于相同有效容量。** mean的q/W均闲置，attention仅q参与计算，dispersion仅W参与。Adam状态数和有效训练参数会不同，须原样记录。q/W同时接受Actor和共享Critic的梯度，本轮不能归因成只改善Actor或消除梯度冲突。

保留原HOLD/BUY/REDUCE/EXIT与条件数量分布、整手/T+1/冻结逻辑、现金残余和合法动作掩码。不得更改size初始化、探索退火、先验、奖励、费用、股票池、特征、共享Critic、采样方式或部署规则。

## 数据与预算

只读复用已封存新浪300股、20特征、lookback1，以及2023/2024历史折。每折训练到当年3月31日，验证为Q2；scaler仅由该折过去训练数据拟合，不重新拟合。所有模型共用5日完整性mask，provider在加载前物理截断；2024初态使用实际2024-03-29收盘，不能虚构周末报价。

两个窗口此前已暴露，不是独立盲测。固定2026股票名单及HFQ研究执行限制仍保留；不下载、不评估2026-07-01及以后绩效，不借新实验重跑旧2025绩效。

| 项目 | 固定安排 |
| --- | --- |
| seeds | 42、123、2026 |
| 正式训练 | 3arm × 2fold × 3seed = 18次 |
| 快照 | short4,096 / long8,192，共36份；short为共享前缀 |
| 实际RL预算 | 18×8,192 = 147,456步 |
| 快照步数之和 | 221,184，含73,728共享前缀，不能算新增训练 |
| 每run episode | 256步，完整8,192步共32个 |
| 初始账户 | mixed覆盖cash/equal/concentrated/random |
| 日期采样 | recent_every4，沿用独立日期/模式/账户/K随机流 |
| train K | 10/20/40；验证仅K20 |
| 初始化路径 | 同fold/seed三臂精确等价后复用，共6组×4账户=24条 |
| 模型验证 | 36快照×4账户=144条 |
| 规则验证 | 两折各3规则×4账户=24条，独立新产物 |
| 总验证路径 | 192条；若每条59步则11,328验证步，报告实际值 |
| 新pilot | 0；不按中途利润加预算或停掉失败臂 |

所有臂使用AShareGPU/CUDA，torch_threads1，hidden256/128，actor_output_gain.1、bias0、initial_log_std=-.7。PPO为n_steps512、batch128、epochs5、LR1e-5、gamma.99、GAE.95、clip.2、targetKL.03、ent0、vf.5、global gradclip.5，保持常数学习率。

无KL提前终止时每run为320次Adam、合计5,760；short160次合计2,880仅为前缀。以实际optimizer state记录为准，不把PPO epoch数或快照数当真实优化次数。不新增上一轮固定批梯度诊断；已有原生Critic拟合和训练日志继续保留。

资本100万元，佣金.0003/最低5元、卖税.0005、滑点.001、成交量参与率.01、lot100、单票5%、单步BUY/REDUCE名义额5%、max_commission_ratio=.001。只有实际完整EXIT可豁免微额经济门槛。奖励仍为逐日扣成本净收益，费用只扣一次，无额外微单惩罚。

## 比较和门槛

先在每fold/seed内平均四账户，再平均同seed两个fold，最后对三个seed计算均值和样本SD；不把账户或六个fold/seed当六次独立随机训练。并列披露每fold、每seed以及最坏账户，不以总体均值掩盖失败窗口。

预指定主要比较为attention-minus-mean、dispersion-minus-mean，分别在两个预算上报告配对净ROI、初始化差、学习增量差、成本、现金、换手及回撤。由于初始函数相同，初始化差理论为0，必须由实际初始控制身份验证，而非事后假定。小样本只作研究观察，不宣称统计显著或普遍泛化。

保留自身初始化控制及HOLD、每日动量、20日动量经济基线，三条规则使用相同有效mask与执行约束。初始化控制复用须同时满足：初始policy.pth完全相同、q/W均0、规范化预测契约只去掉pooling字段后相同；不能复用已训练模型或跨seed/fold控制。

每个arm/预算若作为合格研究候选，两个fold均须满足：

1. 三seed四账户共12条路径的最坏回撤≤10%。
2. 自身初始化增益均值>0，且至少2/3 seed增益>0。
3. 平均净ROI超过同机制20日动量规则。

新增attention/dispersion如要宣称相对mean的改进，还须两个fold各自配对平均净ROI差>0，且每折至少2/3配对seed差>0。分别记录是否“观察均值更好”“稳定配对改善”“通过全部门槛”，不得混称盈利或实盘达标。

合格配置按跨fold逐seed平均净ROI、最坏回撤、成本、较短预算、arm名称排序；无合格者记null。固定代表fold2024/seed42，所有seed个体表现仍并列，不挑最盈利模型。即使合格也不自动升级默认或生成真实订单。

## 实现、验收与产物

独立输出为`reports/runs/ppo_market_summary_20261007`；只新增隔离policy/runner/test/tool文件，不修改旧83份源码或旧研究产物。可信参考为上一轮完整审计SHA256：

`4f9485bd43cec7afe60f423aa1b088e7b3c8b436226ea80deb739cbe36935b7a`

prepare需核验上一轮audit锚点、协议、旧83源码、两折文件身份，并冻结本轮全部源码及本文SHA。正式训练后不得继续修改实现或协议；遇到阻断错误须如实报告，不能边跑边修改再覆盖。

离线测试覆盖同参数及同初始函数、摘要账户独立性、置换与无效padding、全无效/单有效、query/W可训练梯度、保存加载及契约拒绝、数据时间边界、严格缓存、初始化复用及汇总完整性。新工程通过后提交再开始正式训练。

继续封存原生动作/旧logP、精确账户账本、episode随机流、快照共享前缀、实际Adam、全部model/data/scaler/source哈希。重点检查新增参数实际变化，方向/size/topN/资金投影、成交数、费用和随机训练与确定性部署差异，但不根据这些指标事后挑选新部署规则。

研究入口：

```powershell
python -m src.training.research_market_summary --stage prepare
python -m src.training.research_market_summary --stage train
python -m src.training.research_market_summary --stage report
```

完成后另写详细报告与摘要，包含失败项、完整预算和限制。不能把模型文件、单元测试或数据完整性通过视为“收益已提高”。
