# 纯行情摘要优化研究摘要

日期：2026-10-07。**本轮研究已完成：18次训练、36快照、192验证路径，金融训练147,456步；最终回归及独立完整性审计通过。没有配置通过全部收益/风险门槛，候选null，不替换默认模型。** 每日实盘配置模型目标仍未达标。

## 结果

固定Actor、共享Critic、奖励及交易机制，仅比较行情embedding的mean/attention/dispersion。总参数均141,575，q/W零初值，同初始策略函数；有效容量及全局裁剪影响分别保留。mean不是原始指标均值，encoder可隐式保留非线性统计，新结构测试归纳偏置/学习效率，非必然新增信息。

每seed先平均四账户，再平均同seed两折，最后对三个seed计算均值/样本SD；配对差单位为百分点，不是显著性结论。

| 摘要 | 4,096步净ROI | 8,192步净ROI | 短 / 长相对mean |
| --- | ---: | ---: | --- |
| mean | +0.702734% | +0.707718% | 0 / 0 |
| attention | +0.703497% | +0.705927% | +0.000762 / -0.001790 pp |
| dispersion | +0.613680% | +0.131585% | -0.089054 / -0.576133 pp |

没有稳定增益的新摘要。dispersion在2024局部提高+.052518/+.313058pp，在2023降低-.230627/-1.465323pp，不能择年声称改善。mean加长训练两折分别+1.565995/-1.556028pp，跨折几乎抵消，不支持步数越多越好。

六配置2023最坏DD均12.1372%至13.7353%，超过10%且落后同机制20日动量，全部不合格。固定2024/seed42长预算mean/attention为-3.722541%、dispersion为-3.615018%，不是配置均值或部署推荐。

## 学到什么

attention的q非零，但12快照10个全部账户指标与mean相同；uniform偏离最坏界仅.1544%至1.1234%，不是实测注意力/盈利界。固定预算弱干预无效果不等于架构普遍无用。

旧轨迹193万非HOLD请求过滤后仅17.5万映射订单，映射后成交率99.836%，瓶颈主要在独立请求到全局名额/现金分配，非Broker。它是旧研究数值，不混成本轮结果。

本轮[完整投影](../../reports/runs/ppo_market_summary_20261007/projection_diagnostics.json) 则为144模型路径中163.17万非HOLD请求、13.89万映射/13.87万非零成交，映射后成交率99.835%；BUY名义金额仅2.8707%通过映射。它是描述性瓶颈证据，不是移除约束就增收益的因果证明。

本轮mean/seed42 [size诊断](../../reports/runs/ppo_market_summary_20261007/size_selection_diagnostics.json) 显示训练映射可变lot区间分位约66%、确定性验证约50%，新仓偏采样上尾；这是quantity/priority共用信号的联合筛选，非仓位/资本比例。整手/现金/价格及train/val差异使其不能作收益因果证明。下一轮优先独立研究排序数量解耦和预算感知稀疏动作，不混改本轮。

## 验证与限制

金融actual147456步/5760Adam/576episode/288chunk，192路径（144模型+24初始+24规则）共11328验证步。73728短前缀已含，快照步数和221184不是新增训练量。六mean短快照均实证精确重放旧共同张量/Adam，不增加独立seed。

注册与全部54模型metadata Git为b17a31c dirty=true，冻结源码85份、旧83不变。worker/tail真exit0；主曾partialguard在环境/模型创建及learn前exit1，是协调退出非额外训练，随后仅cached汇总真实exit0，env/optimizer新增0，barrier已complete。

正式封存后的最终全套实际**967passed、13deselected、11warnings，260.58秒，exit0**。合成测试与金融预算分开，测试通过非收益达标。原132数值项复算对汇总误差0；绘图/36索引与投影exit0，无额外模型推理/环境/优化。

[唯一独立最终审计](../../reports/runs/ppo_market_summary_20261007/final_audit.json) 真实seal exit0、status=passed，15锚点/1,964 registry-marker关联及全部54模型metadata、85源码、192原生金融曲线重算通过，额外model/grad/env/optimizer均0。SHA256：`fbb932aea3e269b3b46308e0bda21db6ef082bc9d92db033c1c42df26fdfbe41`。完整性通过不等于策略达标。

未下载、未碰2026-07-01后保护期；窗口已暴露、固定池和HFQ研究执行限制仍在。不生成真实订单，真实未复权/公司行为账本、未见时期及未见股票验证仍须补齐。

[详细报告](ppo_market_summary_20261007.md) · [预注册](../ppo_market_summary_protocol_20261007.md) · [研究札记](../market_summary_research_notes_20261007.md) · [机器汇总](../../reports/runs/ppo_market_summary_20261007/study_summary.json) · [对照图](../../reports/runs/ppo_market_summary_20261007/comparison.png) · [36模型索引](../../reports/runs/ppo_market_summary_20261007/model_index.csv) · [最终验收](../../reports/runs/ppo_market_summary_20261007/final_acceptance.json) · [独立审计](../../reports/runs/ppo_market_summary_20261007/final_audit.json)
