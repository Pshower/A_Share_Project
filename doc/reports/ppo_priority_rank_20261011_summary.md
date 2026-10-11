# 排序动作与辅助训练研究摘要

日期：2026-10-11。**已完成24次训练、48快照、393,216实际RL步。八配置未通过全部收益/风险门槛，候选null，默认模型不变。** 最终离线回归、只读工具和独立最终审计通过。研究完成不等于实盘目标达成。

## 结果

每seed先平均四账户，再平均同seed两历史fold，最后对三seed求均值/样本SD；以下ROI为%、SD为pp，不是显著性结论。

| 配置 | 8,192步净ROI±SD | 16,384步净ROI±SD |
| --- | ---: | ---: |
| coupled PPO | -1.066099±1.160042 | -1.161539±1.895211 |
| coupled rank_aux | -1.219838±1.412212 | -0.588144±1.840187 |
| decoupled PPO | +1.190440±3.271436 | -0.608901±1.643927 |
| decoupled rank_aux | +0.881897±2.742447 | -0.548201±1.505629 |

排序/数量解耦的跨折动作主效应短/长为 **+2.179137/+0.296290pp**；辅助方法主效应 **-0.231141/+0.317047pp**，交互 **-0.154803/-0.512694pp**。初始化效应严格0、净效应等于学习增量效应，但跨Actor随机探索函数不同，不能声称已消除全部探索归因。辅助长预算2023改善、2024退步，不是稳定跨年增益。

**更多步数不稳定增加收益：** decoupled PPO加长平均下降1.799341pp，三seed均下降。观察最高均值配置是decoupled PPO短预算，但2023最坏DD12.071505%、2024均值亏损且学习增量为负；固定2024/seed42代表ROI为-3.335610%，不能挑均值或幸运seed晋升。八配置全部2023最坏DD>10%，候选null。

## 本轮做什么

固定纯行情均值摘要、Critic结构/损失及每日扣成本收益，二乘二比较quantity/priority耦合或解耦、普通PPO或五日rank辅助。每trial独立训练自己的Actor/Critic，只是模型内共享编码，不是跨trial共用Critic参数。不下载、不重拟合scaler、不评估保护期。

统一3N动作：方向、quantity、BUY priority。quantity决定整手数量，priority参与资金/名额排序；HOLD/REDUCE/EXIT以及冻结、T+1和剩余现金不改。coupled的BUY priority是quantity拷贝原子，decoupled增加独立Normal；联合概率精确计入实际采样坐标，不拿裁剪或成交结果代替动作。

所有cell总参数 **125,322**，同seed初始张量一致。同Actor内两方法初始随机策略一致，跨Actor确定性初始排序/成交一致；**跨Actor随机执行函数不同**，必须保留各自初始化控制。

辅助任务只用market encoder和零初始化signal head预测训练期五日gross收益的截面rank；系数0.05、Huber delta0.1，每次接受的PPO Adam更新抽1,024个stock-date样本。同次Adam合并，不追加optimizer，KL拒绝前不抽标签，未来标签不进入观察。构造前物理截断到train_end，最后五个决策日purge。

signal head只看单股编码，不看同日市场摘要，学习的是给定单股特征的条件期望排名；不能宣称完整建模动态截面相对位置。相对特征/摘要条件辅助head留待另轮对照。

这轮不是完整稀疏selector，仍保留mapper的topN/现金过滤；没有新风险奖励、BC、GRU、动量先验或新下载数据。

## 固定预算与验收

复用新浪300股、20特征、2023/2024训练折及Q2验证，三seed42/123/2026，用户锁定16K最长预算、未追加32K。实际24trial/48快照、393,216RL、1,536episode/768chunk；快照步数和589,824含196,608共享前缀，不重复计RL。**15,360次joint Adam内7,680次含aux，额外监督optimizer为0**；7,864,320监督stock-date曝光不是新行情或RL步。192模型+48初始化+24规则共264评估路径、15,576步。

针对性合成154pass、root集成4pass；训练前全仓准入1,293pass/442.59秒。正式封存后最终回归实际 **1,293passed、13deselected、11warnings、389.43秒、exit0**；13项联网未执行，重验数不累计。

prepare/两worker/全局汇总真实exit0；注册Git9a52851，48训练模型Git c7998904×8/09c65fe2×40，冻结91src、旧85未改。独立最终审计exit0通过24native、标签/规则、1,104archives/70,728CRC、预算/数学/acceptance/事故lineage；签3,140study文件及源/外部协议等，最终人工MD不纳签，冻结协议MD c055纳签。额外NN/env/grad/Adam及candidate替代均0，避免把完整性通过称收益达标。

每trial64episode、四mode各16，初始账户/状态不重复数44–47/57–62；不存在recent绑定单一mode。recent时间×账户未分层均匀，recent cash仅1–3片段/seed；分层调度留待另轮。

train-only池2023/2024为354,930/425,912样本、1,449/1,690决策日、292/297股票；最后标签端点3月31日/3月29日均未越train_end。历史重叠不算新独立行情，最后五行purge，未用验证标签。

实际CUDA；四cell平均elapsed为1,269.25/1,306.44/1,309.35/1,356.00秒，24trial累计31,446秒。long audit计时含训练/保存/诊断，不含后续金融eval、不是双worker总墙钟。CUDA单trial peak allocated269.49–288.38MiB/reserved344–404MiB，非整卡峰值；合成GPU快18.3倍不等于正式整体加速。

## 执行机制

训练coupled quantity-priority相关exact1、映射数量rank偏移约+0.35；decoupled相关约0.0015、偏移约+0.0004，**独立噪声基本消除训练数量上尾优先偏差**。但确定性验证同日相关仍0.9449–0.9706、偏移+0.1066–0.3301，不能称推理完全解耦；共享body/复制初始化下均值仍相似，相关性不是盈利因果证明。

192模型验证共2,036,389个raw BUY请求、82,344次实际BUY成交，约4.04%；每决策179.77请求→7.27成交。投影仍主导，下一轮优先真正无放回预算/slots感知selector，而非只加步数。见[原生执行诊断](../../reports/runs/ppo_priority_rank_20261011_16k/execution_diagnostics.json)。

正式评估同时比较扣成本收益、最坏回撤、初始化增量及种子稳定性。每配置在两个fold都须通过DD≤10%、均值初始化增量>0、≥2/3 seed正、超过同机制20日动量，才可成为历史研究候选。不因中途收益扩预算或挑种子，不发布默认模型/真实订单。

## 完整性事故与恢复

首chunk曾稳定读为58/CRC坏，后cached与NO_BUFFERING均为原c887/CRC好、ID/mtime不变；用户及三个agent均报告未手动处理。根因unknown，不称人工恢复或硬件归因；无现场坏raw capture，坏variant是**事后精确重建**，不是现场原件。

两worker exit0后透明更正两处旧登记/receipt并reseal，canonical NPZ写0；原completed/receipt bytes已保留。原c311凭据的4path标签以8b47勘误sidecar修正，原凭据/23ccb脚本未覆盖；无额外RL或重训。详见[事故登记](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/incident.json)、[reseal凭据](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/incident_lineage_receipt.json)、[路径勘误](../../reports/runs/ppo_priority_rank_20261011_16k/forensics/receipt_path_correction.json)。

## 当前结论

本轮支持继续研究排序/数量解耦，**未得到通过门槛的每日配置模型**。优先另轮检验无放回/预算slots感知selector、市场摘要条件辅助head和分层时间×账户调度；风险目标与真实成交账本另立对照，不临时改本轮门槛。窗口反复曝光、HFQ和固定2026池仍限制泛化。

费用/lot通过仅表示冻结commission/最低5元、卖税0.0005、滑点0.001和100股规则内部数学一致，不表示各历史日期税费制度和各板块交易单位都真实复原，也不是实盘成交验收；与HFQ经济代理、固定2026股票池和开发窗口反复曝光限制一并保留。

logP审计检查原生逐step有限性、3N/原子/合法支持、来源与prefix；未保存所有旧theta，也不新增forward，所以不声称全部逐步旧神经密度已重算。合成集成测试验证的是源码likelihood路径，不能替代这一缺失的正式数值复算覆盖。

全48checkpoint/receipt、只读工具及数学汇总已完成，最大对账误差0；收益门槛未通过与完整性通过是不同结论。[独立最终审计](../../reports/runs/ppo_priority_rank_20261011_16k/final_audit.json) SHA256：`89bc5f465eada6a845812b9f538886fc510dfcaecd392e54100b3f1f93379ce2`。

[详细报告](ppo_priority_rank_20261011.md) · [预注册](../ppo_priority_rank_protocol_20261011.md) · [机器汇总](../../reports/runs/ppo_priority_rank_20261011_16k/study_summary.json) · [对照图](../../reports/runs/ppo_priority_rank_20261011_16k/comparison.png) · [48模型索引](../../reports/runs/ppo_priority_rank_20261011_16k/model_index.csv)
