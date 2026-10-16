# 训练期经济配对诊断预注册草案

日期：2026-10-16。状态：实现及本地合成验收完成，待冻结提交与prepare；本阶段不预写真实标签统计或收益。候选输出为 `reports/runs/ppo_counterfactual_pilot_20261016`。本轮仅做train-only历史动作配对诊断，不进行Actor预热、PPO正式训练、模型拟合或参数搜索。

## 依据与冻结边界

父研究见[状态修复详细报告](reports/ppo_timing_phase_state_repair_20261015.md)与[摘要](reports/ppo_timing_phase_state_repair_20261015_summary.md)。其四物理run、八快照已完成，linear 8K在2023 Q2有局部增量、2024 Q2持平，筛选未准入；32K出现更多额外调仓及验证退化。不能未经动作级证据把退化单独归因于费用、Critic或过拟合。

硬钉[父唯一审计](../reports/runs/ppo_timing_phase_state_repair_20261015/final_audit.json)SHA `af731fb246da9dfb15999a73a1761d1e4c804a7525f3865e7395398d9e04b78d`，父6,712签署输入及原96份src不改。所有新工具和测试隔离新增，不能回写父协议、模型、buffer、trace或审计；也不从父失败/成功run借取本轮样本。

固定沿用本地同源新浪300股、HFQ研究执行代理、F20、原训练期scaler、129维macro观察、四类初始账户和同一Broker/shadow基线。股票选择、M20排序/数量、整手、T1、冻结、应收、事件、费用与滑点不改；强20日日历M20基线不改。HFQ、静态2026股票名单与历史状态缺口仍是限制，本轮不下载、不换源、不新增特征、标签观察或动作。

## 问题与非目标

问题是：在固定日历策略采到的真实训练状态上，当前先HOLD或先REBALANCE，后续采用同一个固定日历延续时，20步费用后结果是否存在有量级的差异，差异集中在哪些phase、账户与日期层？

本轮不证明策略能预测该差异，不将oracle正值当作可实现收益，不比较新模型与基线，不把标签塞入PPO buffer、GAE或old logP。标签是有限期、固定rule延续下的一次历史动作差，不是Qπ、不含对未来路径的期望保证，也不构成新的合格模型。

## 训练期与采样清单

只使用2023/2024两个fold各自训练期，边界为该年3月31日，构造provider前物理裁剪至train_end。旧Q2验证窗口、2025/2026及2026年7月后保护期不用于标签、调参、评估或按结果筛样本。所有20步标签终点必须不晚于对应train_end。

每fold固定seed42，四账户×三日期层×四phase层的48种组合完整循环一次。诊断专用源episode预先固定为80步，而非父PPO的240步；只改变采样可用起点和人工TimeLimit，不改任何经济执行机制，也不运行80步PPO。这样offset60加H20可完整落在源episode内，并使最近日期层能够覆盖训练末段，避免240步采样器配合短offset只覆盖较早日期。该选择依据时间边界而非标签收益，生成标签后不得改选。

采用两遍真实采集。第一遍48次reset，每个episode按固定日历M20采集60步，在offset0和60记录当前观察与完整状态指纹；冻结全部96项manifest及SHA后，第二遍重新构造相同数据/执行配置的fresh环境，重置seed42，完整重演相同48次reset与日历动作。逐点核对日期、phase、观察及全部状态指纹一致后，才调用配对分支。因此每fold共96次源reset、两遍各48次，但目标状态仍仅96个，不是192个独立标签。

不因信号弱、费用高、标签为负或近零追加reset。offset0指reset后、尚未执行采集动作的决策状态；offset60指已执行60步后的决策状态，不是第60步执行前。

由于60 mod 20=0，同一episode的两个状态phase相同，但日期、持仓、应收/锁仓和市场状态可能不同；不得把它们当作两个独立phase样本。每遍48次reset按主动短采集处理，不冒称完整80步或240步episode。下一episode开始的主动reset不列为自然终止或TimeLimit；offset60分支在第20步恰好抵达80步TimeLimit时如实标记，不加任何bootstrap。

在查看任何配对标签前冻结完整采样manifest及SHA：fold/seed、episode与48层身份、offset、日期/phase、H20端点、129维当前观察、完整状态/RNG指纹和数据/契约身份。恢复采用第二遍真实重演并严格核对完整指纹，不保存或加载任意pickle对象，也不能仅用净值、持仓比例或trace聚合摘要重建账户。原行情引用已签不可变产物；重演不得漏掉可卖/锁仓、应收、成本、事件、调度/RNG或影响未来执行的缓存，任何不一致均停止，不调整指纹掩盖差异。

192条候选记录全部保留。无效状态保留明确拒绝原因，不静默删除、补采或替换；不足192个有效pair时不能宣称完整计划成功。有效标签无论正、负、零或绝对值小均保存，不挑赢家或高绝对Δ样本，不按label排序改变预算。

## 完整状态的两分支执行

对每个冻结状态，从同一完整真实TimingEnv原件独立deepcopy两个分支，串行执行并释放资源；不能先运行一支后从其终点生成另一支，不复用另一支订单。两分支开始的Agent及shadow状态分别逐项相同，起点Agent NAV必须相同且为正。

首步只改变Agent macro token：H分支HOLD，R分支REBALANCE_M20；shadow仍按原phase执行固定日历M20。后续19步两支Agent均按原phase自然推进：phase0 REBALANCE，其余HOLD。首个R不重置phase，也不另起一个20日调仓周期。股票订单须由各支自己的当前账户重新解码，经同一个Broker复核；事件、T1、冻结、整手、费用和滑点照旧执行。

两支shadow从同状态、相同phase和行情出发，必须在全部20步保持完整经济状态及路径相等；Agent分化不能污染shadow。两支及源环境之间不得共享mutable对象；验收需覆盖portfolio、broker/history、事件/应收、可卖/锁仓、市场游标、phase、缓存及独立调度/RNG对象的别名隔离。immutable数据共享只有在不可写与身份一致得到验证时才可接受，不能共享可被分支推进的provider状态。

每个pair执行前后核对源环境完整状态及RNG指纹不变，包含影响step/observe/reset的last/cache字段、调度bag/block、计数和独立RNG；不能只核净值或持仓。另核Python、NumPy、Torch CPU全局RNG；CUDA只在原已初始化时核对其状态，不为标签主动初始化设备。指纹采用登记的typed状态编码，合法拒单history里的NaN价格可以编码保留，但不能据此放宽模型观察、净值或财务字段的finite检查。未知/未登记状态字段不得静默忽略。helper只证明当前观察与当前经济状态一致，因果特征与scaler可得性依赖登记的数据管线，不能将该检查夸大成任意输入都无泄漏。

## H20标签与数值恒等式

固定H=20、gamma=.995。对分支b∈{H,R}，原相对奖励为 `r_b,j = log(NAV_b,j+1 / NAV_b,j) - log(Shadow_b,j+1 / Shadow_b,j)`；沿用原已扣费用与滑点的净NAV，不重复扣费。

主诊断标签为：

```text
Δ = sum(j=0..19) 0.995^j * (r_R,j - r_H,j)
```

另报未折扣 `sum(r_R-r_H)`、`log(NAV_R,20 / NAV_H,20)`、费用/滑点/成交与持仓变化。起点NAV相同、shadow逐步相同条件下，未折扣两式应一致；逐步shadow奖励应消去。主折扣Δ一般不等于终值log比，未折扣统计不是第二个选优label，更不是另行挑选盈利期限。

数值等价验收使用绝对容差1e-10：跨支shadow状态/奖励/净值、相对奖励与Agent净log差的消去、未折扣累加与终值log比；不以放大容差掩盖状态污染。源状态/RNG及离散股数、时间、标记须严格一致，不把数值容差用于掩盖整数或流程差异。

输入仅为当时已有129维观察。未来NAV、quote、费用和成交属于独立label/evidence，不作为观察、mask或采样依据。报价不可用仍走原valid/交易限制，不能为了得到标签填造价格或把缺行当停牌事实。

## 拒绝与终止规则

拒绝未reset、已done、未知状态、129输入非finite、必要经济字段不合法、没有足够20步行情、终点超train_end，或会在H之前跨TimeLimit/自然终止的pair。不能通过reset、改episode时钟、扩大provider日期或续接另一episode凑足H。

第20步末恰好自然终止/TimeLimit可以保留，须如实标记；终止后不继续推进、不加Critic bootstrap。H之前提前终止则不作为完整H20标签，保留拒绝状态和实际已付步骤。异常发生在Agent/shadow推进不同位置时，两个book实际成本分别记录，不强行伪装相同步数或回滚成0。

## 固定预算与计数

| 环节 | 每fold Agent步 | 每fold shadow步 | 两fold Agent步 | 两fold shadow步 |
| --- | ---: | ---: | ---: | ---: |
| 两遍日历采集2×48×60 | 5,760 | 5,760 | 11,520 | 11,520 |
| 配对2分支×96状态×20 | 3,840 | 3,840 | 7,680 | 7,680 |
| 合计 | 9,600 | 9,600 | 19,200 | 19,200 |

完整计划共192目标状态、384分支、38,400个book-step。两遍采集的重复状态只用于重演验证，不计作额外独立样本。它们是历史经济诊断执行，不计为真正PPO RL经历；PPO RL、NN forward、梯度、optimizer和模型拟合均为0。采样清单读写、deepcopy、状态指纹与数值核验不算金融步，但所有真实step调用均计数。

这是计划而非已执行计数。失败分支、提前终止或工程重试的已付步骤额外如实记录，不隐藏、不借旧run冲抵、不重复利用作独立样本；计划外补采或追加预算必须显式另行授权。

## 报告与统计纪律

全量报告Δ的分布/分位数、正负/近零比例、净终值差与费用/滑点差，按fold、账户、日期层、phase/bin和offset分层，同时报告有效/拒绝数量及实际预算。经济signal阈值固定1bp=.0001，只作 `Δ≥.0001`、`Δ≤-.0001` 与中性区间统计，不删样本、不强迫BUY或交易；该数值是折扣log奖励差的阈值，不能直接当作可实现组合ROI百分点。

oracle机会相对原日历首动作计算：phase0原动作R，非phase0原动作H；仅用于描述已知标签后哪支更优。不能将正Δ一律称优于日历，也不能把重叠状态的oracle优势相加为一条可执行净值曲线。

明确同episode两offset、跨episode重叠日期/标签区间、同股历史、两fold训练期嵌套与单seed相关性；不把192条当192次独立市场实验。采集来自固定日历策略分布，未覆盖PPO已偏离后的所有账户状态；不因oracle标签有机会就宣称输入可预测、模型能学到或实盘赚得到。

当前scaler用完整outer-train拟合，即使滚动特征只用过去，早期训练状态仍使用整个outer-train的统计量。本轮只做该已冻结训练口径的诊断；后续若做严格内层时序验证，必须按inner-train重新拟合/冻结scaler及资格规则并处理标签purge/embargo，不能把现outer-train scaler称作内层无泄漏。

## 工程验收与后续准入

已新增 `tools/timing_counterfactual.py`、`tests/test_timing_counterfactual.py`、独立 `tests/test_timing_counterfactual_review.py`，以及 `tools/research_timing_counterfactual_pilot.py`、`tests/test_counterfactual_pilot.py`。prepare显式登记这五份代码/测试与本协议共六份新输入及父身份，不以import旧模块替代依赖冻结。父参考配置仅用于继承经济数据；派生配置明确登记诊断episode从240改80、输出独立目录，不覆盖父配置或冒称运行父PPO预算。

合成验收至少覆盖：完整状态深复制及mutable alias负例、源RNG/缓存不变、首步H/R与原phase连续、shadow逐步一致、T1/应收/事件/冻结/整手/费用保持、H前终止拒绝与H末终止标记、训练日期物理裁剪、未来量只在label、奖励/终值恒等式、48层与96状态清单、零/负/小Δ不筛除，以及实际book步计数/失败成本。验收不得通过额外NN预测、GAE或模型更新实现本轮标签。

只有新代码、作者/独立验收和冻结提交完成，才允许新目录prepare及正式诊断；结果另写详细报告/摘要与实际命令/退出/身份，不回改本协议挑选信号。

冻结前最终联合验收使用 `AShareGPU` 环境，命令如下；root原生session `72488`，最终chunk `bc9296`，exit `0`，`127 passed in 27.97s`。其中标签作者76项、独立复核25项、驱动26项。测试均为合成数据或隔离临时目录，不能替代真实数据诊断，也不证明策略可预测动作收益。

```powershell
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -m pytest tests/test_timing_counterfactual.py tests/test_timing_counterfactual_review.py tests/test_counterfactual_pilot.py -q -p no:cacheprovider
```

正式诊断依次执行以下命令。`prepare`冻结输入身份；两个`run`分别生成各自完整96状态采样清单后重演配对；`report`只能读取已完成且通过哈希与数值恒等式复核的两折，不采新状态、不补标签。任何失败保留实际成本并停止，不覆盖残缺折目录。

```powershell
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.research_timing_counterfactual_pilot --stage prepare
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.research_timing_counterfactual_pilot --stage run --fold 2023
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.research_timing_counterfactual_pilot --stage run --fold 2024
& 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe' -B -u -m tools.research_timing_counterfactual_pilot --stage report
```

Actor预热、监督/Q辅助与PPO训练只是候选后续，须另立协议，保留θ0→θaux→θPPO及同参数/同预算PPO-only对照，监督曝光与真实RL增量分开。固定continuation标签不冒充Qπ，拟合和验证要处理重叠/泄漏/目标冲突。本轮不改变原跨折、多seed、至少+0.5pp及风险最终门槛，不promote、不承诺胜基线。
