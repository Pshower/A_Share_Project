# 项目文档索引

进行中：[因果动作价值预热与PPO正式对照](ppo_counterfactual_learning_protocol_20261017.md)。实现95a27b0、326项合成联合通过，真实prepare exit0，2023/2024开发worker已启动；先构造精确phase teacher，再真实比较原PPO与Ridge预热后PPO，不以监督或oracle收益冒充RL增量。训练与收益待实际完成后报告，多种子确认仍受冻结选择及原风险门槛约束。

已完成：[训练期配对动作诊断协议](ppo_counterfactual_pilot_protocol_20261016.md)、[摘要](reports/ppo_counterfactual_pilot_20261016_summary.md)及[详细报告](reports/ppo_counterfactual_pilot_20261016.md)。192个配对、38,400 book-step、0新增训练；186项合成回归与原生净值/订单冷重算通过。额外调仓的平均动作差跨折异号，不能据此增加调仓频率；phase0每折仅2个相关状态。未证明可预测性、未形成新策略ROI或合格候选，后续先验证动作价值信号能否学习。

已完成但未准入：[修复研究摘要](reports/ppo_timing_phase_state_repair_20261015_summary.md)及[详细报告](reports/ppo_timing_phase_state_repair_20261015.md)。四次训练、131,072新增RL步、5,120 Adam和八快照回测完成；linear8K在2023Q2较基线+0.977594pp、2024持平，长训练均退化。额外调仓增多不等于收益提高；确认0、无合格候选、不部署。最终独审和唯一签章通过，签6,712文件，原96源码及旧成本不变。下一研究为独立train-only配对动作诊断，不改旧协议或保护期。

已完成：[多步收益归因协议](ppo_timing_credit_protocol_20261015.md)、[详细报告](reports/ppo_timing_credit_20261015.md)与[摘要](reports/ppo_timing_credit_20261015_summary.md)。八trial、65,536 RL步、2,560次Adam；relative_log/lambda1在2023比M20高0.977594pp、2024持平，其余三配置两折持平。局部增量未达到跨折准入，确认0、不晋升。四个短trace控制重新训练精确复现；128实际GAE全量重建误差0，终审通过并签4,676文件，旧96源码和前轮输入未改。

已完成：[调仓初始化强度协议](ppo_timing_margin_protocol_20261014.md)、[报告](reports/ppo_timing_margin_20261014.md)与[摘要](reports/ppo_timing_margin_20261014_summary.md)。六trial完成49,152 RL步、1,920次Adam，较弱margin改变了行为却未跨折改善：0.5在2023差-5.427832pp、2024差+0.328781pp；最终未准入，确认0、不部署。margin2新训练精确复现旧参数/轨迹；最终独审通过并签署3,244文件，旧96源码及前轮输入未改。

已完成：[规则锚定调仓控制器协议](ppo_timing_protocol_20261014.md)、[详细报告](reports/ppo_timing_20261014.md)与[摘要](reports/ppo_timing_20261014_summary.md)。保留原20日动量股票引擎，RL学习HOLD/REBALANCE时机；10次训练、131,072 RL步、5,120次Adam。两确认折相对基线平均-0.942373个百分点，未胜出，不切换默认。单次跳过调仓节费但损失后续收益，不能仅由参数更新或正绝对收益认定成功；最终独审通过，2,259文件签名，原91源码不变，调仓decoder固定抽样与完整账本范围分别披露。

综合记录：[A股日频强化学习交易研究记录](research_record_20261011.md)。汇总2026-09-25至2026-10-11的数据来源、处理方法、模型演进、训练方法和13轮测评，区分观察最高、当轮研究候选与实际部署，不将不同口径收益直接排名。

已完成：[排序动作与rank辅助PPO预注册](ppo_priority_rank_protocol_20261011.md)、[报告](reports/ppo_priority_rank_20261011.md)及[摘要](reports/ppo_priority_rank_20261011_summary.md)。24次正式GPU试验、393,216实际RL步、48个8K/16K共享前缀快照、264条固定评估路径；3N数量/优先级对照和训练期辅助任务已实现。观察最高为独立优先级+PPO短预算，净ROI+1.1904%，较同轮共享排序+2.2565个百分点，但所有配置未通过两折门槛，候选None，不升级默认。最终1,293项离线回归通过、13项联网排除；91源码/原85份、训练标签、原生账本与CRC已复核，读取校验异常及仅登记更正的完整历程见报告。真实稀疏selector、截面条件辅助训练、风险奖励和实际成交适配仍待下一轮研究。

已完成：[纯行情摘要优化预注册](ppo_market_summary_protocol_20261007.md)、[报告](reports/ppo_market_summary_20261007.md)及[摘要](reports/ppo_market_summary_20261007_summary.md)。同初始函数的三臂完成18次训练、147,456实际RL步、36快照和192条验证路径；attention收益近似mean，dispersion在2024改善但2023退步，无跨折稳定增益或合格候选。最终967项离线回归通过、13联网排除；源码、原生账本和产物身份复核通过。没有更换Actor/奖励、默认模型或评估保护期，后续交易与风险研究见笔记。

后续候选：[纯行情摘要研究笔记](market_summary_research_notes_20261007.md)。区分相对强弱、市场广度、条件注意力、集合交互、风险及多尺度摘要，包含初始化、数据和Actor执行瓶颈；除本轮三臂外均未实现，不代表收益已提高。

已完成：[同参数归因协议](ppo_attribution_protocol_20261007.md)、[研究报告](reports/ppo_attribution_20261007.md)及[摘要](reports/ppo_attribution_20261007_summary.md)。相同125,063参数的2×2对照覆盖2023/2024两折、三种子、24次训练、48快照和98,304实际RL步。长预算纯行情摘要平均较好、短预算反向，Critic梯度截断无稳定增益；96次固定批诊断与264条验证路径已完成，无配置通过两折全部门槛，不升级默认。最终786项离线测试通过、13联网排除。不是独立盲测或真实成交验收；预注册、旧研究和默认模型不改写。

已完成：[七方向探索摘要](reports/ppo_exploration_20261006_summary.md)、[详细报告](reports/ppo_exploration_20261006.md)及[冻结预登记协议](ppo_exploration_protocol_20261006.md)。全部诊断、四组正式三种子、三个pilot及27快照回测完成，实际104,448 RL步、4,080 Adam更新。行情/账户分离长预算通过研究门槛，配置平均净ROI+2.2784%，但固定seed42模型实际-1.5191%，且未超过20日动量，不发布默认。668项完整离线测试通过、13联网排除；仍是历史回溯研究，保留2026年7月以后的绩效隔离。原协议字节不改写，实际完成和限制统一见报告与实施进度。

已完成：[整手交易与明确保持动作摘要](reports/ppo_trade_actor_20261005_summary.md)、[详细报告](reports/ppo_trade_actor_20261005.md)与[预登记协议](ppo_trade_actor_protocol_20261005.md)。HOLD/BUY/REDUCE/EXIT、冻结与经济整手约束已实现；126,343参数、三种子98,304正式步及8,192 pilot步、6快照、主K20和辅助K10/40四账户开发回测均完成。主K20短/长平均净ROI为-2.5182%/-6.2879%，对自身初始化有平均改善但没有合格候选，不晋升默认。费用进入模拟净值奖励一次，Critic没有另加微单处罚；执行验收不等于收益或现实成交验收。后续稀疏交易与信用分配建议见报告，尚未实施。

已完成：[股票增量Actor研究摘要](reports/ppo_delta_actor_20261004_summary.md)、[详细报告](reports/ppo_delta_actor_20261004.md) 与 [预登记协议](ppo_delta_actor_protocol_20261004.md)。股票输出交易增量，现金为剩余，新增冻结/可卖状态及topN条件；三种子196,608正式步、6快照，另8,192步pilot。主K20短/长净收益-0.4163%/-0.3528%，无合格候选，不晋升默认。训练/回测执行已完成，不代表收益目标或真实账户适配完成；旧模型保留旧动作契约。

已完成：[无先验随机初始化二乘二矩阵](ppo_random_matrix_protocol_20261003.md)，新增6次训练、12快照及393,216步，逐模型/种子评估随机初始化并与旧四格合并比较。没有随机配置通过研究门槛；八格中观察到的最佳PPO仍为先验小模型短预算，且未超过初始化及强规则基线。见 [摘要](reports/ppo_random_matrix_20261003_summary.md) 与 [详细报告](reports/ppo_random_matrix_20261003.md)，含KL限制、成本、风险和代表模型；不晋升实盘默认。

已完成：[训练预算与网络容量二乘二矩阵](ppo_matrix_protocol_20261003.md)，沿用新浪300股，6次训练、12个快照、393,216个正式步。训练增加8倍、参数增加约61倍，未得到稳定收益增量。见 [摘要](reports/ppo_matrix_20261003_summary.md) 与 [详细报告](reports/ppo_matrix_20261003.md)，含经历覆盖、配对效应及诊断缺陷处置；未评估保留测试期或晋升实盘默认。

全仓库审查8项修复已完成：[修复与验收](repository_audit_fixes.md)。含训练/推理掩码统一、下载来源、队列竞争、评估契约、连接释放及浏览器回归；真实300股单日推理已通过一致性对照。

首版已实现：[联网行情与多模型预测整合](market_model_prediction_plan.md)，新浪默认日线、图表信号、多模型研究意图与已存验证表现；连续回放和真实账户建议仍待完成。见 [验收记录](reports/market_model_integration_20260927.md)。

最新研究：[稀疏与先验残差Agent摘要](reports/agent_improvement_20260927_summary.md)、[详细报告](reports/agent_improvement_20260927.md)。找到相对旧PPO收益更高的研究候选，但主要增益来自先验，未证明PPO学习优势，未实盘发布。

新增目标：[每日收盘股票池配置计划](eod_allocation_plan.md)；最新研究：[大网络 CPU/GPU 对照](reports/ppo_capacity_20260927.md)。

当前执行协议：[新浪沪深300与多初始持仓](sina_hs300_protocol_20260927.md)，包含指定300股名单、来源一致性和训练/验证日期边界。

该轮已完成：[新浪沪深300报告摘要](reports/sina_hs300_20260927_summary.md) 与 [详细报告](reports/sina_hs300_20260927.md)。模型训练成功但未证明策略优势，未晋升默认。

更新日期：2026-10-16

建议先读 [项目 Pipeline](../pipeline.md)，再看 [实施进度与验收记录](implementation_plan.md)。阶段四已完成框架及 32 只股票的轻量训练和冻结测试，但尚未证明策略优势。最新结果见 [报告摘要](reports/ppo_lightweight_20260925_summary.md) 和 [详细报告](reports/ppo_lightweight_20260925.md)。

最新增量研究为 19 股、3 配置 × 3 种子的开发验证，见 [多种子摘要](reports/ppo_multiseed_20260926_summary.md) 和 [详细报告](reports/ppo_multiseed_20260926.md)。它与原 32 股实验条件不同，不直接作升级收益比较，也没有重新运行最终测试。

## 文档分工

| 文档 | 负责内容 | 何时阅读 |
| --- | --- | --- |
| [pipeline.md](../pipeline.md) | 当前总体架构、数据流、关键边界与未来候选方向 | 了解项目在做什么 |
| [implementation_plan.md](implementation_plan.md) | 唯一的阶段进度总表、任务勾选、验收证据和下一步 | 查看已完成与未完成事项 |
| [research_workflow.md](research_workflow.md) | 数据产物、预处理和基线回测运行方式 | 重建本地数据或复现基线 |
| [trading_environment.md](trading_environment.md) | 环境输入、动作、账户、成交、事件及约束契约 | 接入策略或修改记账逻辑 |
| [ppo_implementation_plan.md](ppo_implementation_plan.md) | PPO 智能体输入与行为、动作映射、共享网络、扩池和训练设计 | 理解 PPO 为什么这样实现 |
| [ppo_training_framework.md](ppo_training_framework.md) | 实际依赖、检查/训练/冻结测试命令、产物和操作限制 | 运行与排查 PPO 框架 |
| [ppo_lightweight_protocol.md](ppo_lightweight_protocol.md) | 轻量实验预定股票池、预算和测试规则 | 核对是否按预定方案执行 |
| [报告摘要](reports/ppo_lightweight_20260925_summary.md)、[详细报告](reports/ppo_lightweight_20260925.md) | 实际训练审计、验证与最终测试结果、局限 | 阅读本次实验结论 |
| [frontend_workbench_design.md](frontend_workbench_design.md) | 数据选择、训练监控、模型管理、回测和动作预测的设计基线 | 理解交互和契约；首版已实现，扩展项另行标记 |
| [workbench_runbook.md](workbench_runbook.md) | 前端构建、后端启动、任务和账户使用边界 | 运行本地研究工作台 |
| [gpu_environment.md](gpu_environment.md) | AShareGPU 独立环境、驱动兼容、设备选择与验证 | 使用 RTX 3050 运行框架；Graduate 保留供旧实验使用 |
| [工作台验收记录](reports/workbench_acceptance_20260925.md) | 实际浏览器测试、修复问题、性能优化与截图 | 核对前端实现和剩余限制 |
| [online_data_live_decision_plan.md](online_data_live_decision_plan.md) | 联网快照、新日线推理及价格/时效边界 | 设计基线；部分能力已接入，真实连通与账户适配仍受限 |
| [online_market_runbook.md](online_market_runbook.md) | 显式下载、日线研究计划、报价监控和停止 | 使用联网工作区，识别失败和陈旧数据 |
| [bulk_download_runbook.md](bulk_download_runbook.md) | 500 股批量选取、代码文件导入、进度、取消保留与数据集扩充 | 用完整下载批次构建新数据版本 |
| [training_data_linkage.md](training_data_linkage.md) | 多股训练日期交集、观察窗口、原始行情与下载批次关联 | 理解为何某些股票不能进入旧训练区间，以及怎样关联新数据版本 |
| [feature_builder_plan.md](feature_builder_plan.md) | 数据管理追加特征、预览、新版本构建及模型兼容 | 首版已实现；含规划、使用步骤及真实构建验收 |
| [akshare_download_expansion.md](akshare_download_expansion.md) | AKShare 可选数据、五类扩展下载、独立归档和后续候选 | 用户手动采集资料 / 资金流 / 分钟 / 指数 / 分红，不自动进入训练 |
| [training_data_improvement_plan.md](training_data_improvement_plan.md) | 训练数据可得性探测、缺口替代方案与改进建议；设计稿，未实施 | 评估扩充训练数据、改造股票池 / 价格口径或补交易状态时 |
| [data_feature_ppo_integration_plan.md](data_feature_ppo_integration_plan.md) | 数据需求、特征方案、版本发布与 PPO 的统一实验流程 | 下一步集成设计；含准入、任务编排、恢复与落地顺序，尚未实施 |
| [ppo_training_improvement_plan.md](ppo_training_improvement_plan.md) | 超参数、训练监控、模型评价、选模与训练方法改进 | 核心工程已接入；算法候选和实现边界另行标记 |
| [experiment_workbench_runbook.md](experiment_workbench_runbook.md) | 数据与 PPO 实验流程、参数、监控、评价及实现边界 | 已实现核心工程；明确区别于待研究的算法和扩展数据适配 |
| [ppo_multiseed_protocol_20260926.md](ppo_multiseed_protocol_20260926.md) | 19 股、三个配置和三个种子的正式运行协议 | 训练前登记，验证期研究，不重跑最终测试 |
| [market_information_indicator_plan.md](market_information_indicator_plan.md) | 行情信息扩展、手动导入、技术指标计算契约及页面设计 | 待实施设计；区分本地可计算指标与需额外下载的数据 |
| [联网验收记录](reports/online_market_acceptance_20260926.md) | 模拟测试、真实连接失败及未完成项 | 区分代码通过与实时数据实际可用 |

## 统一维护规则

- 总体阶段只使用实施进度中的一至五，不再混用早期模型演进或平台化里程碑编号。
- `[x]` 只表示对应条目已有验收证据；“已实现、待训练验收”不表示已经运行训练，“待数据”不表示接口未实现。
- 阶段状态、测试数量和验收报告索引集中维护在实施进度。设计文档解释契约，运行文档解释命令，不重复维护另一份任务清单。
- 配置值以 `configs/research.json`、`configs/ppo.json` 为准，文档中的值为当前快照。改变配置或行为时同步更新对应文档及验收记录。
- 原始数据、清洗产物、初始化模型和回测报告保留在本地，不纳入 Git。文档中的 `reports/runs/...` 是本地验收证据路径，其他机器须自行按运行说明复现。
- 历史验收数字标记所属阶段；不能把阶段二、三当时的测试数量误当成当前总数。

## 运行入口

所有命令从仓库根目录执行。原 CPU 实验使用 Conda `Graduate`；新 GPU 环境 `AShareGPU` 的安装与验收状态见 [GPU 配置](gpu_environment.md)。依赖安装另见 [研究流程](research_workflow.md) 和 [PPO 运行说明](ppo_training_framework.md)。

```powershell
# 只运行离线测试，默认排除联网测试
python -B -m pytest -q -p no:cacheprovider

# 复现验证集基线，不训练 PPO
python -B -m src.backtest.run --config configs/research.json

# 检查 PPO 接口、保存加载与参数不变，不训练
python -B -m src.training.train --config configs/ppo.json --check-only

# 前端已构建且 Web 依赖齐备时启动本地工作台
python -B -m src.web.serve --port 8765
```

训练必须显式启动；现已有轻量实验的模型和冻结测试报告，默认检查命令仍不训练。最终测试区间已查看，不再用作未见数据调参。当前 `hfq_research` 结果不能等同于现实交易收益。
