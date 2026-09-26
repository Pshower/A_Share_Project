# 项目文档索引

新增目标：[每日收盘股票池配置计划](eod_allocation_plan.md)；最新研究：[大网络 CPU/GPU 对照](reports/ppo_capacity_20260927.md)。

当前执行协议：[新浪沪深300与多初始持仓](sina_hs300_protocol_20260927.md)，包含指定300股名单、来源一致性和训练/验证日期边界。

该轮已完成：[新浪沪深300报告摘要](reports/sina_hs300_20260927_summary.md) 与 [详细报告](reports/sina_hs300_20260927.md)。模型训练成功但未证明策略优势，未晋升默认。

更新日期：2026-09-26

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
