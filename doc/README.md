# 项目文档索引

更新日期：2026-10-24。

建议按以下顺序阅读：

1. [项目 Pipeline](../pipeline.md)：当前架构、数据流和关键边界。
2. [实施进度](implementation_plan.md)：唯一的当前状态、验收结论和剩余工作入口。
3. [本地研究流程](research_workflow.md)或相应 runbook：实际复现命令。
4. 下表中的协议与报告：核对某一轮研究的预定规则、实际结果和失败原因。

索引只保留一句话状态，不复制研究的逐步数字。当前没有经确认门槛认定可部署的新模型，最新结论见[实施进度](implementation_plan.md)。

## 核心文档与运行手册

| 文档 | 职责 |
| --- | --- |
| [pipeline.md](../pipeline.md) | 当前总体架构、数据流、训练评价边界和未实施扩展 |
| [implementation_plan.md](implementation_plan.md) | 当前阶段状态、最新结论、下一步和维护规则 |
| [research_workflow.md](research_workflow.md) | 数据产物、预处理、基线回测和本地复现 |
| [trading_environment.md](trading_environment.md) | 环境输入、动作、账户、成交、公司行为和约束契约 |
| [ppo_implementation_plan.md](ppo_implementation_plan.md) | PPO 智能体、动作映射、共享网络、扩池和训练设计 |
| [ppo_training_framework.md](ppo_training_framework.md) | PPO 依赖、检查/训练/冻结测试命令和产物说明 |
| [workbench_runbook.md](workbench_runbook.md) | 本地工作台启动、页面、任务、账户和测试方式 |
| [experiment_workbench_runbook.md](experiment_workbench_runbook.md) | 实验配置、分阶段批准、多种子、风险选模和当前工程边界 |
| [frontend_workbench_design.md](frontend_workbench_design.md) | 工作台交互与 API 设计基线 |
| [gpu_environment.md](gpu_environment.md) | AShareGPU/Graduate 环境、设备选择与验证 |
| [online_market_runbook.md](online_market_runbook.md) | 显式下载、日线研究计划、报价监控和分来源限制 |
| [bulk_download_runbook.md](bulk_download_runbook.md) | 批量选取、代码导入、进度、取消、重试和数据集扩充 |
| [akshare_download_expansion.md](akshare_download_expansion.md) | 资料、资金流、分钟、指数和分红的独立归档；不自动入模 |
| [training_data_linkage.md](training_data_linkage.md) | 多股日期交集、lookback、原始行情与下载批次关联 |
| [feature_builder_plan.md](feature_builder_plan.md) | 追加指标、预览、新版本构建、模型兼容和验收 |

## 设计、核查与综合记录

这些文档保留设计时的判断；其中“待实施”不等于当前状态。当前完成情况始终回到[实施进度](implementation_plan.md)。

| 文档 | 内容与使用边界 |
| --- | --- |
| [training_data_improvement_plan.md](training_data_improvement_plan.md) | 数据可得性探测、改进建议、清理记录、模型/信息核查及因子基线结果 |
| [data_feature_ppo_integration_plan.md](data_feature_ppo_integration_plan.md) | 数据—特征—PPO 的一体化设计；已实现行为以实验工作区手册为准 |
| [ppo_training_improvement_plan.md](ppo_training_improvement_plan.md) | 超参数、监控、评价和训练方法的设计基线及后续候选 |
| [market_information_indicator_plan.md](market_information_indicator_plan.md) | 行情信息、手动导入和技术指标扩展设计 |
| [eod_allocation_plan.md](eod_allocation_plan.md) | 每日收盘股票池配置目标、数据闸门和收盘边界设计 |
| [market_model_prediction_plan.md](market_model_prediction_plan.md) | 联网行情与多模型预测整合的设计和历史进度 |
| [online_data_live_decision_plan.md](online_data_live_decision_plan.md) | 联网快照、新日线推理、价格和时效边界的设计基线 |
| [repository_audit_fixes.md](repository_audit_fixes.md) | 仓库审查、修复和 300 股一致性验收证据 |
| [research_record_20261011.md](research_record_20261011.md)（[PDF](research_record_20261011.pdf)） | 2026-09-25 至 2026-10-11 的综合研究记录 |
| [ppo_timing_followup_design_20261015.md](ppo_timing_followup_design_20261015.md) | 调仓研究后续设计，保留其历史假设 |
| [market_summary_research_notes_20261007.md](market_summary_research_notes_20261007.md) | 市场摘要后续候选与实现边界 |
| [counterfactual_learning_scheduling_20261017.md](counterfactual_learning_scheduling_20261017.md) | 因果动作价值研究的调度与重入约束 |
| [critic_initialization_probe_20261017.md](critic_initialization_probe_20261017.md) | 独立 Critic 初始化/裁剪探针的工程验收，不是金融结果 |

## 研究协议与证据

摘要/详细报告对是不同阅读层级，不是重复垃圾；失败、部分完成和被修复研究承接的记录同样是证据。冻结协议保留登记时正文，实际完成状态只由报告和[实施进度](implementation_plan.md)表达。

| 日期 | 研究 | 协议/设计 | 结果与证据 | 结论 |
| --- | --- | --- | --- | --- |
| 2026-10-23 | 独立 Ridge 因子基线 | 固定超参探索，无独立预注册 | [第 11.9 节实现与两折结果](training_data_improvement_plan.md#119-p1-首版实现与两折实测)；`tools/factor_baseline.py` | 两折弱于动量，不产生候选 |
| 2026-10-17/18 | 因果动作价值预热与 PPO | [协议](ppo_counterfactual_learning_protocol_20261017.md) | [runlog](reports/ppo_counterfactual_learning_20261017_runlog.md)、[A 臂诊断](reports/ppo_counterfactual_learning_20261017_control_diagnostic.md) | 确认未合格，无候选 |
| 2026-10-16 | 训练期配对动作诊断 | [协议](ppo_counterfactual_pilot_protocol_20261016.md) | [摘要](reports/ppo_counterfactual_pilot_20261016_summary.md)、[详细报告](reports/ppo_counterfactual_pilot_20261016.md) | 诊断完成，不证明可学性或 ROI |
| 2026-10-15/16 | 调仓 phase 状态修复 | [协议](ppo_timing_phase_state_repair_protocol_20261015.md) | [摘要](reports/ppo_timing_phase_state_repair_20261015_summary.md)、[详细报告](reports/ppo_timing_phase_state_repair_20261015.md) | 筛选失败，无候选 |
| 2026-10-15 | 多步收益归因 | [协议](ppo_timing_credit_protocol_20261015.md) | [摘要](reports/ppo_timing_credit_20261015_summary.md)、[详细报告](reports/ppo_timing_credit_20261015.md) | 局部增量，未准入 |
| 2026-10-15 | 原 phase/预算运行 | [协议](ppo_timing_phase_budget_protocol_20261015.md) | [失败与过程报告](reports/ppo_timing_phase_budget_20261015.md) | 原运行失败，由修复研究承接 |
| 2026-10-14 | 调仓初始化强度 | [协议](ppo_timing_margin_protocol_20261014.md) | [摘要](reports/ppo_timing_margin_20261014_summary.md)、[详细报告](reports/ppo_timing_margin_20261014.md) | 行为改变但无跨折改善 |
| 2026-10-14 | 规则锚定调仓控制 | [协议](ppo_timing_protocol_20261014.md) | [摘要](reports/ppo_timing_20261014_summary.md)、[详细报告](reports/ppo_timing_20261014.md) | 确认未胜 M20，不部署 |
| 2026-10-11 | 排序动作与 rank 辅助 | [协议](ppo_priority_rank_protocol_20261011.md) | [摘要](reports/ppo_priority_rank_20261011_summary.md)、[详细报告](reports/ppo_priority_rank_20261011.md) | 所有配置未过门槛 |
| 2026-10-07 | 纯行情摘要优化 | [协议](ppo_market_summary_protocol_20261007.md) | [摘要](reports/ppo_market_summary_20261007_summary.md)、[详细报告](reports/ppo_market_summary_20261007.md) | 无稳定摘要增益 |
| 2026-10-07 | 市场摘要与 Critic 归因 | [协议](ppo_attribution_protocol_20261007.md) | [摘要](reports/ppo_attribution_20261007_summary.md)、[详细报告](reports/ppo_attribution_20261007.md) | 无稳定改进，无候选 |
| 2026-10-06 | 七方向探索 | [协议](ppo_exploration_protocol_20261006.md) | [摘要](reports/ppo_exploration_20261006_summary.md)、[详细报告](reports/ppo_exploration_20261006.md) | 观察最佳仍弱于动量 |
| 2026-10-05 | 整手交易与保持动作 | [协议](ppo_trade_actor_protocol_20261005.md) | [摘要](reports/ppo_trade_actor_20261005_summary.md)、[详细报告](reports/ppo_trade_actor_20261005.md) | 无合格候选 |
| 2026-10-04 | 股票增量 Actor | [协议](ppo_delta_actor_protocol_20261004.md) | [摘要](reports/ppo_delta_actor_20261004_summary.md)、[详细报告](reports/ppo_delta_actor_20261004.md) | 未超过旧先验 |
| 2026-10-03 | 随机初始化八格比较 | [协议](ppo_random_matrix_protocol_20261003.md) | [摘要](reports/ppo_random_matrix_20261003_summary.md)、[详细报告](reports/ppo_random_matrix_20261003.md) | 随机配置未过门槛 |
| 2026-10-03 | 容量与预算矩阵 | [协议](ppo_matrix_protocol_20261003.md) | [摘要](reports/ppo_matrix_20261003_summary.md)、[详细报告](reports/ppo_matrix_20261003.md) | 扩规模无稳定收益增量 |
| 2026-09-27 | 稀疏与先验残差 Agent | [协议](agent_improvement_protocol_20260927.md) | [摘要](reports/agent_improvement_20260927_summary.md)、[详细报告](reports/agent_improvement_20260927.md) | 增益主要来自先验，未发布 |
| 2026-09-27 | 新浪 300 股与多初始持仓 | [协议](sina_hs300_protocol_20260927.md) | [摘要](reports/sina_hs300_20260927_summary.md)、[详细报告](reports/sina_hs300_20260927.md) | 训练完成，未证明优势 |
| 2026-09-27 | 大网络 CPU/GPU 容量对照 | — | [报告](reports/ppo_capacity_20260927.md) | 历史容量证据 |
| 2026-09-27 | 联网行情与模型整合 | [设计](market_model_prediction_plan.md) | [验收报告](reports/market_model_integration_20260927.md) | 首版整合验收 |
| 2026-09-26 | 19 股多种子实验 | [协议](ppo_multiseed_protocol_20260926.md) | [摘要](reports/ppo_multiseed_20260926_summary.md)、[详细报告](reports/ppo_multiseed_20260926.md) | 工程完成，未胜等权 |
| 2026-09-26 | 联网工作区 | [设计](online_data_live_decision_plan.md) | [验收报告](reports/online_market_acceptance_20260926.md) | 离线/页面通过，真实连接受限 |
| 2026-09-25 | 轻量 PPO 与冻结测试 | [协议](ppo_lightweight_protocol.md) | [摘要](reports/ppo_lightweight_20260925_summary.md)、[详细报告](reports/ppo_lightweight_20260925.md) | 流程完成，未优于等权 |
| 2026-09-25 | 本地工作台 | [设计](frontend_workbench_design.md) | [验收报告](reports/workbench_acceptance_20260925.md) | 首版浏览器闭环验收 |

## 维护与运行

- 当前阶段、结论、下一步和完整维护规则只在[实施进度](implementation_plan.md)；架构只在[Pipeline](../pipeline.md)。历史协议、失败报告和 runlog 不因过时或 negative result 而删除。
- 复现、检查、训练和工作台命令分别见[本地研究流程](research_workflow.md)、[PPO 运行说明](ppo_training_framework.md)和[工作台说明](workbench_runbook.md)；环境见[GPU 配置](gpu_environment.md)。
- 原始数据、模型和报告留在本地，`reports/runs/...`、`data/...`、`web/test-results/...` 不随 Git 分发，其他机器须按运行说明复现。历史报告中的截图路径若不存在，属于未入库的本地测试产物，不表示报告结论或验收记录缺失。
