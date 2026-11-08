# 项目文档索引

更新日期：2026-11-08。

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
| [data_inventory.md](data_inventory.md) | 本地数据资产清单：股票池的市场与行业、量价、基本面、宏观、面板边界和已知缺口 |
| [trading_environment.md](trading_environment.md) | 环境输入、动作、账户、成交、公司行为和约束契约 |
| [ppo_implementation_plan.md](ppo_implementation_plan.md) | PPO 智能体、动作映射、共享网络、扩池和训练设计 |
| [ppo_training_framework.md](ppo_training_framework.md) | PPO 依赖、检查/训练/冻结测试命令和产物说明 |
| [workbench_runbook.md](workbench_runbook.md) | 本地工作台启动、页面、动作类型准入、任务、账户和测试方式 |
| [experiment_workbench_runbook.md](experiment_workbench_runbook.md) | 实验配置、分阶段批准、多种子、风险选模和当前工程边界 |
| [frontend_workbench_design.md](frontend_workbench_design.md) | 工作台交互与 API 设计基线 |
| [gpu_environment.md](gpu_environment.md) | AShareGPU/Graduate 环境、设备选择与验证 |
| [online_market_runbook.md](online_market_runbook.md) | 显式历史下载、新浪/东财实时报价、日线研究计划、监控、PIT 可用日、可交易性账本和分来源限制 |
| [daily_advice_program.md](daily_advice_program.md) | 开盘前趋势分析与个股买卖建议：引擎、证据门槛、当前输出和边界；含两折曲线在工作台的查看方式 |
| [exposure_control_plan_20261105.md](exposure_control_plan_20261105.md) | 仓位控制的分阶段规划与预登记：指数检验、三层决策、样本账本和集成方式 |
| [alpha_program_20261106.md](alpha_program_20261106.md) | beta 底座、仓位偏好与 alpha 计划：IC 的经济门槛、检验功效算术、候选族与预登记判据 |
| [alpha_measure_20261106.md](alpha_measure_20261106.md) | alpha 度量机件：时点对齐、Newey-West、注入自检，以及"经济门槛落在噪声里"的功效下限 |
| [ppo_excess_20261106.md](ppo_excess_20261106.md) | 基准超额奖励 + 每个 checkpoint 回测；强制满仓映射；整手不可分这个混淆 |
| [price_feature_ic_20261106.md](price_feature_ic_20261106.md) | 20 个价格特征在新框架下的重测：量级站得住、显著性站不住、方向全部为负 |
| [autocorrelation_20261106.md](autocorrelation_20261106.md) | 自回归结构诊断：时序是轻微动量，截面 1 日反转很强且 T+1 可执行，但卡在成本上 |
| [reversal_backtest_20261106.md](reversal_backtest_20261106.md) | 1 日反转的成本感知回测：毛额 +520%，但零滑点下法定费率已把它压到 +69%，全程输给等权池 |
| [research_direction_20261107.md](research_direction_20261107.md) | 研究方向复核：持有期 × 成本 × 效应量的算术，为什么 1 日算术关闭而 20 日还有余量，以及唯一该做的实验与停止清单 |
| [prior_attribution_20261107.md](prior_attribution_20261107.md) | R1 动量先验的只读归因：残差只有先验振幅的 0.01~0.53%，排序 12 个 run 全部未变；附样本账本修正 A3（R 已被 9 月训练消耗） |
| [baseline_capm_20261107.md](baseline_capm_20261107.md) | 1/N 与 CAPM 的分解：剥掉 β 后七个策略的 α 几乎相同（PPO +17.01% 对等权 +16.82%），α 的水位来自池子偏差而非本事 |
| [capm_protocol_20261107.md](capm_protocol_20261107.md) | 全市场 CAPM 检验的预登记：5,578 只宇宙、市值加权市场组合、T1/T2/T3 判据和四级判定顺序（写在数据到手之前） |
| [capm_full_market_20261108.md](capm_full_market_20261108.md) | 全市场 CAPM 检验（**已完成**）：5,056 只、覆盖 98.50% 流通市值，自建市值加权市场组合；β 配对真实（置换 p=0.000）但斜率只到溢价的四成二、判定"无法区分"（合并 t −0.89，2019-2021 单看 t −2.00）；面板 1/N 超额 +3.99pp/年经指数对照证伪 |
| [equal_weight_crosscheck_20261108.md](equal_weight_crosscheck_20261108.md) | 等权溢价的指数对照与自我更正：面板算出 +7.92pp，同档真实指数（000905/000982）只有 +0.59pp；机制是等权拿到个股算术平均（+15.36%）而中位数只有 +6.50%，差额≈σ²/2。**主报告的 3.99pp 因此作废** |
| [equal_weight_review_20261108.md](equal_weight_review_20261108.md) | 等权投资效果复核与后续研究建议：定位静态中位市值选档的前视偏误（改为调仓日滞后市值动态选档后 CSI500 溢价回落至 +0.44pp，与指数 +0.59pp 吻合），拆解全市场 1/N 的微盘暴露，并提出四条后续工具与实验建议 |
| [selection_diagnostics_20261108.md](selection_diagnostics_20261108.md) | 基准的两种角色与 E1 的诊断栏：池内归因基准必须同池同调仓同本金同整手，实盘底座必须是真实指数；工具实测 300 万本金下 30 只留存 32%、299 只留存 65%，且随机 30 只的规模暴露已达 t=11.14 |
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

摘要与详细报告是不同阅读层级，不是重复内容；失败、部分完成和被修复研究承接的记录同样是证据。冻结协议保留登记时正文，实际完成状态只由报告和[实施进度](implementation_plan.md)表达。

| 日期 | 研究 | 协议/设计 | 结果与证据 | 结论 |
| --- | --- | --- | --- | --- |
| 2026-11-08 | 等权溢价的指数对照与机制复核 | 无独立预注册（外部基准校验与复核） | [指数对照](equal_weight_crosscheck_20261108.md)、[复核与建议](equal_weight_review_20261108.md)、`reports/runs/equal_weight_crosscheck/`、`tools/equal_weight_crosscheck.py` | 中证500档静态中位市值选档给出等权溢价 **+7.92pp**，同档真实指数对（`000905`/`000982`）仅 **+0.59pp**（净约 +0.4pp）；复核证明静态中位市值选档使档内期初市值与涨幅负相关（r=−0.370），改为调仓日滞后市值动态选档后溢价降至 **+0.44pp**（大中盘前 1,800 只各档均在 −0.10~+0.44pp），全市场 1/N 的 +3.99pp 实为 57% 权重压在后 2,400 只小微盘股。**主报告 3.99pp 作废**，1/N 做底座的理由收敛为无需估计期望收益与低换手成本 |
| 2026-11-08 | 全市场 CAPM 检验 | [预登记](capm_protocol_20261107.md) | [报告](capm_full_market_20261108.md)、`reports/runs/capm_full/`、`tools/capm_backtest.py`、`tools/full_market_panel.py` | 5,056 只（覆盖流通市值 **98.50%**），自建市值加权组合与 `399317` 相关 **0.995**、跟踪误差 2.61%；置换对照 p=0.000 证明 β 配对真实，但 SML 斜率 γ1=+0.0031（t 0.67）仅市场溢价四成二，与溢价差 t −0.89（2019-2021 单看 **t −2.00**），判定"**无法区分**"；个股 |t|≥2 仅 0.69%；T3 面板等权超额 3.99pp 经指数对照作废 |
| 2026-11-07 | 1/N 与 CAPM 的策略分解 | 无独立预注册（只读回归） | [报告](baseline_capm_20261107.md)、`reports/runs/baseline_capm/`、`tools/baseline_capm.py` | 对已存 32 股七策略做日超额回归：最高 α 属于零决策的 `buy_hold`（+17.23%），加规模因子（`000905−000300`）后 α 反升 0.86~1.00pp，证明水位来自幸存者偏差；剥离 β 后 PPO 与等权 α 无法区分（选择窗 +17.01% 对 +16.82%，测试窗 +14.27% 对 +14.35%），净收益差距在费用而非选股 |
| 2026-11-07 | R1 动量先验的只读归因 | 无独立预注册（只读诊断） | [报告](prior_attribution_20261107.md)、`reports/runs/prior_attribution/`、`tools/prior_attribution.py` | 训练后与先验排序的 Spearman **0.9999997**、top-20 **20/20 完全重合**、残差只有先验振幅 **0.01%~0.53%**（设计上限 6.25%）。9 月那轮**结构上改不动**，不能当作"RL 学不会截面 alpha"的证据；11 月无先验那轮才是真检验。**附修正 A3：R 已被 9 月训练消耗（训练至 2026-03-31、验证至 2026-06-30），不再是干净样本。** |
| 2026-11-07 | 研究方向复核与仓库整理 | [方向复核](research_direction_20261107.md) | [实施进度](implementation_plan.md)、[本地产物目录](../reports/README.md) | 不产生新样本：算术显示 1 日构造净 **−41.8%/年**、20 日 **+5.3%/年**，内点最优约 14 日；盈亏平衡 IC 0.0310 对实测 0.0311，**恰好压线**。据此把下一步收敛为单个 go/no-go 实验（20 日、298 只、20 特征全取负号等权、top 30），并列出停止清单（RL 策略学习、仓位控制、1 日反转、IC 显著性检验） |
| 2026-11-07 | 1 日反转的成本感知回测 | [报告](reversal_backtest_20261106.md) | `reports/runs/reversal_backtest/`（含 `slippage_sweep.png`）、`tools/reversal_backtest.py` | 零成本下 6 年 **+217%~+520%**（等权参考 +108.6%），与 IC 折算同量级；真券商下 **−60%~−80%**；缓冲带只降换手约 10%（1 日反转排名逐日独立）；**滑点扫描（佣金与印花税始终照收）显示整条曲线都在参考线之下**——滑点 0 时最好也只有 **+69%**，法定费率吃掉约 87% 毛收益；滑点 0.10% 时 −71%，0.20% 时 −92% |
| 2026-11-06 | 自回归结构诊断 | [报告](autocorrelation_20261106.md) | `reports/runs/autocorrelation/`、`tools/autocorrelation_probe.py` | 时序上日收益是**轻微动量**（ρ₁ +0.030、t 2.41；VR(5) 1.062、t 2.19）；截面上 1 日**开盘到开盘**反转很强且 **T+1 可执行**：`body_ratio` IC −0.0258（t **5.69**）、**14/20 特征 \|t\|≥2**，full 窗口 −0.0272（t 7.01）；但 2 日就掉到边缘、5 日消失；毛收益 ≈0.139%/日 对一次往返成本 ≈0.31%，**成败取决于换手** |
| 2026-11-06 | 20 个价格特征重测 | [报告](price_feature_ic_20261106.md) | `reports/runs/price_feature_ic/`、`tools/price_feature_ic.py` | 旧结论"`\|IC\| ≤ 0.035`"在量级上成立（full 窗口最大 0.0358）；但 60 日持有期 **0/20 显著**（lag-h），而 lag-5 会声称 4–5 个显著；**13/20 的 \|IC\| 超过经济门槛 0.0048**，所以特征是"确认不了"而非"太弱"；符号在两个窗口、两个持有期上**全部为负**（反转），与 PPO 的动量先验相反 |
| 2026-11-06 | 对齐基准超额的 PPO 小模型 | [报告](ppo_excess_20261106.md) | `reports/runs/ppo_excess_small/`、`reports/runs/ppo_return_small/`、`reports/runs/ppo_full_invest_3m/`、`reports/runs/ppo_excess_3m/`、`reports/runs/ppo_checkpoints_20261106/`、`reports/runs/ppo_checkpoints_3m_20261106/`、`tools/checkpoint_compare.py` | 各 run 均 32 个检查点完成完整回测；现金模式与强制满仓都没跑赢等权池；**先发现整手不可分**——100 万下连等权参考自己都只投 71%、持币 28.8%，本金提到 300 万后现金降到 11%；满仓把现金压下去但毛额仅等于复制池子，费用翻倍（1.91% 对 0.87%） |
| 2026-11-06 | alpha 度量机件与功效下限 | [度量](alpha_measure_20261106.md)、[alpha 计划 5.4](alpha_program_20261106.md#54-预执行检查的两处修正与一个阻断2026-11-06) | `reports/runs/alpha_ic/`、`tools/alpha_factors.py`、`tools/alpha_ic.py` | 三项注入对照全部符合；实测 IC 波动 0.143、重叠使标准误大 3.6–3.8×，经济门槛 0.0048 仅 0.17 SE，**协议以现行判据无法执行** |
| 2026-11-06 | 仓位算术与 alpha 需求 | [alpha 计划](alpha_program_20261106.md) | `reports/runs/alpha_budget/`、`tools/alpha_budget.py` | 定仓不改变超额 Sharpe；70% 敞口需 alpha 1.41pp，折合 20 日 IC 0.0048；组合层检验在剩余样本上没有功效 |
| 2026-11-05 | 仓位控制阶段二：L1 波动率缩放 | [规划](exposure_control_plan_20261105.md) | [报告](exposure_stage2_20261105.md)、`reports/runs/exposure_stage2/` | 未通过（Calmar 0.075 对 0.100），方向关闭，V 段封存 |
| 2026-11-05 | 仓位控制阶段一：机制度量 | [规划](exposure_control_plan_20261105.md) | [报告](exposure_stage1_20261105.md)、`reports/runs/exposure_stage1/` | 波动率可预测（L1 通过），趋势状态口径不达标（L2 出局） |
| 2026-11-05 | 新数据块截面信号筛选 | [协议](signal_expansion_protocol_20261105.md) | [报告](signal_expansion_20261105.md)、`reports/runs/signal_expansion/` | 只有市值通过，但受幸存者偏差污染，不进入候选 |
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
- `reports/runs/` 自身的目录约定、三层保留策略、逐目录占用表和清理命令见[本地产物目录](../reports/README.md)。结论不在那里，在本索引与[实施进度](implementation_plan.md)。
