# 试验账本：本仓全部判定型试验的统一清单

更新日期：2026-11-23（首版编制；此后**只增不改**，每份新预登记追加一行）。
编制方式：grep 全部 129 份 doc .md 命中试验计数/变体/轮次表述 → 逐份通读判定型文档；范围只收**有门槛或有结论**的研究，排除纯工具验收、数据下载、运维与文档整理。编制由独立子代理执行、父代理抽查 3 条编号全部核实（fundamental_ic 第 26 次、unlock_ic 第 22 次、supply_overlay 第 22/25 次并记）。
上游：[方法学自审](methodology_selfaudit_20261123.md) §5 第 3 条（"总试验账本散落各文档，从未有统一清单"）、[外部深读](external_deepread_20261123.md) §1（2611.18934：重复种子 ≠ 搜索规模）。

---

## 类别口径（四类 + 一类待增设）

| 类别 | 定义 | 多重性含义 |
| --- | --- | --- |
| **搜索** | 同一已花样本上试多个配置并选优 | 最需要校正的一类 |
| **诊断** | 对同一条规则的事后拆解 / 敏感性 | 不选优，不新增多重性 |
| **预登记对照** | 判据写死在运行前的并行 / 配对判定 | 多重性由预登记承担 |
| **复检** | 同一问题的独立复算 | 降多重性，不升 |
| 未分类\* | 见文末脚注（判据与方向事前写死、但单臂无对照） | 多重性由"第 N 次"序号承担 |

## 账本主体（66 行，按日期分组；archive/ 前缀 = doc/archive/）

**2026-09-27 ~ 10-17（RL 线）**

| 序号或计数 | 出处 | 内容一句话 | 类别 |
| --- | --- | --- | --- |
| 4 方案/11 次训练上限 | archive/agent_improvement_protocol_20260927.md | 预登记四方案筛选，限额防试错 | 搜索 |
| 13 轮（末轮 24 trial） | archive/research_record_20261011.md | RL 训练评测轮次表（方向×配置×种子） | 搜索 |
| 2 个开发对照 | archive/counterfactual_learning_scheduling_20261017.md | PPO-only 冻结对照臂并行执行 | 预登记对照 |

**2026-11-05**

| 4 信号×3 期限 | archive/signal_expansion_protocol_20261105.md + archive/signal_expansion_20261105.md | 新源预登记筛查，市值过后又否 | 搜索 |
| 0（判据冻结） | archive/exposure_control_plan_20261105.md | 样本账本 D/V/R/P 与 L1–L4 判据 | 预登记对照 |
| 2 机制 | archive/exposure_stage1_20261105.md | L1 过、L2 出局 | 未分类\* |
| 1 主判+4 变体 | archive/exposure_stage2_20261105.md | L1 缩放 V 段一次性判定未过 | 预登记对照 |

**2026-11-06**

| 10 敞口；5 族未执行 | archive/alpha_program_20261106.md | 定仓不改 Sharpe；alpha 判据被功效阻断 | 诊断 |
| 3 注入对照+7 信号 | archive/alpha_measure_20261106.md | IC 尺子标定，V 段试跑不作证据 | 诊断 |
| 20 特征×2 窗×2 期 | archive/price_feature_ic_20261106.md | 价格特征 IC 重测：量级够、确认不了 | 搜索 |
| 20 特征×5 期限 | archive/autocorrelation_20261106.md | 找到 1 日截面反转结构（描述性） | 诊断 |

**2026-11-07**

| 24 组+滑点 6 档 | archive/reversal_backtest_20261106.md | 1 日反转被成本吃光，方向关闭 | 搜索 |
| 1 次（E1 go/no-go） | archive/research_direction_20261107.md | E1 预登记：判据写死、算术先定价 | 预登记对照 |
| 6 策略×2 窗 | archive/baseline_capm_20261107.md | α 归因：赢的是选样偏差不是技能 | 诊断 |
| 12 run（2 组×3 种子） | archive/prior_attribution_20261107.md | 训练从未改动先验排序 | 诊断 |

**2026-11-08**

| T1/T2/T3+置换 30 | archive/capm_protocol_20261107.md + archive/capm_full_market_20261108.md | CAPM 三检验判定：无法区分 | 预登记对照 |
| "不做第七次" | archive/e1_reversal_20261108.md | E1 不过，关闭价格特征族 | 预登记对照 |
| 同族第 2 次+随机 200 | archive/e2_protocol_20261108.md + archive/e2_result_20261111.md | E2 换池三判据全败，永久关闭 | 预登记对照 |
| 8 源候选 | archive/information_sources_20261108.md | 信息源门槛 t≥2.5/IC 0.031 | 预登记对照 |
| 第 22 次 | archive/unlock_ic_20261108.md | 解禁 IC 过门槛，仅够排除/风控 | 未分类\* |
| 第 23–24 次 | archive/information_sources_result_20261108.md | 预告不过；资金流无历史出局 | 未分类\* |
| 20 取前 5+随机 30 | archive/feature_backtest_20261108.md | top5 特征书样本外与随机无差 | 搜索 |
| S1 1 次（前七次失败） | archive/research_direction_20261108.md | S1 预登记；转向供给侧事件 | 预登记对照 |
| 第 8 次/第 25 次 | archive/supply_overlay_20261108.md | S1 剔解禁 t 2.25 未过，关族 | 预登记对照 |
| 4 指数+1 组合 | archive/allocation_study_20261108.md | 三 ETF 跟踪组合拟合，差 2.7pp | 搜索 |
| 5+4+2=11 变体 | archive/allocation_robustness_20261108.md | 再平衡无用、现金有效、波动目标否 | 搜索 |
| 3 口径 | archive/tail_exposure_20261108.md | 微盘尾缺口主因是北交所缺失 | 诊断 |
| 6 配置+2 样本外窗 | archive/drawdown_target_20261108.md | 7%/10% 仅一配置达标靠上界假设 | 搜索 |
| 5 档×2 口径×3 股息 | archive/dividend_sleeve_20261108.md | 红利五档单调荐 40%（后部分撤回） | 搜索 |
| 5 档×2 指数 | archive/dividend_sleeve_crosscheck_20261108.md | 换国证红利复算：降回撤不成立 | 复检 |
| 4 档+5 宽度×2 | archive/equal_weight_crosscheck_20261108.md | 指数对照推翻面板等权溢价 3.99pp | 复检 |
| 随机 30 次 | archive/satellite_decision_20261108.md | 个股账超额来自池子，信号贡献零 | 诊断 |
| 4 窗+滚动 5 年 | archive/return_context_20261108.md | 5% 是窗口属性；2010-26 仅 0.82% | 诊断 |
| 随机 30 次 | archive/selection_diagnostics_20261108.md | 基准规范与随机对照演示 | 诊断 |
| 第 26 次/4 因子 | archive/fundamental_ic_20261108.md | bp 过门槛但是风格暴露非 alpha | 未分类\* |
| 2 口径/0.42 年 | archive/fund_flow_20261108.md | 不冗余但样本不够判 IC | 未分类\* |

**2026-11-09**

| 5 倍数×4 窗 | cppi_strategy_20261109.md | CPPI 平台 5-7 达标，选 6 | 搜索 |
| 4 窗×5 倍数=20 组 | archive/cppi_gap_risk_20261109.md | m=8 击穿 6 次，上限收到 7 | 搜索 |
| 5 危机×2 情景 | archive/cppi_crisis_20261109.md | 五次危机全守楼层，代价恢复慢 | 诊断 |
| 5×2×4 | archive/cppi_bond_sensitivity_20261109.md | 债券 3.0% 以下 7% 即失败 | 诊断 |
| 231 组合×4 窗 | archive/global_frontier_20261109.md | 无一固定配比跨窗稳健 | 搜索 |

**2026-11-10**

| 6 组合×2 carry | archive/gold_substitution_20261110.md | 黄金替代无效；暴露窗口选择偏差 | 搜索 |
| 68 起点×2 carry | archive/cppi_start_dates_20261110.md | 回撤承诺 68/68，7% 承诺不成立 | 诊断 |

**2026-11-11**

| 第 4 源 | archive/lhb_ic_20261111.md | 龙虎榜方向对但 t 0.83 区分不了 | 未分类\* |
| 第 5 源 | archive/gdhs_ic_20261111.md | 六年同号但 IC 差 9% 不过 | 未分类\* |
| 第 6 候选 | archive/margin_ic_20261111.md | 融资融券首个过预登记门槛 | 未分类\* |
| 第 7 源 | archive/perf_ic_20261111.md | 业绩报表覆盖 99%，效应近零 | 未分类\* |
| 3 判据+随机 200 | archive/margin_portfolio_protocol_20261111.md + archive/margin_portfolio_result_20261111.md | 组合层三判据全败，关多头用法 | 预登记对照 |
| 97 配对块 | archive/crowding_overlay_protocol_20261111.md + archive/crowding_overlay_result_20261111.md | 拥挤度规避 overlay t 1.29 不过 | 预登记对照 |

**2026-11-12**

| 129 起点×2 频率 | plan_backtest_20261112.md | 真实 ETF 回测：≥7% 仅 3.9%（两轮更正） | 诊断 |

**2026-11-14**

| 10 方向/24 run | ppo_value_20261114.md | PPO 线十方向全关，留方法教训 | 搜索 |
| 4K×2 节奏=8 变体 | e1_topk_20261114.md | 无 K 赢等权，K 是风险旋钮不采纳 | 诊断 |
| 2 节奏对照 | shadow_cadence_20261114.md | 影子改 20 日一判，判据先写后看 | 预登记对照 |
| 3 残值档×2 宇宙 | survivorship_20261114.md | 退市上界 0.54-0.78pp 非主误差 | 诊断 |
| 4 闸门规则 | etf_premium_20261114.md | 溢价闸门全否，改 p95 熔断建议 | 搜索 |
| 4 判据×3 数据版 | market_leverage_protocol_20261114.md + market_leverage_result_20261114.md | 比率版 failed，关闭该族 | 预登记对照 |
| 4 判据，守卫 void | market_leverage_growth_protocol_20261114.md + market_leverage_growth_result_20261114.md | 增速版守卫定错阈值，void 关族 | 预登记对照 |
| 6 本金×6 带宽 | plan_sensitivity_20261114.md | 本金不敏感；带宽 3-5% 平台不调 | 诊断 |

**2026-11-15**

| 12 规则×5 判据=45 | unified_strategy_protocol_20261115.md + unified_strategy_20261115.md | 统一多资产 0/12，长样本守不住 10% | 预登记对照 |
| 8 规则+2 对照 | credit_sleeve_20261115.md | 信用债 +9bp 不成立；久期一次性 | 预登记对照 |
| 9 ETF 载体 | fund_vehicles_20261115.md | 载体差 0.35-1.22pp/年，费率上调 | 诊断 |
| 96 组+18 次 | feature_audit_20261115.md | 三族筛查全否；量价结论后作废 | 搜索 |

**2026-11-16**

| 3 执行时点×129 起点 | investment_advice_review_20261116.md | 次日开盘影响在噪声内 | 诊断 |

**2026-11-18**

| 4 册（A–D） | shadow_variants_20261118.md | 四种冻结权重并行影子，60 日首判 | 预登记对照 |

**2026-11-23**

| 5 臂×129 起点 | equity_bond_tail_protocol_20261123.md + equity_bond_tail_20261123.md | 优化臂赢 Sharpe 实为全债退化 | 预登记对照 |
| 3 账本 | shadow_alloc_20261123.md | minvar/cvar 前向影子，252 日首判 | 预登记对照 |

## 汇总与计数链

- **批次（66 行）**：搜索 **17** · 诊断 **18** · 预登记对照 **20** · 复检 **2** · 未分类 **9**。
- **最大累计计数 = 26**（[fundamental_ic](archive/fundamental_ic_20261108.md) §7，"第 26 次检验"）。计数链：20 价格特征 + E1 + 特征筛选 → **解禁 = 第 22 次** → 业绩预告 = 23 → 资金流 = 24 → S1 = 25 → 基本面 = 26。**此后链条冻结**：lhb/gdhs/margin/perf 改用"第 4–7 源"候选计数；feature_audit、e1_topk、equity_bond_tail 等后续判定均未领"第 27 次"及以后的序号。
- **编号的三处不整齐（照实入档，不回改原文）**：
  1. 第 1–21 次从未逐条编号，只被"20 价格特征 + E1 + 特征筛选"打包引用；
  2. **第 21/22 号重号**：[supply_overlay](archive/supply_overlay_20261108.md) 把 E1 记作第 22 次，而 [unlock_ic](archive/unlock_ic_20261108.md) 的算术（20+1+1 之后）里 E1 应为 21；同一份 supply_overlay 内 S1 又同时被记作"第 8 次"（按研究线）与"第 25 次"（按单次检验），两套口径并存未注明；[final_plan](final_plan_20261109.md) 的"26 次（5+20+1+1+1）"括号加总实为 28；
  3. 链条冻结后，**真正在已花样本上做多配置选优的批次（配置/CPPI 族的 11 变体、20 组、231 组合）从头到尾没有"第 N 次"序号**——它们在本表按类别如实记为"搜索"，多重性披露靠本表而非原文。

## 三条编制备注

1. **最易被误记成"搜索"的**：RL 线的"13 轮/24 run"里相当部分是**同配置换种子**（2611.18934 口径：重复种子 ≠ 搜索），真正构成搜索的是十个方向本身（[ppo_value](ppo_value_20261114.md) 的"别再试"清单）；反向提醒：feature_audit 的"96 组"是目标函数 bug 后被作废的搜索，只看计数会高估其披露价值。
2. **盲区**：只读了 doc 与 doc/archive 的 .md；`reports/` 产物不可核（PPO 线 11.94 GB 产物已误删，RL 各轮内部 run/seed 结构只能照抄 research_record 的轮次表）；约 60 份运维/工程/数据文档经标题+grep 判型后未逐字通读。
3. **未分类\*（9 行）**：均为**判据与方向事前写死的单臂阈值判定**（无并行对照、无配置选择）——四类均不贴切故不硬塞；下次修订可增设第五类"预登记单臂判定"。

## 维护规则

- **只增不改**：每份新预登记在开跑前于本表追加一行（序号或计数 / 出处 / 一句话 / 类别）；已入档行不回改，原文的编号不整齐保留为历史证据。
- 类别必须在**运行前**指定；事后改类别 = 新开预登记。
- 本表是 [2611.18934](https://arxiv.org/abs/2611.18934) 六行标准第 4 行（"搜索过多少配置"）在本仓的常设答案。
