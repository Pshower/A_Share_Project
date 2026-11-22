# 用 2611.18934 的审计口径清点本仓 40 份判定型研究

日期：2026-11-23
状态：**已完成**。编码由独立子代理执行（只读，40 份逐份读过，每格一条原文短引文或"未提及"），
父代理抽查 3 条编码全部核实（§7）。无工具、无网络；本文即编码表。
上游：[外部研究扫描](external_research_scan_20261122.md)（[arXiv:2611.18934](https://arxiv.org/abs/2611.18934)）。

---

## 1. 为什么做

[外部研究扫描](external_research_scan_20261122.md) 结论 1 写了一句话：**"本仓自己的门槛（真成本、预登记、PIT、
随机对照）高于这批文献的中位水平。"** 这句话当时是**断言**，不是测量。2611.18934 对 91 篇文献做的事
（预登记 rubric → 逐篇编码 → 双盲复核 → 汇总百分比），对本仓自己的 40 份判定型研究做一遍，
断言才能变成数字——包括它**不成立的那部分**。

## 2. 编码口径（七字段，每格须有原文短引文或"未提及"）

`cost`（none / assumed 常数假设 / data 用数据建模）· `window`（检验窗，折月）· `leakage`
（PIT 池 / 信号滞后 / 训练截止后检验 / 时序一致性 / 随机对照 / 独立复检，任一命中即记）·
`trials`（可数变体数 / 是否报告试验序号）· `prereg`（独立 protocol 或文内预登记章节）·
`verdict`（mechanical 写死门槛 / descriptive 描述比较 / none）· `outcome`（通过 / 关闭 / 进影子 / 背景 / 未判定）。

诚实规则与 2611.18934 同源：**文档里没写的字段记"未提及"，不替作者补全**；
声称与正文不符的地方单独列出（§6），不并入主表。

## 3. 汇总：与 91 篇外部文献并排

| 指标 | 91 篇外部 | 本仓 40 份 |
| --- | --- | --- |
| 成本完全不扣 | 51%（有交易测试的 75 篇） | 全口径 **57.5%**（23/40）；可比口径（21 份组合/交易回测）**33%**（7/21） |
| 成本用数据建模 | 9% | 全仓 2.5%（1/40，[etf_premium](etf_premium_20261114.md)，且是成本专项非策略回测）；**组合回测类 0/21** |
| 报告试验次数 / 多重性进判定 | 仅 9% 报告多重性校正 | 可数变体 **40/40**；多重性进门槛或 Bonferroni **9 份（22.5%）** |
| 任一泄漏控制 | ≈2%（51 篇可查者 98% 三条件全无） | 宽口径 **80%**（32/40，含 4 弱命中）；严格 70%；**按外部同款窄口径（训练截止后/前向）只有 4/40 = 10%** |
| 检验窗中位 | 12 个月（1/4 ≤ 6 个月） | **72 个月**（最短约 5 个月，且该研究自认"不够算 IC"拒绝判定） |
| 三件套（扣成本+泄漏控制+试验数） | 5/91 ≈ 5.5% | **14/40 = 35%**（含弱命中；核心无保留 10/40 = 25%） |

**口径警告（先于任何自满）**：外部的 98% 是"不在训练截止后检验 + 不实盘 + 时序不一致"**三条件联合**；
本仓的 80% 是六项**任一命中**，宽得多。两边只在"方向"上可比，不在数值上可比。
真正可比的窄口径一行已经写明：**训练截止后/前向检验，本仓只有 10%，与外部的 2% 是同一量级。**

一句话总账：本仓**赢**在窗口长度、试验报告、多重性；**平**在泄漏控制的宽口径；
**输**在成本真实性——**组合回测 0/21 用数据建模成本，外部还有 9%**；以及外部最狠的批评
（"全部落在已花样本上"）本仓**同样大面积成立**：40 份里 36 份在已花窗口上判定。

## 4. 缺口分布

26 份未达"三件套"者的缺口：**缺 cost 23（最集中）**、缺 prereg 17（严口径）、缺 leakage 8、缺 trials 0。
缺 cost 的 23 份 = 8 份 IC 层（自认"IC 不扣成本，组合层再算"，逻辑成立但要记账）+
**9 份配置/CPPI/背景** + 3 份回归/共线 + 3 份审计/影子。

## 5. 三条自我批评（编码暴露的，不是编码前知道的）

1. **配置/CPPI 族是全仓方法学最弱的一族**（约 10 份：allocation_study、allocation_robustness、
   drawdown_target、dividend_sleeve×2、cppi×4、gold_substitution、tail_exposure）：
   **0 份预登记、多数不扣成本、无一随机对照**；且多份的"达标/推荐"后来被自家后续文档推翻或条件化
   （dividend_sleeve 次日部分撤回；gold_substitution 与 cppi_start_dates 自认窗口选择偏差；
   drawdown_target 的 7%/10% 双承诺不成立）。**判定纪律几乎只长在选股 alpha 线上，
   配置线的质量全靠事后 crosscheck 补救**——而配置线恰恰是现行方案（LEGS/CPPI 参数）的出处。
   → 本日已开始补：[equity_bond_tail_protocol](equity_bond_tail_protocol_20261123.md) 是本仓
   **第一份配置层预登记**（含成本、含判据、含失败后果）。
2. **"门槛高于文献中位"这句话在成本真实性上不成立**。组合回测全部是常数假设
   （10bp 滑点 / 5bp 往返 / 万三），没有一份按成交额或冲击曲线建模成本；
   且 [dividend_sleeve](archive/dividend_sleeve_20261108.md) 自称"同一成本"而表格无成本列、
   [supply_overlay](archive/supply_overlay_20261108.md) 终表引用了更正前的解禁 IC。
   外部 9% 做到的事，本仓 0%。
3. **本仓从未系统性报告过"训练截止后/前向"这一格**。只有 4 份
   （feature_backtest、allocation_study、drawdown_target 样本外窗、shadow_cadence 前向）
   在未用于构造假设的窗口上做判定；其余 36 份全部在已花的 V/full 窗口上判定
   （多数文档诚实标注"已花窗口"，但诚实标注不等于豁免）。另：总试验账本
   （"第 22/25/26 次"）散落各文档，**从未有一份统一清单**——这正是 2611.18934
   报告的"试验次数不可知"在本仓的温和版本。

## 6. 声称与正文不符（编码时撞见的，全部留档）

| 文档 | 不符 |
| --- | --- |
| [dividend_sleeve](archive/dividend_sleeve_20261108.md) | §5 对照表自称"同一成本"，正文无成本列；§6.3 又自认"费率差异没有建模" |
| [supply_overlay](archive/supply_overlay_20261108.md) | §6 终表用解禁 IC **−0.0686（t −5.26）/覆盖 5.1%**，是 [unlock_ic](archive/unlock_ic_20261108.md) 2026-11-09 更正**前**的数字（更正后 −0.0463 / t −2.82 / 4.0%）——成文早一天，但作为"这一族的最终账"与更正记录矛盾 |
| [feature_audit](feature_audit_20261115.md) | 量价"0/96 过门槛"被自家 11-16 审查作废：`close.shift(-h)` 把横截面 Z-score 当前瞻收益，真实 1 日 IC −0.0277（t −6.75），符号相反、量级 3 倍 |
| [capm_full_market](archive/capm_full_market_20261108.md) | T3 的 −3.99pp 已在文内 ⚠️ 自行作废（同档指数 +0.59pp 毛）——诚实处理，留档 |

（unlock_ic 的窗口 bug 更正、plan_backtest 的零股与分红推断两轮更正均为文内自曝，属诚实处理，不列入不符。）

## 7. 审计的审计（本表的可信度边界）

- 编码者与文档作者是不同代理，rubric 先于编码固定；
- 父代理抽查 3 条，**全部核实**：`allocation_robustness` 的三格（cost=assumed、leakage=none、prereg=none）
  与父代理此前独立读过的原文一致；`supply_overlay` 的陈旧终表数字经 grep 逐行定位（其文第 81 行 vs
  unlock_ic 更正行）；`dividend_sleeve` 的"同一成本"声明与 §6.3 自认经 grep 逐行定位；
- 引用的 8 份独立 protocol 文件全部经 glob 核实存在，其中 market_leverage 两份带提交号（d683715、391f354）；
- **未做**的：20% 双盲复核（2611.18934 做了）；单人格编码的系统性偏差不能排除，尤其是"弱命中"的归类。

## 8. 处置与引用方式

1. 本表是**记录，不是拦截**：不据此关闭或重开任何已判定的线。
2. 未来 U 级新线（如"指数 PE 多年期"）的预登记应引用本表第 3 节的四个基线数字
   （成本 data 建模 0/21、前向检验 4/40、多重性 9/40、三件套 14/40），并**逐项写清本研究如何不低于**；
3. 两条已被本日工作直接回应的缺口：配置层预登记（[equity_bond_tail_protocol](equity_bond_tail_protocol_20261123.md)）
   与统一试验账本（待建，见下）。
4. **不做**的：不为 23 份不扣成本的文档补做成本回测——那是"在已花窗口上重跑搜索"，本仓的规则是不回头重开。

## 9. 复现

无命令。编码表全文如下（附录），字段缩写：mech=mechanical，desc=descriptive；引文备查于各文档。

| # | 研究 | cost | 窗口(月) | leakage | trials | prereg | verdict | outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | [price_feature_ic](archive/price_feature_ic_20261106.md) | none（IC 层） | 72 | NW lag；安慰剂仅引用 | 20 特征×2 窗×2 期 | none | mech | 进影子运行 |
| 2 | [fundamental_ic](archive/fundamental_ic_20261108.md) | none（自认"没过成本"） | 72 | PIT 公告日锚 | 第 26 次/4 因子/Bonferroni | 弱 | mech | 保留为背景 |
| 3 | [lhb_ic](archive/lhb_ic_20261111.md) | none | 72 | 滞后 1 日+NW | 第 4 源 | 文内§2 | mech | 关闭 |
| 4 | [gdhs_ic](archive/gdhs_ic_20261111.md) | none | 72 | PIT+NW | 第 5 源 | 文内§2 | mech | 关闭 |
| 5 | [margin_ic](archive/margin_ic_20261111.md) | none（自认） | 72 | 滞后 1 日+NW | 第 6 候选/Bonferroni | 文内§1（commit 51d17a6） | mech | 通过（IC 层） |
| 6 | [perf_ic](archive/perf_ic_20261111.md) | none | 72 | PIT 法定截止+NW | 第 7 源 | 文内§2 | mech | 关闭 |
| 7 | [unlock_ic](archive/unlock_ic_20261108.md) | none（自认） | 72 | PIT 解禁日先验 | 第 22 次 | 文内 | mech | 通过（仅排除/风控） |
| 8 | [information_sources_result](archive/information_sources_result_20261108.md) | none（自认） | ≈72 | PIT | 第 22–24 次 | 清单+文内 | mech | 关闭（资金流出局） |
| 9 | [feature_backtest](archive/feature_backtest_20261108.md) | assumed（万三/印花税/10bp） | 36 | 选择/测试分离+随机 30 | 20 取前 5（自认未校正） | none | desc | 关闭 |
| 10 | [reversal_backtest](archive/reversal_backtest_20261106.md) | assumed（滑点 0.1%） | 72 | 滞后次日开盘 | 24 组合并列 | none | desc | 关闭 |
| 11 | [e1_reversal](archive/e1_reversal_20261108.md) | assumed（同一 broker） | 96 | 随机对照仅引用 | "不做第七次" | research_direction_20261107§3 | mech | 关闭（价格特征族） |
| 12 | [e2_result](archive/e2_result_20261111.md) | assumed（同 E1） | 96 | PIT 动态池+随机 200+p90 | E1→E2+随机 200 | e2_protocol_20261108 | mech（三判据全败） | 关闭（永久） |
| 13 | [margin_portfolio_result](archive/margin_portfolio_result_20261111.md) | assumed | 72（已花 V） | PIT 动态池+随机 200 | 随机 200 | margin_portfolio_protocol_20261111 | mech | 关闭（多头 alpha） |
| 14 | [crowding_overlay_result](archive/crowding_overlay_result_20261111.md) | assumed | 96 | PIT 池+配对 97 块 | 本线第 3 用法 | crowding_overlay_protocol_20261111 | mech | 关闭（规避用途） |
| 15 | [allocation_study](archive/allocation_study_20261108.md) | none（"费率未核实"） | 72 | 拟合/检验分离 | 4 候选+1 组合 | none | desc | 保留为背景 |
| 16 | [allocation_robustness](archive/allocation_robustness_20261108.md) | assumed（往返 5bp） | 72 | none | 11 变体 | none | desc | 通过（定案现金工具） |
| 17 | [tail_exposure](archive/tail_exposure_20261108.md) | none | 72 | PIT 滞后 1 日 | 3 口径 | none | desc | 保留为背景 |
| 18 | [dividend_sleeve](archive/dividend_sleeve_20261108.md) | none（声称/正文不符） | 72 | none（自认前视） | 5 档×2×3；"前面 26 次" | none | desc | 通过（次日部分撤回） |
| 19 | [dividend_sleeve_crosscheck](archive/dividend_sleeve_crosscheck_20261108.md) | none | 72 | 独立复检 | 5 档×2 指数 | none | desc | 关闭（撤回 40% 建议） |
| 20 | [equal_weight_crosscheck](archive/equal_weight_crosscheck_20261108.md) | assumed（31bp 往返） | 72 | PIT 重选+外部指数校验 | 4 档+5 宽度×2 | none | desc | 保留为背景（3.99pp 作废） |
| 21 | [gold_substitution](archive/gold_substitution_20261110.md) | none | 96 | none（自认挑窗） | 6 变体+旧 4 窗 | none | desc | 关闭（7%→3–6%） |
| 22 | [cppi_gap_risk](archive/cppi_gap_risk_20261109.md) | none | 168 | none | 4 窗×5 倍数 | none | mech（零击穿） | 通过（m=6） |
| 23 | [cppi_crisis](archive/cppi_crisis_20261109.md) | none | 240 | none | 5 危机×2 情景 | none | mech（−10% 楼层） | 通过（附代价） |
| 24 | [cppi_start_dates](archive/cppi_start_dates_20261110.md) | none | 240 跨度 | none（自认挑窗） | 68 起点×2 carry | none | mech（7%/10%） | 通过（仅回撤承诺） |
| 25 | [cppi_bond_sensitivity](archive/cppi_bond_sensitivity_20261109.md) | assumed（单侧 5bp） | 168 | none | 5×2×4 | none | mech | 保留为背景 |
| 26 | [capm_full_market](archive/capm_full_market_20261108.md) | assumed（仅 T3） | 71 | 置换 30 次+滞后 1 日 | 覆盖 10 档+置换 30 | capm_protocol_20261107 | mech（四级） | 未判定（无法区分） |
| 27 | [drawdown_target](archive/drawdown_target_20261108.md) | none（自认） | 72 | 样本外两窗复核 | 6 配置 | none | mech（7%/10%） | 关闭（承诺不成立） |
| 28 | [supply_overlay](archive/supply_overlay_20261108.md) | assumed（"真成本"） | 72 | 72 不重叠配对块 | 第 8/25 次 | research_direction_20261108§6 | mech（t≥2.5 事前） | 关闭（解禁族） |
| 29 | [fund_flow](archive/fund_flow_20261108.md) | none | ≈5 | PIT 归档（只追加） | 2 口径 | 弱 | mech | 未判定（归档中） |
| 30 | [market_leverage_result](market_leverage_result_20261114.md) | none（回归无成本概念） | 72 | NW20+守卫 G1/G2+分段 | 3 数据版本同判据 | market_leverage_protocol_20261114 | mech（四判据 failed） | 关闭 |
| 31 | [market_leverage_growth_result](market_leverage_growth_result_20261114.md) | none | 174 | 守卫 G2+P 段未读 | 族内第 2 份 | market_leverage_growth_protocol_20261114 | mech（守卫触发→void） | 关闭（不再写第三份） |
| 32 | [ppo_value](ppo_value_20261114.md) | assumed（1.91% vs 0.87%） | 未提及 | 随机初始化+多种子+冻结测试 | 10 方向/24 run | none | desc | 关闭（PPO 线） |
| 33 | [survivorship](survivorship_20261114.md) | none | 72 | 跨文档互核（弱） | 3×2×2 | none | desc | 保留为背景（上界 0.54–0.78pp） |
| 34 | [etf_premium](etf_premium_20261114.md) | **data**（实测 −0.09pp/笔） | 154 | 净值平移±1/2 日+拆分剔除 | 4 闸门规则 | none | desc | 保留为背景（熔断 p95） |
| 35 | [e1_topk](e1_topk_20261114.md) | assumed（双边各 0.05%） | 72 | 滞后 t+1+逐位复现 | 4K×2 节奏=8 | none（自称诊断） | desc | 未判定（K 不采纳） |
| 36 | [plan_backtest](plan_backtest_20261112.md) | assumed（自认基金费率未扣） | 131 | 双引擎复现（弱） | 129 起点×2 频率 | none | mech（≥7% 仅 3.9%） | 保留为背景（下调至 4.43%） |
| 37 | [plan_sensitivity](plan_sensitivity_20261114.md) | assumed | 130 | none（自曝 P 段污染） | 6 本金×6 带宽 | none | mech（≤10% 列） | 保留为背景（带宽不调） |
| 38 | [shadow_cadence](shadow_cadence_20261114.md) | none（影子不实盘） | 前向 | 纯前向+判据前置 | 2 节奏对照 | 本文判据先写 | mech（>0 才算有信息） | 进影子运行 |
| 39 | [feature_audit](feature_audit_20261115.md) | none | 96 | HAC+独立窗+前视作废 | 96 组+18 次（多重为作废理由） | none | mech | 关闭（量价结论后作废） |
| 40 | [credit_sleeve](credit_sleeve_20261115.md) | assumed（费率进模型） | 260 | 配对同起点+月/日双频 | 8 规则+2 对照 | 自带"预注册(v4)" | mech（A1–A5） | 关闭（0/8；E1 日频压线留） |

**未纳入**：本日新开的两份（[equity_bond_tail_protocol](equity_bond_tail_protocol_20261123.md) 及其结果）
在编码范围冻结之后产生，不在 40 份内；它们的自我要求已高于本表基线（预登记 + 成本 + 判据 + 失败后果）。
