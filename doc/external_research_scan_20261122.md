# 外部研究扫描：与本仓六类对号的最新文献

日期：2026-11-22。状态：一次性检索快照（非系统综述）；只回答"外界现在在做什么"，不改任何现行判据与参数。

检索口径：arXiv q-fin API（`export.arxiv.org/api/query`，按提交日期倒排）+ OpenAlex（`api.openalex.org/works`，`title_and_abstract.search` 过滤 + 相关度排序），执行于 2026-11-22。Google/DuckDuckGo 直连被拒，cn.bing 只回行情门户页；**知网与券商研报未覆盖（中文文献是盲区）**。OpenAlex 纯按日期排序会浮出掠夺性期刊（含 2027 假日期条目），已按刊物名人工剔除；arXiv 定向查询一度被限流（429），靠慢速重试补全，不排除漏检。下表日期是 arXiv 提交日或期刊出版日，链接当天可达。

## 结论

1. **外界的"最新"大半是本仓已关闭的方向**——LLM 挖 alpha、RL 组合、拥挤度、季节性异象。[91 篇 LLM 交易论文的预登记审计](https://arxiv.org/abs/2611.18934)（2026-11-13）给出这批文献的证据纪律：75 篇有交易测试的论文里 **51% 不扣任何交易成本、仅 9% 用数据建模成本**，**中位测试窗 12 个月**（四分之一 ≤6 个月），51 篇可查者中 **98% 既不在模型训练截止后检验、也不做实盘、时序也不一致**；只有 9% 报告多重性校正结果，全部 91 篇里仅 5 篇同时做到扣成本 + 泄漏控制 + 报告试验次数。本仓关闭这些线的判断得到外部佐证——且本仓自己的门槛（真成本、预登记、PIT、随机对照）高于这批文献的中位水平。
2. **值得借力的三篇**（按对本仓的用处排序）：
   - [Trading Before a Committed Rebalance](https://arxiv.org/abs/2611.25762)（11-19）：机构必须在既定时点完成调仓、为其备货的交易者可选择是否参与的贝叶斯执行博弈——[e1_topk](e1_topk_20261114.md) 实测"每天重判把 +16.3% 打成 +0.1%"的理论对应物，**影子 60 日判读（约 2027-02）前的背景读物**。
   - [Revisiting Equity-Bond Diversification under Tail Risk and Structural Market Change](https://arxiv.org/abs/2611.21897)（11-16）：30 只全球股债/多资产 ETF（2020–2024）上，min-var/CVaR 优化确实压低 VaR/CVaR，但**等权基准的 Sharpe/Sortino/Rachev 最高**——与 [allocation_robustness](archive/allocation_robustness_20261108.md) 的"简单+低换手赢"同向，给四腿结构一个"股债对冲在结构变化下还稳不稳"的尾部视角。
   - [Short Windows, Unmodelled Costs, Unknown Cutoffs](https://arxiv.org/abs/2611.18934)（11-13）：方法论镜子；它提出的"六行报告最低标准"与本仓文档规范同宗——若再开 LLM/新数据源线，成本、窗口、泄漏控制、试验次数四项门槛可直接引用其审计数字。
3. **两个盲区确认**：本轮两个库均未检出 QDII 溢价的日频/熔断研究新作（[etf_premium](etf_premium_20261114.md) 的 p95≈9.5% 熔断仍属自研，无外部对表）；"低利率环境多资产配置"也无高相关命中（低 carry 期望表同样无对表）。

## 一、个股截面选股（本仓已整族关闭，外界仍在产出）

| 文献 | 日期 | 对本仓的意义 |
| --- | --- | --- |
| [Herding, Momentum, and Reversal in China's A-Share Market](https://arxiv.org/abs/2607.27063) | 2026-07 | 反转线关闭后首次见到"为什么会有这条信号"的机制模型（信息扩散的主体网络）。只作背景，不构成重开理由 |
| [AlphaSchema](https://arxiv.org/abs/2607.26642) / [AlphaPROBE](https://arxiv.org/abs/2602.11917) | 2026-07 / 2026-02 | LLM 批量挖 alpha 成了潮流，与已关 PPO 线同源；其证据纪律见第 2 类的 91 篇审计 |
| [The Logic of Factor Portfolios](https://arxiv.org/abs/2611.09834) | 2026-11-06 | 因子组合的布尔规则、交互与**交易成本**——与 [e1_topk](e1_topk_20261114.md)"成本决定成败"同向 |
| [Global Seasonality Factor Investing](https://arxiv.org/abs/2611.15657) | 2026-11-11 | 五个季节性异象全球普遍；快季节模式毛收益强但**换手极高**——与 [shadow_cadence](shadow_cadence_20261114.md) 的教训一致 |
| [Sales Seasonality Premium in the Chinese Stock Market](https://doi.org/10.2139/ssrn.6374738) | 2026-01 | A 股季节性的期刊版（SSRN），方向同上 |
| [The autoencoder asset pricing model in the Chinese stock market](https://doi.org/10.1080/00036846.2025.2560131) | 2025-09 | 深度定价模型在中国市场的复检——若开 U 级"指数 PE 多年期"线，这是参照系 |
| [Jump imbalance and Chinese stock market returns](https://doi.org/10.1016/j.pacfin.2025.102923) | 2025-09 | 高频跳跃不对称定价；需 tick 数据，本仓拿不到，只作记录 |
| [Foreign Capital in the Chinese Stock Market: A Firm-Level Study](https://doi.org/10.1017/s0022109025102081) | 2025-09 | 北向资金的微观定价角色（标题/摘要层） |
| [Do Chinese Retail and Institutional Investors Trade on Anomalies?](https://doi.org/10.2139/ssrn.5112567) | 2025 | 谁在交易异象——A 股 IC 普遍测不出的一个环境解释 |
| [Concept-Driven Trading in China's Stock Market](https://doi.org/10.2139/ssrn.6522518) | 2026 | 概念驱动交易；与资金流/主题方向相邻，记录备查 |

## 二、基准与测量方法论

| 文献 | 日期 | 对本仓的意义 |
| --- | --- | --- |
| [Short Windows, Unmodelled Costs, Unknown Cutoffs](https://arxiv.org/abs/2611.18934) | 2026-11-13 | 见结论 1/2；91 篇、预登记编码、双盲复核 20%；提出六行报告最低标准 |
| [Price-Volume Correlations Are Always Zero](https://arxiv.org/abs/2611.08528) | 2026-11-04 | 论证常规价量相关的经验算法按定义恒为零（源于 VWAP 对均值与量的依赖）——[feature_audit](feature_audit_20261115.md) 的"量价 32 特征 0 过门槛"多一层"测量本身可能错"的注脚 |
| [Specification LASSO and a Flexible Characteristics-Based Asset Pricing Model](https://arxiv.org/abs/2611.03391) | 2026-11-02 | 特征定价的模型设定选择——方法论记录 |

## 三、RL/PPO（本仓已关闭，外界产出照旧）

[ppo_value](ppo_value_20261114.md) 已证 PPO 与等权不可区分并关闭此线；外界 2025-10 以来仍在产出，证据纪律见上节审计：

| 文献 | 日期 | 一句话 |
| --- | --- | --- |
| [Conformal Risk Certificates for Portfolio Optimization: Gated RL](https://arxiv.org/abs/2611.14699) | 2026-11-10 | RL 组合带"风险证书"输出 |
| [Optimal portfolio under no-shorting constraints: exploratory RL](https://arxiv.org/abs/2611.00700) | 2026-10-31 | 探索式表述下的约束组合 |
| [Vibe-Trading: Towards Agent-Native Trading](https://arxiv.org/abs/2611.24845) | 2026-11-18 | LLM 代理交易的新形态 |
| [A Deep RL Framework for Dynamic Portfolio Optimization: China evidence](https://arxiv.org/abs/2412.18563) | 2024-12 | 中国市场 RL 组合（v3） |

## 四、配置与 CPPI（本仓现行方案的来源地）

| 文献 | 日期 | 对本仓的意义 |
| --- | --- | --- |
| [Revisiting Equity-Bond Diversification under Tail Risk](https://arxiv.org/abs/2611.21897) | 2026-11-16 | 见结论 2；等权基准在风险调整口径上战胜 min-var/CVaR 优化 |
| [Mitigating Return Drag and Regime Whipsaws in Sector Rotation](https://arxiv.org/abs/2611.05310) | 2026-11-03 | **regime 抖动**的收益拖累与抑制——与低 carry 分档换挡的稳定性同题 |
| [Simple Portfolios that Deliver Consistent Decent Returns](https://arxiv.org/abs/2610.36491) | 2026-10-27 | 简单组合的可信收益——与 [unified_strategy_final](unified_strategy_final_20261115.md) 的"简单赢"同向 |
| [Dynamic Scenario-Based Portfolio Optimization for Multi-Asset ETFs](https://arxiv.org/abs/2611.16540) | 2026-11-12 | 多资产 ETF 场景优化，记录 |
| Drawdown-Speed clustering for portfolio insurance（Sapienza IRIS 目录条目，无 DOI） | 2026-09 | 用回撤"**速度**"做聚类状态——CPPI 只看深度不看速度，是个新旋钮；仅见目录，未见正文 |
| [Portfolio Insurance with Volatility Control：复现代码](https://doi.org/10.17632/kpm5t7grv8) | 2026-09 | Mendeley Data 上的复现代码包 |
| [Carbon-penalised portfolio insurance strategies](https://doi.org/10.1080/03461238.2026.2682367) | 2026-06 | 部分信息随机因子模型下的组合保险设计（[arXiv 同文](https://arxiv.org/abs/2511.19186)） |
| [On the optimal design of a new class of proportional portfolio insurance](https://arxiv.org/abs/2407.21148) | 2024-07 | 跳扩散框架下的 PPI 设计——与本仓倍数上限 7（[cppi_gap_risk](archive/cppi_gap_risk_20261109.md)）的经验校准互补 |
| [Capital Market Integration and the A-H Premium: Evidence From China's QDII Reform](https://doi.org/10.33423/jaf.v25i4.7935) | 2025-11 | QDII 改革与 A-H 溢价；**刊物层级低**，仅记录 |
| [ETFs and tail dependence: Evidence from Chinese stock market](https://doi.org/10.1016/j.jimonfin.2024.103194) | 2024-09 | 中国 ETF 尾部相依——513500 溢价研究的相邻文献 |

## 五、数据源与数据工程

| 文献 | 日期 | 对本仓的意义 |
| --- | --- | --- |
| [CSMD: Curated Multimodal Dataset for Chinese Stock Analysis](https://arxiv.org/abs/2511.01318) | 2025-11 | 中国股票多模态数据集（公告/新闻/行情）——外部数据源参照，不改变本仓"未检验不评分"的评分卡纪律 |
| [Retail investor AI usage, investor attention, and stock price synchronicity](https://doi.org/10.1016/j.frl.2026.110755) | 2026-09 | 散户用 AI 与股价同步性——散户行为方向的外部动态 |

## 六、执行与验证（影子运行的邻座）

| 文献 | 日期 | 对本仓的意义 |
| --- | --- | --- |
| [Trading Before a Committed Rebalance](https://arxiv.org/abs/2611.25762) | 2026-11-19 | 见结论 2；承诺时点调仓的执行成本理论 |
| [Tracking High-Frequency Trading Activity in China](https://arxiv.org/abs/2610.33400) | 2026-10-24 | 深交所订单级 HFT 代理（毫秒级精确挂单 + 延迟 DiD）——本仓数据够不着毫秒层，只作市场质量背景 |
| [Penalize the Common Trade](https://arxiv.org/abs/2611.03453) | 2026-11-02 | 拥挤交易惩罚层——与已关 crowding overlay 线同题 |
| [Dividend Juicing in Mutual Funds: Catering to Retail in China](https://doi.org/10.1080/1540496x.2026.2756505) | 2026-11-05 | 基金分红迎合散户——载体层（[fund_vehicles](fund_vehicles_20261115.md)）的背景知识 |
| [Determinants of stock attention and behavioral preferences: China retail](https://doi.org/10.1016/j.frl.2025.108584) | 2025-10 | 注意力与行为偏好，记录 |

## 未检索到的东西（盲区清单）

1. **QDII 溢价的日频研究 / 熔断设计**：OpenAlex 命中多为宏观传导与 A-H 溢价，未见 2025 年后的日频溢价新作——p95 熔断仍属自研。
2. **低利率（<2%）环境的多资产配置新作**：两个库都未检出高相关结果——低 carry 期望表（[delivery](delivery_20261116.md) §4）无现成对表。
3. **中文文献层（知网/券商研报）**：检索通道不可用，未覆盖。
4. arXiv 定向查询被限流后靠重试补全，不排除漏检。

## 对下一步的意义

- 本轮**不改任何判据、不触发预登记**——文献按"记录不拦截"入档。
- 影子 60 日判读（约 2027-02）前：读 `2611.25762`；四腿结构若因低 carry 换挡再评估：读 `2611.21897`。
- 若开 U 级"指数 PE 多年期"线：预登记的成本/窗口/泄漏/试验次数四项门槛引用 `2611.18934` 的审计数字。
