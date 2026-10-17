# 训练数据可得性核查与改进方案

日期：2026-10-15。状态：设计稿，尚未实施。本轮只做只读接口探测和文档记录：未下载训练数据、未修改源码、未训练、未评估 2026-07-01 之后的保护期。

本轮在 `AShareGPU` 环境（Python 3.11.16、AKShare 1.18.64）对候选接口执行真实只读探测，记录实际返回、字段和失败原因。探测脚本为 `.workbench/probe_akshare{,2,3,4,5,6}.py`；`.workbench/` 已在 `.gitignore` 内，不进入版本库。探测结果是单机、单网络、单时点的快照，**不构成供应商能力认证**；正式采集仍须由用户显式启动并核对真实样例，与 [AKShare 下载范围扩展](akshare_download_expansion.md) 的规则一致。

总体实施状态以 [实施进度](implementation_plan.md) 为唯一入口，本文不另立阶段编号；架构背景见 [项目 Pipeline](../pipeline.md)，数据关联约束见 [训练数据关联](training_data_linkage.md)。

## 1. 探测环境与网络事实

**东方财富系在当前网络下大面积不可用**，且已有本地证据表明这不是本轮才出现的问题。

| 目标域名 | 本轮探测结果 |
| --- | --- |
| `push2his.eastmoney.com` 小窗口（2024-01-02 至 01-12） | 成功 9 行 |
| `push2his` 三年窗口、`000001` 全历史 | `ProxyError` |
| `push2his` 退市股 `600001` 全历史 | 成功 2,753 行 |
| `push2.eastmoney.com`、`40.push2`、`80.push2`、`17.push2` | 全部 `ProxyError` |

已有下载批次的完成度：

| 批次 | 来源 | 请求 | 成功 | 失败 | 失败原因 |
| --- | --- | ---: | ---: | ---: | --- |
| `h_5cd4a2e818924259` | sina | 300 股 × `hfq` | 300 | 0 | — |
| `h_74c25a4c5d6e4b47` | eastmoney | 300 股 × (`hfq`,`unadjusted`) | 9 | 591 | 5 次 `RemoteDisconnected` 后触发批量熔断 |
| `h_82a51ea875564aaa` | eastmoney | 300 股 × (`hfq`,`unadjusted`) | 0 | 600 | 597 项"取消前未采集"、2 次 `RemoteDisconnected` |

结论：批量采集的可靠路径当前是**新浪（日线）＋ 巨潮（公司行为、行业）＋ 申万（行业指数）**；东方财富只能作为退市股历史的补充，且需要与批量熔断隔离的独立重试策略。此结论只在上述网络环境成立，换网络须重新探测。

## 2. 已核实可回溯的数据

下表全部为本轮真实返回，可指定历史区间或历史报告期，具备进入训练或回测撮合的潜力。

| 用途 | 接口 | 本轮实测 |
| --- | --- | --- |
| 独立交易日历 | `tool_trade_date_hist_sina()` | 8,797 行，1990-12-19 起 |
| 不复权日线 | `stock_zh_a_daily(adjust="")` | `sz000001` 2,367 行，2017-01-03 至 2026-09-30 |
| 后复权日线 | `stock_zh_a_daily(adjust="hfq")` | 同长度 2,367 行，可逐日交叉校验 |
| 流通股本 | 同上返回的 `outstanding_share` | 与不复权价格同源同频；`000001` 全期 15 个变动日 |
| 分红送转事件 | `stock_fhps_em(date=报告期)` | 2017/2020/2023 报告期分别 2,794 / 3,125 / 3,855 行 |
| 逐股分红（含派息日） | `stock_dividend_cninfo(symbol)` | `600519` 31 行，回溯至 2002，最新 2026-06-22 |
| 上市名录与上市日期 | `stock_info_sh_name_code()` / `_sz_` / `_bj_` | 1,702 / 2,905 / 349 行 |
| 退市名录与终止上市日 | `stock_info_sh_delist()` / `_sz_delist()` | 159 / 208 行 |
| 全 A 代码表 | `stock_info_a_code_name()` | 5,574 行 |
| 退市股历史行情 | `stock_zh_a_hist(symbol="600001", adjust="")` | 2,753 行，1998-01-22 至 2009-12-15（**东财，当前不稳定**） |
| 基准指数日线 | `stock_zh_index_daily("sh000300")` | 6,008 行，2002-01-04 至 2026-10-14 |
| 行业指数日线 | `index_hist_sw(symbol="801120", period="day")` | 6,471 行，1999-12-30 至 2026-10-14 |
| 行业成分、权重与纳入日期 | `index_component_sw(symbol="801120")` | 122 行 |
| 逐股行业变更历史 | `stock_industry_change_cninfo(symbol, start, end)` | `600519` 9 行、`002594` 11 行，多套分类标准 |
| 全市场估值时序 | `stock_a_ttm_lyr()` | 262 行月度，2005-01-31 起 |
| 两融明细 | `stock_margin_detail_sse(date)` / `_szse` | 2024-01-05 单日 1,772 行 |
| 限售解禁 | `stock_restricted_release_detail_em(start, end)` | 2024-01 单月 256 行 |
| 股东户数 | `stock_zh_a_gdhs_detail_em(symbol)` | `000001` 68 行，2013-03-01 起 |
| 业绩报表 | `stock_yjbb_em(date=报告期)` | 2023 年报 11,629 行，含最新公告日期 |
| 龙虎榜 | `stock_lhb_detail_em(start, end)` | 按区间可取 |

### 2.1 公司行为账本可用复权因子独立对账

对 `000001` 取新浪后复权与不复权收盘价之比，检测相对跳变超过 0.5% 的日期，与 `stock_fhps_em` 的除权除息日比对：

| 来源 | 日期序列 |
| --- | --- |
| 复权因子断裂 | 2018-07-12、2019-06-26、2020-05-28、2021-05-14、2022-07-22、2023-06-14、2024-06-14、2024-10-10、2025-06-12、2025-10-15、2026-06-12、2026-09-24 |
| `stock_fhps_em` 除权除息日 | 2018-07-12、2019-06-26、2020-05-28、2021-05-14、2022-07-22、2023-06-14、2024-06-14（每 10 股派息 1.36 / 1.45 / 2.18 / 1.80 / 2.28 / 2.85 / 7.19 元） |

两者**逐个精确吻合**。这意味着公司行为账本可以建成，并且能用"因子断裂"与"分红事件"双向核验，与仓库既有的独立审计习惯一致。注意：比值本身带 0.01 元报价舍入噪声（2,367 天中有 1,691 个不同取值），不能直接用比值反推事件，只能用于校验。

## 3. 只有当前快照，不能用于历史回测

这一类最容易误判：接口存在、返回非空，但**没有历史**。`stock_zt_pool_em` 对历史日期返回的是**空表而不是报错**，若在聚合统计中直接使用，会被误读为"当日没有涨停股"。

| 接口 | 本轮验证 | 性质 |
| --- | --- | --- |
| `stock_zt_pool_em(date=近 3 日)` | 46 行 | 仅近期 |
| `stock_zt_pool_em(date=40 天前 / 200 天前)` | **0 行** | 静默空表 |
| `stock_zt_pool_dtgc_em` | 抛错"只能获取最近 30 个交易日" | 硬上限 |
| `news_trade_notify_suspend_baidu(date=2026-10-12)` | 2 行 | 仅近期 |
| `news_trade_notify_suspend_baidu(2018 / 2020 / 2022)` | **全部 0 行** | 无回溯 |
| `stock_zh_a_stop_em()` | 304 行 | 当前停牌快照 |
| `stock_market_activity_legu()` | 12 行 | 当前涨跌家数 |
| `stock_info_change_name(symbol)` | `000001` 3 行，字段仅 `index`/`name` | **有序列、无日期** |

最后一条需要特别记录：简称变更序列可以判定"该股票曾经是 ST"（序列中出现 `ST` 字样），但**无法确定起止日期**。

因此：完整的、权威的**历史停牌 / 涨跌停 / ST 逐日状态在 AKShare 免费接口层面不可得**，只能用第 5 节的替代方案推算并明确披露误差。

## 4. 当前接口层不可得与其他缺陷

**不可得**

- **沪深 300 历史成分股**：`index_stock_cons_csindex("000300")` 返回 300 行但日期字段为 2026-10-14（当前）；`index_detail_hist_cni("399001")` 返回 500 行但日期字段为 2026-09-30（最新一期）。两者都不是 point-in-time 成分。
- 退市股票的**新浪**行情：`stock_zh_a_daily("sh600001")` 抛 `JSONDecodeError`；同一标的东财有完整数据。退市股历史因此绑定在东财可达性上。

**本机 AKShare 1.18.64 缺陷**

- `stock_share_change_cninfo` 抛 `OptionError: No such keys(s): 'future.no_silent_downcasting'`，属 pandas 兼容问题，非数据缺失。
- `sw_index_third_cons` 抛 `ValueError: Length mismatch: Expected axis has 20 elements, new values have 17`，列契约变化。
- `stock_industry_clf_hist_sw` 因申万站点 xls 的 SSL 证书验证失败；同站 `index_hist_sw` 正常，属单接口问题。

按既有规则**不自动升级 AKShare**；这些缺陷记录在案，需要时按单接口适配处理，不整体升级依赖。

## 5. 针对不可得缺口的替代构造方案

| 缺口 | 替代方案 | 误差性质与披露要求 |
| --- | --- | --- |
| 历史停牌 | 该股当日无行情行、但市场其他股票有行情 → 推断停牌；用 `src/data/coverage.py` 已计算的 `raw_start` / `raw_end` 区分"未上市/已退市"与"上市期间停牌" | 无法区分真实停牌与数据源缺行，须逐股披露推断比例 |
| 历史涨跌停 | 不复权 OHLC + 前收盘 + 板块规则推算理论涨跌停价：主板 10%、创业板 2020-08-24 后 20%、科创板 20%、北交所 30%；触及即判定不可成交 | ST 期间比例为 5% 且 ST 区间不可得，须单独披露该档误差 |
| ST 精确区间 | 简称变更序列判定"曾经 ST"，区间用"涨跌幅恰好落在 5% 附近"或年报数据反推 | 只能给出边界，不能保证区间端点精确 |
| 沪深 300 历史成分 | 用第 6.2 节的全市场 point-in-time 宇宙替代 | 这是口径变化而非降级：全市场宇宙比指数成分更抗幸存者偏差，但两者收益分布不可直接对比 |

## 6. 对训练数据的改进建议

以下全部为建议，**均未实施**。当前训练数据为 `data/clean/sina_hs300_study_20260927`：300 只股票、2,364 个交易日、709,200 行矩形面板，其中 641,413 行有源报价（**9.56% 的行没有任何行情记录**），特征缺失率均值 10.7%、最高 92.3%，有效行数最少 139 天（`001280`）。逐折结构本身是干净的：每个 fold 有独立数据集，`train_end` 为 `{fold}-03-31`、`val_end` 为 `{fold}-06-30`，scaler 逐折重拟合。

### 6.1 数据质量：区分两类缺失，并让缺失本身成为特征

源审计显示 300 只股票的 `invalid_price_rows` 合计为 **0**，即 9.56% 的空白全部来自"该股票当天没有行情行"，而不是坏报价。这些空白混合了两种经济含义完全不同的情况：上市前的"未上市"和上市期间的"停牌"。

- 按 `raw_start` / `raw_end` 切分两类缺失，停牌另行标记。
- `src/envs/market.py` 的 `observation()` 把非有限特征填成 0；0 在 Z-score 空间等于训练集均值，等于告诉模型"这天数据很普通"。建议至少增加 `missing_rate_20d` 让模型知道自己被填了零。
- 对 `volume_ratio` / `turnover`，"没有成交"应表达为 0；价格类特征的缺失才应留给掩码。

### 6.2 股票池：改为 point-in-time，并做一次留出股票

当前池子是 `data/stock_list/hs300_20260629.txt`，即 2026-06-29 的沪深 300 快照，被用于 2017 年以来的全部历史，同时引入幸存者偏差和"未来成分股"偏差。

可用的替代构造（全部来自第 2 节已核实接口）：

```text
stock_info_sh_name_code (1,702)  ┐
stock_info_sz_name_code (2,905)  ├─ 含上市日期
stock_info_bj_name_code (349)    ┘
            +
stock_info_sh_delist (159)  ┐
stock_info_sz_delist (208)  ├─ 含终止上市日期
                            ┘
            =
每个交易日的全部在场 A 股
```

配套要求：

- 按数据充分度筛股票时，**只能用该 fold 训练期内的数据筛**，绝不能用全期有效行数，否则重新引入前视。
- 做一次**留出股票**实验。仓库已支持 scaler 拟合股票池与训练股票池分离（`train_codes` 与 `stock_codes` 本就是两个字段），但现有 fold 配置里 `train_codes` 就是全池。跨股票泛化是本仓库唯一未被验证过的核心主张。

### 6.3 特征：增加截面标准化与市场状态

现有 20 个特征全部是时序相对量（除以自身 close 或自身均线），且 Z-score 是**全池、全训练期的逐特征全局均值方差**。后果是 `volatility_20` 长期偏高的股票永久得到正的 z 值，模型学到的是"哪些股票历史上波动大"，而不是"今天哪些股票相对强势"。

- 在 `src/data/preprocess.py` 的 `add_features` 之后、`normalize_features` 之前插入逐日截面变换：`cs_rank(f)` 与 `cs_zscore(f)`，至少覆盖 `momentum_*`、`volatility_*`、`volume_ratio`、`turnover`。截面变换逐日无参数，不引入拟合期问题，也不破坏旧模型契约。
- 补充市场状态特征，全部可用现有面板或第 2 节接口算出：上涨家数占比、截面收益离散度、截面动量离散度、全池成交额相对 20 日均值的变化，以及 `stock_zh_index_daily` 的指数日收益。
- 行业维度使用 `index_hist_sw`（行业指数日线）与 `stock_industry_change_cninfo`（逐股 PIT 归属），不使用当前快照归属。

### 6.4 执行保真：不复权价格与公司行为账本

当前 `price_basis` 为 `hfq_research`，`initial_capital=1,000,000`、`lot_size=100`、`max_weight=0.05` 作用在后复权价格上，得到的股数是虚构值。市场层已支持 `unadjusted` + `corporate_actions` 并带完整字段校验。

**这不是数据缺口，而是没有执行。** `src/data/sina.py` 的 `history()` 已经是 `adjust="hfq" if basis == "hfq" else ""`，即 `unadjusted` 口径**现在就能下载**，只是历史批次只请求过 `bases=["hfq"]`。

字段映射到 `corporate_actions` 需要的四列：

| 目标字段 | 来源 | 换算 |
| --- | --- | --- |
| `date` | `除权除息日` | 直接使用 |
| `cash_per_share` | `现金分红-现金分红比例` | 每 10 股派息 ÷ 10 |
| `share_multiplier` | `送转股份-送股比例` + `送转股份-转股比例` | 1 + (送 + 转) ÷ 10 |
| `payment_date` | `stock_dividend_cninfo` 的派息日 | 直接使用 |
| `share_available_date` | `stock_dividend_cninfo` 的股份到账日 | 直接使用 |

`stock_fhps_em` 缺 `payment_date` 与 `share_available_date`，需用逐股 `stock_dividend_cninfo` 补齐。建议**双轨**：后复权用于特征构造与策略搜索，不复权 + 公司行为用于最终确认，并量化两条路径的差异。

### 6.5 交易日历与容量

- 接入 `tool_trade_date_hist_sina` 作为独立交易日历。当前 manifest 明说 calendar 是"本地行情日期并集"，接入后缺数据才能被正确判定为停牌而非"非交易日"。
- 拿到不复权数据后做容量压力测试：`volume_fraction ∈ {0.5%, 1%, 5%}` × `initial_capital ∈ {100 万, 1000 万, 1 亿}`。若容量是瓶颈，此前所有关于收益差异的结论都需要重新解释。

### 6.6 实验纪律：登记已曝光窗口

2023 / 2024 / 2025 的 Q2 窗口已被多轮研究反复用于筛选，协议中已有"只能称本轮选择未使用的时间确认，不称未曝光盲测"的诚实表述，但依靠自觉维持。

建议机制化：在 `protocol.json` 旁维护 `exposed_windows.json`，任何被任一报告引用过的窗口自动登记并写进后续协议；规定训练窗口不得与"非曝光确认窗口"重叠。仓库已有 `tools/audit_timing_*.py` 这类工具先例，可实现为小工具加测试。

## 7. 建议的采集清单

全部走新浪、巨潮、申万，避开东财；每项均须用户显式授权并核对样例。

| 顺序 | 采集 | 接口 | 规模 | 用途 |
| --- | --- | --- | --- | --- |
| ① | 不复权日线 | `stock_zh_a_daily(adjust="")` | 300 只 × 1 次 | 真实成交价、股数、市值 |
| ② | 分红送转事件 | `stock_fhps_em` | 约 36 个报告期 | 公司行为账本 |
| ③ | 逐股分红明细 | `stock_dividend_cninfo` | 300 次 | 补派息日与股份到账日 |
| ④ | 交易日历 | `tool_trade_date_hist_sina` | 1 次 | 独立日历与停牌判定 |
| ⑤ | 上市与退市名录 | `stock_info_{sh,sz,bj}_name_code` + 两个 `_delist` | 5 次 | point-in-time 宇宙 |
| ⑥ | 行业指数与成分 | `index_hist_sw` + `index_component_sw` | 按需 | 行业特征 |
| ⑦ | 逐股行业变更 | `stock_industry_change_cninfo` | 300 次 | PIT 行业归属 |
| ⑧ | 基准指数 | `stock_zh_index_daily("sh000300")` | 1 次 | 基准与市场状态 |

①②是核心，③④⑤是配套，⑥⑦⑧是特征。`outstanding_share`（流通股本）目前被 `src/data/online.py` 的 `normalize_history` 列白名单丢弃，采集 ① 时应一并保留。

## 8. 边界与不做的事

- 本轮探测未下载任何训练数据，未把任何新数据接入特征、训练或回测撮合。第 2 节列出的接口只是"可获取"，不等于"已接入"或"已验证预测能力"。
- 不把当前快照当作历史：`stock_zt_pool_em` 的历史日期返回空表，聚合使用会造成静默错误；`stock_individual_info_em`、`stock_market_activity_legu` 等当前快照不能反填历史。
- 不为了修复单接口缺陷而整体升级 AKShare；`stock_share_change_cninfo` 的 pandas 兼容问题按单接口处理。
- 不重试东方财富批量下载。若确需退市股历史，应走逐股小窗口、且与批量熔断隔离的独立路径。
- 指数点位不是股票成交价格；供应商分红为税前口径，尚未核验每股基准、到账日期与税务规则，接入账户前必须补足。
- 行业归属需历史成分而非当前名单；财报需按公告可得时间而非报告期末对齐，本轮未核验披露时点，不进入训练。
- 第 5 节的替代推断（停牌、涨跌停、ST 区间）必须与权威状态数据区分标注，不能表述为"已还原真实交易状态"。
- 新增数据、特征与 fold 结构都属于 [实施进度](implementation_plan.md) 阶段一与阶段五的范围，须先预注册协议再执行；本文不替代实施进度中的任务清单。

## 9. 与现有文档的关系

| 文档 | 分工 |
| --- | --- |
| [AKShare 下载范围扩展](akshare_download_expansion.md) | 已实现的扩展下载类型与操作入口；本文只做可得性探测与改进建议 |
| [联网行情与最新判断计划](online_data_live_decision_plan.md) | 联网快照与推理边界 |
| [训练数据关联](training_data_linkage.md) | 多股训练日期交集与行情来源关联 |
| [市场信息与技术指标扩展方案](market_information_indicator_plan.md) | 页面展示层的指标与导入设计 |
| [交易环境](trading_environment.md) | 账户、成交与约束契约；本文 6.4 的字段映射以其 `corporate_actions` 契约为准 |
| [本地研究流程](research_workflow.md) | 数据产物重建与基线复现命令 |
| [实施进度](implementation_plan.md) | 唯一的阶段状态与验收证据入口 |

## 10. 执行记录与仓库大文件核查（2026-10-17）

### 10.1 清理执行

- 删除失败残留 `data/clean/.failed_agent_improvement_20260927_walk_01/all_stocks_features.parquet`（90,584,368 字节），已获用户确认。失败原因与处置记录见 [agent_improvement 报告](reports/agent_improvement_20260927.md)；完成版数据集 `agent_improvement_20260927_walk`（457 MB）保留。
- confirm 首启被窗口关闭中止后，两个未封口的 teacher `inner_train` 部分目录（2025/2026 折）已按协议删除并重采，`data/2025`、`data/2026` 产物未受影响。过程见研究 runlog。

### 10.2 raw 文件命名说明

`configs/research.json` 的 `raw_pattern` 为 `{code}_data_2017_2024_hfq.csv`，但实测该批文件实际覆盖至 2026-06-26（`000001`、`600519` 与新浪 hfq 共同最后交易日 2026-06-26，共同 2,300 个交易日）。文件名为冻结约定保持不改，实际区间以内容为准。

### 10.3 双源后复权口径差（实测）

对 `000001`、`600519` 抽检新浪 hfq（`data/online/history/h_5cd4a2e818924259`）与东财 hfq（`data/raw`）收盘价：

| 股票 | sina/EM 收盘比 min–max | 比值均值 | 日收益相关系数 |
| --- | --- | --- | --- |
| `000001` | 0.8643 – 0.9384 | 0.9032 | 0.998724 |
| `600519` | 1.3216 – 1.4700 | 1.4228 | 0.999277 |

比值在全期不是常数（`000001` 波动约 8%、`600519` 约 10%），说明两源复权基准与分红处理不同，不只是整体缩放；日收益高度一致但非完全相同。结论：单个数据集只能绑定一个源，禁止混用；跨源比较收益分布前必须先统一复权口径。

### 10.4 仓库大文件核查（只读）

- 版本库本身很小：409 个受控文件，`.git` 5.9 MB，最大受控文件为 `doc/research_record_20261011.pdf`（1.34 MB）。
- 本地物理大目录（全部在 `.gitignore` 内）：`.workbench` 2.59 GB（torch wheel 安装介质）；`reports/runs` 26.55 GB（研究证据）；`data/clean` 2.16 GB（各版本数据集，报告与文档引用的证据）；`data/raw` 253 MB；`data/online` 203.5 MB；`web/node_modules` 194.7 MB（可由 `npm install` 重建）。
- 已按用户批准删除（2026-10-17，合计释放 2,615.6 MB）：`data/clean/` 下 11 个 UI/e2e 测试版本目录（`feature_ui_*`×3、`web_check_*`×4、`workflow_ui_*`×3，约 23 MB，由 `web/e2e/*.spec.ts` 以时间戳命名生成，代码未引用）；`.workbench/wheels/torch-2.7.1+cu126-cp311-cp311-win_amd64.whl`（2.59 GB，安装介质，需要时重新下载）。
- 保留候选（用户决定暂不删除）：`data/clean/` 根目录旧构建产物（`panel_data.parquet` 128.7 MB、`all_stocks_features.parquet` 104.7 MB、`train/val/test.parquet` 约 117 MB、300 个个股 `{code}_features.parquet` 约 150 MB、`test000001_000002s2017e2024/` 3 MB）；`web/node_modules`（194.7 MB，可由 `npm install` 重建）。当前配置 `clean_dir` 指向 `data/clean/research_v3`，根目录产物来自版本化之前的构建。
