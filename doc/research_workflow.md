# 本地数据与基线回测

更新日期：2026-09-25。本文负责数据与基线的运行方式；当前阶段状态及最新验收统一见 [实施进度](implementation_plan.md)，完整入口见 [文档索引](README.md)。PPO 的命令单独维护在 [PPO 运行说明](ppo_training_framework.md)。

## 一条命令运行

在仓库根目录、Conda `Graduate` 中执行：

```powershell
python -B -m src.backtest.run --config configs/research.json
```

数据版本不存在时，从已有个股后复权 CSV 构建；存在时检查配置与关键产物哈希并复用。研究流程不导入或调用下载接口，不联网。只运行验证集，配置成测试集会拒绝执行，避免选型阶段误用最终测试集。

数据与基线依赖入口为 `requirements.txt`；要运行包括 PPO 在内的完整测试，请安装 `requirements-ppo.txt`，它会包含前者。安装依赖与下载行情是不同操作，依赖齐备后的本地数据处理和测试不需要网络。新实验会写入独立的 `reports/runs/<run_id>`，可用 `--run-id <名称>` 指定；已存在的实验目录不会覆盖。

## 数据版本

默认配置对应 `data/clean/research_v3`，旧顶层清洗数据和开发中生成的中间版本保留，但不会被默认流程读取。PPO 入口只校验和复用已有版本，不自动触发下面的预处理流程。

| 文件 | 内容 |
| --- | --- |
| `panel_data.parquet` | 标准化的 20 个相对特征，列为 `(code, feature)` |
| `prices.parquet` | 未标准化的后复权开盘、收盘及成交股数，研究用途 |
| `valid_mask.parquet` | 每日每股特征完整性 |
| `source_mask.parquet` | 本地该日该股是否有源记录，不能等同于官方停牌状态 |
| `all_stocks_features.parquet`、个股特征文件 | 对齐后的长表及个股分组 |
| `train.parquet`、`val.parquet`、`test.parquet` | 固定日期边界切分的特征 |
| `scaler_params.json` | 特征顺序、训练日期与股票、拟合行数、均值和缩放值 |
| `source_audit.csv`、`quality.csv` | 来源日期、源文件哈希、缺失股票原因及特征缺失统计 |
| `training_feature_distribution.csv` | 训练集合按特征的分布检查，非逐股标准化目标 |
| `manifest.json` | 数据版本、构建参数、源码哈希、输入及产物哈希和限制 |

新股票推理复用训练参数，不能再拟合。`transform_new_data()` 接受单只股票的完整历史预热数据；调用方应按交易日期对齐，缺失日期保留空行。不能只传当日一行来获得 60 日技术指标。

预处理单独运行：

```powershell
python -B -m src.data.preprocess --config configs/research.json
python -B -m src.data.clean_data_analysis --data data/clean/research_v3
python -B -m src.data.check_missing_shares --data data/clean/research_v3
```

预处理不覆盖已有目录。修改日期、股票集合或特征逻辑时，需同时修改配置中的版本名称及输出目录后重新构建。原始旧文件名虽然包含 `2017_2024`，实际范围由文件内容审计；新下载代码使用完整请求日期命名，但本次未重新下载。

## 基线与口径

- `cash`：始终持有现金，未模拟存款利息。
- `buy_hold`：首次出现有效观测时提交一次等权目标，之后不下单；未成交数量不会重试。
- `equal_weight`：每 20 个交易步骤对当前有效股票提交等权目标。
- `momentum`：每 20 步选择当前 20 日收益排名前 20 只，等权分配并受单票上限约束。共享 Z-score 对同一特征是单调变换，不改变截面排序。

四条策略共享日期、股票池、初始资金、手续费、滑点与成交约束。目标等权不代表实际成交后等权；后复权价格配合整手限制尤其容易造成大量现金留存，必须连同平均现金比例和成交明细解读结果。

信号时点为当日收盘，成交为次日开盘。验证集从 2022-08-08 至 2023-10-20，共 291 步；以 2022-08-05 收盘初始化账户和首个信号。阶段三默认入口仍只做验证；阶段四的 32 只股票轻量实验已完成冻结后的测试回测，见 [报告摘要](reports/ppo_lightweight_20260925_summary.md)。

指标年化使用 252 个交易步骤；Sharpe 无风险利率设零，零波动为 null。最大回撤包含初始资金。换手率为当日买卖成交金额绝对值之和除以上日收盘净值，未除以二。手续费、滑点已体现在净值中，不再次扣减。

## 输出与复现

每个策略保存 `daily.csv`、`positions.parquet`、`orders.csv`。总体保存 `metrics.json/csv`、`config.json`、`experiment.json`、`report.md` 和 `curves.png`。逐日自动检查现金、持仓、总资产及累计费用与成交记录一致。

实验记录 Git commit、未提交状态、源码 SHA-256、数据 manifest 哈希、种子和依赖版本。`benchmark_path` 可指定本地含 `date,close` 的指数 CSV，必须完整覆盖评价日期；缺少文件时记录缺口，不自动获取。指数比较仅为不含费用的价格收益，不代表真实成交组合。

运行离线测试：

```powershell
python -B -m pytest -q -p no:cacheprovider
```

网络集成测试默认排除；其余测试由夹具禁止建立网络连接。数据/基线测试覆盖未来数据扰动、非训练股票扰动、缺失值及价格分离、公司行为记账、基线调仓时点、指标定义和重复运行一致性；完整命令还会执行 PPO 无训练检查。测试数量及本地回归结果只在 [实施进度](implementation_plan.md) 中维护。

## 尚需外部数据

固定 2026 年成分名单仍有选择偏差；当前日历来自本地行情日期并集。未提供全量不复权报价、真实公司行为、历史成分股、ST、停牌及涨跌停状态。环境支持部分对应接口，但无法从现有 OHLCV 可靠还原这些信息。当前报告是 HFQ 简化研究模拟，不是可执行的实盘收益证明。
