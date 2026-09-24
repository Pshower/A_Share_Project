# 阶段二交易环境接口

## 输入契约

`MarketDataProvider` 接受两个独立的 DataFrame 或 Parquet 路径。行索引为严格递增且不重复的交易日期，列为两层索引 `(股票代码, 字段)`。股票代码为字符串；两张表必须包含相同股票集合，环境按特征表股票顺序对齐。

特征表包含已经处理好的模型特征。通过 `feature_cols` 显式选择特征；不要传入股票代码等身份字段。环境不拟合标准化参数，不修复上游未来信息泄漏。

价格表字段：

| 字段 | 必需 | 含义 |
| --- | --- | --- |
| `open` | 是 | 当日开盘成交参考价，未标准化 |
| `close` | 是 | 当日收盘估值价，未标准化 |
| `volume` | 否 | 当日总成交股数；执行时仅使用前一交易日的数据 |
| `suspended` | 否 | 开盘时停牌状态，0 或 1 |
| `limit_up` | 否 | 当日开盘前已知的涨停价 |
| `limit_down` | 否 | 当日开盘前已知的跌停价 |

字段若提供，须覆盖所有股票。可使用 NaN 表示某日未知；不能用标准化价格作为成交价。无效或未知开盘价不成交，收盘缺失可以沿用已有历史收盘价估值，上市前缺价不会后向填充。

`price_basis` 必须显式指定 `unadjusted` 或 `hfq_research`。前者要求不复权报价，后者仅支持简化研究。价格口径属于调用方数据契约，正数检查不能自动判断一组价格是否真实报价。

## 调用示例

以下示例假设调用方已准备好满足契约的独立特征和价格文件，这些示例路径不是仓库现有产物。不要把现有 `panel_data.parquet` 同时用作特征和成交价格。

```python
from src.envs.market import MarketDataProvider
from src.envs.broker import BrokerConfig
from src.envs.trading import TradingEnv

market = MarketDataProvider(
    "data/clean/features_v2.parquet",
    feature_cols=["return", "close_ma_ratio_20", "volatility_20"],
    price_path="data/clean/execution_prices.parquet",
    price_basis="unadjusted",
    start_date="2024-01-02",
    end_date="2024-12-31",
    lookback=20,
)
env = TradingEnv(
    market,
    initial_capital=1_000_000,
    broker_config=BrokerConfig(
        commission=0.0003,
        min_commission=5.0,
        sell_tax=0.0,       # 按实验场景设置，不代表历史实际税率
        slippage=0.001,
        lot_size=100,
        max_weight=0.05,
        volume_fraction=0.01,
    ),
)
obs, info = env.reset(seed=42)
done = False
while not done:
    # 示例只演示接口：保留现金，不构成基线策略或训练实现。
    action = [0.0] * market.n_stocks
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

观测内容：

- `stock_features`：`[股票数, 历史窗口, 特征数]` 的 float32 数组。
- `portfolio`：各股票收盘权重，最后一个元素为现金比例。
- `valid_mask`：当前特征窗口与估值是否有效；无效特征填零，但不允许据此买入。
- `buy_mask`、`sell_mask`：当前已知市场状态的方向掩码，不预告下一日状态。实际成交时重新检查次日开盘约束。卖出不要求买入特征有效，以支持已有持仓退出。

动作是各股票目标权重，有限、非负、总和不超过 1，剩余部分为现金。超过单票上限或总仓位约束的动作报错，不静默归一化。`action_space` 为带总仓位校验的 Box 子类，其 `sample()` 只生成有效目标；未来接入策略网络时仍需在模型端映射动作到合法权重。

## 时间、成交与审计

每次 `step()` 从当前收盘推进到下一交易日，按下一日开盘时资产估值换算目标股数，再先卖后买。新买股数当日不可卖出，下一交易日解锁；重复调用同一天 `start_day()` 不会解锁。

成交价包含配置的买卖方向滑点，并受已提供涨跌停价边界限制。费用按每笔实际成交金额计算；买入包含最低佣金的资金约束通过缩减整手数量满足。卖出收益可用于当天买入。

奖励为 `本日收盘净值 / 上日收盘净值 - 1`，已经包含手续费和滑点影响。到最后一日返回 `terminated=True`，达到可选 `max_steps` 返回 `truncated=True`；结束后继续 `step()` 会报错，需要重新 `reset()`。

`info` 包含日期、净值、现金、持仓、可卖股数、累计费用、当步订单、价格口径和简化假设。订单含目标权重、请求股数、成交股数、成交参考/执行价格、费用、完成状态、拒绝或部分成交原因，以及成交后该股持仓和账户现金。没有价格而无法成交的订单价格可为 NaN，但模型观测必须有限。

`env.history` 保存初始状态及逐日结果，`env.broker.orders` 保存全部订单。`reset()` 会清空两者和持仓账户。环境对象独占传入的市场提供器，多个环境应分别创建提供器。

## 限制与验证

本阶段没有实现公司行为记账、退市回收、历史费率表、各板块申报差异和盘中撮合。阶段一未完成的数据处理问题依然存在。具体简化假设同时写入每次返回的 `info['assumptions']`；不能把缺少事件及交易状态的历史回放视为真实交易回测。

安装环境依赖：`python -m pip install -r requirements-env.txt`。

离线验证：`python -B -m pytest tests/test_trading_env.py tests/test_download.py -q -p no:cacheprovider`。
