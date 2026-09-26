"""Common training coverage on the original calendar, without joining gaps."""

import numpy as np
import pandas as pd


def training_coverage(features, prices, codes, lookback, train_end):
    if not codes or len(set(codes)) != len(codes) or not 1 <= lookback <= 60:
        raise ValueError("Select unique stocks and a valid lookback")
    dates = features.index
    train_indices = np.flatnonzero(dates <= pd.Timestamp(train_end))
    per_stock, masks = [], []
    for code in codes:
        values = features.loc[:, code].to_numpy(dtype=float)
        quote = prices.loc[:, [(code, f) for f in ["open", "close", "volume"]]].to_numpy(dtype=float)
        raw = np.isfinite(quote).all(axis=1) & (quote[:, :2] > 0).all(axis=1) & (quote[:, 2] >= 0)
        valid = pd.Series(np.isfinite(values).all(axis=1), index=dates).rolling(lookback).sum().eq(lookback).to_numpy() & raw
        masks.append(valid)
        def bounds(mask):
            available = dates[mask]
            return (str(available[0].date()), str(available[-1].date())) if len(available) else (None, None)
        raw_start, raw_end = bounds(raw)
        feature_start, feature_end = bounds(valid)
        per_stock.append(dict(code=code, raw_start=raw_start, raw_end=raw_end,
                              feature_start=feature_start, feature_end=feature_end,
                              ready_at_train_end=bool(len(train_indices) and valid[train_indices[-1]])))
    common = np.stack(masks).all(axis=0)
    available = dates[common]
    start = end = None
    count = 0
    reason = None
    if not len(train_indices) or not common[train_indices[-1]]:
        blocked = [r["code"] for r in per_stock if not r["ready_at_train_end"]]
        reason = "训练截止日缺少有效数据：" + "、".join(blocked[:10]) + "；需调整选股或重建日期划分及 scaler"
    else:
        last = int(train_indices[-1])
        first = last
        while first > 0 and common[first - 1]:
            first -= 1
        count = last - first + 1
        start, end = str(dates[first].date()), str(dates[last].date())
        if count < 2:
            reason = "共同连续区间不足两条记录，无法进行下一日成交"
    return dict(stocks=per_stock, common_start=str(available[0].date()) if len(available) else None,
                common_end=str(available[-1].date()) if len(available) else None,
                common_rows=len(available), train_start=start, train_end=end, train_rows=count,
                eligible=reason is None, reason=reason, lookback=lookback,
                policy="intersection", configured_train_end=str(train_end))
