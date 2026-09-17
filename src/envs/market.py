import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

class MarketDataProvider:
    """
    按日期索引提供面板数据，包含特征和价格
    """
    def __init__(self, panel_path: Path, feature_cols: Optional[List[str]] = None):
        self.panel = pd.read_parquet(panel_path)          # MultiIndex列: (code, feature)
        self.dates = self.panel.index                     # DatetimeIndex
        self.n_dates = len(self.dates)
        
        # 获取所有股票代码
        self.stock_codes = self.panel.columns.get_level_values(0).unique().tolist()
        self.n_stocks = len(self.stock_codes)
        
        # 提取收盘价列（假设特征名称为 'close'）
        self.price_cols = [(code, 'close') for code in self.stock_codes]
        self.prices = self.panel[self.price_cols]          # DataFrame: (dates, stocks)
        self.prices.columns = self.stock_codes             # 重命名列
        
        # 特征列（除 close 外其他列）
        if feature_cols is None:
            # 默认使用除 close 外的所有数值特征
            all_features = self.panel.columns.get_level_values(1).unique().tolist()
            feature_cols = [f for f in all_features if f != 'close']
        self.feature_cols = feature_cols
        # 构建特征面板：形状 (dates, stocks, features)
        # 为了效率，我们预提取为3D数组
        self.feature_data = np.stack([
            self.panel[(code, feat)].values for code in self.stock_codes for feat in self.feature_cols
        ], axis=-1).reshape(self.n_dates, self.n_stocks, len(self.feature_cols))
        # 注：索引顺序 [date, stock, feature]
        
        self.current_index = 0
    
    def reset(self):
        self.current_index = 0
        return self.get_state()
    
    def get_state(self, idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回当前日期的特征矩阵 (n_stocks, n_features) 和价格向量 (n_stocks,)
        """
        if idx is None:
            idx = self.current_index
        features = self.feature_data[idx]   # (n_stocks, n_features)
        prices = self.prices.iloc[idx].values  # (n_stocks,)
        return features, prices
    
    def get_next_prices(self) -> np.ndarray:
        """获取下一日价格（用于计算收益）"""
        next_idx = self.current_index + 1
        if next_idx >= self.n_dates:
            return None
        return self.prices.iloc[next_idx].values
    
    def step(self):
        """移动到下一交易日，返回是否到达末尾"""
        self.current_index += 1
        return self.current_index < self.n_dates
    
    @property
    def current_date(self):
        return self.dates[self.current_index] if self.current_index < self.n_dates else None