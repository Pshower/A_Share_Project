# src/data/preprocess.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    股票数据预处理类
    - 读取后复权CSV
    - 清洗（缺失值、异常值）
    - 特征工程（收益率、技术指标）
    - 标准化（Z-Score）
    - 按时间划分数据集
    - 保存为Parquet格式
    """

    def __init__(self,
                 raw_dir: Path,
                 clean_dir: Path,
                 stock_list_path: Path,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.stock_list_path = Path(stock_list_path)
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        # 加载股票代码列表
        self.stock_list = self._load_stock_list()

    def _load_stock_list(self) -> List[str]:
        """从文本文件读取股票代码（每行一个）"""
        with open(self.stock_list_path, 'r', encoding='utf-8') as f:
            codes = [line.strip() for line in f if line.strip()]
        return codes

    def _read_stock_data(self, code: str) -> pd.DataFrame:
        """读取单只股票的后复权CSV文件"""
        # 根据实际文件名模式修改
        file_pattern = f"{code}_data_*_hfq.csv"  # 或精确匹配 "2017_2024"
        candidates = list(self.raw_dir.glob(f"{code}_data_*_hfq.csv"))
        if not candidates:
            # 尝试不带hfq后缀的？但这里我们强制要求后复权
            raise FileNotFoundError(f"未找到股票 {code} 的后复权数据文件")
        file_path = candidates[0]  # 取第一个匹配
        df = pd.read_csv(file_path, parse_dates=['日期'])
        return df

    def clean_single_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗单只股票数据
        - 重命名列（统一为英文）
        - 按日期排序
        - 去除全为NaN的行
        - 处理缺失值（前向填充）
        - 检查异常值（价格非正等）
        """
        # 重命名列（根据实际列名调整）
        rename_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        df.rename(columns=rename_map, inplace=True)

        # 确保日期排序
        df = df.sort_values('date').reset_index(drop=True)

        # 去除停牌日：若 close 为 NaN 或成交量0，视为缺失
        # 但后复权数据通常每日有值，成交量可能为0（停牌），保留这些行但标记
        # 我们前向填充close，其余特征也可填充
        df['close'] = df['close'].ffill()
        # 其他价格列也可前向填充
        for col in ['open', 'high', 'low']:
            df[col] = df[col].ffill()
        # 成交量可能为0，保持0
        df['volume'] = df['volume'].fillna(0)
        df['amount'] = df['amount'].fillna(0)
        df['turnover'] = df['turnover'].fillna(0)

        # 删除开盘价为0的异常行（可能上市前数据）
        df = df[df['open'] > 0]

        # 过滤日期范围
        if self.start_date:
            df = df[df['date'] >= self.start_date]
        if self.end_date:
            df = df[df['date'] <= self.end_date]

        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术特征
        - 日收益率、对数收益率
        - 滚动波动率（20日）
        - 价格相对均线（5,10,20,60日）
        - RSI（14日）
        - MACD
        - 布林带上下轨
        """
        # 收益
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # 滚动波动率（年化可调，这里用日度）
        df['volatility_20'] = df['return'].rolling(20).std()

        # 均线
        for window in [5, 10, 20, 60]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'close_ma_ratio_{window}'] = df['close'] / df[f'ma_{window}'] - 1

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 成交额占比（相对过去20日平均）
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        return df

    def normalize_features(self, df: pd.DataFrame, fit: bool = True,
                           stats: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
        """
        对特征列进行Z-Score标准化（按股票自身历史）
        :param df: 包含特征列的DataFrame
        :param fit: True时计算均值和标准差；False时使用传入的stats
        :param stats: 包含均值和标准差的字典，用于变换
        :return: 标准化后的DataFrame, 统计量字典
        """
        # 需要标准化的列（排除日期、价格、非数值列）
        feature_cols = [col for col in df.columns if col not in ['date', 'code'] and
                        df[col].dtype in ['float64', 'int64']]

        if fit:
            stats = {}
            for col in feature_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:
                    std = 1e-8
                stats[col] = {'mean': mean, 'std': std}
                df[col] = (df[col] - mean) / std
        else:
            if stats is None:
                raise ValueError("stats must be provided when fit=False")
            for col in feature_cols:
                mean = stats[col]['mean']
                std = stats[col]['std']
                df[col] = (df[col] - mean) / std

        return df, stats

    def process_all_stocks(self, normalize: bool = True,
                           save_combined: bool = True) -> pd.DataFrame:
        """
        处理所有股票，返回合并后的DataFrame（按日期对齐）
        - 每个股票处理完后，统一日期索引，合并为面板数据
        - 保存每个股票的清洗特征到clean目录
        - 可选保存合并后的全量数据
        """
        all_dfs = []
        for code in self.stock_list:
            try:
                df = self._read_stock_data(code)
                df = self.clean_single_stock(df)
                df = self.add_features(df)
                df['code'] = code  # 添加股票代码列

                # 标准化（如果启用）
                if normalize:
                    df, _ = self.normalize_features(df, fit=True)

                # 保存单个股票的清洗结果（带日期和code，不含标准化？含标准化？建议保存标准化后的）
                save_path = self.clean_dir / f"{code}_features.parquet"
                df.to_parquet(save_path, index=False)

                all_dfs.append(df)
                print(f"✅ {code} 处理完成，共 {len(df)} 条记录")
            except Exception as e:
                print(f"❌ {code} 处理失败：{e}")

        if save_combined and all_dfs:
            # 合并所有股票数据，按日期对齐（outer join）
            combined = pd.concat(all_dfs, ignore_index=True)
            # 保存合并数据（便于后续使用）
            combined_path = self.clean_dir / "all_stocks_features.parquet"
            combined.to_parquet(combined_path, index=False)
            print(f"📦 合并数据保存至 {combined_path}")
            return combined
        else:
            return None

    def split_time_series(self, df: pd.DataFrame,
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分数据集（保留时间顺序）
        :param df: 包含date列的DataFrame
        :return: train_df, val_df, test_df
        """
        dates = sorted(df['date'].unique())
        n = len(dates)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]

        train_df = df[df['date'].isin(train_dates)]
        val_df = df[df['date'].isin(val_dates)]
        test_df = df[df['date'].isin(test_dates)]

        return train_df, val_df, test_df


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    # 配置路径（根据实际项目调整）
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
    STOCK_LIST_FILE = PROJECT_ROOT / "data" / "stock_list" / "hs300_20260629.txt"

    preprocessor = DataPreprocessor(
        raw_dir=RAW_DIR,
        clean_dir=CLEAN_DIR,
        stock_list_path=STOCK_LIST_FILE,
        start_date="2017-01-01",
        end_date="2024-12-31"
    )

    # 执行处理
    combined_df = preprocessor.process_all_stocks(normalize=True, save_combined=True)

    if combined_df is not None:
        # 划分数据集
        train, val, test = preprocessor.split_time_series(combined_df, train_ratio=0.7, val_ratio=0.15)
        # 保存划分结果（含股票代码和日期）
        train.to_parquet(CLEAN_DIR / "train.parquet", index=False)
        val.to_parquet(CLEAN_DIR / "val.parquet", index=False)
        test.to_parquet(CLEAN_DIR / "test.parquet", index=False)
        print("🎯 数据集划分完成并保存。")