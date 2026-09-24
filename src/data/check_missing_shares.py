from pathlib import Path
import numpy as np
import pandas as pd

clean_dir = Path("data/clean")
stock_list_path = Path("data/stock_list/hs300_20260629.txt")
with open(stock_list_path) as f:
    expected = [line.strip() for line in f if line.strip()]

# 读取合并文件
df = pd.read_parquet(clean_dir / "all_stocks_features.parquet")
actual = df['code'].unique().tolist()

missing = set(expected) - set(actual)
print("缺失的股票代码：", missing)

raw_dir = Path("data/raw")
for code in missing:
    files = list(raw_dir.glob(f"{code}_data_*_hfq.csv"))
    if files:
        df = pd.read_csv(files[0], parse_dates=['日期'])
        print(f"{code} 数据行数: {len(df)}")
        print(f"日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
        print(f"开盘价正数行数: {(df['开盘'] > 0).sum()}")
    else:
        print(f"{code} 未找到数据文件")