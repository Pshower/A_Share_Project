import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_socket import DataFetcher

PROJECT_ROOT = Path(__file__).parent.parent.parent
fetcher = DataFetcher(cache_root=PROJECT_ROOT / "data" / "raw")
with open(PROJECT_ROOT / "data" / "stock_list" / "hs300_20260629.txt", "r", encoding="utf-8-sig") as f:
    stock_list = [line.strip() for line in f]

if __name__ == "__main__":
    # df_stock_name = fetcher.get_stock_list()
    # df_stock_name.to_csv(tmp_path / "data" / "stock_list" / "stock_name.csv", index=False, encoding="utf-8-sig")
    # df = fetcher.get_stock_hist("000001", "20170301", "20240528", "daily", "", use_cache=True)
    # df.to_csv(tmp_path / "data" / "stock_list" / "stock_data.csv", index=False, encoding="utf-8-sig")
    df = fetcher.get_stock_hist(stock_list[0], "20170101", "20260626", "daily", "", use_cache=True)

    for stock in stock_list:
        df_stock = fetcher.get_stock_hist(stock, "20170101", "20260626", "daily", "hfq", use_cache=True)
        df_stock.to_csv(PROJECT_ROOT / "data" / "raw" / f"{stock}_data_2017_2024_hfq.csv", index=False, encoding="utf-8-sig")
        df = pd.concat([df, df_stock], ignore_index=True)
    
    df.to_csv(PROJECT_ROOT / "data" / "raw" / "stock_data_hs300_2017_2024_hfq.csv", index=False, encoding="utf-8-sig")