import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_socket import DataFetcher

PROJECT_ROOT = Path(__file__).parent.parent.parent


def download_stocks(fetcher, stock_list, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for stock in stock_list:
        df_stock = fetcher.get_stock_hist(stock, "20170101", "20260626", "daily", "hfq", use_cache=True)
        df_stock.to_csv(output_dir / f"{stock}_data_2017_2024_hfq.csv", index=False, encoding="utf-8-sig")
        frames.append(df_stock)

    if not frames:
        raise ValueError("Stock list must not be empty")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "stock_data_hs300_2017_2024_hfq.csv", index=False, encoding="utf-8-sig")
    return combined


if __name__ == "__main__":
    fetcher = DataFetcher(cache_root=PROJECT_ROOT / "data" / "raw")
    with open(PROJECT_ROOT / "data" / "stock_list" / "hs300_20260629.txt", "r", encoding="utf-8-sig") as f:
        stock_list = [line.strip() for line in f if line.strip()]
    download_stocks(fetcher, stock_list, PROJECT_ROOT / "data" / "raw")
