import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_socket import DataFetcher

PROJECT_ROOT = Path(__file__).parent.parent.parent


def download_stocks(fetcher, stock_list, output_dir, start_date="20170101", end_date="20260626"):
    stock_list = [str(code).strip().zfill(6) for code in stock_list]
    if not stock_list or len(set(stock_list)) != len(stock_list):
        raise ValueError("Stock list must not be empty or contain duplicates")
    if any(len(code) != 6 or not code.isdigit() for code in stock_list):
        raise ValueError("Invalid stock code")
    if pd.to_datetime(start_date, format="%Y%m%d") > pd.to_datetime(end_date, format="%Y%m%d"):
        raise ValueError("Invalid download date range")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for stock in stock_list:
        df_stock = fetcher.get_stock_hist(stock, start_date, end_date, "daily", "hfq", use_cache=True)
        df_stock.to_csv(output_dir / f"{stock}_data_{start_date}_{end_date}_hfq.csv", index=False, encoding="utf-8-sig")
        frames.append(df_stock)

    if not frames:
        raise ValueError("Stock list must not be empty")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / f"stock_data_hs300_{start_date}_{end_date}_hfq.csv", index=False, encoding="utf-8-sig")
    return combined


if __name__ == "__main__":
    import argparse
    from preprocess import read_config, resolve_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/research.json")
    args = parser.parse_args()
    settings = read_config(args.config)
    raw_dir = resolve_path(settings["dataset"]["raw_dir"])
    stock_list_path = resolve_path(settings["dataset"]["stock_list_path"])
    fetcher = DataFetcher(cache_root=raw_dir)
    with open(stock_list_path, "r", encoding="utf-8-sig") as f:
        stock_list = [line.strip() for line in f if line.strip()]
    download_stocks(fetcher, stock_list, raw_dir, **settings["download"])
