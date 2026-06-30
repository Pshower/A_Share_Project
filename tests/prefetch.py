# tests/prefetch.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径，确保可以导入 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_socket import DataFetcher

fetcher = DataFetcher()

# 生成缓存（所有测试中会用到的接口）
fetcher.get_sse_summary(use_cache=True)
fetcher.get_all_a_spot(use_cache=True, max_age=3600)   # 缓存有效期设长一些
fetcher.get_stock_hist("000001", "20170301", "20240528", "daily", "", use_cache=True)
fetcher.get_market_fund_flow(use_cache=True)
fetcher.get_industry_board_spot(use_cache=True, max_age=3600)
fetcher.get_stock_list(use_cache=True)