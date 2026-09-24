# tests/test_data_socket.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径，确保可以导入 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.data.data_socket import DataFetcher

pytestmark = pytest.mark.network


@pytest.fixture
def fetcher(tmp_path):
    """返回一个使用临时缓存目录的 DataFetcher 实例"""
    return DataFetcher(cache_root=tmp_path / "data" / "raw")


def test_cache_root_creation(fetcher):
    """测试缓存目录是否自动创建"""
    assert fetcher.cache_root.exists()
    assert fetcher.cache_root.is_dir()


def test_get_sse_summary(fetcher):
    """测试获取上交所总貌数据"""
    df = fetcher.get_sse_summary(use_cache=True, max_age=60)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # 预期包含的列：项目, 股票, 科创板, 主板
    expected_cols = {"项目", "股票", "科创板", "主板"}
    assert expected_cols.issubset(df.columns)


def test_get_all_a_spot(fetcher):
    """测试获取全部A股实时行情"""
    df = fetcher.get_all_a_spot(use_cache=True, max_age=30)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # 实时行情应包含的列：序号, 代码, 名称, 最新价, 涨跌幅, ...
    assert "代码" in df.columns
    assert "名称" in df.columns
    assert "最新价" in df.columns


def test_get_stock_hist_cache(fetcher):
    """测试历史行情缓存功能（两次调用应返回相同结果，第二次从缓存读取）"""
    symbol = "000001"
    start = "20170301"
    end = "20240528"
    period = "daily"
    adjust = ""
    df1 = fetcher.get_stock_hist(symbol, start, end, period, adjust, use_cache=True, max_age=None)
    df2 = fetcher.get_stock_hist(symbol, start, end, period, adjust, use_cache=True, max_age=None)
    pd.testing.assert_frame_equal(df1, df2)
    # 检查缓存文件是否存在
    cache_files = list(fetcher.cache_root.glob("stock_hist_*.parquet"))
    assert len(cache_files) >= 1


def test_get_market_fund_flow(fetcher):
    """测试大盘资金流向数据"""
    df = fetcher.get_market_fund_flow(use_cache=True, max_age=60)
    assert isinstance(df, pd.DataFrame)
    # 应包含常用列
    assert "日期" in df.columns or "上证-收盘价" in df.columns
    assert not df.empty


def test_get_industry_board_spot(fetcher):
    """测试行业板块实时行情"""
    df = fetcher.get_industry_board_spot(use_cache=True, max_age=30)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "板块名称" in df.columns or "板块代码" in df.columns


def test_get_stock_list(fetcher):
    """测试获取全部A股列表"""
    df = fetcher.get_stock_list(use_cache=True, max_age=3600)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "code" in df.columns or "代码" in df.columns
    assert "name" in df.columns or "名称" in df.columns


def test_clear_cache(fetcher):
    """测试清除缓存功能"""
    # 先获取数据产生缓存
    fetcher.get_sse_summary(use_cache=True)
    fetcher.get_all_a_spot(use_cache=True)
    assert len(list(fetcher.cache_root.glob("*.parquet"))) >= 2

    # 清除所有缓存
    fetcher.clear_cache()
    assert len(list(fetcher.cache_root.glob("*.parquet"))) == 0

    # 再测试按函数名清除
    fetcher.get_sse_summary(use_cache=True)
    fetcher.get_all_a_spot(use_cache=True)
    fetcher.clear_cache(func_name="sse_summary")
    sse_files = list(fetcher.cache_root.glob("sse_summary_*.parquet"))
    spot_files = list(fetcher.cache_root.glob("all_a_spot_*.parquet"))
    assert len(sse_files) == 0
    assert len(spot_files) == 1


# 可选：跳过实时行情测试，避免频繁请求
@pytest.mark.skip(reason="实时行情请求耗时较长，按需启用")
def test_get_all_a_spot_live(fetcher):
    df = fetcher.get_all_a_spot(use_cache=False)
    assert not df.empty


if __name__ == "__main__":
    # 支持直接运行此文件进行简易测试（不依赖 pytest）
    pytest.main([__file__, "-v", "--tb=short"])
