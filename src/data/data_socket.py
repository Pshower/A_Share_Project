# src/data/data_socket.py
import os
import hashlib
import json
import random
from pathlib import Path
import time
from typing import Optional, Union, Any
from requests.exceptions import JSONDecodeError, ConnectionError
import pandas as pd
import akshare as ak


class DataFetcher:
    """
    数据接口类，封装 AKShare 并支持自动缓存到本地
    缓存根目录：项目顶层目录下的 data/raw/
    """

    def __init__(self, cache_root: Optional[Union[str, Path]] = None):
        """
        初始化数据接口
        :param cache_root: 缓存根目录，默认为项目顶层目录下的 data/raw
        """
        if cache_root is None:
            # 项目顶层目录：假设当前文件在 src/data/ 下，向上两级为项目根目录
            project_root = Path(__file__).parent.parent.parent
            cache_root = project_root / "data" / "raw"
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, func_name: str, params: dict) -> Path:
        """根据函数名和参数生成唯一的缓存文件路径"""
        # 将参数排序后转为字符串，计算哈希作为文件名的一部分
        param_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
        filename = f"{func_name}_{param_hash}.parquet"
        return self.cache_root / filename

    def _load_cache(self, cache_path: Path, max_age: Optional[float] = None) -> Optional[pd.DataFrame]:
        """加载缓存，如果文件存在且未过期则返回 DataFrame，否则返回 None"""
        if not cache_path.exists():
            return None
        if max_age is not None:
            file_age = pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit='s')
            if file_age.total_seconds() > max_age:
                return None
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            return None

    def _save_cache(self, cache_path: Path, data: pd.DataFrame) -> None:
        """保存数据到缓存"""
        data.to_parquet(cache_path, index=False)

    def _fetch_with_cache(self, func_name: str, params: dict, fetcher_func,
                      use_cache: bool = True, max_age: Optional[float] = None,
                      retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
        """
        通用的带缓存和自动重试的获取方法
        """
        if use_cache:
            cache_path = self._get_cache_path(func_name, params)
            df = self._load_cache(cache_path, max_age)
            if df is not None:
                return df

        # 重试循环
        last_exception = None
        for attempt in range(retries):
            try:
                # 调用接口（如果 params 为空则调用无参函数）
                df = fetcher_func(**params) if params else fetcher_func()
                break
            except (JSONDecodeError, ConnectionError, Exception) as e:
                last_exception = e
                if attempt == retries - 1:
                    raise  # 最后一次失败则抛出异常
                wait = backoff * (attempt + 1) + random.uniform(0, 1)
                print(f"⚠️ 请求失败 (尝试 {attempt+1}/{retries})，等待 {wait:.2f}s 后重试: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError("所有重试均失败") from last_exception

        if use_cache and df is not None and not df.empty:
            self._save_cache(cache_path, df)
        return df

    # ==================== 市场总貌 ====================
    def get_sse_summary(self, use_cache: bool = True, max_age: Optional[float] = 86400) -> pd.DataFrame:
        """
        上海证券交易所-股票数据总貌（最近交易日）
        :param use_cache: 是否使用缓存
        :param max_age: 缓存最大有效期（秒），默认 1 天
        """
        return self._fetch_with_cache(
            func_name="sse_summary",
            params={},
            fetcher_func=ak.stock_sse_summary,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_szse_summary(self, date: str, use_cache: bool = True, max_age: Optional[float] = 86400) -> pd.DataFrame:
        """
        深圳证券交易所-市场总貌-证券类别统计
        :param date: 统计日期，格式 "20200619"
        :param use_cache: 是否使用缓存
        :param max_age: 缓存最大有效期（秒），默认 1 天
        """
        params = {"date": date}
        return self._fetch_with_cache(
            func_name="szse_summary",
            params=params,
            fetcher_func=ak.stock_szse_summary,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_szse_area_summary(self, date: str, use_cache: bool = True, max_age: Optional[float] = 86400) -> pd.DataFrame:
        """
        深圳证券交易所-地区交易排序
        :param date: 年月，格式 "202203"
        """
        params = {"date": date}
        return self._fetch_with_cache(
            func_name="szse_area_summary",
            params=params,
            fetcher_func=ak.stock_szse_area_summary,
            use_cache=use_cache,
            max_age=max_age
        )

    # ==================== 实时行情 ====================
    def get_all_a_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """
        沪深京 A 股实时行情（东方财富）
        :param use_cache: 是否使用缓存
        :param max_age: 缓存最大有效期（秒），默认 5 分钟（实时数据短缓存）
        """
        # 实时数据不适合长期缓存，这里允许短时间缓存避免频繁请求
        return self._fetch_with_cache(
            func_name="all_a_spot",
            params={},
            fetcher_func=ak.stock_zh_a_spot_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_sh_a_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """沪 A 股实时行情"""
        return self._fetch_with_cache(
            func_name="sh_a_spot",
            params={},
            fetcher_func=ak.stock_sh_a_spot_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_sz_a_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """深 A 股实时行情"""
        return self._fetch_with_cache(
            func_name="sz_a_spot",
            params={},
            fetcher_func=ak.stock_sz_a_spot_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_bj_a_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """京 A 股实时行情"""
        return self._fetch_with_cache(
            func_name="bj_a_spot",
            params={},
            fetcher_func=ak.stock_bj_a_spot_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_cy_a_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """创业板实时行情"""
        return self._fetch_with_cache(
            func_name="cy_a_spot",
            params={},
            fetcher_func=ak.stock_cy_a_spot_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_kc_a_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """科创板实时行情"""
        return self._fetch_with_cache(
            params={},
            func_name="kc_a_spot",
            fetcher_func=ak.stock_kc_a_spot_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_st_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """风险警示板（ST）实时行情"""
        return self._fetch_with_cache(
            func_name="st_spot",
            params={},
            fetcher_func=ak.stock_zh_a_st_em,
            use_cache=use_cache,
            max_age=max_age
        )

    # ==================== 历史行情 ====================
    def get_stock_hist(self, symbol: str, start_date: str, end_date: str,
                       period: str = "daily", adjust: str = "",
                       use_cache: bool = True, max_age: Optional[float] = None,
                       *, timeout: Optional[float] = None, retries: int = 3) -> pd.DataFrame:
        """
        单个股票历史行情（日/周/月）
        :param symbol: 股票代码，如 "000001"
        :param start_date: 开始日期 "20210101"
        :param end_date: 结束日期 "20211231"
        :param period: 周期 "daily"/"weekly"/"monthly"
        :param adjust: 复权 ""(不复权)/"qfq"(前复权)/"hfq"(后复权)
        :param use_cache: 是否使用缓存
        :param max_age: 缓存最大有效期（秒），None 表示永久（除非手动删除）
        """
        params = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "period": period,
            "adjust": adjust
        }
        if timeout is not None:
            if timeout <= 0:
                raise ValueError("timeout must be positive")
            params["timeout"] = timeout
        return self._fetch_with_cache(
            func_name="stock_hist",
            params=params,
            fetcher_func=ak.stock_zh_a_hist,
            use_cache=use_cache,
            max_age=max_age,
            retries=retries
        )

    # ==================== 资金流向 ====================
    def get_market_fund_flow(self, use_cache: bool = True, max_age: Optional[float] = 86400) -> pd.DataFrame:
        """大盘资金流向（历史数据）"""
        return self._fetch_with_cache(
            func_name="market_fund_flow",
            params={},
            fetcher_func=ak.stock_market_fund_flow,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_individual_fund_flow(self, stock: str, market: str,
                                 use_cache: bool = True, max_age: Optional[float] = 86400) -> pd.DataFrame:
        """
        个股资金流向（近100个交易日）
        :param stock: 股票代码，如 "000425"
        :param market: 市场 "sh"/"sz"/"bj"
        """
        params = {"stock": stock, "market": market}
        return self._fetch_with_cache(
            func_name="individual_fund_flow",
            params=params,
            fetcher_func=ak.stock_individual_fund_flow,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_sector_fund_flow_rank(self, indicator: str = "今日", sector_type: str = "行业资金流",
                                  use_cache: bool = True, max_age: Optional[float] = 3600) -> pd.DataFrame:
        """
        板块资金流排名
        :param indicator: "今日"/"5日"/"10日"
        :param sector_type: "行业资金流"/"概念资金流"/"地域资金流"
        """
        params = {"indicator": indicator, "sector_type": sector_type}
        return self._fetch_with_cache(
            func_name="sector_fund_flow_rank",
            params=params,
            fetcher_func=ak.stock_sector_fund_flow_rank,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_north_south_fund_flow(self, symbol: str = "北向资金",
                                  use_cache: bool = True, max_age: Optional[float] = 3600) -> pd.DataFrame:
        """
        沪深港通资金流向历史数据
        :param symbol: "北向资金"/"南向资金"/"沪股通"/"深股通"/"港股通沪"/"港股通深"
        """
        params = {"symbol": symbol}
        return self._fetch_with_cache(
            func_name="north_south_fund_flow",
            params=params,
            fetcher_func=ak.stock_hsgt_hist_em,
            use_cache=use_cache,
            max_age=max_age
        )

    # ==================== 板块行情 ====================
    def get_industry_board_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """行业板块实时行情"""
        return self._fetch_with_cache(
            func_name="industry_board_spot",
            params={},
            fetcher_func=ak.stock_board_industry_name_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_concept_board_spot(self, use_cache: bool = True, max_age: Optional[float] = 300) -> pd.DataFrame:
        """概念板块实时行情"""
        return self._fetch_with_cache(
            func_name="concept_board_spot",
            params={},
            fetcher_func=ak.stock_board_concept_name_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_industry_board_cons(self, symbol: str, use_cache: bool = True, max_age: Optional[float] = 3600) -> pd.DataFrame:
        """
        指定行业板块的成分股
        :param symbol: 板块名称或代码，如 "小金属" 或 "BK1027"
        """
        params = {"symbol": symbol}
        return self._fetch_with_cache(
            func_name="industry_board_cons",
            params=params,
            fetcher_func=ak.stock_board_industry_cons_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_concept_board_cons(self, symbol: str, use_cache: bool = True, max_age: Optional[float] = 3600) -> pd.DataFrame:
        """
        指定概念板块的成分股
        :param symbol: 板块名称或代码，如 "融资融券" 或 "BK0655"
        """
        params = {"symbol": symbol}
        return self._fetch_with_cache(
            func_name="concept_board_cons",
            params=params,
            fetcher_func=ak.stock_board_concept_cons_em,
            use_cache=use_cache,
            max_age=max_age
        )

    # ==================== 股票基础信息 ====================
    def get_stock_list(self, use_cache: bool = True, max_age: Optional[float] = 86400) -> pd.DataFrame:
        """全部 A 股股票代码及名称"""
        return self._fetch_with_cache(
            func_name="stock_list",
            params={},
            fetcher_func=ak.stock_info_a_code_name,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_stock_individual_info(self, symbol: str, use_cache: bool = True, max_age: Optional[float] = 3600) -> pd.DataFrame:
        """
        个股基本信息（东方财富）
        :param symbol: 股票代码，如 "000001"
        """
        params = {"symbol": symbol}
        return self._fetch_with_cache(
            func_name="stock_individual_info",
            params=params,
            fetcher_func=ak.stock_individual_info_em,
            use_cache=use_cache,
            max_age=max_age
        )

    def get_stock_bid_ask(self, symbol: str, use_cache: bool = True, max_age: Optional[float] = 30) -> pd.DataFrame:
        """
        个股买卖五档报价（东方财富）—— 更新频率极高，建议短缓存或禁用缓存
        :param symbol: 股票代码，如 "000001"
        """
        # 五档行情变化非常快，默认只缓存30秒
        params = {"symbol": symbol}
        return self._fetch_with_cache(
            func_name="stock_bid_ask",
            params=params,
            fetcher_func=ak.stock_bid_ask_em,
            use_cache=use_cache,
            max_age=max_age
        )

    # ==================== 辅助功能 ====================
    def clear_cache(self, func_name: Optional[str] = None) -> None:
        """
        清除缓存文件
        :param func_name: 指定函数名清除对应缓存，不传则清除所有缓存
        """
        if func_name is None:
            for f in self.cache_root.glob("*.parquet"):
                f.unlink()
        else:
            for f in self.cache_root.glob(f"{func_name}_*.parquet"):
                f.unlink()
