import os
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data import data_socket


@pytest.mark.parametrize("method,args,api,expected", [
    ("get_szse_summary", ["20240101"], "stock_szse_summary", {"date": "20240101"}),
    ("get_szse_area_summary", ["202401"], "stock_szse_area_summary", {"date": "202401"}),
    ("get_individual_fund_flow", ["000001", "sz"], "stock_individual_fund_flow", {"stock": "000001", "market": "sz"}),
    ("get_sector_fund_flow_rank", ["x", "y"], "stock_sector_fund_flow_rank", {"indicator": "x", "sector_type": "y"}),
    ("get_north_south_fund_flow", ["x"], "stock_hsgt_hist_em", {"symbol": "x"}),
    ("get_industry_board_cons", ["x"], "stock_board_industry_cons_em", {"symbol": "x"}),
    ("get_concept_board_cons", ["x"], "stock_board_concept_cons_em", {"symbol": "x"}),
    ("get_stock_individual_info", ["000001"], "stock_individual_info_em", {"symbol": "000001"}),
    ("get_stock_bid_ask", ["000001"], "stock_bid_ask_em", {"symbol": "000001"}),
])
def test_parameter_forwarding_without_network(tmp_path, monkeypatch, method, args, api, expected):
    fake = Mock(return_value=pd.DataFrame({"value": [1]}))
    monkeypatch.setattr(data_socket.ak, api, fake)
    fetcher = data_socket.DataFetcher(tmp_path)
    assert len(getattr(fetcher, method)(*args)) == 1
    fake.assert_called_once_with(**expected)


def test_cache_hit_expiry_corruption_and_retry(tmp_path, monkeypatch):
    fetcher = data_socket.DataFetcher(tmp_path)
    frame = pd.DataFrame({"value": [3]})
    fake = Mock(side_effect=[RuntimeError("temporary"), frame, frame, frame])
    sleep = Mock()
    monkeypatch.setattr(data_socket.time, "sleep", sleep)
    first = fetcher._fetch_with_cache("test", {}, fake, retries=2)
    pd.testing.assert_frame_equal(first, fetcher._fetch_with_cache("test", {}, fake))
    assert fake.call_count == 2 and sleep.call_count == 1
    cache = fetcher._get_cache_path("test", {})
    os.utime(cache, (0, 0))
    fetcher._fetch_with_cache("test", {}, fake, max_age=1)
    assert fake.call_count == 3
    cache.write_bytes(b"corrupt")
    fetcher._fetch_with_cache("test", {}, fake)
    assert fake.call_count == 4
    with pytest.raises(RuntimeError, match="permanent"):
        fetcher._fetch_with_cache("bad", {}, Mock(side_effect=RuntimeError("permanent")), retries=2)
