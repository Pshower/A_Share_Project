"""Optional live integration checks; excluded from the default test run."""

import akshare as ak
import pytest

pytestmark = pytest.mark.network


@pytest.mark.parametrize("name,kwargs", [
    ("stock_zh_a_spot_em", {}),
    ("stock_zh_a_hist", {"symbol": "000001", "period": "daily",
                         "start_date": "20170301", "end_date": "20240528", "adjust": ""}),
    ("stock_board_industry_name_em", {}),
    ("stock_info_a_code_name", {}),
])
def test_live_connection(name, kwargs):
    result = getattr(ak, name)(**kwargs)
    assert result is not None and not result.empty
