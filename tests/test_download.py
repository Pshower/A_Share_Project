import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import download_stocks
from src.data.rebuild_history import rebuild_history


def test_download_aggregates_each_stock_once_with_hfq(tmp_path):
    codes = ["000001", "600000"]
    frames = [
        pd.DataFrame({"date": ["2024-01-02", "2024-01-03"],
                      "code": [code, code], "close": [100.0, 110.0]})
        for code in codes
    ]
    fetcher = Mock()
    fetcher.get_stock_hist.side_effect = frames

    combined = download_stocks(fetcher, codes, tmp_path)

    assert fetcher.get_stock_hist.call_count == len(codes)
    for call, code in zip(fetcher.get_stock_hist.call_args_list, codes):
        assert call.args == (code, "20170101", "20260626", "daily", "hfq")
        assert call.kwargs == {"use_cache": True}
    expected = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(combined, expected)
    assert not combined.duplicated(["date", "code"]).any()
    saved = pd.read_csv(tmp_path / "stock_data_hs300_20170101_20260626_hfq.csv",
                        dtype={"code": str})
    pd.testing.assert_frame_equal(saved, expected)
    for code, frame in zip(codes, frames):
        individual = pd.read_csv(tmp_path / f"{code}_data_20170101_20260626_hfq.csv",
                                 dtype={"code": str})
        pd.testing.assert_frame_equal(individual, frame)


def test_empty_stock_list_does_not_overwrite_aggregate(tmp_path):
    aggregate = tmp_path / "stock_data_hs300_2017_2024_hfq.csv"
    aggregate.write_text("existing data", encoding="utf-8")
    fetcher = Mock()
    with pytest.raises(ValueError, match="Stock list must not be empty"):
        download_stocks(fetcher, [], tmp_path)
    fetcher.get_stock_hist.assert_not_called()
    assert aggregate.read_text(encoding="utf-8") == "existing data"


def test_rebuild_uses_local_hfq_files(tmp_path):
    codes = ["000001", "600000"]
    stock_list = tmp_path / "stocks.txt"
    stock_list.write_text("\n".join(codes), encoding="utf-8-sig")
    frames = []
    for code in codes:
        frame = pd.DataFrame({"日期": ["2024-01-02"], "股票代码": [code],
                              "收盘": [100.0]})
        frame.to_csv(tmp_path / f"{code}_data_2017_2024_hfq.csv", index=False)
        frames.append(frame)
    target = tmp_path / "stock_data_hs300_2017_2024_hfq.csv"
    pd.concat([frames[0], *frames]).to_csv(target, index=False)

    rebuilt = rebuild_history(tmp_path, stock_list)

    expected = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(rebuilt, expected)
    pd.testing.assert_frame_equal(pd.read_csv(target, dtype={"股票代码": str}), expected)


def test_rebuild_missing_source_preserves_aggregate(tmp_path):
    stock_list = tmp_path / "stocks.txt"
    stock_list.write_text("000001", encoding="utf-8")
    target = tmp_path / "stock_data_hs300_2017_2024_hfq.csv"
    target.write_text("existing data", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        rebuild_history(tmp_path, stock_list)

    assert target.read_text(encoding="utf-8") == "existing data"
