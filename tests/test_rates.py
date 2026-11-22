"""The curve refresher must be conservative: append only what is newer, refuse schema drift,
and prove the file it wrote is the one the nightly job reads -- with the same reader.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tools import rates

COLUMNS = ["日期", "中国-国债收益率2年", "中国-国债收益率5年", "中国-国债收益率10年",
           "中国-国债收益率30年", "中国-国债收益率10年-2年", "中国GDP平减指数",
           "美国国债收益率2年", "美国国债收益率5年", "美国国债收益率10年",
           "美国国债收益率30年", "美国国债收益率10年-2年", "美国GDP平减指数"]


def curve_file(root):
    path = root / "yields" / "china_treasury.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(COLUMNS)
    rows = ["2026-11-18," + ",".join(["1.44"] * 12),
            "2026-11-19," + ",".join(["1.45"] * 12),
            "2026-11-20," + ",".join(["1.5803"] * 12)]
    path.write_text("\ufeff" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def fetched(dates):
    data = {"日期": [str(day) for day in dates]}
    for column in COLUMNS[1:]:
        data[column] = ["1.44"] * len(dates)
    return pd.DataFrame(data)


def test_refresh_appends_only_newer_rows_and_keeps_the_bom(tmp_path):
    path = curve_file(tmp_path)
    frame = fetched(["2026-11-19", "2026-11-20", "2026-11-21", "2026-11-23"])
    result = rates.refresh(online=tmp_path, frame=frame)
    assert result["status"] == "appended"
    assert result["appended"] == 2                          # 11-21 and 11-23; the overlap stays put
    assert result["last_date"] == "2026-11-23"
    text = path.read_text(encoding="utf-8-sig")
    assert text.startswith("\ufeff") or path.read_bytes()[:3] == b"\xef\xbb\xbf"
    assert "2026-11-21,1.44" in text
    assert text.count("\n") == 6                            # header + 5 data rows, no duplicates
    assert not path.with_name(path.name + ".tmp").exists()


def test_refresh_verifies_with_the_nightly_reader(tmp_path):
    curve_file(tmp_path)
    frame = fetched(["2026-11-21"])
    result = rates.refresh(online=tmp_path, frame=frame)
    from tools import asset_panel
    curve = asset_panel.curve(tmp_path)
    assert str(curve.index[-1].date()) == result["last_date"]
    assert curve["y10"].iloc[-1] == pytest.approx(1.44)      # column 3 of the file, as curve reads it


def test_a_schema_drift_is_refused_and_the_file_is_untouched(tmp_path):
    path = curve_file(tmp_path)
    before = path.read_text(encoding="utf-8-sig")
    frame = fetched(["2026-11-21"])
    frame.columns = ["日期"] + ["改名%d" % i for i in range(12)]
    with pytest.raises(ValueError, match="列名已变"):
        rates.refresh(online=tmp_path, frame=frame)
    assert path.read_text(encoding="utf-8-sig") == before


def test_no_newer_rows_is_a_clean_noop(tmp_path):
    path = curve_file(tmp_path)
    before = path.read_text(encoding="utf-8-sig")
    result = rates.refresh(online=tmp_path, frame=fetched(["2026-11-19", "2026-11-20"]))
    assert result["status"] == "fresh"
    assert result["appended"] == 0
    assert path.read_text(encoding="utf-8-sig") == before
