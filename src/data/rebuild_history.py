"""Rebuild the historical aggregate from local per-stock HFQ CSV files."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def rebuild_history(raw_dir, stock_list_path):
    raw_dir = Path(raw_dir)
    codes = Path(stock_list_path).read_text(encoding="utf-8-sig").splitlines()
    codes = [code.strip() for code in codes if code.strip()]
    if not codes or len(codes) != len(set(codes)):
        raise ValueError("Stock list must be nonempty and unique")

    frames = []
    for code in codes:
        source = raw_dir / f"{code}_data_2017_2024_hfq.csv"
        frame = pd.read_csv(source, dtype={"股票代码": str})
        if frame.empty or not frame["股票代码"].eq(code).all():
            raise ValueError(f"Empty data or mismatched stock code: {source}")
        if frames and list(frame.columns) != list(frames[0].columns):
            raise ValueError(f"Inconsistent columns: {source}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(combined["日期"], errors="raise")
    if dates.isna().any() or combined.duplicated(["股票代码", "日期"]).any():
        raise ValueError("Missing dates or duplicate stock/date rows")

    target = raw_dir / "stock_data_hs300_2017_2024_hfq.csv"
    previous_rows = len(pd.read_csv(target, usecols=["日期"])) if target.exists() else 0
    # Validate serialization before replacing the existing aggregate.
    temporary = target.with_suffix(".csv.tmp")
    combined.to_csv(temporary, index=False, encoding="utf-8-sig")
    restored = pd.read_csv(temporary, dtype={"股票代码": str})
    pd.testing.assert_frame_equal(restored, combined)
    temporary.replace(target)
    print(f"stocks={len(codes)} previous_rows={previous_rows} rows={len(combined)}")
    print(f"dates={dates.min().date()}..{dates.max().date()} duplicates=0")
    print(f"Saved: {target}")
    return combined


if __name__ == "__main__":
    rebuild_history(
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "stock_list" / "hs300_20260629.txt",
    )
