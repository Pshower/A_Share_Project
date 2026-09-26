"""Conservative local close-bar checks, not exchange calendar certification."""

from datetime import datetime, time

import pandas as pd

from src.data.online import CHINA


def require_close_capture(as_of, received_at, now):
    date = pd.Timestamp(as_of)
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("Close date must be a timezone-free calendar date")
    captured = datetime.fromisoformat(received_at)
    if captured.tzinfo is None or now.tzinfo is None:
        raise ValueError("Capture and current timestamps must include timezone")
    # 16:00 is our conservative operational buffer, not a vendor finality guarantee.
    ready_after = datetime.combine(date.date(), time(16), tzinfo=CHINA)
    if now < ready_after or captured < ready_after:
        raise ValueError("Close-date bars must be captured after 16:00 Asia/Shanghai")
    if captured > now:
        raise ValueError("Capture timestamp is in the future")
    return date


def require_close_bar(frame, as_of, code):
    date = pd.Timestamp(as_of)
    if frame.index.has_duplicates or date not in frame.index:
        raise ValueError(f"Missing or duplicate requested close-date bar for {code}")
    return frame.loc[:date].copy()
