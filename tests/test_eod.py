"""Offline close-date gate tests; no provider calls or training."""

from datetime import datetime

import pandas as pd
import pytest

from src.inference.eod import require_close_bar, require_close_capture


def test_close_capture_after_buffer_and_timezone_conversion():
    now = datetime.fromisoformat("2026-09-25T17:00:00+08:00")
    assert str(require_close_capture("2026-09-25", "2026-09-25T08:00:00+00:00", now).date()) == "2026-09-25"


@pytest.mark.parametrize("capture,now", [
    ("2026-09-25T15:59:00+08:00", "2026-09-25T18:00:00+08:00"),
    ("2026-09-25T16:00:00+08:00", "2026-09-25T15:00:00+08:00"),
    ("2026-09-24T17:00:00+08:00", "2026-09-25T18:00:00+08:00"),
    ("2026-09-25T19:00:00+08:00", "2026-09-25T18:00:00+08:00"),
    ("2026-09-25T17:00:00", "2026-09-25T18:00:00+08:00"),
])
def test_close_capture_rejects_partial_stale_future_or_naive(capture, now):
    with pytest.raises(ValueError):
        require_close_capture("2026-09-25", capture, datetime.fromisoformat(now))


def test_close_bar_never_falls_back_or_exposes_future():
    frame = pd.DataFrame({"close": [1., 2., 999.]}, index=pd.to_datetime(["2026-09-23", "2026-09-24", "2026-09-25"]))
    assert require_close_bar(frame, "2026-09-24", "000001").close.tolist() == [1., 2.]
    with pytest.raises(ValueError, match="Missing"):
        require_close_bar(frame, "2026-09-22", "000001")
    with pytest.raises(ValueError, match="duplicate"):
        require_close_bar(pd.concat([frame, frame.iloc[:1]]), "2026-09-24", "000001")
