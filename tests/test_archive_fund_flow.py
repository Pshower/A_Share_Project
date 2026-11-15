"""The fund-flow archive is append-only and deduped; a rerun must not rewrite or double rows.

It is the only tool whose scheduling outlived the branch cut -- the weekly job calls it, so if it
is missing the job fails silently on a Saturday evening.  These tests cover the parts that decide
what a rerun does; ``fetch`` itself needs the network and is left to the job.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tools import archive_fund_flow as flow


def test_market_of_routes_each_board():
    assert flow.market_of("600519") == "sh"
    assert flow.market_of("688981") == "sh"
    assert flow.market_of("000001") == "sz"
    assert flow.market_of("300750") == "sz"


def test_the_default_universe_is_the_spot_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "a_share_spot.json"
    snapshot.write_text(json.dumps({"rows": [{"code": "600519"}, {"code": "1"}]}), encoding="utf-8")
    monkeypatch.setattr(flow, "UNIVERSE", snapshot)
    assert flow.universe_codes() == ["600519", "000001"]
    assert flow.universe_codes(1) == ["600519"]


def test_an_empty_archive_reads_as_the_declared_columns(tmp_path):
    frame = flow.existing(tmp_path / "missing.parquet")
    assert list(frame.columns) == list(flow.KEEP)
    assert frame.empty


def test_an_existing_archive_is_returned_unchanged(tmp_path):
    path = tmp_path / "index.parquet"
    pd.DataFrame([{column: 0 for column in flow.KEEP}]).to_parquet(path, index=False)
    frame = flow.existing(path)
    assert len(frame) == 1
    assert list(frame.columns) == list(flow.KEEP)


def test_the_skip_target_is_the_trading_day_not_the_stores_own_maximum(tmp_path, monkeypatch):
    """The archive froze because a code was skipped once it reached the store's max date.

    Every healthy code sat at that same maximum, so after the first capture nothing was ever
    fetched again.  The target has to come from the calendar, not from the data being written.
    """
    import pandas as pd
    from tools import archive_fund_flow as flow

    store = tmp_path / "store"
    store.mkdir()
    index = tmp_path / "flow.parquet"
    old = pd.DataFrame(dict(date=pd.to_datetime(["2026-11-06"] * 2), code=["600519", "000001"],
                            close=[1.0, 1.0], pct_change=[0.0, 0.0], main_net=[0.0, 0.0],
                            main_pct=[0.0, 0.0]))
    old.to_parquet(index, index=False)
    monkeypatch.setattr(flow, "existing", lambda path: old.copy())
    monkeypatch.setattr(flow, "last_trading_day", lambda today=None: pd.Timestamp("2026-11-13"))
    fetched = []

    def fake_fetch(code, pause=0.0):
        fetched.append(code)
        frame = old.head(1).copy()
        frame["code"] = code
        frame["date"] = pd.Timestamp("2026-11-13")
        return frame

    monkeypatch.setattr(flow, "fetch", fake_fetch)
    summary = flow.run(codes=["600519", "000001"], store=store, log=lambda *_: None)
    assert sorted(fetched) == ["000001", "600519"]      # both were stale against the target
    assert summary["skipped"] == 0
    assert summary["rows_added"] == 2                    # counted after de-duplication


def test_a_code_already_at_the_target_is_skipped(tmp_path, monkeypatch):
    import pandas as pd
    from tools import archive_fund_flow as flow

    store = tmp_path / "store"
    store.mkdir()
    index = tmp_path / "flow.parquet"
    fresh = pd.DataFrame(dict(date=pd.to_datetime(["2026-11-13"]), code=["600519"],
                              close=[1.0], pct_change=[0.0], main_net=[0.0], main_pct=[0.0]))
    monkeypatch.setattr(flow, "existing", lambda path: fresh.copy())
    monkeypatch.setattr(flow, "last_trading_day", lambda today=None: pd.Timestamp("2026-11-13"))
    summary = flow.run(codes=["600519"], store=store, log=lambda *_: None)
    assert summary["skipped"] == 1 and summary["fetched"] == 0
