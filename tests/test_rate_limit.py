from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import sqlite3

import pytest
import requests

from src.data.online import Eastmoney, normalize_history, retry
from src.data.rate_limit import AdaptivePacer, retry_delay, transient
from src.runtime import events


def http_error(status, after=None):
    response = requests.Response()
    response.status_code = status
    if after is not None:
        response.headers["Retry-After"] = after
    return requests.HTTPError(str(status), response=response)


@pytest.fixture
def clock(monkeypatch):
    from src.data import rate_limit
    now = [1000.0]
    monkeypatch.setattr(rate_limit.time, "time", lambda: now[0])
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(rate_limit.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr("src.data.online.random.uniform", lambda *_: 0)
    return now


def test_error_classification_and_retry_after():
    assert transient(http_error(429, "9")) and retry_delay(http_error(429, "9")) == 9
    assert transient(http_error(503)) and transient(requests.ConnectionError())
    assert not transient(http_error(403)) and not transient(ValueError("schema"))
    assert retry_delay(http_error(429, "NaN")) == 0
    assert retry_delay(http_error(429, "bad")) == 0
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60), usegmt=True)
    assert 58 <= retry_delay(http_error(429, future)) <= 60


def test_shared_cooldown_applies_to_new_pacer_and_recovers_slowly(tmp_path, clock):
    first = AdaptivePacer(tmp_path)
    first.acquire()
    assert first.failure(10) == 2
    second = AdaptivePacer(tmp_path)
    second.success()  # In-flight success cannot cancel the active cooldown.
    second.acquire()
    assert clock[0] >= 1010
    for _ in range(5):
        second.success()
    with sqlite3.connect(first.path) as conn:
        assert conn.execute("SELECT interval FROM throttle").fetchone()[0] == pytest.approx(1.6)
    before = clock[0]
    first.acquire()
    assert clock[0] - before >= 2 - 1e-8


def test_retry_throttles_real_attempts_and_preserves_success(tmp_path, clock, monkeypatch):
    limiter = AdaptivePacer(tmp_path)
    calls, log = [], []
    monkeypatch.setattr(events, "emit", lambda kind, **kw: log.append(dict(type=kind, **kw)))
    def fetch():
        limiter.acquire(deadline=1100)
        calls.append(clock[0])
        if len(calls) < 3:
            raise http_error(429, "7")
        return "saved"
    assert retry(fetch, limiter=limiter, deadline=1100) == "saved"
    assert len(calls) == 3 and min(b - a for a, b in zip(calls, calls[1:])) >= 7
    assert len([r for r in log if r["type"] == "download_retry"]) == 2


def test_long_retry_after_never_sends_early_past_budget(tmp_path, clock):
    limiter = AdaptivePacer(tmp_path)
    calls = []
    def fetch():
        limiter.acquire(deadline=1003)
        calls.append(clock[0])
        raise http_error(429, "3600")
    with pytest.raises(TimeoutError):
        retry(fetch, limiter=limiter, deadline=1003)
    assert len(calls) == 1


def test_permanent_error_not_retried_and_transient_attempts_bounded(clock):
    calls = []
    def fetch(error):
        calls.append(1)
        raise error
    with pytest.raises(ValueError):
        retry(lambda: fetch(ValueError("invalid schema")))
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(requests.ConnectionError):
        retry(lambda: fetch(requests.ConnectionError("disconnect")))
    assert len(calls) == 5


def test_cancellation_interrupts_shared_wait(tmp_path, clock, monkeypatch):
    limiter = AdaptivePacer(tmp_path)
    limiter.failure(60)
    def cancelled():
        raise events.Cancelled("stop")
    monkeypatch.setattr(events, "check_cancel", cancelled)
    with pytest.raises(events.Cancelled):
        limiter.acquire()
    assert clock[0] == 1000


def test_history_checks_http_before_json_and_preserves_basis(tmp_path, clock, monkeypatch):
    provider = Eastmoney(tmp_path)
    error = http_error(429, "12")
    monkeypatch.setattr("src.data.online.requests.get", lambda *a, **kw: error.response)
    with pytest.raises(requests.HTTPError) as caught:
        provider.history("000001", "2020-01-01", "2020-01-03", "hfq")
    assert retry_delay(caught.value) == 12
    calls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return dict(rc=0, data=dict(code="000001", klines=["2020-01-02,10,11,12,9,100,10000,3,2,1,0.5"]))
    def get(*args, **kwargs):
        calls.append(kwargs["params"])
        return Response()
    monkeypatch.setattr("src.data.online.requests.get", get)
    for basis in ["hfq", "unadjusted"]:
        frame = provider.history("000001", "2020-01-01", "2020-01-03", basis)
        normalized = normalize_history(frame, "000001", "2020-01-01", "2020-01-03")
        assert normalized.volume.iloc[0] == 10000 and normalized.close.iloc[0] == 11
    assert [c["fqt"] for c in calls] == ["2", "0"]


def test_quote_lease_rechecked_after_cooldown(tmp_path, clock, monkeypatch):
    provider = Eastmoney(tmp_path)
    provider.limiter.failure(10)
    def deny_after_wait():
        assert clock[0] >= 1010
        raise events.Cancelled("Lease expired while waiting")
    def forbidden(*args, **kwargs):
        pytest.fail("No HTTP request after lease expiry")
    monkeypatch.setattr("src.data.online.requests.get", forbidden)
    with pytest.raises(events.Cancelled):
        provider.quote("000001", permission_check=deny_after_wait)
