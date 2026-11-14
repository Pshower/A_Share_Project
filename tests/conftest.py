"""Default tests fail immediately if they accidentally open a network connection.

They also fail if they touch the real account books.  That is not a hypothetical: on 2026-11-14
``test_mark_when_ready`` marked the ``backtest`` book for a day the export never covered and left
a row whose cash came from a replay no export produced -- the page then showed 1,613,595 where the
published track record is 1,687,443.  A test that writes outside ``tmp_path`` has to break the
build, not the ledger.
"""

import hashlib
import socket
import ipaddress
from pathlib import Path

import pytest

ACCOUNT_ROOT = Path(__file__).resolve().parents[1] / "reports" / "account"


def _fingerprint():
    """Content hashes of every real book file, so a write anywhere shows up as a difference."""
    if not ACCOUNT_ROOT.exists():
        return {}
    return {str(path.relative_to(ACCOUNT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ACCOUNT_ROOT.rglob("*")) if path.is_file()}


@pytest.fixture(autouse=True)
def offline_by_default(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests")

    original_connect = socket.socket.connect

    def connect(sock, address):
        # Windows asyncio implements its internal wakeup pipe with a loopback pair.
        if request.node.get_closest_marker("local_ipc") and isinstance(address, tuple):
            try:
                if ipaddress.ip_address(address[0]).is_loopback:
                    return original_connect(sock, address)
            except ValueError:
                pass
        return blocked()

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def account_books_are_read_only():
    """Fail the test that writes to ``reports/account``, naming the files it changed."""
    before = _fingerprint()
    yield
    after = _fingerprint()
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    assert not changed, ("this test wrote to the real account books: " + ", ".join(changed)
                         + " -- point ACCOUNT_DIR or SNAPSHOTS at tmp_path instead")
