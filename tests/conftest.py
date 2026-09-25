"""Default tests fail immediately if they accidentally open a network connection."""

import socket
import ipaddress

import pytest


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
