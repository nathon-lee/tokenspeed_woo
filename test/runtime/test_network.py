import socket

import pytest

from tokenspeed.runtime.utils import network


class FakeSocket:
    def __init__(self, family: int, fail_connect: bool):
        self.family = family
        self.fail_connect = fail_connect
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def connect(self, address):
        if self.fail_connect:
            raise OSError("network unavailable")

    def getsockname(self):
        if self.family == socket.AF_INET:
            return ("192.0.2.1", 0)
        return ("2001:db8::1", 0, 0, 0)


def _patch_network(monkeypatch, *, fail_ipv4: bool):
    sockets = []

    def create_socket(family, socket_type):
        fake_socket = FakeSocket(family, fail_ipv4 and family == socket.AF_INET)
        sockets.append(fake_socket)
        return fake_socket

    monkeypatch.setattr(network.socket, "socket", create_socket)
    monkeypatch.setattr(network.socket, "gethostname", lambda: "localhost")
    monkeypatch.setattr(
        network.socket,
        "gethostbyname",
        lambda hostname: (_ for _ in ()).throw(OSError("resolution failed")),
    )
    monkeypatch.delenv("TOKENSPEED_HOST_IP", raising=False)
    monkeypatch.delenv("HOST_IP", raising=False)
    return sockets


@pytest.mark.parametrize("probe", [network.get_local_ip_by_remote, network.get_ip])
def test_network_probe_closes_socket_on_ipv4_success(monkeypatch, probe):
    sockets = _patch_network(monkeypatch, fail_ipv4=False)

    assert probe() == "192.0.2.1"
    assert len(sockets) == 1
    assert all(fake_socket.closed for fake_socket in sockets)


@pytest.mark.parametrize("probe", [network.get_local_ip_by_remote, network.get_ip])
def test_network_probe_closes_sockets_on_ipv6_fallback(monkeypatch, probe):
    sockets = _patch_network(monkeypatch, fail_ipv4=True)

    assert probe() == "2001:db8::1"
    assert [fake_socket.family for fake_socket in sockets] == [
        socket.AF_INET,
        socket.AF_INET6,
    ]
    assert all(fake_socket.closed for fake_socket in sockets)
