"""Offline guard for the HTTP process.

WorldForge is offline-by-design. The sandbox does not download datasets
or weights. :func:`install_offline_guard` wraps socket connect so a
non-loopback TCP connection raises :class:`OfflineError`. Loopback stays
open so ``worldforge serve`` can accept clients on ``127.0.0.1``. The
guard is idempotent and is not a switch that can be turned off.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Callable

OFFLINE_ENV = "WORLDFORGE_OFFLINE"

# Set when the guard installs. These names are not WorldForge dependencies.
# They stop optional libraries from phoning home if one is imported later.
_OFFLINE_DEFAULTS = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "DO_NOT_TRACK": "1",
}

_installed = False
_original_connect: Callable[..., object] | None = None
_original_connect_ex: Callable[..., object] | None = None
_original_create_connection: Callable[..., object] | None = None


class OfflineError(OSError):
    """An outbound connection was refused because WorldForge stays offline."""


def offline_guard_installed() -> bool:
    """Return whether :func:`install_offline_guard` has wrapped the socket."""

    return _installed


def install_offline_guard() -> None:
    """Refuse non-loopback TCP connects for the rest of this process."""

    global _installed, _original_connect, _original_connect_ex, _original_create_connection
    os.environ[OFFLINE_ENV] = "1"
    for name, value in _OFFLINE_DEFAULTS.items():
        os.environ.setdefault(name, value)
    if _installed:
        return
    _original_connect = socket.socket.connect
    _original_connect_ex = socket.socket.connect_ex
    _original_create_connection = socket.create_connection

    def guarded_connect(self: socket.socket, address: object) -> object:
        _reject_remote(address)
        assert _original_connect is not None
        return _original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: object) -> object:
        _reject_remote(address)
        assert _original_connect_ex is not None
        return _original_connect_ex(self, address)

    def guarded_create_connection(address: object, *args: object, **kwargs: object) -> object:
        _reject_remote(address)
        assert _original_create_connection is not None
        return _original_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    _installed = True


def _reject_remote(address: object) -> None:
    host = _peer_host(address)
    if host is None or _is_loopback(host):
        return
    raise OfflineError(
        f"WorldForge is offline-by-design and refused a connection to {host}"
    )


def _peer_host(address: object) -> str | None:
    """Return the host from a TCP address. Unix sockets have no host."""

    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, str):
            return host
        if isinstance(host, bytes):
            return host.decode("utf-8", "replace")
    return None


def _is_loopback(host: str) -> bool:
    text = host.strip().lower()
    if text == "localhost":
        return True
    try:
        parsed = ipaddress.ip_address(text)
    except ValueError:
        return False
    return bool(parsed.is_loopback)
