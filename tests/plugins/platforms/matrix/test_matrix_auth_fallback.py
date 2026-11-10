"""Regression tests for Matrix token-auth (whoami) failure handling (#152609).

``_connect_authenticate`` used to abort the whole connect on ANY whoami error, so a transient
homeserver outage (e.g. matrix.org returning 503 ``MUnknown`` during the 2026-11-10 outage) left
the lane down even though the stored token was still valid. It now classifies the failure with
the sync loop's ``_is_permanent_matrix_auth_error``: transient errors retry in-adapter (bounded)
and then fall back to ``MATRIX_PASSWORD`` when one is configured; permanent errors fall back to
password immediately; only a failure with no fallback aborts.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import plugins.platforms.matrix.adapter as matrix_adapter
from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter


class _MatrixError(Exception):
    """Stand-in for mautrix's MatrixRequestError: classification reads only errcode/http_status."""

    def __init__(self, message, *, errcode=None, http_status=None):
        super().__init__(message)
        self.errcode = errcode
        self.http_status = http_status


def _make_adapter(*, password=None):
    extra = {"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}
    if password is not None:
        extra["password"] = password
    return MatrixAdapter(PlatformConfig(enabled=True, token="syt_x", extra=extra))


def _make_client(whoami_side_effect):
    whoami_ok = SimpleNamespace(user_id="@bot:example.org", device_id="DEV123")
    client = SimpleNamespace(
        whoami=AsyncMock(side_effect=whoami_side_effect if isinstance(whoami_side_effect, list) else [whoami_side_effect]),
        login=AsyncMock(return_value=SimpleNamespace(device_id="DEV999")),
        query_keys=AsyncMock(return_value=SimpleNamespace(device_keys={})),
        mxid="@bot:example.org",
        device_id="",
    )
    client._whoami_ok = whoami_ok
    return client


def _make_api():
    api = SimpleNamespace(token="", session=MagicMock())
    api.session.close = AsyncMock()
    return api


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    """Retry sleeps are policy, not behaviour under test — run them at zero delay."""
    monkeypatch.setattr(matrix_adapter, "_TOKEN_AUTH_RETRY_DELAY_S", 0, raising=False)


@pytest.mark.asyncio
async def test_transient_whoami_failure_retries_then_succeeds(caplog):
    """A 503 whoami (the matrix.org outage shape) is retried, not fatal."""
    adapter = _make_adapter()
    transient = _MatrixError("Unable to introspect the access token", errcode="M_UNKNOWN", http_status=503)
    client = _make_client([transient, SimpleNamespace(user_id="@bot:example.org", device_id="DEV123")])
    api = _make_api()

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
        assert await adapter._connect_authenticate(client, api) is True

    assert client.whoami.await_count == 2
    assert client.device_id == "DEV123"
    assert "transiently" in caplog.text


@pytest.mark.asyncio
async def test_transient_whoami_exhausted_falls_back_to_password():
    """Bounded retry, then the configured password recovers the connection."""
    adapter = _make_adapter(password="pw")
    transient = _MatrixError("degraded_performance", errcode="M_UNKNOWN", http_status=503)
    client = _make_client([transient] * 3)
    api = _make_api()

    assert await adapter._connect_authenticate(client, api) is True

    assert client.whoami.await_count == 3  # _TOKEN_AUTH_MAX_ATTEMPTS, then fallback
    client.login.assert_awaited_once()
    assert api.token == ""  # token cleared before the password login
    assert client.device_id == "DEV999"


@pytest.mark.asyncio
async def test_transient_whoami_exhausted_without_password_aborts():
    """No password to fall back to: the connect aborts after the bounded retry."""
    adapter = _make_adapter()
    transient = _MatrixError("timeout", errcode=None, http_status=None)
    client = _make_client([transient] * 3)
    api = _make_api()

    assert await adapter._connect_authenticate(client, api) is False

    assert client.whoami.await_count == 3
    client.login.assert_not_awaited()
    api.session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_permanent_whoami_failure_falls_back_without_retry(caplog):
    """A revoked token is permanent: no retry, straight to the password fallback."""
    adapter = _make_adapter(password="pw")
    permanent = _MatrixError("Unknown access token", errcode="M_UNKNOWN_TOKEN", http_status=401)
    client = _make_client([permanent])
    api = _make_api()

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
        assert await adapter._connect_authenticate(client, api) is True

    assert client.whoami.await_count == 1
    client.login.assert_awaited_once()
    assert "permanently" in caplog.text


@pytest.mark.asyncio
async def test_permanent_whoami_failure_without_password_aborts():
    """A revoked token with no password fallback aborts without retrying whoami."""
    adapter = _make_adapter()
    permanent = _MatrixError("Unknown access token", errcode="M_UNKNOWN_TOKEN", http_status=401)
    client = _make_client([permanent])
    api = _make_api()

    assert await adapter._connect_authenticate(client, api) is False

    assert client.whoami.await_count == 1
    client.login.assert_not_awaited()
    api.session.close.assert_awaited_once()
