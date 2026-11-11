"""Reserve-before-rotate coverage for pooled single-use OAuth refresh (#151973).

The refresh POST consumes the old single-use refresh token, so the rotated
pair must have somewhere durable to land BEFORE it is spent. ``_refresh_entry``
now probes the store ``_persist`` will write (temp + fsync + rename beside
``auth.json``) while holding the auth-store lock, and skips the refresh entirely
when the probe fails — the still-valid access token keeps working until it
expires, and no other process ever sees the consumed token as current.

Companion: ``test_anthropic_credential_persist_failure.py`` covers the
Anthropic *singleton* commit failing after the POST; these tests cover the
pool-row persist itself, which for pool-owned sources (``manual:*``) is the
only durable copy — there is no singleton to re-seed from.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from agent import anthropic_credentials as AA
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)

# Far enough in the past that every ``_entry_needs_refresh`` check fires.
_EXPIRED_MS = 1_000

# Synthetic, non-functional token material.
_STALE_ACCESS = "sk-ant-oat01-stale"
_STALE_REFRESH = "sk-ant-ort01-stale"
_ROTATED_ACCESS = "sk-ant-oat01-rotated"
_ROTATED_REFRESH = "sk-ant-ort01-rotated"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Real on-disk HERMES_HOME so the probe targets a real auth.json."""
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True
    )
    return home


def _rotating_refresh(*_a, **_kw):
    """Stand-in for the token endpoint: always rotates the pair."""
    return {
        "access_token": _ROTATED_ACCESS,
        "refresh_token": _ROTATED_REFRESH,
        "expires_at_ms": int(time.time() * 1000) + 3_600_000,
    }


def _entry(source: str) -> PooledCredential:
    return PooledCredential(
        provider="anthropic",
        id="anthropic-1",
        label="anthropic oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token=_STALE_ACCESS,
        refresh_token=_STALE_REFRESH,
        expires_at_ms=_EXPIRED_MS,
    )


def _break_temp_creation(monkeypatch):
    """Make creating a temp file beside auth.json fail with ENOSPC.

    The atomic writer stages its temp file the same way, so this is the exact
    first failure a full disk produces for the persist.
    """
    real_mkstemp = tempfile.mkstemp

    def _failing_mkstemp(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tempfile, "mkstemp", _failing_mkstemp)
    return real_mkstemp


def test_refresh_skipped_when_store_cannot_take_write(hermes_home, monkeypatch):
    """ENOSPC on the store must skip the refresh, not spend the token.

    ``manual:hermes_pkce`` has no authoritative singleton, so the pool row in
    auth.json is the only durable copy: POSTing here would leave the rotated
    pair in memory only while every other process reads the consumed token.
    """
    posts: list[str] = []

    def _counting_refresh(refresh_token, use_json=False):
        posts.append(refresh_token)
        return _rotating_refresh()

    monkeypatch.setattr(AA, "refresh_anthropic_oauth_pure", _counting_refresh)
    _break_temp_creation(monkeypatch)
    before = (hermes_home / "auth.json").read_bytes()

    entry = _entry("manual:hermes_pkce")
    pool = CredentialPool("anthropic", [entry])

    result = pool._refresh_entry(entry, force=True)

    assert posts == [], "the refresh POST must never fire when the store cannot take the write"
    assert result is entry
    assert result.access_token == _STALE_ACCESS
    assert result.refresh_token == _STALE_REFRESH
    assert result.last_status is None, "the credential is fine — the disk is not; never bench it"
    assert (hermes_home / "auth.json").read_bytes() == before
    assert [p.name for p in hermes_home.iterdir() if ".hermes-reserve-" in p.name] == []


def test_reserve_probe_succeeds_and_leaves_no_scratch_files(hermes_home):
    """The happy-path probe proves writability without touching the store."""
    pool = CredentialPool("anthropic", [_entry("manual:hermes_pkce")])
    before = (hermes_home / "auth.json").read_bytes()

    pool._reserve_rotation_persist()

    assert (hermes_home / "auth.json").read_bytes() == before
    assert [p.name for p in hermes_home.iterdir() if ".hermes-reserve-" in p.name] == []


def test_reserve_probe_fails_when_rename_fails(hermes_home, monkeypatch):
    """The probe exercises the rename leg of the atomic write too."""
    pool = CredentialPool("anthropic", [_entry("manual:hermes_pkce")])
    real_replace = os.replace

    def _failing_replace(src, dst):
        if ".hermes-reserve-" in os.fspath(src):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _failing_replace)

    with pytest.raises(OSError):
        pool._reserve_rotation_persist()

    assert [p.name for p in hermes_home.iterdir() if ".hermes-reserve-" in p.name] == []


def test_reserve_probe_fails_when_store_parent_missing(hermes_home, monkeypatch):
    """A missing store directory fails the probe instead of creating it."""
    pool = CredentialPool("anthropic", [_entry("manual:hermes_pkce")])
    monkeypatch.setattr(
        pool, "_rotation_persist_target", lambda: hermes_home / "gone" / "auth.json"
    )

    with pytest.raises(OSError):
        pool._reserve_rotation_persist()

    assert not (hermes_home / "gone").exists()
