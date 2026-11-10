"""``prompt.submit`` without ``submission_id``/``input_id`` is admitted as a fresh send.

Regression for #151715: in v0.23.0 the owner refused an identityless ``prompt.submit`` with a bare
``4001 invalid_params`` (no field name, no log line), breaking pre-0.23 clients. The owner now
mints the id server-side, as the Desktop shim does client-side and the legacy wire did — and the
``prompt.submit`` params contract lists both fields.
"""

from __future__ import annotations

import asyncio

from gateway.session_contract import SessionRef
from tests.gateway.test_session_follow_profile_config import _owner


def _subscribed_connection(tmp_path, monkeypatch):
    """A real owner authority with one live session this connection subscribes to."""
    from gateway import run
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, "_hermes_home", home)
    # A resolvable model route, as the harness's own tests write it: without one the create
    # policy is invalid and session.create is refused as storage_unavailable.
    import json

    (home / "config.yaml").write_text(json.dumps({"model": {
        "provider": "custom", "base_url": "http://127.0.0.1:1/v1", "default": "m", "api_key": "k"}}))
    store, _runner, authority, connection = _owner(home)

    async def create():
        response = await connection.dispatch({"id": "s1", "method": "session.create", "params": dict(
            request_id="s1", source="gui", cwd=str(home), toolsets=[],
            follow_profile_config=True)})
        assert "result" in response, response
        return response["result"]["session_id"]

    sid = asyncio.run(create())
    ref = SessionRef(authority.profile_id, sid)
    connection.subscriptions[sid] = object()
    return connection, ref, store


def _close(connection, store):
    asyncio.run(connection.close())
    store.close_all_db_handles()


def test_identityless_submit_is_admitted_with_a_minted_id(tmp_path, monkeypatch):
    connection, ref, store = _subscribed_connection(tmp_path, monkeypatch)
    try:
        first = connection.submission(ref, {"session_id": ref.session_id, "text": "hi"})
        second = connection.submission(ref, {"session_id": ref.session_id, "text": "hi again"})
        assert isinstance(first.request_id, str) and first.request_id
        assert isinstance(second.request_id, str) and second.request_id
        # Each mint is unique: a retry of an identityless send admits a new turn (as the legacy
        # wire did); clients that need idempotent retries send their own submission_id.
        assert first.request_id != second.request_id
        assert first.payload["text"] == "hi"
    finally:
        _close(connection, store)


def test_empty_or_non_string_id_is_minted_not_refused(tmp_path, monkeypatch):
    connection, ref, store = _subscribed_connection(tmp_path, monkeypatch)
    try:
        for params in (
            {"session_id": ref.session_id, "text": "hi", "submission_id": ""},
            {"session_id": ref.session_id, "text": "hi", "input_id": ""},
            {"session_id": ref.session_id, "text": "hi", "submission_id": 42},
        ):
            minted = connection.submission(ref, params)
            assert isinstance(minted.request_id, str) and minted.request_id
    finally:
        _close(connection, store)


def test_client_supplied_id_is_preserved(tmp_path, monkeypatch):
    connection, ref, store = _subscribed_connection(tmp_path, monkeypatch)
    try:
        kept = connection.submission(
            ref, {"session_id": ref.session_id, "text": "hi", "submission_id": "client-123"})
        assert kept.request_id == "client-123"
        legacy = connection.submission(
            ref, {"session_id": ref.session_id, "text": "hi", "input_id": "legacy-9"})
        assert legacy.request_id == "legacy-9"
    finally:
        _close(connection, store)


def test_prompt_submit_params_contract_lists_the_identity_fields():
    """The published ``prompt.submit`` params name ``submission_id``/``input_id`` (both optional)."""
    from tui_gateway.contracts.prompt_voice import PromptSubmitParams

    params = PromptSubmitParams(session_id="s", text="hi", submission_id="client-123")
    assert params.submission_id == "client-123"
    assert params.input_id is None
    assert PromptSubmitParams(session_id="s", text="hi").submission_id is None
