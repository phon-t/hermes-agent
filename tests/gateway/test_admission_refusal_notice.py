"""A native-admission refusal notice carries no operator language to the sender (#151644).

The notice is returned as the chat reply, which on email/DM platforms goes verbatim to an
external correspondent: it must not name an operator or a log file, and it must not leak the
admission clause for trust/provenance checks. Diagnostics already live in ``gateway.log``
via ``_refuse``.
"""
import pytest

from gateway.platforms.event import MessageEvent
from gateway.session_ingress import admit_message
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authority_cancel_settlement import _authority


async def _refusal_reply(tmp_path, monkeypatch, field):
    db, authority = _authority(tmp_path, monkeypatch)

    async def admit_native(event):
        raise RuntimeStoreError('invalid_params', field=field)

    monkeypatch.setattr(authority, 'admit_native', admit_native)
    with db:
        return await admit_message(authority, MessageEvent(text='hi', source=authority.sessions['s'].source))


@pytest.mark.asyncio
async def test_refusal_notice_names_no_operator_or_log(tmp_path, monkeypatch):
    reply = await _refusal_reply(tmp_path, monkeypatch, 'text')
    assert reply is not None, reply
    assert 'refused this message' in reply, reply
    assert 'operator' not in reply.lower(), reply
    assert 'gateway.log' not in reply, reply
    assert 'log' not in reply.lower(), reply


@pytest.mark.asyncio
async def test_refusal_notice_keeps_provenance_clause_generic(tmp_path, monkeypatch):
    # A trust/provenance clause (e.g. ``metadata``) stays generic in the notice: the sender
    # learns the message was refused, not which admission check fired.
    reply = await _refusal_reply(tmp_path, monkeypatch, 'metadata')
    assert reply is not None, reply
    assert 'metadata' not in reply, reply
    assert 'operator' not in reply.lower(), reply
    assert 'gateway.log' not in reply, reply
