"""A prompt turn fans its accepted user message out on ``message.start``.

Regression for #151742: when two clients are attached to one session, the client that did NOT
submit a prompt never received the prompt's text live — only the assistant stream fanned out,
so the prompt appeared only after the next history refetch (~10 s gap on Desktop). The turn's
``message.start`` now carries the accepted user message (durable row id, text, timestamp,
attachments) so a non-submitting viewer can merge it by row identity at once.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import prompt_turn, server


class _InlineThread:
    """Run threads synchronously so tests observe final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


def _agent():
    def _run(message, conversation_history=None, stream_callback=None, **_kwargs):
        return {"final_response": "done"}

    return types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )


@pytest.fixture()
def emits(monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    return captured


@pytest.fixture()
def marker_home(monkeypatch, tmp_path):
    """Point the server's marker storage at a temp HERMES_HOME."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    return tmp_path


@pytest.fixture()
def turn_env(monkeypatch, tmp_path, marker_home):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _message_start_payloads(emits):
    return [p for e, _s, p in emits if e == "message.start"]


def test_user_prompt_turn_emits_user_payload_on_message_start(emits, turn_env, marker_home):
    """The non-submitting viewer learns the prompt live, with its durable row identity."""
    session = _session(
        agent=_agent(),
        running=True,
        _submit_user_row={
            "role": "user", "content": "hello from A",
            "_row_id": 42, "timestamp": 1730000000.0,
        },
    )

    server._run_prompt_submit("rid", "sid", session, "hello from A")

    starts = _message_start_payloads(emits)
    assert starts, "expected a message.start emit"
    assert {
        "user_row_id": 42, "user_text": "hello from A", "user_timestamp": 1730000000.0,
    } in starts


def test_turn_without_submit_row_emits_bare_message_start(emits, turn_env, marker_home):
    """Auto-continue / system turns carry no user message: message.start stays as today."""
    session = _session(agent=_agent(), running=True)

    server._run_prompt_submit(
        "rid", "sid", session, server._auto_continue_note("the original prompt"),
        display_kind="auto_continue",
    )

    assert _message_start_payloads(emits) == [None]


def test_stale_submit_row_is_not_attributed_to_the_turn():
    """A staged row from an earlier send must not label this turn's prompt."""
    session = {"_submit_user_row": {"content": "old prompt", "_row_id": 7, "timestamp": 1.0}}

    assert prompt_turn._message_start_user_payload(session, "new prompt", None) is None


def test_builder_includes_images_and_skips_missing_timestamp():
    session = {"_submit_user_row": {"content": "see this", "_row_id": 9}}

    assert prompt_turn._message_start_user_payload(
        session, "see this", ["/tmp/a.png", "/tmp/b.png"]) == {
        "user_row_id": 9, "user_text": "see this", "user_images": ["/tmp/a.png", "/tmp/b.png"],
    }


def test_builder_needs_a_durable_row_id():
    """A staged row without ``_row_id`` (write failed, turn writes its own) carries no identity."""
    session = {"_submit_user_row": {"content": "hello", "timestamp": 1.0}}

    assert prompt_turn._message_start_user_payload(session, "hello", None) is None
