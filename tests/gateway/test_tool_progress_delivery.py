"""``tool.progress`` delivery to the adapters that already receive ``tool.started`` (#152515).

The agent emits ``tool.progress`` with real server-reported values through
``bound_tool_progress``; the two structured sinks that already carry ``tool.started`` must
carry it too, gated by the same tool-progress setting and scoped per session /
``tool_call_id``: the shared execution event stream (``progress_callback``) and the API
observers (``wire_api_observers``). ``tool.completed`` / ``tool.complete`` is the clear
signal that lets a client drop the progress state, so the API test asserts it arrives
after the progress frames.
"""
import queue
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner


def _turn(platform, published, *, tool_progress_enabled=True):
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(
        source=SimpleNamespace(platform=platform, chat_id='chat'),
        mute_notification_reply=False, scheduled_heartbeat=False,
        streaming_tts_consumer_holder=[None], stream_consumer_holder=[None],
        interim_assistant_messages_enabled=False, user_config={},
        resolve_display_setting=lambda *a: None, _run_still_current=lambda: True,
        _voice_ack_guild=[None], _native_slack_task_cards=False, progress_mode='all',
        session_id='s', progress_queue=queue.Queue(), _live_status_adapter=None,
        _live_status_mode='off', log_queue=None, tool_progress_enabled=tool_progress_enabled,
        long_tool_hint_fired=[True],
    )
    turn._runner = SimpleNamespace(config=None, _delivery_adapter_for=lambda source: None)
    live = SimpleNamespace(tool_progress_mode=None)
    authority = SimpleNamespace(sessions={'s': live})
    turn._approval_owner = (authority, 's', 1)
    turn._publish_execution = lambda kind, payload: published.append((kind, payload)) or True
    turn._publish_api_tool = lambda *a, **k: None
    turn._schedule = lambda coro, _message: coro.close()
    return turn


def test_messaging_turn_publishes_tool_progress_to_execution_stream():
    """A remote (Telegram) turn with tool progress on publishes ``tool.progress`` with the
    server-reported values, scoped by ``tool_call_id``; a missing total stays missing."""
    published = []
    turn = _turn(Platform.TELEGRAM, published)
    turn.progress_callback('tool.progress', 'mcp__prog__slow', 'halfway', None,
                           tool_call_id='call-7', progress=2.0, total=4.0, percent=50)
    turn.progress_callback('tool.progress', 'mcp__prog__slow', None, None,
                           tool_call_id='call-8', progress=1.0, total=None, percent=None)
    assert published == [
        ('tool.progress', {'tool_id': 'call-7', 'name': 'mcp__prog__slow', 'progress': 2.0,
                          'total': 4.0, 'percent': 50, 'message': 'halfway'}),
        ('tool.progress', {'tool_id': 'call-8', 'name': 'mcp__prog__slow', 'progress': 1.0,
                          'total': None, 'percent': None, 'message': None}),
    ]


def test_messaging_turn_with_tool_progress_off_publishes_nothing():
    published = []
    turn = _turn(Platform.TELEGRAM, published, tool_progress_enabled=False)
    turn.progress_callback('tool.progress', 'mcp__prog__slow', 'halfway', None,
                           tool_call_id='call-7', progress=2.0, total=4.0, percent=50)
    assert published == []


def test_local_turn_still_publishes_tool_progress():
    published = []
    turn = _turn(Platform.LOCAL, published, tool_progress_enabled=False)
    turn.progress_callback('tool.progress', 'mcp__prog__slow', 'halfway', None,
                           tool_call_id='call-7', progress=2.0, total=4.0, percent=50)
    assert published == [
        ('tool.progress', {'tool_id': 'call-7', 'name': 'mcp__prog__slow', 'progress': 2.0,
                          'total': 4.0, 'percent': 50, 'message': 'halfway'}),
    ]


def _api_turn_runner(owner, ref):
    from gateway.run_turn_runner import TurnRunner
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(_native_slack_task_cards=False, _voice_ack_guild=[None], progress_mode='off')
    turn._approval_owner = (owner, ref.session_id, owner.db.get_session(ref.session_id)['runtime_generation'])
    return turn


def _wired_agent(turn):
    """The agent's tool sinks as TurnRunner wires them for an API admission."""
    from gateway.session_api_turn import wire_api_observers
    agent = SimpleNamespace(tool_progress_callback=None, status_callback=None, interim_assistant_callback=None,
                            tool_start_callback=turn.combined_tool_start_callback,
                            tool_complete_callback=turn.combined_tool_complete_callback)
    wire_api_observers(agent, turn._approval_owner, False)
    return agent


@pytest.mark.asyncio
async def test_api_progress_observer_receives_tool_progress_and_completion(api, owner):
    """A progress-only API observer receives ``tool.progress`` verbatim (``total``/``percent``
    only when the server reported them) and then ``tool.completed`` as the clear signal."""
    from gateway.session_api_turn import admit_api_turn, observe_api_turn
    admitted = admit_api_turn(api, session_id='prog', user_message='hello', conversation_history=[])
    _, ref, _ = admitted
    seen = []

    async def handle(event):
        from gateway.session_results import execution_result
        turn = _api_turn_runner(owner, ref)
        progress_cb = _wired_agent(turn).tool_progress_callback
        progress_cb('tool.progress', 'mcp__prog__slow', 'halfway', None,
                    tool_call_id='call-7', progress=2.0, total=4.0, percent=50)
        progress_cb('tool.progress', 'mcp__prog__slow', None, None,
                    tool_call_id='call-8', progress=1.0, total=None, percent=None)
        progress_cb('tool.completed', 'mcp__prog__slow', None, None, is_error=False)
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    await observe_api_turn(admitted, tool_progress_callback=lambda *a, **kw: seen.append((a, kw)))
    assert seen == [
        (('tool.progress', 'mcp__prog__slow', 'halfway', None),
         {'tool_call_id': 'call-7', 'progress': 2.0, 'total': 4.0, 'percent': 50}),
        (('tool.progress', 'mcp__prog__slow', None, None),
         {'tool_call_id': 'call-8', 'progress': 1.0, 'total': None, 'percent': None}),
        (('tool.completed', 'mcp__prog__slow', None, None), {'is_error': False}),
    ]
