"""In-chat ``/resume`` / ``/sessions`` and bare ``hermes -c`` reach cross-surface sessions.

Regression for #151584: sessions created in Hermes Desktop (``source="gui"``) never appeared in
the classic CLI's in-session lists or via bare ``-c`` / ``--resume latest``. The in-chat list now
shows this CLI's rows first, then an "other surfaces" section (still one numbered list, so
``/resume <number>`` resolves both), and bare ``-c`` falls back to the owner's latest chat on
another local surface when this interface has no history yet.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_cli.gateway_session_store import SurfaceLabelReader
from hermes_state import SessionDB


def _populate(db: SessionDB, rows: list[tuple[str, str, str, float, str]]) -> None:
    for sid, source, _created, _started, _title in rows:
        db.create_session(sid, source=source)
    with db._lock:
        db._conn.executemany(
            "UPDATE sessions SET created_source=?, started_at=?, message_count=1, title=? WHERE id=?",
            [(created, started, title, sid) for sid, _source, created, started, title in rows],
        )
        db._conn.commit()


def _recent_ids(db: SessionDB) -> list[tuple[str, bool]]:
    reader = SurfaceLabelReader(db)
    rows = CLISessionMixin._list_recent_sessions(
        SimpleNamespace(_session_db=reader, session_id="current-session"), limit=10,
    )
    return [(row["id"], bool(row.get("is_other_surface"))) for row in rows]


def test_list_recent_sessions_shows_gui_rows_after_cli_rows(tmp_path):
    """Desktop rows land in a second section after the CLI rows, in the same numbered list."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        ("cli-a", "cli", "cli", 1_000_000, "CLI chat"),
        ("gui-b", "gui", "gui", 2_000_000, "Desktop chat"),
        ("tool-c", "tool", "tool", 3_000_000, "Integration"),
        ("kanban-d", "kanban", "kanban", 4_000_000, "Board worker"),
    ])
    try:
        assert _recent_ids(db) == [("cli-a", False), ("gui-b", True)]
    finally:
        db.close()


def test_list_recent_sessions_gui_only_still_lists(tmp_path):
    """A Desktop-first user with no CLI history still sees their chats in-chat."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [("gui-a", "gui", "gui", 1_000_000, "Desktop chat")])
    try:
        assert _recent_ids(db) == [("gui-a", True)]
    finally:
        db.close()


def test_list_recent_sessions_local_cli_row_is_not_an_other_surface(tmp_path):
    """A CLI chat stored as ``local`` (gateway relabel) stays in the CLI section, listed once."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        ("local-cli", "local", "cli", 1_000_000, "CLI chat"),
        ("local-gui", "local", "desktop", 2_000_000, "Desktop chat"),
    ])
    try:
        assert _recent_ids(db) == [("local-cli", False), ("local-gui", True)]
    finally:
        db.close()


def test_show_recent_sessions_renders_other_surfaces_section(capsys):
    """Both sections render with continuous numbering and the surfaces named."""
    cli_obj = SimpleNamespace(session_id="current-session")
    cli_obj._list_recent_sessions = MagicMock(return_value=[
        {"id": "cli-a", "title": "CLI chat", "preview": "hello", "last_active": None},
        {"id": "gui-b", "title": "Desktop chat", "preview": "hi", "last_active": None,
         "source": "gui", "is_other_surface": True},
    ])
    shown = CLISessionMixin._show_recent_sessions(cli_obj, reason="resume")
    output = capsys.readouterr().out

    assert shown is True
    assert "CLI chat" in output
    assert "Desktop chat" in output
    assert "Other surfaces (Desktop):" in output
    # The Desktop row keeps its number on the one shared list.
    assert "  2   Desktop chat" in output


def test_show_recent_sessions_without_cli_rows_starts_numbering_at_other_section(capsys):
    cli_obj = SimpleNamespace(session_id="current-session")
    cli_obj._list_recent_sessions = MagicMock(return_value=[
        {"id": "gui-b", "title": "Desktop chat", "preview": "hi", "last_active": None,
         "source": "gui", "is_other_surface": True},
    ])
    shown = CLISessionMixin._show_recent_sessions(cli_obj, reason="sessions")
    output = capsys.readouterr().out

    assert shown is True
    assert "Other surfaces (Desktop):" in output
    assert "  1   Desktop chat" in output


def _make_cli():
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "current_session"
    cli_obj._resumed = False
    cli_obj._pending_title = None
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._session_db = MagicMock()
    cli_obj._pending_resume_sessions = None
    cli_obj.resume_display = "minimal"
    return cli_obj


def test_resume_by_number_resolves_a_gui_row():
    """``/resume 2`` switches to the Desktop row numbered in the other-surfaces section."""
    cli_obj = _make_cli()
    cli_obj._list_recent_sessions = MagicMock(return_value=[
        {"id": "cli-a", "title": "CLI chat"},
        {"id": "gui-b", "title": "Desktop chat", "is_other_surface": True},
    ])
    cli_obj._session_db.get_session.return_value = {"id": "gui-b", "title": "Desktop chat"}
    cli_obj._session_db.get_resume_conversations.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ], [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    cli_obj._session_db.resolve_resume_session_id.return_value = "gui-b"

    with (
        patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
        patch("cli._cprint") as mock_cprint,
    ):
        cli_obj._handle_resume_command("/resume 2")

    printed = " ".join(str(call) for call in mock_cprint.call_args_list)
    assert cli_obj.session_id == "gui-b"
    assert "Desktop chat" in printed


def test_resume_all_arms_the_combined_list():
    """``/resume all`` shows the same all-sources list as bare ``/resume``."""
    cli_obj = _make_cli()
    combined = [
        {"id": "cli-a", "title": "CLI chat"},
        {"id": "gui-b", "title": "Desktop chat", "is_other_surface": True},
    ]
    cli_obj._list_recent_sessions = MagicMock(return_value=combined)
    with patch.object(cli_obj, "_show_recent_sessions", return_value=True) as mock_show:
        cli_obj._handle_resume_command("/resume all")

    mock_show.assert_called_once_with(reason="resume")
    assert cli_obj._pending_resume_sessions == combined


# ---------------------------------------------------------------------------
# Bare ``hermes -c`` / ``--resume latest``: CLI history first, other local
# surfaces only when this interface has no history yet.
# ---------------------------------------------------------------------------


def _seed_home(tmp_path, monkeypatch, rows):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, rows)
    db.close()
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: None)
    monkeypatch.setattr("hermes_cli.main.get_hermes_home", lambda: tmp_path)


def test_latest_session_falls_back_to_gui_when_no_cli_history(tmp_path, monkeypatch):
    """A Desktop-first user opening a terminal continues their Desktop chat with bare ``-c``."""
    _seed_home(tmp_path, monkeypatch, [
        ("gui-b", "gui", "gui", 2_000_000, "Desktop chat"),
    ])
    from hermes_cli.main import _latest_session_id

    assert _latest_session_id(use_tui=False) == "gui-b"


def test_latest_session_cli_history_still_wins_over_newer_gui(tmp_path, monkeypatch):
    """Backward-compatible priority: an existing CLI session still wins ``-c`` over a newer
    Desktop one; the Desktop chat is reached from in-chat ``/resume`` instead."""
    _seed_home(tmp_path, monkeypatch, [
        ("cli-a", "cli", "cli", 1_000_000, "CLI chat"),
        ("gui-b", "gui", "gui", 2_000_000, "Desktop chat"),
    ])
    from hermes_cli.main import _latest_session_id

    assert _latest_session_id(use_tui=False) == "cli-a"


def test_latest_session_fallback_skips_platform_rows(tmp_path, monkeypatch):
    """Neither a platform chat nor a platform chat relabelled ``local`` can steal the ``-c``
    fallback; the Desktop chat wins."""
    _seed_home(tmp_path, monkeypatch, [
        ("gui-b", "gui", "gui", 2_000_000, "Desktop chat"),
        ("tg-c", "telegram", "telegram", 3_000_000, "Telegram chat"),
        ("local-tg", "local", "telegram", 4_000_000, "Telegram relabelled"),
    ])
    from hermes_cli.main import _latest_session_id

    assert _latest_session_id(use_tui=False) == "gui-b"
