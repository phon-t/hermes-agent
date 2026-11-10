"""A rename sent to a compressed conversation lands on its live tip (#152624).

Renaming through the runtime mutation path with an ended segment's id wrote the
title onto that dead segment, so the listing (which projects a compressed lineage
onto its tip) kept showing the tip's automatic title and the rename looked
reverted. The mutation must resolve the compression tip before writing the title;
a name an ended ancestor already holds moves to the tip through the existing
compression-ancestor rule in ``_resolve_title_conflict``.
"""
import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    database.close()


def _compressed(db, epoch):
    """``chat`` compressed into ``chat-2``; the tip carries an automatic title."""
    db.create_session("chat", source="cli")
    db.publish_compression_child(
        parent_session_id="chat",
        child_session_id="chat-2",
        source="cli",
        messages=[{"role": "user", "content": "hi"}],
        require_compression_lease=False,
    )
    db.set_auto_title("chat-2", "Plugin para sessoes", source=db.TITLE_SOURCE_LLM)
    return db.get_session("chat")


def _rename(db, epoch, session_id, title, request_id="rename-once"):
    snap = db.get_session(session_id)
    return rt.mutate_runtime_session(
        db, epoch=epoch, principal_id="human", session_id=session_id,
        request_id=request_id, expected_revision=snap["runtime_revision"],
        operation="sidebar", payload={"title": title})


def test_sidebar_rename_via_stale_segment_id_lands_on_live_tip(db):
    epoch = rt.begin_runtime_epoch(db, instance_id="owner")
    _compressed(db, epoch)

    _rename(db, epoch, "chat", "SESSION TALK")

    assert db.get_session("chat-2")["title"] == "SESSION TALK"
    assert db.get_session("chat")["title"] != "SESSION TALK"


def test_rename_moves_name_held_by_ended_ancestor_to_tip(db):
    """Residue of the bug: a previous rename already wrote the name onto the ended
    segment. Renaming to the same name through that id must move it to the tip."""
    epoch = rt.begin_runtime_epoch(db, instance_id="owner")
    _compressed(db, epoch)
    db.set_session_title("chat", "SESSION TALK")

    _rename(db, epoch, "chat", "SESSION TALK", request_id="rename-again")

    assert db.get_session("chat-2")["title"] == "SESSION TALK"
    assert db.get_session("chat")["title"] is None


def test_sidebar_rename_of_uncompressed_session_is_unchanged(db):
    """Control: a rename of a session with no compression child lands on it."""
    epoch = rt.begin_runtime_epoch(db, instance_id="owner")
    db.create_session("solo", source="cli")

    _rename(db, epoch, "solo", "SESSION TALK")

    assert db.get_session("solo")["title"] == "SESSION TALK"
