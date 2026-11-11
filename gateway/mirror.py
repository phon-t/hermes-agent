"""Session mirroring for cross-platform message delivery.

When a message is sent to a platform (send_message or cron delivery), append a
"delivery-mirror" record to the target session's transcript so the receiving-side
agent knows what was sent.  Standalone: works from CLI, cron and gateway contexts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"
_SESSIONS_INDEX_AT_IMPORT = _SESSIONS_INDEX


def _resolve_sessions_index() -> Path:
    """Active profile's ``sessions.json`` at call time: the patched ``_SESSIONS_INDEX`` when a test
    changed it, else live profile-scoped HERMES_HOME — under the multiplexed gateway one process
    serves every profile, so the import-time constant would resolve every profile's pre-migration
    session lookup against the launch profile's index."""
    return (_SESSIONS_INDEX if _SESSIONS_INDEX != _SESSIONS_INDEX_AT_IMPORT
            else get_hermes_home() / "sessions" / "sessions.json")


def _candidate_state_db_paths() -> list[Path]:
    """state.db files that may own a mirrored session, preferred first.

    The launch (current) store first, then the default profile's store, then
    every named profile's store under the hermes root. Only files that exist:
    probing must never create a store for a profile that has none (#151923).
    """
    candidates: list[Path] = []

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved not in candidates and path.is_file():
            candidates.append(resolved)

    _add(get_hermes_home() / "state.db")
    try:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root()
    except Exception:
        root = None
    if root is not None:
        _add(root / "state.db")
        profiles_dir = root / "profiles"
        try:
            children = sorted(profiles_dir.iterdir(), key=lambda p: p.name) if profiles_dir.is_dir() else []
        except OSError:
            children = []
        for child in children:
            if child.is_dir():
                _add(child / "state.db")
    return candidates


def _origin_user_id(entry: dict) -> str:
    return str((entry.get("origin") or {}).get("user_id") or "")


def mirror_to_session(
    platform: str, chat_id: str, message_text: str, source_label: str = "cli", thread_id: Optional[str] = None,
    user_id: Optional[str] = None, role: str = "assistant", session_id: Optional[str] = None,
) -> bool:
    """Append a delivery-mirror message to the target session's SQLite transcript.

    Pass ``session_id`` when the caller already holds the exact session (e.g. the
    cron in_channel seed) to skip the origin scan, which refuses to guess on a
    populated chat (flat + N thread sessions per chat_id) and would drop the mirror.
    An explicit id is routed by existence: the write lands in whichever profile
    store actually holds the session row, so a cron-seeded id from a satellite
    profile is not forced into the launch profile's store (#151923).
    Text that is NOT the agent speaking (e.g. a cron brief) must pass
    ``role="user"``: ``mirror`` metadata is dropped at the SQLite boundary, so an
    assistant-role mirror replays as a real turn and yields assistant→assistant
    pairs that break strict-alternation providers, while a user-role mirror
    collapses safely via the consecutive-user merge.
    Returns True if mirrored, False if no matching session or error. Never raises.

    ``role`` defaults to ``"assistant"`` — correct for the interactive ``send_message`` mirror, where the
    mirrored text is the agent's own outgoing reply (a genuine assistant turn). See #2221.
    """
    try:
        if not session_id:
            session_id = _find_session_id(platform, str(chat_id), thread_id=thread_id, user_id=user_id)
        if not session_id:
            logger.warning(
                "Mirror: no session found for %s:%s thread=%s user=%s (explicit_id=none, origin-scan bailed)",
                platform, chat_id, thread_id, user_id,
            )
            return False
        _append_to_sqlite(session_id, {
            "role": role, "content": message_text, "timestamp": datetime.now().isoformat(),
            "mirror": True, "mirror_source": source_label,
        })
        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True
    except Exception as e:
        # WARNING, not debug: a silent mirror drop is the cron continuation-amnesia bug.
        logger.warning("Mirror failed for %s:%s thread=%s user=%s session=%s: %s", platform, chat_id, thread_id, user_id, session_id, e)
        return False


def _find_session_id(platform: str, chat_id: str, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
    """Active session_id for a platform + chat_id pair.

    state.db is primary; sessions.json is the pre-migration fallback.  DM keys
    don't embed the chat_id ("agent:main:telegram:dm"), so match on the persisted
    origin.  With *user_id*, exact sender matches win; several same-chat candidates
    with no user match → None rather than contaminate another participant's session.

    Queries state.db gateway session rows (primary source since #9006); falls back to scanning sessions.json
    for pre-migration databases.
    """
    try:
        from hermes_state_registry import acquire, release_or_close
        db = acquire()
        try:
            finder = getattr(db, "find_session_by_origin", None)
            session_id = finder(platform=platform, chat_id=chat_id, thread_id=thread_id, user_id=user_id) if callable(finder) else None
            if session_id:
                return str(session_id)
        finally:
            release_or_close(db)
    except Exception as e:
        logger.debug("Mirror state.db session lookup failed: %s", e)

    sessions_index = _resolve_sessions_index()
    if not sessions_index.exists():
        return None
    try:
        data = json.loads(sessions_index.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    def _matches(entry: dict) -> bool:
        origin = entry.get("origin") or {}
        return ((origin.get("platform") or entry.get("platform", "")).lower() == platform.lower()
                and str(origin.get("chat_id", "")) == str(chat_id)
                and (thread_id is None or str(origin.get("thread_id") or "") == str(thread_id)))

    # Keys starting with "_" (e.g. the gateway's "_README") are metadata sentinels.
    candidates = [e for k, e in data.items() if not str(k).startswith("_") and isinstance(e, dict) and _matches(e)]
    if not candidates:
        return None
    if user_id:
        exact_user_matches = [e for e in candidates if _origin_user_id(e) == str(user_id)]
        if exact_user_matches:
            candidates = exact_user_matches
        elif len(candidates) > 1:
            return None
    elif len(candidates) > 1 and len({u.strip() for u in map(_origin_user_id, candidates) if u.strip()}) > 1:
        return None
    return max(candidates, key=lambda entry: entry.get("updated_at", "")).get("session_id")


def _append_to_sqlite(session_id: str, message: dict) -> None:
    """Append a message to the SQLite session database.

    Raises on failure: ``mirror_to_session`` reports ``False`` (and warns) only when the
    exception reaches it — swallowing it here made every failed write look mirrored (#10130).

    The owning store is resolved by existence (#151923): an explicit session id handed over
    from another profile's store — the multiplexed gateway's cron seed creates the row in
    the job profile's ``state.db`` while the mirror runs in the launch scope — must land
    in the store that actually holds the row, not the launch profile's, or the write fails
    the ``messages.session_id`` foreign key on every catch-up delivery. A session no store
    holds (retired or ghost id) raises ``LookupError`` instead, so the mirror skips it
    cleanly rather than failing the FK as an exception every time. A probe failure on one
    store never blocks the scan; a failed append to the owning store still raises.
    """
    from hermes_state_registry import acquire, release_or_close

    for db_path in _candidate_state_db_paths():
        db = acquire(db_path)
        try:
            try:
                owns_session = db.get_session_list_row(session_id) is not None
            except Exception as exc:
                logger.debug("Mirror owner probe failed on %s: %s", db_path, exc)
                continue
            if not owns_session:
                continue
            db.append_message(session_id=session_id, role=message.get("role", "assistant"), content=message.get("content"))
            return
        finally:
            release_or_close(db)
    raise LookupError(f"no session store holds session {session_id!r}; skipping mirror")
