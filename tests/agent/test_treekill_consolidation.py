"""#85125 Phase 4d — site-local tree-kills delegate to agent.deadline.

Each migrated wrapper keeps its caller-facing contract (signature, all
failures swallowed, ``None`` return) while routing the actual tree
termination through :func:`agent.deadline.kill_process_tree`:

* ``hermes_cli._subprocess_compat.kill_process_tree(proc)`` — also consumed
  by ``agent.shell_hooks`` by name; falls back to
  ``_legacy_kill_process_tree`` when delegation fails.
* ``tools.code_execution_tool._kill_process_group(proc, escalate=...)`` —
  SIGTERM tree first, then (escalate) bounded wait + SIGKILL tree.

Plus one real end-to-end probe: a setsid'd grandchild must die through the
compat wrapper (i.e. the psutil descendant sweep in agent.deadline is
actually reached from the delegating call site).
"""

from __future__ import annotations

import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest

import agent.deadline as deadline_mod


class _FakeProc:
    def __init__(self, pid=54321):
        self.pid = pid
        self.kill_calls = 0

    def kill(self):
        self.kill_calls += 1


# ---------------------------------------------------------------------------
# (1) hermes_cli._subprocess_compat.kill_process_tree
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# (2) tools.code_execution_tool._kill_process_group
# ---------------------------------------------------------------------------

class TestCodeExecutionDelegation:

    @pytest.mark.platforms("linux")
    def test_escalate_waits_then_sigkills_tree(self, monkeypatch):
        import signal as _signal

        from tools import code_execution_tool

        calls = []
        monkeypatch.setattr(
            deadline_mod,
            "kill_process_tree",
            lambda pid, sig=None: calls.append((pid, sig)) or True,
        )
        proc = MagicMock()
        proc.pid = 6666
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
        code_execution_tool._kill_process_group(proc, escalate=True)
        assert calls == [(6666, _signal.SIGTERM), (6666, _signal.SIGKILL)]
        proc.wait.assert_called_once_with(timeout=5)

    def test_swallows_delegation_raise_falls_back_to_plain_kill(self, monkeypatch):
        from tools import code_execution_tool

        def _boom(pid, **kw):
            raise PermissionError("nope")

        monkeypatch.setattr(deadline_mod, "kill_process_tree", _boom)
        proc = _FakeProc(pid=7777)
        code_execution_tool._kill_process_group(proc)  # must not raise
        assert proc.kill_calls == 1

    def test_even_proc_kill_raise_is_swallowed(self, monkeypatch):
        from tools import code_execution_tool

        def _boom(pid, **kw):
            raise PermissionError("nope")

        monkeypatch.setattr(deadline_mod, "kill_process_tree", _boom)
        proc = MagicMock()
        proc.pid = 8888
        proc.kill.side_effect = OSError("already reaped")
        code_execution_tool._kill_process_group(proc)  # must not raise


# ---------------------------------------------------------------------------
# End-to-end: escalation SIGKILLs a SIGTERM-ignoring descendant (#154418)
# ---------------------------------------------------------------------------

@pytest.mark.live_system_guard_bypass
@pytest.mark.platforms("posix")  # POSIX signal semantics
def test_escalate_sigkills_sigterm_ignoring_child_of_promptly_dying_parent(tmp_path):
    """Escalation used to watch only the root: when the root died promptly on
    SIGTERM, no SIGKILL pass ever ran, so a descendant that ignored SIGTERM
    outlived it — reparented to PID 1, where no fresh tree sweep can find it
    (the interrupted-``execute_code`` leak, #154418)."""
    pytest.importorskip("psutil")
    import psutil

    from tools import code_execution_tool

    ready = tmp_path / "child_ready"
    pid_file = tmp_path / "child.pid"
    child_py = tmp_path / "child.py"
    child_py.write_text(
        "import pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(ready)!r}).write_text('x')\n"
        "time.sleep(120)\n"
    )
    parent_py = tmp_path / "parent.py"
    parent_py.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, {str(child_py)!r}])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    proc = subprocess.Popen([sys.executable, str(parent_py)])
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        while not (ready.exists() and pid_file.exists()) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists() and pid_file.exists(), "child never armed — test harness broken"
        child_pid = int(pid_file.read_text().strip())

        code_execution_tool._kill_process_group(proc, escalate=True)

        _gone, alive = psutil.wait_procs([psutil.Process(child_pid)], timeout=10)
        assert not alive, "SIGTERM-ignoring child survived escalation"
    finally:
        if proc.poll() is None:
            proc.kill()
        if child_pid is not None:
            try:
                psutil.Process(child_pid).kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# End-to-end: setsid grandchild dies through the compat wrapper
# ---------------------------------------------------------------------------

@pytest.mark.live_system_guard_bypass
@pytest.mark.platforms("posix")  # POSIX session semantics
def test_e2e_setsid_grandchild_killed_via_compat_wrapper(tmp_path):
    """The delegating wrapper must reach descendants that setsid'd out of the
    child's process group — the exact capability the shared primitive adds
    over the old local killpg-only body (scaffold shape lifted from
    tests/agent/test_deadline.py::test_kills_descendant_in_its_own_session).
    """
    pytest.importorskip("psutil")
    import psutil

    from hermes_cli._subprocess_compat import kill_process_tree

    started = tmp_path / "grandchild_started"
    marker = tmp_path / "grandchild_survived"
    grandchild_py = tmp_path / "grandchild.py"
    grandchild_py.write_text(
        "import pathlib, time\n"
        f"pathlib.Path({str(started)!r}).write_text('x')\n"
        "time.sleep(10)\n"
        f"pathlib.Path({str(marker)!r}).write_text('x')\n"
    )
    parent_py = tmp_path / "parent.py"
    parent_py.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(grandchild_py)!r}], start_new_session=True)\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen([sys.executable, str(parent_py)], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), "grandchild never spawned — test harness broken"
        # Snapshot the tree BEFORE the kill so we can assert zero survivors.
        descendants = psutil.Process(proc.pid).children(recursive=True)

        kill_process_tree(proc)

        proc.wait(timeout=5)
        _gone, alive = psutil.wait_procs(descendants, timeout=5)
        assert not alive, f"survivors after tree kill: {alive}"
        time.sleep(1.0)
        assert not marker.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
