"""`python -m gateway.run` must load gateway/run.py once (#152713).

Started as `python -m gateway.run`, run.py runs as `__main__`; a later
`from gateway.run import X` loaded a second module object with its own
globals, so e.g. `sys.modules["gateway.run"]._gateway_runner_ref` could
see the other copy. The `__main__` block aliases it first (the pattern
`hermes_cli/main.py` already uses).

The probe runs the real entry point in a subprocess with `--help`: argparse
exits 0 before the gateway starts, but only after the `__main__` block ran.
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DRIVER = """
import runpy
import sys

sys.argv = ["gateway/run.py", "--help"]
try:
    runpy.run_module("gateway.run", run_name="__main__", alter_sys=True)
except SystemExit as exc:
    code = exc.code
else:
    code = 0
assert code == 0, f"gateway.run --help exited {code!r}"
mod = sys.modules.get("gateway.run")
assert mod is not None, "gateway.run was never registered in sys.modules"
assert mod.__name__ == "__main__", (
    f"gateway.run is {mod.__name__!r}, not the __main__ module"
)
import gateway.run as again

assert again is mod, "import gateway.run loaded a second module object"
"""


def test_python_m_gateway_run_is_one_module_object(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home)}
    result = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # the return code is asserted below, with the probe output
    )
    assert result.returncode == 0, (
        f"probe failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
