"""Thin wrapper around the vendored `wasi_test_runner` package that
makes the hard-coded 5s `do_wait` timeout configurable via the
`WAMR_TESTSUITE_TIMEOUT` env var (seconds, defaults to the upstream
5s when unset).

Background — issue #583 A7:
`tests/wasi-testsuite/test-runner/wasi_test_runner/test_suite_runner.py`
hard-codes `self._wait(5)` inside `TestCaseRunner.do_wait`. That
matches GitHub Actions runner timings (every `wasm32-wasip3` fixture
exits in <5s on the CI hosts) but is tight on slow developer VMs
(`http-fields.wasm` takes ~11s on the project Azure dev VM, which
causes the upstream runner to mark the test as
`timeout expired` even though the wamr subprocess would finish
correctly a few seconds later).

The adapter/runtime-adapter contract (see `runtime_adapter.py`) does
not expose a timeout hook, so we cannot plumb the override through
`tests/wasi-testsuite-adapter/wamr-zig.py`. Wrapping the wamr binary
in `timeout(1)` also does not help — the wait is enforced by
Python's `subprocess.communicate(timeout=...)`, not by the wamr
binary's own runtime.

Rather than vendor a forked copy of `test_suite_runner.py`, this
wrapper imports the upstream package unmodified and monkey-patches
`TestCaseRunner.do_wait` before delegating to the upstream `main()`.
The override reads `WAMR_TESTSUITE_TIMEOUT` (parsed as a positive
float, in seconds) and falls back to the upstream 5s default when
the variable is unset or invalid. The submodule contents at
`tests/wasi-testsuite/` are left untouched so the upstream pin can
move forward without rebasing a vendored runner copy.

This file is wired into `build.zig`'s `wasi-testsuite` /
`wasi-p3-testsuite` steps. Run locally with e.g.
`WAMR_TESTSUITE_TIMEOUT=30 zig build wasi-p3-testsuite`.
"""

import os
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_UPSTREAM_RUNNER_DIR = _REPO_ROOT / "tests" / "wasi-testsuite" / "test-runner"

# Make the vendored `wasi_test_runner` package importable without
# requiring the user to set PYTHONPATH manually.
sys.path.insert(0, str(_UPSTREAM_RUNNER_DIR))

from wasi_test_runner import __main__ as _upstream_main  # noqa: E402
from wasi_test_runner.test_suite_runner import TestCaseRunner  # noqa: E402
from wasi_test_runner.test_case import Wait  # noqa: E402


_UPSTREAM_DO_WAIT_TIMEOUT_SECS = 5.0


def _resolve_timeout() -> float:
    raw = os.environ.get("WAMR_TESTSUITE_TIMEOUT")
    if raw is None or raw == "":
        return _UPSTREAM_DO_WAIT_TIMEOUT_SECS
    try:
        value = float(raw)
    except ValueError:
        print(
            f"warning: ignoring non-numeric WAMR_TESTSUITE_TIMEOUT={raw!r}; "
            f"falling back to upstream {_UPSTREAM_DO_WAIT_TIMEOUT_SECS}s",
            file=sys.stderr,
        )
        return _UPSTREAM_DO_WAIT_TIMEOUT_SECS
    if value <= 0:
        print(
            f"warning: ignoring non-positive WAMR_TESTSUITE_TIMEOUT={raw!r}; "
            f"falling back to upstream {_UPSTREAM_DO_WAIT_TIMEOUT_SECS}s",
            file=sys.stderr,
        )
        return _UPSTREAM_DO_WAIT_TIMEOUT_SECS
    return value


def _patched_do_wait(self: TestCaseRunner, wait: Wait) -> None:
    timeout = _resolve_timeout()
    try:
        exit_code, out, err = self._wait(timeout)  # pylint: disable=protected-access
        if wait.exit_code != exit_code:
            from wasi_test_runner.test_suite_runner import (  # noqa: E402
                _append_stdout_and_stderr,
            )
            msg = f"{wait} failed: expected {wait.exit_code}, got {exit_code}"
            msg = _append_stdout_and_stderr(msg, out, err)
            self.fail_expectation(msg)
    except subprocess.TimeoutExpired:
        self.fail_expectation(
            f"{wait} failed: timeout expired after {timeout}s"
        )


TestCaseRunner.do_wait = _patched_do_wait  # type: ignore[assignment]


if __name__ == "__main__":
    sys.exit(_upstream_main.main())
