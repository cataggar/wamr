"""Launch the vendored `wasi_test_runner` package without installing it.

The upstream runner now obtains its wait timeout from the runtime adapter.
`wamr-zig.py` implements that hook using `WAMR_TESTSUITE_TIMEOUT`, so this
launcher makes the vendored package importable and delegates to its entry
point. When `WAMR_PROFILE_TIMINGS` is set, it also records process execution
time after adapter-side precompilation. The hook is entirely opt-in and does
not wrap the guest process or alter its streams/argv.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_UPSTREAM_RUNNER_DIR = _REPO_ROOT / "tests" / "wasi-testsuite" / "test-runner"

# Make the vendored `wasi_test_runner` package importable without
# requiring the user to set PYTHONPATH manually.
sys.path.insert(0, str(_UPSTREAM_RUNNER_DIR))

from wasi_test_runner import __main__ as _upstream_main  # noqa: E402
from wasi_test_runner import test_suite_runner as _suite_runner  # noqa: E402


def _append_event(event: dict[str, Any]) -> None:
    output = os.getenv("WAMR_PROFILE_TIMINGS")
    if not output:
        return
    encoded = (json.dumps(event, sort_keys=True) + "\n").encode("UTF-8")
    fd = os.open(output, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)


def _install_profile_hooks() -> None:
    if not os.getenv("WAMR_PROFILE_TIMINGS"):
        return

    original_run = _suite_runner.TestCaseRunner.do_run
    original_wait = _suite_runner.TestCaseRunner._wait

    def timed_run(self: Any, run: Any) -> None:
        original_run(self, run)
        if self._proc is None:
            return
        self._wamr_started_ns = time.perf_counter_ns()

    def timed_wait(self: Any, timeout: float | None) -> tuple[int, str, str]:
        started_ns = getattr(self, "_wamr_started_ns", None)
        try:
            result = original_wait(self, timeout)
            return result
        finally:
            finished_ns = time.perf_counter_ns()
            if started_ns is not None:
                fixture = Path(self._test_path).stem
                with Path(self._test_path).open("rb") as wasm:
                    artifact_kind = (
                        "component"
                        if wasm.read(8)[4:8] == b"\x0d\x00\x01\x00"
                        else "core"
                    )
                _append_event(
                    {
                        "schema_version": 1,
                        "event": "phase_timing",
                        "run_id": os.getenv("WAMR_PROFILE_RUN_ID", ""),
                        "mode": os.getenv(
                            "WAMR_PROFILE_MODE",
                            "jit" if os.getenv("WAMR_JIT_TESTSUITE") else "aot",
                        ),
                        "fixture": fixture,
                        "phase": "fixture_execution",
                        "artifact_kind": artifact_kind,
                        "cache": "n/a",
                        "duration_ns": finished_ns - started_ns,
                        "pid": os.getpid(),
                    }
                )

    _suite_runner.TestCaseRunner.do_run = timed_run
    _suite_runner.TestCaseRunner._wait = timed_wait


_install_profile_hooks()


if __name__ == "__main__":
    sys.exit(_upstream_main.main())
