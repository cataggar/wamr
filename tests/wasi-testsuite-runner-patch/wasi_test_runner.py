"""Launch the vendored `wasi_test_runner` package without installing it.

The upstream runner now obtains its wait timeout from the runtime adapter.
`wamr-zig.py` implements that hook using `WAMR_TESTSUITE_TIMEOUT`, so this
launcher makes the vendored package importable and delegates to its entry
point. When `WAMR_PROFILE_TIMINGS` is set, it also records process execution
time after adapter-side precompilation. The hook is entirely opt-in and does
not wrap the guest process or alter its streams/argv.

It also installs unconditional deadlines on the upstream runner's
harness-side blocking operations (see `_install_operation_deadlines`).
"""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

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


# ── Harness-side operation deadlines (#929) ───────────────────────────
#
# The upstream runner's `operations` fixtures perform blocking work on the
# *harness* side: `do_connect` does an unbounded `readline()` on the guest's
# stdout to discover `host:port`, and `do_read`/`do_send`/`do_recv` block on
# a pipe or socket with no timeout. The adapter's `WAMR_TESTSUITE_TIMEOUT`
# bounds only `do_wait`'s `Popen.wait`, so none of those are covered.
#
# The practical consequence (#929) is severe: a guest that starts but never
# reaches its first `println!` leaves the harness parked in `readline()`
# forever. One wedged fixture then hangs the entire suite instead of failing
# a single test, which is how `sockets-echo` consumed 900s samples on ARM64
# AOT while reporting nothing at all.
#
# Rather than reimplement upstream's stream handling, each blocking op is
# wrapped in a watchdog that kills the guest process when the deadline
# expires. Killing the guest is what actually unblocks the harness: its
# stdout hits EOF (so `readline`/`read` return promptly) and its sockets are
# closed by the kernel (so `recv` returns empty and `send` raises EPIPE).
# Upstream's existing error paths then turn that into an ordinary failure.
_DEFAULT_OPERATION_DEADLINE_SECONDS = 60.0


def _operation_deadline_seconds(runner: Any) -> float:
    """Seconds any single harness-side blocking operation may take.

    `WAMR_TESTSUITE_OP_TIMEOUT` wins when set. Otherwise the adapter's
    guest-wait budget is reused, so callers that already tighten
    `WAMR_TESTSUITE_TIMEOUT` (the profiling workflow uses 30s/120s) get
    proportionally tight operation deadlines for free.
    """
    raw = os.getenv("WAMR_TESTSUITE_OP_TIMEOUT")
    if raw:
        try:
            explicit = float(raw)
        except ValueError:
            explicit = 0.0
        if explicit > 0:
            return explicit
        print(
            f"warning: ignoring invalid WAMR_TESTSUITE_OP_TIMEOUT={raw!r}",
            file=sys.stderr,
        )
    try:
        inherited = float(runner._runtime.get_timeout_seconds())
    except Exception:  # pylint: disable=broad-except
        inherited = 0.0
    if inherited > 0:
        return inherited
    return _DEFAULT_OPERATION_DEADLINE_SECONDS


def _kill_guest(runner: Any) -> None:
    proc = getattr(runner, "_proc", None)
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.kill()
    except OSError:
        pass


def _bounded(runner: Any, label: str, call: Callable[[], None]) -> bool:
    """Run `call`, killing the guest if it outlives the deadline.

    Returns True when the deadline fired. The timer thread only kills the
    guest; it never touches runner state, so the failure is always recorded
    from this (the main) thread after `call` unblocks and returns.
    """
    seconds = _operation_deadline_seconds(runner)
    fired = threading.Event()

    def on_deadline() -> None:
        fired.set()
        _kill_guest(runner)

    timer = threading.Timer(seconds, on_deadline)
    timer.daemon = True
    timer.start()
    try:
        call()
    finally:
        timer.cancel()

    if fired.is_set():
        runner.fail_unexpected(
            f"{label} exceeded the {seconds:g}s harness operation deadline; "
            "the guest process was killed to unblock the suite "
            "(WAMR_TESTSUITE_OP_TIMEOUT)"
        )
        return True
    return False


def _install_operation_deadlines() -> None:
    runner_cls = _suite_runner.TestCaseRunner

    def bound_method(name: str) -> None:
        original = getattr(runner_cls, name)

        def wrapper(self: Any, op: Any) -> None:
            _bounded(self, f"{name}({op})", lambda: original(self, op))

        wrapper.__name__ = name
        setattr(runner_cls, name, wrapper)

    for method in ("do_read", "do_connect", "do_send", "do_recv"):
        bound_method(method)

    # Belt-and-braces for the socket ops: an explicit socket timeout makes
    # `send`/`recv` fail on their own even in the case where the guest
    # process has already exited but a peer socket somehow stays open.
    original_add_socket = runner_cls.add_socket

    def add_socket(self: Any, name: str, sock: socket.socket) -> None:
        try:
            sock.settimeout(_operation_deadline_seconds(self))
        except OSError:
            pass
        original_add_socket(self, name, sock)

    runner_cls.add_socket = add_socket


_install_operation_deadlines()


if __name__ == "__main__":
    sys.exit(_upstream_main.main())
