#!/usr/bin/env python3
"""Launch a process stopped, attach macOS `sample`, then resume it."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command after --")
    if sys.platform != "darwin":
        parser.error("macOS sampling profiles require Darwin")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    child = os.fork()
    if child == 0:
        os.kill(os.getpid(), signal.SIGSTOP)
        os.execvp(command[0], command)

    _, stopped = os.waitpid(child, os.WUNTRACED)
    if not os.WIFSTOPPED(stopped):
        return 125

    def forward(signum, _frame):
        try:
            os.kill(child, signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, forward)

    with args.log.open("wb") as log:
        sampler = subprocess.Popen(
            [
                "/usr/bin/sample",
                str(child),
                "300",
                "1",
                "-file",
                str(args.output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        # The target remains stopped while sample resolves and attaches.
        time.sleep(0.25)
        os.kill(child, signal.SIGCONT)
        while True:
            try:
                _, status = os.waitpid(child, 0)
                break
            except InterruptedError:
                continue
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.terminate()
            sampler.wait(timeout=10)

    if not args.output.is_file() or args.output.stat().st_size == 0:
        return 125
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 125


if __name__ == "__main__":
    sys.exit(main())
