#!/usr/bin/env python3
"""Run the complete P3 corpus and assert its executed/failure contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_SUITE = _ROOT / "tests" / "wasi-testsuite" / "tests" / "rust" / "testsuite" / "wasm32-wasip3"
_RUNNER = _ROOT / "tests" / "wasi-testsuite-runner-patch" / "wasi_test_runner.py"
_ADAPTER = _ROOT / "tests" / "wasi-testsuite-adapter" / "wamr-zig.py"
_EXPECTED_FIXTURES = 41


def _remove_jit_sidecars() -> None:
    for pattern in ("*.cwasm", "*.cwasm.json"):
        for sidecar in _SUITE.glob(pattern):
            sidecar.unlink()


def main() -> int:
    jit = bool(os.environ.get("WAMR_JIT_TESTSUITE"))
    if jit:
        _remove_jit_sidecars()
        leftovers = [
            path.name
            for pattern in ("*.cwasm", "*.cwasm.json")
            for path in _SUITE.glob(pattern)
        ]
        if leftovers:
            print(
                f"error: JIT P3 run has component sidecars: {', '.join(leftovers)}",
                file=sys.stderr,
            )
            return 2

    report_override = os.environ.get("WAMR_P3_REPORT")
    report = (
        Path(report_override)
        if report_override
        else _ROOT
        / "zig-out"
        / ("wasi-p3-unfiltered-jit.json" if jit else "wasi-p3-unfiltered-aot.json")
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        str(_RUNNER),
        "--test-suite",
        str(_SUITE),
        "--runtime-adapter",
        str(_ADAPTER),
        "--json-output-location",
        str(report),
    ]
    print(" ".join(cmd), flush=True)
    runner = subprocess.run(cmd, check=False)

    if not report.is_file():
        print(f"error: P3 runner did not create {report}", file=sys.stderr)
        return runner.returncode or 2

    data = json.loads(report.read_text())
    results = data.get("results", [])
    if len(results) != 1:
        print(f"error: expected one suite report, got {len(results)}", file=sys.stderr)
        return 2
    suite = results[0]
    tests = suite.get("tests", [])
    executed = [test for test in tests if test.get("executed")]
    failures = {test["name"] for test in executed if test.get("failures")}

    if len(tests) != _EXPECTED_FIXTURES or len(executed) != _EXPECTED_FIXTURES:
        print(
            "error: unfiltered P3 contract violated: "
            f"fixtures={len(tests)}, executed={len(executed)}, "
            f"expected={_EXPECTED_FIXTURES}",
            file=sys.stderr,
        )
        return 2
    if failures:
        print(
            f"error: unfiltered P3 failures: {sorted(failures)}",
            file=sys.stderr,
        )
        return 2
    if runner.returncode != 0:
        print(
            f"error: P3 runner exited with status {runner.returncode}",
            file=sys.stderr,
        )
        return 2

    print(
        f"unfiltered P3 contract: executed={len(executed)}/{_EXPECTED_FIXTURES}, "
        f"passed={suite.get('passed')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
