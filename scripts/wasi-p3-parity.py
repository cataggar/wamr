#!/usr/bin/env python3
"""Run the wamr + Wasmtime sides of the `wasm32-wasip3` conformance
suite and feed both JSON reports into `diff-testsuite-reports.py`.

Wired into `build.zig`'s `wasi-p3-parity` step (issue #583 C1). The
step needs an orchestrator because the standalone Wasmtime run can
exit non-zero (Wasmtime 44.0.1 fails 4 / 40 fixtures —
`http-service`, `sockets-tcp-{connect,listen}`, `sockets-udp-send`,
all tracked under `tests/wasi-p3-parity-skip.json`), which would
otherwise abort the build before the diff script gets a chance to
*classify* the deltas. The diff exit code is the step's exit code:
0 if no wamr regressions and no stale skip-list entries, 1 if wamr
fails a fixture the parity runtime still passes, or if a fixture in
the parity-skip list is no longer in the wamr-pass / parity-fail
shape (e.g. the upstream wasmtime / fixture fix has landed and the
entry must be retired).

Resolves the `WAMR` / `WASMTIME` binaries from env vars (matching the
upstream adapters). Defaults are the just-built `zig-out/bin/wamr`
and `wasmtime` on `PATH`.

Usage
-----

    wasi-p3-parity.py --output-dir <DIR>

The script writes
* `<DIR>/wamr-p3.json` — wamr-side wasi-testsuite report
* `<DIR>/wasmtime-p3.json` — Wasmtime-side report
* `<DIR>/wasi-p3-parity.json` — classifier summary (regressions,
  undocumented fixture-bugs, documented fixture-bugs, stale skip
  entries, shared failures)

and forwards the diff script's exit code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_RUNNER = _REPO_ROOT / "tests" / "wasi-testsuite-runner-patch" / "wasi_test_runner.py"
_SUITE = _REPO_ROOT / "tests" / "wasi-testsuite" / "tests" / "rust" / "testsuite" / "wasm32-wasip3"
_WAMR_ADAPTER = _REPO_ROOT / "tests" / "wasi-testsuite-adapter" / "wamr-zig.py"
_WASMTIME_ADAPTER = _REPO_ROOT / "tests" / "wasi-testsuite-adapter" / "wasmtime.py"
_SKIPLIST = _REPO_ROOT / "tests" / "wasi-p3-testsuite-skip.json"
_PARITY_SKIP = _REPO_ROOT / "tests" / "wasi-p3-parity-skip.json"
_DIFF = _HERE / "diff-testsuite-reports.py"


def _run_suite(label: str, adapter: Path, out: Path) -> int:
    cmd = [
        sys.executable,
        str(_RUNNER),
        "--test-suite",
        str(_SUITE),
        "--runtime-adapter",
        str(adapter),
        "--exclude-filter",
        str(_SKIPLIST),
        "--json-output-location",
        str(out),
    ]
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    # The Wasmtime run may exit non-zero when wasmtime fails fixtures
    # that wamr passes — that's not a parity-gate failure, the diff
    # step does the classification. Surface the actual exit code in
    # the log but let the orchestrator carry on.
    print(f"=== {label} exit={proc.returncode} ===", flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the two JSON reports + parity summary into.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat fixture/runtime-bug-class deltas as hard failures "
            "(forwarded to diff-testsuite-reports.py)."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wamr_out = args.output_dir / "wamr-p3.json"
    wasmtime_out = args.output_dir / "wasmtime-p3.json"
    summary_out = args.output_dir / "wasi-p3-parity.json"

    wamr_rc = _run_suite("wamr", _WAMR_ADAPTER, wamr_out)
    wasmtime_rc = _run_suite("wasmtime", _WASMTIME_ADAPTER, wasmtime_out)

    if not wamr_out.exists():
        print(
            f"error: wamr report missing at {wamr_out} (runner exit "
            f"{wamr_rc}); aborting parity diff.",
            file=sys.stderr,
        )
        return wamr_rc or 2
    if not wasmtime_out.exists():
        print(
            f"error: wasmtime report missing at {wasmtime_out} (runner "
            f"exit {wasmtime_rc}); aborting parity diff.",
            file=sys.stderr,
        )
        return wasmtime_rc or 2

    diff_cmd = [
        sys.executable,
        str(_DIFF),
        str(wamr_out),
        str(wasmtime_out),
        "--parity-skip",
        str(_PARITY_SKIP),
        "--json",
        str(summary_out),
    ]
    if args.strict:
        diff_cmd.append("--strict")

    print(f"\n=== diff ===\n{' '.join(diff_cmd)}", flush=True)
    proc = subprocess.run(diff_cmd, check=False)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
