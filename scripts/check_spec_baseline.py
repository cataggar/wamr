#!/usr/bin/env python3
"""Compare live spec-test results against tests/spec-baseline.tsv.

Runs every `.wast` file in the vendored WebAssembly/testsuite through
`spec-test-runner` and fails (exit code 1) on any drift:

    - fewer passes than baseline,
    - more failures than baseline,
    - more skips than baseline,
    - a crash / timeout / no-output (always a failure, never baselined),
    - a baseline file missing from the testsuite,
    - a testsuite .wast file missing from the baseline.

Each file runs in its own subdirectory because the runner writes
`.wamr-test-<name>` scratch directories into the current directory.

Pass --update to overwrite the baseline with current results instead of
checking.

Usage:
    zig build -Doptimize=ReleaseSafe
    python3 scripts/check_spec_baseline.py
    python3 scripts/check_spec_baseline.py --update
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNNER = REPO_ROOT / "zig-out" / "bin" / "spec-test-runner"
DEFAULT_SUITE = REPO_ROOT / "third_party" / "testsuite"
DEFAULT_BASELINE = REPO_ROOT / "tests" / "spec-baseline.tsv"

COUNTS_RE = re.compile(r"pass=(\d+)\s+fail=(\d+)\s+skip=(\d+)")

HEADER = """\
# Spec-test baseline. Each line lists the expected pass/fail/skip counts
# for one .wast file under third_party/testsuite/, as produced by
# zig-out/bin/spec-test-runner. CI compares live output against this file
# and fails on any drift. Regenerate with:
#     python3 scripts/check_spec_baseline.py --update
"""


def load_baseline(path: Path) -> dict[str, tuple[int, int, int]]:
    baseline: dict[str, tuple[int, int, int]] = {}
    if not path.exists():
        return baseline
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, p, f, s = line.split("\t")
        baseline[name] = (int(p), int(f), int(s))
    return baseline


def run_one(args: tuple[Path, Path, Path]) -> tuple[str, tuple[int, int, int] | None, str]:
    wast, runner, scratch = args
    workdir = scratch / wast.stem
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [str(runner), str(wast)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return wast.name, None, "TIMEOUT"

    output = proc.stdout + proc.stderr
    match = COUNTS_RE.search(output)
    if proc.returncode != 0:
        if "Segmentation fault" in output:
            return wast.name, None, "SEGFAULT"
        return wast.name, None, f"EXIT{proc.returncode}"
    if match is None:
        return wast.name, None, "NO-OUTPUT"
    return wast.name, tuple(int(x) for x in match.groups()), "OK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--update", action="store_true")
    opts = parser.parse_args()

    if not opts.runner.exists():
        print(f"error: runner not found: {opts.runner}", file=sys.stderr)
        print("hint: zig build -Doptimize=ReleaseSafe", file=sys.stderr)
        return 1

    wast_files = sorted(opts.suite.glob("*.wast"))
    if not wast_files:
        print(f"error: no .wast files under {opts.suite}", file=sys.stderr)
        print("hint: git submodule update --init third_party/testsuite", file=sys.stderr)
        return 1

    scratch = Path(tempfile.mkdtemp(prefix="wamr-spec-baseline-"))
    try:
        work = [(w, opts.runner, scratch) for w in wast_files]
        results: dict[str, tuple[tuple[int, int, int] | None, str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=opts.jobs) as pool:
            for name, counts, status in pool.map(run_one, work):
                results[name] = (counts, status)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if opts.update:
        lines = [HEADER]
        for name in sorted(results):
            counts, status = results[name]
            if status != "OK" or counts is None:
                print(f"error: refusing to baseline {name}: {status}", file=sys.stderr)
                return 1
            lines.append(f"{name}\t{counts[0]}\t{counts[1]}\t{counts[2]}\n")
        opts.baseline.write_text("".join(lines))
        total = sum(c[0][0] for c in results.values() if c[0])
        print(f"wrote {opts.baseline} ({len(results)} files, {total} passing assertions)")
        return 0

    baseline = load_baseline(opts.baseline)
    if not baseline:
        print(f"error: baseline {opts.baseline} is empty or missing", file=sys.stderr)
        return 1

    drift: list[str] = []

    for name in sorted(set(baseline) | set(results)):
        if name not in results:
            drift.append(f"{name}: in baseline but missing from testsuite")
            continue
        if name not in baseline:
            drift.append(f"{name}: in testsuite but missing from baseline")
            continue
        counts, status = results[name]
        if status != "OK" or counts is None:
            drift.append(f"{name}: {status} (baseline {baseline[name]})")
            continue
        bp, bf, bs = baseline[name]
        p, f, s = counts
        if p < bp:
            drift.append(f"{name}: passes {p} < baseline {bp}")
        if f > bf:
            drift.append(f"{name}: failures {f} > baseline {bf}")
        if s > bs:
            drift.append(f"{name}: skips {s} > baseline {bs}")

    total_pass = sum(c[0][0] for c in results.values() if c[0])
    total_base = sum(v[0] for v in baseline.values())
    print(f"{len(results)} files, {total_pass} passing assertions (baseline {total_base})")

    if drift:
        print(f"\nspec baseline drift ({len(drift)}):", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nIf this is an intentional improvement, regenerate with:\n"
            "    python3 scripts/check_spec_baseline.py --update",
            file=sys.stderr,
        )
        return 1

    print("no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
