#!/usr/bin/env python3
"""Diff two `wasi-testsuite` JSON reports and classify the deltas as
either *regressions* (wamr regressed on a fixture the parity runtime
still passes — exit non-zero) or *fixture/runtime bugs* (the parity
runtime fails a fixture wamr still passes — warn but exit zero).

Closes the Wasmtime-parity gate of issue #583 C1: #489 originally
proposed running the same wasm32-wasip3 fixtures through Wasmtime so
a wamr regression that Wasmtime *also* exhibits surfaces as a fixture
bug rather than a wamr bug. Only the wamr-side gate landed in PR
#518; this script + the `wasi-p3-testsuite-wasmtime` step + the
matching CI job close the loop.

Usage
-----

    diff-testsuite-reports.py WAMR.json WASMTIME.json
        [--label-a wamr]
        [--label-b wasmtime]
        [--strict]
        [--json OUTPUT.json]

Exit codes
----------

* 0 — no true regressions (parity-runtime failures on fixtures wamr
  also fails, or parity-runtime failures on fixtures wamr passes,
  are classified as *fixture/runtime bugs* and downgraded to
  warnings on stderr).
* 1 — at least one true regression detected: wamr fails a fixture
  the parity runtime still passes.
* 2 — usage / input error.

`--strict` upgrades fixture/runtime-bug warnings to hard failures so
the parity gate can also enforce wasmtime-side hygiene once the
wasm32-wasip3 baseline ships its own conformance-runtime tests.

The two reports must come from `wasi_test_runner --json-output-location
<path>` against the *same* test-suite paths (the suite name is used
as the join key). Mismatched suites are flagged and the join falls
back to the intersection.

The format produced by the upstream JSON reporter is
documented in
`tests/wasi-testsuite/test-runner/wasi_test_runner/reporters/json.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Mapping from join key (suite_name, test_name) → fixture metadata
# extracted from a single wasi-testsuite JSON report.
FixtureKey = Tuple[str, str]


def _load_report(path: Path) -> Dict[FixtureKey, Dict]:
    try:
        with path.open(encoding="UTF-8") as fp:
            doc = json.load(fp)
    except OSError as exc:
        print(f"error: cannot open {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"error: {path} is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(2)

    fixtures: Dict[FixtureKey, Dict] = {}
    for suite in doc.get("results", []):
        suite_name = suite.get("name", "<unnamed-suite>")
        for test in suite.get("tests", []):
            key = (suite_name, test["name"])
            executed = bool(test.get("executed", True))
            failures = list(test.get("failures") or [])
            fixtures[key] = {
                "executed": executed,
                # A fixture *passes* iff it ran and produced no
                # failures; skipped fixtures are *not* passes.
                "passed": executed and not failures,
                "failures": failures,
            }
    return fixtures


def _runtime_label(path: Path) -> str:
    """Best-effort label extracted from the report's first
    suite — used when no `--label-*` override was supplied. Falls
    back to the JSON path's stem.
    """
    try:
        with path.open(encoding="UTF-8") as fp:
            doc = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return path.stem
    suites = doc.get("results", [])
    if suites and "runtime" in suites[0]:
        name = suites[0]["runtime"].get("name") or path.stem
        version = suites[0]["runtime"].get("version") or ""
        return f"{name} {version}".strip()
    return path.stem


def _format_failures(failures: List[str], limit: int = 1) -> str:
    if not failures:
        return ""
    head = failures[0].splitlines()[0]
    if len(failures) > limit or "\n" in failures[0]:
        head += " …"
    return f" — {head}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diff two wasi-testsuite JSON reports (wamr-side vs a "
            "parity-runtime side) and classify deltas as regressions "
            "(hard fail) or fixture/runtime bugs (warning)."
        ),
    )
    parser.add_argument(
        "report_a",
        type=Path,
        help="Path to the wamr-side JSON report (`zig build wasi-p3-testsuite`).",
    )
    parser.add_argument(
        "report_b",
        type=Path,
        help=(
            "Path to the parity-runtime JSON report "
            "(`zig build wasi-p3-testsuite-wasmtime`)."
        ),
    )
    parser.add_argument(
        "--label-a",
        help="Override the runtime label inferred from report_a.",
    )
    parser.add_argument(
        "--label-b",
        help="Override the runtime label inferred from report_b.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat fixture/runtime-bug-class deltas (parity runtime "
            "fails a fixture wamr passes) as hard failures."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write a machine-readable diff summary.",
    )
    args = parser.parse_args()

    fixtures_a = _load_report(args.report_a)
    fixtures_b = _load_report(args.report_b)

    label_a = args.label_a or _runtime_label(args.report_a)
    label_b = args.label_b or _runtime_label(args.report_b)

    keys_a = set(fixtures_a.keys())
    keys_b = set(fixtures_b.keys())

    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    regressions: List[FixtureKey] = []
    fixture_bugs: List[FixtureKey] = []
    shared_failures: List[FixtureKey] = []

    for key in common:
        a = fixtures_a[key]
        b = fixtures_b[key]
        if a["passed"] and b["passed"]:
            continue
        if not a["passed"] and not b["passed"]:
            shared_failures.append(key)
            continue
        if not a["passed"] and b["passed"]:
            regressions.append(key)
        else:  # a passed, b failed
            fixture_bugs.append(key)

    def _emit_section(title: str, items: List[FixtureKey], src) -> None:
        if not items:
            return
        print(f"\n{title}", file=src)
        for suite_name, test_name in items:
            label = f"  • {suite_name} :: {test_name}"
            failures = (
                fixtures_a.get((suite_name, test_name), {}).get("failures") or []
                if src is sys.stderr
                else fixtures_b.get((suite_name, test_name), {}).get("failures") or []
            )
            print(label + _format_failures(failures), file=src)

    print(f"Comparing {label_a} (A) vs {label_b} (B)", file=sys.stderr)
    print(
        (
            f"  A only: {len(only_in_a)} fixtures, "
            f"B only: {len(only_in_b)} fixtures, "
            f"common: {len(common)} fixtures"
        ),
        file=sys.stderr,
    )

    if only_in_a or only_in_b:
        print(
            "warning: fixture sets are not identical — comparing the "
            "intersection.",
            file=sys.stderr,
        )
        if only_in_a:
            print(f"  fixtures only in {label_a}:", file=sys.stderr)
            for k in only_in_a:
                print(f"    • {k[0]} :: {k[1]}", file=sys.stderr)
        if only_in_b:
            print(f"  fixtures only in {label_b}:", file=sys.stderr)
            for k in only_in_b:
                print(f"    • {k[0]} :: {k[1]}", file=sys.stderr)

    if regressions:
        print(
            f"\nREGRESSIONS ({len(regressions)}): "
            f"{label_a} fails but {label_b} passes — true wamr regressions:",
            file=sys.stderr,
        )
        for suite_name, test_name in regressions:
            failures = fixtures_a[(suite_name, test_name)]["failures"]
            print(
                f"  • {suite_name} :: {test_name}" + _format_failures(failures),
                file=sys.stderr,
            )

    if fixture_bugs:
        print(
            f"\nfixture/runtime bugs ({len(fixture_bugs)}): "
            f"{label_b} fails but {label_a} passes — likely fixture or "
            f"parity-runtime bug, not a wamr issue:",
            file=sys.stderr,
        )
        for suite_name, test_name in fixture_bugs:
            failures = fixtures_b[(suite_name, test_name)]["failures"]
            print(
                f"  • {suite_name} :: {test_name}" + _format_failures(failures),
                file=sys.stderr,
            )

    if shared_failures:
        print(
            f"\nshared failures ({len(shared_failures)}): both runtimes "
            "fail — almost certainly a fixture bug:",
            file=sys.stderr,
        )
        for suite_name, test_name in shared_failures:
            print(f"  • {suite_name} :: {test_name}", file=sys.stderr)

    summary = {
        "labels": {"a": label_a, "b": label_b},
        "only_in_a": [list(k) for k in only_in_a],
        "only_in_b": [list(k) for k in only_in_b],
        "regressions": [list(k) for k in regressions],
        "fixture_bugs": [list(k) for k in fixture_bugs],
        "shared_failures": [list(k) for k in shared_failures],
    }
    if args.json is not None:
        with args.json.open("w", encoding="UTF-8") as fp:
            json.dump(summary, fp, indent=2)
            fp.write("\n")

    # Decide exit code.
    if regressions:
        print(
            f"\n::error::Wasmtime parity diff: {len(regressions)} wamr "
            "regression(s) detected.",
            file=sys.stderr,
        )
        return 1
    if args.strict and fixture_bugs:
        print(
            f"\n::error::Wasmtime parity diff (strict): "
            f"{len(fixture_bugs)} fixture/runtime bug(s) detected.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nWasmtime parity diff: 0 regressions, "
        f"{len(fixture_bugs)} fixture/runtime-bug warning(s), "
        f"{len(shared_failures)} shared failure(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
