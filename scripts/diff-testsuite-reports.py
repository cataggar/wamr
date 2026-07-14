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
        [--parity-skip PATH]
        [--strict]
        [--json OUTPUT.json]

Exit codes
----------

* 0 — no true regressions (parity-runtime failures on fixtures wamr
  also fails, or parity-runtime failures on fixtures wamr passes,
  are classified as *fixture/runtime bugs* and downgraded to
  warnings on stderr — or to *documented* deltas when the fixture
  is listed in `--parity-skip`).
* 1 — at least one true regression detected: wamr fails a fixture
  the parity runtime still passes. Also returned when a fixture in
  the `--parity-skip` list is no longer a fixture-bug (wamr-pass /
  parity-fail) so the skip-list does not silently rot.
* 2 — usage / input error.

`--strict` upgrades fixture/runtime-bug warnings to hard failures so
the parity gate can also enforce wasmtime-side hygiene once the
wasm32-wasip3 baseline ships its own conformance-runtime tests.
Entries in `--parity-skip` are exempt from `--strict`: they are
treated as already-tracked work and never fail the gate as long as
the wamr-pass / parity-fail shape still holds.

`--parity-skip PATH` consumes a JSON file mapping a fixture's
`test_name` to a human-readable tracking pointer (typically the URL
of an upstream issue):

    {
      "_comment": "Wasmtime parity-skip — fixtures wamr passes but wasmtime fails. Keyed by fixture name (matches `<test>.wasm` under `tests/rust/testsuite/wasm32-wasip3/`).",
      "http-service": "tracking https://github.com/WebAssembly/wasi-testsuite/issues/228",
      "sockets-tcp-connect": "tracking https://github.com/bytecodealliance/wasmtime/issues/13396"
    }

Keys starting with `_` (e.g. `_comment`) are ignored so the file
can carry its own documentation. The skip list lives at
`tests/wasi-p3-parity-skip.json` in this repo and is the
authoritative inventory of known-tracked wasmtime / fixture-side
deltas.

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


def _load_parity_skip(path: Path) -> Dict[str, str]:
    """Load a `--parity-skip` JSON map of `test_name → tracking pointer`.

    Keys starting with `_` (e.g. `_comment`) are filtered out so the
    file can carry inline documentation. Returns an empty dict when
    `path` is `None`.
    """
    if path is None:
        return {}
    try:
        with path.open(encoding="UTF-8") as fp:
            doc = json.load(fp)
    except OSError as exc:
        print(
            f"error: cannot open parity-skip file {path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(
            f"error: parity-skip file {path} is not valid JSON: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not isinstance(doc, dict):
        print(
            f"error: parity-skip file {path} must be a JSON object "
            "(`{\"<fixture>\": \"<tracking pointer>\"}`).",
            file=sys.stderr,
        )
        sys.exit(2)
    skip: Dict[str, str] = {}
    for k, v in doc.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        if not isinstance(k, str) or not isinstance(v, str):
            print(
                f"error: parity-skip entry {k!r}={v!r} in {path} is not "
                "a `<fixture>: <tracking pointer>` string pair.",
                file=sys.stderr,
            )
            sys.exit(2)
        skip[k] = v
    return skip


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
        "--parity-skip",
        type=Path,
        help=(
            "Path to a JSON file listing fixtures whose wasmtime "
            "failure is documented under a tracking issue. Each entry "
            "is `\"<fixture>\": \"<tracking pointer>\"`. Matched "
            "fixture-bugs are downgraded to *documented* deltas that "
            "never fail the gate (even with --strict). If a fixture "
            "listed here is no longer in the wamr-pass / parity-fail "
            "shape, the gate fails so the list cannot rot."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat fixture/runtime-bug-class deltas (parity runtime "
            "fails a fixture wamr passes) as hard failures. Fixtures "
            "listed in --parity-skip are exempt — they are tracked "
            "via the cited issue and never fail the gate."
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
    parity_skip = _load_parity_skip(args.parity_skip)

    label_a = args.label_a or _runtime_label(args.report_a)
    label_b = args.label_b or _runtime_label(args.report_b)

    keys_a = set(fixtures_a.keys())
    keys_b = set(fixtures_b.keys())

    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    regressions: List[FixtureKey] = []
    fixture_bugs: List[FixtureKey] = []
    documented: List[FixtureKey] = []
    shared_failures: List[FixtureKey] = []
    shared_skips: List[FixtureKey] = []
    execution_mismatches: List[FixtureKey] = []

    for key in common:
        a = fixtures_a[key]
        b = fixtures_b[key]
        if not a["executed"] or not b["executed"]:
            if not a["executed"] and not b["executed"]:
                shared_skips.append(key)
            else:
                execution_mismatches.append(key)
            continue
        if a["passed"] and b["passed"]:
            continue
        if not a["passed"] and not b["passed"]:
            shared_failures.append(key)
            continue
        if not a["passed"] and b["passed"]:
            regressions.append(key)
        else:  # a passed, b failed
            if key[1] in parity_skip:
                documented.append(key)
            else:
                fixture_bugs.append(key)

    # Fixtures listed in the skip-list that are *not* currently in
    # the documented shape (wamr-pass / parity-fail) are stale — fail
    # the gate so the list cannot rot. Two sub-cases:
    #
    #   * wamr fails (`regressions` or `shared_failures`) — would have
    #     been caught above as a regression / shared failure;
    #     additionally call out the entry so the maintainer knows the
    #     skip-list entry can be retired.
    #   * both wamr and the parity runtime now pass — the upstream
    #     wasmtime / fixture fix has landed; remove the entry.
    stale_skip: List[Tuple[str, str]] = []
    documented_test_names = {k[1] for k in documented}
    for test_name, tracking in sorted(parity_skip.items()):
        if test_name in documented_test_names:
            continue
        # Resolve the matching fixture key (skip-list is keyed by
        # test_name only; resolve the full (suite, test) key from the
        # reports so the operator sees the same identity as elsewhere
        # in the diff output).
        matching_keys = [k for k in common if k[1] == test_name]
        if not matching_keys:
            stale_skip.append(
                (test_name, "fixture not present in either report")
            )
            continue
        # If the parity runtime now passes the listed fixture too the
        # entry is stale and the upstream bug is fixed.
        k = matching_keys[0]
        if fixtures_a[k]["passed"] and fixtures_b[k]["passed"]:
            stale_skip.append(
                (test_name, "both runtimes now pass — drop entry")
            )

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

    if documented:
        print(
            f"\ndocumented fixture/runtime bugs ({len(documented)}): "
            f"{label_b} fails but {label_a} passes — tracked under the "
            "cited upstream issue(s):",
            file=sys.stderr,
        )
        for suite_name, test_name in documented:
            failures = fixtures_b[(suite_name, test_name)]["failures"]
            tracking = parity_skip.get(test_name, "<no tracking>")
            print(
                f"  • {suite_name} :: {test_name} [{tracking}]"
                + _format_failures(failures),
                file=sys.stderr,
            )

    if stale_skip:
        print(
            f"\nstale parity-skip entries ({len(stale_skip)}): listed "
            "in the skip-list but the fixture is no longer in the "
            "wamr-pass / parity-fail shape:",
            file=sys.stderr,
        )
        for test_name, why in stale_skip:
            tracking = parity_skip.get(test_name, "<no tracking>")
            print(
                f"  • {test_name} [{tracking}] — {why}",
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

    if shared_skips:
        print(
            f"\nshared skips ({len(shared_skips)}): both runtimes excluded "
            "the same expected fixture(s):",
            file=sys.stderr,
        )
        for suite_name, test_name in shared_skips:
            print(f"  • {suite_name} :: {test_name}", file=sys.stderr)

    if execution_mismatches:
        print(
            f"\nexecution mismatches ({len(execution_mismatches)}): one "
            "runtime executed a fixture that the other skipped:",
            file=sys.stderr,
        )
        for suite_name, test_name in execution_mismatches:
            print(f"  • {suite_name} :: {test_name}", file=sys.stderr)

    summary = {
        "labels": {"a": label_a, "b": label_b},
        "only_in_a": [list(k) for k in only_in_a],
        "only_in_b": [list(k) for k in only_in_b],
        "regressions": [list(k) for k in regressions],
        "fixture_bugs": [list(k) for k in fixture_bugs],
        "documented": [
            {"suite": k[0], "test": k[1], "tracking": parity_skip.get(k[1], "")}
            for k in documented
        ],
        "stale_skip": [
            {"test": t, "why": w, "tracking": parity_skip.get(t, "")}
            for t, w in stale_skip
        ],
        "shared_failures": [list(k) for k in shared_failures],
        "shared_skips": [list(k) for k in shared_skips],
        "execution_mismatches": [list(k) for k in execution_mismatches],
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
    if stale_skip:
        print(
            f"\n::error::Wasmtime parity diff: {len(stale_skip)} stale "
            "parity-skip entry(ies) — update tests/wasi-p3-parity-skip.json.",
            file=sys.stderr,
        )
        return 1
    if execution_mismatches:
        print(
            f"\n::error::Wasmtime parity diff: "
            f"{len(execution_mismatches)} execution mismatch(es) detected.",
            file=sys.stderr,
        )
        return 1
    if args.strict and fixture_bugs:
        print(
            f"\n::error::Wasmtime parity diff (strict): "
            f"{len(fixture_bugs)} undocumented fixture/runtime bug(s) "
            "detected (add to tests/wasi-p3-parity-skip.json with a "
            "tracking issue, or fix upstream).",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nWasmtime parity diff: 0 regressions, "
        f"{len(fixture_bugs)} undocumented fixture/runtime-bug warning(s), "
        f"{len(documented)} documented fixture/runtime-bug(s), "
        f"{len(shared_failures)} shared failure(s), "
        f"{len(shared_skips)} shared skip(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
