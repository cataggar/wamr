# WASI Preview 1 + Preview 2 conformance audit

Tracking issue: [#583](https://github.com/cataggar/wamr/issues/583) section C
item 2.

This document is the audit log behind the empty
[`tests/wasi-testsuite-skip.json`](./wasi-testsuite-skip.json). The skip-list
is the source of truth for what gets gated; this file records the matching
fixture counts and how the audit was performed so a future contributor can
re-run it without re-deriving the methodology.

## Result

Date: **2026-05-16**.
Submodule: `tests/wasi-testsuite` @
`40c1f7d35823abfe58d3896501dd660dcf3ff7a7`.
Runner: `tests/wasi-testsuite/test-runner/wasi_test_runner.py` via the in-tree
adapter [`tests/wasi-testsuite-adapter/wamr-zig.py`](./wasi-testsuite-adapter/wamr-zig.py).
Driver: `zig build wasi-testsuite` (see
[`build.zig`](../build.zig) lines 214-241).

| Suite                                   | Fixtures | Passed | Failed | Skipped |
| --------------------------------------- | -------: | -----: | -----: | ------: |
| WASI C tests [wasm32-wasip1]            |       14 |     14 |      0 |       0 |
| WASI Rust tests [wasm32-wasip1]         |       46 |     46 |      0 |       0 |
| WASI Assemblyscript tests [wasm32-wasip1] |     12 |     12 |      0 |       0 |
| **Total**                               |   **72** | **72** |  **0** |   **0** |

Runner output:

```
===== Test results =====
wamr-zig dev: PASS: 72 tests passed
```

## Findings

### F1. No `wasm32-wasip2` corpus exists upstream

The upstream `WebAssembly/wasi-testsuite` repository skipped Preview 2 as a
distinct fixture corpus. Evidence:

- `tests/wasi-testsuite/tests/c/build.py:23`:
  `VERSIONS = ['wasip1'] # + ['wasip2', 'wasip3']` — P2 and P3 are commented
  out for the C suite.
- `tests/wasi-testsuite/doc/writing-tests.md:102`: "Note that until wasip3 is
  released, we use the wasip2 toolchain for wasip3." — the `wasm32-wasip2`
  Rust toolchain target is used only as a *compiler target* for building P3
  fixtures, not for a P2 fixture corpus.
- `find tests/wasi-testsuite -name manifest.json` lists only
  `wasm32-wasip1` (C, Rust, AssemblyScript) and `wasm32-wasip3` (Rust)
  manifests.

**No follow-up issue filed:** there is nothing to gate. If upstream adds a
Preview 2 corpus the audit needs to be rerun and `build.zig` extended to
include it; until then there is no observable gap.

### F2. AssemblyScript suite was running ungated

`build.zig` line 227 already includes
`tests/wasi-testsuite/tests/assemblyscript/testsuite/wasm32-wasip1`, but the
previous skip-list only enumerated the C and Rust suites. The upstream
`JSONTestExcludeFilter` (`tests/wasi-testsuite/test-runner/wasi_test_runner/filters.py`)
treats a missing suite-name entry as "no fixtures skipped", so all 12
AssemblyScript fixtures were already running and passing — but the empty
skip-list silently claimed only two suites.

**Fix in this PR:** added
`"WASI Assemblyscript  tests [wasm32-wasip1]": {}` to
`tests/wasi-testsuite-skip.json` (note the double space — that matches the
upstream `manifest.json` `name` field verbatim) so the empty-skip claim is
explicit for all three suites.

### F3. P3 baseline `http-fields` dev-VM flake (unchanged, out of scope)

`zig build wasi-p3-testsuite` reports `40/40` on CI runners but
`39/40` on this slow dev VM with `http-fields` failing as
`Wait(exit_code=0) failed: timeout expired`. This is the hard-coded 5 s
timeout in
`tests/wasi-testsuite/test-runner/wasi_test_runner/test_suite_runner.py:241`
that's borderline on this VM (~11 s observed) and stays under 5 s on CI
runners. Already tracked under
[#583](https://github.com/cataggar/wamr/issues/583) section A item 7. **Not
in scope for this audit** — listed here so a future re-runner does not
double-file.

## Re-audit: 2026-07-10

Date: **2026-07-10**. Submodule unchanged:
`40c1f7d35823abfe58d3896501dd660dcf3ff7a7`.

PR #661 (AOT-only CLI policy) landed after the audit above and
reintroduced widespread failures, tracked in #662 — the skip-list was
rewritten to exclude all 14 C + 46 Rust + (initially all, later 3)
AssemblyScript fixtures. #662 was later closed claiming its four phase
PRs (#663–#666) fixed the underlying AOT gaps and all skip-lists were
emptied again, but the skip-list on `main` was never actually updated
to match — this re-audit corrects that.

| Suite                                     | Fixtures | Passed | Failed | Skipped |
| ------------------------------------------ | -------: | -----: | -----: | ------: |
| WASI C tests [wasm32-wasip1]               |       14 |     13 |      0 |       1 |
| WASI Rust tests [wasm32-wasip1]            |       46 |     46 |      0 |       0 |
| WASI Assemblyscript  tests [wasm32-wasip1] |       12 |     11 |      0 |       1 |
| **Total**                                  |   **72** | **70** |  **0** |   **2** |

Runner output (after removing the now-stale skip-list entries):

```
===== Test results =====
wamr-zig dev: PASS: 70 tests passed (2 skipped)
```

**Fix in this re-audit:** cleared the `WASI Rust tests [wasm32-wasip1]`
section of `tests/wasi-testsuite-skip.json` to `{}` (all 46 entries now
pass) and reduced the AssemblyScript section from 3 entries to 1
(`environ_sizes_get-no-variables`, which still genuinely fails — its
existing rationale, an environ-inheritance behavioral mismatch rather
than an AOT crash, was already accurate and is unchanged). The C
section was reduced from 14 to 1 entry: `clock_gettime-monotonic` was
initially also cleared, but turned out to be genuinely flaky (not an
AOT/JIT regression) — the upstream fixture compares two
`clock_gettime` reads' `tv_nsec` fields directly without accounting
for `tv_sec` rollover, so it intermittently fails whenever the two
reads straddle a second boundary. Reproduces at similar rates under
both the two-step AOT path and the in-process JIT path, confirmed via
several repeated local runs of each. Re-added with an accurate
rationale rather than left passing-by-luck in CI.

The **P3** (`tests/wasi-p3-testsuite-skip.json`) side of #662's closing
claim did *not* hold up under that re-audit — all 40 fixtures were excluded,
and unfiltered execution showed 38/40 failing with
`UnsupportedCrossInstanceSource`. This was filed as
[#881](https://github.com/cataggar/wamr/issues/881).

## P3 re-audit: 2026-07-13

#881 now resolves inline canonical-builtin sources through the shared AOT
dispatcher, supports async canon-lower ABI/status handling, and preserves
wide scalar cross-instance calls used by socket address records. The complete
corpus executes in both modes:

| Mode | Executed | Passed | Failed | Skipped |
| ---- | -------: | -----: | -----: | -------: |
| persisted `wamrc compile-component` manifest AOT | 40 | 39 | 1 (`sockets-echo`) | 0 |
| no-sidecar `-Djit=true`, `WAMR_JIT_TESTSUITE=1` | 40 | 39 | 1 (`sockets-echo`) | 0 |

`sockets-echo` is unrelated to #881: its TCP listener is created, but the
CLI returns after the initial pending async `wasi:cli/run` lift before the
callback can resume the accept/read task. It is tracked by
[#905](https://github.com/cataggar/wamr/issues/905) and is the only entry in
the filtered convenience skip list. `zig build wasi-p3-testsuite-unfiltered`
and its JIT counterpart do not consume that list: they require all 40
fixtures to execute and assert the exact one-fixture exception.

## Method

```bash
git submodule update --init --recursive --depth 1 tests/wasi-testsuite
zig build wasi-testsuite        # exercised here
zig build wasi-p3-testsuite     # the existing P3 gate, not modified
```

To re-audit, bump the `Submodule` SHA above, re-run, and update the table.
If a new fixture starts failing add a one-line rationale + tracking issue to
the matching suite block in `tests/wasi-testsuite-skip.json`.

## References

- Tracking issue: [#583](https://github.com/cataggar/wamr/issues/583) §C2.
- Wave-6 P3 gate landing: PR [#582](https://github.com/cataggar/wamr/pull/582).
- Build step:
  [`build.zig`](../build.zig) lines 214-241.
- Adapter:
  [`tests/wasi-testsuite-adapter/wamr-zig.py`](./wasi-testsuite-adapter/wamr-zig.py).
- P3 sibling audit format:
  [`tests/wasi-p3-testsuite-skip.json`](./wasi-p3-testsuite-skip.json).
