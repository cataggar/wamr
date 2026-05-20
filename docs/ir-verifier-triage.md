# IR verifier — differential corpus triage (issue #627)

The IR verifier landed in [#626](https://github.com/cataggar/wamr/pull/626) /
[#624](https://github.com/cataggar/wamr/issues/624) is currently opt-in for the
differential test suite (`InitOptions.verify_ir = false`). Flipping the default
to `true` requires that every test currently passing under the legacy
(verifier-off) pipeline also pass under verification. This page records the
shape of that work: which passes the verifier currently catches red-handed,
which invariants they violate, and which sub-issue tracks the fix.

## Reproducing

```sh
zig build test -Dverify-ir-triage --summary all
```

Each compile that fails verification prints a `triage:` line to stderr that
encodes the failing pass + invariant + location. The test itself fails with
`error.CompileFailed`, exactly so the test runner stamps the test name above
the verifier diagnostic.

When every entry in the table below has been fixed, flip the default in
`src/tests/aot_harness.zig:InitOptions.verify_ir` to `true` and remove this
file's "currently failing" caveat. That is the closing task of #627.

## Currently failing (as of branch `issue-627-fuzz-verifier-triage`)

11 of 200 differential tests fail under `-Dverify-ir-triage`. They reduce to
**3 distinct (pass, invariant) pairs**:

| # | Failing pass | Invariant | Tests | Sub-issue |
|---|--------------|-----------|-------|-----------|
| 1 | `inlineSmallFunctions` | `MissingPredecessor` | 8 SIMD param/result tests | [#630](https://github.com/cataggar/wamr/issues/630) |
| 2 | `inlineSmallFunctions` | `MultipleTerminators` | `two-function linked list (build + traverse)` | [#631](https://github.com/cataggar/wamr/issues/631) |
| 3 | `promoteLocalsToSSA`   | `MissingPredecessor` | `crcu8 CoreMark CRC kernel`, `linked list traversal in memory` | [#632](https://github.com/cataggar/wamr/issues/632) |

Verifier messages (verbatim):

- `IR verifier: MissingPredecessor after pass 'inlineSmallFunctions' func #1 block #1 — predecessor list omits a block whose terminator targets this block`
- `IR verifier: MissingPredecessor after pass 'promoteLocalsToSSA' func #0 block #1 — predecessor list omits a block whose terminator targets this block`
- `IR verifier: MultipleTerminators after pass 'inlineSmallFunctions' func #1 block #4 inst #10 — terminator before end of block`

### Tests, grouped by (pass, invariant)

`inlineSmallFunctions` / `MissingPredecessor`:
- `differential SIMD: direct v128 param preserves f32 NaN payload bits`
- `differential SIMD: direct v128 param preserves f32 signed zero bits`
- `differential SIMD: direct v128 result preserves f32 NaN payload bits`
- `differential SIMD: direct v128 result preserves f32 signed zero bits`
- `differential SIMD: excess v128 params use stack`
- `differential SIMD: global v128 value passed as param`
- `differential SIMD: local v128 value passed as param`
- `differential SIMD: mixed scalar and v128 params`

`inlineSmallFunctions` / `MultipleTerminators`:
- `differential: two-function linked list (build + traverse)`

`promoteLocalsToSSA` / `MissingPredecessor`:
- `differential: crcu8(0x53, 0xe9f5) — CoreMark CRC kernel`
- `differential: linked list traversal in memory`

## Fuzz corpora

`tests/fuzz/regression/wasi/` is empty in-tree; there is no committed corpus to
replay against `fuzz_aot` / `fuzz_diff`. The fuzz targets themselves already
opt in to `verify_ir = true` (see `src/tests/fuzz/aot.zig` and
`src/tests/fuzz/diff.zig`), so as soon as a corpus is reintroduced the
verifier will exercise it. Adding a checked-in regression corpus is tracked
separately (the empty `regression/wasi` directory is the existing hook).
