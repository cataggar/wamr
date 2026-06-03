# IR optimizer property tests

Issue #736 adds an in-process IR interpreter and deterministic IR generators so
optimizer passes can be checked by behavior instead of output shape alone.

`zig build test` runs a small CI-sized matrix:

```sh
unset ZIG_LOCAL_CACHE_DIR
export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
zig build test --summary failures
```

For a larger local/nightly run, increase the per-pass/per-shape iteration count:

```sh
unset ZIG_LOCAL_CACHE_DIR
export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
zig build test -Dir-property-iters=100 --summary failures
```

On mismatch, `src/compiler/ir/property_test.zig` prints:

```text
IR property mismatch: pass=<pass> seed=0x<seed> shape=<shape>
```

To promote a failing case into a named regression, add a focused test to
`property_test.zig` that calls `checkPassPreservesCase` with the reported
`seed`, `shape`, and pass. Keep the generated shape deterministic rather than
serializing IR text.

Current scope is intentionally scalar: i32/i64 arithmetic, locals/globals,
memory, branches, phis, parallel copies, and mock calls as memory barriers.
Floats, SIMD, EH, tables, atomics, and bulk-memory ops are outside this first
oracle slice and should either remain ungenerated or return `unsupported` from
the interpreter until explicitly modeled.
