# `String.prototype.slice` regression (#754)

Minimal reproducer for the wamr-AOT codegen bug tracked in
[issue #754](https://github.com/cataggar/wamr/issues/754) — the actual
root cause of the cryptic "Resolved to / which is outside package
file://" error from #752.

## What it does

`test.js` defines a `wasi:cli/run.run` export that simply tests
`"hello".slice(0, 3)`:

```js
const got = "hello".slice(0, 3);
if (got === "hel") console.log("PASS slice=...");
else                console.log("FAIL slice=...");
```

Under wasmtime: `PASS slice="hel"`.
Under wamr (current `main`, including this branch's MAX_SLOTS bump):
**`FAIL slice=""`**.

## Why it's not a tiny `.wasm` checked into the repo

`jco componentize` produces a ~13 MiB wasm that bundles the entire
StarlingMonkey/SpiderMonkey engine — large enough that we don't want
to commit it. Rebuild on demand:

```sh
# One-time setup (uses the jco install vendored under
# azure-sdk-for-zig/codegen/tcgc-component/node_modules).
JCO=/path/to/jco

# Build the .wasm from this directory:
$JCO componentize test.js \
    --wit wit/ \
    --world-name repro \
    --out repro.wasm
```

## Wamr invocation

```sh
unset ZIG_LOCAL_CACHE_DIR
export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
zig build -Doptimize=ReleaseSafe

# Bug requires MAX_SLOTS >= ~1000 because the standalone componentize-js
# wasm has more wasi imports than the legacy 256-slot trampoline pool
# can hold — that bump landed in this branch as a prerequisite for
# even getting the JS code to run.
./zig-out/bin/wamrc compile-component -O0 repro.wasm
./zig-out/bin/wamr run repro.wasm   # → "FAIL slice=\"\""
```

## One-command differential (#757)

Had `wamrc verify` (issue #757) existed at the time of the original
bisect, the single command that would have surfaced the bug — and
saved most of the ~6 hours of probe-building — is:

```sh
./zig-out/bin/wamrc verify repro.wasm
```

That spawns both `wasmtime run repro.wasm` and `wamr run repro.wasm`
under the hood, captures both stdouts, and prints the first-divergence
offset with hex+ASCII context on each side. Exit 1 on divergence,
0 on match, 2 on setup error. See `.github/skills/aot-diff-debug/SKILL.md`
for the broader workflow.

## Bisect status (in progress)

- ✅ Disproved: WASI fs, IR-opt passes, canon-lift/lower, `memory.grow`
  external-string dangling (PR #753 made `data.ptr` stable; bug persists).
- ✅ Established: `charCodeAt`, `at`, `[i]`, `codePointAt`,
  `String.fromCharCode`, `length`, string concatenation all work
  correctly. Only `slice` / `substring` / `substr` are broken.
- ❌ Pending: which wasm instruction in
  `js::SubstringKernel` / `js::NewDependentString` / mozjemalloc's
  small-allocation fast path is miscompiled by wamr-AOT?

A simpler-than-StarlingMonkey reproducer (Zig wasm doing
memcpy + heap alloc + extern-struct writes) **passes** under wamr-AOT,
so the bug is not in those primitives. Likely candidates remaining:
SIMD shuffle on host without v128 detection,
`i32.atomic.rmw.cmpxchg` in slab allocation,
specific `i64.shl` shift used to encode dependent-string flag bits,
or a `call_indirect` against the per-encoding slice helper vtable.
