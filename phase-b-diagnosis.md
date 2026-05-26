# Phase B Diagnosis — AS startup abort under `wamr run` (#662)

## TL;DR

The abort is **not** an AOT codegen bug. It is a missing‑data bug in the in‑process
AOT‑compile helper used by `wamr run` for plain core wasm
(`src/component/aot.zig::compileCoreWasm`). That helper passes `null` to
`emit_aot.emit` for both the **globals** and **element segments**, so the
emitted `.cwasm` has an empty globals section. AssemblyScript modules rely on
a mutable `i32` global as their shadow stack pointer; when the loader brings
it up at `0`, every stack‑frame prologue underflows and the runtime traps
inside AS's own `abort()` handler — which then formats and prints the famous
`abort:  in (1:1)` line via `fd_write` and calls `proc_exit(255)`.

The fix is small (one helper). It is **not** related to Phase A's calling
convention work; it is a regression introduced by the in‑process compile path
added for `runCoreAot`/#644 deliberately leaving globals + elems "for phase 3".

## Reproducer

```
cd /work/wamr-662b
zig build
./zig-out/bin/wamr run tests/wasi-testsuite/tests/assemblyscript/testsuite/wasm32-wasip1/fd_write-to-stdout.wasm
# → abort:  in (1:1)  ; exit 255

# Compile the SAME wasm with the standalone wamrc (no -O0!) and run the .cwasm:
./zig-out/bin/wamrc compile --no-verify-ir \
  tests/wasi-testsuite/tests/assemblyscript/testsuite/wasm32-wasip1/fd_write-to-stdout.wasm \
  -o /tmp/fd.cwasm
./zig-out/bin/wamr run /tmp/fd.cwasm
# → hello5  ; exit 0
```

Same IR pipeline, same target, same passes — but the auto‑AOT path aborts and
the file‑based path succeeds. The only difference between the two is which
`emit_aot.emit` arguments are populated.

## The two paths

### `wamrc compile` (works) — `src/compiler/main.zig` line 422‑531

Builds `global_entries` (imports first, then locals, evaluating each
`init_expr`) and `elem_entries`, then calls:

```zig
const aot_binary = try emit_aot.emit(
    allocator,
    compiled.code,
    compiled.offsets,
    exports.items,
    .{ .arch = arch_name },
    if (data_segs.items.len > 0) data_segs.items else null,
    if (import_entries.items.len > 0) import_entries.items else null,
    if (mem_entries.items.len > 0) mem_entries.items else null,
    if (global_entries.items.len > 0) global_entries.items else null,   // ← populated
    if (elem_entries.items.len > 0) elem_entries.items else null,       // ← populated
    module.start_function,
    …
);
```

### `wamr run …wasm` (fails) — `src/component/aot.zig::compileCoreWasm` line 308‑330

```zig
return emit_aot.emit(
    allocator,
    code,
    offsets,
    exports.items,
    .{ .arch = arch_name },
    if (data_segs.items.len > 0) data_segs.items else null,
    if (imports.items.len > 0) imports.items else null,
    if (mem_entries.items.len > 0) mem_entries.items else null,
    // Globals + elems are not yet populated by this helper. The
    // motivating workload (componentize-js cores) initialises its
    // own globals through data segments + start code, and the
    // existing AOT loader tolerates empty global/elem sections …
    // Wired in fully when phase 3 lights up canon-lift onto AOT
    // exports for components that exercise top-level globals.
    null,                                                             // ← globals dropped
    null,                                                             // ← elems dropped
    module.start_function,
    …
);
```

The block comment explicitly states that globals + elems are deferred. That
assumption is only true for componentize‑js cores whose globals are managed
by canon‑lift wrappers; it is **false** for plain `wasm32-wasip1` modules
(AssemblyScript, Rust/wasi‑sdk, clang/wasi‑libc), all of which carry an
explicit `(global (mut i32))` shadow stack pointer in the module.

## Why these particular AS tests fail

`fd_write-to-stdout.wasm` declares **1 memory + 1 mutable i32 global** with
`init_expr = i32.const 17804` (the AS shadow stack base; address grows
downward). Every function prologue does:

```
global.get 0       ;; load SP
i32.const N
i32.sub
global.set 0       ;; reserve N bytes
```

With the global missing the loader allocates a backing slot zero‑initialized,
so SP = 0. The first frame computes `0 - N` (huge unsigned) and stores it
back as the new SP. Subsequent loads/stores at `[SP+k]` either fault on a
linear‑memory bounds check or scribble over the data segment region; either
way AS's runtime detects the corruption and dives into
`~lib/builtins/abort` with `message=null, fileName=null, line=1, col=1` —
exactly the observed output.

The 6 currently‑passing AS fixtures all have one of:

* No user‑level locals beyond their parameters → no `global.get $sp` /
  `global.set $sp` pair in the prologue (`proc_exit-success`,
  `proc_exit-failure`).
* Only sizes_get calls whose AS wrappers happen to optimize the prologue
  out at compile time (the `*_sizes_get-no-*` and `_sizes_get-multiple-*`
  tests).
* `random_get-non-zero-length` happens to bring SP up only by 16 bytes and
  then `wasi.random_get` writes that 16‑byte buffer — and the buffer write
  lands inside the data segment region, which the AOT loader does populate
  via `data_segs`, so the test "accidentally" survives until `proc_exit`.

In other words: passing vs. failing splits along *whether the body uses the
shadow stack*, not along "passes a non‑empty pointer". The user's hypothesis
(arg‑buffer access) and the actual cause (SP global at 0) happen to correlate
because anything that touches strings touches the shadow stack.

## How this was confirmed

1. **Pass bisection**: with `runPassesWithOptions` short‑circuited to a no‑op
   (every IR opt pass effectively disabled) the failing `wamr run` path still
   aborts. Rules out every IR pass (`inlineSmallFunctions`,
   `promoteLocalsToSSA`, `forwardLocalGet`, `foldConstantBranches`,
   `scrubUnreachableBlocks`, …) as the culprit.

2. **Path comparison**: compiling the wasm with `wamrc compile --no-verify-ir`
   (full optimization pipeline, same target, same passes) and then running the
   resulting `.cwasm` with `wamr run` succeeds. Confirms the divergence is not
   in the IR / codegen, but in the *arguments to* `emit_aot.emit`.

3. **Source diff**: comparing `runCompile` (`src/compiler/main.zig`) against
   `compileCoreWasm` (`src/component/aot.zig`) shows the global_entries and
   elem_entries build is missing in the latter.

## Side observation (not the root cause)

`./zig-out/bin/wamrc compile` (which uses debug runtime safety) trips the IR
verifier on every AS fixture:

```
IR verifier: MissingPredecessor after pass 'inlineSmallFunctions'
             func #0 block #1 — predecessor list omits a block whose
             terminator targets this block
```

This is a **verifier‑only** issue: `BasicBlock.predecessors.items` is read
only by `verifier.zig` and `print.zig`. The actual passes
(`promoteLocalsToSSA`, dominator computation, liveness) recompute
predecessors on the fly via `analysis.buildPredecessors`. The bug is that
the frontend never populates `block.predecessors`, and
`inlineSmallFunctionsCount` only refreshes it for caller functions that were
actually mutated (`caller_changed[i]`). So a never‑inlined‑into function
reaches the post‑pass `Verify.check` with empty `predecessors` while the
verifier derives its expected set from the LAST‑instruction terminator. The
verifier complains, but codegen is unaffected.

Worth filing as a separate small issue (`refresh predecessors for every
function, or have the frontend populate predecessors`) but it is **not** the
runtime abort and we should not change it under #662 Phase B.

## Proposed fix

Port the globals + elements construction from `runCompile`
(`src/compiler/main.zig:422-483`) into `compileCoreWasm`
(`src/component/aot.zig`). The helpers
`defaultZeroValue` / `valueToI64` / `valueToV128` are currently file‑private
in `src/compiler/main.zig`; either:

* (a) make them `pub` and import, **or**
* (b) duplicate them into `src/component/aot.zig` (they are 30 lines total
  and trivially correct).

I will do (b) inside this PR to keep the change strictly additive and
local to `src/component/aot.zig`. A follow‑up can dedupe.

The element segment build is straightforward — for every non‑declarative
segment, evaluate the offset (`i32_const → u32`) and pack the funcref list,
mapping a missing index to the `0xFFFFFFFF` null sentinel that the loader
already understands.

The global build needs `wamr.instance.evalInitExpr`, which is already
re‑exported through `wamr` and is what `runCompile` uses today.

## Size estimate

* Fix in `src/component/aot.zig`: ~60 lines (globals build + elem build +
  three value helpers). Well under the "small" threshold.
* No new module structure required.
* No IR / codegen changes.
* No interp fallback (still AOT‑only).

## Validation plan

* Run all 12 AS fixtures via `wamr run`:
  * 6 currently passing must still pass.
  * 6 currently failing must produce either success or the **AS‑level**
    abort with a source location (`src/...ts(L:C)`). Both indicate the SP
    global is now alive; the `(L:C)` aborts are the upstream test asserting
    that argv/env didn't match — that's wasi-testsuite harness territory,
    not Phase B's responsibility, and matches `-O0` behaviour today.
* `zig build test --summary failures` must remain clean.
* `tests/wasi-testsuite-skip.json` "WASI Assemblyscript  tests
  [wasm32-wasip1]" entry can drop the 6 fixed test names.

## Out‑of‑scope follow‑ups (NOT in this PR)

* Frontend should populate `BasicBlock.predecessors`, or
  `inlineSmallFunctionsCount` should refresh **every** function (cheap)
  rather than only caller‑changed ones, so safe‑build verifier passes match
  release codegen behaviour.
* C and Rust wasi‑testsuite fixtures still fail under `wamr run` — those
  appear to need additional work (likely the same globals fix may help a
  subset, but the WASI ABI shims/argv handling for `wasm32-wasip1` also
  surfaced earlier failures). Tracked under #662 Phases A/C.
