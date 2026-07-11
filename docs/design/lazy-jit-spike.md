# Lazy JIT design spike: per-function on-demand compilation

Status: **NARROW PROTOTYPE** (leaf functions only, currently implemented on x86_64 and aarch64; see [Scope](#scope) and [Deferred to follow-up work](#deferred-to-follow-up-work)).

Tracking: [#862](https://github.com/cataggar/wamr/issues/862) (Milestone 4 of the in-process JIT plan, [#863](https://github.com/cataggar/wamr/issues/863)). Depends on the core-wasm in-process JIT ([#853](https://github.com/cataggar/wamr/issues/853)) and the thread-safety audit ([#859](https://github.com/cataggar/wamr/issues/859)).

## Motivation

The in-process JIT (#852-#861) compiles a module's **entire** function set up front, synchronously, before the first export call returns. For a large module where most functions are never actually invoked in a given run (a common shape — CLI tools, WASI components with big generated bindings, etc.), that's wasted compile latency squarely on the user-visible cold-start path this whole JIT effort exists to shrink. wasmtime/V8-class JITs solve this with **lazy, per-function compilation**: compile a stub for every function at load time, and only do the real work the first time each stub is actually called.

This spike wires up the existing (until now, unused) `config.lazy_jit` build flag stub (`src/config.zig:103`) with a real, narrowly-scoped prototype, to de-risk the approach before committing to a full implementation.

## What already exists that makes this tractable

- Both native backends are organized per function: `pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8` exists independently in `src/compiler/codegen/x86_64/compile.zig` and `src/compiler/codegen/aarch64/compile.zig`. The primitive to "compile just function N" already exists.
- `platform.mapExecutableCode` (#858) already handles the W^X-safe mmap→copy→icache-flush→mprotect sequence for a single blob of native code — exactly what a lazily-compiled function's code needs.
- `mapCodeExecutable` (`runtime.zig`) already builds the `func_addrs[]`/`funcptrs[]` table used to resolve a function's native entry point — the natural patch point for "not compiled yet."

## What actually made this hard

### Direct calls are relative-branch patches computed at whole-module compile time

Both backends' `compileModuleWithOptions`/`compileModuleCachedWithOptions` compile every function into one contiguous code blob, then resolve every direct `.call` as a `BL`(aarch64)/`call rel32`(x86_64) whose target offset is only known **after all functions have been compiled and laid out**. A function whose compilation is deferred has no offset yet — any *other* function's `.call` to it can't be patched. This is the crux of why "lazy" can't simply mean "don't call `compileFunction` yet" for arbitrary functions.

### `call_indirect` / `ref.func` assume every `func_addrs[]` entry is resolved at load time

`runtime.zig`'s `getFuncAddr`, the `ptr_to_sig` sorted map, and `call_indirect`'s codegen all assume `func_addrs[i]` is a real, callable native address for every `i` by the time `mapCodeExecutable` returns. Deferring a function reachable via `call_indirect` (i.e. placed in a table by an active element segment) would need that codegen path itself — and the runtime's table dispatch — to route through some form of "is this a stub? if so, compile now" check. That's a materially bigger change than this spike attempts.

## The narrow spike's design

Given the two problems above, this prototype restricts "lazy-eligible" to functions where **neither problem can occur**:

1. **Leaf**: the function's own IR contains no `.call`/`.call_indirect` instruction. (If it called something, that something would need to already be compiled — leaf functions have no such dependency.)
2. **Never a direct call target**: no other function's IR contains a `.call` naming this function. (So no BL/`call rel32` patch will ever need this function's offset.)
3. **Never reachable via `call_indirect`**: conservatively, if the module has **any** element segments at all, no function in it is eligible. A function placed in any table by any active element segment could be invoked via `call_indirect`, which this prototype's hook does not cover.

See `src/compiler/lazy_jit.zig`'s `findLazyEligibleLeaves` for the exact analysis (with unit tests covering all three rules).

### Compile-time: skip codegen, not IR lowering/optimization

`compileModuleCachedWithOptions` (now on both native backends) gained an additive `CompileOptions.lazy_skip: []const bool` field (indexed by local function index, empty by default — zero behavior change for every existing caller). For an index where `lazy_skip[i]` is true, the per-function loop writes a **zero-byte placeholder** instead of calling into the real per-function codegen. This is safe only because of the two invariants above: the function is never a patch target (nothing references its now-nonexistent offset) and it makes no outgoing calls (nothing for the eager pass to fail to resolve).

Note this spike does **not** skip running IR optimization passes on lazy-eligible functions — only the final codegen (instruction selection / register allocation / emission) is deferred. Skipping passes too is a real follow-up opportunity (see below); per the earlier CoreMark pass-timing analysis from #860, GVN/loop-invariant-hoisting/dominator-based redundant-load-forwarding are the expensive ones, and none of that is wasted work here in the sense that it *could* still be avoided later without touching this spike's core mechanism.

`PrecompileOptions` gained a `lazy_jit: bool` opt-in flag and `CompileCacheCtx` gained a `lazy_jit_out: ?*LazyJitOut`. When both are set, `compileCoreWasmCached` computes the eligible set, threads it into the selected backend's codegen call, and — instead of freeing the lowered IR module at the end like it normally does — **moves it** into the caller-supplied `LazyJitOut` (along with the compact list of deferred local indices), so the deferred functions' IR survives past the call for later on-demand compilation.

### Runtime: a per-instance hook in `callFuncScalar`, not a native trampoline

The "real" version of this feature needs `func_addrs[i]` for a not-yet-compiled function to point at a small native trampoline: a per-architecture shim (much like the existing host-import trampolines in `host_trampolines.zig`) that calls back into the runtime to compile the function, then either tail-jumps into the freshly compiled code or (simpler) just forwards the call through a cached-after-first-time indirection. `host_trampolines.zig`'s existing `TrampolinePool`/`genericDispatcher` machinery is very nearly reusable off the shelf for this — its "forward up to 10 lowered flat-u64 args to a Zig callback, get a u64 back" stub shim is exactly shaped for a leaf function with i32/i64/pointer-class params and a single scalar result. The reason this spike **doesn't** do that: `TrampolinePool` today is only wired up for `ComponentInstance` (the component-model host-import path), not plain `AotInstance` — extending it to core-wasm-only instances is itself nontrivial plumbing, and doing it *and* validating the trampoline machine code on the narrow leaf-only case in one session was too much for a spike.

Instead, this prototype intercepts at the **Zig level**, in `runtime.zig`'s `callFuncScalar` — the entry point the CLI (and this spike's own test) uses to invoke an exported function. `AotInstance` gained a comptime-gated (`config.lazy_jit`) `LazyJitState`:

```zig
pub const LazyJitState = struct {
    pending: []bool = &.{},               // local func idx -> still pending?
    compiled: []?LazyCompiledFunc = &.{},  // local func idx -> resolved code
    compile_ctx: ?*anyopaque = null,       // type-erased, see below
    compile_fn: ?*const fn (ctx: *anyopaque, local_idx: u32) ?LazyCompiledFunc = null,
};
```

`callFuncScalar` checks `pending[local_idx]` before resolving the call address; if true, it invokes `compile_fn` (which compiles the one function via the same real per-function codegen entry point the eager path uses, maps it executable via `platform.mapExecutableCode`, and returns the address+size), stores the result, clears `pending`, and proceeds with the call as normal. `AotInstance.destroy()` frees the tracking arrays and `munmap`s every compiled function's own mapping.

**Why type-erased?** `AotInstance`/`runtime.zig` must not depend on compiler types directly — the plain (non-`-Djit`) `wamr` binary links no compiler at all (#695's whole point). The actual "compile function N" logic — which needs `ir.IrFunction`, the per-arch codegen module, etc. — lives in `src/component/aot_compile.zig`'s new `LazyCompileDriver`, which owns the retained `LazyJitOut` and is wired into an `AotInstance` via the also-new `setupLazyJit(inst, lazy_out, allocator)` helper. This exactly mirrors the existing `TrampolinePool.ctx: *anyopaque` pattern.

**Scope of what this hook covers**: only calls that go through `callFuncScalar` — i.e., the host invoking an exported function directly. It does **not** cover `call_indirect` or a direct intra-module `.call`/`ref.func` reaching a lazy function (rules 2/3 above make sure that can't happen for anything currently marked eligible, so this is a consistent, if narrow, restriction — not a latent bug).

## Validation

`src/tests/lazy_jit_spike_test.zig` (gated on `config.lazy_jit`, native x86_64/aarch64 only):

1. **Correctness**: a 3-function fixture (`add1`, `mul2`, `never_called`, no calls between them, no tables) compiles with `opts.lazy_jit = true`; all 3 are confirmed lazy-eligible and the emitted `.cwasm`'s text section is genuinely empty (0 compiled bytes) up front. Calling `add1`/`mul2` through `callFuncScalar` triggers on-demand compilation and returns the correct result both on the first call (compiles) and a second call (reuses). `never_called` remains pending throughout — proving the deferral is real, not a same-work-different-order relabeling.
2. **Measured effect**: a 200-leaf-function synthetic module (`fN(x) = x + N`, no calls, no tables — all 200 eligible) compiled with `opts.lazy_jit = true` vs. the eager default. Measured (this repo's benchmark hardware, x86_64, ReleaseFast, 2026-07-10):

   ```
   eager (compile all 200): 359,923 µs
   lazy  (defer all 200):   289,020 µs
   ```

   ~20% reduction. Smaller than one might expect for skipping 100% of codegen — for these deliberately tiny (3-IR-instruction) functions, IR lowering/pass-running and `emit_aot`'s per-export table construction dominate total compile time more than actual instruction selection/regalloc does. A module with larger unused functions (more IR per function → proportionally more codegen work skipped, same fixed lowering/emit overhead) would show a bigger win; the CoreMark-based measurement in #860's PR description already established that codegen (not lowering) is where the bulk of *large*-function compile time goes.

Both the default and `-Djit=true` (no `-Dlazy_jit`) builds are verified byte-for-byte/behaviorally unaffected — every new field/branch is either an additive, empty-by-default struct field (`CompileOptions.lazy_skip`, `PrecompileOptions.lazy_jit`, `CompileCacheCtx.lazy_jit_out`) or comptime-gated to `void`/a no-op on `AotInstance` (`lazy_jit: if (config.lazy_jit) LazyJitState else void`).

## Deferred to follow-up work

This spike deliberately does not attempt (tracked as follow-up work, not filed as a separate issue yet — see #862's own acceptance criteria for "a follow-up tracking issue"):

- **Non-leaf functions.** Requires either indirect-call-only dispatch (give up the direct BL/`call rel32` patch for calls to any potentially-lazy function, replacing it with a load-from-`func_addrs`-and-call-indirect sequence — a codegen change with its own throughput cost even for already-compiled functions) or a two-phase layout (reserve space / use a trampoline for not-yet-known offsets, patch later) — either is real design + implementation work.
- **`call_indirect`/`ref.func` reachable functions.** Needs the native-trampoline approach sketched above (extending `TrampolinePool`/`host_trampolines.zig`'s dispatch machinery to plain `AotInstance`, or a dedicated lazy-compile trampoline kind), so `func_addrs[i]` can safely be "not yet compiled" for anything the runtime might jump to directly.
- **Components.** `precompileComponentInMemory` compiles each core module independently already, so the per-core mechanism here composes in principle, but the `LazyJitOut`/`LazyCompileDriver` plumbing only threads through the single-core `compileCoreWasmCached` path today.
- **Interaction with the JIT code cache (#857).** `JitCodeCache` currently tracks resident bytes at `mapCodeExecutable` time; a lazily-compiled function's separate `platform.mapExecutableCode` call in `LazyCompileDriver.compileFn` does **not** register with `JitCodeCache` in this spike, so the budget/tracking in #857 currently undercounts lazy-compiled code. A real implementation should route through `JitCodeCache.register`/`unregister` too.
- **Skipping IR-optimization passes for lazy functions**, not just codegen (see the note above).
- **Tiering / background re-compilation to a higher optimization level** (this spike's lazily-compiled functions always use the real regalloc-based codegen at whatever preset was requested — there's no "quick baseline now, optimize later in the background" tier).
- **Thread-safety of concurrent first-calls to the *same* lazy function.** `LazyJitState.compile_fn` is called at most once per index in the tests here, but nothing prevents two threads racing to compile the *same* still-pending function concurrently if an embedder calls `callFuncScalar` on the same `AotInstance` from multiple threads — matching the general "configure before spawning concurrent work" contract from #859's thread-safety audit, but worth an explicit note since this is genuinely new mutable per-instance state, not a startup-only global.
