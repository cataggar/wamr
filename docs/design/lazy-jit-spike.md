# Lazy JIT design spike: per-function on-demand compilation

Status: **x86_64/aarch64 PROTOTYPE** (non-leaf direct-call graphs via stable entry stubs on x86_64 (#887); `call_indirect` / `ref.func` / `call_ref` for trampoline-compatible leaf signatures via native trampolines, x86_64 only (#888); leaf-only zero-text deferral on aarch64 (#890); see [Scope](#scope) and [Deferred to follow-up work](#deferred-to-follow-up-work)).

Tracking: [#862](https://github.com/cataggar/wamr/issues/862) (Milestone 4 of the in-process JIT plan, [#863](https://github.com/cataggar/wamr/issues/863)). Depends on the core-wasm in-process JIT ([#853](https://github.com/cataggar/wamr/issues/853)) and the thread-safety audit ([#859](https://github.com/cataggar/wamr/issues/859)).

## Motivation

The in-process JIT (#852-#861) compiles a module's **entire** function set up front, synchronously, before the first export call returns. For a large module where most functions are never actually invoked in a given run (a common shape — CLI tools, WASI components with big generated bindings, etc.), that's wasted compile latency squarely on the user-visible cold-start path this whole JIT effort exists to shrink. wasmtime/V8-class JITs solve this with **lazy, per-function compilation**: do the real work only on first entry, retaining a stable entrypoint only where a local call or function reference needs one before then.

This spike wires up the existing (until now, unused) `config.lazy_jit` build flag stub (`src/config.zig:103`) with a real, narrowly-scoped prototype, to de-risk the approach before committing to a full implementation.

## What already exists that makes this tractable

- Both native backends are organized per function: `pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8` exists independently in `src/compiler/codegen/x86_64/compile.zig` and `src/compiler/codegen/aarch64/compile.zig`. The primitive to "compile just function N" already exists.
- `platform.mapExecutableCode` (#858) already handles the W^X-safe mmap→copy→icache-flush→mprotect sequence for a single blob of native code — exactly what a lazily-compiled function's code needs.
- `mapCodeExecutable` (`runtime.zig`) already builds the `func_addrs[]`/`funcptrs[]` table used to resolve a function's native entry point — the natural patch point for "not compiled yet."

## What actually made this hard

### Direct calls are relative-branch patches computed at whole-module compile time

Both backends' `compileModuleWithOptions`/`compileModuleCachedWithOptions` compile every function into one contiguous code blob, then resolve every direct `.call` as a `BL`(aarch64)/`call rel32`(x86_64) whose target offset is only known **after all functions have been compiled and laid out**. A function whose compilation is deferred has no offset yet — any *other* function's `.call` to it can't be patched. This is the crux of why "lazy" can't simply mean "don't call `compileFunction` yet" for arbitrary functions.

### `call_indirect` / `ref.func` need a stable native pointer at load time

`runtime.zig`'s `ptr_to_sig` sorted map, table initialization, and the x86_64 `call_indirect` / `call_ref` / `ref.func` codegen all assume `func_addrs[i]` is a real, callable native address for every `i` by the time `mapCodeExecutable` returns. Issue #888 implements that with a per-instance stable trampoline pointer for each deferred local: `mapCodeExecutable` publishes the trampoline in `funcptrs` / tables up front, and the trampoline compiles the target on first use while keeping its own address stable for the instance lifetime.

## The two dispatch mechanisms

Given the two problems above, this prototype supports a zero-text root path plus two stable-entry mechanisms, selected by `src/compiler/lazy_jit.zig`'s `findLazyEligibleFunctions`:

1. **Stub mechanism (#887)**: a lazy function that is a direct intra-module `.call`/tail-call target receives a small, always-resident entry stub at its normal text-section offset. Patches resolve against that stub exactly as if the function had been compiled eagerly; it compiles the real body on first entry, then forwards through it. A direct-call-graph root with no incoming local call emits no text and resolves through `callFuncScalar` before first entry. Non-leaf functions are fine here because their lazy bodies lower local direct calls through `vmctx.funcptrs_ptr`, never relying on a separately mapped body's rel32 patch.
2. **Trampoline mechanism (#888)**: for LEAF functions that are never a direct `.call` target but ARE reachable through a table, `ref.func`, or `call_indirect`/`call_ref`, and whose lowered ABI fits the trampoline envelope (x86_64 only, no retptr, no `v128`, at most one scalar result, at most 9 user args after `vmctx`) — a per-instance native trampoline (`host_trampolines.TrampolinePool`) is allocated up front and its stable address is published into `func_addrs`/`funcptrs`/tables wherever the runtime normally mirrors callable pointers. The trampoline compiles the target on first use while keeping its own address stable for the instance lifetime.

A function that itself contains `.call_indirect` or `.call_ref` is never lazy-eligible under either mechanism.

See `src/compiler/lazy_jit.zig`'s `findLazyEligibleFunctions` for the exact analysis (with unit tests covering both mechanisms).

### Compile-time: skip per-function IR work and codegen

`compileModuleCachedWithOptions` (on both native backends now — x86_64 and aarch64) gained an additive `CompileOptions.lazy_skip: []const bool` field (indexed by local function index, empty by default — zero behavior change for every existing caller). It emits no body for each skipped function; x86_64's `lazy_entry_stubs` subset emits a stable stub only for a direct-call target. Trampoline-mechanism functions (x86_64 only) publish their trampoline instead, and aarch64 only admits leaf functions that are never direct-call targets.

Issue #892 threads the **same** local-index set into `passes.RunOptions.lazy_skip`, so eager optimization neither inlines a lazy caller/callee pair nor runs that function's per-function pipeline (`promoteLocalsToSSA`, `lowerPhisToLocals`, the preset-selected fixpoint loop, and final `scrubUnreachableBlocks`). `LazyJitOut` retains the chosen pass preset plus a small `RunOptions` snapshot, and `LazyCompileDriver.compileFn` replays that exact per-function pipeline immediately before x86_64 codegen on first call.

`PrecompileOptions` gained a `lazy_jit: bool` opt-in flag and `CompileCacheCtx` gained a `lazy_jit_out: ?*LazyJitOut`. When both are set, `compileCoreWasmCached` computes the eligible set, threads it into both the eager pass manager and the selected backend's codegen call, and — instead of freeing the lowered IR module at the end like it normally does — **moves it** into the caller-supplied `LazyJitOut` (along with the compact list of deferred local indices, parallel `needs_trampoline` flags, the target backend so `LazyCompileDriver` dispatches first-call compilation to the matching codegen, and the deferred-pass metadata), so the deferred functions' IR survives past the call for later on-demand compilation. For benchmarks/tests that still need the historical "#862 codegen-only lazy" baseline, `PrecompileOptions.lazy_defer_passes = false` preserves the old behavior.

### Runtime: per-instance native trampolines for plain `AotInstance`

Issue #888 extends `host_trampolines.zig`'s existing `TrampolinePool` / `genericDispatcher` machinery to plain `AotInstance`. `setupLazyJit()` now runs before `mapCodeExecutable()`, allocates one stable trampoline stub per deferred local, stores those pointers in `LazyJitState.trampolines`, and lets `mapCodeExecutable()` publish them everywhere the runtime already mirrors callable pointers (`funcptrs`, `ptr_to_sig`, active element initialization, multi-table `tables_info`). `runtime.zig` adds a new `wamrAotDispatchLazyLocalAot` arm that compile-on-first-call's the target local and then forwards through the resolved native code while keeping the trampoline pointer stable.

`AotInstance` still keeps comptime-gated (`config.lazy_jit`) `LazyJitState`:

```zig
pub const LazyJitState = struct {
    pub const SlotState = enum(u8) { inactive, pending, compiling, ready };
    slot_states: []std.atomic.Value(u8) = &.{}, // local func idx -> per-slot lazy state
    compiled: []?LazyCompiledFunc = &.{},       // local func idx -> resolved code
    compile_ctx: ?*anyopaque = null,            // type-erased, see below
    compile_fn: ?*const fn (ctx: *anyopaque, local_idx: u32) RuntimeError!LazyCompiledFunc = null,
};
```

`callFuncScalar` (and the #888 trampoline dispatcher) now resolve lazy locals through a **per-slot atomic state machine**: lazy-eligible slots start `pending`, one winning thread CASes a contended slot to `compiling`, publishes `compiled[local_idx]`, then release-stores `ready`; waiters acquire-load that `ready` state before reading the published `LazyCompiledFunc`, so same-slot races never observe torn/half-written code pointers. If `compile_fn` returns an error (including `error.CodeBudgetExceeded` from the tracked mapping path, propagated as-is so callers can distinguish "budget exceeded, retryable" from a hard mapping failure), the winning thread release-stores `pending` again before re-raising that same error, so the slot stays retryable and waiters never wedge forever (they may immediately compete to retry in the same burst). `AotInstance.destroy()` still frees the tracking arrays, the trampoline pool, and `munmap`s every compiled function's own mapping (and unregisters it from `JitCodeCache`).

**Why type-erased?** `AotInstance`/`runtime.zig` must not depend on compiler types directly — the plain (non-`-Djit`) `wamr` binary links no compiler at all (#695's whole point). The actual "compile function N" logic — which needs `ir.IrFunction`, the per-arch codegen module, etc. — lives in `src/component/aot_compile.zig`'s new `LazyCompileDriver`, which owns the retained `LazyJitOut` and is wired into an `AotInstance` via the also-new `setupLazyJit(inst, lazy_out, allocator)` helper. This exactly mirrors the existing `TrampolinePool.ctx: *anyopaque` pattern.

**Scope of what this hook covers**: exported direct calls (`callFunc` / `callFuncScalar`), direct intra-module `.call`s to non-leaf lazy functions (#887, via a text-section entry stub), and x86_64 `call_indirect` / `ref.func` / `call_ref` that consume a published trampoline pointer (#888).

## Validation

`src/tests/lazy_jit_spike_test.zig` (gated on `config.lazy_jit`, native x86_64/aarch64 only):

1. **Correctness**: a 3-function fixture (`add1`, `mul2`, `never_called`, no calls between them, no tables) compiles with `opts.lazy_jit = true`; all 3 are confirmed lazy-eligible and the emitted `.cwasm`'s text section is genuinely empty (0 compiled bytes) up front. Calling `add1`/`mul2` through `callFuncScalar` triggers on-demand compilation and returns the correct result both on the first call (compiles) and a second call (reuses). `never_called` remains pending throughout — proving the deferral is real, not a same-work-different-order relabeling.
2. **Same-slot contention**: an 8-thread stress test shares one lazily prepared `AotInstance`, forces all 8 threads to first-call the SAME exported lazy leaf together, and wraps `compile_fn` with an atomic counter. Exactly one compile entry is observed for the contended slot; every thread gets the correct result; unrelated lazy locals remain pending until explicitly used.
3. **Failure/retry path**: the same 8-thread setup wraps `compile_fn` so the first winning compile attempt returns `null`. One caller gets `CodeMappingFailed`, another waiter wins the reset-to-`pending` retry, the slot reaches `ready`, and later calls reuse the successfully published code.
4. **Non-leaf direct-call graphs (#887)**: a fixture mixing eager↔lazy and lazy↔lazy direct callers/callees (including a tail call) proves stable entry stubs let direct `.call`/tail-call patches resolve against still-pending lazy targets, with each nested first-call compiling exactly once.
5. **Table / funcref reachability (#888)**: x86_64 lazy-JIT tests cover an active-element `call_indirect` path and a `ref.func + table.set + call_indirect` path, both compiling a leaf target through its stable trampoline on first use while leaving `inst.funcptrs[target]` unchanged before and after compilation.
6. **JitCodeCache integration (#891)**: deferred mappings are confirmed to register with (and unregister from) `JitCodeCache`'s resident-bytes/mapping-count accounting, and a low `budget_bytes` correctly rejects a deferred first-call compile with `error.CodeBudgetExceeded` while leaving the slot retryable.
7. **Deferred IR-pass pipeline (#892)**: a fixture adds a branchy/local-heavy leaf function (`phi_local`) alongside `add1`, `mul2`, and `never_called`. A dedicated test proves the eager `runPassesWithOptions` call genuinely skips `promoteLocalsToSSA`/`lowerPhisToLocals`/the preset fixpoint loop for lazy-marked functions (via a pass-dump probe), and that `runFunctionPassesWithOptions` correctly replays that exact pipeline for one retained function on demand. A second test confirms `phi_local` compiles correctly on first call (replaying the deferred passes then running x86_64 codegen) and executes both branches correctly, with the same pending/ready deferral proof as item 1.
8. **Measured effect**: a 200-leaf-function synthetic module (`fN(x) = x + N`, no calls, no tables — all 200 eligible) compiled with `opts.lazy_jit = true` vs. the eager default, and — since #892 — a three-way comparison also against the historical codegen-only lazy baseline (`lazy_defer_passes = false`). Measured (this repo's benchmark hardware, x86_64, ReleaseFast, 2026-07-10):

   ```
   lazy_skip_passes_ns <= lazy_codegen_only_ns <= eager_ns
   ```

   This captures the original #862 ~20% codegen-only win while ensuring #892's additional pass deferral moves the 200-leaf case further in the right direction.

Both the default and `-Djit=true` (no `-Dlazy_jit`) builds are verified byte-for-byte/behaviorally unaffected — every new field/branch is either an additive, empty-by-default struct field (`CompileOptions.lazy_skip`, `PrecompileOptions.lazy_jit`, `CompileCacheCtx.lazy_jit_out`) or comptime-gated to `void`/a no-op on `AotInstance` (`lazy_jit: if (config.lazy_jit) LazyJitState else void`).

## Deferred to follow-up work

This spike deliberately does not attempt (tracked as follow-up work, not filed as a separate issue yet — see #862's own acceptance criteria for "a follow-up tracking issue"):

- **Non-scalar / wide trampoline signatures.** The plain-AOT trampoline arm still only admits the existing `genericDispatcher` envelope (x86_64 only, no retptr, no `v128`, max 9 user args, at most one scalar result).
- **Non-leaf functions and the trampoline mechanism on aarch64.** #890 ports leaf-only zero-text deferral to the aarch64 backend, but #887's non-leaf `via_funcptrs` local-call lowering and #888's native trampolines remain x86_64-only; `findLazyEligibleFunctions` keeps aarch64 restricted to leaf, never-directly-called functions accordingly.
- **Components with persisted manifests.** The in-memory component path now carries one lazy sidecar per compiled core and attaches it at instantiation time (#889), including nested sub-components matched by `core_wasm` slice identity. The on-disk manifest / `wamrc compile-component` path still remains eager; no lazy sidecar metadata is persisted in the manifest format.
- **Interaction with the JIT code cache (#857).** Resolved later by #891: deferred function mappings now go through the same runtime-owned tracked mapping helper as eager text blobs, so `residentBytes()`, `mappingCount()`, and `WAMR_JIT_CODE_BUDGET_BYTES` all account for lazy first-call compilation too.
- **Tiering / background re-compilation to a higher optimization level** (this spike's lazily-compiled functions always use the real regalloc-based codegen at whatever preset was requested — there's no "quick baseline now, optimize later in the background" tier).
