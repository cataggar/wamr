# Lazy JIT tiering spike: `.fast` first-call, background `.full` upgrade

Status: **DESIGN SPIKE ONLY**. This note proposes the narrow x86_64/core-wasm follow-up to the existing lazy-JIT prototype in [`docs/design/lazy-jit-spike.md`](./lazy-jit-spike.md). It does **not** add code by itself.

Tracking: [#893](https://github.com/cataggar/wamr/issues/893). Builds on [#862](https://github.com/cataggar/wamr/issues/862), [#861](https://github.com/cataggar/wamr/issues/861), [#860](https://github.com/cataggar/wamr/issues/860), and [#859](https://github.com/cataggar/wamr/issues/859).

## Goal

Keep the current lazy-JIT scope from #862 (x86_64 only, current lazy-eligible leaf functions only, `callFuncScalar` path only), but let a function:

1. stay deferred at module compile time,
2. compile **synchronously** to a cheap `.fast` baseline on first call,
3. optionally queue a **background** `.full` recompilation, and
4. publish the better entry point for later callers without blocking them.

This is still **not** OSR, deopt, `call_indirect`, `ref.func`, or general multi-tier JITing. The current #862 lazy-eligibility contract remains the guardrail.

## Constraints from the current code

- `src/component/aot_compile.zig:221-241` computes lazy-eligible functions before passes, but the retained `LazyJitOut` today only preserves the **post-pass IR** for the module chosen by one `PrecompileOptions.pass_preset`.
- `src/component/aot_compile.zig:620-703`'s `LazyCompileDriver.compileFn` can only do final x86_64 codegen + `platform.mapExecutableCode` from that retained IR.
- `src/compiler/ir/passes.zig:8972-9038` already has the only two presets we need: `.fast` and `.full`.
- `src/compiler/codegen/x86_64/compile.zig:1626-1815` already has both relevant codegen seams:
  - `CompileOptions.lazy_skip` in the module loop, and
  - `compileFunctionRAWithGlobalOffsetsPublic(...)` for one-function emission.
- `src/runtime/aot/runtime.zig:1468-1494,2546-2605` currently has only a boolean/pointer-style lazy state and a synchronous first-call hook. There is no tier state, queue, or publication protocol.
- `src/runtime/aot/runtime.zig:35-73,1828-1836,2007-2032`'s `JitCodeCache` still only accounts for the main module mapping; lazy per-function mappings bypass it today.

## Proposed per-function state machine

Add a per-function slot under `AotInstance.lazy_jit`:

| State | Meaning | Caller behavior | Worker behavior |
|---|---|---|---|
| `pending` | no code compiled yet | first caller claims baseline compile | none |
| `fast_compiling` | caller thread is building `.fast` code | same-function callers wait | none |
| `fast_ready` | baseline published | call baseline entry | may enqueue `.full` |
| `full_queued` | queued for worker | call baseline entry | worker may dequeue |
| `full_running` | worker is compiling `.full` | call baseline entry | compile off-thread |
| `full_ready` | optimized entry published | call optimized entry | none |
| `failed` / `full_failed` | baseline failed, or full upgrade failed | propagate baseline failure, or keep using baseline | none |

Minimal slot shape:

```zig
const LazyTierSlot = struct {
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    state: State = .pending,
    active_addr: std.atomic.Value(usize) = .init(0),
    baseline: ?LazyCompiledFunc = null,
    optimized: ?LazyCompiledFunc = null,
    retired: std.ArrayListUnmanaged(LazyCompiledFunc) = .{},
};
```

`active_addr` is the only field callers need on the hot path after publication; everything else stays behind the mutex.

## 1) How a function first compiled from `.fast` state later reaches `.full`

### Chosen design

Retain **two views** of each lazy function when the module is initially compiled with `.fast`:

1. a **pre-pass clone** of the lazy function (or minimal one-function IR seed), and
2. the current **post-`.fast` IR** used for the baseline compile.

That keeps first-call latency low while still preserving enough state to rerun the optimizer at `.full` later.

### Concrete flow

1. In `src/component/aot_compile.zig:221-241`, after `lazy_jit.findLazyEligibleLeaves(...)` identifies the current lazy set but **before** `passes.runPassesWithOptions(...)`, clone each lazy function's pre-pass IR into a new `LazyJitOut.full_seeds`.
2. Continue running the module's normal `.fast` preset through `passes.passesForPreset(opts.target_arch, .fast)` so eager functions and the retained lazy baseline IR stay on the current cold-start-optimized path.
3. Keep the current lazy skip in `src/compiler/codegen/x86_64/compile.zig:1626-1692`, so those lazy functions still emit zero bytes into the module text blob.
4. Extend `src/component/aot_compile.zig:620-703`'s `LazyCompileDriver` so it becomes tier-aware:
   - `compileBaseline(local_idx)` codegens from the retained post-`.fast` IR using the existing `compileFunctionRAWithGlobalOffsetsPublic(...)`.
   - `compileOptimized(local_idx)` clones the saved pre-pass seed into a scratch one-function `IrModule`, runs `passes.passesForPreset(.x86_64, .full)` on that scratch module, then calls the same x86_64 codegen entry point.

### Why this is the minimal fit

- It avoids reparsing/re-lowering the original wasm on the worker thread.
- It preserves the current #862 first-call behavior: the blocking path is still “codegen + map”, not “re-run `.fast` passes over the whole module”.
- It uses the existing preset split in `src/compiler/ir/passes.zig:8972-9038` instead of inventing a third optimization mode.

### Important rule

If the module was already compiled with `WAMR_JIT_FULL_OPT=1` / `pass_preset = .full`, **do not queue a tier-up job**. The function is already at the top tier.

## 2) What thread/queue does the background work

### Chosen design

Use **one worker thread per `AotInstance`**, started with raw `std.Thread.spawn`, plus a tiny mutex/condition-protected FIFO owned by `LazyJitState`.

That matches the repo's existing ad-hoc thread patterns (`runtime.zig` watch thread and `wasi_cli_adapter.zig` HTTP workers) without inventing a shared runtime-wide job system.

### Queue model

- `LazyJitState` owns:
  - `queue_mutex`
  - `queue_cond`
  - `queue: std.ArrayListUnmanaged(u32)` of local function indices
  - `worker_thread: ?std.Thread`
  - `shutting_down: bool`
- After a caller publishes baseline `.fast` code, it transitions the slot from `fast_ready` to `full_queued` exactly once and appends `local_idx` to the queue.
- The worker waits on the condition variable, dequeues a single `local_idx`, flips the slot to `full_running`, compiles `.full`, then publishes or records `full_failed`.

### Why per-instance, not global

- It stays inside the current #859 “caller-owned state” model.
- `AotInstance.destroy()` can quiesce the worker by setting `shutting_down = true`, signalling the condition variable, and `join()`ing the one known thread before any lazy mappings or retained IR are freed.
- It avoids a repo-wide worker pool for a feature that is still intentionally narrow.

### Same-function race contract

- `pending -> fast_compiling` is claimed under the slot mutex.
- Any second caller that arrives while the same function is in `fast_compiling` waits on the slot condition instead of compiling the same function twice.
- Callers arriving during `full_queued` / `full_running` do **not** wait; they keep using the published baseline entry.

That is the minimum synchronization worth adding beyond #862's current “single-threaded per-instance” caveat.

### Process-global config contract

This design does **not** relax #859: process-global knobs like `WAMR_JIT_FULL_OPT`, debug flags, and code-budget settings still have to be configured before concurrent work begins. Tiering only adds per-instance worker state, not a new global compiler pool.

## 3) How the optimized entry point is published without racing callers

### Chosen design

Publish via an **atomic pointer swap** on a per-function `active_addr`, with all code generation and mapping completed before the swap.

### Publication sequence

1. Worker compiles `.full` into a fresh private buffer.
2. Worker maps that buffer executable with `platform.mapExecutableCode(...)`.
3. Worker registers the new mapping in `JitCodeCache` **before** publication.
4. Worker acquires the slot mutex, confirms the slot is still `full_running` and the instance is not shutting down.
5. Worker stores `optimized = mapped_full`.
6. Worker swaps `old_addr = active_addr.swap(new_addr, .acq_rel)` (or `store(..., .release)` if the old pointer is already saved in `baseline`).
7. Worker moves the old mapping record to `retired` and marks the slot `full_ready`.
8. Later callers load `active_addr` with `.acquire` and jump to whichever fully-initialized entry was visible at that instant.

### Why this is race-safe for the current scope

- Lazy-eligible functions are still the #862 leaf-only set, so there are no intra-module direct-call patch sites to rewrite.
- Code pages are immutable after mapping. The only mutable shared datum is the entry pointer.
- A caller therefore sees either:
  - the old `.fast` mapping, or
  - the new `.full` mapping,
  but never partially generated code.

### `callFuncScalar` rule

`src/runtime/aot/runtime.zig:2546-2605` should change from “compile if pending, else check `compiled[local_idx]`” to:

1. fast-path load of `slot.active_addr`,
2. if zero, take the slot mutex and drive the `pending` / `fast_compiling` logic,
3. after unlock, call the loaded address.

That keeps the common post-publication path down to one atomic load.

## 4) How superseded mappings are reclaimed and accounted for in `JitCodeCache`

### Chosen design

**Do not unmap the superseded baseline immediately.** Retire it and reclaim it at `AotInstance.destroy()`.

### Why immediate free is the wrong first step

The pointer swap alone does not prove no thread is still executing inside the old baseline code. Without OSR, safepoints, or per-call reference counting, immediate `munmap` would risk tearing code out from under an in-flight caller.

### Minimal reclaim policy

- Every lazy `.fast` mapping registers with `JitCodeCache` when mapped.
- Every lazy `.full` mapping registers with `JitCodeCache` when mapped.
- When `.full` supersedes `.fast`, the `.fast` mapping moves to `slot.retired` but remains mapped.
- `AotInstance.destroy()`:
  1. stops and joins the per-instance worker,
  2. unmaps `baseline`, `optimized`, and every `retired` mapping,
  3. unregisters each exact byte size from `JitCodeCache`,
  4. then frees retained IR / queue state.

This makes `JitCodeCache.residentBytes()` reflect **actual** mapped code, not just “currently active entry points”.

### Required `JitCodeCache` follow-up

Because tiering introduces background compilation, `src/runtime/aot/runtime.zig:35-73`'s current plain `usize` counters and unsynchronized budget check are no longer sufficient. The follow-up implementation should make `checkBudget/register/unregister` thread-safe, either by:

- a small mutex around budget reservation + counter updates, or
- atomics plus a compare/exchange reservation loop.

The simplest first implementation is a mutex.

### Future optimization (explicitly deferred)

If code-size pressure becomes material, a later phase can add per-slot in-flight call counts and free retired mappings once the old generation's active-call count reaches zero. That is **not** needed for the design spike.

## Benchmark plan

Use the existing #861 numbers as the acceptance envelope:

- current in-process JIT `.fast`: **1.430 ms** cold-start median, **12,002.8** CoreMark iter/s
- current in-process JIT `.full`: **1.463 ms** cold-start median, **12,536.5** CoreMark iter/s

from [`docs/bench/jit-cold-start-comparison-2026-07-10.md`](../bench/jit-cold-start-comparison-2026-07-10.md), plus the current lazy-leaf compile win of **359,923 µs → 289,020 µs** from [`docs/design/lazy-jit-spike.md`](./lazy-jit-spike.md).

### Proposed measurements

1. **Cold-start guardrail (`noop`, existing #861 harness)**
   - Add a tiering-on row to the existing noop subprocess benchmark.
   - Expectation: tiering plumbing stays close to the current `.fast` median (**1.430 ms**), because noop has effectively no worthwhile background work.

2. **Throughput recovery (`CoreMark`, existing #861 harness)**
   - Add a tiering-on row that starts from default `.fast`.
   - Measure two points:
     - first-run / pre-upgrade behavior (should remain close to `.fast`), and
     - post-queue-drain behavior (target should move toward `.full`'s **12,536.5** iter/s instead of staying at **12,002.8**).
   - The harness needs one deterministic “queue drained” signal for this row (debug log, counter, or bench-only env knob).

3. **Mechanism check for the current #862 leaf-only scope**
   - Re-run the existing 200-leaf synthetic from `docs/design/lazy-jit-spike.md`.
   - This is the workload that actually exercises today's lazy-eligible contract, so it remains the best place to confirm:
     - cold-start still benefits from deferring codegen,
     - first call pays only baseline compilation, and
     - the background worker later upgrades entries without blocking callers.

### Honest limitation

With the current #862 eligibility rules, CoreMark is mainly a **ceiling/reference** benchmark for the eventual throughput target; the 200-leaf synthetic is still the direct mechanism test for the narrow spike.

## Exact follow-up code touch points

### `src/component/aot_compile.zig`

- `PrecompileOptions` / `CompileCacheCtx` / `LazyJitOut`
- `compileCoreWasmCached(...)` around the current lazy-skip setup (`221-241`) so it preserves both:
  - pre-pass seeds for future `.full`, and
  - post-`.fast` IR for baseline codegen
- `LazyCompileDriver` / `setupLazyJit(...)` (`620-703`) so compile callbacks become tier-aware and can enqueue background work

### `src/runtime/aot/runtime.zig`

- `JitCodeCache` (`35-73`) for thread-safe budget/accounting of lazy mappings
- `LazyJitState` (`1468-1494`) to replace the current `pending[]`/`compiled[]` shape with per-slot tier state, queue state, and worker handle
- `AotInstance.destroy()` (`1828-1836`) to stop/join the worker before freeing mappings and retained compiler state
- `callFuncScalar(...)` (`2546-2605`) for:
  - baseline-compile claiming/waiting,
  - queueing `.full`,
  - atomic entry loads

### `src/compiler/ir/passes.zig`

- `PassPreset` / `passesForPreset(...)` (`8972-9038`)
- add the small helper that runs an existing preset on a cloned single-function scratch module, rather than inventing a new preset

### `src/compiler/codegen/x86_64/compile.zig`

- `CompileOptions.lazy_skip` and the `compileModuleCachedWithOptions(...)` loop (`1626-1815`)
- keep the existing eager-module skip behavior
- reuse `compileFunctionRAWithGlobalOffsetsPublic(...)` for both baseline and optimized one-function emission

### Additional likely touch points

- `src/main.zig` for an explicit tiering env/flag (separate from `WAMR_JIT_FULL_OPT`, which should keep meaning “compile everything eagerly at `.full`”)
- `src/tests/lazy_jit_spike_test.zig` plus a new tiering-focused test file for:
  - same-function concurrent first call,
  - background publish while callers continue succeeding,
  - destroy while work is queued/running,
  - `JitCodeCache` accounting of baseline + upgraded + retired mappings
- `scripts/bench_jit_coldstart.py` (or equivalent) for the new tiering benchmark rows

## Recommendation

Implement this as a **narrow follow-up on top of #862**, not as a general JIT framework:

- keep the current lazy-eligible leaf set,
- compile baseline `.fast` on the caller thread,
- use one per-instance background worker,
- publish with an atomic entry pointer,
- reclaim retired mappings only at destroy,
- and make `JitCodeCache` count every mapped tier.

That is the smallest design that answers #893's four acceptance questions without drifting into OSR, trampolines, or a full scheduler.
