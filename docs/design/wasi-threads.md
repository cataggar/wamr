# `wasi:threads` design — multi-threaded interpreter state isolation

Status: **DRAFT** (shared-memory/parking foundation implemented; opcode
atomicity and production spawning remain incomplete).

Tracking: [#616 B1.3](https://github.com/cataggar/wamr/issues/616).

Author wave: W10-3.

This document captures the design exploration the wave-7–9 PRs deferred:
how `wamr` should support guest threads given that the current
interpreter is single-threaded by construction. It is intentionally
specific enough that a future implementation wave can execute the plan
without re-doing the upstream survey or the state inventory.

## Table of contents

- [Overview](#overview)
- [Current implementation status](#current-implementation-status)
- [Upstream state](#upstream-state)
- [Inventory of single-threaded assumptions](#inventory-of-single-threaded-assumptions)
- [Design options](#design-options)
- [Recommendation](#recommendation)
- [Implementation plan (multi-wave)](#implementation-plan-multi-wave)
- [Open questions](#open-questions)
- [Acceptance criteria for the design to land](#acceptance-criteria-for-the-design-to-land)
- [Out of scope](#out-of-scope)
- [See also](#see-also)

## Overview

**Goal.** Support guest-driven thread spawning under `wamr` — initially
the existing `wasi:threads@0.1.0` (Preview-1) prototype that
[`src/wasi/thread_manager.zig`](../../src/wasi/thread_manager.zig)
sketches, ultimately a Component-Model-native binding once upstream
stabilises one. Each WASI thread runs on a real `std.Thread`; guest
atomics (`*.atomic.*`, `memory.atomic.wait`, `memory.atomic.notify`)
operate on the shared linear memory and synchronise across threads.

**Constraints.**

1. **No single-threaded performance regression.** The single-threaded
   path — every `zig build wasi-testsuite` / `wasi-p3-testsuite` /
   `wasi-p2-testsuite` fixture today — must stay on the existing fast
   path. Any mutex or atomic added for thread safety has to be a no-op
   or near-no-op when `lib_wasi_threads = false`.
2. **No data races.** Resource-table mutation, `TaskManager` state,
   adapter-wide singletons (allow-list templates, deinit-managed
   allocations, pending async-fetch lists) must be safe under
   concurrent guest invocations.
3. **No new dependencies.** Zig 0.16 standard library + `std.Thread`
   only. The existing `thread_manager.zig` shows the pattern (custom
   spinlock `Mutex` because Zig 0.16 moved `std.Thread.Mutex` behind
   `Io`).
4. **`zig build test` stays green.**

`-Dlib_wasi_threads=true` is currently a configuration contract, not a
production-support switch. It implies shared memory, the thread manager,
WebAssembly atomics, and heap auxiliary stacks, requires a 64-bit native
multithreaded interpreter build, and rejects AOT/JIT configurations. The
public `config.wasi_threads` report keeps target, backend, configured
capabilities, and implementation readiness separate; thread-spawn imports
are rejected before allocation while readiness remains false.

## Current implementation status

The tree contains scaffolding in `ThreadManager`, `cloneForThread`,
the atomic-opcode dispatcher, and the `wasi.thread-spawn` host import.
These are prototypes, not production support. No default CLI or
component path currently offers gated guest-thread execution.

Shared memories now use a refcounted control block with an immutable base.
Instantiation reserves the declared maximum up front and fails if that
reservation cannot be created; it never falls back to relocating `realloc`.
Grow commits in place under one lock, preserves zero fill, and publishes
current bytes/pages with release stores consumed by acquire accessors.

`src/platform/parking_lot.zig` provides keyed wait32/wait64, exact notify,
monotonic deadlines, address/all cancellation, and draining shutdown. The
value comparison and waiter insertion share one bucket lock. Its native wait
word uses Linux futex, macOS ulock, or Windows WaitOnAddress; Windows linear
memory uses NT reserve/commit. These APIs are intentionally independent of
opcode lowering so the remaining atomic load/store/RMW audit can use them.

The remaining production work follows the ordered plan below: make all
shared runtime and host resources safe; complete atomic opcode and fence
behavior; spawn in both interpreter and AOT modes; isolate cancellation and
per-thread component/WASI context; and pass the end-to-end correctness,
conformance, and performance gates.

## Upstream state

There are three relevant upstream proposals; the task spec refers to
the (non-existent) "`wasi:threads@0.3.x`" by analogy with the WASI 0.3
versioning scheme used elsewhere in the matrix. The real landscape is:

| Proposal                                                                                            | Phase                                          | Status                                                                                                                          |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [`wasi-threads`](https://github.com/WebAssembly/wasi-threads) (Preview-1 era)                       | Phase 1, explicitly **legacy** as of late 2024 | Single core-wasm import `wasi.thread-spawn` + guest export `wasi_thread_start`. Champion: Alexandru Ene. Wasmtime supports it.  |
| [`shared-everything-threads`](https://github.com/WebAssembly/shared-everything-threads)             | Phase 1, **under active development**          | Adds `shared` annotations across tables / globals / functions, thread-local globals, atomic GC, **Component-Model built-ins** for thread spawning. |
| Core [`threads`](https://github.com/WebAssembly/threads) proposal                                   | Phase 4 (standard)                             | Shared memories + atomic loads/stores/RMW + `wait`/`notify`. Already partly implemented in `src/runtime/interpreter/interp.zig` (`atomic_prefix`, opcodes `0x10`–`0x4F`). |

**Key fact.** Upstream `wasi-threads/README.md` (commit at the time of
writing) explicitly says:

> this proposal is considered a legacy proposal, retained for engines
> that can only support WASI v0.1 (`preview1`). After much debate,
> future work on threads will happen in the
> [shared-everything-threads](https://github.com/WebAssembly/shared-everything-threads)
> proposal which adds component model built-ins for thread spawning…

So **there is no `wasi:threads@0.3.x` interface** in the Component
Model sense. The Component-Model successor is a set of *canon
built-ins* (`thread.spawn`, …) defined by the
`shared-everything-threads` proposal, not a `wasi:threads/<iface>`
named interface like `wasi:keyvalue/store@0.2.0-draft2`.

**Wasmtime status.**

* Wasmtime ships the legacy [`wasi-threads` crate](https://github.com/bytecodealliance/wasmtime/tree/main/crates/wasi-threads)
  behind `--wasi-threads`. Its `README.md` warns: *"this crate is
  experimental and not yet suitable for use in multi-tenant
  embeddings. As specified, a trap or WASI exit in one thread must end
  execution for all threads. Due to the complexity of stopping
  threads, however, this implementation currently exits the process
  entirely."*
* For Component-Model threads, Wasmtime is tracking
  `shared-everything-threads` but has no stable user-facing surface
  yet (as of November 2025).
* Wasmtime does **not** ship a `wasi:threads/spawn@0.3.x` host
  adapter, because that interface does not exist.

**Conclusion.** "Pin to an upstream Wasmtime version" — the recipe
that worked for `wasi:keyvalue` ([commit `fb6e23d`](https://github.com/WebAssembly/wasi-keyvalue/tree/fb6e23d11d41d0704b41cdd6362536c5750e0329))
and `wasi:config` (`0.2.0-rc.1`) — does **not** work for threads
today. The design must instead aim at the legacy Preview-1 surface
(which is what real-world threaded `wasm32-wasi-threads` Rust /
`pthreads` C binaries actually link against) and treat
`shared-everything-threads` as a future-compat target.

## Inventory of single-threaded assumptions

The current runtime touches three layers of mutable state that a
threaded execution model has to either share-with-locking or
clone-per-thread.

### 1. `Interpreter` / `ExecEnv` ([`src/runtime/common/exec_env.zig`](../../src/runtime/common/exec_env.zig:76))

The execution environment is already documented as
*"per-thread interpreter state"*. Its canonical fields are:

| Field                              | Lifetime    | Threading classification                              |
| ---------------------------------- | ----------- | ----------------------------------------------------- |
| `module_inst: *ModuleInstance`     | Shared      | Borrowed pointer — the instance is shared across threads (see (2)). |
| `operand_stack: []Value`           | Per-thread  | Allocated in `ExecEnv.create`; never shared.          |
| `sp: u32`                          | Per-thread  | —                                                     |
| `call_stack: []CallFrame`          | Per-thread  | —                                                     |
| `call_depth: u32`                  | Per-thread  | —                                                     |
| `exception`, `pending_exception_*` | Per-thread  | —                                                     |
| `exception_refs[8]`, `…_count`     | Per-thread  | —                                                     |
| `allocator: std.mem.Allocator`     | Per-thread  | Passed in; embedder owns thread-safety.               |
| `thread_manager: ?*ThreadManager`  | Shared      | Already a borrowed pointer (today only non-null when `lib_wasi_threads = true`). |
| `tid: i32`                         | Per-thread  | Already populated by `thread_manager.zig:230`.        |
| `host_trap: ?HostTrapInfo`         | Per-thread  | Trap diagnostic, not racy by construction.            |
| `wasi_ctx: ?*anyopaque`            | Shared      | Opaque pointer to embedder's WASI context — see (3).  |

The `ExecEnv` struct itself is already thread-local: one
`ExecEnv.create` call per spawned thread, no shared mutable fields.
`dispatchLoopWithFuel` takes the `ExecEnv` by pointer and does **not**
share any per-call state across `env` instances — every mutation lands
in `env.operand_stack`, `env.call_stack`, or via `env.module_inst.*`
(which is what makes (2) and (3) the hard parts of the problem).

The big locked-in choice in the interpreter is the **single-threaded
fast path for global access**:
[`interp.zig:1994`](../../src/runtime/interpreter/interp.zig)

```zig
.global_get => { … env.module_inst.globals[idx].value … },
.global_set => { … env.module_inst.globals[idx].value = try env.pop(); … },
```

Globals are bare loads/stores. If we share `globals` across threads,
we either (a) clone globals per thread (Preview-1's
`cloneForThread` already does this — `types.zig:835`), or (b) add
atomic accessors for `shared` globals once
`shared-everything-threads` lands.

### 2. `ModuleInstance` ([`src/runtime/common/types.zig`](../../src/runtime/common/types.zig:791))

| Field                  | Today's role                                                | Threading classification               |
| ---------------------- | ----------------------------------------------------------- | -------------------------------------- |
| `module`               | Borrowed pointer to immutable parsed `WasmModule`.          | Shared (immutable).                    |
| `memories: []*MemoryInstance` | `MemoryInstance` is ref-counted (`retain`/`release`).  | **Shared across threads** — guests rely on it. |
| `tables: []*TableInstance`    | Ref-counted.                                          | **Shared across threads.**             |
| `globals: []*GlobalInstance`  | Mutable global state.                                 | **Cloned per thread** (Preview-1 model). |
| `import_functions`, `host_functions`, `host_func_entries` | Resolved during `linkImports`. | Shared (immutable after link).         |
| `tags`                 | Exception tags.                                             | Shared (immutable after instantiation). |
| `thread_manager`       | Borrowed pointer to the ThreadManager.                      | Shared.                                |
| `dropped_elems`, `dropped_data`, `cached_elem_values` | Active-segment drop state.    | **Currently per-instance, not protected.** |
| `gc_objects`           | GC heap (struct/array refs).                                | **Currently per-instance, not protected.** |

`cloneForThread` ([`types.zig:835`](../../src/runtime/common/types.zig))
already encodes the Preview-1 "shared memories+tables, cloned globals"
model. The unprotected fields above (`dropped_elems`, `dropped_data`,
`gc_objects`, `cached_elem_values`) are the bug-farm that
the implementation wave will have to clean up — Preview-1's
`destroyClonedInstance` (`thread_manager.zig:244`) duplicates
`dropped_elems` per thread but `gc_objects` is dangerous: two
threads doing `struct.new` simultaneously will corrupt the array.

### 3. `WasiCliAdapter` + `ComponentInstance`

The Component-Model resource tables live on `ComponentInstance`
([`src/component/instance.zig:196`](../../src/component/instance.zig))
and the Preview-2/3 host adapter ([`src/component/wasi_cli_adapter.zig:3779`](../../src/component/wasi_cli_adapter.zig)).

**Per-`ComponentInstance` (`std.AutoHashMapUnmanaged` / `ArrayList…` —
all single-threaded today):**

* `resource_tables` — keyed by component resource type idx.
* `exported_funcs`, `imports` — built once at link time.
* `trampoline_ctxs`, `canon_builtin_ctxs`, `canon_builtin_ctx_by_canon_idx`,
  `forwarding_ctxs`, `synthetic_host_instances`.
* `pending_core_starts`, `sub_instances`.
* `implicit_task_context[N_CONTEXT_SLOTS]`.
* `futures`, `streams`, `error_contexts`, `waitable_sets`,
  `next_async_handle` — the Preview-3 async handle tables.
* `current_task_manager: ?*async_mod.TaskManager`.

**Per-`WasiCliAdapter` (resource tables — every `*_table` field):**

Counted by `grep -cE "^\s*\w+_table\s*:"` on the adapter: **8** direct
`_table` fields, but the full set of guest-handle-indexed
`ArrayListUnmanaged(?Slot)` collections is wider:

* `stream_table`, `input_stream_table` — I/O streams (and the owned-lists
  `owned_input_streams`, `owned_output_streams`).
* `fs_descriptor_table`, `fs_preopens`, `dir_entry_stream_table` — filesystem.
* `network_table`, `socket_table`, `udp_incoming_streams`,
  `udp_outgoing_streams`, `resolve_streams`.
* `sockets_p3_stream_ctxs`, `fs_write_stream_ctxs`, `fs_read_stream_ctxs`.
* `pending_udp_receives`, `pending_http_fetches`,
  `pending_http_fetches_p3` — host-side async backlogs.
* `http_fields_table`, `http_outgoing_requests`, `http_incoming_requests`,
  `http_outgoing_responses`, `http_incoming_responses`,
  `http_request_options`, `http_response_outparams`,
  `http_incoming_bodies`, `http_outgoing_bodies`,
  `http_future_responses`, `http_future_trailers`,
  `http_requests_p3`, `http_responses_p3`, `http_request_options_p3`.
* `keyvalue_buckets`.
* `pollable_table`.
* `timer_futures`, `timer_future_ready`.

**Adapter-wide singletons / borrowed slices (read-mostly):**

* `argv`, `env`, `config_store` — borrowed slices set at startup;
  read-only during run.
* `sockets_allow_list_template: []IpCidr` — read-mostly; replaced
  atomically by `setSocketsAllowList`. Today a single
  `[]IpCidr` slice, so updates aren't lock-free safe under concurrent
  reads but no production code path mutates it after startup.
* `wall_clock_override`, `monotonic_clock_override`, `log_level` —
  injected once at startup.
* `stdin`, `stdout`, `stderr` — `streams.OutputStream` / `InputStream`
  instances; the `OutputStream` write paths funnel through host file
  descriptors which **the kernel already serialises**.
* `exit_code: ?u32` — written once when the guest calls `wasi:cli/exit.exit`.

**`TaskManager` ([`src/component/async.zig:157`](../../src/component/async.zig)):**

The async task manager is per-component-instance, and within an
instance it tracks `current_task` for `context.{get,set}` / `task.yield`.
The single-threaded executor assumes only one task is on the dispatch
stack at a time; with real threads, **each thread needs its own
TaskManager** (since each thread has its own "currently active task"),
or the design must declare that `wasi:threads`-spawned threads never
run async-lifted entry points.

## Design options

Three options exist for how to slot real threads into the runtime.
They are not mutually exclusive — the recommendation is a *staged*
adoption of Option A → Option B.

### Option A — One `Interpreter` per WASI thread, **separate** linear memory

* Each spawned thread gets a fresh `ModuleInstance` whose `memories`
  slice points at *its own* freshly-allocated linear memory.
* Each thread gets its own `ExecEnv`.
* Resource tables on the host side become per-thread by cloning the
  `WasiCliAdapter` (or by making the adapter ref-counted and giving
  each thread its own slot pool).

**Pros.**

* Trivial isolation; no races at the wasm level by construction.
* No mutex on host resource tables — each thread has its own.
* Matches the "one-component-instance-per-task" pattern Wasmtime uses
  for its **async** runtime.

**Cons — fatal for the use case.**

* Defeats the entire reason `wasi-threads` exists: real threaded code
  (Rust's `wasm32-wasi-threads`, `pthreads` via `wasi-libc`, Emscripten
  ports) **needs** shared linear memory because that is where the
  shared heap, locks, and TLS regions live.
* `i32.atomic.rmw.add` on a per-thread memory is just a non-atomic add;
  guests that depend on atomics-as-synchronisation are silently broken.
* Not what Wasmtime's `wasi-threads` does. Components built for
  `wasi-threads` would crash or deadlock under this model.

**Verdict.** Useful only as an intermediate scaffolding step
("can we even spawn a `std.Thread`?") before Option B; ship-worthy
only for embedders that don't run real threaded guests.

### Option B — One `Interpreter` per WASI thread, **shared** linear memory + atomics

This is the model upstream `wasi-threads` actually specifies and the
one wamr's existing prototype ([`thread_manager.zig`](../../src/wasi/thread_manager.zig))
already targets.

* **Memory:** `MemoryInstance` is ref-counted and pointed at by every
  thread's `ModuleInstance.memories[]` slot. All `i32.load` /
  `i32.store` already operate on the shared bytes; `i32.atomic.*` etc.
  add the synchronisation. The core-wasm `threads` proposal
  (atomic loads/stores/RMW/`wait`/`notify`) is already partly wired in
  `interp.zig:3349` (`atomic_prefix`) — `atomic.fence` is a no-op
  ("Fence is a no-op for single-threaded execution.") and most RMW
  opcodes are implemented as **non-atomic** sequences. **This is the
  single biggest correctness gap to close before Option B is real.**
* **Globals:** Cloned per thread via `cloneForThread` (today's
  behaviour). When `shared-everything-threads` lands, `shared` globals
  will need atomic access; for now, the Preview-1 model (mutable
  globals are thread-local) is correct.
* **Tables:** Ref-counted, shared. Mutation through `table.set` etc. on
  a `shared` table needs atomic stores once `shared-everything-threads`
  lands; until then, only the `__indirect_function_table` (function
  references, written-once at init) is realistically touched.
* **Host resource tables:** Stay on one `WasiCliAdapter` per process,
  but every `*_table` mutation acquires a per-table mutex. The
  uncontended fast path is a single atomic CAS (`Mutex.tryLock`), so
  the single-threaded performance hit is the cost of one CAS per
  resource-table mutation — measurable on `coremark`-style microbenchmarks
  but should land within the 2 % budget for non-microbench workloads.
* **`TaskManager`:** Per-thread (each thread's `ExecEnv` carries a
  pointer); async-lifted entry points run only on the spawning thread
  unless and until we extend the async runtime to be MT-aware.
* **Trap propagation:** When any thread traps, every other thread must
  see the trap. The existing `ThreadManager.signalTrap`/`hasTrap`
  scheme already provides this; `dispatchLoopWithFuel` already polls
  `env.thread_manager.?.hasTrap()` once per fuel tick
  (`interp.zig:1534`).

**Pros.**

* Matches both `wasi-threads@0.1.0` (legacy Preview-1) and the future
  `shared-everything-threads` runtime model.
* Compatible with real `wasm32-wasi-threads` Rust binaries and
  `pthreads`-based C libraries.
* Builds on the prototype already in `src/wasi/thread_manager.zig` —
  most of the bones exist, the gap is correctness + adapter
  thread-safety + atomic-opcode hardening.

**Cons.**

* Every `*_table` access on the hot path (every stream read/write,
  every fd lookup, every HTTP fields mutation) pays a mutex acquire.
  Measurable single-threaded regression must be budgeted.
* Atomics in the interpreter are currently *implemented but not
  actually atomic*. Closing that gap is non-trivial — Zig's
  `std.atomic` doesn't give us pointer-into-`[]u8`-with-ordering
  directly; we need explicit `@atomicLoad` / `@atomicRmwOp` casts.
* `task.cancel` semantics across threads are undefined by the current
  proposal — see [Open questions](#open-questions).

### Option C — Async-only (model threads as cooperative tasks)

* Spawn no real `std.Thread`; instead, the `thread-spawn` host fn
  schedules a new `Task` on the existing single-threaded executor.
* "Threads" run interleaved, switching at `task.yield` or async-await
  points.

**Pros.**

* Zero new concurrency primitives; reuses `TaskManager` directly.
* No mutex on resource tables; no atomic-opcode hardening required.
* Determinism is easier to argue.

**Cons — fatal for the use case.**

* `pthreads`-compiled guests have **no `task.yield` points**. A
  compute-bound thread blocks the executor indefinitely.
* Atomic-RMW + `wait`/`notify` would have to be emulated as scheduler
  hints, which is a research project.
* Does not match upstream semantics; Wasmtime users would not be able
  to run their existing `--wasi-threads` binaries against wamr.

**Verdict.** Insufficient for any real-world thread workload.

## Recommendation

**Option B**, staged in over multiple waves. Option A is a useful
scaffold only insofar as Wave 2 *temporarily* uses it to validate
`std.Thread.spawn` plumbing before Wave 3 enables real shared-memory
atomics.

Concretely:

* Target **`wasi:threads@0.1.0`** (the legacy Preview-1 surface) as
  the first guest-visible binding, because it is what every existing
  `wasm32-wasi-threads` binary actually imports.
* Treat **`shared-everything-threads`** as a future-compat checkpoint:
  the canon built-ins (`thread.spawn`, `thread.exit`, …) will land in
  a follow-up wave once upstream pins a WIT shape and Wasmtime ships a
  reference implementation.
* Do **not** invent a `wasi:threads@0.3.x` interface name. The
  `WasiCliAdapter`'s `populateWasiProviders` version-multiplex
  routes by interface name, and no such interface exists. The
  Preview-1 surface is bound at the core-wasm import layer
  (`wasi.thread-spawn`), not the component layer, so the existing
  `host_functions.zig` registration path (already prototype'd in
  `src/wasi/host_functions.zig:3209`) is the right seam.

## Implementation plan (multi-wave)

Each wave is a discrete PR with its own conformance gate.

### Wave 1 — Thread-safe resource tables on the host adapter

**Scope.**

* Introduce a `ResourceTable(T)` helper in `src/component/instance.zig`
  that wraps `std.ArrayListUnmanaged(?T)` + a `Mutex` (the same
  custom-spinlock pattern as `thread_manager.zig:11`, until Zig 0.16
  reinstates `std.Thread.Mutex` without `Io`).
* Migrate every `*_table` field on `WasiCliAdapter` and every
  `…_table` / handle-keyed hashmap on `ComponentInstance` to the new
  wrapper.
* Gate the mutex acquire behind `comptime config.lib_wasi_threads`:
  when threads are disabled, the wrapper is a transparent newtype with
  no lock. **No runtime cost when `lib_wasi_threads = false`.**

**Verification.**

* `zig build test` stays green.
* Run `coremark.wasm` / a microbenchmark from `tests/perf/` (or set
  one up alongside this wave if none exists) with `lib_wasi_threads`
  off, before vs after. Document the delta in the PR; target ≤2 %.
* No fixture regressions on `wasi-testsuite` / `wasi-p2-testsuite` /
  `wasi-p3-testsuite`.

### Wave 2 — `std.Thread` spawning for interpreter and AOT execution

**Scope.**

* Promote `ThreadManager.spawnThread` to non-experimental: today it's
  gated by `config.lib_wasi_threads` (off by default). Re-enable it
  under a new build flag (`-Dlib_wasi_threads=true` already exists in
  [`build.zig:67`](../../build.zig)).
* Wire both interpreter and AOT entry paths. A prototype that spawns
  only an interpreter loop is not production-complete.
* Pin down `cloneForThread` semantics: confirm the cloned globals,
  shared memories, shared tables, shared host_functions match
  `wasi-threads` spec section "instance-per-thread".
* Wire the existing `wasi_thread_start` smoke tests in
  `thread_manager.zig` into the default test step (currently they
  pass under unit tests but spawning is opt-in).

**Verification.**

* `zig build test -Dlib_wasi_threads=true` green.
* New fixture: `tests/wasi-threads-fixtures/spawn-and-join.wasm` that
  imports `wasi.thread-spawn`, exports `wasi_thread_start`, increments
  a shared atomic counter from 8 threads, and asserts the final value
  under both interpreter and AOT execution.
* CRC mismatch indicates either the `cloneForThread` race or the
  non-atomic RMW gap — drive the diagnosis with the
  [`aot-diff-debug`](../../) skill plus the new
  thread-spawn fixture.

### Wave 3 — Wasm `threads` proposal: real atomics

**Scope.**

* Audit `interp.zig:3349`–`3700+` (`atomic_prefix` dispatch). Replace
  every non-atomic load/store/RMW with `@atomicLoad`, `@atomicStore`,
  `@atomicRmw`, `@cmpxchgStrong` against the underlying `[]u8`.
* Route `memory.atomic.wait32`, `memory.atomic.wait64`, and
  `memory.atomic.notify` through the implemented per-memory keyed parking
  lot. The runtime helpers are wired; opcode validation and the surrounding
  atomic load/store/RMW audit remain part of this wave.
* Treat `atomic.fence` as `@fence(.seq_cst)` (today it is a documented
  no-op).
* Reject `shared` memories that are not declared `shared` in the
  parsed `MemoryType` (validation gap).

**Verification.**

* Re-run the Wave-2 smoke fixture; it must now produce *deterministic*
  final counters even on x86-64 (where non-atomic RMW happens to work
  most of the time).
* Add the upstream
  [`testsuite-threads`](https://github.com/WebAssembly/threads/tree/main/test/core)
  spec-test subset to `zig build spec-tests`. Drop any fixture that
  doesn't pass with a skip-list entry + tracking issue.

### Wave 4 — `wasi.thread-spawn` host import binding (Preview-1 surface)

**Scope.**

* The host function is already prototype'd at
  [`src/wasi/host_functions.zig:46`](../../src/wasi/host_functions.zig) +
  registered at `:3209`. Productionise it: today it lives behind
  `config.lib_wasi_threads`; flip the default for the WASI Preview-1
  test suite when the suite contains threaded fixtures.
* Wire `ThreadManager` lifetime to `ComponentInstance` /
  `WasiCliAdapter` so a `wasi.thread-spawn` from inside a component
  is observable. The prototype wiring reaches **core-module-only**
  runs (`runWasm` in `src/main.zig`) but not `runLoadedComponent` and
  is not a production path. Fix that by retaining a per-`WasiCliAdapter`
  `ThreadManager` and threading it into every `ExecEnv` the adapter
  creates.
* Document the
  Wasmtime-compat caveat: per spec, "a trap or WASI exit in one
  thread must end execution for all threads." We already have
  `signalTrap`; verify it observably stops other threads' dispatch
  loops before `proc_exit` returns control to the embedder.

**Verification.**

* End-to-end fixture: compile a tiny C `pthread_create` /
  `pthread_join` program with `wasi-sdk -pthread`, run via
  `wamr -Dlib_wasi_threads=true run -- foo.wasm`, assert the
  expected stdout (`"hello from thread 0..7"` in arrival order).

### Wave 5+ — Cancellation, joining, TLS, future-compat for `shared-everything-threads`

**Scope (one mini-wave each).**

* **5a — Cancellation across threads.** Bind `wasi:task.cancel` to set
  `ThreadManager.trap_flag` so cooperative cancel points in
  `dispatchLoopWithFuel` (the same `hasTrap` poll) eject other
  threads. Caveat: only safe if we audit that no thread holds a
  resource-table mutex when it observes the cancel — wave-5a must
  pair the cancel-check with a "release every held lock" routine in
  the trap exit path.
* **5b — Thread joining for embedders.** `ThreadManager.joinAll` is
  process-wide. Add `tid`-scoped `joinOne(tid: i32)` so embedders that
  call `runLoadedComponent` repeatedly can serialise.
* **5c — Per-thread WASI context.** Today every thread shares one
  `ExecEnv.wasi_ctx`. For correctness under threaded `errno` reads,
  give each `ExecEnv` its own `wasi_ctx` clone (the underlying
  `WasiCliAdapter` resource tables stay shared via Wave-1 mutexes).
* **5d — `shared-everything-threads` canon built-ins.** When upstream
  pins a WIT, add `populateWasiThreads` to `populateWasiProviders`
  and route `thread.spawn` (component-level) to the same
  `ThreadManager.spawnThread` already used by `wasi.thread-spawn`
  (core-level).
* **5e — Thread-local storage.** Preview-1 punts TLS to the guest
  (the `start_arg` is opaque). `shared-everything-threads` adds
  `thread-local globals`; binding them requires a wasm-runtime
  feature (one global slot per thread). Defer until 5d.

## Open questions

1. **Does upstream Wasmtime accept guest threads sharing host resource
   tables?** Probably yes — Wasmtime's per-`Store` resource tables are
   already protected by Rust's `&mut` borrow checker, and the
   `wasi-threads` crate runs all threads against the same `Store`.
   But the crate's README explicitly disclaims multi-tenant safety.
   We should at minimum match Wasmtime's "trap-in-one-thread ⇒
   process-exit" backstop until we have audited every resource table
   for lock-then-trap deadlocks.
2. **`task.cancel` across thread boundaries.** Today
   `task.cancel` is wired only for the single-threaded executor's
   timer-future path (#583 B1). When a future is owned by thread A and
   thread B calls `task.cancel`, the cancel-set bit has to be visible
   to A's `pollable.block` exit path. The `trap_flag` field on
   `ThreadManager` is the right hook, but it conflates "trap" and
   "cancel". Split it into `trap_flag` + `cancel_flag` (or
   per-tid bitset) in Wave 5a.
3. **One `WasiCliAdapter` per process vs per thread?** Decision:
   **one per process**. The argument for per-thread is "no mutexes" but
   it forces every preopened directory, every stdio buffer, every
   socket allow-list to be cloned — and the WASI semantics require
   that two threads sharing memory **observe the same fd table** (the
   `wasi-libc` `__wasilibc_register_preopened_fd` table is in shared
   linear memory, indexing into the host fd space). One adapter +
   per-table mutexes is the only model that matches the spec.
4. **Interaction with the async ABI broadening (#488 follow-ups).**
   Today the executor assumes a single `current_task_manager` per
   `ComponentInstance`. If thread A is in an async-lifted call and
   thread B spawns and also enters an async-lifted call, the second
   `currentTaskManager()` swap clobbers the first. **Decision:**
   `wasi-threads`-spawned threads MUST NOT enter async-lifted entry
   points until Wave 5+ moves `current_task_manager` from
   `ComponentInstance` to `ExecEnv`. The Preview-1 surface only ever
   re-enters at the exported `wasi_thread_start` core function, so
   this is naturally enforced today.
5. **Memory-grow under concurrent atomics.** Resolved for shared memory:
   reserve the declared maximum, keep the base immutable, serialize commit,
   and release/acquire-publish the visible extent. Non-shared memories keep
   their compatible allocator/reservation behavior.
6. **Should each thread have its own `WasiCliAdapter` slot for
   `stdin` / `stdout` / `stderr`?** No — stdio writes go through the
   kernel which already serialises. The OutputStream's internal
   buffer is a `std.ArrayListUnmanaged(u8)` and would race; wrap it
   in a `Mutex` in Wave 1 (it falls out naturally from the resource-
   table migration).

## Acceptance criteria for the design to land

A future "Wave 1 lands" PR may declare this design accepted iff:

1. At least one upstream surface is firm enough to pin against. **Today
   that is `wasi:threads@0.1.0` legacy + the WebAssembly `threads`
   proposal's atomic opcodes.** A `shared-everything-threads` WIT pin
   is **not** required; we will revisit when upstream and Wasmtime
   ship matching versions.
2. Single-threaded performance baseline measured **before and after**
   the Wave-1 resource-table-mutex migration: `coremark` (or, lacking
   that, a stable wamr microbenchmark like `tests/perf/dispatch_loop`)
   regression ≤ 2 % under `-Doptimize=ReleaseFast`,
   `-Dlib_wasi_threads=false`. Documented in the wave's PR body.
3. A working `wasi.thread-spawn` smoke test under `wamr` CLI in both
   interpreter and AOT modes:
   ```console
   $ zig build -Dlib_wasi_threads=true
   $ ./zig-out/bin/wamr run tests/wasi-threads-fixtures/spawn-and-join.wasm
   spawned 8 threads
   final counter = 8000000
   ```
4. `zig build wasi-testsuite` + `zig build wasi-p2-testsuite` +
   `zig build wasi-p3-testsuite` stay green with the default build
   (`lib_wasi_threads = false`). No regression on the 72 + 5 + 41
   fixture baseline.

## Out of scope

* **Preemptive scheduling.** Upstream `wasi-threads` explicitly leaves
  scheduling to the host OS scheduler ("optionally, spawn a new
  host-level thread"). We use `std.Thread` and inherit OS preemption;
  no userspace scheduler is in scope.
* **Thread-local storage beyond what the wasm `threads` proposal
  provides.** TLS in the Preview-1 model is a guest-side
  trampoline reading the user start function out of `start_arg`.
  First-class TLS via `shared-everything-threads` thread-local globals
  is Wave 5e.
* **GPU / accelerator threads.** `wasi-parallel` is a separate
  proposal; we are not targeting it.
* **`tokio` / async-runtime-style fairness.** No M:N scheduling on the
  guest side; threads map 1:1 to `std.Thread`.
* **Cross-component thread sharing.** A thread spawned by component A
  cannot enter component B. The `cloneForThread` semantics + the
  `ComponentInstance`-keyed task manager rule this out by
  construction.
* **Reverse engineering Wasmtime's `wasi-threads` private internals.**
  We follow the spec and our own resource-table model; we don't claim
  bit-for-bit Wasmtime parity beyond passing the same test programs.

## See also

* [`docs/wasi.md`](../wasi.md) — WASI feature matrix; this design
  doc is linked from the Roadmap section.
* [`src/wasi/thread_manager.zig`](../../src/wasi/thread_manager.zig)
  — the existing `wasi:threads@0.1.0` prototype (off by default and
  not production-wired).
* [`src/wasi/host_functions.zig`](../../src/wasi/host_functions.zig)
  (`wasiThreadSpawn`, line ~46; registration line ~3209) — the
  `wasi.thread-spawn` host import.
* [`src/runtime/common/exec_env.zig`](../../src/runtime/common/exec_env.zig)
  — `ExecEnv` declared per-thread; the `thread_manager` and `tid`
  fields already exist.
* [`src/runtime/common/types.zig`](../../src/runtime/common/types.zig)
  (`ModuleInstance.cloneForThread`, line ~835) — the
  shared-memory-cloned-globals split.
* [`src/runtime/interpreter/interp.zig`](../../src/runtime/interpreter/interp.zig)
  (`atomic_prefix`, line ~3349; the `// Fence is a no-op` comment is
  the load-bearing TODO Wave 3 has to flip) — the atomic-opcode
  dispatch table.
* [`src/component/wasi_cli_adapter.zig`](../../src/component/wasi_cli_adapter.zig)
  (line ~3779: `WasiCliAdapter` struct; ~3945: the resource tables)
  — Wave 1 migration target.
* [`src/component/instance.zig`](../../src/component/instance.zig)
  (line ~196) — `ComponentInstance` resource tables.
* [`src/component/async.zig`](../../src/component/async.zig)
  (line ~157: `TaskManager`) — the async runtime that Wave 5d has to
  make thread-aware.
* Upstream proposals:
  * [`wasi-threads`](https://github.com/WebAssembly/wasi-threads) —
    Preview-1 era, legacy. Champion: Alexandru Ene.
  * [`threads`](https://github.com/WebAssembly/threads) — core wasm
    atomics; Phase 4 (standard).
  * [`shared-everything-threads`](https://github.com/WebAssembly/shared-everything-threads)
    — Phase 1, active development. Future Component-Model target.
* [`crates/wasi-threads`](https://github.com/bytecodealliance/wasmtime/tree/main/crates/wasi-threads)
  — Wasmtime's reference implementation. README warns it is not
  multi-tenant safe.
