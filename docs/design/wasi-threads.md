# `wasi:threads` design — multi-threaded interpreter state isolation

Status: **DRAFT** (shared-memory/parking, atomic semantics, and the
thread-group lifecycle are implemented; production host binding remains
incomplete).

Tracking: [#616 B1.2-B1.3, B1.6](https://github.com/cataggar/wamr/issues/616).

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

The tree contains a hardened `ThreadManager` lifecycle plus scaffolding in
`cloneForThread`, the atomic-opcode dispatcher, and the
`wasi.thread-spawn` host import. No default CLI or component path currently
offers gated guest-thread execution.

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
shared runtime and host resources safe; spawn in both interpreter and AOT
modes; isolate cancellation and per-thread component/WASI context; and pass
the end-to-end correctness, conformance, and performance gates.

The Preview-1/core-resource portion of B1.1 is now established. `WasiCtx`
uses a conditional descriptor table: thread-enabled builds map guest fds to
stable lease-retained nodes, while disabled builds keep the direct hash-map
representation and compile out locks. Directory locking covers only guest-fd
and preopen-name bookkeeping; host descriptor destruction happens after
unlock and waits for the final lease. Preopen names are copied while the
directory lock is held rather than returning container-backed slices.
`MemoryInstance`, `TableInstance`, and `GlobalInstance` ownership counts are
atomic only in thread-enabled builds, and shared table access is serialized
without holding a lock across guest calls. Thread clones also roll back
partial retains/allocation failures and copy mutable segment state.

The `ComponentInstance` slice is also established. Canonical resource handles
now carry non-wrapping slot generations; Preview-3 futures, streams,
error-contexts, and waitable sets are reached through conditional stable
leases; task/waitable mutation is synchronized; and callbacks/destructors run
after releasing runtime locks. Disabled builds compile out locks and atomics;
callback in-flight markers defer reentrant destruction without per-lookup
reference updates. Execution-local task managers, implicit context slots,
canon-lower bindings, and realloc state remain on `ThreadExecutionContext` as
established by the context split.

The `WasiCliAdapter` slice is established as well. Its stream, filesystem,
socket, HTTP, keyvalue, pollable, callback-context, timer, UDP, and outbound
HTTP operation tables use the same conditional ownership model: disabled
builds retain compact direct arrays, while enabled builds use stable retained
nodes with non-wrapping generations. Child resources retain their exact
descriptor/socket/stream owners; pending operations claim completion exactly
once; shutdown cancels and joins producers before retiring their futures and
parents. Destructors and ComponentInstance callbacks run outside table locks,
and host I/O/waits hold only stable leases plus zero-sized-when-disabled
operation claims. Worker-backed HTTP tables remain synchronized in all builds
because those workers exist independently of `lib_wasi_threads`.

This completes B1.1's resource-lifetime slices. It does not wire production
interpreter/AOT spawning or final group cancellation.

The thread-group lifecycle now publishes a generation-stamped, manager-owned
record before a native child can enter guest code. The child waits on a start
gate and crosses the manager lock once after the gate opens, proving that
publication has completed and the publisher no longer holds the lock. Child
exit only records an outcome; the native handle, cloned instance, execution
environment, and auxiliary stack remain owned by the record until an exact
`joinOne` or batched, unbounded `joinAll` claims them. All joins and destruction
run outside the manager lock.

The host that owns `ThreadManager` also owns group shutdown. `shutdown` first
rejects new spawns, drains spawn calls that began before closure, and joins all
unclaimed records; `deinit` performs the same shutdown before freeing the
registry and stack pool. Threads are never detached. Shutdown deliberately
does not add cancellation or preemption semantics: running guest code must
finish cooperatively or observe the existing group trap flag.

The process/execution split is now explicit:

* `WasiProcessState` owns its argument and environment strings and the
  synchronized Preview-1 descriptor/preopen table. `WasiCtx` remains a source
  compatibility alias.
* `ProcessStateRef` is the type-erased retained handle used by runtime-common
  code. `ModuleInstance`, thread clones, `ExecEnv`, and `AotInstance` each
  acquire exactly one reference for their own lifetime.
* `ThreadExecutionContext` owns thread ID, opaque `start_arg`, optional
  auxiliary-stack/TLS metadata, implicit task-context slots, the active
  `TaskManager`, cancellation/trap flags, and temporary backend/host-call
  bookkeeping. Sibling contexts begin with fresh execution-local state and
  inherit only the retained process reference.
* Component canon task and lower-call state no longer lives on the shared
  `ComponentInstance`. AOT keeps all existing codegen-addressed `VmCtx`
  offsets stable and appends only a thread-context pointer.

This context work deliberately does not add production host bindings, AOT
thread spawning, or final group cancellation.

### Process/context lifetime contract

| Event | Ownership rule |
| --- | --- |
| CLI creates process state | The CLI owns the initial reference. Args/env are deep-copied before sharing. |
| Attach to interpreter/AOT instance | The instance acquires one reference; replacing/detaching releases exactly one. |
| Create `ExecEnv` | The environment acquires its own reference from the instance. |
| Clone a thread instance | The child acquires one process reference; allocation rollback releases it with all partially retained core resources. |
| Parent guest entry returns | Parent execution-local state may be destroyed without affecting a live child; the hardened thread record retains the child instance/env until join. |
| Child completion | Completion records an outcome only; `joinOne`/`joinAll` destroy the child `ExecEnv`, return its auxiliary stack, and destroy the clone. |
| Group shutdown | Shutdown drains in-flight spawns and joins all records. The final process reference closes descriptors/preopens exactly once. |

The Preview-1 ABI's `start_arg` is passed bit-for-bit to
`wasi_thread_start(tid, start_arg)`. The runtime does not inspect the
wasi-libc payload: wasi-libc's assembly trampoline reads its stack pointer at
offset 0 and TLS base at offset 4, then initializes `__stack_pointer` and
`__tls_base`. Runtime auxiliary-stack metadata therefore remains independent
of that guest-owned payload.

Atomic opcode and fence behavior is complete on both execution tiers.
Interpreted atomic loads, stores, RMW and `cmpxchg` are `seq_cst`, bounds-
and alignment-checked; `wait`/`notify` use the monotonic parking lot; and
`atomic.fence` emits a real barrier through `platform.memoryFenceSeqCst`
rather than the no-op it used to be. That last point matters because the AOT
backends have always emitted `MFENCE` / `DMB ISH` for the same instruction,
so a no-op in the interpreter was a tier-dependent memory model — the kind of
divergence that only surfaces as a rare race once threads actually run.

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

**Per-`ComponentInstance`:**

* `resource_tables` — a synchronized registry of heap-stable per-type tables;
  table slots use stable leases and non-wrapping generations when threads are
  enabled, and the compact direct-slot path when disabled.
* `exported_funcs`, `imports` — built once at link time.
* `trampoline_ctxs`, `canon_builtin_ctxs`, `canon_builtin_ctx_by_canon_idx`,
  `forwarding_ctxs`, `synthetic_host_instances`.
* `pending_core_starts`, `sub_instances`.
* `futures`, `streams`, `error_contexts`, `waitable_sets` — conditional
  stable keyed tables. Stream/future entries have short value locking;
  waitable sets synchronize their own queues.
* `next_async_handle` — atomic only in thread-enabled builds and never wraps.
* Implicit task context, the active task manager, cancellation state, and
  canon-lower call context live on the active `ThreadExecutionContext`, not
  mutable instance-global fields.

**Per-`WasiCliAdapter` resource and operation tables:**

All guest-visible adapter handles now go through
`src/shared/adapter_resource.zig`. First-generation handle values retain the
existing ABI; enabled slot reuse advances an encoded generation, while
disabled reuse preserves the direct-array numbering.

* `stream_table`, `input_stream_table` — I/O streams. Owned output streams use
  a small shared owner so an outgoing HTTP body remains valid after the guest
  drops its stream handle.
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
* `keyvalue_buckets`, `cas_table`.
* `pollable_table`.
* `timer_futures`, `timer_future_ready`.

Directory streams, TCP/UDP stream children, pending UDP receives, and
socket-backed ComponentInstance stream callbacks retain their exact parent
node. Poll waits snapshot backend descriptors while holding a lease, release
operation claims before blocking, and therefore cannot race descriptor close
or hold a table/value lock across `poll`.

**Adapter-wide singletons / borrowed slices (read-mostly):**

* `argv`, `env`, `config_store` — borrowed slices set at startup;
  read-only during run.
* `sockets_allow_list_template: []IpCidr` — snapshotted/replaced under a
  conditional operation claim; old storage is freed after publication.
* `wall_clock_override`, `monotonic_clock_override`, `log_level` —
  injected once at startup.
* `stdin`, `stdout`, `stderr` — `streams.OutputStream` / `InputStream`
  instances with shared per-sink operation claims, so aliases serialize
  without a table lock around host I/O.
* `exit_code`, insecure PRNG state, preopen metadata, timer-ready flags, and
  keyvalue persistence state each have conditional operation ownership.

**`TaskManager` ([`src/component/async.zig`](../../src/component/async.zig)):**

Task records and context slots are synchronized and task handles are never
reused. The current-task selection is TLS, while the active manager pointer is
bound on `ThreadExecutionContext`; concurrent host threads therefore cannot
overwrite one another. Interpreter and AOT entry paths supply their execution
context explicitly, including nested callbacks.

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
  (atomic loads/stores/RMW/`wait`/`notify`) is wired up in
  `interp.zig` (`atomic_prefix`): loads, stores, RMW and `cmpxchg` all
  lower to genuine `seq_cst` Zig atomics, `wait`/`notify` go through
  the [`parking_lot`](../../src/platform/parking_lot.zig) with
  monotonic deadlines and cancellation wakeups, and `atomic.fence`
  issues a real barrier via `platform.memoryFenceSeqCst` (`MFENCE` on
  x86-64, `DMB ISH` on AArch64) — the same instruction the AOT
  backends emit for the `atomic_fence` IR op. Bounds and alignment are
  checked on every atomic access.
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

* Use the conditional stable-handle/lease primitives for shared resource
  families. Disabled canonical-resource slots retain their compact direct
  representation; keyed async tables use heap-stable nodes but no locks,
  atomics, or per-lookup reference updates.
* `WasiCtx` and `ComponentInstance` are migrated. Every `*_table` field on
  `WasiCliAdapter` remains the separate adapter-resource slice.
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
* **5b — Thread joining for embedders.** The lifecycle now provides exact
  `joinOne(tid: i32)` and process-wide batched `joinAll`; wiring those APIs
  into repeated `runLoadedComponent` ownership remains part of the host-binding
  work.
* **5c — Process/per-thread context split.** Implemented: each execution
  environment has private task/cancel/trap/TLS metadata and a retained handle
  to one shared, synchronized WASI process state.
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
   Resolved for execution-local state: active task managers, implicit context
   slots, cancellation state, and canon-lower call bindings now live on the
   active `ThreadExecutionContext`, not `ComponentInstance`. The component
   async resource tables now use synchronized stable ownership; the
   `WasiCliAdapter` resource families remain the final B1.1 migration before
   component threads can be exposed.
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
  provides.** The runtime records per-thread TLS metadata but does not
  reinterpret Preview-1's guest-owned `start_arg`; wasi-libc's trampoline
  initializes `__tls_base`. First-class TLS via
  `shared-everything-threads` thread-local globals is Wave 5e.
* **GPU / accelerator threads.** `wasi-parallel` is a separate
  proposal; we are not targeting it.
* **`tokio` / async-runtime-style fairness.** No M:N scheduling on the
  guest side; threads map 1:1 to `std.Thread`.
* **Cross-component thread sharing.** A thread spawned by component A
  cannot enter component B. The `cloneForThread` semantics, execution-local
  task-manager binding, and instance-owned resource tables rule this out by
  construction.
* **Reverse engineering Wasmtime's `wasi-threads` private internals.**
  We follow the spec and our own resource-table model; we don't claim
  bit-for-bit Wasmtime parity beyond passing the same test programs.

## See also

* [`docs/wasi.md`](../wasi.md) — WASI feature matrix; this design
  doc is linked from the Roadmap section.
* [`src/wasi/thread_manager.zig`](../../src/wasi/thread_manager.zig)
  — the generation-safe `wasi:threads@0.1.0` lifecycle (off by default and
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
  (`TaskManager`) — synchronized task records and execution-local current-task
  binding used by the component async runtime.
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
