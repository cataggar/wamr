# Embedded Component Deployment, `wit-zigar` Comptime Bindings & Zero-Copy WASI 0.3 Streams (`#1134`)

This guide explains how `#1134` changes component packaging, host/guest function dispatch, and WASI 0.3 (`wasip3`) stream I/O compared to WAMR's standard deployment workflow, along with usage examples and benchmark measurements.

---

## How This Deployment Process Differs for a New User

### 1. Packaging: External Sidecar Files vs. Single Self-Contained Binary

* **Standard Deployment:**
  1. Build your `.wasm` component.
  2. Run `wamrc compile-component app.wasm` to emit a sidecar `.cwasm.json` manifest and `core-0.cwasm`, `core-1.cwasm`, etc. on disk.
  3. Ship the host binary **plus** the `.wasm`, `.cwasm.json`, and `.cwasm` files together; at startup, the host reads and verifies them from the filesystem (`aot.loadComponentManifest`).
* **`#1134` Embedded Deployment:**
  1. In `build.zig`, call `addEmbeddedComponent(b, wamrc, .{ .component_path = ..., .aot = true })` (from [`build/component_embed.zig`](../build/component_embed.zig)).
  2. Zig invokes `wamrc` at build time and packages the `.wasm`, manifest metadata, and precompiled `.cwasm` core slices directly into the host binary via `@embedFile`.
  3. At runtime, call `aot.resolveEmbeddedCoresForComponent(...)` (or `aot.resolveEmbeddedManifestForComponent(...)`) and `instance.instantiateWithOptions(...)` — **zero filesystem I/O**, full SHA-256 verification, and a **single native executable** to deploy.

---

### 2. Host & Guest Calls: Runtime `InterfaceValue` Boxing vs. Comptime Zig Types

* **Standard Deployment:**
  * Host imports receive `[]const abi.InterfaceValue` and write `[]abi.InterfaceValue`. Every call walks `TypeRegistry` at runtime and heap-allocates intermediate slices.
  * Calling a guest export (`executor.callComponentFunc`) requires manually packing and unpacking `InterfaceValue` tagged unions and allocating an `ExecEnv` per call.
* **`#1134` Comptime Bindings (`src/component/canon_comptime.zig`):**
  * Register ordinary Zig functions directly with `canon_comptime.makeHostFunc(myZigFn)` (or `Instance.bindTypedHostFunc`). Arguments and return values — primitives, strings, slices, records, tuples, variants, enums, flags, `Option`, `Result`, `Own<T>`, and `Borrow<T>` — are lifted and lowered at comptime directly between flat core registers / linear memory (`HostFunc.call_flat`) across both `wasm32` and `wasm64` (`memory64`) with **zero heap allocations**.
  * Call guest exports directly as typed Zig calls via `canon_comptime.callTypedExport(FnType, .{}, inst, "export-name", .{ arg1, arg2 }, allocator)` (or `Instance.callTypedExport`), reusing a cached `ExecEnv` (`ComponentInstance.call_env`) on non-reentrant interpreter calls and inline `AotFrame` stack buffers on AOT calls.

---

### 3. WASI 0.3 (`wasip3`) Streams & Stackful Async Tasks

* **Standard Deployment:**
  * `stream<u8>` transfers allocate and copy through an intermediate host FIFO buffer (`AsyncStream.buffer`), and `waitable-set.wait` polls in fixed intervals.
* **`#1134` Deployment:**
  * `WaitableSet` ([`src/component/async.zig`](../src/component/async.zig)) is backed by Zig 0.17 `std.Io.Event` readiness notifications (`waitReady`) for immediate wakeup without fixed-cap polling.
  * Stackful guest coroutines ([`src/component/coro.zig`](../src/component/coro.zig)) isolate per-task linear-memory `__stack_pointer` ranges via `ShadowStackPool` / `TaskContext` with 64-bit canary bounds checking (`0xC0DEC0FFEEFACADE`) while preserving persistent `ExecEnv` stacks across suspensions (`[waitable-set]wait`, `task.yield`, `thread.suspend*`, `thread.switch-to`, `thread.yield*`).
  * `wasi:cli/{stdout,stderr}@0.3.0#write-via-stream`, `wasi:cli/stdin@0.3.0#read-via-stream`, `wasi:filesystem` `read-via-stream`, and parked guest-to-guest `[stream]read` / `[stream]write` transfer bytes **directly between validated guest linear-memory slices (`readGuestBytes64` / `writableGuestBytes64`) and host `std.Io` streams (`HostStreamDriver`)** with zero intermediate buffer allocations or `cabi_realloc` calls.

---

## Quick-Start Examples

### 1. Build-Time Component Embedding (`build.zig`)

```zig
const component_embed = @import("build/component_embed.zig");

// Inside build(b: *std.Build):
const embedded_app = component_embed.addEmbeddedComponent(b, wamrc_exe, .{
    .name = "embedded_app",
    .component_path = b.path("guests/app.component.wasm"),
    .aot = true, // Precompile all core modules to .cwasm at build time
});
exe.root_module.addImport("embedded_app", embedded_app);
```

### 2. Runtime Instantiation & Comptime-Typed Host/Guest Calls

```zig
const std = @import("std");
const wamr = @import("wamr");
const embedded_app = @import("embedded_app");

const Point = struct { x: i32, y: i32 };

fn hostCompute(pt: Point, scale: i32) i32 {
    return (pt.x + pt.y) * scale;
}

pub fn run(allocator: std.mem.Allocator) !i32 {
    // 1. Decode component & bind embedded .cwasm cores in-memory (no disk I/O)
    var decoded = try wamr.component.loader.loadComponent(allocator, embedded_app.component_bytes);
    defer decoded.deinit(allocator);

    const resolved = try wamr.component.aot.resolveEmbeddedCoresForComponent(
        allocator,
        &decoded.component,
        embedded_app.component_bytes,
        embedded_app.precompiled_cores,
    );
    defer wamr.component.aot.freeEmbeddedResolution(allocator, resolved);

    // 2. Instantiate with embedded AOT cores and register comptime-typed host import
    var inst = try wamr.component.instance.instantiateWithOptions(allocator, &decoded.component, .{
        .precompiled_cores = resolved,
        .aot_only = true,
    });
    defer inst.deinit();

    try inst.addHostFunc("host:math/compute", wamr.component.canon_comptime.makeHostFunc(hostCompute));
    try inst.linkImports();

    // 3. Call guest export directly with typed Zig arguments & return type
    return try wamr.component.canon_comptime.callTypedExport(
        fn (Point, i32) i32,
        .{},
        inst,
        "run",
        .{ Point{ .x = 7, .y = 11 }, 3 },
        allocator,
    );
}
```

---

## Performance Measurements vs. Standard Deployment

Benchmarked in [`src/tests/component_wit_zigar_test.zig`](../src/tests/component_wit_zigar_test.zig) (`zig build test-wit-zigar` & `zig build test-wit-zigar -Doptimize=ReleaseFast`) and PR [#1135](https://github.com/cataggar/wamr/pull/1135) CI:

| Workload | Standard Deployment | `#1134` (`wit-zigar` + `std.Io` + Embedded AOT) | Speedup |
| :--- | :--- | :--- | :--- |
| **1. Host Import Dispatch (`canon.lower` trampoline, record + scalar params, 50k calls)** | **ReleaseFast:** `218.6 ns/call`<br>**Debug:** `27.76 µs/call`<br>*(Runtime `TypeRegistry` walk + heap `[]InterfaceValue`)* | **ReleaseFast:** `3.1 ns/call`<br>**Debug:** `204.5 ns/call`<br>*(Comptime `CanonLift` / `HostFunc.call_flat`, 0 heap allocs)* | **71.3x faster** (`ReleaseFast`)<br>**135.8x faster** (`Debug`) |
| **2. Guest Export Call — Interpreter (`canon.lift`, 10k calls)** | **ReleaseFast:** `33.64 µs/call`<br>**Debug:** `380.05 µs/call`<br>*(`executor.callComponentFunc` + per-call `ExecEnv` + `InterfaceValue`)* | **ReleaseFast:** `55.7 ns/call`<br>**Debug:** `1.24 µs/call`<br>*(Comptime `callTypedExport` + cached `call_env`, 0 heap allocs)* | **603.9x faster** (`ReleaseFast`)<br>**305.9x faster** (`Debug`) |
| **3. End-to-End Guest Export + Host Import Round-Trip — AOT `.cwasm` (`10k calls`)** | **ReleaseFast:** `934.4 ns/call` (`483.3 ns` w/ inline `AotFrame`)<br>**Debug:** `106.20 µs/call`<br>*(`callComponentFunc` + dynamic host trampoline)* | **ReleaseFast:** `332.1 ns/call`<br>**Debug:** `19.29 µs/call`<br>*(`resolveEmbeddedCoresForComponent` + `callTypedExport` + `call_flat` + inline `AotFrame`)* | **2.81x faster** (`ReleaseFast`, **1.46x** over inline `AotFrame` alone)<br>**5.51x faster** (`Debug`) |
| **4. WASI 0.3 `stream<u8>` 4 KiB Transfer (`20k ops`)** | **ReleaseFast:** `4,307.7 ns/op` (`~0.95 GiB/s`)<br>**Debug:** `46.97 µs/op`<br>*(Intermediate `AsyncStream.buffer` FIFO alloc/copy/free)* | **ReleaseFast:** `53.2 ns/op` (`~75.2 GiB/s`)<br>**Debug:** `60.9 ns/op`<br>*(Zero-copy `HostStreamDriver` / direct guest slice rendezvous, 0 `cabi_realloc`)* | **81.0x faster** (`ReleaseFast`)<br>**771.4x faster** (`Debug`) |
| **5. Existing Core/WASI CI Benchmarks (PR `#1135` vs `main`)** | `1.0000x` baseline | `0.9998x–1.0002x` across AOT hot, atomic, spawn-join, wait-notify, coldstart, and wasi-microbench | **0% regression** |

---

## Verification

Run the dedicated `#1134` test suite (covering embedded AOT `.cwasm` resolution, comptime `CanonLift`/`CanonLower`, `wasm32`/`wasm64` memory64 bounds validation, `std.Io` `WaitableSet` wakeup, stackful coroutines, and zero-copy `stream<u8>`):

```console
$ zig build test-wit-zigar --summary all
```
