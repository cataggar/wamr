# Minimal native WASI context

`src/wasi/minimal.zig` (build module `minimal-wasi`) supplies the **snapshot-0**
`wasi_unstable` subset needed by the two tracked CoreMark workloads. It is
allocation-free and has no dependency on the interpreter, compiler, hosted
`WasiProcessState`, filesystem, preopens, networking, or host threads.

This context is a building block for #1045, **not evidence of a working Unikraft
image or compiler-free AOT execution**. Native import registration and terminal
unwinding must be connected to the native AOT embedding backend. Genuine runs of
both workloads must still preserve their CRC validation output and classify
return, `proc_exit`, and trap separately. Test clocks below do not qualify a
production run or a benchmark score.

## Ownership and platform callbacks

Create one `Context` for each native instance with explicit `Options`:

- `args` and `environment`: borrowed immutable byte strings, without trailing
  NULs. Embedded NULs and wasm32-unrepresentable tables are rejected at creation.
  Keep the slices and their backing strings alive and immutable until the
  instance is destroyed. No host arguments or environment are inherited.
- `descriptors`: independent open/closed state for descriptors 0, 1, and 2.
  Each can optionally identify a real terminal. No other descriptor exists.
- `output`: an optional synchronous callback plus opaque caller-owned state.
  It receives the actual bytes and descriptor (1 = stdout, 2 = stderr), including
  embedded NULs. It must not retain guest slices, modify guest memory, reenter
  the guest, or report consuming more bytes than supplied.
- `clock`: an optional real platform clock callback plus opaque caller-owned
  state and resolutions. The context never manufactures a timestamp.

The platform state must outlive the instance. These are single-execution,
non-reentrant contexts: no global context, host-thread state, or global exit
flag is used. Closing a descriptor closes its **logical guest handle**, not the
platform output device shared with the embedding application.

### Output results and failure handling

`Output.write(userdata, fd, bytes)` returns
`WriteResult { written: usize, errno: Errno }`. `written` is the actual consumed
count, including on failure. The guest `nwritten` receives the total across
completed callbacks. A short or zero successful write stops the operation
without retrying or spinning. An error stops the operation and is returned to
the guest **even when earlier bytes were consumed**; it is never hidden as
success. WASI only guarantees result values on success, but this implementation
also records the actual count on error for embedding diagnostics.

Callback errors are WASI errno numbers, not host errno numbers. All WASI values
0–76 are accepted. Invalid numbers and overreported byte counts become `EIO`;
an overreported callback has violated the contract, so only the count from
earlier valid callbacks is known and recorded. Do not use this contract to
invent a successful output count or replace output with a debug logger.

### Clock contract

`Clock.resolution_ns[0..4]` declares support and the actual platform resolution:
zero means unsupported; positive values are nanoseconds. Supported IDs and
their timestamp semantics are:

| ID | Clock | Timestamp meaning |
|---:|---|---|
| 0 | realtime | Nanoseconds since the Unix epoch |
| 1 | monotonic | Nondecreasing nanoseconds from a fixed unspecified origin |
| 2 | process CPU | Nanoseconds of actual process CPU consumption |
| 3 | thread CPU | Nanoseconds of actual executing-thread CPU consumption |

`Clock.read(userdata, id, precision_ns)` receives the guest-requested precision
unchanged and returns either `timestamp_ns` or an explicit WASI error. Precision
does not change the units and does not promise a resolution better than the
platform's declaration. Do not map uptime to realtime or wall time to CPU time.
A known but unsupported ID returns `ENOTSUP`; an unknown ID returns `EINVAL`.
A failed read preserves its error; a failure carrying `ESUCCESS` becomes `EIO`.
Unsupported clocks and failed reads never modify the guest output timestamp.
The embedding platform is responsible for real sampling and truthful capability
metadata. Deterministic timestamps occur only in unit-test fixtures.

### Unikraft Hyper-V platform mapping

The intended native provider is pinned to
`cataggar/unikraft@40d8fc70`, `plat/hyperv/time.c`:

- `ukplat_monotonic_clock` (line 892) returns
  `hyperv_reference_delta_ns(hyperv_reference_time(), hyperv_boot_ref)`.
  Its C ABI return is `__nsec`/u64, already in nanoseconds. This maps to
  WASI monotonic clock ID 1 without another unit conversion.
- `ukplat_wall_clock` (line 898) returns
  `hyperv_wall_time_ns(hyperv_epoch_ns, hyperv_efi_ref,
  hyperv_reference_time())`, also `__nsec`/u64 nanoseconds. It maps to realtime
  ID 0 **only when a genuine EFI wall-clock epoch is available and qualified**.
  The existence of the symbol alone does not establish that capability.
- Neither hook supplies actual process or thread CPU consumption. Keep IDs
  2 and 3 unsupported unless the platform supplies separate qualified CPU
  clocks. Never substitute boot time or wall time for them.

The platform adapter must declare the source's real resolution in
`resolution_ns`, not infer one-nanosecond resolution from the return unit.
If wall-clock epoch availability or source resolution cannot be established,
do not advertise that clock as supported. The callback seam and this mapping
are a software integration contract, not a claim that these hooks have been
linked or exercised in a native image.

Native execution additionally depends on the backend's qualified memory
provider. Unikraft's `LIBUKVMEM` is off by default; reserve/map/protect/grow
callbacks alone do not demonstrate that the resulting image enables and
exercises its required paging and allocation capabilities. Those platform
enablement and linked-image checks belong to the native backend and the
Unikraft integration work, not this allocation-free WASI context.

## Exact imported ABI

`imports` contains the names below, their literal `wasi_unstable` namespace,
and raw Wasm parameter/result type bytes. `resolve` requires an exact match and
does **not** alias `wasi_snapshot_preview1`. All functions return an i32 WASI
errno except `proc_exit`, which has no Wasm results.

| Import | Wasm parameters |
|---|---|
| `fd_prestat_get` | i32, i32 |
| `fd_prestat_dir_name` | i32, i32, i32 |
| `environ_sizes_get`, `environ_get` | i32, i32 |
| `args_sizes_get`, `args_get` | i32, i32 |
| `clock_time_get` | i32, i64, i32 |
| `proc_exit` | i32 |
| `fd_fdstat_get` | i32, i32 |
| `fd_close` | i32 |
| `fd_seek` | i32, i64, i32, i32 |
| `fd_write` | i32, i32, i32, i32 |

The snapshot-0 seek enum is **CUR=0, END=1, SET=2**, unlike preview1. This
subset never grants seek rights: valid seek requests return `ENOTCAPABLE`,
unknown whence values return `EINVAL`, and absent/closed descriptors return
`EBADF`. The 24-byte fdstat layout has type at byte 0, flags at byte 2, base
rights at byte 8, and inheriting rights at byte 16, all little-endian. Only
`FD_WRITE` (bit 6) is granted to open stdout/stderr when an output callback
exists; stdin grants no I/O rights. No descriptor grants inheriting rights.
Nonterminal output is `unknown`, not a fabricated character device.

There are no preopens: both prestat functions return `EBADF`, including for
stdio handles. `fd_close` updates instance state and rejects a repeated close.

## Memory and terminal outcomes

All guest addresses are wasm32 offsets. Overflow-safe bounds checks cover
strings, pointer arrays, scalar outputs, fdstat, iovec arrays, and every iovec
buffer. Invalid ranges return `EFAULT`. `args_get`/`environ_get` write exactly
one pointer per string and concatenated NUL-terminated strings; they do not
append an extra pointer sentinel. Overlapping pointer/string output regions
return `EINVAL`. Each operation validates all its output/input ranges before
writing or invoking platform callbacks. The complete write byte count must
fit u32; otherwise `EOVERFLOW` is returned before output.

`Context.dispatch(memory, function, raw_arguments)` returns a tagged `Outcome`:

- `.returned = errno`: return the errno as the import's i32 result.
- `.exited = code`: **unwind native guest execution immediately**, preserving
  the complete u32 code, even zero. Do not return to the next Wasm instruction.

Argument count mismatch is an adapter error, not a successful WASI call.
`proc_exit` records a sticky instance-local code. Subsequent dispatches return
the same terminal outcome without performing an operation. The native adapter
must distinguish this outcome from guest traps and ordinary entry-point return;
the context itself does not implement platform stack unwinding.

## Focused validation

```sh
zig build test-minimal-wasi -j2
```

The same step is included in `zig build test`. Tests cover exact argv/env
layouts, absent preopens, rights and descriptor lifecycle/isolation, actual
output bytes, short/zero/error writes, invalid/overflowing guest pointers,
iovec/result aliasing, supported/unsupported/failed clocks, exact namespace and
signatures, and terminal zero/nonzero exits.

Fixture tests embed the **unchanged tracked inputs**, validate SHA-256, and
decode their type/import sections to verify all 12 required imports:

| Workload | SHA-256 |
|---|---|
| `tests/benchmarks/coremark/coremark_wasi.wasm` | `f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b` |
| `tests/benchmarks/coremark/coremark_wasi_nofp.wasm` | `24c0cc1bd52b641cf9e8ae74d1be188cba38d74cdb7ac18378de47382aab9541` |

These fixture checks do not execute CoreMark. Native AOT integration must
compile these exact inputs ahead of time, load only the native artifacts in
the consumer, use a real supported clock, preserve stdout/stderr bytes, and
retain `Correct operation validated.` plus the workload's CRC output.
