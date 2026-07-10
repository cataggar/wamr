# JIT compile→map→execute W^X hygiene

Status: **IMPLEMENTED** (see `platform.mapExecutableCode`, `src/platform/platform.zig`).

Tracking: [#858](https://github.com/cataggar/wamr/issues/858) (part of the in-process JIT plan, [#863](https://github.com/cataggar/wamr/issues/863)).

## Scope

Every JIT compile produces native code that must go from "just written
by the compiler, sitting in a wasm module's `text_section`" to "mapped,
executable, and callable" without a window where a page is
simultaneously writable **and** executable (W^X) — and, on Apple
Silicon, without violating macOS's `MAP_JIT` requirements for
JIT-compiled code. This note documents the guarantee for the one call
site that matters for the in-process JIT: `runtime.zig:mapCodeExecutable`,
which every `wamr run`/`wamr serve` JIT compile (core wasm or
per-component core) and every `.cwasm`-file load routes through.

The host-import trampoline pool (`src/runtime/aot/host_trampolines.zig`)
has its own, pre-existing W^X handling for a structurally different
allocation pattern (one big pool mapped once, individual trampolines
written into sub-slots lazily) and is **out of scope** here — it
already uses the same underlying primitives (`platform.jitWriteProtect`,
`platform.macos_jit`) this note describes, just composed differently
because it's a pool allocator rather than a single-blob mapper. See
["Why `host_trampolines.zig` wasn't refactored"](#why-host_trampolineszig-wasnt-refactored) below.

## The primitive: `platform.mapExecutableCode`

```zig
pub fn mapExecutableCode(code: []const u8) ?[*]u8
```

Maps `code.len` bytes, copies `code` in, flushes the instruction cache,
and returns the region in its final executable-and-not-writable state
— or `null` on any failure. The caller `platform.munmap`s it when done
(`AotInstance.destroy` already does this; see `JitCodeCache`, #857).

Two strategies, selected at comptime via `platform.macos_jit`
(`= is_macos and builtin.cpu.arch == .aarch64`):

### Linux, Windows, x86_64 macOS — RW → RX `mprotect`

1. `mmap` a fresh **RW** (not exec) region.
2. `memcpy` the compiled code in.
3. `icacheFlush` (a no-op on x86/x86_64; real cache-line invalidation
   on aarch64/arm).
4. `mprotect` the region to **RX** (not writable).

**Guarantee**: the region is RW-only until step 4's single `mprotect`
call, and RX-only from that point on. Nothing writes to it after step
4 (no call site retains a mutable view past `mapCodeExecutable`
returning). No other thread can observe a W+X page: exec permission
doesn't exist until the `mprotect` syscall completes, and by then the
write is already finished.

This is the path exercised by the `linux-x64`, `linux-arm64`, and
`windows-x64` CI jobs (`.github/workflows/ci.yml`) and by the
`mapExecutableCode: mapped code is genuinely callable` unit test
(`src/platform/platform.zig`), which maps a tiny real function body and
calls it through a function pointer — proving the region is truly
executable, not just readable.

### macOS aarch64 (Apple Silicon) — `MAP_JIT` + per-thread toggle

Apple Silicon's hardened-runtime JIT model doesn't support "mmap RW,
then mprotect to RX" for JIT-compiled code: a region has to be mapped
`MAP_JIT` (RWX at the VMA/mapping level) up front, and the actual
write-vs-execute enforcement is a **per-thread** toggle via
`pthread_jit_write_protect_np`, which Zig's `platform.jitWriteProtect`
wraps. A `MAP_JIT` region's protection cannot be changed after the
fact with a second `mprotect` the way a normal page's can.

1. `mmap` with `MAP_JIT` set and `RWX` requested — the OS grants the
   *capability*, not simultaneous access for every thread.
2. `jitWriteProtect(false)` — this thread flips to write-enabled
   (execute-disabled) for its view of the mapping.
3. `memcpy` the compiled code in.
4. `jitWriteProtect(true)` — this thread flips back to
   execute-enabled (write-disabled).
5. `icacheFlush`.

**Guarantee**: at every point in time, *this thread's* view of the
page is either write-enabled-and-execute-disabled or
execute-enabled-and-write-disabled, never both — the OS enforces this
per-thread, not by revoking the underlying RWX mapping. A new thread
that later calls into this code starts in the execute-enabled default
state for `MAP_JIT` regions, so no additional toggling is needed on
the executing thread as long as it never itself calls
`jitWriteProtect(false)` on this mapping.

This mirrors the pattern `host_trampolines.zig`'s `TrampolinePool`
already established and validated for the host-import trampoline pool
(used by every AOT component test that exercises cross-instance /
canon-lower dispatch).

## What's verified vs. what isn't

| Target | Verified how |
|---|---|
| Linux x86_64 | CI (`linux-x64` job) + local dev VM; `mapExecutableCode` unit test calls mapped code through a function pointer. |
| Linux aarch64 | CI (`linux-arm64` self-hosted runner); same unit test, plus the aarch64 `icacheFlush` d-cache/i-cache line invalidation path. |
| Windows x86_64 | CI (`windows-x64` job); same unit test (NT API `mmap`/`mprotect` path, shared with the non-macOS-JIT branch since Windows isn't `macos_jit`). |
| macOS x86_64 | **Not covered by CI** (no macOS runner in `.github/workflows/ci.yml`) — but this target isn't `macos_jit`, so it takes the same well-covered RW→RX `mprotect` path as Linux/Windows. Low risk. |
| macOS aarch64 (Apple Silicon) | **Not covered by CI or local testing** — no macOS hardware was available to implement this. The `MAP_JIT` + `jitWriteProtect` code path is new for `mapCodeExecutable` (though the same primitives are already exercised in production by `host_trampolines.zig`'s trampoline pool, which every AOT component end-to-end test exercises indirectly). **Follow-up filed**: [#874](https://github.com/cataggar/wamr/issues/874) asks a maintainer with Apple Silicon hardware to validate `wamr run <core.wasm>` / `wamr serve <component.wasm>` on a `-Djit=true` macOS aarch64 build and confirm no `MAP_JIT`-related crash or trap, rather than leaving this a silent gap. |

## Why `host_trampolines.zig` wasn't refactored

`TrampolinePool.initWithCap` maps one large region up front (sized for
`cap` slots), zero-fills it, and then writes individual trampoline
stubs into sub-slots **lazily**, one at a time, as imports are
resolved — each stub write itself wrapped in its own
`jitWriteProtect(false) … jitWriteProtect(true)` pair (see
`TrampolinePool.installStub`-equivalent call sites at
`host_trampolines.zig:625-631`). This is a pool-allocator pattern:
many writes over the pool's lifetime, not "compile once, map once."

`mapExecutableCode` is deliberately the opposite shape: one call maps
*and* writes *and* finalizes a single, complete code blob — the shape
`mapCodeExecutable` needs (a whole AOT module's `text_section` is
final before mapping). Forcing the pool to go through
`mapExecutableCode` would mean either (a) writing all `cap` slots
before the first mapping call (defeating the pool's lazy-fill design
and cache locality benefits), or (b) `mapExecutableCode` growing a
second lazy-write mode that only the pool would ever use. Neither is
worth the risk for this issue's scope; the pool already correctly uses
the same two underlying primitives (`platform.macos_jit`,
`platform.jitWriteProtect`) this note describes, just composed for its
own allocation shape.

## See also

- `src/platform/platform.zig` — `mapExecutableCode`, `jitWriteProtect`, `macos_jit`.
- `src/runtime/aot/runtime.zig` — `mapCodeExecutable` (the JIT/AOT call site this note covers).
- `src/runtime/aot/host_trampolines.zig` — the pre-existing, differently-shaped W^X handling for the host-import trampoline pool.
- [#857](https://github.com/cataggar/wamr/issues/857) — `JitCodeCache` registry (tracks/bounds resident JIT code across repeated compiles; orthogonal to this note's mapping-strategy concern).
