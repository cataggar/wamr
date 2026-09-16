# Optional native in-process compiler

This is **dependent software integration for #1044**, not native Unikraft JIT
qualification. It depends on the native AOT API introduced by draft #1050;
that API must work and be qualified first. Physical image/adapter integration
and boot evidence remain pending in `cataggar/unikraft#156`. Linux tests and QEMU
are correctness evidence only. No CoreMark speedup, Azure result, image size,
or hard real-time latency claim follows from this work.

## Explicit build and API boundary

```sh
# Unchanged compiler-free default guest profile:
zig build -Dprofile=unikraft-aot -Doptimize=ReleaseSafe -j2
# Explicit opt-in, separate artifact and Zig module:
zig build -Dprofile=unikraft-jit -Doptimize=ReleaseSafe -j2
```

The latter exports Zig module `wamr-jit`, namespaces `jit` and `aot`, and builds
`lib/libwamr-jit.a`. It targets `x86_64-freestanding-none`, SysV, single-threaded,
no red zone, libc, stack protector, unwind tables, or Zig error tracing.
This is an application-object contract, not an ISR ABI.

The public optional-compiler interface is **Zig**, not a new C ABI. Existing
compiler-free C entry points remain in `wamr_aot.h`; they cannot request the
new metered artifact's mandatory budgets and therefore reject it. The two
`wamr_jit_*_link_check` symbols retain real compile and load/start/call paths in
the freestanding link audit. They are not application entry points. Do not
execute the link-check ELF as a program or call it a bootable image.

The compiler uses the existing `component/aot_compile.zig` **core** pipeline:
core parser, frontend, `passesForPreset(.fast/.full)`, x86 codegen and emitter.
Compile-time gates exclude component execution, hosted AOT runtime, filesystem
cache/sidecars, hosted diagnostics, lazy tiering and environment-dependent
bisection. The parser lives in an `interpreter/` directory but no interpreter
execution engine is linked. No subprocess, filesystem, scheduler, POSIX layer,
environment variable or interpreter fallback is required or consulted.
Unsupported feature flags fail during build configuration; unsupported module
features/options fail explicitly before executable mapping.

Following [Unikraft's architecture](https://unikraft.org/docs/internals/architecture),
the optional compiler and compiler-free executor are fine-grained static
modules, sharing a checked API rather than bypassing it. Allocator selection is
separate from native page policy. Shared kernel address space is not protection:
bounds checks, typed imports, explicit traps and RW-to-RX transitions remain
mandatory. Platform callbacks still need the concrete native implementation
described in [unikraft-aot.md](unikraft-aot.md).

```zig
const native = @import("wamr-jit");
var compiled = try native.jit.compile(allocator, @embedFile("workload.wasm"), .{
    .preset = .fast, // .full reuses the existing full pass preset
    .verify = .after_each_pass,
    .max_compiler_bytes = 64 * 1024 * 1024,
});
defer compiled.deinit();
const instance = try native.aot.Instance.load(allocator, platform,
    compiled.bytes, imports, native.jit.runtime_options);
defer instance.deinit();
const startup = try instance.start(); // inspect terminal outcome
_ = startup;
var results: [1]native.aot.Value = undefined;
const outcome = try instance.call("workload", &.{.{ .i32 = 2000 }}, &results);
// Read results only after checking outcome.returned == 1.
_ = outcome;
```

Compilation is always explicit. `Instance.load` still accepts only precompiled
bytes and never compiles. JIT artifacts pass the same native admission policy
as `wamrc --profile=unikraft-x86_64`, before optimization and after lowering.
They use the explicit **fuel subprofile** flag `0x554b0002`, contract 1 and the
same native CPU/ABI requirements; ordinary Linux containers are never relabelled.
The original AOT subprofile `0x554b0001` and its compiler-free build remain
unchanged. Native text is still trusted matching-compiler output.

## Enforced limits and actual granularity

`jit.Options` caps input bytes, retained compiler allocations, emitted code,
function count/bytecode/locals, per-function IR blocks/instructions, and poll
count. Scratch storage is a caller-allocated **capped arena**: peak accounting
counts retained backing chunks, including fragmentation, not just logically
live IR. This deliberately trades some memory for complete reclamation when
an optimization pass fails or is cancelled. The separately owned result copy
is included in the cap. Caller allocator bookkeeping and caller-owned input
storage are not included; the input size has its own cap.

Compiler cancellation/deadline checks run on scratch backing-allocation growth,
at phase boundaries, between function lowerings/codegens, before/after each
scheduled per-function pass, and between inliner rounds. IR complexity is
checked at those function/pass boundaries. Parsing/validation of a capped input,
a single bounded function lowering/codegen, individual pass/analysis internals,
metadata emission and allocator callbacks are **not preempted**. Cancellation
can therefore overshoot an absolute deadline by such work; this is cooperative
admission/cancellation, not a hard latency bound. An absolute deadline without
a caller clock is rejected. Clock errors propagate. No clock means phase times
are null, never fabricated zeros. Verification mode is preserved.

The native loader requires explicit nonzero `max_run_fuel` and explicit
`max_heap_bytes`, `max_reserved_bytes`, `max_code_bytes` for metered artifacts.
`jit.runtime_options` supplies finite defaults. Heap accounting includes the
instance and retained metadata arena. The reservation cap includes rounded RX
code plus the entire maximum linear-memory reservation; table storage falls
under the heap cap. Existing page/table limits and wasm maxima still apply.
Allocation/commit/protection failure rolls ownership back.

Fuel decrements at every generated function entry and backward branch target.
Exhaustion traps before the block executes, including an allocation-free
infinite loop. Each checked export/start invocation receives a fresh budget.
One unit is a poll, **not one wasm instruction or one nanosecond**. Branch
selection, optimization and presets can change consumption. An unmetered AOT
artifact rejects a requested fuel budget rather than claiming to enforce it.
No runtime wall-clock deadline option is offered. Callers must bound their own
host callbacks and page/allocator services; fuel cannot interrupt them.

For generated native stack admission, this JIT subset rejects recursion and
indirect/reference calls, limits direct call depth to 32, and rejects generated
frames exceeding 8192 bytes including saved registers and scalar call argument
allowance. Provision at least 256 KiB for those generated frames **in addition
to** the embedder/compiler/helper/host-callback stack requirements. The compiler
itself and external callbacks use the caller's native stack; no independent
stack switch or native fault recovery is claimed. Threads/shared memories,
SIMD, GC/EH, memory64/multiple memories, components and HTTP are unsupported.
Scalar ABI limits remain five import parameters, sixteen function parameters,
one result and sixty-four imports; rejection precedes lowering.

## Metrics and evidence

`Artifact.metrics` exposes real caller-clock `parse_ns`, `lower_ns`,
`optimize_ns`, `codegen_ns`, `emit_ns`, plus native `code_bytes`.
`compiler_peak_bytes`, `compiler_retained_bytes` and `polls` are reported
separately. Phase buckets omit outer API bookkeeping, result-copy and cleanup;
measure the complete `compile` call separately when reporting cold latency.
`Instance.memoryStats()` exposes live/peak heap bytes, code bytes/reservation,
and linear reserved/committed bytes. These are explicit software allocation
metrics, **not RSS, page-table overhead, or image size**. Respect the supplied
clock's actual resolution.

The shared native loader's `Options.timings = &load_timings` composes with
`jit.runtime_options`: retain its mandatory caps and set that optional pointer.
`LoadTimings.completed` bit 0 validates `load_ns` (metadata/import admission);
bit 1 validates `instantiate_ns` (allocation, initialization and RW-to-RX).
Failed loads do not make unfinished phases valid. Export start, first result
and repeated calls remain distinct outer measurements, not loader phases.

The Linux producer/matched report work in #1046 owns complete compile,
load/instantiate, first-result and steady-state measurements. Compare the same
wasm and exact output, record fast/full and fuel instrumentation differences,
and measure the final compiler-free/JIT images separately after #156 integration.
Archive/library file lengths may be recorded now but must not substitute for
image measurements or measured process memory. QEMU timings are not performance
evidence.

```sh
zig build test-native-jit test-native-aot test-native-aot-abi -j2
zig build native-jit-fixture -j2
# Matching compiler-free fixture for the same deterministic source:
zig-out/bin/wamrc compile --target=x86_64 --profile=unikraft-x86_64 \
  zig-out/fixtures/native-jit.wasm -o zig-out/fixtures/native-jit.cwasm
```

Tests execute embedded arithmetic/control/volatile-memory work for exact
results with both presets, stable growth, bounds traps, repeated instances,
finite cancellation of allocation-free loops, cap rejection, and every
compiler backing-allocation failure with complete teardown. The existing AOT
regressions still validate imports, checked dispatch, traps and page failures.
x86 Linux CI runs natively; arm64 development uses pre-provisioned
`qemu-x86_64 -cpu max` only for correctness. These checks do not close #1044's
pending native/image and matched benchmark acceptance.
