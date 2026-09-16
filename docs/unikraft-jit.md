# Optional native in-process compiler

This is **dependent software integration for #1044**, not native Unikraft JIT
qualification. It depends on the corrected native AOT API from #1050 and the
staged producer API from #1046. Development integration uses producer base
`7b0b7701eb882cf36ba02455a00b9dd64823cff6`; this is not a protected-main merge
or deployment qualification receipt. Physical image/adapter integration
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
Allocation/commit/protection failure rolls ownership back. Allocation-free target
admission rejects missing/zero metered budgets before allocating the instance
or copying input. The capped allocator then covers metadata parsing.
`loadModule` checks parsed code, combined code/maximum-memory reservation, and
table admission before any executable or guest mapping. `instantiate` freezes
**every non-timing option**, including all four new caps; a changed option
poisons that attempt instead of permitting a staged bypass or retry. Only the
optional timing destination may differ.

Fuel decrements at every generated function entry and backward branch target.
Exhaustion traps before the block executes, including an allocation-free
infinite loop. Each checked export/start invocation receives a fresh budget.
One unit is a poll, **not one wasm instruction or one nanosecond**. Branch
selection, optimization and presets can change consumption. Fuel 1 traps at
entry; fuel 2 permits a one-entry leaf to finish. The last charged unit traps
before the block, so a budget of N permits N-1 continuing polls. Import callbacks
return normally before checked host exit unwinds; callbacks themselves are not
metered or preempted. An unmetered AOT
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

## Explicit matched sampler

The optional `native-jit-bench` target builds **two separately linked** static
x86_64 Linux executables: `wamr-native-jit-aot-compare` has no compiler and only
accepts `aot`; `wamr-native-jit-bench` embeds the compiler and requires `fast` or
`full`. Both use the exact same embedded deterministic source. The comparator
embeds a matching, build-time `wamrc --profile=unikraft-x86_64` artifact, never
compiles at runtime, and reports compile time as null. Tests inspect both
binaries' symbols as well as their behavior.

The shared sampler calls the checked native API with no imports. It measures
the complete in-process compile (JIT only), compiler subphases, staged load,
instantiate, explicit start, first call, and three repeated calls separately.
Each 2000-round call initializes its volatile memory **inside the timed call**.
A separately timed `grow(1)` after the first call must return two previous
pages; before/after memory samples prove growth from two to three committed
pages with a stable eight-page reservation. This is not the #1046 producer's
eight-domain snapshot replay or a claim about uninterrupted steady state.
Both modes have the same heap/code/reservation limits; JIT additionally has
100,000 polls per invocation. The original AOT artifact has no fuel guarantee.
The fixed workload's finite input/control flow and the host process timeout
bound this comparator experiment; do not generalize that to arbitrary AOT.

```sh
zig build native-jit-bench test-native-jit-bench -Doptimize=ReleaseSafe -j2
python3 -m scripts.native_jit_benchmark \
  --aot zig-out/bin/wamr-native-jit-aot-compare \
  --jit zig-out/bin/wamr-native-jit-bench \
  --output /path/to/new-correctness-capture
# On arm64 development add: --runner 'qemu-x86_64 -cpu max'
```

The helper always runs all three explicit modes, with a 120-second per-process
host timeout (configurable 1-600 seconds). It archives a nonce-bound request,
raw stdout/stderr and statuses, including failures/timeouts. Strict revalidation
checks raw hashes, request identity, actual matching wasm identities, exact
results, every phase, growth and complete teardown before writing a comparison.
The separate `WAMR_JIT_SAMPLE=` marker cannot be consumed as the existing
`WAMR_BENCH_RESULT=` protocol. `scripts/native_benchmark.py` remains strict v2
compiler-free AOT by default, unchanged.

Without independent evidence the capture is **correctness-only**, has no image
size value and makes no performance claim. Executable file-size growth,
requested caller allocation peak growth, compiler peak/retained allocations,
and logical page commitment are distinct fields. None is RSS, allocator
bookkeeping, physical residency, a bootable image, or guest RAM overhead.
The JIT output artifact stays allocated through sampling, then both it and the
instance are released; this retained copy is included in caller memory peaks.
QEMU timing values are diagnostic observations only.

The helper's opt-in measurement path additionally requires **all** of
`--receipt`, `--trusted-receipt-sha256`, `--aot-image`, and `--jit-image`, and
rejects emulation/non-x86_64/non-Linux execution before starting a producer.
It does not build, deploy, boot, or qualify any image. Receipt schema version 1,
kind `wamr-jit-independent-image-deployment-receipt`, is exact:

| Fields | Required binding |
| --- | --- |
| `issuer`, `source_commit`, `deployment_receipt_sha256` | Independent issuer token, source commit, deployment evidence digest |
| `os`, `arch`, `hardware_execution`, `platform` | Linux/x86_64, hardware attestation, existing native platform identity shape |
| `executables`, `images` | Separate `aot`/`jit` SHA256 and byte counts of actual executable and complete-image files |
| `wasm`, `aot_module` | Actual sampled source and comparator module SHA256/byte counts |
| `compiler_embedded`, `runtime_linkage` | Exactly `{"aot":false,"jit":true}` and `static` |
| `safety` | `bounds_checks`, `checked_imports`, `checked_traps`, `wx` all true |
| `lifecycle`, `allocator`, `page_policy` | Exact sampler lifecycle, independently identified caller allocator and page policy |

The trusted hash is SHA256 of canonical JSON (sorted keys, compact separators);
it must be supplied out of band, not emitted by the producer. Receipt hashes
bind attestations; they are **not hardware proof or independent verification
of an issuer's claims**. The operator must qualify the images and deployment
independently first. This repository supplies no such receipt. In particular
Unikraft measurements remain rejected until the actual image adapter/receipt
integration is qualified separately. No CoreMark JIT or cloud action is added.

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
`qemu-x86_64 -cpu max` only for correctness. The matched software path is present;
these checks do not close #1044's pending physical native/image qualification.
