# Optional native in-process compiler

This is **dependent software integration for #1044**, not native Unikraft JIT
qualification. It depends on the corrected native AOT API from #1050 and the
staged producer API from #1046. Development integration uses producer base
`7b0b7701eb882cf36ba02455a00b9dd64823cff6`; this is not a protected-main merge
or deployment qualification receipt. Physical image/adapter qualification
and boot evidence remain pending in `cataggar/unikraft#156`. The freestanding
sampler and external native evidence import described below are application
building blocks, not a qualified image or deployment. Linux tests and QEMU
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
`lib/libwamr-jit.a`. It targets `x86_64-freestanding-none`, SysV, position-independent, single-threaded,
no red zone, libc, stack protector, unwind tables, or Zig error tracing.
This is an application-object contract, not an ISR ABI.

The public optional-compiler interface is **Zig**, not a new C ABI. Existing
compiler-free C entry points remain in `wamr_aot.h`; they cannot request the
new metered artifact's mandatory budgets and therefore reject it. The
`wamr_jit_*_link_check` symbols retain real compile and load/start/call paths in
the freestanding link audit, including complete sampling and serial formatting.
They are not application entry points. Do not
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

## Freestanding image-application sampler

The native JIT profile additionally exports:

| Zig module | Interface/artifact |
| --- | --- |
| `wamr-jit` | `sample.run` and `sample.Capture`, in `libwamr-jit.a` |
| `wamr-jit-aot-sample` | Separately linked, compiler-free `sample.run` and `sample.Capture`, in `libwamr-jit-aot-sample.a` |
| `wamr-jit-workload` | `wasm`, the embedded deterministic workload also installed as `native-jit-bench/matched.wasm` |

`native-jit-check` links both complete freestanding sampler/serialization paths,
not just unused declarations, and separate downstream-style consumers of the
public modules with the embedded workload. Native objects explicitly use PIC and
all four audits link as PIE, including the compiler-free comparator, so EFI image
links do not depend on fixed-address 32-bit absolute relocations. These are link
audits, not EFI boot qualification. The two archive audits link the actual static
libraries with compiler-runtime fallback disabled; the other two cover public
module composition. The sampler helpers import neither Linux pages
nor `std.process`; the Linux drivers now call these same helpers. Image
applications supply a caller allocator, native `aot.Platform` (including its
monotonic clock), actual clock resolution, request SHA256, and serial writer.
No console, process, image launcher, or boot entry is supplied. Downstream
application roots using compiler logging can reuse `wamr-jit.std_options` (the
freestanding no-hosted-logging configuration).

```zig
const native = @import("wamr-jit");
const workload = @import("wamr-jit-workload");
pub const std_options = native.std_options;

// Inside the embedding application, with qualified native capabilities:
var capture: native.sample.Capture = .{};
native.sample.run(allocator, platform, workload.wasm, .fast,
    request_sha256, clock_resolution_ns, &capture) catch {};
try capture.writeRecord(serial_writer);
if (capture.report.failure != null) return error.SampleFailed;
```

Use `.full` explicitly for the other JIT preset. For the separate comparator
application, import **only** `wamr-jit-aot-sample` and `wamr-jit-workload`, and call
`sample.run(allocator, platform, workload.wasm, @embedFile("matched.cwasm"),
request_sha256, clock_resolution_ns, &capture)`. Generate that matching AOT
artifact with the hosted `native-jit-bench` build above; it installs both source
and compiled bytes. The compiler-free module is exported by the opt-in sampler
profile for composition, but does not import or link the compiler module/library.
The independent image receipt must bind the actual embedded source and artifact;
the comparator does not compile to check their relationship.

`Capture` must stay at a stable address until serialization completes: report
identity slices point into its own fixed arrays. No borrowed input or compiler
allocation survives `run`. Keep the same object in place, rather than returning
it by value. `run` initializes even failure reports, releases the instance and
retained compiler artifact before returning, and records requested allocation
peaks and reservation teardown through wrappers around the caller's actual
capabilities. Infallible `Platform.unmap` remains an adapter obligation; the
wrapper's zero reservation count is **not** independent physical-release proof.
The serial writer receives at most 8192 bytes, including prefix and newline.
Propagate write errors and sampler failures to the adapter's capture status.

The fixed sampler caps, verification mode, workload, four invocations, and real
two-to-three-page growth are unchanged. The native stack provisioning,
non-preempted allocator/page/import callbacks, cooperative compiler polling and
fuel-not-time limitations above apply without relaxation.

## Explicit external native evidence transport

Linux capture and its version-1 receipt are unchanged: Linux execution cannot be
relabelled Unikraft. A **separate**, host-OS/architecture-independent path accepts
an independently qualified downstream Unikraft adapter's evidence. This path
never invokes an adapter, starts a process, builds/deploys/boots an image, or
dispatches cloud work. It has no synthetic/correctness-to-measurement bypass.
Correctness-only Linux/QEMU observations remain correctness-only.

The integrator must independently qualify its adapter, images and deployment,
and supply a trusted hash of its canonical version-2 image receipt. The helper
hashes the **actual files** for both application executables and complete images;
an executable cannot stand in for a complete image. Prepare the challenge:

```sh
python3 -m scripts.native_jit_benchmark \
  --native-prepare --aot /private/aot-app.elf --jit /private/jit-app.elf \
  --aot-image /private/aot-complete-image --jit-image /private/jit-complete-image \
  --receipt /private/independent-image-receipt.json \
  --trusted-receipt-sha256 "$INDEPENDENT_IMAGE_RECEIPT_SHA256" \
  --output /private/new-native-capture
```

This writes only `request.json`, binding a fresh nonce, 24-hour validity window,
file identities, and the independent receipt. The downstream adapter must pass
the SHA256 of those **exact request bytes**, plus the explicit mode, into each
already-built image. Do not embed the nonce into an image after hashing it:
that changes its identity and creates a request/image hash cycle.

The adapter separately supplies `capture.json`, `aot.serial`, `fast.serial`,
and `full.serial`. An independently trusted hash of canonical `capture.json`
is required **in addition** to the image receipt hash. Import with the same
image/executable files and image receipt arguments:

```sh
python3 -m scripts.native_jit_benchmark \
  --native-import /private/adapter-output \
  --aot /private/aot-app.elf --jit /private/jit-app.elf \
  --aot-image /private/aot-complete-image --jit-image /private/jit-complete-image \
  --receipt /private/independent-image-receipt.json \
  --trusted-receipt-sha256 "$INDEPENDENT_IMAGE_RECEIPT_SHA256" \
  --trusted-capture-sha256 "$INDEPENDENT_CAPTURE_RECEIPT_SHA256" \
  --output /private/new-native-capture
```

This does not require a Linux/x86_64 **collector**. The receipt still requires
qualified x86_64 **Unikraft execution** on hardware, exact native ABI/options,
static linkage, safety protections, and a compiler-free AOT comparator.

### Exact native receipt contracts

Image receipt version 2 retains kind
`wamr-jit-independent-image-deployment-receipt`. All fields are required;
unknown fields are rejected. No sample successful deployment receipt is supplied.

| Fields | Required binding |
| --- | --- |
| `evidence_kind`, `os`, `arch`, `hardware_execution` | `measurement`, `unikraft`, `x86_64`, true |
| `issuer`, `source_commit`, `deployment_receipt_sha256` | Independent issuer token, exact WAMR commit and prior independent deployment/qualification evidence digest |
| `platform`, `executables`, `images`, `wasm`, `aot_module` | Existing platform identity shape and SHA256/positive byte counts of actual qualified artifacts |
| `compiler_embedded`, `runtime_linkage`, `safety`, `lifecycle`, `allocator`, `page_policy` | Same strict bindings as version 1, but describing the actual native image implementation |
| `target`, `options` | Exactly `NATIVE_TARGET` and `NATIVE_OPTIONS` in `scripts/native_jit_benchmark.py`: native contract/subprofiles, both presets, after-each-pass verification, every compiler/runtime cap and fixed workload lifecycle |
| `build_options` | One shared `optimize` (`Debug`, `ReleaseSafe`, `ReleaseFast`, or `ReleaseSmall`) for both images, plus exactly `NATIVE_BUILD_OPTIONS`: PIC, single-threaded, no red zone/libc/stack checker/stack protector/unwind tables/error tracing |
| `adapter` | `name`, `source_commit`, `qualification_receipt_sha256`; independently qualified native adapter, not a Linux adapter name |
| `clock` | `method` token, positive `resolution_ns`, `scope` exactly `guest-monotonic-execution` |
| `native_stack` | `generated_frames_bytes` at least 256 KiB, positive separately qualified `compiler_embedder_callbacks_bytes`, and `provisioned_bytes` covering their sum |
| `memory_observer` | Exact fields described below; actual physical backing observations, not sampler allocation requests |

The external capture receipt has schema version 1, kind
`wamr-jit-native-capture-receipt`, `evidence_kind: measurement`,
`hardware_execution: true`, `request_sha256`, `image_receipt_sha256`,
`adapter_qualification_sha256`, and `records` containing exactly `aot`, `fast`,
`full`. Every record contains:

* `image` and `executable`: the exact corresponding file hash/byte-count objects;
* `serial`: hash/byte count of the entire raw serial file;
* `outcome: success`, `capture_complete: true`: attestation of the complete run,
  not merely receipt of a success-looking line before a later guest failure;
* `started_at`, `completed_at`: UTC capture window within the request validity
  interval. These timestamps are never converted to guest execution durations;
* `memory`: nonnegative `before_bytes`, `observed_max_bytes`,
  `after_teardown_bytes`, plus `observation_count` of at least two. The observed
  maximum must be positive and at least the other values.

`memory_observer` declares a public `method` token, `quantity` exactly
`physical-backing-bytes`, `coverage`, `excludes`, `sampling`, and `interval_ns`.
Coverage is either `whole-guest` (excludes `["hypervisor"]`) or
`caller-allocator-and-native-pages` (excludes
`["image","native-stack","other-kernel-allocations","page-tables"]`). The latter
must include real backing for **all** caller allocator traffic, including
compiler scratch/result storage, and native code/linear-memory pages. It is not
whole-guest overhead. Sampling is `continuous-high-water`, `periodic`, or
`phase-boundaries`; only periodic sampling supplies a positive `interval_ns`,
otherwise null. Boundary/periodic maxima are observed maxima, not a guaranteed
true peak. Independent qualification must substantiate the declared method and
coverage. Requested bytes or logical page commitment cannot supply these values.

Reports retain complete image and executable identities/sizes, native observations
and their method/coverage separately from compiler peak/retained allocation,
caller requested peaks and logical memory growth. No serial-arrival or collector
latency becomes a guest duration or compiler phase. No RSS or whole-image RAM
claim is inferred from partial observation coverage.

Import archives use a private 0700 directory and 0600 files. Boot noise and
non-UTF8 bytes remain opaque; exactly one line-starting, newline-terminated,
bounded UTF8 `WAMR_JIT_SAMPLE=` record is allowed per raw stream. Duplicate
markers, truncated/oversized records, duplicate JSON keys, failures, bad results,
missing memory evidence, stale request/artifact identities and altered raw bytes
reject publication. Raw imports are bounded to 16 MiB per serial file and 128 KiB
for the capture receipt; an oversized input retains at most the bound plus one
byte with a failed import status. Failed attempts retain private evidence and
cannot be retried in place; prepare a fresh challenge. Public comparisons do not
include boot logs. Both trusted hashes pin **supplied attestations**, not proof
generated by this tool. Synthetic fixtures exist only inside unit tests and are
explicitly labelled as not deployment evidence.

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
