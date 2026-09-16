# Freestanding native AOT guest producer

The `unikraft-aot` profile exports **`@import("wamr-aot").benchmark`** and
**`.runner`**, in addition to its existing `.aot` API. This is a reusable Zig
integration boundary for #1045/#1046, not an Unikraft application or qualified
image. Actual native pages, clocks, transport, image assembly and deployment
acceptance remain [cataggar/unikraft#156](https://github.com/cataggar/unikraft/issues/156).
No Linux adapter, filesystem, environment variable, Python, subprocess, compiler,
interpreter, JIT, thread manager or hosted runtime is imported by the producer.

## Build and import

```sh
zig build -Dprofile=unikraft-aot -Doptimize=ReleaseSafe -j2
zig build -Dprofile=unikraft-aot native-aot-guest-check \
  -Doptimize=ReleaseFast -j2

# Hosted correctness tests using matching, actually emitted native bytes:
zig build test-native-aot-guest -Doptimize=ReleaseSafe -j2
zig build test-native-bench-unit -Doptimize=ReleaseSafe -j2
```

`native-aot-guest-check` retains exported wrappers reaching `benchmark.run`,
Session setup/invocation/reset, all error paths and streaming evidence in an
`x86_64-freestanding-none` ELF. The default native install also depends on this
audit, alongside the existing complete C API audit. Neither audit ELF is bootable
or executable as an application. Neither links libc or resolves hosted syscalls.
The compiler remains external; building this profile never constructs `wamrc`.

Downstream, use the existing `wamr-aot` dependency module:

```zig
const native = @import("wamr-aot");
const bench = native.benchmark;
// native.aot == bench.aot == native.runner.aot
// bench.wasi is the same minimal-wasi module used by the native import adapter.
```

Add `dependency.module("wamr-aot")` to the application's imports using the native
profile/target. Do **not** separately import copies of `src/api/aot.zig`,
`native_runner.zig` or WASI files into another module: that creates distinct Zig
types. No Linux source or duplicated Platform/Context declarations are needed.
The optional Zig producer is not a new C ABI; existing C library users are unchanged.

## One bounded request, explicit capabilities

`bench.run(options: bench.Options) !void` consumes exactly the existing
[`native-request` v2 JSON](unikraft-native.md), plus its separately delivered
canonical `config_sha256`. It does not introduce another request/result schema.
Normal return means the result was transmitted, not that its `outcome` succeeded.
Pass these **caller-owned**, lifetime-stable values:

| Option | Required input |
|---|---|
| `request_json`, `config_sha256` | Host request bytes and control-channel digest; no path or environment lookup. |
| `build` | Trusted independent `Build` facts: source, pre-image runtime artifact, external compiler, ABI and actual allocator/page policy. |
| `deployment` | Independently transported receipt JSON, complete-image identity, placement/RAM attestations and qualification state. Never populate it from `request_json`. |
| `workloads` | Embedded `Workload { name, wasm, aot_bytes }` slices from the tracked fixture and matching external compiler. |
| `cpu` | Actual guest CPU model token and active CPU count from the platform, not SKU capacity, affinity alone or the incoming request. |
| `native` | The existing `aot.Platform`: real reserve/eager commit/protect/unmap/monotonic callbacks. |
| `clock` | Actual monotonic source/resolution and `bench.wasi.Clock`; unsupported WASI clock IDs have zero resolution. |
| `pages` | Callback-accounted code/linear-memory `Footprint`, with a precise public method token. |
| `allocator` | Caller allocator for Session/runtime/snapshot/output; no implicit backing allocator. |
| `workspace` | Fixed, disjoint scratch/report storage; released for reuse after `run` returns. |
| `result`, `evidence` | `*std.Io.Writer` transports. Evidence must not allocate through the Session allocator and must report write/flush failures. |

The two writers may share a synchronous serial transport. Keep their buffers
disjoint from inputs, workspace and guest memory. All callback contexts and
borrowed byte slices must outlive the call. Do not reenter the runtime from a
callback. The integrator can pass a bounded/fallible native allocator; it remains
responsible for real allocation, W^X and transactional page ownership.

For example, after the integrator has obtained these real inputs:

```zig
try bench.run(.{
    .request_json = request_bytes,
    .config_sha256 = control_channel_digest,
    .build = independently_generated_build,
    .deployment = independent_deployment,
    .workloads = embedded_workloads,
    .cpu = observed_cpu,
    .native = native_platform,
    .clock = actual_clock_capability,
    .pages = retained_page_accounting,
    .allocator = caller_allocator,
    .workspace = report_workspace,
    .result = serial_result_writer,
    .evidence = private_evidence_writer,
});
```

Defaults bound each request/receipt to 256 KiB, JSON nesting to 24, total
stdout+stderr to 1 MiB per setup/invocation, and repeats to 10,000. The caller may
lower the byte/repeat limits. JSON parsing, canonical hashing, reconciliation and
retained invocation reports all use the fixed workspace; there is no fallback
allocator. Its required size depends on receipt size, repeat count and output.
Runtime allocations use the separate caller allocator and existing AOT limits.
Report-budget exhaustion fails capture; it does not silently drop an invocation.
The JSON allocating writer may expose workspace exhaustion as `WriteFailed`.

Only compiler-free AOT, the explicit native compile profile, v2 phase contract,
and exact eight-domain snapshot lifecycle are admitted. Requests cannot choose
another export, argv, environment, AOT file or workload bytes. Both CoreMarks use
`_start` and argv `WORKLOAD 0 0 0 400000 0`; `compute` and `memory` use the tracked
`unroll4.wasm`/`iv_store.wasm`, `_start`, and no arguments after argv[0].
Environment is empty. There is no short-run/calibration measurement override.

## Identity without a circular image digest

Admission hashes the actual embedded Wasm/AOT bytes, checks the pinned Wasm
digests, checks AOT bytes/size against the independently supplied receipt, and
reconciles the receipt, complete-image attestation, build facts, observed CPU,
actual optimization/safety settings and request. A changed request cannot become
an observation merely by changing its config hash.

`Identity.fromBytes(bytes, *[64]u8)` hashes actual bytes without allocating.
Build tooling can use it (or its ordinary SHA-256 implementation) to establish
the exact external compiler and **complete linked pre-image runtime archive or
consumer object**, including the producer code actually used by the application.
Do not hash a separately rebuilt C-only archive if the guest instead imports and
links different Zig consumer objects. Source `tree_sha256` must hash the actual canonical source-input set,
including dirty/untracked inputs used by the build; record the recipe, source
commit, tracked-diff digest, Zig version and complete build/compile commands
in private build evidence. These are trusted **build-time byte observations and
lineage attestations**, not guest discovery of files that do not exist at boot.
The guest reconciles them independently of the request; it does not reproduce
the source build or prove that a caller supplied truthful facts.

Neither the runtime object nor the final image may contain its own digest.
Compile the actual consumer object with build metadata supplied separately, hash
that object, bind its identity/source facts via a separate manifest data object
at final image assembly, then hash the **whole completed image**
externally. Supply that identity and its boot/deployment relationship over the
independent trusted deployment channel. The complete-image receipt must not be
embedded back into the image it hashes. No `.text`-only hash, fabricated receipt,
request echo, source ref alone or unrelated runtime artifact is sufficient.

`Deployment.qualification` defaults to `correctness_only`, which emits
`WAMR_NATIVE_CHECK_RESULT=`. The host measurement importer rejects that prefix.
It carries the v2-shaped payload solely for software validation, not a newly
accepted measurement evidence kind. Set `independently_qualified_hardware` only
when the downstream integrator has genuine reviewed image/boot/hardware evidence;
this selects the existing `WAMR_BENCH_RESULT=` transport. The enum is **not a
qualification procedure**. CPU observations and hashes cannot prove Azure
placement, physical hardware, complete-image boot identity or configured RAM.
No local/QEMU test, successful link or edited label establishes those facts.

## Exact lifecycle and failure evidence

The shared Session genuinely stages `loadModule`, `instantiate` and module start.
Instantiation includes WASI binding/context creation and module start; artifact
acquisition/hashing and report transport are outside phase timing. Snapshot
allocation/copy has its own `lifecycle_setup_ticks`. Each reset and each `_start`
call has separate opening/closing native clock reads. No compilation occurs.

The captured guest path persists module-start terminal and exact output privately
before teardown or clearing. Successful module-start descriptor changes become
the WASI reset baseline. Start output does not leak into first-invocation output,
and module start is not rerun. The first/repeated invocation stdout is preserved
as canonical base64; private evidence additionally retains stderr and diagnostics.
`proc_exit` keeps the entire unsigned 32-bit status; returned/trap/API/host error
are distinct. Only proc_exit has an exit code.
Version 2 has no separate module-start event field; module-start terminal/output
is retained in private evidence, not misreported as a first invocation.

Every executed call and attempted reset is flushed privately before report
allocation or state reuse. A closing failure, backwards reading or saturated
`UINT64_MAX` clock retains the actual executed terminal/output with null duration.
An opening failure creates no invocation. A completed reset followed by an opening
clock failure remains a final, uncalled reset event. Failed reset protection
poisons the instance and suppresses unreliable memory observations.

Pending output errors prevent reset before state mutation; neither context
replacement nor output clearing can erase them. Truncated output retains its
known prefix and completeness/error flags. OOM, missing memory observation or
transport failure never produces successful placeholder values. On a returned
error, the caller must mark capture failed and preserve any already flushed
private records. Do not silently retry and overwrite the failed attempt.

Reset restores the same instance's globals, output, linear access protection,
contents and logical size, passive-segment drop state, table entries/signatures
and WASI context. This is **snapshot replay**, not warm steady-state execution.
The host reports reset cost and snapshot call-only distributions separately;
`steady_state_seconds` stays null for this lifecycle.

## Memory coverage and validation limits

`Pages.sample` returns **reserved virtual-address bytes** and **retained committed
backing bytes** for live native code/linear mappings. Snapshot access revocation
does not itself decommit backing; providers must not reduce retained commitment
merely because logical `memory.size` shrank. Infallible unmap releases ownership.
The provider's method must describe its real accounting, not inferred RSS.

The v2 memory record is deliberately `partial-guest`. Runtime/snapshot allocator
storage, tables/globals, producer buffers, guest stacks, kernel, image and other
allocations are explicit omissions. Configured VM RAM remains a deployment
attestation, never a memory sample. A separate actual counting allocator emits
private `caller-requested-bytes` live/peak diagnostics, with zero live bytes after
release. These exclude allocator bookkeeping/slack and report/input storage,
and are not added to committed/reserved page counters. Protocol v2 has no
canonical requested-allocation sample field.

Tests exercise actual emitted native compute/memory, module-start output and
descriptor state, full-u32 exit, traps, known binary output prefixes, reset
domains, clock faults/overflow, every runtime allocator failure, late report OOM,
and transport failure. Existing short CoreMark CRC checks remain correctness-only.
`test-native-aot-guest-protocol` additionally feeds real execution payloads under
explicitly synthetic test-only metadata through the **unchanged Python v2
validator**, including failed/untimed calls and uncalled resets. Linux behavior is
covered by the existing producer tests. None supplies native Unikraft boot,
Azure performance, whole-guest RAM, paired campaign or EEMBC qualification.
