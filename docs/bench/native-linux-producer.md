# Compiler-free Linux embedding producer

`native-bench-linux` builds `zig-out/bin/wamr-native-bench`, an **x86_64 Linux
SysV** consumer of trusted `.cwasm` bytes produced ahead of time by the matching
`wamrc --profile=unikraft-x86_64`. It imports `src/api/aot.zig` and the real
`src/wasi/native_aot.zig` adapter, not the hosted Harness, interpreter, JIT,
component engine, or compiler. Building the consumer does not build `wamrc`.
This is integration software for #1045/#1046, **not** Unikraft image or Azure
performance qualification. The native ELF/EFI application and image/boot/hardware
acceptance remain [cataggar/unikraft#156](https://github.com/cataggar/unikraft/issues/156).

**Measurement is currently fail-closed with `LifecycleProtocolPending`.** The
host-owned protocol must bind the same full-snapshot/reset-outside-timing policy
into both targets' requests, receipts and observed results before this producer
can publish steady-invocation measurements. A prose-only reset description is not
enough. No unknown v1 field or ordinary steady-state record is emitted as a
substitute. Correctness checks and private failure evidence remain runnable.

## Build and correctness checks

From the repository root, with Zig 0.16:

```sh
zig build native-bench-linux native-bench-fixtures -Doptimize=ReleaseFast -j2
zig build test-native-bench -Doptimize=ReleaseFast -j2

zig-out/bin/wamr-native-bench --check zig-out/native-bench/coremark.cwasm 100 3 coremark
zig-out/bin/wamr-native-bench --check zig-out/native-bench/coremark-nofp.cwasm 100 3 coremark-nofp
```

The fixture step invokes the matching external host compiler on both tracked
CoreMark files and the pinned `unroll4.wasm`/`iv_store.wasm` workloads. It also
builds genuine deterministic compute/memory and WASI callback regression modules.
It does not generate replacement CoreMark bytes. Its artifacts are correctness
inputs, not build/image qualification receipts.

On an ARM Linux development host, the tests use an already-installed
`qemu-x86_64 -cpu max`. For manual checks, prefix the executable with that same
command. QEMU results are **correctness-only**. The check command emits
`WAMR_NATIVE_CHECK_RESULT=`, never `WAMR_BENCH_RESULT=`. Host `native-capture`
rejects a check record as measurement while retaining its raw transcript.
No receipts, Azure names, image identities, or performance results are generated
by this mode.

The arguments are `--check ARTIFACT ITERATIONS TOTAL_INVOCATIONS WORKLOAD`.
An optional final `closing-clock`, `backwards-clock` or `report-oom` injects a
**correctness-only negative-test failure**, never a measurement capability.
These modes execute the real guest and verify that private capture retains its
actual terminal/output when the closing timer or later report allocation fails.
Iteration counts affect CoreMark only. Tests use 100 iterations and independently
check `seedcrc=e9f5`, `crclist=e714`, `crcmatrix=1fd7`, `crcstate=8e3a`, and
repeatable `crcfinal=988c` in both variants. These short runs intentionally retain
CoreMark's “Must execute for at least 10 secs”/“Errors detected” diagnostics.
The no-floating-point variant may print zero whole seconds. **CRC correctness
is not a valid throughput result**, and those diagnostics are never removed.

`test-native-bench` tests actual Linux page/clock callbacks, real WASI argv/env
copying and stdout/stderr, returned/proc_exit(0)/proc_exit(23)/trap outcomes,
partial-output failures, allocation rollback, staged lifecycle, memory reset,
both CoreMarks, and repeated no-import computation/memory checks. Its existing
Python unittest integration exercises both pinned no-import workloads and the
native validator using explicitly synthetic metadata around actual execution
payloads. That test envelope never becomes measurement evidence.
`test-native-bench-build` does not execute x86 code. CI executes the native suite
only on Linux x86_64; Windows/ARM hosted CI is not forced to execute native code.

## Real phase boundaries and repeat semantics

All timers call Linux `clock_gettime(CLOCK_MONOTONIC)` and report nanoseconds.
Resolution comes from `clock_getres`, not a hard-coded precision claim.
The WASI adapter exposes actual Linux realtime, monotonic, process-CPU and
thread-CPU clock readings and their individual resolutions. No fake ticking
clock is used for CoreMark. These clocks are not guest-instruction counters.

* **Load:** `Instance.loadModule` allocates/copies and validates already resident
  `.cwasm` bytes, including the native contract/CPU checks. Disk acquisition and
  artifact hashing are excluded. This native format has no deferred relocation
  work. No guest/code mapping or import binding occurs in this stage.
* **Instantiate:** real WASI context/descriptor construction, import binding,
  memory/table/global/data initialization, code publication RW→RX, execution
  environment setup, and any module start section. The old `Instance.load`
  convenience function still performs both stages for existing embedders.
  A failed staged instantiation is not retryable: its owner must call `deinit`.
  Non-timing options are frozen at load and must match at instantiation; a later
  stage cannot silently change memory/table/CPU or subsequently added budget caps.
* **First invocation:** export lookup and `_start` call through the terminal
  returned/proc_exit/trap/host-error outcome, including output callbacks.
* **Steady invocations:** the **same instance and native code** are called again
  after a full initialized-state snapshot reset. This is not an assumption of
  libc command reentrancy and is not warm-libc-state reuse.

The reusable `src/bench/native_runner.zig` snapshots initialized linear memory,
globals, table contents/signatures and passive-segment state. Before each repeat,
`Instance.restoreMemory` revokes the grown suffix with `protect(.none)`, restores
the original bytes and logical bounds, and leaves the original reservation in
place. A later `memory.grow` commits and zeroes pages normally. Table-size changes
are rejected rather than silently approximated. A fresh WASI Context restores
descriptors, deferred output errors and sticky exit state; output buffers clear.
Exact repeated CoreMark CRCs and real memory-grow/reset regressions qualify this
specific reset path. Reset/snapshot costs are outside invocation timing; do not
describe these numbers as uninterrupted process-state or end-to-end steady-state
throughput. The snapshots consume memory, which the coverage declaration below
explicitly excludes.

The instance is made non-callable **before** revoking protection, and becomes
callable again only after the whole restoration succeeds. If protection fails,
the owner stays poisoned rather than assuming the provider left the mapping
unchanged. Calls, growth and reset retries are rejected;
only teardown is supported. The failed measurement omits memory snapshots instead
of reporting potentially stale commitment counts.

Every executed call is written to private stderr first as one
`WAMR_NATIVE_INVOCATION=` JSON line with ordinal/phase, actual terminal reason,
guest exit status, optional ticks, timing error, host/output diagnostic and
canonical base64 stdout/stderr. This streaming path uses fixed stack buffers and
does not allocate or copy output after execution. `Invocation` borrows the
Session's buffers until another invocation, reset or teardown; callers must
persist evidence before then.

A failed/backwards closing clock leaves `ticks: null` but preserves the true
terminal reason and bytes. Later report-allocation failure cannot erase the
already-written private record. If a canonical timed result cannot represent
that untimed call, the producer fails without fabricating a duration or silently
omitting its actual terminal from the private evidence. Existing host capture
saves stderr and its digest before validation. Transport/write failure can still
truncate a stream and must be handled as a failed capture, not a successful event.

Per-invocation stdout in a supported benchmark result is preserved as canonical
padded `stdout_base64` inside the single-line UTF-8 result JSON, including
non-UTF-8 bytes. Only `proc_exit` has an exit code; returned/trap/error records
use null. Nonzero WASI exit codes remain unsigned guest codes;
collector process return codes are separate.

## Qualified hardware deployment

Measurement follows the existing
[native benchmark request/result protocol](unikraft-native.md); it does not
create a new host schema, cloud pipeline, or image receipt. First obtain genuine
image/build/ABI qualification and create a host plan with sufficient common
CoreMark iterations (the existing protocol fixes both variants at 400000).
The producer performs no calibration or iteration override in measurement mode.
The setup below remains gated until the coordinated lifecycle schema is integrated.

The image integrator supplies a **private, independent deployment configuration**,
not values copied from the incoming request. Its fields are:

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `kind` | `wamr-linux-producer-deployment` |
| `qualification` | `hardware`; an explicit integrator attestation, not automatic proof |
| `producer_sha256` | Hash of the actual deployed `wamr-native-bench` ELF |
| `receipt_path` | Local qualified `wamr-native-image-receipt` file |
| `image_path` | Exact complete boot/deployment image, not `.text` or an unrelated binary |
| `runtime_path` | The same complete compiler-free producer ELF used here |
| `compiler_path` | Exact external matching `wamrc` used ahead of time |
| `aot_paths` | Workload-name → local `.cwasm` file |
| `wasm_paths` | Workload-name → original tracked `.wasm` file |
| `environment` | Empty array for these matched pinned measurements |

All paths are relative to the producer's working directory unless absolute.
The host sets `WAMR_BENCH_REQUEST` and `WAMR_BENCH_CONFIG_SHA256`:

```sh
python3 scripts/bench_coremark.py native-capture \
  --manifest PRIVATE_PLAN --run-id RUN_ID --out NEW_PRIVATE_CAPTURE --timeout 900 \
  -- zig-out/bin/wamr-native-bench --deployment PRIVATE_DEPLOYMENT_CONFIG
```

The consumer strictly parses the request, recomputes its canonical digest, compares
the independent receipt with the request, and hashes the actual image, runtime,
compiler, AOT and wasm files. Complete image/compiler files are hashed as streams,
not loaded into the timed runtime or limited to ELF sections. `/proc/self/exe`
must hash to both the deployment's producer identity and the receipt's runtime.
Thus Linux's runtime-size observation is the **whole linked compiler-free consumer
ELF**, including Linux integration code, not an isolated archive/text-size claim.
The receipt's optimize mode must also match the executing binary.

The receipt must declare `runtime_linkage: static` and all required safety
properties enabled. Its `memory_policy` must describe this actual implementation:
`allocator_by_phase.load` and `.instantiate` are `counted-zig016-debug-and-arena`;
`.first` and `.steady` are `counted-zig016-debug-output`. The single-threaded, libc-free
Zig 0.16 process supplies DebugAllocator in all optimization modes; loader/
instance arenas use that backing allocator and invocation output callbacks use
it directly. `page_policy.reservation` is `linux-mmap-prot-none`,
`.commitment` is `linux-mprotect-rw-zero`, and `.release` is
`linux-munmap-full`. These tokens are checked independently, not copied from the
request. Allocator policy identity is not by itself a measured allocation-byte count.

CPU model is observed from `/proc/cpuinfo`'s first `model name`; each character
outside `[A-Za-z0-9_.:+-]` becomes `_`. Active guest CPU count is read from Linux's
`/sys/devices/system/cpu/online` ranges, not affinity, SKU capacity or configured
offline CPUs. Both must match the independent receipt. A missing x86 Linux
identity fails closed, which also excludes the supported ARM-host QEMU check
environment from measurement mode.

**Identity provenance matters:** CPU/online-count and loaded-file hashes are
observations; Azure SKU/region, source/compile-option lineage, configured VM RAM
and the relationship between the complete image and this boot are **qualified
receipt attestations**. They cannot be independently discovered from arbitrary
request strings or proved by hashing a local image file. Protocol v1's `observed`
object consolidates those two sources; it must not be described as guest discovery
of Azure placement or proof of physical hardware. Review deployment/receipt origin
privately. No programmatic receipt or hardware-certification bypass is provided.
An edited `"hardware"` label is not valid qualification.

The private build evidence must bind the source commit, complete actual source
input digest, tracked-diff digest, Zig version, compiler binary and full commands.
Build from a clean committed tree when possible; do not silently attribute dirty
or untracked producer changes to the base commit. No cloud credentials or private
raw logs are needed or published by this software.

## Memory meaning and remaining qualification

The record uses `coverage: partial-guest`,
`method: linux-mmap-accessible-ranges`. Native callbacks count real reserved code
and linear-memory address ranges and their currently RW/RX-accessible committed
prefixes at after-instantiation, after-first and after-steady boundaries.
Revoking a grown suffix reduces this accessible count; retained physical pages
and Linux overcommit are not inferred from `mprotect`.

This is **not RSS, whole-process committed RAM, physical residency or complete
guest memory usage**. Runtime/snapshot allocator allocations, tables/globals,
producer buffers, stacks, kernel, other processes and image RAM are explicitly
omitted. Configured VM RAM is receipt-attested and never substituted for measured
memory. The full producer ELF and complete image byte sizes remain separate
artifact identities. Whole-guest allocation/page/residency accounting and Unikraft
`ukalloc`/`uk_vma` qualification require the native image integration; no missing
values are invented here.

Separately, a real counting allocator records successful alloc/resize/remap/free
requests for the Session, native API, snapshots and invocation-output buffers.
Private stderr includes `WAMR_NATIVE_ALLOCATOR stage=... method=caller-requested-bytes
live=... peak=...` after instantiation, every invocation and final release. `peak`
is cumulative for that attempt; final `live` must be zero. These counts exclude
input/request/JSON transport allocations and allocator bookkeeping/backing-page
slack. They are not combined with mmap bytes, which would double-count or mix
incompatible meanings. Protocol v1 has no allocation-byte sample field: its
canonical memory receipt stays explicitly partial; these supplementary diagnostic
counts are retained privately by capture, not promoted into a paired public
allocation result. The reusable counter and actual release are regression-tested.

No real Linux/Azure throughput, paired Linux–Unikraft performance, native Unikraft
boot, or EEMBC compliance claim follows from passing these software tests.
