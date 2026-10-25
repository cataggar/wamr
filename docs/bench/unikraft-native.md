# Matched Linux / native Unikraft AOT evidence

This is the **host-side software portion** of
[#1046](https://github.com/cataggar/wamr/issues/1046), not a performance
result or completion claim. [cataggar/unikraft#156](https://github.com/cataggar/unikraft/issues/156)
completed the initial source-pinned, compiler-free tiny-AOT image integration,
direct Azure Gen2 execution and ownership-checked cleanup, and supplies
[baseline image and memory observations](https://github.com/cataggar/unikraft/issues/156#issuecomment-5722701992).
That baseline does not satisfy #1046's matched workloads, phase timing,
peak-memory coverage or paired Linux/Unikraft comparison. The earlier
[cataggar/unikraft#88](https://github.com/cataggar/unikraft/issues/88)
pipeline may be reused externally; these commands do not create cloud
resources, dispatch jobs, download serial logs, or publish evidence.

The existing `scripts/bench_coremark.py` entry point provides `native-plan`,
`native-request`, `native-capture`, `native-import`, and `native-report`, implemented
by its `native_benchmark.py` adapter. Its existing hosted benchmarks are unchanged.
Python runs **only on the external host**. Neither that hosted CLI nor Python is
assumed to exist inside Unikraft.

## What is runnable now, and what is not

The host can verify local artifact bytes and image receipts, generate a counterbalanced
plan, export per-run requests, launch a supplied **Linux embedding producer**, import
native serial evidence, validate every result, and produce a paired JSON report.
The Python adapter does not implement an embedding runner or obtain native runtime
phase timers by timing a subprocess. The separate
[compiler-free Linux embedding producer](native-linux-producer.md) supplies real
API phase clocks and qualified same-instance snapshot-reset execution. Native
Unikraft image integration and both targets' hardware/deployment qualification
remain necessary before real paired measurements can be collected.
Linux measurement binds the explicit version 2 snapshot-replay policy to requests,
receipts and results, with separate snapshot setup/reset costs and retained partial
failure evidence. It rejects warm-instance or version 1 declarations for this path.
Standalone parser/capture tests do not depend on those APIs.

The optional [freestanding AOT guest producer](native-guest-producer.md) now
exposes that Session lifecycle through `wamr-aot.benchmark`, consumes the same v2
requests without host OS services, and emits bounded results/evidence through
caller-owned transports. Actual Unikraft pages/clocks, independent build/image/
deployment attestations and hardware/campaign qualification remain downstream.

Concrete integration dependencies identified during the native API handoff:

* A combined `Instance.load(...)` outer-call duration cannot honestly populate
  separate `load_ticks` and `instantiate_ticks`. The producer must use genuine
  staging/internal phase instrumentation and check completion indicators.
  Duplicating an outer duration or inventing zero instantiation is not an adapter.
* The standalone `src/wasi/minimal.zig` context supplies imports, bytes and tagged
  terminal outcomes, not a native executable, load/instantiate timers or memory
  snapshots. Those belong to the native embedding producer.
* WASI exit state is deliberately sticky. Warm continuation needs actual restart
  qualification; snapshot replay instead needs a complete qualified restoration
  boundary. Version 2 distinguishes those policies and records excluded reset cost.
  Repeated CRC success alone does not prove all state/resource lifecycles are safe.

Neither an API build nor context/import fixture tests establish native CoreMark CRC
success, valid clock execution or benchmark measurements. These dependencies must
be resolved by the native integration before this host adapter can accept a complete
real campaign.

The [Unikraft architecture guidance](https://unikraft.org/docs/internals/architecture)
informs this boundary: use statically linked, compiler-free AOT and narrow native/WASI
modules, not POSIX, filesystem or scheduler dependencies added merely for compatibility.
Run-to-completion and phase-specific allocator selection are possible levers, not
measured advantages. Keep the caller allocator separate from reserve/commit page
policy; avoid steady-call allocations only where the qualified lifecycle permits.
The pinned native Zig build is authoritative, not generic Make/Kconfig examples.
All kernel components share a protection domain: bounds checks, W^X, import checks
and trap isolation remain mandatory. No network/storage scope is introduced here.

The unit tests use explicitly `synthetic` artifacts, clocks, outputs, CPU descriptions
and receipts. A test-only Python keyword enables them; **no CLI option accepts
synthetic measurement evidence**. Relabeling test data as real is not evidence.
Receipts are trusted attestations supplied by the image/guest integrator, not digital
signatures or an automatic proof of their truth. Review their origin privately.

## Pinned workloads

No fixtures are regenerated. `native-plan` hashes the tracked bytes:

| Key | Tracked path | SHA-256 |
|---|---|---|
| `coremark` | `tests/benchmarks/coremark/coremark_wasi.wasm` | `f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b` |
| `coremark-nofp` | `tests/benchmarks/coremark/coremark_wasi_nofp.wasm` | `24c0cc1bd52b641cf9e8ae74d1be188cba38d74cdb7ac18378de47382aab9541` |
| `compute` | `tests/benchmarks/loop-passes/unroll4.wasm` | `6870b3373e4098117c82b6736d0ca7cbcc7d8d747fe87ca5b7d1ebf0e4d12890` |
| `memory` | `tests/benchmarks/loop-passes/iv_store.wasm` | `b1979dd330c14d5f898b8a7c8c313e58f6521ad6eb7db9a6e7d7e78788ac6e72` |

Both CoreMark variants always call `_start` with guest arguments
`0 0 0 400000 0`, independent of profile. No calibration or per-target iteration
override is allowed. The pinned no-import fixtures take no arguments and trap on
incorrect computation/memory contents; successful return without output is their
correctness contract. They are focused microbenchmarks, not whole-system workloads.
Select all four for the full intended workload coverage; subsets remain explicitly
listed in the report and cannot establish coverage of omitted workloads.

Each CoreMark invocation must have exactly one 2K performance marker, fixed iteration
field, positive throughput, success marker, positive total ticks and total seconds,
`seedcrc=e9f5`, `crclist=e714`, `crcmatrix=1fd7`, `crcstate=8e3a`, and one hexadecimal
`crcfinal`. The final CRC must agree across all invocations of the **same pinned
variant**. First and steady invocations are checked separately. Error/ambiguous output
is rejected even when the process exits zero.

Both pinned fixtures implement `time_in_secs` using **1,000 ticks per second**.
For `coremark`, seconds and throughput use the fixture's `%f` six-decimal formatting;
seconds must equal ticks / 1000 and throughput must equal iterations / that duration
within half a printed decimal unit plus floating-point representation tolerance.
For `coremark-nofp`, printed seconds truncate `ticks / 1000` to an integer, and
throughput uses integer division by those truncated seconds. The integer fields must
match exactly; rounded-up seconds or floating-point formatting are not interchangeable.

The **untruncated tick-derived duration**, not the potentially truncated display
value, must fit inside the enclosing guest invocation. The permitted measurement
quantization is one inner millisecond tick plus the declared enclosing clock
resolution (and floating-point representation tolerance). For example, 20,999 ticks
may print 20 seconds in nofp, but cannot fit inside a 20.1-second invocation.
Reports retain the raw ticks, their 1,000 Hz frequency, tick-derived seconds,
displayed seconds and formatting variant separately.

The independent minimum-duration check requires reported CoreMark time, tick-derived
time and guest invocation time each to be at least ten seconds. `authoritative` is only a sample-count
profile (2 warmups, 10 measured runs); `ci` conventionally means 0 warmups, 3 runs.
Overridden counts remain visible. Neither profile, CRC success, nor minimum duration
certifies compliance with all EEMBC source, porting, workload and publication rules.
Reports explicitly say `coremark_compliance: "not-certified"`.

## Plan input and image receipts

Run from the repository root. Keep local artifacts under a private directory, for
example `.bench-coremark/native/`, which is already ignored. Commands never overwrite
an existing evidence output. Relative input artifact paths are relative to the
invoking host's working directory.

`native-plan --config` accepts a JSON object with **exactly** these fields:

* `profile`: `authoritative` or `ci`; `warmups`: nonnegative integer; `runs`: positive
  integer; `steady_invocations`: positive integer; `valid_hours`: 0.01–168 hours.
  Plans are limited to 10,000 total scheduled attempts and 10,000 steady invocations
  per attempt. A capture timeout must fit inside the remaining validity window.
* `workloads`: unique list of the keys above.
* `producer_source`: source identity of the host tooling; its commit must match the
  executing tooling checkout. The adapter also hashes the actual executing
  `bench_coremark.py`, `native_benchmark.py`, and shared `benchmark_schema.py`.
* `targets`: exactly `linux` and `unikraft`, each described below.
* `abi_proofs`: map from every selected workload to a local proof file path for
  identical AOT bytes, or JSON `null` for target-specific AOT bytes.

A **source identity** has exactly `commit` (40 lowercase hex), `tree_sha256` and
`tracked_diff_sha256` (64 lowercase hex). The tree digest must identify the actual
source input set used by the build, with its computation documented in the private
build evidence. A ref name is not an identity. Dirty source is not silently equated
with the base commit.

Each target configuration has:

* `runtime_path`: exact linked WAMR runtime artifact (the archive/object used in the
  native image, or the actual static compiler-free Linux embedding ELF executed,
  **not** an unrelated hosted `wamr` executable). The runtime ELF is not automatically
  a complete deployment image; their identities and byte counts remain separate.
* `compiler_path`, `compiler_version`: exact external `wamrc` executable and public
  version token. Both targets must use the same compiler bytes and source identity.
* `compile_profile`: `null` when the compiler's profile option was omitted, or the
  explicit `unikraft-x86_64` profile. This per-target compiler/ABI setting is not
  concealed in the common optimization/safety options. The native profile requires
  x86_64 and its strict native artifact contract; a normal hosted artifact is not
  interchangeable. Unikraft currently requires `unikraft-x86_64`. A Linux driver using the same isolated embedding API may use
  that native profile too, subject to actual ABI qualification.
* `source`: runtime/compiler source identity.
* `target_abi`: explicit public ABI token, e.g. an integration-defined ABI version.
* `platform`: exactly `arch` (`x86_64` or `aarch64`), `cpu_model`, `active_cpu_count`,
  `azure_sku`, `azure_region`. Strings are public tokens: letters, digits, `. _ : + -`,
  without spaces or resource paths. Use a stable CPU model token in both producers.
  The CPU count is **actually active guest CPUs**, not Azure SKU capacity, a pinning
  mask alone, host `os.cpu_count()`, or configured-but-offline CPUs.
* `options`: exactly `optimize` (`Debug`, `ReleaseSafe`, `ReleaseFast`, `ReleaseSmall`)
  and boolean `bounds_checks`, `wx_enforced`, `import_checks`, `trap_isolation`,
  `stack_checks`, `simd`, `threads`, `memory64`.
  All must describe the actual linked/runtime/AOT settings and match across targets.
  Bounds checks, W^X, import checks and trap isolation must all be `true`; this adapter
  rejects weakened isolation even if both targets declare the same weakened options.
* `memory_policy`: exactly `allocator_by_phase` and `page_policy`. The former maps
  `load`, `instantiate`, `first`, `steady` to public caller-allocator implementation/
  configuration tokens; the latter maps `reservation`, `commitment`, `release` to
  separate public page-policy tokens. These are configuration identities bound to
  the exact image receipt, not allocator byte counters or evidence of zero allocations.
  Policy choices may differ across targets and remain visible for experimental review.
* `execution_lifecycle`: the explicit common lifecycle/reset policy described below.
  It must match across Linux/Unikraft, match the exact-image receipt, and match every
  result. It is copied explicitly into each request and included in its config hash.
* `image_path`: exact **complete boot/deployment image file**, not ELF sections or
  stripped text size; `image_receipt_path`: its native integration receipt.
* `aot_paths`: map from every selected workload to the actual AOT artifact file.

Sources, compiler identities, options, architecture, CPU model, active CPU count,
SKU and region must match. Source/build settings or guest CPU counts that cannot
be established are pending evidence, not `"unknown"` measurements.
For measurement plans, both targets must use the currently supported native
Unikraft SDK WAMR source commit
`bb24e3e3f0ce5b998c0b24f6c06007d9a8ad7d20` (the Zig 0.17.0 compatibility pin
published by `cataggar/wamr#1081` and consumed by `cataggar/unikraft#270`
`support/apps/wamr-aot/build-tool-contract.zig`), or the preserved historical
Zig 0.16.0 pin `a53205d77be3b880eb8f8b96679512ba58e2331a` (the pin consumed by
cataggar/unikraft#163). Both targets and both compilers in a campaign must
still report the exact same commit; mixing the current Zig 0.17.0 and
historical Zig 0.16.0 pins across targets is rejected. A different WAMR revision requires a separately qualified
native image integration and an explicit adapter pin update; matching two
unsupported revisions is not sufficient. The commit check validates a declared
source identity against the supported SDK, not the actual build or deployed image:
retain independent source-closure, receipt and deployment evidence. Synthetic
offline tests remain unpinned and cannot produce measurement evidence.

The image integrator must supply a JSON receipt with exactly:

```
schema_version = 2
kind = "wamr-native-image-receipt"
evidence_kind = "measurement"
os = "linux" | "unikraft"
image = {sha256, bytes}
runtime = {sha256, bytes}
source = {commit, tree_sha256, tracked_diff_sha256}
compiler = {binary: {sha256, bytes}, source, version}
compile_profile = null | "unikraft-x86_64"
options = {optimize, bounds_checks, wx_enforced, import_checks, trap_isolation,
           stack_checks, simd, threads, memory64}
memory_policy = {allocator_by_phase: {load, instantiate, first, steady},
                 page_policy: {reservation, commitment, release}}
execution_lifecycle = {mode, reset_policy, reset_before, reset_timing, reset_scope}
target_abi = public ABI token
platform = {arch, cpu_model, active_cpu_count, azure_sku, azure_region}
configured_vm_ram_bytes = positive integer
compiler_embedded = false
runtime_linkage = "static"
aot_modules = {selected workload key: {sha256, bytes}, ...}
```

This is a field specification, **not a success-shaped sample receipt**. `native-plan`
checks receipt identities against real local image/runtime/compiler/AOT bytes and
retains the exact receipt file hash. The guest must independently establish the
image receipt and observed identities; blindly echoing a request is not measurement.

### Evidence origin and hardware qualification

The v2 `observed` container reconciles producer observations with independent
receipt attestations. Its name does **not** mean every field was discovered by the
guest or that physical hardware was certified. The fixed provenance boundary is:

| Evidence | Origin and limit |
| --- | --- |
| Loaded Wasm/AOT, runtime/compiler and complete-image file hashes/sizes | Actual bytes checked by the host and producer; a local image hash alone does not prove that image is running. |
| Guest architecture, CPU model and active CPU count | Native platform observations checked against the independent receipt, not copied from the request or inferred from SKU capacity. |
| Source/build lineage, options and ABI/image compatibility | Qualified build/integration attestations, with independently checkable runtime settings reconciled by the producer. |
| Azure SKU/region, configured VM RAM and image-to-boot relationship | Independent deployment/receipt attestations; arbitrary request strings or hashing local files cannot establish these facts. |
| Phase durations and covered memory counters | Actual clock/counter observations with their explicit resolution, method and coverage; configured RAM is not measured memory. |

The deployment must be qualified before producing measurement evidence. A
`hardware` label is an integrator attestation, not automatic proof, permission for
a paid run, or a way to turn QEMU correctness runs into performance evidence.
Missing, unqualified or mismatched deployment evidence must fail closed.
Review the private build/deployment evidence independently; this host protocol
does not replace it or establish native image acceptance.

Every public report states
`identity_assurance: "observations-and-receipt-attestations"` and
`hardware_qualification: "requires-independent-deployment-evidence"` in `status`.
Even `paired_measurement_complete: true` means the declared campaign's required
records were accepted, not that the report certified their physical deployment.

Different AOT bytes require different explicit `target_abi` values and retain both
hashes/sizes. Identical bytes require a separate ABI qualification attestation:
exactly `{schema_version: 2, kind: "wamr-aot-abi-proof", evidence_kind: "measurement",
compatible: true, source: SOURCE, target_abis: [LINUX_ABI, UNIKRAFT_ABI],
aot_sha256: HASH}`. Its hash is retained. Sharing a filename or architecture does
not establish ABI compatibility. Actual compatibility tests remain the native
integration's responsibility; the host checks that its attestation binds these bytes.
Identical bytes must also have identical `compile_profile` declarations.

## Exact invocation

Prepare the config/receipts from actual local image evidence, then:

```sh
python3 scripts/bench_coremark.py native-plan \
  --config .bench-coremark/native/config.json \
  --out .bench-coremark/native/plan.json

python3 scripts/bench_coremark.py native-request \
  --manifest .bench-coremark/native/plan.json --run-id run-0001 \
  --out .bench-coremark/native/request-0001.json
```

The plan contains a fresh **public random campaign UUID**, validity interval,
source/tool hashes, full artifact identities, workload settings and a forward/reverse
counterbalanced schedule. Run IDs are public `run-NNNN` ordinals, not cloud execution
IDs. Every scheduled warmup and measured run is a new load/instance.

The request is `{"config": CONFIG, "config_sha256": HASH}`. `HASH` is SHA-256 of
`json.dumps(CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("UTF-8")`.
`CONFIG` contains `campaign_id`, the exact schedule `run` object, `target`, `workload`,
`steady_invocations`, `execution_lifecycle`, and `phase_contract: "wamr-embedding-v2"`.
Run objects contain `run_id`, `target`, `workload`, `phase`, `attempt: 1`, `position`.

For a Linux producer implementing this contract:

```sh
python3 scripts/bench_coremark.py native-capture \
  --manifest .bench-coremark/native/plan.json --run-id run-0001 \
  --out .bench-coremark/native/run-0001 --timeout 900 \
  -- /path/to/qualified-linux-embedding-producer
```

The host sets `WAMR_BENCH_REQUEST` to the private absolute request filename and
`WAMR_BENCH_CONFIG_SHA256` to its configuration digest. The producer, not Python,
uses the embedding APIs and guest/native timers. This command refuses a Unikraft
target: it must not accidentally launch a hosted CLI in place of a native guest.

For externally collected Unikraft stdout/serial data, use `native-import`. Supply
actual host-side observation boundaries/monotonic duration from the collector:

```sh
python3 scripts/bench_coremark.py native-import \
  --manifest .bench-coremark/native/plan.json --run-id run-0002 \
  --stdout .bench-coremark/native/serial.bin \
  --stderr .bench-coremark/native/collector-stderr.bin \
  --started-at "$OBSERVATION_START_UTC" --completed-at "$OBSERVATION_END_UTC" \
  --observation-seconds "$OBSERVATION_SECONDS" \
  --control-plane-seconds "$CONTROL_PLANE_SECONDS" \
  --transport-outcome exited --process-returncode 0 \
  --out .bench-coremark/native/run-0002
```

Omit `--control-plane-seconds` if not measured; `null` is not zero.
`--process-returncode` describes the **collector**, never the guest. For collector
timeouts/launch errors, pass `--transport-outcome timeout|launch_error` and omit
`--process-returncode`. No terminal guest result is invented in these cases.

Both adapters retain `stdout.bin`, `stderr.bin`, and `observation.json` with POSIX mode
0600 in a new directory (0700). On Windows, use an appropriately private inherited
directory ACL; POSIX mode bits do not establish NTFS access control.
Raw evidence is saved **before** result validation,
including malformed output, failures, traps, timeouts and missing terminal results.
Import observation timestamps must lie inside the campaign validity window.

## Native result protocol v2

All native plans, image/ABI receipts, observations, results and reports now use
`schema_version: 2`; requests declare `phase_contract: "wamr-embedding-v2"`. Version 1
is explicitly rejected, not silently upgraded, because it could not represent full
snapshot replay or a completed call with missing duration/output-copy evidence.
Regenerate the plan/request and update the producer together. Do not relabel old
evidence or rely on ignored unknown keys.

Emit exactly one complete UTF-8 line on stdout/serial, starting at column zero:

```
WAMR_BENCH_RESULT=<one JSON object, no line breaks>
```

Boot output may precede/follow it and may contain arbitrary non-UTF-8 bytes, retained
privately without decoding. The result line itself must be valid UTF-8 JSON.
Fragmented/truncated JSON, extra result lines,
duplicate JSON keys, nonfinite numbers, unknown fields, stale campaign/config IDs,
wrong artifacts, unexpected run IDs and incomplete campaigns are errors.
The object has **exactly** these fields:

* `schema_version: 2`, `kind: "wamr-native-benchmark-result"`,
  `evidence_kind: "measurement"`, `campaign_id`, `run_id`, `config_sha256`,
  `image_receipt_sha256`.
* `observed`: exactly `image_sha256`, `runtime_sha256`, `aot_sha256`, `wasm_sha256`,
  `platform`, `options`, `compile_profile`, `mode: "aot"`, `jit_preset: null`, independently checked by
  the producer against loaded native artifacts, active platform and independent
  receipts according to the evidence-origin boundary above.
* `outcome`: `success`, `trap`, `timeout`, `abort`, or `error`.
  `exit_code`: the final invocation's complete unsigned 32-bit `proc_exit` status,
  or `null` for ordinary return, trap/error, or no invocation. Successful `proc_exit`
  requires zero; an ordinary successful return has **no invented exit status**.
  Preserve `proc_exit(0)` separately from normal return and preserve nonzero
  `proc_exit` without POSIX eight-bit truncation or signed conversion. A nonzero
  invocation exit must match this enclosing status and stop further invocations.
  Host process/collector return codes remain separate observation fields.
  A trapped or nonreturning
  workload must never be rewritten into a successful terminal event.
* `phase_contract: "wamr-embedding-v2"`, `execution_lifecycle`: identical to the
  request, target and image receipt.
* `clock`: exactly `source` (public monotonic clock token), `unit` (`ns`, `us`, `ms`),
  `ticks_per_second` (respectively 1000000000, 1000000, 1000), `resolution_ticks`
  (positive integer from the supported native clock resolution). Raw TSC cycles
  without a qualified conversion are not supported. Record zero durations when
  below resolution rather than inventing precision; the resolution remains visible.
  Resolution coarser than one second is not supported by this benchmark contract.
  The total of sequential phase durations cannot exceed the campaign validity
  window: this sanity check rejects overflow/wraparound evidence such as
  `UINT64_MAX`, without treating host observation latency as an execution timer.
  If no supported clock exists, an early **failed, unstarted** attempt may use
  `clock: null`, with all scalar phases `null`, empty steady/invocation/reset lists and
  `memory: null`. It is not a timed sample. Do not fabricate a clock resolution to
  report that failure. For a WASI-provided phase clock, use the monotonic ID's actual
  resolution (ID 1); real-time epoch or CPU-time clocks are not substitutes.
* `phases`: `compile_ticks: null`, `load_ticks`, `instantiate_ticks`, `lifecycle_setup_ticks`,
  `first_invocation_ticks`, `steady_state_ticks` (list).
* `reset_events`: ordered list of `{before_invocation: ORDINAL, outcome:
  "completed"|"error", elapsed_ticks: INTEGER|null}`. Ordinals are one-based
  invocation numbers, so the first reset is before invocation 2. Every repeated
  call needs a completed, timed preceding reset. Failed runs may retain one
  additional reset event before the next, uncalled invocation. Missing reset time
  is `null`, not zero, and forbids proceeding to that invocation.
* `invocations`: list, first then steady, each exactly `{phase: "first"|"steady",
  outcome: "returned"|"proc_exit"|"trap"|"error", exit_code: u32|null,
  stdout_base64: STRING, stdout_complete: BOOLEAN, measurement_errors: LIST}`.
  Only `proc_exit` has a non-null exit code (including zero);
  returned/trap/error use `null`. `stdout_base64` is canonical padded RFC 4648 base64
  of the **exact callback bytes**, including NUL and non-UTF-8 bytes; empty output is
  `""`. Never use replacement decoding. Successful CoreMark output must decode as
  ASCII before correctness checks. Include separate complete output for every
  invocation; a shared serial transcript cannot substitute for it.
  `stdout_complete` describes capture/copy completeness, not whether the intended
  guest output was successful. `measurement_errors` is a sorted unique subset of
  `post-call-clock`, `stdout-copy`, `pending-output`, normally `[]`. The failure
  rules below preserve the actual guest terminal tag independently of these errors.
* `memory`: the native memory receipt below, or `null` on a failed attempt.

### Accepted lifecycle/reset policies

The `execution_lifecycle` object has exactly five fields. Both supported modes use
`reset_before: "each-steady-invocation"` and
`reset_timing: "excluded-from-invocation"`. Exclusion is explicit and the reset
duration is separately required; it is not permission to omit expensive work.

| `mode` | `reset_policy` | Exact sorted `reset_scope` |
|---|---|---|
| `same-instance-warm` | `invocation-state-only` | `["invocation-state", "stdout-capture"]` |
| `snapshot-replay` | `restore-post-start-snapshot` | `["globals", "invocation-output", "linear-memory-access-protection", "linear-memory-contents", "linear-memory-logical-size", "passive-segment-drop-state", "table-entries-signatures", "wasi-context"]` |

Warm mode continues the same initialized instance with its mutable Wasm state intact.
Only qualified per-call execution/output rearming occurs outside the call timer.
It must not restore memory, globals, tables, descriptors or a fresh WASI context.
If repeated command/libc execution is unsafe, warm mode is unsupported.

Snapshot replay uses a baseline taken after successful instantiation and the Wasm
module start section, before the first exported `_start` call. It restores linear
contents/logical size/access protection, globals, table entries/signature backing,
passive-segment drop flags, the initial WASI context and invocation-output boundary.
Table-size changes are rejected before restore rather than silently truncated.
Grown linear suffixes become inaccessible and logical size returns to the baseline;
actual committed high-water bytes do **not** shrink merely because access is revoked.
Report a physical reduction only if the measurement method establishes a real unmap
or decommit.

The same instance, code mapping/import pointers, reservation bases, immutable artifact
and completed module-start state are reused; the module start section is not rerun.
SDK pending-terminal clearing occurs inside the timed API call, not by snapshot
reset overwriting private runtime fields. Fresh WASI context initialization restores
the original immutable arguments/environment and initial descriptor/open/exit/error
contract, without acquiring new external file/socket resources. Reset must reject
pending output errors before they can be erased. Captured output may reuse capacity,
but bytes belonging to an earlier invocation must remain stable until serialized.
Context replacement alone is not a full snapshot, nor does CRC success prove complete
restoration. The producer must qualify resource lifetimes and reset failure behavior.
This mode is **restored-state replay**, not warm libc/state continuation or fresh
load/instantiation. Both targets must use the same declared semantics.

For snapshot mode, `lifecycle_setup_ticks` measures snapshot construction after
instantiation/module start and before the first invocation, including allocation/copy.
It is required on success. Warm mode uses `null` because there is no snapshot setup.
`reset_events` measures the entire relevant restore/rearm and output/context setup
before each repeat, on the declared guest clock, separately from the call.
Run-level warmup/measured labels do not change this per-instance lifecycle.

Reports expose `repeated_invocation_seconds`, `reset_seconds`,
`reset_and_invocation_seconds`, and `lifecycle_setup_seconds`. Only warm mode
populates `steady_state_seconds`; replay populates `snapshot_replay_seconds` instead.
The legacy wire name `steady_state_ticks` denotes repeated call-only durations
under the explicit v2 lifecycle, not an implicit warm-state performance claim.
Reset+call excludes initial load, instantiation and snapshot construction, which
remain separate reported phases.

### Phase boundaries

All phase timings come from the **same native guest clock** and exclude host
observation/control-plane time:

1. `load_ticks`: from supplying the in-memory precompiled artifact bytes through
   byte-copy and completed format/ABI/CPU/import validation. Excludes boot,
   transport, disk acquisition, compilation and artifact hashing.
2. `instantiate_ticks`: instance allocation, code/linear mapping and relocation,
   memory/table/global initialization, executable permissions/helper wiring, WASI
   binding and execution-environment setup after validation, before lookup/call.
   Invoke the module start section once during this initialization scope, distinct
   from the exported `_start` calls. Native timing completion indicators must establish the phases actually finished;
   separately timed caller setup must not overlap or double-count native intervals.
3. `first_invocation_ticks`: lookup/call of `_start` on that fresh instance through
   normal return or caught WASI `proc_exit(0)`, including workload I/O.
4. `steady_state_ticks`: repeated `_start` calls on the same loaded instance, using
   the explicitly declared warm-continuation or snapshot-replay policy. Reset/rearm
   is separately timed outside the call. Do not silently recreate/load an instance,
   include first invocation in repeats, or mix an internal function with full `_start`.

Successful records require all three main scalar phases, the lifecycle-specific
setup/reset evidence and exactly the requested repeated invocation count. Each call
has its own actual terminal/output evidence.

### Post-call and partial failure evidence

A post-call clock or output-copy failure must not discard a completed guest call or
invent its duration. Retain that invocation's actual `returned`, `proc_exit(u32)`,
`trap` or `error` tag. The enclosing run must be unsuccessful and cannot contribute
to successful phase statistics.

If an end timestamp fails, retain `first_invocation_ticks: null` with the first
invocation record present, or a `null` entry in `steady_state_ticks` paired with
the actual repeated invocation. Require `measurement_errors: ["post-call-clock"]`
(plus any other actual errors). A first scalar `null` with **no** invocation still
means unstarted; a repeated array entry always means a call actually occurred.
The known clock provider/unit/resolution remains required. No later reset or call
may follow incomplete measurement evidence.

If copying captured output fails, retain the exact known bytes/prefix in
`stdout_base64`, set `stdout_complete: false`, and include `stdout-copy`.
Empty known output is allowed but cannot masquerade as complete capture. A copy
failure may accompany a valid duration or a clock failure. Public reports preserve
completeness/error flags, terminal status and the hash of retained bytes, not private
bytes themselves. CRC/correctness is unavailable for incomplete capture.

For complete CoreMark output with a missing enclosing duration, the parser still
checks the pinned CRC/iterations/ticks/display/rate relationships; it marks
`invocation_timing_available: false` and minimum timing unqualified. It does not
invent an enclosing duration from the printed workload time.

After a native call, inspect deferred per-fd output errors before success/rearm.
Use `pending-output` while preserving the actual guest terminal and known output;
do not convert a returned call into a fabricated trap. No subsequent reset may
erase the failure. If the producer cannot serialize even the terminal/known-byte
record safely, retain raw private diagnostics and fail capture explicitly; do not
emit a successful measurement placeholder.

### Memory receipt

`memory` has exactly:

* `image_sha256`: exact complete native image identity.
* `configured_vm_ram_bytes`: the image receipt's configured VM RAM, **not usage**.
* `coverage`: `complete-guest` or `partial-guest`.
* `method`: public measurement-method token.
* `policy`: the observed caller allocator-by-phase and page-policy identities,
  matching the target's `memory_policy` and exact-image receipt. Do not replace the
  measurement method with an allocator's name, or confuse allocator selection with
  virtual reservation/physical commitment policy.
* `covered_regions`, `omitted_regions`: disjoint lists of public region tokens.
  Complete coverage requires no omissions; partial coverage requires explicit
  omissions. Runtime heap alone is partial coverage.
* `samples`: three objects in order, with `stage` respectively
  `after_instantiation`, `after_first`, `after_steady`, and measured
  `reserved_address_bytes`, `committed_bytes` for the **same covered regions**.
  A failed attempt may retain just the nonempty prefix actually observed, or use
  `memory: null` if no snapshot was obtained.

These are phase snapshots, **not peaks**. Report both byte counts and measured
coverage. Virtual reservation is not resident/committed memory; configured VM RAM
is neither. Allocator totals must never be called RSS. Native page-accounting coverage
and excluded kernel/device/stack/code regions must come from exact-image evidence.
Comparing partial coverage across unlike region sets requires further review.
Snapshot storage, reset scratch and fresh context allocations must also be included
or explicitly listed among omitted regions; reused mappings do not make that memory
free or invisible.
Flat committed-memory snapshots do not establish allocation-free steady execution;
that claim would require separate actual allocator-event evidence. This report does
not infer allocator call counts or attribute performance to a policy label.

## Reporting, failures and privacy

Pass every planned capture directory, including warmups and failures:

```sh
python3 scripts/bench_coremark.py native-report \
  --manifest .bench-coremark/native/plan.json \
  --capture .bench-coremark/native/run-0001 \
  --capture .bench-coremark/native/run-0002 \
  --out .bench-coremark/native/report.json
```

Repeat `--capture` for **all** schedule entries; the two-entry example only completes
a plan with two scheduled attempts. Reports use schema version 2 and kind
`wamr-native-matched-comparison`, distinct from hosted CoreMark schema version 2.
They retain the plan, hashes/byte counts, each run/configuration/terminal event,
guest phase durations and clock, per-invocation correctness, external observations,
memory coverage and separate successful-sample statistics. Failed campaigns still
write their report when their evidence is well formed. Successful-only statistics
are explicitly counted and do not erase unsuccessful attempts.

There are no automatic retries. A repeated run ID is an error, not a replacement;
start a new campaign for a retry and retain the previous campaign. A missing/malformed
guest result on an otherwise successful transport is a protocol error: preserve
the private capture and repair the producer, not the data.

Exit codes: 0 = operation/attempt successful; 1 = a well-formed unsuccessful attempt
or campaign (evidence/report persisted); 2 = invalid/mismatched/incomplete evidence.
No report is emitted for an invalid campaign. Public JSON deliberately excludes raw
workload output, boot logs, local paths, cloud resource/run IDs and credentials.
It retains content hashes so private raw evidence can be audited. Public invocation
records retain `stdout_sha256`, not the output's base64 bytes. Only supply reviewed
public tokens in identity/measurement fields; do not put private IDs into labels.
Artifacts and raw logs are never uploaded by this tooling.

Serial-download arrival is observation latency, **not execution time**. Azure
control-plane duration is an optional separate value. There is no performance gate,
speedup threshold, OS-benefit claim, or Wasmtime OS-attribution baseline. Wasmtime
may be collected separately as a correctness/reference engine.

## Future optional JIT (#1044)

Version 2 accepts only `mode: "aot"`, `jit_preset: null`, a compiler-free image receipt,
and `compile_ticks: null`. These explicit fields reserve a non-conflating extension:
qualified JIT must declare `fast` or `full`, time compilation separately, retain
its compiler-bearing complete image and memory growth versus this compiler-free
AOT comparator, and disclose any differing safety/optimization settings.
Simply changing a mode label or treating load time as compile time is rejected.
Native JIT APIs and qualification are not prerequisites for these host/parser tests.

## Validation

```sh
python3 -m unittest discover -s scripts -p test_bench_coremark.py
```

`zig build test` also runs the portable `NativeBenchmarkTests` class through the
existing Python harness-command mechanism on ordinary Linux, ARM and Windows CI.
The existing hosted CoreMark tests continue to run in the CoreMark workflows.

The tests pin all four tracked fixtures, execute an explicitly synthetic Linux
producer subprocess, exercise timeouts and retained raw evidence, and reject
duplicate/stale/partial records, mode/identity/clock/memory mismatches and ambiguous
CoreMark output. They are software validation, not native measurements.

## Local Linux vs Unikraft x86_64 KVM correctness comparison (Zig 0.17.0)

Local side-by-side correctness validation on Zig `0.17.0` (`cataggar/wamr`
`bb24e3e3f0ce5b998c0b24f6c06007d9a8ad7d20` + `cataggar/unikraft` `47ef7f85` /
[#331](https://github.com/cataggar/unikraft/pull/331), `correctness-only`, not a
cloud `Standard_D2s_v5` performance claim) confirms byte-identical `.cwasm`
compilation, runtime heap/code/page commitment, and execution results across the
Linux producer (`wamr-native-bench`, `wamr-native-jit-aot-compare`,
`wamr-native-jit-bench`) and the Unikraft Hyper-V EFI guest under x86_64 KVM:

### Matched AOT / `fast` JIT / `full` JIT sampler (`matched.wasm`)

Both Linux (`scripts.native_jit_benchmark`, `CLOCK_MONOTONIC` `1 ns` resolution)
and Unikraft x86_64 KVM (`support/apps/wamr-aot/sampler.zig`, Hyper-V reference
monotonic `100 ns` resolution) execute the `309` B `matched.wasm` fixture
(`135511ae68829ab7feeff41de03451ba860906f5fc1cd053ec8d6f3a7e1dd9e5`, workload
`volatile-compute-memory-2000`, expected `2241491208`, `growth_previous_pages = 2`):

| Metric | Linux `aot` | Unikraft KVM `sample-aot` | Linux `fast` | Unikraft KVM `fast` | Linux `full` | Unikraft KVM `full` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `.cwasm` bytes (`sha256` prefix) | `1,156` B (`d6a89bd3...`) | `1,156` B (`d6a89bd3...`) | `1,331` B (`6ccddd51...`) | `1,331` B (`6ccddd51...`) | `1,324` B (`47dc8f33...`) | `1,324` B (`47dc8f33...`) |
| `compile_ns` (`parse`/`lower`/`opt`/`cg`/`emit`) | `null` | `null` | `533,040` (`57,256`/`40,481`/`250,736`/`140,731`/`9,214`) | `1,658,400` (`245,800`/`234,600`/`543,900`/`545,100`/`60,200`) | `613,570` (`43,024`/`31,478`/`362,534`/`121,462`/`6,760`) | `2,019,000` (`246,100`/`245,300`/`891,700`/`543,300`/`62,200`) |
| `compiler_polls` / `fuel_per_invocation` | `0` / `null` | `0` / `null` | `332` / `100,000` | `316` / `100,000` | `534` / `100,000` | `494` / `100,000` |
| `load_ns` / `instantiate_ns` / `start_ns` / `growth_ns` | `15,012` / `68,193` / `60` / `24,587` | `60,800` / `15,200` / `100` / `7,300` | `14,342` / `69,905` / `30` / `19,880` | `37,900` / `15,400` / `100` / `6,500` | `12,008` / `66,460` / `30` / `23,215` | `39,400` / `16,300` / `100` / `6,600` |
| `invocations[0..3].ns` (`value = 2241491208`) | `5,719` / `3,886` / `3,766` / `3,765` | `4,000` / `3,600` / `3,600` / `3,500` | `19,539` / `3,756` / `3,615` / `3,606` | `3,900` / `3,400` / `3,300` / `3,300` | `6,330` / `6,129` / `6,009` / `5,989` | `6,500` / `5,900` / `5,800` / `5,800` |
| `compiler_peak_bytes` / `compiler_retained_bytes` | `0` / `0` B | `0` / `0` B | `203,835` / `1,331` B | `230,051` / `1,331` B | `329,882` / `1,324` B | `339,036` / `1,324` B |
| `caller_peak_bytes` | `2,980` B | `2,980` B | `203,835` B | `230,051` B | `329,882` B | `339,036` B |
| Runtime `heap_peak_bytes` / `code_bytes` (`reserved`) | `2,980` / `815` B (`4,096` B) | `2,980` / `815` B (`4,096` B) | `3,242` / `990` B (`4,096` B) | `3,242` / `990` B (`4,096` B) | `3,232` / `983` B (`4,096` B) | `3,232` / `983` B (`4,096` B) |
| Linear memory committed `before -> after` (`reserved`) | `131,072 -> 196,608` B (`524,288` B) | `131,072 -> 196,608` B (`524,288` B) | `131,072 -> 196,608` B (`524,288` B) | `131,072 -> 196,608` B (`524,288` B) | `131,072 -> 196,608` B (`524,288` B) | `131,072 -> 196,608` B (`524,288` B) |
| Post-teardown live / reserved bytes | `0` / `0` B | `0` / `0` B | `0` / `0` B | `0` / `0` B | `0` / `0` B | `0` / `0` B |

### Matched AOT snapshot-replay producer (`wamr-native-bench --check`)

On the four pinned v2 workloads (`unikraft-x86_64` profile, `snapshot-replay`
lifecycle, 100-iteration correctness check for CoreMark), the Linux producer
and Unikraft x86_64 KVM guest consume identical `.cwasm` artifacts and produce
matching CRCs (`seedcrc=0xe9f5`, `crclist=0xe714`, `crcmatrix=0x1fd7`,
`crcstate=0x8e3a`, `crcfinal=0x988c`):

| Workload | `.cwasm` bytes (`sha256` prefix) | `load_ticks` (ns) | `instantiate_ticks` (ns) | `lifecycle_setup_ticks` (ns) | `first_invocation_ticks` (ns) | `steady_state_ticks` (ns) | `reset_events` (ns) | Reserved / committed (`after_first`) | Caller peak / post-release |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `compute` (`unroll4.wasm`) | `819` B (`cec63990...`) | `15,543` | `27,662` | `481` | `172,207,932` | `167,396,782`, `164,394,266` | `751`, `731` | `4,096` / `4,096` B | `3,194` / `0` B |
| `memory` (`iv_store.wasm`) | `646` B (`b347ba32...`) | `15,173` | `71,377` | `44,406` | `799,401,924` | `775,941,080`, `744,231,572` | `7,701`, `7,972` | `16,781,312` / `135,168` B | `199,604` / `0` B |
| `coremark` (`coremark_wasi.wasm`, 100 iter) | `170,318` B (`f342cdfc...`) | `103,345` | `145,178` | `50,937` | `6,517,371` | `5,841,647`, `5,598,122` | `17,266`, `16,915` | `16,941,056` / `360,448` B | `455,054` / `0` B |
| `coremark-nofp` (`coremark_wasi_nofp.wasm`, 100 iter) | `168,866` B (`4741c445...`) | `77,246` | `121,833` | `46,770` | `5,605,642` | `5,593,615`, `5,572,363` | `11,828`, `8,803` | `16,941,056` / `360,448` B | `452,837` / `0` B |

### Closed CoreMark six-mode image-chain correctness profile (`coremark-qcow2-derived-vhd`)

In `cataggar/unikraft` `e1222136`
([#332](https://github.com/cataggar/unikraft/pull/332), building on standalone
`uk-wamr-aot-build olddefconfig` `build/.config` seeding in `15e6b592`,
[#331](https://github.com/cataggar/unikraft/pull/331)),
`support/build/wamr-native-ci` adds the closed `coremark-qcow2-derived-vhd`
profile (`ProductionProfile.coremark_exact_v2` /
`CompatibleRecordSet.coremark_v2_qcow2_derived_vhd` /
`Compatibility.coremark_qcow2_derived_vhd_v2`) alongside the unchanged default
`qcow2-derived-vhd` (`tiny_exact_v2`) profile:

* **Closed profile selection and SDK pin**: `uk-wamr-native-ci build --coremark`
  (or `--profile coremark-qcow2-derived-vhd`) passes `--coremark` to
  `uk-wamr-aot-build prepare` and requires the Zig `0.17.0` WAMR SDK pin
  `bb24e3e3f0ce5b998c0b24f6c06007d9a8ad7d20` (`identity_coremark` rejects the
  historical `a53205d77be3b880eb8f8b96679512ba58e2331a` pin).
* **Bound artifacts and archive layout**: `coremark-qcow2-derived-vhd` embeds the
  tiny fixture (`answer() == 42`, `memory.grow(1) == 2`, `trap()`) together with
  both pinned CoreMark WASI modules compiled via
  `wamrc compile --target=x86_64 --profile=unikraft-x86_64` (`coremark.wasm` /
  `coremark.cwasm` and `coremark-nofp.wasm` / `coremark-nofp.cwasm`). It binds
  `30` artifact slots (`artifact_names_coremark_v2` vs `26` for `tiny_v2`), `89`
  regular public-source ZIP members (`coremark_v2_zip_member_count` vs `85` for
  `tiny_v2`), and `10` sorted `identity_coremark` keys (`minimal_wasi: true`),
  refusing cross-profile relabeling between `qcow2-derived-vhd` and
  `coremark-qcow2-derived-vhd`.
* **Six-mode KVM execution and serial validation**: executes all six local
  x86_64 KVM/OVMF boot modes (`raw-x2apic`, `raw-legacy-apic`, `qcow2-x2apic`,
  `qcow2-legacy-apic`, `vpc-x2apic`, `vpc-legacy-apic`) with `WAMR_NATIVE_WASI=`
  admitted only in `coremark_exact_v2` (still forbidden in `tiny_exact_v2`).
  Both `uk-wamr-log-validate tiny --output json-v1` and
  `support/build/wamr-native-ci/handoff/candidate.zig` (`checkSerial`) enforce
  `minimal_wasi: true`, qualified UEFI UTC + Hyper-V reference realtime support
  (`realtime_supported: true`), exact 100-iteration (`coremark 0 0 0 100 0`)
  stdout fields (`Iterations: 100`, `seedcrc: 0xe9f5`, `[0]crclist: 0xe714`,
  `[0]crcmatrix: 0x1fd7`, `[0]crcstate: 0x8e3a`, `[0]crcfinal: 0x988c`, plus the
  short-duration warning markers, mirrored on the host by
  `native_benchmark.validate_coremark_short_crc_output`), and zero post-teardown
  counters (`teardown: 0`).
* **Boundary with #1046 measurement campaigns**: `coremark-qcow2-derived-vhd`
  closes six-mode native Unikraft image-lineage and 100-iteration WASI CoreMark
  CRC correctness (`cataggar/wamr#1045` / `cataggar/unikraft#156`) with
  `hardware_acceptance: "not_established"`, `cloud_authority: "not_admitted"`,
  `benchmark: "not_measured"`, and all six direct-compute approval flags `false`.
  It does **not** emit `WAMR_BENCH_RESULT=` (`wamr-embedding-v2`), does **not**
  run the authoritative `400,000`-iteration (`>= 10 s`) CoreMark benchmark
  configuration (`coremark 0 0 0 400000 0`, `[0]crcfinal: 0x382f`), and is
  rejected by `native_benchmark.coremark_correctness` / `validate_result` if
  presented as a successful v2 measurement record. An authorized paired Linux vs
  Azure Unikraft hardware campaign (`Standard_D2s_v5`) remains separate; no cloud
  resources were provisioned and no speedup is claimed.


