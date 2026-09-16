# Matched Linux / native Unikraft AOT evidence

This is the **host-side software portion** of [#1046](https://github.com/cataggar/wamr/issues/1046),
not a performance result or completion claim. Native image qualification, deployed-image
receipts, guest integration and paired real measurements remain
[cataggar/unikraft#156](https://github.com/cataggar/unikraft/issues/156).
The earlier #88 pipeline may be reused externally; these commands do not create cloud
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
phase timers by timing a subprocess. The Linux and Unikraft native embedding
producers must implement the contract below before real measurements can be collected.
Standalone parser/capture tests do not depend on those APIs.

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
is rejected even when the process exits zero. Reported workload time cannot exceed
the enclosing guest invocation time (10 ms print-rounding tolerance). Throughput
must agree with fixed iterations / reported seconds (one iteration/s or 0.1%
print-rounding tolerance, whichever is larger).

The independent minimum-duration check requires both reported CoreMark time and guest
invocation time to be at least ten seconds. `authoritative` is only a sample-count
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
  native image, **not** an unrelated hosted `wamr` executable).
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
  and boolean `bounds_checks`, `stack_checks`, `simd`, `threads`, `memory64`.
  All must describe the actual linked/runtime/AOT settings and match across targets.
* `image_path`: exact **complete boot/deployment image file**, not ELF sections or
  stripped text size; `image_receipt_path`: its native integration receipt.
* `aot_paths`: map from every selected workload to the actual AOT artifact file.

Sources, compiler identities, options, architecture, CPU model, active CPU count,
SKU and region must match. Source/build settings or guest CPU counts that cannot
be established are pending evidence, not `"unknown"` measurements.

The image integrator must supply a JSON receipt with exactly:

```
schema_version = 1
kind = "wamr-native-image-receipt"
evidence_kind = "measurement"
os = "linux" | "unikraft"
image = {sha256, bytes}
runtime = {sha256, bytes}
source = {commit, tree_sha256, tracked_diff_sha256}
compiler = {binary: {sha256, bytes}, source, version}
compile_profile = null | "unikraft-x86_64"
options = {optimize, bounds_checks, stack_checks, simd, threads, memory64}
target_abi = public ABI token
platform = {arch, cpu_model, active_cpu_count, azure_sku, azure_region}
configured_vm_ram_bytes = positive integer
compiler_embedded = false
aot_modules = {selected workload key: {sha256, bytes}, ...}
```

This is a field specification, **not a success-shaped sample receipt**. `native-plan`
checks receipt identities against real local image/runtime/compiler/AOT bytes and
retains the exact receipt file hash. The guest must independently establish the
image receipt and observed identities; blindly echoing a request is not measurement.

Different AOT bytes require different explicit `target_abi` values and retain both
hashes/sizes. Identical bytes require a separate ABI qualification attestation:
exactly `{schema_version: 1, kind: "wamr-aot-abi-proof", evidence_kind: "measurement",
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
`steady_invocations`, and `phase_contract: "wamr-embedding-v1"`.
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

## Native result protocol v1

Emit exactly one complete UTF-8 line on stdout/serial, starting at column zero:

```
WAMR_BENCH_RESULT=<one JSON object, no line breaks>
```

Boot output may precede/follow it. Fragmented/truncated JSON, extra result lines,
duplicate JSON keys, nonfinite numbers, unknown fields, stale campaign/config IDs,
wrong artifacts, unexpected run IDs and incomplete campaigns are errors.
The object has **exactly** these fields:

* `schema_version: 1`, `kind: "wamr-native-benchmark-result"`,
  `evidence_kind: "measurement"`, `campaign_id`, `run_id`, `config_sha256`,
  `image_receipt_sha256`.
* `observed`: exactly `image_sha256`, `runtime_sha256`, `aot_sha256`, `wasm_sha256`,
  `platform`, `options`, `compile_profile`, `mode: "aot"`, `jit_preset: null`, independently checked by
  the producer against deployed/loaded native artifacts and active platform.
* `outcome`: `success`, `trap`, `timeout`, `abort`, or `error`.
  `exit_code`: integer or `null`; success requires zero. A trapped or nonreturning
  workload must never be rewritten into a successful terminal event.
* `phase_contract: "wamr-embedding-v1"`.
* `clock`: exactly `source` (public monotonic clock token), `unit` (`ns`, `us`, `ms`),
  `ticks_per_second` (respectively 1000000000, 1000000, 1000), `resolution_ticks`
  (positive integer from the supported native clock resolution). Raw TSC cycles
  without a qualified conversion are not supported. Record zero durations when
  below resolution rather than inventing precision; the resolution remains visible.
  If no supported clock exists, an early **failed, unstarted** attempt may use
  `clock: null`, with all scalar phases `null`, empty steady/invocation lists and
  `memory: null`. It is not a timed sample. Do not fabricate a clock resolution to
  report that failure. For a WASI-provided phase clock, use the monotonic ID's actual
  resolution (ID 1); real-time epoch or CPU-time clocks are not substitutes.
* `phases`: `compile_ticks: null`, `load_ticks`, `instantiate_ticks`,
  `first_invocation_ticks`, `steady_state_ticks` (list).
* `invocations`: list, first then steady, each exactly `{phase: "first"|"steady",
  outcome: "returned"|"proc_exit"|"trap"|"error", exit_code: integer|null,
  stdout: STRING}`. Include separate complete workload output for every invocation;
  a shared serial transcript cannot substitute for per-invocation output.
* `memory`: the native memory receipt below, or `null` on a failed attempt.

### Phase boundaries

All phase timings come from the **same native guest clock** and exclude host
observation/control-plane time:

1. `load_ticks`: from supplying the in-memory precompiled artifact bytes to the
   embedding loader through completed load/relocation/validation. Excludes boot,
   transport, disk acquisition, compilation and artifact hashing.
2. `instantiate_ticks`: instance creation, memory/table/global initialization,
   WASI binding setup and execution-environment setup after load, before lookup/call.
3. `first_invocation_ticks`: lookup/call of `_start` on that fresh instance through
   normal return or caught WASI `proc_exit(0)`, including workload I/O.
4. `steady_state_ticks`: repeated `_start` invocations on the **same instance**,
   re-arming supported WASI invocation/exit/output state outside the timed call.
   Do not silently recreate/load an instance, include first invocation in steady
   state, or mix a function microbenchmark with complete `_start` execution.

Successful records require all three scalar phases and exactly the requested
steady invocation count. Each timed invocation has corresponding terminal/output
evidence. Failed records retain the completed or interrupted phases that actually
have timings; unstarted phases are `null` and unstarted steady calls are omitted.
A producer unable to safely repeat the pinned command module is pending integration,
not permission to silently change the phase contract.

### Memory receipt

`memory` has exactly:

* `image_sha256`: exact complete native image identity.
* `configured_vm_ram_bytes`: the image receipt's configured VM RAM, **not usage**.
* `coverage`: `complete-guest` or `partial-guest`.
* `method`: public measurement-method token.
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
a plan with two scheduled attempts. Reports use schema version 1 and kind
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
It retains content hashes so private raw evidence can be audited. Only supply reviewed
public tokens in identity/measurement fields; do not put private IDs into labels.
Artifacts and raw logs are never uploaded by this tooling.

Serial-download arrival is observation latency, **not execution time**. Azure
control-plane duration is an optional separate value. There is no performance gate,
speedup threshold, OS-benefit claim, or Wasmtime OS-attribution baseline. Wasmtime
may be collected separately as a correctness/reference engine.

## Future optional JIT (#1044)

Version 1 accepts only `mode: "aot"`, `jit_preset: null`, a compiler-free image receipt,
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
