# WASI pthread and atomic benchmark

This harness supplies the threaded performance evidence required before #963
adds more AOT cancellation polls. It runs checked-in, deterministic core-Wasm
fixtures through the interpreter and AOT on x86_64 and AArch64.

## Workloads

`threaded.wasm` is a real wasi-libc pthread program. It is built with the
`wasm32-wasi-threads` target and exercises:

| workload | operation |
|---|---|
| `hot` | one back-edge-heavy integer loop per pthread |
| `atomic` | contended shared `atomic_fetch_add` RMW |
| `wait-notify` | controller/worker `memory.atomic.wait32` + `notify` hand-offs |
| `spawn-join` | batches of wasi-libc `pthread_create` + `pthread_join` |

Each workload scales over 1/2/4/8 worker pthreads. The report includes aggregate
and per-thread throughput. Caller-provided 16 KiB pthread stacks make the
fixture's memory footprint deterministic and avoid mixing allocator growth into
the spawn/join measurement.

`single.wasm` contains the exact same `bench_hot_kernel` from `kernel.h`, but is
built for ordinary `wasm32-wasi` and has no thread-spawn import. The harness runs
that one identical byte fixture on `-Dlib_wasi_threads=false` and `true`
runtimes. A module that imports `wasi.thread-spawn` cannot validly load in the
disabled runtime, so this separate no-spawn module is the conditional
infrastructure comparison requested by #616.

Every invocation prints one JSON object. The driver rejects an incorrect
workload, thread count, iteration count, operation count, checksum, extra output,
non-zero exit, watchdog timeout, malformed/duplicate timing, a corrected
interval shorter than 100 ms, or measured timer/barrier overhead of 1% or more.

## Metric definitions

The #979 reports timed the host process from `Popen` through process exit.
Those observations include process creation, module loading, AOT mapping,
passive-data start initialization, pthread lifecycle, serialization, and
teardown. They are retained as historical whole-process diagnostics only and
must not be used as throughput or #957 cancel-poll evidence.

The corrected schema-v2 metric is guest-reported WASI monotonic time:

- `hot`, `atomic`, and `wait-notify` spawn their workers first, warm code where
  applicable, and wait until every worker is ready;
- five empty synchronized epochs measure timer + release/completion-barrier
  overhead in the same process, and the minimum is subtracted;
- the clock starts immediately before releasing the work epoch and stops only
  after all intended operations complete;
- pthread join, JSON serialization, runtime teardown, and process startup are
  outside the kernel interval;
- `spawn-join` is explicitly a separate `spawn-join-lifecycle` metric whose
  guest interval intentionally contains pthread create + join, but still
  excludes module/process startup and teardown.

The report retains raw guest time, corrected guest time, overhead and overhead
ppm, plus host wall time as a watchdog/lifecycle diagnostic. Throughput uses
only corrected guest time. Default inputs target at least hundreds of
milliseconds per sample: 64 million hot-loop iterations per worker, 20 million
atomic RMWs per worker, 10,000 wait/notify hand-offs per worker, and 600
spawn/join rounds.

## Rebuild the fixtures

The pinned toolchain is wasi-sdk 25.0:

- archive:
  `wasi-sdk-25.0-x86_64-linux.tar.gz`
- SHA-256:
  `52640dde13599bf127a95499e61d6d640256119456d1af8897ab6725bcf3d89c`
- clang: `19.1.5-wasi-sdk`
- wasi-libc:
  `574b88da481569b65a237cb80daf9a2d5aeaf82d`

```sh
export WASI_SDK_PATH=/path/to/wasi-sdk-25.0-x86_64-linux
export TMPDIR="$PWD/zig-out/wasi-thread-fixture-tmp"
tests/benchmarks/wasi-threads/build-fixtures.sh
git diff --exit-code -- tests/benchmarks/wasi-threads
```

The script verifies `fixtures.sha256`. CI downloads the exact archive only in
the x86_64 fixture-integrity job; AArch64 benchmark jobs use the same checked-in
bytes.

## Run

```sh
python3 scripts/bench_wasi_threads.py \
  --profile authoritative \
  --output-dir "$PWD/zig-out/wasi-thread-bench" \
  --no-budget
```

The authoritative profile alternates each pair, discards two warmups, and keeps
ten measured samples. `report.json` follows `report.schema.json` and records
raw warmups/samples, commands, host/CPU/compiler/runtime identities, fixture and
source hashes, explicit pair direction, guest and host timing, build cache keys,
medians/ranges, paired ratios, immutable commit/platform/plan identities, and
every correctness result. JSON replacement is an fsynced same-directory atomic
rename that preserves an existing report's mode.

The AOT hot loop is compiled twice from the same `threaded.wasm`. The
`--benchmark-disable-cancel-points` compiler flag is accepted only by a wamrc
built with `-Dbenchmark-cancel-point-toggle=true`; normal production compilers
reject it. Normal runtime semantics therefore cannot accidentally ship with
cancellation disabled. During the timed `hot` interval the only variable work
is the intended kernel back-edge; barrier/timer cost is separately measured and
subtracted. The report records one dynamic cancel-poll opportunity per hot-loop
operation when polls are enabled, zero when disabled. Untimed wasi-libc and
initialization polls no longer enter the metric.

For local AArch64 validation on x86_64:

```sh
python3 scripts/bench_wasi_threads.py \
  --profile smoke --samples 1 --warmups 0 \
  --modes aot --thread-counts 1 \
  --single-iterations 100000 --hot-iterations 100000 \
  --atomic-iterations 10000 --wait-iterations 10 --spawn-iterations 1 \
  --min-interval-ms 1 \
  --target aarch64-linux-musl --aot-target aarch64 \
  --runner qemu-aarch64 \
  --output-dir "$PWD/zig-out/wasi-thread-bench-aarch64" \
  --no-budget
```

## Hosted calibration and budgets

The workflow remains path-filtered/manual and runs with `--no-budget`. Shared
local machines are not an authoritative calibration source. Do not start a
calibration cohort until the post-#979 methodology correction has passed
independent review and one validation run on each hosted platform.

Manual runs require a lowercase immutable 40-character `target_sha`; their
concurrency group uses `github.run_id` and never cancels another calibration
run. `scripts/wasi_thread_cohort.py` dispatches a bounded number in parallel and
records only run/artifact metadata, not build caches or large downloads:

```sh
python3 scripts/wasi_thread_cohort.py dispatch \
  --target-sha <40-char-main-commit> \
  --runs 20 --max-in-flight 2 \
  --output /d/wasi-thread-cohort-dispatch.json
```

After downloading only the retained report artifacts, validate the cohort:

```sh
python3 scripts/wasi_thread_cohort.py validate \
  --input-dir /d/wasi-thread-reports \
  --minimum-reports 20 \
  --output /d/wasi-thread-cohort.json
```

Validation rejects mixed commit, source, fixture, plan, profile, platform, or
duplicate workflow-run identity. A calibrated schema-v2 budget must identify
that exact cohort and contain platform-scoped thresholds for every explicit
left/right pair and every planned scenario/condition. Empty, partial, unknown,
duplicate, direction-inverted, or identity-mismatched budgets fail closed.

Until that cohort exists, claiming a statistically sound hard gate would be
fabricating evidence. Issue #966 must remain open and #963 remains dependent on
the published hosted baseline.
