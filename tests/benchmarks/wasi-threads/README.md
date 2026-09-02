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
non-zero exit, or watchdog timeout.

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
source hashes, build cache keys, medians/ranges, paired ratios, commit, and every
correctness result. JSON replacement is an fsynced same-directory atomic rename
that preserves an existing report's mode.

The AOT hot loop is compiled twice from the same `threaded.wasm`. The
`--benchmark-disable-cancel-points` compiler flag is accepted only by a wamrc
built with `-Dbenchmark-cancel-point-toggle=true`; normal production compilers
reject it. Normal runtime semantics therefore cannot accidentally ship with
cancellation disabled.

For local AArch64 validation on x86_64:

```sh
python3 scripts/bench_wasi_threads.py \
  --profile smoke --samples 1 --warmups 0 \
  --modes aot --thread-counts 1 \
  --target aarch64-linux-musl --aot-target aarch64 \
  --runner qemu-aarch64 \
  --output-dir "$PWD/zig-out/wasi-thread-bench-aarch64" \
  --no-budget
```

## Hosted calibration and budgets

The first merged workflow is deliberately path-filtered/manual and runs with
`--no-budget`. Shared local machines are not an authoritative calibration
source. To turn it into a hard gate:

1. Dispatch `.github/workflows/wasi-thread-bench.yml` with
   `profile=authoritative`, `warmups=2`, and `samples=10` at one exact main
   commit on at least ten separate occasions.
2. Retain at least 20 valid reports for each hosted platform
   (`ubuntu-22.04` x86_64 and `ubuntu-24.04-arm` AArch64).
3. Derive limits from that cohort, update `budget.json` with
   `calibrated=true`, paired elapsed-delta limits and minimum median
   throughputs, change the workflow to pass `--budget`, and verify an injected
   regression fails the job.

Until that cohort exists, claiming a statistically sound hard gate would be
fabricating evidence. Issue #966 must remain open and #963 remains dependent on
the published hosted baseline.
