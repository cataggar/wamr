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

The corrected metric is guest-reported WASI monotonic time:

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
milliseconds per sample: 128 million hot-loop iterations per worker, 64
million atomic RMWs per worker with a 256 million aggregate floor, 512,000
total wait/notify hand-offs divided evenly across the selected workers, and
3,000 spawn/join rounds. The aggregate atomic floor lengthens the low-thread
cells that are most exposed to hosted scheduling while retaining the existing
per-worker floor for 4/8-thread scaling. Keeping the wait/notify operation
total fixed avoids turning its intentionally serialized controller/worker
protocol into a multi-minute sample at higher thread counts.

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

The paired-revision mode takes two immutable checkouts. The driver builds each
revision into a separate output subtree, verifies that both use the same
fixture set and plan, and tags every warmup and measured sample with its
revision commit and build-source hash:

```sh
python3 scripts/bench_wasi_threads.py \
  --baseline-repo /path/to/immutable-baseline \
  --candidate-repo /path/to/immutable-candidate \
  --profile authoritative \
  --host-pair-id "$GITHUB_RUN_ID/$RUNNER_ARCH" \
  --runner-environment github-hosted \
  --output-dir "$PWD/zig-out/wasi-thread-bench" \
  --no-budget
```

Until the workflow PR supplies the two checkouts, the existing single-revision
CLI remains valid:

```sh
python3 scripts/bench_wasi_threads.py \
  --profile authoritative \
  --output-dir "$PWD/zig-out/wasi-thread-bench" \
  --no-budget
```

That transitional invocation records separate `baseline` and `candidate`
identities which intentionally point to the same checkout and sets
`plan.revision_mode` to `single-revision-compatibility`. It is schema-valid but
is not regression evidence between two source revisions.

The authoritative profile alternates each pair, discards two warmups, and keeps
ten measured samples. Within every sample index, baseline/candidate execution
order alternates as a balanced pair; the left/right condition order continues
to alternate independently. `report.json` follows `report.schema.json` and
records raw warmups/samples, commands, host/CPU/compiler/runtime identities,
fixture and source hashes, explicit pair and revision direction, guest and host
timing, build cache keys, medians/ranges, immutable commit/platform/plan
identities, and every correctness result. JSON replacement is an fsynced
same-directory atomic rename that preserves an existing report's mode.

The report exposes four metric layers:

1. `summaries`: raw absolute elapsed time and throughput by revision and
   condition. These remain diagnostic and are not portable performance gates.
2. `paired_summaries`: the internal right/left condition ratio for each
   revision.
3. `comparison_summaries`: matched candidate/baseline elapsed and throughput
   ratios for the same condition and sample index.
4. `ratio_of_ratios_summaries`: candidate internal right/left ratio divided by
   the baseline internal right/left ratio. A throughput value below 1 means the
   candidate's internal-pair relationship regressed; an elapsed value above 1
   means it regressed.

Runner metadata separates the diagnostic, high-cardinality `runner_name` from
`runner_environment` and the stable host fingerprint used as the performance
class. The fingerprint includes system, machine, CPU, logical CPU count, runner
image/OS/architecture, and runner environment. It deliberately excludes runner
name, workflow run ID, attempt, and workflow name. `host_pair.id` identifies
the particular matched baseline/candidate run without becoming a performance
class.

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
  --atomic-iterations 10000 --atomic-total-iterations 10000 \
  --wait-iterations 10 --spawn-iterations 1 \
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

The commands below describe the currently checked-in single-revision dispatch
plumbing. They cannot produce a schema-v3 paired calibration cohort by
themselves. The dependent workflow PR must create immutable baseline and
candidate checkouts, pass their paths and one host-pair ID to the harness, and
update cohort aggregation before calibration is enabled.

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

Schema version 3 separates the actual report revisions from budget calibration
provenance. `metadata.revisions.baseline` and `.candidate` always describe the
code that produced the current samples. A calibrated budget separately records
the baseline and candidate revisions used to derive its thresholds. The
current report baseline must match the calibrated baseline, fixture, plan, and
profile. The current candidate commit and build-source hash are expected to
change and are never required to equal the calibration candidate.

Budgets contain only paired ratio limits:

- a minimum candidate/baseline throughput ratio and maximum
  candidate/baseline elapsed ratio for every condition;
- a minimum throughput ratio-of-ratios and maximum elapsed ratio-of-ratios for
  every internal left/right pair.

Absolute throughput and elapsed values remain report diagnostics. A calibrated
budget must cover every platform, condition, and internal pair in the exact
declared direction. Missing or duplicate revisions, samples, conditions, or
thresholds; inverted directions; unsupported or mixed hosts; mixed plan or
fixture identities; stale baseline provenance; and partial platform coverage
all fail closed. There is no success-shaped fallback to an absolute value or an
uncalibrated threshold.

Rebaseline only after an intentional baseline change or a reviewed methodology,
fixture, plan, hosted runner class, or toolchain change. Retain the complete
hosted paired cohort, update both calibration revision identities and every
derived ratio threshold together, and keep `calibrated` and `enforcement`
false until the replacement cohort meets the declared report-count and platform
requirements. A candidate-only source change never requires rebaselining.

Until that cohort exists, claiming a statistically sound hard gate would be
fabricating evidence. Issue #966 must remain open and #963 remains dependent on
the published hosted baseline.
