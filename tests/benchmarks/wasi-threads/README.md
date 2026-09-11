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
interval shorter than 1.25 seconds, or measured timer/barrier overhead of 1%
or more.

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
only corrected guest time. Evidence counts are selected once per job by a
versioned deterministic sizing algorithm. The following checked-in table is
used only for pilots:

| mode/workload | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| interpreter `hot` | 28M | 28M | 20M | 10M |
| AOT `hot` | 1.8B | 1.8B | 900M | 450M |
| interpreter `atomic` | 72M | 40M | 28M | 14M |
| AOT `atomic` | 850M | 180M | 64M | 64M |
| interpreter `wait-notify` | 128K | 64K | 32K | 16K |
| AOT `wait-notify` | 1.5M | 64K | 32K | 16K |
| interpreter `spawn-join` | 9K | 4.5K | 2.25K | 1.25K |
| AOT `spawn-join` | 10K | 5K | 2.5K | 1.25K |
| AOT `cancel-hot` | 1.9B | 1.9B | 950M | 475M |

Single-hot uses 30M interpreter iterations and 1.9B AOT iterations.
Baseline and candidate always execute identical frozen work for the same cell.
Interpreter/AOT conditions may use different counts; throughput is normalized
by each record's validated operation count and corrected guest interval.

The 1.25-second quality floor remains derived independently from the retained
12.456 ms barrier target and the strict `<1%` rule. Before evidence, the
harness runs exactly one fixed-count pilot for every baseline/candidate and
left/right condition that uses a cell. The pilot order is canonical and every
pilot is retained; there are no retries, discarded pilots, adaptive sleeps, or
replacement observations.

This replaces fixed-count provenance after #1008 run 34173856468, job
101899360065 measured the AOT single-hot/one-thread threads-disabled warmup at
1.054664 seconds for 1.9 billion iterations on AMD EPYC 9V45, 71.95% faster
than the retained sizing host. That observation demonstrates that no finite
fixed host margin is a defensible contract.

For each cell, sizing version 8 selects the fastest valid pilot across all
revisions and conditions. With pilot iterations `P` and corrected elapsed
nanoseconds `E`, the general count is
`round_up_3_significant_digits(ceil(P*target_duration_ns*safety_numerator/(E*safety_denominator)))`,
where the target is 1.75 seconds and the safety factor is `10/7`. This projects
exactly 2.5 seconds at the fastest pilot rate: a portable 2x
measurement-to-pilot rate envelope over the 1.25-second evidence floor.

Three wholly new immutable-tag smoke cohorts retained ordinary-cell
accelerations above the old 1.925-second general target's `1.54x` tolerance:

| evidence | cell | fastest pilot rate (ops/s) | acceleration | old count | v8 count | projected failure |
|---|---|---:|---:|---:|---:|---:|
| run 34322503291 | AOT hot/8 | 207,202,544.631 | 1.567876679x | 399M | 519M | 1.597572s |
| run 34337339856 | interpreter wait/4 | 7,028.109 | 1.724465573x | 13.6K | 17.6K | 1.452177s |
| run 34351942414 | interpreter atomic/2 | 8,375,789.863 | 1.603974260x | 16.2M | 21.0M | 1.563134s |

The distinct modes, workloads, and thread counts make continued cell-specific
exceptions indefensible. The general 2x envelope has
`2 / 1.7244655732767034 - 1 = 0.15977960418180248`, or 15.9780%, margin over
the retained maximum. The three diagnostics are pinned respectively by
SHA-256 `95b9990bd9d0b1c73b26b0636c22555fb72af2b3942363a86619e5b5e2b1a995`,
`d05ca21d0557074aa356574b29d8ddf488c7e4a7ca31976e35bb6da92bb9f447`,
and `098a466f98da6899f2e7dba657078eb3853500a9bfca4e06727c23b51c0cbed3`.
Replaying every retained pilot under the 2x rule projects at most
4,205.813468912 seconds, or 70.10 minutes, below the 111-minute benchmark
limit.

One stronger declarative cell envelope additionally covers
`mode=aot, workload=wait-notify, threads=1` on every architecture. Its count is
`round_up_3_significant_digits(ceil(P*1250000000*16/E))`, and the selected
count is the maximum of the general and envelope counts. Equivalently, the
cell-specific formula is
`round_up_3_significant_digits(max(ceil(P*1750000000*10/(E*7)), ceil(P*1250000000*16/E)))`.
The `16/1` rate-acceleration envelope projects 20.0 seconds at the pilot rate,
so a later measurement may accelerate by up to 16 times and still meet the
1.25-second floor.

The retained failed hosted attempts and their ordered-rate evidence are:

| evidence | fastest pilot rate (ops/s) | later/pilot acceleration | v8 general count | v8 selected count | projected later duration |
|---|---:|---:|---:|---:|---:|
| attempt 1 | 739,133.627 | 2.192262005x | 1.85M | 14.8M | 9.133691s |
| attempt 2 | 853,510.803 | 1.836297620x | 2.14M | 17.1M | 10.910484s |
| attempt 3 | 262,449.465 | 1.737047051x | 657K | 5.25M | 11.516010s |
| #1020 run 34284891299 | 182,566.439 | 5.367617735x | 457K | 3.66M | 3.734896s |
| v4 run 34528848245 | 74,831.976 | 9.723491827x | 188K | 1.50M | 2.061493s |

The exact relative margin over the retained `9.723491826731898x` boundary is
`16 / 9.723491826731898 - 1 = 0.645499403415209`, or 64.5499%.
Replaying all 88 retained pilots and the 12 evidence invocations per
observation gives a maximum 4,193.807262141-second projection on the v4
failure, bounded by 69.90 minutes and still below 111 minutes. The diagnostic
is pinned by SHA-256
`8e8c9c7f0075ac7c9e36917dde73294a415f6756784107b06fb0055317035496`.
This stronger rule is a cell-specific rate envelope, not an x86 host
allowance: the selector
contains no CPU, runner, platform, or architecture identity and therefore
applies identically to every canonical platform.

There is no pilot-count floor: a slow host downsizes a long pilot to
target-sized evidence, while a fast host sizes up. The fastest baseline result
is retained as the explicit lower-bound derivation: a slower candidate cannot
lower work, while a faster candidate or condition increases it. The selected
count is then frozen for both revisions, both conditions, all warmups, and all
samples.

Pilots must pass guest correctness and operation assertions, have positive
timing fields, and provide at least a 1 ms corrected interval for clock
resolution. They are intentionally not required to satisfy the evidence
`99B < E` rule at their short unsized count. After the frozen count is known,
every retained pilot's barrier `B` is checked against its linearly projected
corrected evidence interval `E`: `99B < E`. Every projected interval must also
be at least both the 1.25-second evidence floor and the exact 2.5-second
`1.75s * 10/7` sizing target; AOT wait/notify/1 projects at least 20.0 seconds.
Actual warmups and samples
independently enforce the unchanged `99B < E_actual` and 1.25-second floor.
There is no hidden retry or count increase if a later rate exceeds its
declared 2x or 16x envelope: even a just-over-boundary observation that falls
below 1.25 seconds is retained and fails closed.

Selected counts must fit uint64 operations, the wait/notify signed-32-bit epoch
limit, checksum arithmetic, and declared workload caps. A pilot is rejected
immediately above 30 seconds corrected time or 33 seconds host-wall time; the
30-second corrected cap is above the retained approximately 23-second worst
cell while preventing 88 watchdog-length pilots from exhausting a job. Every
condition's projected invocation must remain strictly below the 90-second
watchdog.

The workflow reserves 69 minutes for non-benchmark work, leaving a 111-minute
benchmark limit. Before and during the full 88-pilot authoritative plan,
admission uses the hard 48-minute-24-second pilot bound (`88 * 33s`), the
51-minute minimum evidence bound
(`86 * 12 * 2.5s + 2 * 12 * 20s`), and the 10-minute auxiliary allowance.
Their sum is 109 minutes 24 seconds; adding the 69-minute reserve is 178
minutes 24 seconds, strictly below the 180-minute job timeout. The
harness accumulates actual pilot corrected and wall time after each one-shot
pilot and aborts immediately when the remaining hard bound cannot fit.

The reserve reduction is bounded by retained v3 AArch64 smoke job
102467148543. Its benchmark step occupied 4,701 seconds; retained pilots and
records account for 1,258.552313986 seconds, leaving 3,442.447686014 seconds
for builds, orchestration, preflights, and gaps. The 69-minute reserve alone is
4,140 seconds, 20.2633% above that unaccounted wall time; combined with the
10-minute auxiliary allowance, 4,740 seconds is 37.6927% above it. The report
hash, timestamps, and replay arithmetic are pinned in
`sizing-simulation-provenance.json`.

After all pilots finish, the hard pre-admission pilot allowance is no longer
charged. The final `projected_evidence_limit_ns` is exactly
`111m - actual_retained_pilot_wall - 600s`, and
`projected_benchmark_ns` is exactly
`actual_retained_pilot_wall + projected_evidence_wall + 600s`. Reports retain
both the distinct `maximum_pre_admission_pilot_bound_ns` and
`pilot_host_wall_elapsed_ns`; validation replays both. Any failure before
evidence retains all completed pilots in the diagnostic.

The hot-kernel expected checksum is prepared with an exact jump-ahead. After
unrolling the recurrence, terms with the same iteration index modulo 64 share
one rotation. Each of the 64 residue classes becomes the XOR of a consecutive
58-bit range plus fixed low six bits, so the result takes at most 64 groups per
worker regardless of an iteration count in the billions. All operations are
explicit unsigned 64-bit rotate/XOR arithmetic; no byte representation or host
endianness is involved. The report records the worst and total preparation
time for all unique plan keys before benchmark execution.

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
  --comparison-purpose candidate-evaluation \
  --profile authoritative \
  --host-pair-id "$GITHUB_RUN_ID/$RUNNER_ARCH" \
  --runner-environment github-hosted \
  --output-dir "$PWD/zig-out/wasi-thread-bench" \
  --no-budget
```

The workflow always supplies two checkouts. The single-revision CLI remains
available only for local and required-check compatibility:

```sh
python3 scripts/bench_wasi_threads.py \
  --profile authoritative \
  --output-dir "$PWD/zig-out/wasi-thread-bench" \
  --no-budget
```

That transitional invocation builds, runs, and emits only the `candidate`
role, sets `plan.revision_mode` and `plan.comparison_purpose` to
`single-revision-compatibility`, and emits no candidate/baseline comparisons.
It preserves current PR smoke coverage without doubling every measurement and
is not regression evidence between two source revisions.

The authoritative profile alternates each pair, discards two warmups, and keeps
ten measured samples. The smoke default is four measured samples. Paired mode
requires an even measured sample count: baseline and candidate each occupy the
first revision position exactly half the time. Warmups may have any count
because balance is enforced across the measured indices even when their
starting parity is shifted by the warmups. Within every sample index, the
left/right condition order continues to alternate independently. `report.json`
follows `report.schema.json` and
records raw warmups/samples, commands, host/CPU/compiler/runtime identities,
fixture and source hashes, explicit pair and revision direction, guest and host
timing, build cache keys, medians/ranges, immutable commit/platform/plan
identities, and every correctness result. JSON replacement is an fsynced
same-directory atomic rename that preserves an existing report's mode.

Each guest invocation has a fixed 90-second watchdog. Before evidence, the
harness checks every pilot-derived per-condition projection against that
watchdog and checks the complete projected benchmark path against the strict
111-minute limit. The workflow retains its 180-minute bound and 69-minute
non-benchmark reserve. Twenty
sequential trusted x86 jobs at the full job timeout still take 60 hours,
leaving 12 hours before the unchanged 72-hour dispatcher deadline.

Reports carry two plan identities. `plan_sha256` is the audit identity of the
complete plan, including `comparison_purpose`.
`measurement_plan_sha256` is version 9 of a purpose-independent portable
identity. It excludes only `comparison_purpose`, host-resolved evidence counts,
pilot outcomes, and their projections. It includes the workload/scenario
definitions, fixed pilot counts and order, sizing algorithm/version, target,
safety factor, declarative cell envelopes, rounding, caps, timeouts, modes,
thread counts, pairs, profile, samples, warmups, optimization, and quality
policy. `plan_sha256` hashes the complete report-specific plan, including every
pilot and selected count. `validate_report` recomputes both and independently
replays sizing.

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

Paired checkouts must resolve to different paths so both roles are independently
built. `candidate-evaluation` is the only purpose eligible for budget
enforcement. A same-revision A/A run is allowed only as explicit
`noise-calibration`, from two distinct checkout paths:

```sh
python3 scripts/bench_wasi_threads.py \
  --baseline-repo /path/to/calibration-a \
  --candidate-repo /path/to/calibration-b \
  --comparison-purpose noise-calibration \
  --samples 10 \
  --no-budget
```

Both noise-calibration checkouts must have identical commit, tracked-diff, and
build-source identities. Noise reports are always non-enforcing. Conversely,
budget loading accepts only candidate-evaluation reports and rejects identical
baseline/candidate commits or build-source identities. An accidentally reused
checkout can therefore never produce a passing gate.

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

## Hosted calibration, dispatch security, and cohorts

The workflow remains path-filtered/manual and always runs with `--no-budget`.
Every run is explicitly marked non-enforcing. No threshold is declared or
enabled by this workflow/cohort phase.

A successful hosted smoke run validates only smoke coverage; it does not
validate the 88-pilot authoritative profile. This change requires a full
hosted authoritative run before merge.

Pull requests and `main` pushes use GitHub-hosted x86_64 and AArch64 runners
only. A PR compares its trustworthy base commit with the tested merge commit. A
push compares the valid `before` commit with the current commit; branch creation
or another all-zero/invalid `before` value safely becomes an explicit same-SHA
`noise-calibration`. Both revisions are checked out into distinct directories,
even for same-SHA calibration, and each checked-out commit is verified before
any benchmark code runs. Benchmark output, SDK files, and caches live in an
explicit run/attempt/platform directory under `RUNNER_TEMP`; reports upload
before an `always()` cleanup removes exactly that guarded directory.

Manual dispatch requires lowercase immutable baseline and candidate SHAs,
purpose, profile, warmup/sample counts, and a runner target. The
`trusted-calibration` target is restricted to `workflow_dispatch` on `main`,
accepts only same-SHA `noise-calibration`, and verifies on a GitHub-hosted
preparation job that the target is commit history reachable from `main`. Only
its x86 job can select the repository-scoped `wamr-temp-20260906` label;
AArch64 remains `ubuntu-24.04-arm`. Both trusted-calibration jobs explicitly
enable the same fixed scheduler/barrier quality preflight; ordinary hosted
PR/push diagnostics do not enable it.

The preflight runs before any warmup or measured record. For each selected
thread count it runs exactly four AOT `hot` invocations through the checked-in
threaded guest's normal five-epoch release/completion barrier and runtime path:
16 fixed probes for the default 1/2/4/8 plan. It never retries, discards a
probe, adapts work, or waits for quiet. Every probe is retained. Against the
retained empirical one-in-257 tail, 16 independent probes have only
`1 - (256/257)^16 ≈ 6.05%` detection power. This is therefore only a cheap
fixed fail-fast check of current host state; it is not authoritative and
cannot promise that a later sample will not stall.

A separate atomic-wait stress preflight runs the one-thread AOT atomic path
eight times for smoke and 64 times for authoritative runs, per revision. Each
short probe uses exactly 1,000,000 iterations and remains fail-closed for
process exits, watchdog timeouts, malformed output, checksum mismatches, and
classified atomic-wait failures. These probes are correctness-only: they are
not retained as performance evidence, so clock/barrier overhead and the
measurement interval do not decide their acceptance. The measurement identity
records both the fixed work and this timing-quality policy. Every sizing pilot,
warmup, measured sample, and trusted scheduler/barrier probe keeps its strict
timing-quality gate.

With corrected interval `E` and measured barrier `B`, the unchanged strict
quality rule is `B / (E + B) < 0.01`, equivalently `99B < E`. The predeclared
barrier target is 12,456,000 ns, exactly twice the retained 6,228,000 ns
observation. It requires `E > 1,233,144,000 ns`. The fixed 1,250,000,000 ns
floor adds exactly 16,856,000 ns of headroom; at that floor the largest
accepted integer barrier is 12,626,262 ns. The later per-invocation timing gate
is authoritative and fail-closed for every warmup and measured sample.

The report retains the preflight values, min/median/max barrier summary, CPU
affinity and availability, load average, Linux CPU pressure when available,
the observable `Runner.Worker` process count, and runner/job identity. These
host diagnostics are explanatory only. Every run captures
`host_quiescence_at_start`, including hosted runs. A failed preflight or later
timing-quality failure takes a fresh `host_quiescence_at_failure` snapshot
rather than reusing the start state, and records the ratio at the fixed minimum
interval. It writes `failure-diagnostic.json` and
`failure-diagnostic.md` before exiting. The always-running artifact step uploads
those files even when no normal report exists, and cleanup remains after upload.

**Evidence validity and stop rule:** a run is eligible for cohort evidence only
when the preflight passed and the complete normal report exists and validates.
If any preflight probe or later sample fails, stop that workflow run, retain its
diagnostic artifact, exclude the run rather than retrying or replacing its
sample, and investigate host quiescence before starting a separately declared
fresh run. Previously failed or partial evidence never becomes valid.

The temporary runner was registered with
`--no-default-labels`, so its complete job-routing inventory is the single
`wamr-temp-20260906` label and its exact registered name is
`vm31e-wamr-temp-20260906`.

For a public repository, this route is protected only while the repository's
fork approval setting remains `approval_policy: all_external_contributors` and
maintainers never approve a fork PR that adds or changes a job targeting
`wamr-temp-20260906`. The committed workflow restricts its current routes, but
does not by itself make future approved fork access impossible. Deregister the
temporary runner immediately after calibration completes, or immediately when
the calibration is abandoned or finally closed. Benchmark jobs have only
`contents: read`; the separate hosted PR-comment job alone receives
`pull-requests: write`.

`scripts/wasi_thread_cohort.py` resolves and records the workflow ref's exact
head before dispatch, sends all paired inputs unchanged, retains failed run
metadata without retrying, and predeclares a sequence-based training/holdout
split before results exist. Trusted calibration requires an immutable
`wasi-thread-calibration-*` tag whose commit is reachable from `main`; baseline
and candidate must both equal that tagged commit. Every dispatched run must
report the same resolved workflow head, so moving the tag after dispatch fails
closed rather than mixing workflow revisions:

Before a `trusted-calibration` cohort resolves or dispatches any workflow, the
script makes one bounded Actions runner inventory query against the target
repository. It requires the complete returned inventory to contain exactly one
runner with custom label `wamr-temp-20260906`; that runner must have exact name
`vm31e-wamr-temp-20260906`, the custom label must be its sole label, and it must
be online and idle. Missing or duplicate label matches, offline or busy state,
and name or label drift abort with a contextual error before workflow dispatch.
The GitHub-hosted target does not query runner inventory. This exact name/label
pair is operational identity for the temporary registration, not a portable
performance class; report host fingerprints remain the relevant performance
identity. The x86 job also requires at least 80 GiB available on its `/d`
filesystem before checkout or compilation. This catches deleted run caches
that remain held open by a cancelled compiler process even when directory
inspection appears clean.

```sh
python3 scripts/wasi_thread_cohort.py dispatch \
  --workflow-ref wasi-thread-calibration-966-vN \
  --baseline-sha <40-char-tagged-main-commit> \
  --candidate-sha <same-40-char-tagged-main-commit> \
  --purpose noise-calibration \
  --profile authoritative --warmups 2 --samples 10 \
  --runner-target trusted-calibration \
  --runs 20 --training-runs 16 --max-in-flight 2 \
  --timeout-seconds 259200 \
  --output /d/wasi-thread-cohort-dispatch.json
```

Do not run that command until this workflow has merged and the temporary runner
route is intentionally ready. Trusted calibration caps `--max-in-flight` at 2:
GitHub concurrency preserves only one pending run in addition to the active
run. The dispatcher's validated wall-clock timeout defaults to 72 hours, which
covers the authoritative job timeouts while ensuring a queued or never-scheduled
run fails loudly. Do not manually dispatch this workflow while a cohort is in
progress; any unrelated manual dispatch introduces uncontrolled contention and
invalidates the cohort, and a displaced pending cohort run fails the dispatcher.
After downloading every retained report artifact, validate against the exact
dispatch manifest:

```sh
python3 scripts/wasi_thread_cohort.py validate \
  --input-dir /d/wasi-thread-reports \
  --dispatch-state /d/wasi-thread-cohort-dispatch.json \
  --output /d/wasi-thread-cohort.json
```

Paired validation requires exactly one baseline/candidate report for every
platform and workflow run: no missing, duplicate, inverted, partial, unexpected,
legacy, or cherry-picked report is accepted. It verifies the immutable workflow
head and target SHAs, purpose/profile/warmup/sample plan, fixture and plan
identity, balanced sample ordering, the shared baseline/candidate host identity
inside each report, and stable baseline/candidate build identities. Trusted
calibration additionally requires every x86 report to identify runner
`vm31e-wamr-temp-20260906` and one exact host fingerprint across the cohort.
GitHub-hosted x86 and AArch64 reports may have heterogeneous hosts across runs;
validation retains every observation and summarizes their fingerprint, CPU,
and runner-image distributions instead of rejecting normal hosted-runner
variation. The output
lists every retained observation with its exact workflow run ID and predeclared
training/holdout partition, and records an empty exclusion list. The old
single-revision validation path remains non-authoritative compatibility only
and cannot enter a dispatch-backed paired cohort.

Validation also embeds each report's canonical SHA-256 identity and the raw
per-sample comparison, ratio-of-ratios, and candidate
`single-infrastructure/*` ratios. That makes subsequent derivation
self-contained: it never re-discovers reports, changes membership, or selects
observations after seeing results.

## Deterministic budget derivation

`derive` accepts only the previously validated authoritative paired cohort and
a separately reviewed policy document. The policy format is
`derivation-policy.schema.json`; `derivation-policy.synthetic.json` is test
data only and must not be treated as a production policy. Production policy
ceilings and the rounding cushion must be declared before derivation:

```sh
python3 scripts/wasi_thread_cohort.py derive \
  --cohort /d/wasi-thread-cohort.json \
  --policy /path/to/reviewed-derivation-policy.json \
  --budget-output /d/wasi-thread-budget.candidate.json \
  --evidence-json-output /d/wasi-thread-budget-evidence.json \
  --evidence-markdown-output /d/wasi-thread-budget-evidence.md
```

For each condition and internal pair, derivation first takes the median of that
report's retained raw ratio samples, then works in natural-log ratio space on
TRAINING reports only. The adverse one-sided noise bound is the more permissive
of:

1. the worst observed training deviation from ratio 1; and
2. the adverse endpoint of `median ± 6 × 1.4826 × MAD`.

The predeclared log rounding cushion is added to that selected bound. Derivation
fails if the result exceeds the separately declared engineering-policy ceiling.
No observation is removed: points outside the robust interval are listed only
as diagnostic outliers. Every untouched HOLDOUT report must satisfy every
candidate threshold or no output is written.

Every report on both architectures must also contain the direct candidate
`threads-enabled / threads-disabled` single-infrastructure raw ratios. Both the
absolute median throughput delta and absolute median elapsed delta must be
strictly below the issue's 2% policy; missing proof or any failure aborts with
the exact run/platform evidence.

The emitted candidate budget has complete thresholds and calibration
provenance, but intentionally sets `"enforcement": false`. A later proof/final
PR must explicitly change that field after reviewing the machine-readable
evidence. The evidence records workflow run IDs and immutable SHAs,
fixture/plan/profile/purpose, predeclared split, policy inputs, every formula
input/result, holdout results, and host/fingerprint/CPU/image distributions.
It contains no future candidate revision identity.

Schema version 3 separates the actual report revisions from budget calibration
provenance. Paired reports use `metadata.revisions.baseline` and `.candidate`;
single-revision compatibility reports contain only `.candidate`. These entries
describe the code that produced the current samples. A calibrated budget separately records
the baseline and candidate revisions plus the explicit `noise-calibration`
purpose used to derive its thresholds. The current report baseline must match
the calibrated baseline, fixture, purpose-independent measurement-plan
identity, and profile. The full
noise-calibration `plan_sha256` remains in provenance for audit but is not
compared to a candidate-evaluation full-plan hash, because the purpose field is
intentionally different. The current candidate commit and build-source hash
are expected to change and are never required to equal the calibration
candidate.

Budgets contain only paired ratio limits:

- a minimum candidate/baseline throughput ratio and maximum
  candidate/baseline elapsed ratio for every condition;
- a minimum throughput ratio-of-ratios and maximum elapsed ratio-of-ratios for
  every internal left/right pair.

Absolute throughput and elapsed values remain report diagnostics. A calibrated
budget must cover every platform, condition, and internal pair in the exact
declared direction. Missing or duplicate revisions, samples, conditions, or
thresholds; inverted directions; unsupported hosts; mixed canonical sizing or
fixture identities; incomplete pilots; stale baseline provenance; and partial
platform coverage all fail closed. Host-resolved selected counts and full plan
hashes may differ, and their distributions are retained without exclusions.
There is no success-shaped fallback to an absolute value or an uncalibrated
threshold.

Rebaseline only after an intentional baseline change or a reviewed methodology,
fixture, plan, hosted runner class, or toolchain change. Retain the complete
hosted paired cohort, update both calibration revision identities and every
derived ratio threshold together. Keep both `calibrated` and `enforcement`
false while evidence is incomplete; after successful deterministic derivation,
the candidate budget may set `calibrated` true but must keep `enforcement`
false until the proof/final PR explicitly enables it. A candidate-only source
change never requires rebaselining.

Schema-v3 fixed-plan reports and reports produced before measurement-plan
identity version 8, including version-2 reports from #1013, version-3/4
#1016 attempts, version-5 #1020 evidence, and version-6/7 cell-envelope
evidence, are invalid for a new authoritative cohort. Fresh
evidence with the canonical one-shot sizing identity is mandatory for
calibration and derivation. Reports with different legitimate host-selected
counts may mix;
reports with altered pilot specifications, target, safety factor, cell
envelope, rounding, caps, timeouts, or algorithm identity may not. Earlier
failed/partial runs and the retained timing-quality failure remain excluded
and cannot be retried, relabelled, or mixed into the fresh cohort.

Until that cohort exists, claiming a statistically sound hard gate would be
fabricating evidence. Issue #966 must remain open and #963 remains dependent on
the published hosted baseline.
